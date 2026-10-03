"""Bounded parallel JPEG decode/resize; augmentation and sampling stay unchanged."""
from collections import OrderedDict
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import random

from data_backend import PrefixCache, fixed_seed


class TensorLRU(MutableMapping):
    def __init__(self, budget):
        self.budget = budget
        self.used = 0
        self.entries = OrderedDict()

    def __getitem__(self, key):
        value = self.entries[key]
        self.entries.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self.entries:
            self.__delitem__(key)
        self.entries[key] = value
        self.used += value.numel() * value.element_size()
        while self.used > self.budget and len(self.entries) > 1:
            oldest = next(iter(self.entries))
            self.__delitem__(oldest)

    def __delitem__(self, key):
        value = self.entries.pop(key)
        self.used -= value.numel() * value.element_size()

    def __contains__(self, key):
        if key in self.entries:
            self.entries.move_to_end(key)
            return True
        return False

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)


class ParallelDecodeCache(PrefixCache):
    def __init__(self, *args, workers=8, decoded_capacity=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jpeg")
        self.decoded_capacity = decoded_capacity
        self.decoded = OrderedDict()
        self.pending = {}
        self.decode_lock = Lock()

    def _decoded(self, image_id):
        return super().decode(image_id)

    def prefetch(self, pairs):
        for image_id in dict.fromkeys(p["image_id"] for p in pairs):
            with self.decode_lock:
                if image_id not in self.decoded and image_id not in self.pending:
                    self.pending[image_id] = self.pool.submit(self._decoded, image_id)

    def decode(self, image_id):
        with self.decode_lock:
            if image_id in self.decoded:
                self.decoded.move_to_end(image_id)
                return self.decoded[image_id].copy()
            future = self.pending.get(image_id)
            if future is None:
                future = self.pool.submit(self._decoded, image_id)
                self.pending[image_id] = future
        image = future.result()
        with self.decode_lock:
            self.pending.pop(image_id, None)
            self.decoded[image_id] = image
            self.decoded.move_to_end(image_id)
            while len(self.decoded) > self.decoded_capacity:
                self.decoded.popitem(last=False)
        return image.copy()

    def batch(self, pairs, step, view, physical):
        self.prefetch(pairs)
        return super().batch(pairs, step, view, physical)

    def next_step(self, labeled, unlabeled, step):
        selected_l = random.Random(fixed_seed(self.seed, step, "pair-sampler")).sample(labeled, 16)
        selected_u = random.Random(fixed_seed(self.seed, step, "u-sampler")).sample(unlabeled, 32)
        self.prefetch(selected_l + selected_u)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        self.decoded.clear()


def apply_options(runner, options):
    import data_backend
    from albef_ssl.model import get_tokenizer
    from common import ROOT

    data_backend.PREFIX_BATCH = options["prefix_batch"]
    runner.physical = options["physical_pairs"]
    for model in (runner.model, runner.ema):
        model.activation_checkpointing = options["activation_checkpointing"]
        model.text_encoder.config.gradient_checkpointing = options["activation_checkpointing"]
        model.fusion_chunk_size = runner.physical
    cache_cls = ParallelDecodeCache if options["decode_workers"] else PrefixCache
    kwargs = {"workers": options["decode_workers"]} if options["decode_workers"] else {}
    runner.cache = cache_cls(runner.model, get_tokenizer(str(ROOT / "assets/tokenizer")),
                            runner.assets, runner.manifest["token_guard"],
                            runner.cfg["seed"], runner.device, **kwargs)
    if options["decode_workers"]:
        runner.cache.images = TensorLRU(8 * 1024**3)
        runner.cache.text = TensorLRU(1024**3)


def close_cache(runner):
    if isinstance(runner.cache, ParallelDecodeCache):
        runner.cache.close()

