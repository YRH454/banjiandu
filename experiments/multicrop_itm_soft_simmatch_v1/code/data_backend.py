"""Public pair loader and frozen-prefix cache with unchanged full captions."""
from __future__ import annotations
import csv
import hashlib
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import RandAugment
from common import ROOT, digest, read

U_FIELDS = ["pair_id", "image_id", "image_path", "text", "text_sha256"]
L_FIELDS = ["image_id", "image_relpath", "positive_text", "negative_text",
            "source_text_sha256", "negative_text_sha256", "image_sha256"]
# Frozen prefix execution is independent of cache hits, optimizer microbatch
# fallback and caption lengths. Padding duplicates are discarded before use.
PREFIX_BATCH = 4

def fixed_seed(seed, step, purpose):
    return int.from_bytes(hashlib.sha256(f"{seed}\0{step}\0{purpose}".encode()).digest()[:8], "little")

def load_contract(crop, budget):
    manifest_path = ROOT / "data" / crop / "manifest.json"
    manifest = read(manifest_path)
    asset_meta = manifest["asset_index"]
    if digest(ROOT / asset_meta["path"]) != asset_meta["sha256"]:
        raise RuntimeError("Asset index modified")
    assets = read(ROOT / asset_meta["path"])
    def rows(contract, fields):
        path = ROOT / contract["path"]
        if digest(path) != contract["sha256"]:
            raise RuntimeError(f"Input hash mismatch {path}")
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fields:
                raise RuntimeError("Input has forbidden/missing columns")
            output = list(reader)
        for r in output:
            if r["image_id"] not in assets:
                raise RuntimeError("Unregistered image ID")
            for text_key, hash_key in (("text", "text_sha256"),) if fields == U_FIELDS else (
                    ("positive_text", "source_text_sha256"), ("negative_text", "negative_text_sha256")):
                if hashlib.sha256(r[text_key].encode()).hexdigest() != r[hash_key]:
                    raise RuntimeError("Full-caption hash changed")
        return output
    l = rows(manifest["budgets"][budget]["l"], L_FIELDS)
    u = rows(manifest["budgets"][budget]["u"], U_FIELDS)
    val = rows(manifest["validation"], L_FIELDS)
    l_ids, u_ids, v_ids = ({r["image_id"] for r in rs} for rs in (l, u, val))
    if l_ids & u_ids or (l_ids | u_ids) & v_ids or len(val) != 400:
        raise RuntimeError("Train/Validation membership contract changed")
    if len({r["pair_id"] for r in u}) != len(u):
        raise RuntimeError("Duplicate U pair IDs")
    return manifest, assets, l, u, val

def flatten_l(rows):
    return [dict(image_id=r["image_id"], pair_id=r["image_id"] + ":" + kind,
                 text=r[text_key], label=label)
            for r in rows for kind, text_key, label in (("pos", "positive_text", 1.), ("neg", "negative_text", 0.))]

class PrefixCache:
    def __init__(self, model, tokenizer, assets, guard, seed, device):
        self.model, self.tokenizer, self.assets = model, tokenizer, assets
        self.guard, self.seed, self.device = guard, seed, device
        self.text, self.images, self.verified = {}, {}, set()
        self.augment = RandAugment(num_ops=2, magnitude=10)
        self.mean = torch.tensor([.48145466, .4578275, .40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([.26862954, .26130258, .27577711], device=device).view(1, 3, 1, 1)

    def decode(self, image_id):
        meta = self.assets[image_id]
        path = ROOT / meta["path"]
        if image_id not in self.verified:
            if digest(path) != meta["sha256"]:
                raise RuntimeError(f"Image hash changed {image_id}")
            self.verified.add(image_id)
        with Image.open(path) as image:
            return image.convert("RGB").resize((384, 384), Image.Resampling.BICUBIC)

    def pixels(self, images):
        a = np.stack([np.asarray(im, dtype=np.uint8).copy() for im in images])
        x = torch.from_numpy(a).permute(0, 3, 1, 2).to(self.device).float() / 255.
        return (x - self.mean) / self.std

    @torch.no_grad()
    def texts(self, strings):
        missing = {hashlib.sha256(s.encode()).hexdigest(): s for s in strings
                   if hashlib.sha256(s.encode()).hexdigest() not in self.text}
        items = list(missing.items())
        for begin in range(0, len(items), PREFIX_BATCH):
            group = items[begin:begin + PREFIX_BATCH]
            strings_fixed = [s for _, s in group]
            strings_fixed += [strings_fixed[0]] * (PREFIX_BATCH-len(strings_fixed))
            encoded = self.tokenizer(strings_fixed, truncation=False, padding="max_length",
                                     max_length=self.guard, return_tensors="pt")
            if encoded["input_ids"].shape[1] > self.guard:
                raise ValueError("Caption over guard: refusing truncation")
            ids = encoded["input_ids"].to(self.device)
            mask = encoded["attention_mask"].to(self.device)
            with torch.autocast("cuda", dtype=torch.float16):
                values = self.model.text_prefix(ids, mask).detach().cpu().half()
            for (key, _), value, length in zip(group, values, mask.sum(1).cpu().tolist()):
                self.text[key] = value[:int(length)].contiguous()
        entries = [self.text[hashlib.sha256(s.encode()).hexdigest()] for s in strings]
        length = max(x.shape[0] for x in entries)
        values = torch.zeros(len(entries), length, 768, dtype=torch.float16)
        mask = torch.zeros(len(entries), length, dtype=torch.long)
        for i, x in enumerate(entries):
            values[i, :len(x)] = x
            mask[i, :len(x)] = 1
        return values.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def batch(self, pairs, step, view, physical):
        flips = [0 if view == "validation" else fixed_seed(self.seed, step, "flip/" + p["image_id"]) & 1 for p in pairs]
        if view in ("weak", "validation"):
            missing = list(dict.fromkeys((p["image_id"], flip) for p, flip in zip(pairs, flips)
                                         if (p["image_id"], flip) not in self.images))
            for begin in range(0, len(missing), PREFIX_BATCH):
                group = missing[begin:begin + PREFIX_BATCH]
                ims = [self.decode(k) for k, _ in group]
                ims = [im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if f else im for im, (_, f) in zip(ims, group)]
                ims += [ims[0]] * (PREFIX_BATCH-len(ims))
                with torch.autocast("cuda", dtype=torch.float16):
                    prefix = self.model.image_prefix(self.pixels(ims)).detach().cpu().half()
                for key, value in zip(group, prefix):
                    self.images[key] = value.contiguous()
            image = torch.stack([self.images[p["image_id"], f] for p, f in zip(pairs, flips)]).to(self.device)
        elif view == "strong":
            ims = []
            for p, flip in zip(pairs, flips):
                im = self.decode(p["image_id"])
                if flip:
                    im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                seed = fixed_seed(self.seed, step, "strong/" + p["pair_id"])
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    im = self.augment(im)
                a = np.asarray(im).copy()
                rng = random.Random(seed)
                center_x, center_y = rng.randrange(384), rng.randrange(384)
                half = 24
                a[max(0, center_y-half):min(384, center_y+half), max(0, center_x-half):min(384, center_x+half)] = 127
                ims.append(Image.fromarray(a))
            parts = []
            for begin in range(0, len(ims), PREFIX_BATCH):
                group = ims[begin:begin + PREFIX_BATCH]
                actual = len(group)
                group += [group[0]] * (PREFIX_BATCH-actual)
                with torch.autocast("cuda", dtype=torch.float16):
                    parts.append(self.model.image_prefix(self.pixels(group)).detach().half()[:actual])
            image = torch.cat(parts)
        else:
            raise ValueError("Unknown view")
        text, mask = self.texts([p["text"] for p in pairs])
        return image, text, mask

def chunks(batch, size):
    for begin in range(0, len(batch[0]), size):
        yield tuple(x[begin:begin + size] for x in batch)
