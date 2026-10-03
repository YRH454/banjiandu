"""Cassava-adapted ITM + Pair-USA backend, based on the registered Apple trainer.

The module is deliberately inert on import.  Normal runs require the negative
quality gate, never read the test split, and execute one CUDA worker at a time.
"""

from __future__ import annotations

import argparse
import copy
import gc
import csv
import ctypes
import hashlib
import io
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
CORE = Path("/path/to/albef/code")
V2_DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
SEED = 20260825
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073))[None, :, None, None]
STD = torch.tensor((0.26862954, 0.26130258, 0.27577711))[None, :, None, None]
BUDGET_ORDER = ("005", "020", "001", "010", "030", "100")
PLAN = json.loads((ROOT / "configs/plan.json").read_text(encoding="utf-8"))
EXPECTED_COUNTS = PLAN["l_counts"]
MAX_TEXT_TOKENS = PLAN["max_text_tokens_including_special_tokens"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_fingerprint(path: Path) -> str:
    """A stable digest for a source file or an immutable tokenizer snapshot tree."""
    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).replace("\\", "/").encode("utf-8"))
            digest.update(sha256(child).encode("ascii"))
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_with_retry(source: Path, destination: Path) -> None:
    deadline = time.monotonic() + 5.0
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".atomic-", suffix=".tmp", dir=str(path.parent), delete=False) as out:
            tmp = Path(out.name)
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        replace_with_retry(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def atomic_json(path: Path, item: object) -> None:
    atomic_bytes(path, (json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_torch(path: Path, item: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".checkpoint-", suffix=".pt", dir=str(path.parent), delete=False) as out:
            tmp = Path(out.name)
        torch.save(item, tmp)
        replace_with_retry(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def append_jsonl(path: Path, item: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        out.flush()


@contextmanager
def process_lock() -> Iterator[None]:
    """A Windows exclusive lock prevents two launchers from sharing the GPU/output."""
    import msvcrt
    path = ROOT / ".queue.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    try:
        stream.seek(0)
        stream.write(" ")
        stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another cassava ITM launcher holds the queue lock") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()} started_utc={now()}\n")
        stream.flush()
        yield
    finally:
        try:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        stream.close()


@contextmanager
def prevent_system_sleep() -> Iterator[None]:
    """Keep only this process awake; no permanent Windows power-plan change."""
    if os.name != "nt":
        yield
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    result = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if result == 0:
        raise OSError("SetThreadExecutionState failed")
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def source_fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    source_files = [CODE / "legacy_parent.py", CODE / "pair_model.py", CODE / "prepare_pairs.py",
                    ROOT / "reports/执行登记_木薯S1-S4_20261001.md", ROOT / "configs/plan.json",
                    ROOT / "data/manifest.json",
                    ROOT / "data/u_manifest.json", ROOT / "data/splits/split_manifest.json",
                    ROOT / "audit/negative_quality_gate.json", ROOT / "audit/source_audit.json",
                    CORE / "albef_ssl/model.py", CORE / "albef_ssl/vendor/albef/vit.py",
                    CORE / "albef_ssl/vendor/albef/xbert.py", CORE / "albef_ssl/vendor/albef/bert_config.json"]
    for key in ("train", "validation"):
        path = manifest.get("files", {}).get(key)
        if path:
            source_files.append(ROOT / path)
    for path in manifest.get("files", {}).get("budgets", {}).values():
        source_files.append(ROOT / path)
    result = {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path)
              for path in source_files if path.exists()}
    tokenizer = Path(model_config()["tokenizer_path"])
    result["tokenizer_snapshot"] = path_fingerprint(tokenizer)
    return result


def model_config() -> dict[str, Any]:
    """The exact registered LoRA topology, copied from v2 rather than learned from results."""
    return {
        "checkpoint": "/path/to/albef/weights/ALBEF_4M.pth",
        "checkpoint_md5_expected": "3c876d776a8e0ce61e2285fc9897f0b3",
        "tokenizer_path": "/path/to/huggingface-cache/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
        "image_size": 384, "vision_lora_layers": [10, 11], "vision_lora_rank": 4,
        "vision_lora_alpha": 8, "cross_lora_layers": [6, 7, 8, 9, 10, 11],
        "cross_lora_rank": 8, "cross_lora_alpha": 16, "max_text_length": 384,
        "seed": SEED, "initial_student_temperature": 0.07,
        "fusion_chunk_size": 8, "activation_checkpointing": True,
    }


def make_configs(manifest: dict[str, Any], smoke_steps: int | None = None) -> list[dict[str, Any]]:
    steps = int(smoke_steps or 1600)
    result = []
    for code in BUDGET_ORDER:
        for method in ("A_bce", "B_bce_pairusa"):
            run_id = f"apple_itm_random_{method}_l{code}_s{SEED}"
            if smoke_steps is not None:
                run_id += f"_smoke{steps}"
            result.append({
                "run_id": run_id, "budget": code, "method": method,
                "seed": SEED, "pair_file": manifest["files"]["budgets"][code],
                "validation_file": manifest["files"]["validation"], "image_root": manifest["image_root"],
                "model": model_config(), "training": {
                    "max_steps": steps, "logical_positive_batch": 16, "physical_positive_batch": 16,
                    "warmup_steps": min(80, max(1, steps // 4)) if smoke_steps else 80,
                    "checkpoint_every": min(50, max(1, steps)), "eval_every": min(100, max(1, steps)),
                    "lora_lr": 1e-4, "head_lr": 2e-4, "weight_decay": 0.01, "max_grad_norm": 1.0,
                    "pairusa_teacher_temp": 1.0, "pairusa_student_temp_init": 0.07,
                    "pairusa_lambda": 0.10, "pairusa_start": 0 if smoke_steps else 100,
                    "pairusa_ramp_end": 1 if smoke_steps else 200,
                    "teacher_batch": 16, "teacher_lr": 2e-4,
                    "teacher_max_epochs": min(2, steps) if smoke_steps else 20,
                    "teacher_patience": 3,
                },
            })
    return result


def configure_files(manifest: dict[str, Any]) -> None:
    config_dir = ROOT / "configs"
    configs = make_configs(manifest)
    config_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        path = config_dir / f"{cfg['run_id']}.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != cfg:
                raise RuntimeError(f"Registered queue configuration changed: {path.name}")
        else:
            atomic_json(path, cfg)
    expected = {"queue_order": [c["run_id"] for c in configs], "source_manifest_sha256": sha256(ROOT / "data/manifest.json")}
    queue_path = config_dir / "queue.json"
    if queue_path.exists():
        existing = json.loads(queue_path.read_text(encoding="utf-8"))
        if {key: existing.get(key) for key in expected} != expected:
            raise RuntimeError("Registered queue manifest/order changed")
    else:
        atomic_json(queue_path, {"created_utc": now(), **expected})


def load_manifest(require_gate: bool) -> dict[str, Any]:
    manifest_path = ROOT / "data/manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("No pair manifest. Run the approved negative-pair preparation first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("negative_policy") != PLAN["negative_policy"]:
        raise RuntimeError("This queue only accepts the newly authorized random full-caption policy")
    needed = {"image_root", "files"}
    if not needed <= set(manifest):
        raise RuntimeError("Pair manifest is missing image_root or files")
    if require_gate:
        gate_path = ROOT / "audit/negative_quality_gate.json"
        if not gate_path.exists():
            raise RuntimeError("Negative-pair quality gate is absent; formal queue will not start")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("negative_policy") != manifest["negative_policy"]:
            raise RuntimeError("Structural gate and manifest negative policies disagree")
        if gate.get("ready_for_training") is not True:
            raise RuntimeError("Negative-pair quality gate is not approved: " + "; ".join(gate.get("reasons", [])))
        admitted = gate.get("admitted_file_sha256")
        if not isinstance(admitted, dict):
            raise RuntimeError("Negative-pair quality gate lacks admitted_file_sha256 binding")
        admitted_paths = [manifest["files"]["validation"], *manifest["files"]["budgets"].values()]
        for value in admitted_paths:
            path = csv_path(manifest, value)
            relative = str(path.relative_to(ROOT)).replace("\\", "/")
            if admitted.get(relative) != sha256(path):
                raise RuntimeError(f"Quality gate does not admit the current pair file: {relative}")
    return manifest


def read_pairs(path: Path, allow_unqualified_smoke: bool = False) -> list[dict[str, str]]:
    required = {"image_id", "image_relpath", "positive_text", "negative_text", "edit_type", "review_status", "negative_source_image_id",
                "rule_id", "source_text_sha256", "negative_text_sha256"}
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Invalid pair file schema: {path}")
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not row["positive_text"].strip() or not row["negative_text"].strip() for row in rows):
        raise ValueError(f"Empty text or duplicate image ID in {path}")
    if not allow_unqualified_smoke and any(row["review_status"] != "random_other_case_unverified" for row in rows):
        raise ValueError(f"Unexpected annotation status for authorized random pairing in {path}")
    by_id = {row['image_id']: row for row in rows}
    if set(row['negative_source_image_id'] for row in rows) != set(by_id):
        raise ValueError(f"Random negatives are not a within-file caption permutation: {path}")
    for row in rows:
        donor = by_id[row['negative_source_image_id']]
        if donor['image_id'] == row['image_id'] or donor['positive_text'] != row['negative_text']:
            raise ValueError(f"Self-pair or modified donor caption for {row['image_id']}")
        if row['negative_text'] == row['positive_text']:
            raise ValueError(f"Identical positive/negative caption for {row['image_id']}")
        if hashlib.sha256(row["positive_text"].encode("utf-8")).hexdigest() != row["source_text_sha256"]:
            raise ValueError(f"Positive-caption hash mismatch for {row['image_id']}")
        if hashlib.sha256(row["negative_text"].encode("utf-8")).hexdigest() != row["negative_text_sha256"]:
            raise ValueError(f"Negative-caption hash mismatch for {row['image_id']}")
    return rows


def source_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["image_id"] for row in csv.DictReader(stream)}


def csv_path(manifest: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def decode_image(root: Path, row: dict[str, str]) -> np.ndarray:
    path = root / row["image_relpath"]
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((384, 384), Image.Resampling.BICUBIC), dtype=np.uint8).copy()


def fixed_seed(seed: int, step: int, purpose: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{step}\0{purpose}".encode()).digest()[:8], "little")


def choose_unique(rows: list[dict[str, str]], step: int, count: int = 16) -> list[dict[str, str]]:
    if len(rows) < count:
        raise RuntimeError(f"Only {len(rows)} approved anchors; need {count} unique anchors per Pair-USA batch")
    rng = random.Random(fixed_seed(SEED, step, "pair-sampler"))
    return rng.sample(rows, count)


class PairCache:
    """Caches frozen prefixes keyed by image viewing state and caption SHA-256."""
    def __init__(self, model: Any, tokenizer: Any, image_root: Path) -> None:
        self.model, self.tokenizer, self.image_root = model, tokenizer, image_root
        self.tokens: dict[str, torch.Tensor] = {}
        self.text: dict[str, torch.Tensor] = {}
        self.images: dict[tuple[str, int], torch.Tensor] = {}
        self.truncations = 0

    def _text_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def prime_text(self, texts: list[str]) -> None:
        unique = {self._text_key(t): t for t in texts}
        missing = [(key, t) for key, t in unique.items() if key not in self.text]
        if not missing:
            return
        self.model.eval()
        for begin in range(0, len(missing), 64):
            group = missing[begin:begin + 64]
            raw = self.tokenizer([t for _, t in group], padding=False, truncation=False)["input_ids"]
            too_long = [key for (key, _), ids in zip(group, raw) if len(ids) > MAX_TEXT_TOKENS]
            if too_long:
                raise ValueError(f"Caption exceeds registered {MAX_TEXT_TOKENS}-token limit ({len(too_long)} captions); refusing truncation")
            encoded = self.tokenizer([t for _, t in group], padding=True, truncation=False,
                                     max_length=MAX_TEXT_TOKENS, return_tensors="pt")
            ids, mask = encoded["input_ids"].to(DEVICE), encoded["attention_mask"].to(DEVICE)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                prefix = self.model.text_prefix(ids, mask).detach().cpu().half()
            lengths = mask.sum(1).cpu().tolist()
            for (key, _), feature, length in zip(group, prefix, lengths):
                self.text[key] = feature[:int(length)].contiguous()

    def prime_images(self, rows: list[dict[str, str]], flips: bool = True) -> None:
        missing = [row for row in rows if (row["image_id"], 0) not in self.images]
        if not missing:
            return
        self.model.eval()
        mean, std = MEAN.to(DEVICE), STD.to(DEVICE)
        for begin in range(0, len(missing), 16):
            group = missing[begin:begin + 16]
            arrays = [decode_image(self.image_root, row) for row in group]
            views = arrays + [np.ascontiguousarray(x[:, ::-1, :]) for x in arrays] if flips else arrays
            pixels = torch.from_numpy(np.stack(views)).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
            pixels = (pixels - mean) / std
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                prefixes = self.model.image_prefix(pixels).detach().cpu().half()
            for index, row in enumerate(group):
                self.images[(row["image_id"], 0)] = prefixes[index].contiguous()
                if flips:
                    self.images[(row["image_id"], 1)] = prefixes[index + len(group)].contiguous()

    def pair_batch(self, rows: list[dict[str, str]], step: int, augment: bool) -> tuple[torch.Tensor, ...]:
        pos, neg = [r["positive_text"] for r in rows], [r["negative_text"] for r in rows]
        self.prime_text(pos + neg)
        if augment:
            flip = [fixed_seed(SEED, step, f"flip/{r['image_id']}") & 1 for r in rows]
        else:
            flip = [0] * len(rows)
        image = torch.stack([self.images[(r["image_id"], f)] for r, f in zip(rows, flip)]).to(DEVICE)
        def text_batch(strings: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
            values = [self.text[self._text_key(s)] for s in strings]
            length = max(v.shape[0] for v in values)
            prefix = torch.zeros((len(values), length, values[0].shape[-1]), dtype=torch.float16)
            mask = torch.zeros((len(values), length), dtype=torch.long)
            for index, value in enumerate(values):
                prefix[index, :value.shape[0]] = value
                mask[index, :value.shape[0]] = 1
            return prefix.to(DEVICE), mask.to(DEVICE)
        pt, pm = text_batch(pos)
        nt, nm = text_batch(neg)
        return image, pt, pm, nt, nm


def schedule(step: int, warmup: int, total: int) -> float:
    if step <= warmup:
        return step / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def pairusa_weight(step: int, train: dict[str, Any]) -> float:
    if step <= train["pairusa_start"]:
        return 0.0
    if step >= train["pairusa_ramp_end"]:
        return float(train["pairusa_lambda"])
    return float(train["pairusa_lambda"]) * (step - train["pairusa_start"]) / (train["pairusa_ramp_end"] - train["pairusa_start"])


def optimizer_for(model: Any, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    lora, heads = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (lora if "lora_" in name else heads).append(parameter)
    if not lora or not heads:
        raise RuntimeError("Expected trainable LoRA and ITM/projection head parameters")
    train = cfg["training"]
    return torch.optim.AdamW([
        {"params": lora, "lr": train["lora_lr"], "initial_lr": train["lora_lr"]},
        {"params": heads, "lr": train["head_lr"], "initial_lr": train["head_lr"]},
    ], weight_decay=train["weight_decay"])


def validation_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    flat_y, flat_scores = y.ravel(), scores.ravel()
    candidates = np.unique(flat_scores)
    best = (-float("inf"), float("inf"), 0.0)
    for threshold in candidates:
        f1 = f1_score(flat_y, flat_scores >= threshold)
        candidate = (float(f1), -abs(float(threshold)), -float(threshold))
        if candidate > best:
            best = candidate
    return -best[2]


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    probability = 1.0 / (1.0 + np.exp(-scores))
    threshold = validation_threshold(y, scores)
    decisions = scores >= threshold
    return {"paired_accuracy": float(np.mean(np.where(scores[:, 0] > scores[:, 1], 1.0,
                                                          np.where(scores[:, 0] == scores[:, 1], 0.5, 0.0)))),
            "auroc": float(roc_auc_score(y.ravel(), scores.ravel())),
            "average_precision": float(average_precision_score(y.ravel(), scores.ravel())),
            "validation_selected_threshold": float(threshold),
            "balanced_accuracy_at_validation_threshold": float(balanced_accuracy_score(y.ravel(), decisions.ravel())),
            "f1_at_validation_threshold": float(f1_score(y.ravel(), decisions.ravel())),
            "positive_recall_at_validation_threshold": float(np.mean(decisions[:, 0])),
            "false_caption_accept_rate_at_validation_threshold": float(np.mean(decisions[:, 1])),
            "mean_positive_probability": float(probability[:, 0].mean()),
            "mean_negative_probability": float(probability[:, 1].mean()),
            "mean_margin": float((scores[:, 0] - scores[:, 1]).mean()), "n_anchors": int(len(y))}


def validation(model: Any, cache: PairCache, rows: list[dict[str, str]]) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    all_scores = []
    with torch.inference_mode():
        for begin in range(0, len(rows), 16):
            group = rows[begin:begin + 16]
            image, pt, pm, nt, nm = cache.pair_batch(group, 0, augment=False)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits, _ = model.forward_cached(image, pt, pm, nt, nm)
            all_scores.append(logits.float().cpu().numpy())
    scores = np.concatenate(all_scores)
    return metrics(np.tile(np.array([[1, 0]], dtype=np.int64), (len(scores), 1)), scores), scores


def validation_prediction_csv(rows: list[dict[str, str]], scores: np.ndarray, threshold: float) -> bytes:
    """Keep every validated anchor/edit label beside its two logits for later strata reports."""
    fields = list(rows[0].keys()) + ["positive_logit", "negative_logit", "validation_threshold",
                                     "positive_accepted", "negative_accepted"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row, value in zip(rows, scores):
        writer.writerow({**row, "positive_logit": f"{float(value[0]):.9g}", "negative_logit": f"{float(value[1]):.9g}",
                         "validation_threshold": f"{threshold:.9g}", "positive_accepted": int(value[0] >= threshold),
                         "negative_accepted": int(value[1] >= threshold)})
    return stream.getvalue().encode("utf-8")


def teacher_descriptors(rows: list[dict[str, str]], descriptor: Any, tokenizer: Any, image_root: Path,
                        cache: dict[str, torch.Tensor], horizontal_flip: bool = False) -> torch.Tensor:
    # Teacher features deliberately have a separate cache and use only this budget's L caption pairs.
    keys = [(row["image_id"], text, horizontal_flip) for row in rows for text in (row["positive_text"], row["negative_text"])]
    outputs: list[torch.Tensor] = []
    by_id = {row["image_id"]: row for row in rows}
    mean, std = MEAN.to(DEVICE), STD.to(DEVICE)
    descriptor.eval()
    for begin in range(0, len(keys), 16):
        group = keys[begin:begin + 16]
        missing = [(image_id, text, view) for image_id, text, view in group
                   if hashlib.sha256((image_id + "\0" + str(int(view)) + "\0" + text).encode()).hexdigest() not in cache]
        if missing:
            images = []
            for image_id, text, view in missing:
                row = by_id[image_id]
                array = decode_image(image_root, row)
                images.append(np.ascontiguousarray(array[:, ::-1, :]) if view else array)
            pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
            pixels = (pixels - mean) / std
            raw = tokenizer([t for _, t, _ in missing], padding=False, truncation=False)["input_ids"]
            if any(len(ids) > MAX_TEXT_TOKENS for ids in raw):
                raise ValueError(f"Teacher caption exceeds registered {MAX_TEXT_TOKENS}-token limit; refusing truncation")
            encoded = tokenizer([t for _, t, _ in missing], padding=True, truncation=False, max_length=MAX_TEXT_TOKENS, return_tensors="pt")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                image_v = descriptor.encode_image(pixels)
                text_v = descriptor.encode_text(encoded["input_ids"].to(DEVICE), encoded["attention_mask"].to(DEVICE))
                relations = descriptor.relation_features(image_v, text_v).float().cpu()
            for (image_id, text, view), item in zip(missing, relations):
                cache[hashlib.sha256((image_id + "\0" + str(int(view)) + "\0" + text).encode()).hexdigest()] = item.contiguous()
        outputs.extend(cache[hashlib.sha256((image_id + "\0" + str(int(view)) + "\0" + text).encode()).hexdigest()]
                       for image_id, text, view in group)
    return torch.stack(outputs)


def train_teacher(cfg: dict[str, Any], train_rows: list[dict[str, str]], val_rows: list[dict[str, str]],
                  tokenizer: Any, source_fp: dict[str, Any]) -> Path:
    """Train the per-budget relation teacher only on its own positive/negative L pairs."""
    from pair_model import FrozenALBEFDescriptor, PairRelationTeacher
    smoke_tag = cfg["run_id"].split("_smoke", 1)[1] if "_smoke" in cfg["run_id"] else None
    teacher_name = f"teacher_l{cfg['budget']}_s{SEED}" + (f"_smoke{smoke_tag}" if smoke_tag else "")
    out = OUTPUTS / teacher_name
    out.mkdir(parents=True, exist_ok=True)
    last, best, result_path = out / "last.pt", out / "best.pt", out / "result.json"
    provenance = {
        "source": source_fp, "budget": cfg["budget"], "model": cfg["model"], "kind": "pairusa_teacher",
        "smoke_inspection_only": smoke_tag is not None,
        "teacher_training": {key: cfg["training"][key] for key in ("teacher_batch", "teacher_lr", "teacher_max_epochs", "teacher_patience")},
        "train_pair_file_sha256": sha256(Path(cfg["pair_file"]) if Path(cfg["pair_file"]).is_absolute() else ROOT / cfg["pair_file"]),
        "validation_pair_file_sha256": sha256(Path(cfg["validation_file"]) if Path(cfg["validation_file"]).is_absolute() else ROOT / cfg["validation_file"]),
    }
    target_path = out / "positive_targets.pt"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["provenance"] != provenance:
            raise RuntimeError("Completed teacher provenance changed")
        target = torch.load(target_path, map_location="cpu", weights_only=False)
        if target["provenance"] != provenance:
            raise RuntimeError("Saved teacher targets provenance changed")
        return best
    descriptor = FrozenALBEFDescriptor(cfg["model"]).to(DEVICE).eval()
    for parameter in descriptor.parameters(): parameter.requires_grad_(False)
    teacher = PairRelationTeacher(SEED).to(DEVICE)
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg["training"]["teacher_lr"], weight_decay=0.01)
    cache: dict[str, torch.Tensor] = {}
    train_x = teacher_descriptors(train_rows, descriptor, tokenizer, Path(cfg["image_root"]), cache)
    val_x = teacher_descriptors(val_rows, descriptor, tokenizer, Path(cfg["image_root"]), cache)
    train_y = torch.tensor([1, 0] * len(train_rows), dtype=torch.float32)
    val_y = torch.tensor([1, 0] * len(val_rows), dtype=torch.float32)
    start_epoch, best_auc, stale = 0, -float("inf"), 0
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        if ckpt["provenance"] != provenance: raise RuntimeError("Teacher checkpoint provenance changed")
        teacher.load_state_dict(ckpt["teacher"]); opt.load_state_dict(ckpt["optimizer"])
        start_epoch, best_auc, stale = ckpt["epoch"], ckpt["best_auc"], ckpt["stale"]
    for epoch in range(start_epoch + 1, cfg["training"]["teacher_max_epochs"] + 1):
        teacher.train()
        # One logical teacher batch is 16 distinct anchors: 16 true + 16 false pairs.
        order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(fixed_seed(SEED, epoch, cfg["budget"])))
        loss_total = 0.0
        for begin in range(0, len(order), cfg["training"]["teacher_batch"]):
            anchors = order[begin:begin + cfg["training"]["teacher_batch"]]
            idx = torch.stack((2 * anchors, 2 * anchors + 1), dim=1).flatten()
            _, logit = teacher(train_x[idx].to(DEVICE))
            loss = F.binary_cross_entropy_with_logits(logit.float(), train_y[idx].to(DEVICE))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite teacher loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True); loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in teacher.parameters()):
                raise FloatingPointError(f"Nonfinite teacher gradient at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0); opt.step()
            loss_total += float(loss.detach()) * len(idx)
        teacher.eval()
        with torch.inference_mode(): _, val_logit = teacher(val_x.to(DEVICE))
        auc = float(roc_auc_score(val_y.numpy(), val_logit.float().cpu().numpy()))
        record = {"epoch": epoch, "train_bce": loss_total / len(train_x), "val_auroc": auc, "utc": now()}
        append_jsonl(out / "train.jsonl", record)
        if auc > best_auc + 1e-12:
            best_auc, stale = auc, 0
            atomic_torch(best, {"provenance": provenance, "epoch": epoch, "teacher": teacher.state_dict(), "val_auroc": auc})
        else: stale += 1
        atomic_torch(last, {"provenance": provenance, "epoch": epoch, "teacher": teacher.state_dict(), "optimizer": opt.state_dict(), "best_auc": best_auc, "stale": stale})
        if stale >= cfg["training"]["teacher_patience"]: break
    ck = torch.load(best, map_location="cpu", weights_only=False)
    teacher.load_state_dict(ck["teacher"]); teacher.eval()
    def infer_positive(descriptors: torch.Tensor) -> torch.Tensor:
        pieces = []
        with torch.inference_mode():
            for begin in range(0, len(descriptors), 256):
                pieces.append(teacher(descriptors[begin:begin + 256].to(DEVICE))[0].float().cpu())
        return torch.cat(pieces, dim=0)[::2].contiguous()
    positive_vectors = infer_positive(train_x)
    flipped_x = teacher_descriptors(train_rows, descriptor, tokenizer, Path(cfg["image_root"]), cache,
                                    horizontal_flip=True)
    flipped_positive_vectors = infer_positive(flipped_x)
    # Diagnostic only: a fixed 16-positive subset is sufficient to catch a constant/nonfinite target.
    probe = positive_vectors[:16].to(DEVICE)
    unit = F.normalize(probe.float(), dim=-1); sim = unit @ unit.T
    offdiag = sim[~torch.eye(len(sim), dtype=torch.bool, device=sim.device)]
    if not torch.isfinite(offdiag).all() or float(offdiag.var()) <= 1e-12:
        raise RuntimeError("Pair-USA teacher relation is numerically collapsed")
    atomic_torch(target_path, {"provenance": provenance, "image_ids": [r["image_id"] for r in train_rows],
                               "view_ids": ["canonical", "horizontal_flip"],
                               "positive_vectors": torch.stack((positive_vectors, flipped_positive_vectors), dim=1).half()})
    atomic_json(result_path, {"state": "completed", "best_val_auroc": ck["val_auroc"], "provenance": provenance,
                               "non_diagonal_similarity_variance": float(offdiag.var()), "completed_utc": now()})
    return best


def run_student(cfg: dict[str, Any], train_rows: list[dict[str, str]], val_rows: list[dict[str, str]],
                tokenizer: Any, source_fp: dict[str, Any]) -> dict[str, Any]:
    from pair_model import PairITMModel, pair_usa_loss
    out = OUTPUTS / cfg["run_id"]; out.mkdir(parents=True, exist_ok=True)
    base_provenance = {"source": source_fp, "run": cfg, "test_evaluation": "disabled_by_user"}
    result_path, last_path, best_path = out / "result.json", out / "last.pt", out / "best.pt"
    if result_path.exists():
        completed = json.loads(result_path.read_text(encoding="utf-8"))
        existing = completed.get("provenance", {})
        if existing.get("base", existing) != base_provenance:
            raise RuntimeError("Completed student provenance changed")
        if cfg["method"].startswith("B"):
            artifacts = existing.get("teacher_artifacts", {})
            if not artifacts:
                raise RuntimeError("Completed Pair-USA result is missing teacher artifact provenance")
            for file_name, expected in artifacts.items():
                if sha256(Path(file_name)) != expected:
                    raise RuntimeError("Completed Pair-USA teacher artifact changed")
        return completed
    provenance: dict[str, Any] = {"base": base_provenance}
    atomic_json(out / "status.json", {"state": "preparing_teacher" if cfg["method"].startswith("B") else "initializing_model",
                                       "step": 0, "target_steps": cfg["training"]["max_steps"], "updated_utc": now(), "provenance": provenance})
    teacher_cpu: torch.Tensor | None = None
    if cfg["method"].startswith("B"):
        # Train/cache the separate frozen teacher before retaining student activations on the GPU.
        teacher_path = train_teacher(cfg, train_rows, val_rows, tokenizer, source_fp)
        teacher_target_path = teacher_path.parent / "positive_targets.pt"
        teacher_target = torch.load(teacher_target_path, map_location="cpu", weights_only=False)
        if teacher_target["provenance"]["source"] != source_fp:
            raise RuntimeError("Teacher targets do not match this registered input")
        if teacher_target["image_ids"] != [row["image_id"] for row in train_rows]:
            raise RuntimeError("Teacher targets are not aligned with this budget's approved anchors")
        teacher_cpu = teacher_target["positive_vectors"].float().contiguous()
        if teacher_target.get("view_ids") != ["canonical", "horizontal_flip"] or teacher_cpu.shape != (len(train_rows), 2, 256):
            raise RuntimeError("Teacher positive targets do not match the registered canonical/flip views")
        provenance["teacher_artifacts"] = {str(teacher_path): sha256(teacher_path),
                                            str(teacher_target_path): sha256(teacher_target_path)}
    model = PairITMModel({**cfg["model"], "use_pairusa": cfg["method"].startswith("B")}).to(DEVICE)
    initial = model.trainable_state()
    init_hash = hashlib.sha256(b"".join(v.numpy().tobytes() for k, v in sorted(initial.items()) if "student_projection" not in k and "temperature" not in k)).hexdigest()
    opt, scaler = optimizer_for(model, cfg), torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda", init_scale=1024.0)
    cache = PairCache(model, tokenizer, Path(cfg["image_root"]))
    atomic_json(out / "status.json", {"state": "caching_frozen_prefixes", "target_steps": cfg["training"]["max_steps"], "updated_utc": now(), "provenance": provenance})
    print(f"[{now()}] {cfg['run_id']}: caching {len(train_rows)} training and {len(val_rows)} validation anchors", flush=True)
    cache.prime_images(train_rows + val_rows, flips=True)
    cache.prime_text([x for r in train_rows + val_rows for x in (r["positive_text"], r["negative_text"])])
    model.train()
    teacher = teacher_cpu.to(DEVICE).detach() if teacher_cpu is not None else None
    step0, best_step, best_key = 0, 0, (-float("inf"), -float("inf"))
    if last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        if ckpt["provenance"] != provenance: raise RuntimeError("Student checkpoint provenance changed")
        required = {"torch_rng", "cuda_rng", "python_rng", "numpy_rng", "physical_batch",
                    "best_snapshot", "best_predictions"}
        if not required <= set(ckpt) or ckpt["physical_batch"] not in (16, 8, 4):
            raise RuntimeError("Incomplete crop student recovery state")
        if set(ckpt["trainable_state"]) != set(model.trainable_state()) or any(
                not torch.isfinite(v).all() for v in ckpt["trainable_state"].values()):
            raise RuntimeError("Recovery trainable tensors are invalid")
        model.load_trainable_state(ckpt["trainable_state"]); opt.load_state_dict(ckpt["optimizer"]); scaler.load_state_dict(ckpt["scaler"])
        step0, best_step, best_key = ckpt["step"], ckpt["best_step"], tuple(ckpt["best_key"])
        if not 1 <= step0 <= cfg["training"]["max_steps"] or step0 % cfg["training"]["checkpoint_every"]:
            raise RuntimeError("Invalid durable student step")
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in ckpt["cuda_rng"]])
        random.setstate(ckpt["python_rng"]); np.random.set_state(ckpt["numpy_rng"])
        model.fusion_chunk_size = ckpt["physical_batch"]
        # A validation written after last.pt may be ahead of the durable state.
        from ot_stage_fast import reconcile_logs
        reconcile_logs(out, step0)
        if ckpt["best_snapshot"] is not None:
            import shutil
            for name in ("best.pt", "best_validation_predictions.csv"):
                path = out / name
                if path.exists():
                    backup = out / "recovery_backups" / f"{time.time_ns()}_{name}"
                    backup.parent.mkdir(exist_ok=True)
                    shutil.copy2(path, backup)
            atomic_torch(best_path, ckpt["best_snapshot"])
            atomic_bytes(out / "best_validation_predictions.csv", ckpt["best_predictions"])
    atomic_json(out / "status.json", {"state": "running", "step": step0, "target_steps": cfg["training"]["max_steps"], "provenance": provenance, "updated_utc": now()})
    params = [p for p in model.parameters() if p.requires_grad]
    train_index = {row["image_id"]: index for index, row in enumerate(train_rows)}
    for step in range(step0 + 1, cfg["training"]["max_steps"] + 1):
        model.train()
        selected = choose_unique(train_rows, step)
        indexes = torch.tensor([train_index[row["image_id"]] for row in selected], device=DEVICE)
        views = torch.tensor([fixed_seed(SEED, step, f"flip/{row['image_id']}") & 1 for row in selected], device=DEVICE)
        image, pt, pm, nt, nm = cache.pair_batch(selected, step, augment=True)
        factor = schedule(step, cfg["training"]["warmup_steps"], cfg["training"]["max_steps"])
        for group in opt.param_groups: group["lr"] = group["initial_lr"] * factor
        success = False
        retry = 0
        tick = time.perf_counter()
        while retry < 10:
            opt.zero_grad(set_to_none=True)
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                    logits, student_vec = model.forward_cached(image, pt, pm, nt, nm)
                    itm = 0.5 * (F.binary_cross_entropy_with_logits(logits[:, 0].float(), torch.ones(len(logits), device=DEVICE)) + F.binary_cross_entropy_with_logits(logits[:, 1].float(), torch.zeros(len(logits), device=DEVICE)))
                    lam = pairusa_weight(step, cfg["training"])
                    relation_loss = torch.zeros((), device=DEVICE)
                    if teacher is not None and lam:
                        relation_loss = pair_usa_loss(teacher[indexes, views], student_vec, cfg["training"]["pairusa_teacher_temp"], model.student_temperature())
                    loss = itm + lam * relation_loss
                if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite loss at step {step}")
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                if model.fusion_chunk_size <= 4:
                    raise
                opt.zero_grad(set_to_none=True)
                model.fusion_chunk_size //= 2
                logits = student_vec = itm = relation_loss = loss = None
            else:
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(params, cfg["training"]["max_grad_norm"], error_if_nonfinite=False)
                old_scale = scaler.get_scale(); scaler.step(opt); scaler.update()
                if torch.isfinite(norm): success = True; break
                if scaler.get_scale() >= old_scale: raise FloatingPointError("Nonfinite gradient without AMP scale backoff")
                retry += 1
            gc.collect(); torch.cuda.empty_cache()
        if not success: raise FloatingPointError(f"AMP overflow recovery exhausted at step {step}")
        record = {"step": step, "itm_bce": float(itm.detach()), "pairusa": float(relation_loss.detach()), "lambda": lam,
                  "loss": float(loss.detach()), "grad_norm": float(norm), "loss_scale": scaler.get_scale(), "overflow_retries": retry,
                  "physical_batch": model.fusion_chunk_size, "step_seconds": time.perf_counter() - tick, "utc": now()}
        append_jsonl(out / "train.jsonl", record)
        evaluated = step % cfg["training"]["eval_every"] == 0
        if evaluated:
            val, val_scores = validation(model, cache, val_rows); append_jsonl(out / "validation.jsonl", {"step": step, **val, "utc": now()})
            key = (val["paired_accuracy"], val["auroc"])
            if key > best_key:
                best_key, best_step = key, step
                atomic_torch(best_path, {"provenance": provenance, "step": step, "trainable_state": model.trainable_state(), "validation": val, "initial_shared_sha256": init_hash})
                atomic_bytes(out / "best_validation_predictions.csv", validation_prediction_csv(val_rows, val_scores, val["validation_selected_threshold"]))
        if evaluated or step % cfg["training"]["checkpoint_every"] == 0:
            atomic_torch(last_path, {"provenance": provenance, "step": step, "trainable_state": model.trainable_state(),
                "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "best_step": best_step, "best_key": list(best_key),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "physical_batch": model.fusion_chunk_size,
                "best_snapshot": torch.load(best_path, map_location="cpu", weights_only=False) if best_path.exists() else None,
                "best_predictions": (out / "best_validation_predictions.csv").read_bytes()
                    if (out / "best_validation_predictions.csv").exists() else None})
            atomic_json(out / "status.json", {"state": "running", "step": step, "target_steps": cfg["training"]["max_steps"], "best_step": best_step, "best_paired_accuracy": best_key[0], "best_auroc": best_key[1], "provenance": provenance, "updated_utc": now()})
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    result = {"run_id": cfg["run_id"], "state": "completed_validation", "best_step": selected["step"], "best_validation": selected["validation"], "initial_shared_sha256": init_hash, "provenance": provenance, "completed_utc": now()}
    atomic_json(result_path, result); atomic_json(out / "status.json", {"state": "completed_validation", "step": cfg["training"]["max_steps"], "best_step": selected["step"], "provenance": provenance, "updated_utc": now()})
    return result


def run_queue(smoke_steps: int | None = None, allow_unqualified_smoke: bool = False) -> None:
    if DEVICE.type != "cuda": raise RuntimeError("CUDA is required for this registered experiment")
    torch.set_num_threads(8)
    manifest = load_manifest(require_gate=smoke_steps is None)
    configure_files(manifest)
    source_fp = source_fingerprint(manifest)
    if md5(Path(model_config()["checkpoint"])) != model_config()["checkpoint_md5_expected"]:
        raise RuntimeError("ALBEF_4M checkpoint MD5 differs from registered provenance")
    if str(CODE) not in sys.path: sys.path.insert(0, str(CODE))
    if str(CORE) not in sys.path: sys.path.insert(0, str(CORE))
    from albef_ssl.model import get_tokenizer
    tokenizer = get_tokenizer(model_config()["tokenizer_path"])
    configs = make_configs(manifest, smoke_steps)
    if smoke_steps is not None: configs = [c for c in configs if c["budget"] == "005"][:2]
    queue_path = OUTPUTS / ("smoke_pipeline_state.json" if smoke_steps else "pipeline_state.json")
    queue = {"state": "starting", "run_ids": [c["run_id"] for c in configs], "source_fingerprint": source_fp, "test_evaluation": "disabled_by_user", "updated_utc": now()}
    if queue_path.exists():
        old = json.loads(queue_path.read_text(encoding="utf-8"))
        if old["source_fingerprint"] != source_fp or old["run_ids"] != queue["run_ids"]: raise RuntimeError("Queue provenance changed; refusing resume")
        queue = old
    atomic_json(queue_path, queue)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark = True
    initial_shared: dict[str, str] = {}
    allowed_validation = source_ids(V2_DATA / "validation.csv")
    for cfg in configs:
        queue.update(state="running", active_run=cfg["run_id"], updated_utc=now()); atomic_json(queue_path, queue)
        train_rows = read_pairs(csv_path(manifest, cfg["pair_file"]), allow_unqualified_smoke=allow_unqualified_smoke)
        val_rows = read_pairs(csv_path(manifest, cfg["validation_file"]), allow_unqualified_smoke=allow_unqualified_smoke)
        if len(train_rows) < 16:
            raise RuntimeError(f"Budget {cfg['budget']} has only {len(train_rows)} qualified anchors; needs 16 unique anchors")
        if len(train_rows) != EXPECTED_COUNTS[cfg["budget"]] or len(val_rows) != 400:
            raise RuntimeError(f"Random policy requires the complete registered L budget and 400 validation anchors")
        allowed_train = source_ids(V2_DATA / "budgets" / f"train_{cfg['budget']}.csv")
        train_ids, validation_ids = {row["image_id"] for row in train_rows}, {row["image_id"] for row in val_rows}
        if not train_ids <= allowed_train:
            raise RuntimeError(f"Pair table includes IDs outside registered L={cfg['budget']} source split")
        if not validation_ids <= allowed_validation or train_ids & validation_ids:
            raise RuntimeError("Pair tables violate the registered train/validation ID boundary")
        result = run_student(cfg, train_rows, val_rows, tokenizer, source_fp)
        if cfg["method"] == "A_bce":
            initial_shared[cfg["budget"]] = result["initial_shared_sha256"]
        elif initial_shared.get(cfg["budget"]) != result["initial_shared_sha256"]:
            raise RuntimeError(f"A/B shared initialization mismatch for budget {cfg['budget']}")
    queue.update(state="completed_validation", active_run=None, completed_utc=now(), updated_utc=now()); atomic_json(queue_path, queue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-configs", action="store_true")
    parser.add_argument("--smoke-steps", type=int, metavar="N", help="isolated 5%% A/B inspection queue; never test evaluation")
    parser.add_argument("--allow-unqualified-smoke", action="store_true", help="allow candidate pairs only for the isolated smoke inspection")
    args = parser.parse_args()
    manifest = load_manifest(require_gate=False)
    if args.write_configs:
        configure_files(manifest); print(ROOT / "configs/queue.json"); return
    if args.smoke_steps is not None and not 2 <= args.smoke_steps <= 20:
        parser.error("--smoke-steps must be between 2 and 20")
    if args.allow_unqualified_smoke and args.smoke_steps is None:
        parser.error("--allow-unqualified-smoke requires --smoke-steps")
    with process_lock(), prevent_system_sleep():
        run_queue(args.smoke_steps, args.allow_unqualified_smoke)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        atomic_json(OUTPUTS / "pipeline_failure.json", {"error": repr(exc), "traceback": traceback.format_exc(), "utc": now()})
        raise
