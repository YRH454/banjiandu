"""Single-GPU, single-seed cassava S1->S2/S3->S4 ITM queue.

--check validates immutable inputs without allocating a model.
--smoke performs a real-data S1 CUDA update outside formal outputs.
--run executes/resumes 22 registered student stages in dependency order.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import legacy_parent as base
import ot_stage as ot
import pair_model
from albef_ssl.model import get_tokenizer


PLAN = ROOT / "configs/plan.json"
SOURCE = Path(r"X:\PATH\datasets\EGOISTyrh\mushubanjiandu")
OUTPUTS = ROOT / "outputs"
SEED = 20260825
BUDGETS = ("005", "020", "001", "010", "030", "100")
COUNT = {"001": 85, "005": 429, "010": 858, "020": 1716, "030": 2574, "100": 8583}
EXTRA = frozenset({"student_projection.0.weight", "student_projection.0.bias",
                   "student_projection.1.weight", "student_projection.1.bias",
                   "log_student_temperature"})
OT_CONFIG = {"epsilon": 0.1, "max_iterations": 100, "tolerance": 1e-5,
             "warmup_start": 100, "warmup_end": 200, "max_weight": 0.1,
             "solver_dtype": "float64", "cost_dtype": "float32",
             "residual": "both_marginals_same_final_plan"}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return base.sha256(path)


def tensor_digest(state: dict) -> str:
    return hashlib.sha256(b"".join(value.detach().cpu().contiguous().numpy().tobytes()
                                    for _, value in sorted(state.items()))).hexdigest()


def immutable_json(path: Path, value: dict) -> None:
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"Registered artifact differs: {path}")
    else:
        base.atomic_json(path, value)


def run_id(stage: str, budget: str) -> str:
    return f"cassava_itm_{stage.lower()}_l{budget}_s{SEED}"


def source_fingerprint(manifest: dict) -> dict:
    source = base.source_fingerprint(manifest)
    paths = [ROOT / "code" / name for name in ("cassava_queue.py", "ot_stage.py", "ot_loss.py", "test_ot_loss.py")]
    paths += [ROOT / "configs/plan.json", ROOT / "reports/执行登记_木薯S1-S4_20261001.md",
              ROOT / "data/source_snapshot.json", ROOT / "data/image_index.jsonl",
              ROOT / "data/u_manifest.json", ROOT / "audit/pair_preparation.json",
              ROOT / "audit/ot_numerical_checks.json",
              ROOT / "data/splits/split_manifest.json"]
    paths += [ROOT / f"data/splits/{name}_ids.txt" for name in ("train", "validation", "test")]
    u = read(ROOT / "data/u_manifest.json")
    paths += [ROOT / item["path"] for item in u["budgets"].values()]
    source.update({str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in paths})
    source["source_revision"] = read(ROOT / "data/source_snapshot.json")["source_revision"]
    source["ALBEF_4M_md5"] = base.md5(Path(base.model_config()["checkpoint"]))
    return source


def check_inputs() -> tuple[dict, dict]:
    plan = read(PLAN)
    if (plan["training_seed"] != SEED or plan["budget_order"] != list(BUDGETS)
            or plan["max_text_tokens_including_special_tokens"] != 384
            or plan["lineage"]["S4"] != "same_budget_S3_best"
            or plan["test_evaluation"] is not False):
        raise RuntimeError("Current plan is not the authorized single-seed 384-token S3->S4 protocol")
    snapshot = read(ROOT / "data/source_snapshot.json")
    source_audit = read(ROOT / "audit/source_audit.json")
    split = read(ROOT / "data/splits/split_manifest.json")
    if (snapshot["source_revision"] != plan["source_revision"]
            or not source_audit["source_integrity_passed"]
            or source_audit["manifest_sha256"] != split["manifest_sha256"]
            or source_audit["captions_sha256"] != split["captions_sha256"]):
        raise RuntimeError("Source/split audit binding failed")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE, text=True).strip()
    if revision != plan["source_revision"] or dirty:
        raise RuntimeError("Downloaded dataset revision changed or has local modifications")
    if (digest(SOURCE / "image_manifest.jsonl") != snapshot["image_manifest_sha256"]
            or digest(SOURCE / "captions.json") != snapshot["captions_sha256"]):
        raise RuntimeError("Source manifest or captions changed")
    manifest = base.load_manifest(require_gate=True)
    gate = read(ROOT / "audit/negative_quality_gate.json")
    u_manifest = read(ROOT / "data/u_manifest.json")
    pair_audit = read(ROOT / "audit/pair_preparation.json")
    numerical = read(ROOT / "audit/ot_numerical_checks.json")
    if (not numerical["passed"] or numerical["check_count"] != 27
            or numerical["source_sha256"]["ot_loss.py"] != digest(ROOT / "code/ot_loss.py")
            or numerical["source_sha256"]["test_ot_loss.py"] != digest(ROOT / "code/test_ot_loss.py")):
        raise RuntimeError("OT numerical contract does not match this solver/source")
    if (not pair_audit["passed"] or u_manifest["max_text_tokens"] != 384
            or u_manifest["source_pair_manifest_sha256"] != digest(ROOT / "data/manifest.json")
            or manifest["source_counts"] != COUNT or manifest["validation_anchors"] != 400):
        raise RuntimeError("Pair/U registration mismatch")
    for budget, item in u_manifest["budgets"].items():
        if (budget == "100" or item["l_count"] != COUNT[budget]
                or item["n_images"] != 8583 - COUNT[budget]
                or item["n_pairs"] != 2 * item["n_images"]
                or digest(ROOT / item["path"]) != item["sha256"]):
            raise RuntimeError(f"U {budget} input mismatch")
        ot.load_u_rows(ROOT / item["path"], item["sha256"])
    train_ids = base.source_ids(ROOT / "data/train_pool_ids.csv")
    val_ids = base.source_ids(ROOT / "data/validation.csv")
    if len(train_ids) != 8583 or len(val_ids) != 1073 or train_ids & val_ids:
        raise RuntimeError("Train/validation ID boundary changed")
    val_rows = base.read_pairs(ROOT / manifest["files"]["validation"])
    if len(val_rows) != 400 or not {row["image_id"] for row in val_rows} <= val_ids:
        raise RuntimeError("Fixed 400-image validation boundary changed")
    for budget in COUNT:
        rows = base.read_pairs(ROOT / manifest["files"]["budgets"][budget])
        allowed = base.source_ids(ROOT / f"data/budgets/train_{budget}.csv")
        if len(rows) != COUNT[budget] or {row["image_id"] for row in rows} != allowed or not allowed <= train_ids:
            raise RuntimeError(f"L budget {budget} boundary changed")
    model = base.model_config()
    if base.md5(Path(model["checkpoint"])) != model["checkpoint_md5_expected"]:
        raise RuntimeError("ALBEF-4M source weight changed")
    fp = source_fingerprint(manifest)
    return manifest, fp


def base_config(manifest: dict, budget: str, stage: str) -> dict:
    original = next(cfg for cfg in base.make_configs(manifest)
                    if cfg["budget"] == budget and cfg["method"] == ("A_bce" if stage == "S1" else "B_bce_pairusa"))
    cfg = copy.deepcopy(original)
    cfg["run_id"] = run_id(stage, budget)
    cfg["stage"] = stage
    cfg["model"]["use_pairusa"] = stage == "S2"
    if stage == "S2":
        source = checked_parent("S1", budget)
        cfg["method"] = "B_bce_pairusa_warmstart_S1_best"
        cfg["warmstart"] = source
        cfg["model"]["warmstart"] = {**source,
            "audit_path": str(OUTPUTS / cfg["run_id"] / "initialization_audit.json")}
    return cfg


def checked_parent(stage: str, budget: str) -> dict:
    identity = run_id(stage, budget)
    folder = OUTPUTS / identity
    result = read(folder / "result.json")
    best_path = folder / "best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if (result["state"] != "completed_validation" or result["run_id"] != identity
            or best["provenance"] != result["provenance"]
            or best["step"] != result["best_step"]
            or best["validation"] != result["best_validation"]
            or len(best["trainable_state"]) != 34
            or any(not torch.isfinite(value).all() for value in best["trainable_state"].values())):
        raise RuntimeError(f"Parent {identity} best/result mismatch")
    return {"parent_kind": stage, "parent_run_id": identity,
            "parent_best_path": str(best_path), "parent_best_sha256": digest(best_path),
            "parent_best_step": best["step"], "parent_result_sha256": digest(folder / "result.json")}


def warmstart_s2(cfg: dict):
    original = pair_model.PairITMModel

    class S1BestWarmModel(original):
        def __init__(self, config):
            super().__init__(config)
            spec = config["warmstart"]
            if digest(Path(spec["parent_best_path"])) != spec["parent_best_sha256"]:
                raise RuntimeError("S1 best changed before S2 initialization")
            checkpoint = torch.load(spec["parent_best_path"], map_location="cpu", weights_only=False)
            initial, source = self.trainable_state(), checkpoint["trainable_state"]
            if len(source) != 34 or set(initial) - set(source) != EXTRA or set(source) - set(initial):
                raise RuntimeError("S1->S2 does not have registered 34+5 key layout")
            for key, value in source.items():
                if initial[key].shape != value.shape or not torch.isfinite(value).all():
                    raise RuntimeError(f"Invalid S1 best tensor: {key}")
            fresh = {key: initial[key].clone() for key in EXTRA}
            self.load_trainable_state({**initial, **source})
            loaded = self.trainable_state()
            if any(not torch.equal(loaded[key], value) for key, value in source.items()):
                raise RuntimeError("S2 did not exactly copy S1 best")
            audit = {"parent_kind": "S1", "parent_best_sha256": spec["parent_best_sha256"],
                     "parent_best_step": spec["parent_best_step"], "copied_keys": 34,
                     "fresh_keys": sorted(EXTRA), "parent_trainable_sha256": tensor_digest(source),
                     "loaded_shared_sha256": tensor_digest({key: loaded[key] for key in source}),
                     "fresh_usa_sha256": tensor_digest(fresh), "exact_copy": True}
            immutable_json(Path(spec["audit_path"]), audit)

    pair_model.PairITMModel = S1BestWarmModel
    return original


def teacher_binding(budget: str, fp: dict, train_rows: list[dict]) -> dict:
    directory = OUTPUTS / f"teacher_l{budget}_s{SEED}"
    result = read(directory / "result.json")
    if (result["state"] != "completed" or result["provenance"]["source"] != fp
            or result["provenance"]["budget"] != budget):
        raise RuntimeError("Cassava teacher provenance mismatch")
    target = directory / "positive_targets.pt"
    ckpt = torch.load(target, map_location="cpu", weights_only=False)
    if (ckpt["image_ids"] != [row["image_id"] for row in train_rows]
            or ckpt["view_ids"] != ["canonical", "horizontal_flip"]
            or tuple(ckpt["positive_vectors"].shape) != (len(train_rows), 2, 256)):
        raise RuntimeError("Teacher targets do not match this L budget")
    return {str(directory / name): digest(directory / name) for name in ("best.pt", "positive_targets.pt")}


def ot_config(manifest: dict, fp: dict, budget: str, stage: str) -> dict:
    parent_stage = "S1" if stage == "S3" else "S3"
    warm = checked_parent(parent_stage, budget)
    s1 = read(ROOT / "configs" / f"{run_id('S1', budget)}.json")
    u = read(ROOT / "data/u_manifest.json")["budgets"][budget]
    l_rows = base.read_pairs(ROOT / manifest["files"]["budgets"][budget])
    warm["s1_best_sha256"] = checked_parent("S1", budget)["parent_best_sha256"]
    if stage == "S4":
        audit_path = OUTPUTS / run_id("S2", budget) / "initialization_audit.json"
        audit = read(audit_path)
        if not audit["exact_copy"] or audit["copied_keys"] != 34:
            raise RuntimeError("S2 initialization audit is incomplete")
        warm["c_initialization_audit"] = str(audit_path)
        warm["c_initialization_sha256"] = digest(audit_path)
        warm["teacher_artifacts"] = teacher_binding(budget, fp, l_rows)
        warm["teacher_result_sha256"] = digest(OUTPUTS / f"teacher_l{budget}_s{SEED}" / "result.json")
    cfg = {"run_id": run_id(stage, budget), "stage": stage,
           "variant": "G3" if stage == "S3" else "G4", "budget": budget, "seed": SEED,
           "model": copy.deepcopy(s1["model"]), "training": copy.deepcopy(s1["training"]),
           "image_root": manifest["image_root"],
           "pair_file": str(ROOT / manifest["files"]["budgets"][budget]),
           "validation_file": str(ROOT / manifest["files"]["validation"]),
           "u_file": str(ROOT / u["path"]), "u_sha256": u["sha256"],
           "u_n_images": u["n_images"], "u_n_pairs": u["n_pairs"],
           "warmstart": warm, "ot": OT_CONFIG,
           "u_weak": {"resize": 384, "interpolation": "bicubic", "normalization": "CLIP"},
           "u_strong": {"resize": 384, "brightness": 0.1, "contrast": 0.1,
                        "saturation": 0.0, "hue": 0.0, "crop": False, "flip": False,
                        "normalization": "CLIP"},
           "unlabeled_batch": 32, "max_cpu_prefix_cache_gib": 48,
           "caption_protocol": "training_pool_blind_captions_available_matching_labels_masked_on_U",
           "test_evaluation": False}
    cfg["model"]["use_pairusa"] = stage == "S4"
    cfg["model"]["fusion_chunk_size"] = 8
    return cfg


def run_stage(manifest: dict, fp: dict, tokenizer, stage: str, budget: str) -> dict:
    l_rows = base.read_pairs(ROOT / manifest["files"]["budgets"][budget])
    v_rows = base.read_pairs(ROOT / manifest["files"]["validation"])
    if stage in ("S1", "S2"):
        cfg = base_config(manifest, budget, stage)
    else:
        cfg = ot_config(manifest, fp, budget, stage)
    immutable_json(ROOT / "configs" / f"{cfg['run_id']}.json", cfg)
    if stage == "S2":
        original = warmstart_s2(cfg)
        try:
            result = base.run_student(cfg, l_rows, v_rows, tokenizer, fp)
        finally:
            pair_model.PairITMModel = original
        proof = read(OUTPUTS / cfg["run_id"] / "initialization_audit.json")
        if result["initial_shared_sha256"] != proof["parent_trainable_sha256"]:
            raise RuntimeError("S2 shared initialization differs from S1 best")
        return result
    if stage == "S1":
        return base.run_student(cfg, l_rows, v_rows, tokenizer, fp)
    return ot.run_one(cfg, fp)


def smoke(manifest: dict) -> None:
    if base.DEVICE.type != "cuda":
        raise RuntimeError("Real-data smoke requires CUDA")
    import pair_model as models
    cfg = base_config(manifest, "100", "S1")
    rows = base.read_pairs(ROOT / cfg["pair_file"])
    tokenizer = get_tokenizer(cfg["model"]["tokenizer_path"])
    ranked = []
    for begin in range(0, len(rows), 128):
        group = rows[begin:begin + 128]
        lengths = tokenizer([row["positive_text"] for row in group],
                            padding=False, truncation=False)["input_ids"]
        ranked.extend((len(ids), row) for ids, row in zip(lengths, group))
    selected = [row for _, row in sorted(ranked, key=lambda item: (-item[0], item[1]["image_id"]))[:16]]
    if max(length for length, _ in ranked) != 360:
        raise RuntimeError("Observed maximum blind-caption length changed")
    model = models.PairITMModel(cfg["model"]).to(base.DEVICE)
    cache = base.PairCache(model, tokenizer, Path(cfg["image_root"]))
    cache.prime_images(selected, flips=True)
    cache.prime_text([text for row in selected for text in (row["positive_text"], row["negative_text"])])
    inputs = cache.pair_batch(selected, 1, augment=True)
    optimizer = base.optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    before = tensor_digest(model.trainable_state())
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        logits, _ = model.forward_cached(*inputs)
        loss = 0.5 * (F.binary_cross_entropy_with_logits(logits[:, 0].float(), torch.ones(16, device=base.DEVICE))
                      + F.binary_cross_entropy_with_logits(logits[:, 1].float(), torch.zeros(16, device=base.DEVICE)))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError("Real-data smoke loss/gradient is nonfinite")
    scaler.step(optimizer)
    scaler.update()
    if tensor_digest(model.trainable_state()) == before:
        raise RuntimeError("Real-data smoke made no student update")
    proof = {"passed": True, "scope": "engineering_only_not_formal_result",
             "budget": "100", "step": 1, "loss": float(loss.detach()),
             "grad_norm": float(norm), "max_text_tokens": 384,
             "max_observed_caption_tokens": 360,
             "model_keys": len(model.trainable_state()),
             "source_fingerprint": source_fingerprint(manifest)}
    immutable_json(ROOT / "audit/real_gpu_smoke_long_caption_v2.json", proof)
    print(json.dumps({"smoke_passed": True, "loss": proof["loss"],
                      "trainable_keys": proof["model_keys"]}), flush=True)


def queue_order() -> list[tuple[str, str]]:
    return [(stage, budget) for budget in BUDGETS
            for stage in (("S1", "S2") if budget == "100" else ("S1", "S2", "S3", "S4"))]


def run(manifest: dict, fp: dict) -> None:
    smoke_audit = read(ROOT / "audit/real_gpu_smoke_long_caption_v2.json")
    if not smoke_audit["passed"] or smoke_audit["source_fingerprint"] != fp:
        raise RuntimeError("Exact-source real GPU smoke has not passed")
    if base.DEVICE.type != "cuda":
        raise RuntimeError("Formal cassava queue requires CUDA")
    torch.set_num_threads(8)
    state_path = OUTPUTS / "pipeline_state.json"
    state = {"state": "starting", "run_ids": [run_id(stage, budget) for stage, budget in queue_order()],
             "source_fingerprint": fp, "test_evaluation": False, "seed": SEED,
             "started_utc": base.now()}
    with base.process_lock(), base.prevent_system_sleep():
        if state_path.exists():
            old = read(state_path)
            if old["source_fingerprint"] != fp or old["run_ids"] != state["run_ids"]:
                raise RuntimeError("Existing queue provenance differs; refusing resume")
            state["started_utc"] = old["started_utc"]
        base.atomic_json(state_path, state)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED % (2**32))
        random.seed(SEED)
        torch.backends.cudnn.benchmark = True
        tokenizer = get_tokenizer(base.model_config()["tokenizer_path"])
        try:
            for stage, budget in queue_order():
                state.update(state="running", active_run=run_id(stage, budget), updated_utc=base.now())
                base.atomic_json(state_path, state)
                result = run_stage(manifest, fp, tokenizer, stage, budget)
                print(json.dumps({"completed": result["run_id"],
                                  "best_step": result["best_step"],
                                  "paired_accuracy": result["best_validation"]["paired_accuracy"]}), flush=True)
            state.update(state="completed_validation", active_run=None,
                         completed_utc=base.now(), updated_utc=base.now())
            base.atomic_json(state_path, state)
        except BaseException as exc:
            state.update(state="failed", error=repr(exc), updated_utc=base.now())
            base.atomic_json(state_path, state)
            base.atomic_json(OUTPUTS / "pipeline_failure.json",
                             {"error": repr(exc), "traceback": traceback.format_exc(), "utc": base.now()})
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    manifest, fp = check_inputs()
    if args.check:
        print(json.dumps({"inputs_passed": True, "source_files": len(fp),
                          "student_stages": len(queue_order())}, ensure_ascii=False))
    elif args.smoke:
        smoke(manifest)
    else:
        run(manifest, fp)


if __name__ == "__main__":
    main()
