"""Versioned post-5% cassava queue. Never rewrites the first four stages.

The old queue must be handed off only after 5% S4 has a complete result.
This entry point has separate run IDs, configs, provenance and pipeline state.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cassava_queue as q
import ot_stage_fast as fast_ot
from albef_ssl.model import get_tokenizer


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs/speed_profile_v1.json"
FAST_PIPELINE = ROOT / "outputs/fast_pipeline_state.json"
FAST_BUDGETS = ("020", "001", "010", "030", "100")
ORIGINAL_BUDGET = "005"
OLD_BASE_CONFIG = q.base_config
OLD_OT_CONFIG = q.ot_config


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fast_run_id(stage: str, budget: str) -> str:
    return f"cassava_itm_fast_{stage.lower()}_l{budget}_s{q.SEED}"


def configure_modules(profile: dict) -> None:
    """Patch only the new process; the old launcher retains its loaded code."""
    q.run_id = fast_run_id
    q.BUDGETS = FAST_BUDGETS
    q.ot = fast_ot
    q.OT_CONFIG = {**q.OT_CONFIG, "tolerance": profile["ot_tolerance"]}

    def base_config(manifest: dict, budget: str, stage: str) -> dict:
        cfg = OLD_BASE_CONFIG(manifest, budget, stage)
        cfg["speed_profile"] = profile["profile"]
        cfg["model"]["fusion_chunk_size"] = profile["fusion_physical_microbatch"]
        return cfg

    def ot_config(manifest: dict, fp: dict, budget: str, stage: str) -> dict:
        cfg = OLD_OT_CONFIG(manifest, fp, budget, stage)
        cfg["speed_profile"] = profile["profile"]
        cfg["model"]["fusion_chunk_size"] = profile["fusion_physical_microbatch"]
        cfg["ot"]["tolerance"] = profile["ot_tolerance"]
        return cfg

    q.base_config = base_config
    q.ot_config = ot_config


def verify_profile(profile: dict) -> None:
    expected = {"profile": "post_5pct_fast_v1", "applies_to_budgets": list(FAST_BUDGETS),
                "preserved_completed_budget": ORIGINAL_BUDGET, "training_seed": q.SEED,
                "fusion_physical_microbatch": 16, "oom_microbatch_backoff": [16, 8, 4],
                "logical_labeled_positive_anchors": 16, "logical_unlabeled_pairs": 32,
                "amp_dtype": "float16", "ot_tolerance": 1e-4, "ot_epsilon": 0.1,
                "ot_max_iterations": 100, "student_steps_per_stage": 1600}
    if any(profile.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Speed profile differs from the registered post-5% acceleration")


def fingerprint(original: dict) -> dict:
    result = copy.deepcopy(original)
    for rel in ("code/cassava_fast_queue.py", "code/ot_stage_fast.py",
                "configs/speed_profile_v1.json", "reports/加速交接登记_20261001.md"):
        result[rel] = q.digest(ROOT / rel)
    return result


def check_completed_five_percent(original_fp: dict) -> dict:
    receipt = {}
    for stage in ("S1", "S2", "S3", "S4"):
        identity = f"cassava_itm_{stage.lower()}_l005_s{q.SEED}"
        directory = ROOT / "outputs" / identity
        result_path, best_path, status_path = (directory / name for name in ("result.json", "best.pt", "status.json"))
        result, status = read(result_path), read(status_path)
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        provenance = result["provenance"]
        result_source = provenance.get("source", provenance.get("base", {}).get("source"))
        if (result["run_id"] != identity or result["state"] != "completed_validation"
                or status["state"] != "completed_validation" or status["step"] != 1600
                or result_source != original_fp or best["provenance"] != provenance
                or best["step"] != result["best_step"]
                or best["validation"] != result["best_validation"]):
            raise RuntimeError(f"Five-percent parent result is not complete and immutable: {identity}")
        receipt[identity] = {"result_sha256": q.digest(result_path),
                             "best_sha256": q.digest(best_path), "best_step": best["step"]}
    return receipt


def check_input_state(require_boundary: bool = False) -> tuple[dict, dict, dict, dict]:
    profile = read(PROFILE_PATH)
    verify_profile(profile)
    manifest, original_fp = q.check_inputs()
    original_pipeline = read(ROOT / "outputs/pipeline_state.json")
    expected_ids = [f"cassava_itm_{stage.lower()}_l{budget}_s{q.SEED}"
                    for budget in ("005", *FAST_BUDGETS)
                    for stage in (("S1", "S2") if budget == "100" else ("S1", "S2", "S3", "S4"))]
    if (original_pipeline["source_fingerprint"] != original_fp
            or original_pipeline["run_ids"] != expected_ids):
        raise RuntimeError("Original queue identity or input fingerprint differs")
    if require_boundary:
        check_completed_five_percent(original_fp)
    configure_modules(profile)
    return manifest, fingerprint(original_fp), profile, original_fp


def smoke(manifest: dict, fp: dict, profile: dict) -> dict:
    if q.base.DEVICE.type != "cuda":
        raise RuntimeError("Acceleration smoke requires CUDA")
    cfg = q.base_config(manifest, "100", "S1")
    rows = q.base.read_pairs(ROOT / cfg["pair_file"])
    tokenizer = get_tokenizer(cfg["model"]["tokenizer_path"])
    ranked = []
    for begin in range(0, len(rows), 128):
        group = rows[begin:begin + 128]
        lengths = tokenizer([row["positive_text"] for row in group], padding=False,
                            truncation=False)["input_ids"]
        ranked.extend((len(ids), row) for ids, row in zip(lengths, group))
    selected = [row for _, row in sorted(ranked, key=lambda item: (-item[0], item[1]["image_id"]))[:16]]
    model = q.pair_model.PairITMModel(cfg["model"]).to(q.base.DEVICE)
    cache = q.base.PairCache(model, tokenizer, Path(cfg["image_root"]))
    cache.prime_images(selected, flips=True)
    cache.prime_text([text for row in selected for text in (row["positive_text"], row["negative_text"])])
    inputs = cache.pair_batch(selected, 1, augment=True)
    optimizer = q.base.optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    before = q.tensor_digest(model.trainable_state())
    torch.cuda.reset_peak_memory_stats()
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        logits, _ = model.forward_cached(*inputs)
        loss = 0.5 * (F.binary_cross_entropy_with_logits(logits[:, 0].float(), torch.ones(16, device=q.base.DEVICE))
                      + F.binary_cross_entropy_with_logits(logits[:, 1].float(), torch.zeros(16, device=q.base.DEVICE)))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError("Acceleration smoke loss or gradient is nonfinite")
    scaler.step(optimizer)
    scaler.update()
    if q.tensor_digest(model.trainable_state()) == before:
        raise RuntimeError("Acceleration smoke made no optimizer update")
    proof = {"passed": True, "scope": "engineering_only_not_formal_result", "physical_microbatch": 16,
             "max_caption_tokens": max(length for length, _ in ranked),
             "loss": float(loss.detach()), "grad_norm": float(norm),
             "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "source_fingerprint": fp,
             "profile_sha256": q.digest(PROFILE_PATH), "utc": q.base.now()}
    q.base.atomic_json(ROOT / "audit/acceleration_smoke.json", proof)
    return proof


def run(manifest: dict, fp: dict, profile: dict, original_fp: dict) -> None:
    completed = check_completed_five_percent(original_fp)
    proof = read(ROOT / "audit/acceleration_smoke.json")
    if (not proof["passed"] or proof["source_fingerprint"] != fp
            or proof["physical_microbatch"] != profile["fusion_physical_microbatch"]):
        raise RuntimeError("Exact fast-source real GPU smoke has not passed")
    if q.base.DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required")
    run_ids = [fast_run_id(stage, budget) for stage, budget in q.queue_order()]
    state = {"state": "starting", "run_ids": run_ids, "source_fingerprint": fp,
             "completed_5pct_sha256": completed, "speed_profile": profile,
             "seed": q.SEED, "test_evaluation": False, "started_utc": q.base.now()}
    with q.base.process_lock(), q.base.prevent_system_sleep():
        if FAST_PIPELINE.exists():
            old = read(FAST_PIPELINE)
            if (old["source_fingerprint"] != fp or old["run_ids"] != run_ids
                    or old["completed_5pct_sha256"] != completed):
                raise RuntimeError("Accelerated queue provenance differs; refusing resume")
            state["started_utc"] = old["started_utc"]
        q.base.atomic_json(FAST_PIPELINE, state)
        torch.manual_seed(q.SEED)
        torch.cuda.manual_seed_all(q.SEED)
        np.random.seed(q.SEED % (2**32))
        random.seed(q.SEED)
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(8)
        tokenizer = get_tokenizer(q.base.model_config()["tokenizer_path"])
        try:
            for stage, budget in q.queue_order():
                state.update(state="running", active_run=fast_run_id(stage, budget), updated_utc=q.base.now())
                q.base.atomic_json(FAST_PIPELINE, state)
                result = q.run_stage(manifest, fp, tokenizer, stage, budget)
                print(json.dumps({"completed": result["run_id"], "best_step": result["best_step"],
                                  "paired_accuracy": result["best_validation"]["paired_accuracy"]}), flush=True)
            state.update(state="completed_validation", active_run=None,
                         completed_utc=q.base.now(), updated_utc=q.base.now())
            q.base.atomic_json(FAST_PIPELINE, state)
        except BaseException as exc:
            state.update(state="failed", error=repr(exc), updated_utc=q.base.now())
            q.base.atomic_json(FAST_PIPELINE, state)
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    manifest, fp, profile, original_fp = check_input_state(require_boundary=args.run)
    if args.check:
        print(json.dumps({"inputs_passed": True, "future_stages": len(q.queue_order()),
                          "physical_microbatch": profile["fusion_physical_microbatch"],
                          "ot_tolerance": profile["ot_tolerance"]}, ensure_ascii=False))
    elif args.smoke:
        print(json.dumps(smoke(manifest, fp, profile), ensure_ascii=False), flush=True)
    else:
        run(manifest, fp, profile, original_fp)


if __name__ == "__main__":
    main()
