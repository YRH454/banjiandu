# -*- coding: utf-8 -*-
"""Actual-device engineering checks for the OT continuation implementation.

This program only writes audit/training_verification.json and artifacts in
audit/integration_smoke/. Its successful steps are not scientific results.
It never starts the formal queue or modifies registered source experiments.
"""
from __future__ import annotations

import builtins
import contextlib
import copy
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
SMOKE = ROOT / "audit/integration_smoke"
AB = ROOT.parent / "apple_itm_pairusa_random_v2"
C_ROOT = ROOT.parent / "apple_itm_pairusa_warmstart_v1"
PUBLIC_FIELDS = {"pair_id", "image_id", "image_path", "text", "text_sha256"}
USA_KEYS = {"log_student_temperature", "student_projection.0.bias", "student_projection.0.weight",
            "student_projection.1.bias", "student_projection.1.weight"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tensor_digest(state):
    return hashlib.sha256(b"".join(value.detach().cpu().contiguous().numpy().tobytes()
                                  for _, value in sorted(state.items()))).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_queue():
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    specification = importlib.util.spec_from_file_location("new_ot_queue_engineering_check", CODE / "train_queue.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@contextlib.contextmanager
def deny_private_data_reads():
    """Make accessing hidden construction provenance impossible during training."""
    blocked_root = str((ROOT / "audit_private").resolve()).casefold()
    attempts = []
    observed = set()

    def wrap(original):
        def checked(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                path = str(Path(os.fsdecode(file)).resolve()).casefold()
                if path == blocked_root or path.startswith(blocked_root + os.sep):
                    attempts.append(path)
                    raise AssertionError("Training attempted to access hidden U provenance")
                observed.add(path)
            return original(file, *args, **kwargs)
        return checked

    stats = {"attempts": attempts, "observed": observed}
    with mock.patch("builtins.open", wrap(builtins.open)), mock.patch("io.open", wrap(io.open)), mock.patch("os.open", wrap(os.open)):
        yield stats


def compare_trees(left, right, atol=2e-6):
    """Check model, optimizer, scaler, and RNG state, not only training loss."""
    stats = {"atol": atol, "rtol": 0.0, "tensors": 0, "scalars": 0, "max_abs_difference": 0.0}

    def check(a, b, path):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                raise AssertionError(f"Tensor metadata mismatch: {path}")
            a, b = a.detach().cpu(), b.detach().cpu()
            stats["tensors"] += 1
            if a.is_floating_point():
                if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
                    raise AssertionError(f"Nonfinite state: {path}")
                difference = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
                stats["max_abs_difference"] = max(stats["max_abs_difference"], difference)
                if difference > atol:
                    raise AssertionError(f"State mismatch {path}: max abs {difference} > {atol}")
            elif not torch.equal(a, b):
                raise AssertionError(f"Integer/RNG tensor mismatch: {path}")
        elif isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or not np.array_equal(a, b):
                raise AssertionError(f"NumPy/RNG state mismatch: {path}")
        elif isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                raise AssertionError(f"Dictionary key mismatch: {path}")
            for key in a:
                check(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (list, tuple)):
            if not isinstance(b, type(a)) or len(a) != len(b):
                raise AssertionError(f"Sequence mismatch: {path}")
            for index, (first, second) in enumerate(zip(a, b)):
                check(first, second, f"{path}/{index}")
        elif isinstance(a, float):
            stats["scalars"] += 1
            if not math.isfinite(a) or not isinstance(b, (float, int)) or abs(a - b) > atol:
                raise AssertionError(f"Float mismatch: {path}: {a} / {b}")
        else:
            stats["scalars"] += 1
            if a != b:
                raise AssertionError(f"Scalar mismatch: {path}: {a} / {b}")

    check(left, right, "state")
    return stats


def find_mapping(value, required):
    if isinstance(value, dict):
        if set(required) <= set(value):
            return value
        for child in value.values():
            result = find_mapping(child, required)
            if result is not None:
                return result
    return None


def find_value(value, keys):
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for child in value.values():
            result = find_value(child, keys)
            if result is not None:
                return result
    return None


def check_ot_record(record):
    diagnostics = find_mapping(record, ("q_min", "q_max", "row_residual", "col_residual"))
    if diagnostics is None:
        raise AssertionError("Training did not expose actual Sinkhorn diagnostics")
    if not diagnostics["converged"] or max(diagnostics["row_residual"], diagnostics["col_residual"]) > 1e-5:
        raise AssertionError("Actual training OT failed the registered bilateral residual limit")
    if not (0 <= diagnostics["q_min"] <= diagnostics["q_max"] <= 1):
        raise AssertionError("Actual OT targets outside [0,1]")
    if diagnostics["q_requires_grad"]:
        raise AssertionError("OT targets were not detached")
    if (diagnostics["n_anchors"], diagnostics["n_queries"], diagnostics["feature_dim"]) != (32, 32, 768):
        raise AssertionError("Actual OT matrix or representation differs from registered 32x32/768")
    if not (1 <= diagnostics["iterations"] <= 100):
        raise AssertionError("OT iteration count outside frozen setting")
    strong_difference = find_value(record, ("u_strong_weak_pixel_mae", "u_strong_weak_prefix_mae",
                                           "strong_weak_pixel_mae", "strong_weak_prefix_mae"))
    if strong_difference is None or not float(strong_difference) > 0:
        raise AssertionError("No evidence that actual strong U view differs from its weak view")
    return {"sinkhorn": diagnostics, "strong_weak_difference": float(strong_difference)}


def initialization_check(rt, variant="G4"):
    a_path = AB / "outputs/apple_itm_random_A_bce_l005_s20260825/best.pt"
    c_audit_path = C_ROOT / "outputs/apple_itm_random_C_from_A_best_pairusa_l005_s20260825/initialization_audit.json"
    historical = json.loads(c_audit_path.read_text(encoding="utf-8-sig"))
    original = torch.load(a_path, map_location="cpu", weights_only=False)
    a_state, actual = original["trainable_state"], rt.model.trainable_state()
    expected_fresh = USA_KEYS if variant == "G4" else set()
    if len(a_state) != 34 or set(actual) - set(a_state) != expected_fresh or len(actual) != 34 + len(expected_fresh):
        raise AssertionError("Incorrect G3/G4 inherited/shared/new trainable parameter keys")
    for key in a_state:
        if not torch.equal(a_state[key], actual[key]):
            raise AssertionError(f"A-best shared weight was not copied exactly: {key}")
    shared_digest = tensor_digest({key: actual[key] for key in a_state})
    fresh_digest = tensor_digest({key: actual[key] for key in expected_fresh}) if expected_fresh else None
    if shared_digest != historical["loaded_shared_sha256"] or (expected_fresh and fresh_digest != historical["fresh_usa_sha256"]):
        raise AssertionError("Current G4 initialization differs from original C initialization audit")
    if sha256(a_path) != historical["a_best_sha256"] or original["step"] != historical["a_best_step"]:
        raise AssertionError("A-best source changed relative to historical C")
    if rt.opt.state:
        raise AssertionError("New-phase optimizer unexpectedly has inherited moments")
    if rt.scaler.get_scale() != 1024:
        raise AssertionError("New-phase scaler is not initialized to 1024")
    if (variant == "G3") != (rt.teacher is None):
        raise AssertionError("USA teacher presence is inconsistent with G3/G4")
    return {"variant": variant, "a_best_sha256": sha256(a_path), "a_best_step": original["step"], "shared_keys": 34,
            "fresh_usa_keys": sorted(expected_fresh), "shared_weights_exact": True,
            "shared_sha256": shared_digest, "fresh_usa_sha256": fresh_digest,
            "historical_c_audit_sha256": sha256(c_audit_path),
            "new_optimizer_empty": True, "initial_scale": 1024}


def checkpoint_guard_check(queue, snapshot):
    """Synthetic metadata fixtures test schema checks; they are never resumable runs."""
    provenance = {"scope": "synthetic_checkpoint_schema_test_only"}
    fixture = {**copy.deepcopy(snapshot), "provenance": provenance, "step": 50,
               "best_snapshot": None, "best_scores": None, "seen_u_pairs": [],
               "seen_u_images": [], "active_seconds": 1.0}
    expected = set(snapshot["trainable_state"])
    queue.validate_resume_checkpoint(fixture, provenance, expected)
    rejected = []

    def must_reject(name, change):
        damaged = copy.deepcopy(fixture)
        change(damaged)
        try:
            queue.validate_resume_checkpoint(damaged, provenance, expected)
        except RuntimeError:
            rejected.append(name)
        else:
            raise AssertionError(f"Resume guard accepted damaged schema fixture: {name}")

    must_reject("missing_optimizer", lambda value: value.pop("optimizer"))
    must_reject("wrong_provenance", lambda value: value.update(provenance={"wrong": True}))
    must_reject("non_durable_step", lambda value: value.update(step=51))
    must_reject("nonpositive_scale", lambda value: value["scaler"].update(scale=0.0))
    must_reject("missing_optimizer_moments", lambda value: value["optimizer"].update(state={}))
    must_reject("nonfinite_model", lambda value: next(iter(value["trainable_state"].values())).fill_(float("nan")))
    must_reject("future_best_step", lambda value: value.update(
        step=200, best_snapshot={"step": 300, "trainable_state": value["trainable_state"], "validation": {}},
        best_scores=torch.zeros((400, 2))))
    return {"scope": "synthetic_metadata_schema_fixtures_not_executed_training_steps",
            "valid_prevalidation_shape_accepted": True, "damaged_cases_rejected": rejected}


def original_c_step(queue, rt, cfg, step):
    """An independent copy of the original C/parent mathematical update path."""
    parent = queue.parent
    from pair_model import pair_usa_loss

    device = queue.DEVICE
    selected = parent.choose_unique(rt.train_rows, step)
    rt.cache.prime_images(selected, flips=True)
    image, pt, pm, nt, nm = rt.cache.pair_batch(selected, step, augment=True)
    rt.model.train()
    index = torch.tensor([rt.train_index[row["image_id"]] for row in selected], device=device)
    views = torch.tensor([parent.fixed_seed(parent.SEED, step, f"flip/{row['image_id']}") & 1 for row in selected], device=device)
    factor = parent.schedule(step, cfg["training"]["warmup_steps"], cfg["training"]["max_steps"])
    for group in rt.opt.param_groups:
        group["lr"] = group["initial_lr"] * factor
    parameters = [value for value in rt.model.parameters() if value.requires_grad]
    teacher = rt.teacher.to(device) if rt.teacher is not None else None
    for retry in range(10):
        rt.opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, vector = rt.model.forward_cached(image, pt, pm, nt, nm)
            bce = 0.5 * (
                F.binary_cross_entropy_with_logits(logits[:, 0].float(), torch.ones(len(logits), device=device))
                + F.binary_cross_entropy_with_logits(logits[:, 1].float(), torch.zeros(len(logits), device=device))
            )
            coefficient = parent.pairusa_weight(step, cfg["training"]) if teacher is not None else 0.0
            usa = torch.zeros((), device=device)
            if teacher is not None and coefficient:
                usa = pair_usa_loss(teacher[index, views], vector, cfg["training"]["pairusa_teacher_temp"],
                                    rt.model.student_temperature())
            loss = bce + coefficient * usa
        if not bool(torch.isfinite(loss)):
            raise AssertionError("Original C reference loss is nonfinite")
        rt.scaler.scale(loss).backward()
        rt.scaler.unscale_(rt.opt)
        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["training"]["max_grad_norm"], error_if_nonfinite=False)
        old_scale = rt.scaler.get_scale()
        rt.scaler.step(rt.opt)
        rt.scaler.update()
        if bool(torch.isfinite(norm)):
            return {"step": step, "itm_bce": float(bce.detach()), "pairusa": float(usa.detach()),
                    "loss": float(loss.detach()), "lambda": coefficient, "overflow_retries": retry,
                    "l_image_ids": [row["image_id"] for row in selected]}
        if rt.scaler.get_scale() >= old_scale:
            raise AssertionError("Original C reference did not back off after overflow")
    raise AssertionError("Original C reference exhausted overflow retry budget")


def verify():
    queue = load_queue()
    manifest, configs, fingerprint = queue.prepare()
    cfg = next(item for item in configs if item["budget"] == "005" and "_G4_" in item["run_id"])
    if queue.DEVICE.type != "cuda":
        raise RuntimeError("This check must run on the actual CUDA training device")
    report = {"passed": False, "scope": "engineering_only_not_formal_results",
              "run_config": cfg, "fingerprint": fingerprint,
              "source_files": {"verify_training.py": sha256(Path(__file__))},
              "manifest_sha256": sha256(ROOT / "data/manifest.json"), "checks": {}}
    print("Loading one G4 5% model for real GPU engineering checks.", flush=True)
    start_time = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with queue.queue_lock(), queue.parent.prevent_system_sleep(), deny_private_data_reads() as reads:
        rt = queue.Runtime(cfg, eager=False)
        report["checks"]["initialization"] = initialization_check(rt)
        if not rt.u_rows or any(set(row) != PUBLIC_FIELDS for row in rt.u_rows):
            raise AssertionError("Actual U loader did not preserve the strict label-free schema")
        report["checks"]["u_loader_schema"] = {"rows": len(rt.u_rows), "fields": sorted(PUBLIC_FIELDS),
                                                       "hidden_labels_or_donor_metadata": False}
        initial = queue.capture_training_state(rt)
        print("Running actual BCE+USA+OT update at logical smoke step 200.", flush=True)
        record200 = queue.perform_step(rt, 200, ot_enabled=True)
        report["checks"]["ot_step200"] = check_ot_record(record200)
        snapshot200 = queue.capture_training_state(rt)
        if tensor_digest(initial["trainable_state"]) == tensor_digest(snapshot200["trainable_state"]):
            raise AssertionError("Actual optimization did not change trainable weights")
        if not snapshot200["optimizer"]["state"] or not snapshot200["scaler"]:
            raise AssertionError("Full smoke checkpoint lacks optimizer/scaler state")
        report["checks"]["checkpoint_guard"] = checkpoint_guard_check(queue, snapshot200)
        SMOKE.mkdir(parents=True, exist_ok=True)
        checkpoint_path = SMOKE / "complete_after_logical_step200.pt"
        torch.save({"scope": report["scope"], "successful_smoke_steps": 1, "logical_step": 200,
                    "fingerprint": fingerprint, "training_state": snapshot200}, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if loaded["fingerprint"] != fingerprint:
            raise AssertionError("Saved engineering checkpoint fingerprint differs")
        report["checks"]["checkpoint_serialization"] = compare_trees(snapshot200, loaded["training_state"], atol=0)
        print("Running step 201, restoring the full step-200 checkpoint, and replaying step 201.", flush=True)
        record201 = queue.perform_step(rt, 201, ot_enabled=True)
        state201 = queue.capture_training_state(rt)
        report["checks"]["ot_step201"] = check_ot_record(record201)
        queue.restore_training_state(rt, copy.deepcopy(loaded["training_state"]))
        replay201 = queue.perform_step(rt, 201, ot_enabled=True)
        replay_state = queue.capture_training_state(rt)
        report["checks"]["restore_replay_state"] = compare_trees(state201, replay_state, atol=2e-6)
        report["checks"]["restore_replay_losses"] = compare_trees(
            {key: record201[key] for key in ("loss", "itm_bce", "pairusa", "ot", "mu", "lambda")},
            {key: replay201[key] for key in ("loss", "itm_bce", "pairusa", "ot", "mu", "lambda")}, atol=2e-6)
        for possible in (("l_image_ids", "selected_l_image_ids"), ("u_pair_ids", "selected_u_pair_ids")):
            first, second = find_value(record201, possible), find_value(replay201, possible)
            if first is None or second is None or first != second:
                raise AssertionError(f"Resume sampling evidence missing/different for {possible}")
        strong1 = find_value(record201, ("strong_view_digest", "u_strong_digest"))
        strong2 = find_value(replay201, ("strong_view_digest", "u_strong_digest"))
        if strong1 is None or strong1 != strong2:
            raise AssertionError("Resume strong-view evidence missing/different")
        report["checks"]["restore_replay_samples_and_views"] = {"l_ids_equal": True,
                "u_pair_ids_equal": True, "strong_views_equal": True, "strong_digest": strong1}
        print("Comparing OT-disabled update against the original C forward/loss/update path.", flush=True)
        queue.restore_training_state(rt, copy.deepcopy(snapshot200))
        no_ot = queue.perform_step(rt, 201, ot_enabled=False)
        no_ot_state = queue.capture_training_state(rt)
        queue.restore_training_state(rt, copy.deepcopy(snapshot200))
        reference = original_c_step(queue, rt, cfg, 201)
        reference_state = queue.capture_training_state(rt)
        report["checks"]["original_c_update_equivalence"] = compare_trees(no_ot_state, reference_state, atol=2e-6)
        report["checks"]["original_c_loss_equivalence"] = compare_trees(
            {key: no_ot[key] for key in ("loss", "itm_bce", "pairusa", "lambda")},
            {key: reference[key] for key in ("loss", "itm_bce", "pairusa", "lambda")}, atol=2e-6)
        no_ot_l = find_value(no_ot, ("l_image_ids", "selected_l_image_ids"))
        if no_ot_l != reference["l_image_ids"]:
            raise AssertionError("L selection changed relative to original C")
        write_json(SMOKE / "step_records.json", {"step200": record200, "step201": record201,
                   "replay201": replay201, "ot_disabled201": no_ot, "original_c201": reference})
        print("Releasing G4, loading G3, and verifying its real BCE+OT update with 34 keys.", flush=True)
        del rt, initial, snapshot200, loaded, state201, replay_state, no_ot_state, reference_state
        gc.collect()
        torch.cuda.empty_cache()
        g3_cfg = next(item for item in configs if item["budget"] == "005" and "_G3_" in item["run_id"])
        rt = queue.Runtime(g3_cfg, eager=False)
        report["g3_run_config"] = g3_cfg
        report["checks"]["g3_initialization"] = initialization_check(rt, variant="G3")
        g3_before = tensor_digest(rt.model.trainable_state())
        g3_record = queue.perform_step(rt, 200, ot_enabled=True)
        report["checks"]["g3_ot_step200"] = check_ot_record(g3_record)
        if g3_record["lambda"] != 0 or g3_record["pairusa"] != 0 or g3_record["mu"] != 0.1:
            raise AssertionError("G3 has unexpected auxiliary loss coefficients")
        if len(rt.model.trainable_state()) != 34 or tensor_digest(rt.model.trainable_state()) == g3_before:
            raise AssertionError("G3 failed to update exactly its 34-key model")
        write_json(SMOKE / "g3_step200.json", g3_record)
        report["checks"]["g3_no_usa_real_update"] = {"updated": True, "trainable_keys": 34,
                                                         "pairusa": 0.0, "ot_mu": 0.1}
        g3_start = queue.capture_training_state(rt)
        print("Comparing G3 OT-disabled pure BCE update against the original BCE path.", flush=True)
        g3_no_ot = queue.perform_step(rt, 201, ot_enabled=False)
        g3_no_ot_state = queue.capture_training_state(rt)
        queue.restore_training_state(rt, copy.deepcopy(g3_start))
        g3_reference = original_c_step(queue, rt, g3_cfg, 201)
        g3_reference_state = queue.capture_training_state(rt)
        report["checks"]["g3_original_bce_update_equivalence"] = compare_trees(g3_no_ot_state, g3_reference_state, atol=2e-6)
        report["checks"]["g3_original_bce_loss_equivalence"] = compare_trees(
            {key: g3_no_ot[key] for key in ("loss", "itm_bce", "pairusa", "lambda")},
            {key: g3_reference[key] for key in ("loss", "itm_bce", "pairusa", "lambda")}, atol=2e-6)
        if find_value(g3_no_ot, ("l_image_ids", "selected_l_image_ids")) != g3_reference["l_image_ids"]:
            raise AssertionError("G3 L selection differs from the original BCE path")
        write_json(SMOKE / "g3_bce_equivalence.json", {"ot_disabled": g3_no_ot, "original_bce": g3_reference})
        report["checks"]["private_data_reads"] = {"denied_during_runtime_and_steps": True,
                                                       "attempted_reads": len(reads["attempts"])}
    report["smoke_checkpoint_sha256"] = sha256(checkpoint_path)
    report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["elapsed_seconds"] = time.monotonic() - start_time
    report["device"] = torch.cuda.get_device_name(queue.DEVICE)
    report["passed"] = True
    return report


def main():
    result_path = ROOT / "audit/training_verification.json"
    try:
        result = verify()
    except Exception as error:
        failure = {"passed": False, "scope": "engineering_only_not_formal_results",
                   "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                   "source_files": {"verify_training.py": sha256(Path(__file__))}}
        write_json(result_path, failure)
        raise
    write_json(result_path, result)
    print(json.dumps({"passed": result["passed"], "audit": str(result_path),
                      "elapsed_seconds": result["elapsed_seconds"],
                      "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
