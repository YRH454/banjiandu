# -*- coding: utf-8 -*-
"""Real GPU verification of the failed budget's equivalent numerical repair.

Only audit/recovery_integration/ and recovery_verification_20260930.json are
written. The active formal last100 and all four historical results are immutable.
"""
from __future__ import annotations

import copy
import json
import time
import traceback
import zipfile
from pathlib import Path

import torch

import recover_queue as recovery
import verify_training as checks

ROOT = recovery.ROOT
SMOKE = ROOT / "audit/recovery_integration"


def validate_record(row):
    diagnostics = checks.check_ot_record(row)
    if row["mu"] <= 0 or row["lambda"] != 0 or row["pairusa"] != 0:
        raise AssertionError("Failed branch must execute G3's BCE+OT without USA")
    return diagnostics


def verify():
    queue = recovery.load_queue()
    with queue.queue_lock(), queue.parent.prevent_system_sleep():
        snapshot, migration, configs, fingerprint, preserved = recovery.verified_inputs(queue)
        cfg = next(item for item in configs if item["run_id"] == recovery.ACTIVE)
        source_path = ROOT / "outputs" / recovery.ACTIVE / "last.pt"
        if recovery.sha256(source_path) != recovery.ORIGINAL_LAST_SHA:
            raise RuntimeError("GPU recovery verification requires untouched original last100")
        pipeline_path = ROOT / "outputs/pipeline_state.json"
        pipeline_before = recovery.sha256(pipeline_path)
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        queue.validate_resume_checkpoint(source, source["provenance"], source["trainable_state"])
        if source["step"] != 100 or source["provenance"]["source"] != snapshot["legacy_fingerprint"]:
            raise RuntimeError("Wrong source checkpoint for recovery verification")
        report = {"passed": False, "scope": "engineering_replay_only_not_formal_results",
                  "fingerprint": fingerprint, "source_last_sha256": recovery.ORIGINAL_LAST_SHA,
                  "source_step": 100, "completed_runs_preserved": preserved, "checks": {}}
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        fixture_path = ROOT / "audit/ot_failure_20260930/failure_inputs_step140.pt"
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            q, diagnostics = queue.soft_ot_targets(fixture["anchors"].to(queue.DEVICE),
                fixture["labels"].to(queue.DEVICE), fixture["queries"].to(queue.DEVICE), **fixture["solver_kwargs"])
        if (not diagnostics["converged"] or diagnostics["iterations"] > 100
                or max(diagnostics["row_residual"], diagnostics["col_residual"]) > 1e-5
                or q.requires_grad or not bool(torch.isfinite(q).all())):
            raise AssertionError("Exact formerly failing matrix still fails the registered solver constraints")
        report["checks"]["exact_failure140_inputs_on_gpu"] = {
            "fixture_sha256": recovery.sha256(fixture_path), "diagnostics": diagnostics,
            "q": q.detach().cpu().tolist(), "same_epsilon_tolerance_max_iterations": True}
        del q
        print("Exact failed-step140 matrix now converges on GPU; replaying original last100 through150.", flush=True)
        SMOKE.mkdir(parents=True, exist_ok=True)
        formal_logs = {row["step"]: row for row in (json.loads(line) for line in
                       (ROOT / "outputs" / recovery.ACTIVE / "train.jsonl").read_text(encoding="utf-8").splitlines())}
        records = []
        seen_pairs, seen_images = set(), set()
        with checks.deny_private_data_reads() as reads:
            torch.manual_seed(queue.SEED)
            torch.cuda.manual_seed_all(queue.SEED)
            torch.backends.cudnn.benchmark = True
            runtime = queue.Runtime(cfg, eager=False)
            queue.restore_training_state(runtime, copy.deepcopy(source))
            if any(set(row) != checks.PUBLIC_FIELDS for row in runtime.u_rows):
                raise AssertionError("Runtime U data exposes unregistered fields")
            runtime.cache.prime_images(runtime.train_rows + runtime.val_rows, flips=True)
            runtime.cache.prime_text([text for row in runtime.train_rows + runtime.val_rows
                                      for text in (row["positive_text"], row["negative_text"])])
            for step in range(101, 151):
                row = queue.perform_step(runtime, step)
                validate_record(row)
                seen_pairs.update(row["u_pair_ids"])
                seen_images.update(row["u_image_ids"])
                if step in formal_logs:
                    row["legacy_scalar_differences"] = {key: abs(row[key] - formal_logs[step][key])
                                                        for key in ("loss", "itm_bce", "ot")}
                records.append(row)
                if step % 5 == 0:
                    print(json.dumps({"engineering_step": step, "ot_iterations": row["diagnostics"]["iterations"],
                                      "row_residual": row["diagnostics"]["row_residual"],
                                      "seconds": time.monotonic() - started}), flush=True)
            snapshot150 = queue.capture_training_state(runtime)
            engineering_checkpoint = {**snapshot150, "step": 150,
                "provenance": {"source": fingerprint, "run": cfg, "test_evaluation": False},
                "best_snapshot": copy.deepcopy(source["best_snapshot"]),
                "best_scores": source["best_scores"].clone(), "best_step": source["best_step"],
                "seen_u_pairs": sorted(seen_pairs), "seen_u_images": sorted(seen_images),
                "active_seconds": time.monotonic() - started,
                "engineering_scope": report["scope"], "new_successful_engineering_updates": 50,
                "original_formal_checkpoint_sha256": recovery.ORIGINAL_LAST_SHA}
            queue.validate_resume_checkpoint(engineering_checkpoint, engineering_checkpoint["provenance"], snapshot150["trainable_state"])
            engineering_path = SMOKE / "engineering_step150.pt"
            queue.atomic_torch(engineering_path, engineering_checkpoint)
            with zipfile.ZipFile(engineering_path) as archive:
                if archive.testzip() is not None:
                    raise AssertionError("Engineering checkpoint ZIP CRC failed")
            loaded150 = torch.load(engineering_path, map_location="cpu", weights_only=False)
            recovery.assert_equal_tree(engineering_checkpoint, loaded150)
            print("Verifying full 150->151 checkpoint restoration and identical sampling/views.", flush=True)
            first151 = queue.perform_step(runtime, 151)
            state151 = queue.capture_training_state(runtime)
            queue.restore_training_state(runtime, copy.deepcopy(loaded150))
            replay151 = queue.perform_step(runtime, 151)
            replay_state151 = queue.capture_training_state(runtime)
            validate_record(first151)
            validate_record(replay151)
            report["checks"]["step151_state_replay"] = checks.compare_trees(state151, replay_state151, atol=2e-6)
            report["checks"]["step151_losses_replay"] = checks.compare_trees(
                {key: first151[key] for key in ("itm_bce", "ot", "loss", "grad_norm", "mu")},
                {key: replay151[key] for key in ("itm_bce", "ot", "loss", "grad_norm", "mu")}, atol=2e-6)
            for key in ("l_image_ids", "u_pair_ids", "u_image_ids"):
                if first151[key] != replay151[key]:
                    raise AssertionError(f"Recovery replay sampling differs: {key}")
            if first151["diagnostics"]["strong_view_digest"] != replay151["diagnostics"]["strong_view_digest"]:
                raise AssertionError("Recovery replay strong views differ")
            report["checks"]["private_data_reads"] = {"blocked_during_training": True, "attempted_reads": len(reads["attempts"])}
            report["checks"]["replay_samples_and_views"] = {"exact": True,
                "strong_view_digest": first151["diagnostics"]["strong_view_digest"]}
        checks.write_json(SMOKE / "replay_records.json", {"scope": report["scope"], "steps101_to150": records,
                                                         "step151": first151, "replayed_step151": replay151})
        if recovery.sha256(source_path) != recovery.ORIGINAL_LAST_SHA or recovery.sha256(pipeline_path) != pipeline_before:
            raise AssertionError("Engineering verification modified a formal checkpoint/pipeline")
        recovery.verify_completed(snapshot["legacy_fingerprint"])
        if recovery.recovery_fingerprint(queue, migration) != fingerprint:
            raise AssertionError("Recovery code/inputs changed during verification")
        report["checks"]["progress_past_failure"] = {"executed_steps": [101, 150], "successful_updates": 50,
            "original_failure_step140_succeeded": True, "formal_files_unchanged": True,
            "step140_diagnostics": records[39]["diagnostics"], "last_engineering_step": 151}
        report["checks"]["checkpoint_serialization"] = {"full_state_exact": True,
            "sha256": recovery.sha256(engineering_path), "optimizer_and_scaler_restored": True}
        report["pipeline_before_sha256"] = pipeline_before
        report["replay_records_sha256"] = recovery.sha256(SMOKE / "replay_records.json")
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["passed"] = True
        return report


def main():
    try:
        report = verify()
    except Exception as error:
        report = {"passed": False, "scope": "engineering_replay_only_not_formal_results",
                  "error": str(error), "error_type": type(error).__name__, "traceback": traceback.format_exc()}
        checks.write_json(recovery.VERIFICATION, report)
        raise
    checks.write_json(recovery.VERIFICATION, report)
    print(json.dumps({"passed": True, "verification": str(recovery.VERIFICATION),
                      "elapsed_seconds": report["elapsed_seconds"],
                      "formal_recovery_started": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
