# -*- coding: utf-8 -*-
"""Snapshot immutable originals, then register the repaired solver inputs.

No GPU model is created and no formal output is edited. --snapshot only
copies originals; --finalize freezes inputs before GPU verification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import recover_queue as recovery

ROOT, CODE = recovery.ROOT, recovery.CODE
FAILURE_DIR = ROOT / "audit/ot_failure_20260930"
FIRST_FAILURE = ROOT / "outputs" / recovery.ACTIVE / "recovery_backups/pre_exact_retry_20260930_043959/failure.json"
SECOND_FAILURE = ROOT / "outputs" / recovery.ACTIVE / "failure.json"
FIRST_SHA = "f2cb4831e7c3f5cc44964308fdf4554f347c668bf1e49c443f5b36acc82f9c32"
SECOND_SHA = "8360cae7031b910f1328f3510f82dc930870fba2ef06baccb5c0ca08d78ee5d6"


def snapshot():
    if recovery.SNAPSHOT.exists():
        registered = recovery.read_json(recovery.SNAPSHOT)
        for relative, expected in registered["backups"].items():
            if recovery.sha256(ROOT / relative) != expected:
                raise RuntimeError(f"Existing source backup differs: {relative}")
        recovery.verify_completed(registered["legacy_fingerprint"])
        return registered
    queue = recovery.load_queue()
    with queue.queue_lock():
        pipeline_path = ROOT / "outputs/pipeline_state.json"
        pipeline = recovery.read_json(pipeline_path)
        legacy_fp = pipeline["fingerprint"]
        if (pipeline["state"] != "failed" or pipeline["active_run"] != recovery.ACTIVE
                or legacy_fp["code/ot_loss.py"] != recovery.LEGACY_SOLVER_SHA):
            raise RuntimeError("Expected the exact failed original pipeline")
        completed = recovery.verify_completed(legacy_fp)
        active = ROOT / "outputs" / recovery.ACTIVE
        if recovery.sha256(active / "last.pt") != recovery.ORIGINAL_LAST_SHA or recovery.sha256(active / "best.pt") != recovery.ORIGINAL_BEST_SHA:
            raise RuntimeError("Active durable checkpoints changed")
        checkpoint = torch.load(active / "last.pt", map_location="cpu", weights_only=False)
        queue.validate_resume_checkpoint(checkpoint, checkpoint["provenance"], checkpoint["trainable_state"])
        if checkpoint["step"] != 100 or checkpoint["provenance"]["source"] != legacy_fp:
            raise RuntimeError("Incorrect active checkpoint step/source")
        if recovery.sha256(FIRST_FAILURE) != FIRST_SHA or recovery.sha256(SECOND_FAILURE) != SECOND_SHA:
            raise RuntimeError("Pinned first/second failure evidence changed")
        first, second = recovery.read_json(FIRST_FAILURE), recovery.read_json(SECOND_FAILURE)
        for key in ("attempted_step", "error", "provenance", "ot_diagnostics"):
            if first[key] != second[key]:
                raise RuntimeError(f"Two exact-source failures disagree: {key}")
        if second["attempted_step"] != 140:
            raise RuntimeError("Unexpected numerical failure step")
        backups = {}

        def save(source, relative, expected=None):
            target = recovery.BACKUP / relative
            digest = recovery.checked_copy(source, target, expected)
            backups[target.relative_to(ROOT).as_posix()] = digest

        for relative, expected in legacy_fp.items():
            source = ROOT / relative
            if recovery.sha256(source) != expected:
                if relative == "code/ot_loss.py":
                    source = FAILURE_DIR / "legacy_ot_loss.py"
                else:
                    raise RuntimeError(f"Unexplained changed legacy source: {relative}")
            save(source, "legacy_source/" + relative, expected)
        for run_id, files in recovery.PINNED_COMPLETED.items():
            for name, expected in files.items():
                save(ROOT / "outputs" / run_id / name, f"completed/{run_id}/{name}", expected)
        for path in sorted(active.iterdir()):
            if path.is_file():
                save(path, "active/" + path.name)
        save(pipeline_path, "pipeline_state.json")
        save(FIRST_FAILURE, "failures/first_failure.json", FIRST_SHA)
        save(SECOND_FAILURE, "failures/second_failure.json", SECOND_SHA)
        for name in ("launcher_retry_exact_20260930_043959_stdout.log", "launcher_retry_exact_20260930_043959_stderr.log"):
            save(ROOT / "outputs" / name, "failures/" + name)
        for name in ("legacy_ot_numerical_checks.json", "legacy_training_verification.json", "legacy_test_ot_loss.py"):
            save(FAILURE_DIR / name, "legacy_checks/" + name)
        registered = {"version": 1, "scope": "immutable_pre_numerical_repair_snapshot",
            "legacy_fingerprint": legacy_fp, "completed_artifacts": recovery.PINNED_COMPLETED,
            "completed_checks": completed, "run_ids": pipeline["run_ids"],
            "pipeline_sha256": recovery.sha256(pipeline_path), "backups": backups,
            "active": {"run_id": recovery.ACTIVE, "durable_step": 100,
                       "last_sha256": recovery.ORIGINAL_LAST_SHA, "best_sha256": recovery.ORIGINAL_BEST_SHA,
                       "original_provenance": checkpoint["provenance"]},
            "repeated_failure": {"attempted_step": 140, "first_sha256": FIRST_SHA, "second_sha256": SECOND_SHA,
                                 "identical_full_ot_diagnostics": True, "identical_provenance": True,
                                 "row_residual": second["ot_diagnostics"]["row_residual"],
                                 "col_residual": second["ot_diagnostics"]["col_residual"]}}
        recovery.immutable_json(recovery.SNAPSHOT, registered)
        return registered


def finalize():
    registered = snapshot()
    queue = recovery.load_queue()
    _, configs, base_fp = queue.prepare()
    changed = {key for key in set(base_fp) | set(registered["legacy_fingerprint"])
               if base_fp.get(key) != registered["legacy_fingerprint"].get(key)}
    if changed != {"code/ot_loss.py"}:
        raise RuntimeError(f"Unexpected change beyond the equivalent solver repair: {sorted(changed)}")
    numerical_path = ROOT / "audit/ot_numerical_checks.json"
    numerical = recovery.read_json(numerical_path)
    if not numerical.get("passed") or numerical.get("source_sha256", {}).get("ot_loss.py") != base_fp["code/ot_loss.py"]:
        raise RuntimeError("Repaired solver numerical tests are not bound/passed")
    reproduction_path = FAILURE_DIR / "reproduction.json"
    reproduction = recovery.read_json(reproduction_path)
    if (not reproduction["expected_failure_reproduced"] or not reproduction["source_checkpoint_unchanged"]
            or reproduction["source_last_sha256"] != recovery.ORIGINAL_LAST_SHA):
        raise RuntimeError("Real failure reproduction is not tied to the immutable last100")
    if any(row["scalar_differences"]["loss"] != 0 for row in reproduction["replayed_steps"]):
        raise RuntimeError("Old failure reconstruction did not exactly replay registered losses")
    artifacts = [numerical_path, FAILURE_DIR / "failure_inputs_step140.pt", reproduction_path,
                 FAILURE_DIR / "solver_comparison_cpu.json", FAILURE_DIR / "legacy_ot_loss.py"]
    entry = {"version": 1, "scope": "equivalent_balanced_entropic_OT_numerical_solver_recovery",
        "snapshot_sha256": recovery.sha256(recovery.SNAPSHOT),
        "legacy_fingerprint": registered["legacy_fingerprint"], "new_base_fingerprint": base_fp,
        "recovery_code_sha256": {f"code/{name}": recovery.sha256(CODE / name) for name in recovery.RECOVERY_CODE},
        "evidence_sha256": {p.relative_to(ROOT).as_posix(): recovery.sha256(p) for p in artifacts},
        "preserved_completed_runs": list(recovery.COMPLETED_ORDER),
        "active_run": recovery.ACTIVE, "active_original_checkpoint_sha256": recovery.ORIGINAL_LAST_SHA,
        "resume_durable_step": 100, "failed_attempted_step": 140,
        "remaining_runs": [c["run_id"] for c in configs if c["run_id"] not in recovery.COMPLETED_ORDER],
        "unchanged_scientific_definition": {"epsilon": 0.1, "max_iterations": 100,
            "both_marginal_tolerance": 1e-5, "total_successful_steps": 1600, "l_pairs_per_step": 32,
            "u_pairs_per_step": 32, "same_data_caption_sampling_augmentation_LR": True},
        "solver_transition": {"old_sha256": recovery.LEGACY_SOLVER_SHA,
            "new_sha256": base_fp["code/ot_loss.py"], "unchanged_standard_first_iterations": 50,
            "acceleration_counts_inside_total_100": True,
            "completed_max_iterations": max(value["max_ot_iterations"] for value in registered["completed_checks"].values())},
        "historical_results_rewritten": False, "checkpoint_migration": "provenance_wrapper_only_exact_nonprovenance_state",
        "gpu_verification_required_after_input_freeze": True}
    recovery.immutable_json(recovery.MIGRATION, entry)
    return entry


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--snapshot", action="store_true")
    action.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    value = finalize() if args.finalize else snapshot()
    print(json.dumps({"prepared": str(recovery.MIGRATION if args.finalize else recovery.SNAPSHOT),
                      "formal_outputs_modified": False, "gpu_training_started": False,
                      "snapshot_or_manifest_sha256": recovery.sha256(recovery.MIGRATION if args.finalize else recovery.SNAPSHOT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
