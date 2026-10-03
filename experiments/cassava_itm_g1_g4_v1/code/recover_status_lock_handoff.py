"""Recover original 5% S4, then hand off at its completed-result boundary.

All registered training sources/configs and checkpoint provenance remain intact.
The only runtime adapter retries identical Windows output-file replacements
for at most five seconds; it never changes tensors, samples, or loss code.
There is no concurrent high-frequency status reader in this recovery path.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch

import cassava_fast_queue as fast

q = fast.q
ROOT = fast.ROOT
OUT = ROOT / "outputs"
CURRENT = OUT / q.run_id("S4", "005")
NEXT = OUT / q.run_id("S1", "020")
RECEIPT = ROOT / "audit/acceleration_handoff.json"
REPLACE = os.replace
RETRIES = []


def output_replace(src, dst):
    """Retry an identical rename only, not a training update or checkpoint save."""
    target = Path(dst).resolve()
    if OUT.resolve() not in target.parents:
        return REPLACE(src, dst)
    deadline = time.monotonic() + 5.0
    count = 0
    while True:
        try:
            result = REPLACE(src, dst)
            if count:
                event = {"path": str(target), "retries": count, "utc": q.base.now()}
                RETRIES.append(event)
                print(json.dumps({"output_replace_recovered": event}), flush=True)
            return result
        except PermissionError as exc:
            if os.name != "nt" or exc.winerror not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            count += 1
            time.sleep(min(0.02 * count, 0.2))


def assert_no_workers():
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.pid == os.getpid() or "python" not in (proc.info["name"] or "").lower():
            continue
        try:
            command = " ".join(proc.info["cmdline"] or []).lower()
            if Path(proc.cwd()).resolve() == ROOT.resolve() and any(
                    marker in command for marker in ("cassava_queue.py", "cassava_fast_queue.py",
                                                      "handoff_after_5pct.py", Path(__file__).name.lower())):
                raise RuntimeError(f"Another cassava launcher/waiter is alive: {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def reject_old_next_updates():
    if (NEXT / "result.json").exists():
        raise RuntimeError("Old 20% S1 already completed; cannot use this boundary recovery")
    if (NEXT / "status.json").exists() and int(q.read(NEXT / "status.json").get("step") or 0) > 0:
        raise RuntimeError("Old 20% S1 already has successful steps; leave it intact")
    if (NEXT / "train.jsonl").exists() and (NEXT / "train.jsonl").stat().st_size:
        raise RuntimeError("Old 20% S1 has training records; boundary recovery is not authorized")


def completed_snapshot(fp):
    snapshot = {}
    for stage in ("S1", "S2", "S3"):
        directory = OUT / q.run_id(stage, "005")
        result = q.read(directory / "result.json")
        status = q.read(directory / "status.json")
        best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        source = result["provenance"].get("source", result["provenance"].get("base", {}).get("source"))
        if (result["state"] != "completed_validation" or result["run_id"] != directory.name
                or status["state"] != "completed_validation" or status["step"] != 1600
                or source != fp or best["provenance"] != result["provenance"]
                or best["step"] != result["best_step"] or best["validation"] != result["best_validation"]):
            raise RuntimeError(f"Completed history is not consistent: {directory.name}")
        for name in ("result.json", "best.pt", "last.pt", "status.json"):
            snapshot[str((directory / name).relative_to(ROOT))] = q.digest(directory / name)
    return snapshot


def preflight():
    assert_no_workers()
    reject_old_next_updates()
    if RECEIPT.exists() or fast.FAST_PIPELINE.exists():
        raise RuntimeError("Handoff already exists; recover the registered fast queue instead")
    manifest, fp = q.check_inputs()
    state = q.read(OUT / "pipeline_state.json")
    if state["source_fingerprint"] != fp or state["run_ids"] != [q.run_id(s, b) for s, b in q.queue_order()]:
        raise RuntimeError("Original pipeline identity differs")
    original_smoke = q.read(ROOT / "audit/real_gpu_smoke_long_caption_v2.json")
    if not original_smoke["passed"] or original_smoke["source_fingerprint"] != fp:
        raise RuntimeError("Original registered GPU verification differs")
    profile = q.read(fast.PROFILE_PATH)
    fast.verify_profile(profile)
    fast_fp = fast.fingerprint(fp)
    smoke = q.read(ROOT / "audit/acceleration_smoke.json")
    if (not smoke["passed"] or smoke["source_fingerprint"] != fast_fp
            or smoke["physical_microbatch"] != 16):
        raise RuntimeError("Registered fast verification differs")
    cfg = q.ot_config(manifest, fp, "005", "S4")
    if q.read(ROOT / "configs" / f"{cfg['run_id']}.json") != cfg:
        raise RuntimeError("Recovery config is not exactly the original S4 config")
    checkpoint = torch.load(CURRENT / "last.pt", map_location="cpu", weights_only=False)
    parent = torch.load(Path(cfg["warmstart"]["parent_best_path"]), map_location="cpu", weights_only=False)
    expected_keys = set(parent["trainable_state"]) | q.EXTRA
    q.ot.validate_resume_checkpoint(checkpoint, {"source": fp, "run": cfg, "test_evaluation": False}, expected_keys)
    if len(expected_keys) != 39:
        raise RuntimeError("S4 key layout differs")
    proof = {"passed": True, "utc": q.base.now(), "durable_step": checkpoint["step"],
             "last_sha256": q.digest(CURRENT / "last.pt"), "source_fingerprint": fp,
             "fast_source_fingerprint": fast_fp, "completed_history": completed_snapshot(fp),
             "runtime_adapter_sha256": q.digest(Path(__file__)), "checkpoint_migration": False,
             "training_source_migration": False, "runtime_only_change": "bounded_identical_output_replace_retry",
             "replace_retry_limit_seconds": 5, "no_concurrent_status_waiter": True}
    return manifest, fp, state, proof


def run(manifest, fp, state, proof):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = ROOT / "audit/recovery_backups_status_lock" / stamp
    with q.base.process_lock(), q.base.prevent_system_sleep():
        assert_no_workers()
        reject_old_next_updates()
        if q.digest(CURRENT / "last.pt") != proof["last_sha256"]:
            raise RuntimeError("Durable checkpoint changed after preflight")
        backup.mkdir(parents=True, exist_ok=False)
        for path in (CURRENT / "last.pt", CURRENT / "best.pt", CURRENT / "status.json", CURRENT / "failure.json",
                     CURRENT / "train.jsonl", CURRENT / "validation.jsonl", OUT / "pipeline_state.json",
                     ROOT / "code/handoff_after_5pct.py"):
            if path.exists():
                target = backup / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        q.base.atomic_json(ROOT / "audit/status_lock_recovery_20261001.json", {**proof, "backup": str(backup)})
        os.replace = output_replace
        state = {key: value for key, value in state.items() if key != "error"}
        state.update(state="running", active_run=CURRENT.name, updated_utc=q.base.now(),
                     runtime_recovery="audit/status_lock_recovery_20261001.json", pid=os.getpid())
        q.base.atomic_json(OUT / "pipeline_state.json", state)
        torch.manual_seed(q.SEED)
        torch.cuda.manual_seed_all(q.SEED)
        np.random.seed(q.SEED % (2**32))
        random.seed(q.SEED)
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(8)
        try:
            tokenizer = q.get_tokenizer(q.base.model_config()["tokenizer_path"])
            result = q.run_stage(manifest, fp, tokenizer, "S4", "005")
            if result["final_step"] != 1600 or result["state"] != "completed_validation":
                raise RuntimeError("Original S4 did not produce a completed result")
            for rel, digest in proof["completed_history"].items():
                if q.digest(ROOT / rel) != digest:
                    raise RuntimeError(f"Completed history changed: {rel}")
            reject_old_next_updates()
            completed = fast.check_completed_five_percent(fp)
            q.base.atomic_json(ROOT / "audit/original_pipeline_before_fast_handoff.json", state)
            receipt = {"state": "handed_off_after_5pct", "old_pid": os.getpid(), "utc": q.base.now(),
                       "completed_5pct": completed, "original_next_status_preserved": None,
                       "fast_source_fingerprint": proof["fast_source_fingerprint"], "old_source_fingerprint": fp,
                       "handoff_code_sha256": q.digest(Path(__file__)), "recovered_durable_step": proof["durable_step"],
                       "runtime_output_retry_events": RETRIES, "recovery_audit": "audit/status_lock_recovery_20261001.json"}
            q.base.atomic_json(RECEIPT, receipt)
            state.update(state="handed_off_after_5pct", active_run=None, updated_utc=q.base.now(),
                         handoff_receipt=str(RECEIPT))
            q.base.atomic_json(OUT / "pipeline_state.json", state)
            print(json.dumps({"boundary_recovery_complete": result["run_id"], "final_step": 1600}), flush=True)
        except BaseException as exc:
            state.update(state="failed", error=repr(exc), updated_utc=q.base.now())
            q.base.atomic_json(OUT / "pipeline_state.json", state)
            q.base.atomic_json(ROOT / "audit/status_lock_recovery_failure_20261001.json",
                              {"error": repr(exc), "traceback": traceback.format_exc(), "utc": q.base.now()})
            raise
        finally:
            os.replace = REPLACE
    # The original-stage GPU lock is released before a distinct fast process starts.
    stdout, stderr = (OUT / f"launcher_fast_recovery_{stamp}.{suffix}.log" for suffix in ("stdout", "stderr"))
    with stdout.open("wb") as out, stderr.open("wb") as err:
        child = subprocess.Popen([sys.executable, "-X", "utf8", "-u", "code/cassava_fast_queue.py", "--run"],
                                 cwd=ROOT, stdout=out, stderr=err,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print(json.dumps({"fast_queue_pid": child.pid, "stdout": str(stdout), "stderr": str(stderr)}), flush=True)
    time.sleep(5)
    if child.poll() is not None:
        raise RuntimeError(f"Fast queue exited early: {child.returncode}; inspect {stderr}")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    manifest, fp, state, proof = preflight()
    if args.check:
        print(json.dumps({"recovery_checks_passed": True, "durable_step": proof["durable_step"],
                          "registered_sources_unchanged": True, "checkpoint_migration": False}), flush=True)
    else:
        run(manifest, fp, state, proof)


if __name__ == "__main__":
    main()
