"""Read-only CPU inspection of the actual first formal worker and full state."""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import argparse
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import torch
from common import ROOT, read, digest, code_hashes, atomic_json, now
from train import check
from dispatcher import pid_ticks, alive

parser = argparse.ArgumentParser()
parser.add_argument("--require-validation", action="store_true")
args = parser.parse_args()
rid = "apple_001_softmatch_s20260825"
cfg, _, prov, _ = check(rid)
pipeline = read(ROOT / "outputs/pipeline_state.json")
status = read(ROOT / "outputs" / rid / "status.json")
assert pipeline["state"] == "running" and pipeline["phase"] == "run" and pipeline["current_run"] == rid
assert alive(pipeline["pid"], pipeline["pid_start_ticks"])
assert alive(pipeline["child_pid"], pipeline["child_start_ticks"]) and status["pid"] == pipeline["child_pid"]
cmd = Path(f"/proc/{status['pid']}/cmdline").read_bytes().decode().replace("\0", " ")
assert "code/train.py" in cmd and rid in cmd and "--run" in cmd
assert status["step"] > 50 and status["target_steps"] == 2200
assert read(ROOT / "audit/cpu_preflight.json")["source"] == code_hashes()
out = ROOT / "outputs" / rid
last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
assert last["provenance"] == prov and last["step"] >= 50
assert last["format"] == "pair_ssl_full_v1" and last["algorithm_checkpoint_version"] == "soft_sim_full_v1"
assert len(last["trainable_state"]) == len(last["ema"]) == len(last["optimizer"]["state"]) == 34
assert set(last["rng"]) == {"python", "numpy", "torch", "cuda"} and len(last["rng"]["cuda"]) == 1
assert last["scaler"]["scale"] > 0 and last["algorithm_state"]["statistics"]["p_model"] is not None
assert all({"step", "exp_avg", "exp_avg_sq"} <= set(v) for v in last["optimizer"]["state"].values())
gates = {}
for gate_rid in (rid, "apple_001_simmatch_s20260825"):
    passed = read(ROOT / "audit/gpu_gates" / gate_rid / "passed.json")
    assert passed["passed"] and passed["provenance"] == check(gate_rid)[2]
    gates[gate_rid] = {"passed": True, "max_parameter_replay_difference": passed["max_parameter_replay_difference"],
                       "formal_steps_added": passed["formal_successful_steps_added"],
                       "cuda_peak_allocated_bytes": passed["cuda_peak_allocated_bytes"]}
result = {"verified": True, "formal_started": True, "run_id": rid, "status_at_observation": status,
    "pipeline_at_observation": pipeline, "full_last_step": last["step"], "full_last_sha256_at_observation": digest(out / "last.pt"),
    "full_student_ema_optimizer_scaler_rng_algorithm_state": True,
    "observed_progress_beyond_last50": status["step"] > 50, "gates": gates,
    "gpu_observation": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], text=True).strip(),
    "cuda_compute_observation": subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip(),
    "process_command": cmd, "single_gpu_serial_queue": True, "assigned_configurations": 40,
    "source": code_hashes(), "validation100_verified": False, "Test_evaluated": False,
    "monitor_automation_id": "linux", "monitor_hours": 1, "created_utc": now()}
if args.require_validation:
    metrics = read(out / "validation_0100.json")
    predictions = read(out / "predictions_validation_0100.json")
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert metrics["step"] == 100 and metrics["validation_anchors"] == 400 and metrics["evaluation_model"] == "ema"
    assert predictions["split"] == "validation" and len(predictions["labels"]) == len(predictions["probabilities"]) == 800
    assert len(predictions["image_ids"]) == 400 and last["step"] >= 100
    assert best["provenance"] == prov and best["algorithm_checkpoint_version"] == "soft_sim_full_v1"
    assert all(k in best for k in ("optimizer", "scaler", "rng", "ema", "algorithm_state"))
    result.update(validation100_verified=True, validation100_metrics=metrics,
                  full_best_step_at_observation=best["step"], full_best_sha256_at_observation=digest(out / "best.pt"))
atomic_json(ROOT / "audit/deployment_verified.json", result)
print({"verified": True, "run": rid, "status_step": status["step"], "last_step": last["step"],
       "validation100": result["validation100_verified"], "gpu": result["gpu_observation"]})
