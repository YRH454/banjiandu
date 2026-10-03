"""One serial dispatcher; do not shadow Python's stdlib queue module."""
from __future__ import annotations
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from common import ROOT, read, atomic_json, now, exclusive, digest

STATE = ROOT / "outputs/pipeline_state.json"

def pid_ticks(pid):
    try:
        # Linux /proc comm may contain spaces: fields after the closing ')' start at #3.
        after = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return after[19]
    except (FileNotFoundError, ProcessLookupError):
        return None

def alive(pid, ticks):
    return bool(pid and ticks and pid_ticks(pid) == str(ticks))

def gpu_idle(uuid):
    got = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip().splitlines()
    if got != [uuid]:
        raise RuntimeError("GPU physical identity/count changed")
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).splitlines()
    if any(r.strip().isdigit() for r in rows):
        raise RuntimeError("GPU already has a compute worker; no implicit termination or duplicate launch")

def wait_child(command, mode, run_id, state, env):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{run_id}_{mode}_{stamp}.stdout.log"
    stderr_path = log_dir / f"{run_id}_{mode}_{stamp}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, close_fds=True)
        state.update(current_run=run_id, phase=mode, child_pid=p.pid, child_start_ticks=pid_ticks(p.pid),
                     child_command=command, stdout=stdout_path.relative_to(ROOT).as_posix(),
                     stderr=stderr_path.relative_to(ROOT).as_posix(), updated_utc=now())
        atomic_json(STATE, state)
        while p.poll() is None:
            state["updated_utc"] = now()
            atomic_json(STATE, state)
            time.sleep(10)
        code = p.wait()
    state.update(child_exit_code=code, child_pid=None, child_start_ticks=None, updated_utc=now())
    atomic_json(STATE, state)
    if code != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"{run_id} {mode} exited {code}; preserve full checkpoint; {tail}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    args = parser.parse_args()
    queue = read(ROOT / "configs/queue.json")
    if queue["protocol_sha256"] != digest(ROOT / "configs/protocol.json") or len(queue["queue"]) != 40 or len(set(queue["queue"])) != 40:
        raise RuntimeError("Queue registration changed")
    env = dict(os.environ)
    uuid = queue["server"]["gpu_uuid"]
    env.update(CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUTF8="1",
               TOKENIZERS_PARALLELISM="false", USE_TF="0", USE_FLAX="0", OMP_NUM_THREADS="4")
    old = read(STATE) if STATE.exists() else {}
    if alive(old.get("child_pid"), old.get("child_start_ticks")):
        raise RuntimeError("Registered child is still alive; do not launch a second dispatcher/worker")
    state = {"state": "running", "pid": os.getpid(), "pid_start_ticks": pid_ticks(os.getpid()),
             "assigned_count": 40, "completed": [], "current_run": None, "phase": "registration",
             "gpu_uuid": uuid, "created_utc": now(), "updated_utc": now(), "queue_protocol_sha256": queue["protocol_sha256"]}
    atomic_json(STATE, state)
    try:
        from train import verify_result
        for run_id in queue["queue"]:
            if (ROOT / "outputs" / run_id / "result.json").exists():
                verify_result(run_id)
                state["completed"].append(run_id)
                atomic_json(STATE, state)
                continue
            gpu_idle(uuid)
            command = [sys.executable, "-X", "utf8", "-u", "code/train.py", "--run-id", run_id, "--gpu-uuid", uuid]
            wait_child(command+["--check"], "check", run_id, state, env)
            wait_child(command+["--gate"], "gate", run_id, state, env)
            gpu_idle(uuid)
            wait_child(command+["--run"], "run", run_id, state, env)
            verify_result(run_id)
            state["completed"].append(run_id)
            state["updated_utc"] = now()
            atomic_json(STATE, state)
        summaries = {rid: verify_result(rid) for rid in queue["queue"]}
        atomic_json(ROOT / "reports/summary.json", {"configurations": 40, "results": summaries,
            "single_seed": 20260825, "no_Test": True, "equal_compute_claim": False, "completed_utc": now()})
        state.update(state="completed", current_run=None, phase="complete", completed_utc=now(), updated_utc=now())
        atomic_json(STATE, state)
    except BaseException:
        state.update(state="failed", error=traceback.format_exc(), updated_utc=now())
        atomic_json(STATE, state)
        raise

if __name__ == "__main__":
    with exclusive(ROOT / "locks/dispatcher.lock"):
        main()
