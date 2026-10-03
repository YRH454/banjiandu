"""Two method-specific serial lanes on one GPU, adopting the existing worker."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import atomic_json, digest, exclusive, now, read


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19], fields[0]
    except (FileNotFoundError, ProcessLookupError):
        return None, None


def alive(pid, ticks):
    actual, state = process_identity(pid)
    return actual == str(ticks) and state not in ("Z", "X", None)


def verify(run_id, uuid):
    registered = (ROOT / "audit/performance_runs" / (run_id + ".json")).exists()
    command = ([sys.executable, str(HERE / "performance_train.py"), "--mode", "verify", "--gpu-uuid", uuid]
               if registered else [sys.executable, "-X", "utf8", "code/train.py", "--verify-result"])
    result = subprocess.run(command + ["--run-id", run_id], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"Completed run {run_id} failed verification: {result.stderr[-3000:]}")


def launch(run_id, mode, uuid):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stdout = ROOT / "logs" / f"{run_id}_perf_{mode}_{stamp}_{os.getpid()}.stdout.log"
    stderr = ROOT / "logs" / f"{run_id}_perf_{mode}_{stamp}_{os.getpid()}.stderr.log"
    command = [sys.executable, "-X", "utf8", "-u", str(HERE / "performance_train.py"),
               "--run-id", run_id, "--gpu-uuid", uuid, "--mode", mode]
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUTF8="1",
               TOKENIZERS_PARALLELISM="false", USE_TF="0", USE_FLAX="0", OMP_NUM_THREADS="4")
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err, close_fds=True)
    return {"run_id": run_id, "phase": mode, "pid": child.pid,
            "start_ticks": process_identity(child.pid)[0], "process": child, "adopted": False,
            "stdout": str(stdout.relative_to(ROOT)), "stderr": str(stderr.relative_to(ROOT)),
            "command": command}


def serial_parent(old):
    parent, ticks = int(old["pid"]), str(old["pid_start_ticks"])
    if not alive(parent, ticks):
        raise RuntimeError("Original dispatcher changed; do not take over another process")
    command = Path(f"/proc/{parent}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "code/dispatcher.py --run" not in command:
        raise RuntimeError("Original PID is not the registered serial dispatcher")
    return parent, ticks


def main(args):
    queue = read(ROOT / "configs/queue.json")
    if queue["protocol_sha256"] != digest(ROOT / "configs/protocol.json"):
        raise RuntimeError("Scientific protocol changed")
    groups = {"softmatch": [], "simmatch": []}
    for run_id in queue["queue"]:
        cfg = read(ROOT / "configs/runs" / (run_id + ".json"))
        groups[cfg["method"]].append(run_id)
    if len(set(queue["queue"])) != 40 or any(len(rows) != 20 for rows in groups.values()):
        raise RuntimeError("Expected two disjoint 20-configuration methods")
    uuid = queue["server"]["gpu_uuid"]
    proof_path = HERE / "bench_v2/benchmark.json"
    proof = read(proof_path)
    if not proof["passed"] or not proof["cold_replay_gate"]["passed"] or not proof["original_processes_resumed"]:
        raise RuntimeError("Successful isolated benchmark and full replay are required")
    options = proof["selected"]
    policy = {"version": "gpu_parallel_perf_20261002_v1", "options": options,
              "benchmark_sha256": digest(proof_path),
              "worker_files": {name: digest(HERE / name) for name in ("performance_train.py", "perf_cache.py")}}
    policy_path = HERE / "accepted_policy.json"
    if policy_path.exists() and read(policy_path) != policy:
        raise RuntimeError("Existing accepted execution version differs")
    atomic_json(policy_path, policy)
    old = read(ROOT / "outputs/pipeline_state.json")
    parent, parent_ticks = serial_parent(old)
    if old["state"] != "running" or old["phase"] != "run" or not alive(old["child_pid"], old["child_start_ticks"]):
        raise RuntimeError("Only an active registered training worker can be adopted")
    active_method = read(ROOT / "configs/runs" / (old["current_run"] + ".json"))["method"]
    completed = []
    for run_id in queue["queue"]:
        if (ROOT / "outputs" / run_id / "result.json").exists():
            verify(run_id, uuid)
            completed.append(run_id)
    adopted = {"run_id": old["current_run"], "phase": "run", "pid": old["child_pid"],
               "start_ticks": old["child_start_ticks"], "adopted": True, "process": None,
               "stdout": old["stdout"], "stderr": old["stderr"], "command": old["child_command"]}
    state = {"state": "running", "pid": os.getpid(), "pid_start_ticks": process_identity(os.getpid())[0],
             "assigned_count": 40, "completed": completed, "gpu_uuid": uuid,
             "phase": "run", "created_utc": now(), "updated_utc": now(),
             "execution_mode": "two_algorithm_parallel", "maximum_gpu_workers": 2,
             "performance_execution_version": policy["version"], "method_queues": groups,
             "queue_protocol_sha256": queue["protocol_sha256"],
             "scientific_targets_unchanged": True, "legacy_current_worker_not_restarted": True}
    lanes = {method: None for method in groups}
    lanes[active_method] = adopted
    if args.dry_run:
        print(json.dumps({"passed": True, "method_counts": {m: len(r) for m, r in groups.items()},
                          "completed": completed, "adopted_run": old["current_run"],
                          "adopted_pid": old["child_pid"], "options": policy["options"],
                          "new_lane_next": {m: next((rid for rid in r if rid not in completed), None)
                                            for m, r in groups.items() if m != active_method}}))
        return
    os.kill(parent, signal.SIGSTOP)
    try:
        if not alive(old["child_pid"], old["child_start_ticks"]) or not alive(parent, parent_ticks):
            raise RuntimeError("Worker/dispatcher changed during takeover")
        atomic_json(HERE / "legacy_pipeline_state_before_parallel.json", old)
        atomic_json(HERE / "parallel_execution_authorization.json", {
            "created_utc": now(), "user_requested_two_algorithm_programs": True,
            "serial_gpu_lock_exception_explicit": True, "maximum_gpu_workers": 2,
            "scientific_protocol_not_edited": True, "old_parent": parent,
            "adopted_child": old["child_pid"], "completed_preserved": completed,
            "queues_grouped_by_method_not_array_position": True})
    except BaseException:
        if alive(parent, parent_ticks):
            os.kill(parent, signal.SIGCONT)
        raise
    os.kill(parent, signal.SIGTERM)
    if alive(parent, parent_ticks):
        os.kill(parent, signal.SIGCONT)
    until = time.monotonic() + 5
    while alive(parent, parent_ticks) and time.monotonic() < until:
        time.sleep(.1)
    if alive(parent, parent_ticks):
        raise RuntimeError("Original scheduler did not release control; no duplicate launch")
    with exclusive(ROOT / "locks/dispatcher.lock"):
        try:
            while len(completed) < 40:
                for method, rows in groups.items():
                    worker = lanes[method]
                    if worker is not None:
                        child = worker["process"]
                        running = alive(worker["pid"], worker["start_ticks"]) if worker["adopted"] else child.poll() is None
                        if running:
                            continue
                        if child is not None and child.returncode != 0:
                            error = (ROOT / worker["stderr"]).read_text(encoding="utf-8", errors="replace")[-5000:]
                            raise RuntimeError(f"{worker['run_id']} {worker['phase']} failed: {error}")
                        if worker["phase"] == "check":
                            lanes[method] = launch(worker["run_id"], "gate", uuid)
                            continue
                        if worker["phase"] == "gate":
                            lanes[method] = launch(worker["run_id"], "run", uuid)
                            continue
                        verify(worker["run_id"], uuid)
                        completed.append(worker["run_id"])
                        lanes[method] = None
                    remaining = [rid for rid in rows if rid not in completed]
                    if remaining:
                        lanes[method] = launch(remaining[0], "check", uuid)
                workers = [{k: v for k, v in w.items() if k != "process"} for w in lanes.values() if w]
                primary = lanes.get("simmatch") or lanes.get("softmatch")
                state.update(completed=completed, workers=workers, updated_utc=now(),
                             current_run=primary["run_id"] if primary else None,
                             child_pid=primary["pid"] if primary else None,
                             child_start_ticks=primary["start_ticks"] if primary else None,
                             stdout=primary["stdout"] if primary else None,
                             stderr=primary["stderr"] if primary else None)
                atomic_json(ROOT / "outputs/pipeline_state.json", state)
                atomic_json(HERE / "parallel_state.json", state)
                time.sleep(5)
            atomic_json(ROOT / "reports/summary.json", {
                "configurations": 40, "results": {rid: read(ROOT / "outputs" / rid / "result.json") for rid in queue["queue"]},
                "single_seed": 20260825, "no_Test": True, "equal_compute_claim": False,
                "execution_mode": "two_algorithm_parallel", "completed_utc": now()})
            state.update(state="completed", phase="complete", current_run=None, completed_utc=now(), updated_utc=now())
            atomic_json(ROOT / "outputs/pipeline_state.json", state)
            atomic_json(HERE / "parallel_state.json", state)
        except BaseException:
            state.update(state="failed", error=traceback.format_exc(), updated_utc=now())
            state["workers"] = [{k: v for k, v in w.items() if k != "process"} for w in lanes.values() if w]
            atomic_json(ROOT / "outputs/pipeline_state.json", state)
            atomic_json(HERE / "parallel_state.json", state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--takeover-original", action="store_true", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with exclusive(ROOT / "locks/parallel_dispatcher.lock"):
        main(args)
