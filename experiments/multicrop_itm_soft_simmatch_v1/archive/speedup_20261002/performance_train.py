"""Versioned performance worker; registered scientific inputs/code remain untouched."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import base_train as base
import train as legacy
from common import atomic_json, digest, exclusive, read
from perf_cache import ParallelDecodeCache, TensorLRU, apply_options, close_cache

ORIGINAL_PROVENANCE = base.provenance
ORIGINAL_RUNNER = legacy.Runner
POLICY = read(HERE / "accepted_policy.json")


def performance_provenance(cfg, manifest):
    original = ORIGINAL_PROVENANCE(cfg, manifest)
    files = {name: digest(HERE / name) for name in ("performance_train.py", "perf_cache.py")}
    if files != POLICY["worker_files"]:
        raise RuntimeError("Accepted performance source changed; do not overwrite active version")
    execution = {"version": POLICY["version"], "options": POLICY["options"],
                 "worker_files": files, "benchmark_sha256": POLICY["benchmark_sha256"],
                 "two_algorithm_parallel_authorized": True,
                 "cpu_feature_cache_bytes": {"image": 8 * 1024**3, "text": 1024**3},
                 "scientific_protocol_sha256": cfg["protocol_sha256"]}
    receipt = ROOT / "audit/performance_runs" / (cfg["run_id"] + ".json")
    if receipt.exists():
        if read(receipt) != execution:
            raise RuntimeError("This run was registered for another performance version")
    else:
        atomic_json(receipt, execution)
    return {**original, "performance_execution": execution}


class PerformanceRunner(ORIGINAL_RUNNER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure()

    def configure(self, restored_physical=None):
        options = dict(POLICY["options"])
        if restored_physical is not None:
            options["physical_pairs"] = min(options["physical_pairs"], restored_physical)
        close_cache(self)
        apply_options(self, options)
        if not isinstance(self.cache.images, TensorLRU):
            self.cache.images = TensorLRU(8 * 1024**3)
            self.cache.text = TensorLRU(1024**3)

    def load(self, payload):
        super().load(payload)
        self.configure(restored_physical=payload["physical"])

    def train_step(self):
        if isinstance(self.cache, ParallelDecodeCache):
            self.cache.next_step(self.l, self.u, self.step + 1)
            self.cache.next_step(self.l, self.u, self.step + 2)
        result = super().train_step()
        result["performance_execution_version"] = POLICY["version"]
        return result


base.provenance = performance_provenance
base.Runner = PerformanceRunner
legacy.Runner = PerformanceRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--mode", choices=("check", "gate", "run", "verify"), required=True)
    args = parser.parse_args()
    if args.gpu_uuid != read(ROOT / "configs/protocol.json")["server"]["gpu_uuid"]:
        raise RuntimeError("GPU identity changed")
    try:
        if args.mode == "check":
            _, _, _, step = legacy.check(args.run_id)
            print(json.dumps({"passed": True, "run_id": args.run_id, "resume_step": step}))
        elif args.mode == "verify":
            result = legacy.verify_result(args.run_id)
            print(json.dumps({"passed": True, "run_id": args.run_id, "steps": result["successful_steps"]}))
        else:
            # Two distinct run locks, not the old GPU-wide serial lock. This
            # exception is explicit in each run's performance provenance.
            with exclusive(ROOT / "locks" / (args.run_id + ".lock")):
                if args.mode == "gate":
                    legacy.gate(args.run_id, args.gpu_uuid)
                else:
                    base.run(args.run_id, args.gpu_uuid)
    except BaseException:
        atomic_json(ROOT / "audit/failures" / f"{args.run_id}_performance_{os.getpid()}.json",
                    {"run_id": args.run_id, "mode": args.mode, "error": traceback.format_exc(),
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        raise


if __name__ == "__main__":
    main()
