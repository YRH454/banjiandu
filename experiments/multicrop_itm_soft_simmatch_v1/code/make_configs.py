"""Allocate only the final two baselines, all frozen datasets/budgets, single GPU."""
from common import ROOT, read, digest, atomic_json, now

def main():
    p = read(ROOT / "configs/protocol.json")
    queue = []
    for crop in p["datasets"]:
        for budget in p["budgets"]:
            for method in p["targets"]:
                run_id = f"{crop}_{budget}_{method}_s{p['seed']}"
                cfg = {"run_id": run_id, "dataset": crop, "budget": budget, "method": method,
                    "seed": p["seed"], "target_steps": p["targets"][method], "launch_enabled": True,
                    "protocol_sha256": digest(ROOT / "configs/protocol.json"),
                    "model": p["model"], "training": p["training"], "method_settings": p[method],
                    "evaluation": p["evaluation"], "augmentation": p["augmentation"],
                    "initialization": p["initialization"], "legacy_common_tensor_sha256": p["legacy_common_tensor_sha256"]}
                path = ROOT / "configs/runs" / (run_id + ".json")
                if path.exists() and read(path) != cfg:
                    raise RuntimeError("Immutable config differs: " + run_id)
                if not path.exists():
                    atomic_json(path, cfg)
                queue.append(run_id)
    atomic_json(ROOT / "configs/queue.json", {"queue": queue, "server": p["server"],
        "assigned_count": len(queue), "protocol_sha256": digest(ROOT / "configs/protocol.json")})
    atomic_json(ROOT / "audit/assignment_registry.json", {"created_utc": now(), "assigned": queue,
        "host": p["server"]["ssh_host"], "port": p["server"]["ssh_port"], "gpu": p["server"]["gpu_uuid"],
        "single_gpu_serial": True, "windows_first_two_runs_untouched": True,
        "duplicate_MT_FixMatch_assignments": False})
    print("Registered", len(queue), "single-GPU serial configurations")

if __name__ == "__main__":
    main()
