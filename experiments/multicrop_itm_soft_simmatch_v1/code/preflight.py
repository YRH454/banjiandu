"""Read-only CPU checks of all 20 exported contracts and 40 allocated configs."""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import sys
import importlib.metadata
import torch
from common import ROOT, read, digest, code_hashes, now, atomic_json
from data_backend import load_contract
sys.path.insert(0, str(ROOT / "core"))
from albef_ssl.model import get_tokenizer

def main():
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "numpy", "Pillow", "transformers", "timm", "scikit-learn", "huggingface-hub")}
    if versions["torch"] != "2.5.1+cu124" or versions["numpy"] != "1.26.4":
        raise RuntimeError("Unregistered base runtime/version")
    atomic_json(ROOT / "audit/environment.json", {"python": sys.version, "executable": sys.executable,
        "versions": versions, "torch_cuda": torch.version.cuda, "platform": sys.platform, "created_utc": now()})
    assets = read(ROOT / "audit/model_assets.json")
    for path, meta in assets["files"].items():
        if digest(ROOT / path) != meta["sha256"]:
            raise RuntimeError("Official asset SHA256 mismatch")
    initial = read(ROOT / "audit/common_initialization_export.json")
    if not initial["passed"] or initial["optimizer_updates"] != 0 or digest(ROOT / initial["path"]) != initial["sha256"]:
        raise RuntimeError("Registered step-zero initialization transfer mismatch")
    tensors = torch.load(ROOT / initial["path"], map_location="cpu", weights_only=True)
    import hashlib
    legacy = hashlib.sha256(b"".join(v.numpy().tobytes() for _, v in sorted(tensors.items()))).hexdigest()
    if len(tensors) != 34 or legacy != read(ROOT / "configs/protocol.json")["legacy_common_tensor_sha256"]:
        raise RuntimeError("Common initial values do not exactly equal the original reference")
    if digest(ROOT / "assets/ALBEF_4M.pth", "md5") != "3c876d776a8e0ce61e2285fc9897f0b3":
        raise RuntimeError("Not official ALBEF4M")
    tokenizer = get_tokenizer(str(ROOT / "assets/tokenizer"))
    rows = []
    for crop in ("apple", "cassava", "rice", "banana"):
        for budget in ("001", "005", "010", "020", "030"):
            manifest, index, l, u, v = load_contract(crop, budget)
            export = read(ROOT / "audit" / f"export_{crop}.json")
            if digest(ROOT / "data" / crop / "manifest.json") != export["manifest_sha256"]:
                raise RuntimeError("Source-registered dataset manifest mismatch")
            captions = list({p[k] for p in l+v for k in ("positive_text", "negative_text")} | {p["text"] for p in u})
            lengths = [len(ids) for ids in tokenizer(captions, truncation=False, padding=False)["input_ids"]]
            if max(lengths) > manifest["token_guard"]:
                raise RuntimeError("Full caption guard violated; no truncation")
            rows.append({"crop": crop, "budget": budget, "L": len(l), "U_pairs": len(u), "Validation_images": len(v),
                         "max_text_tokens": max(lengths), "guard": manifest["token_guard"]})
            print("CPU public contract passed", crop, budget, flush=True)
    queue = read(ROOT / "configs/queue.json")
    for run_id in queue["queue"]:
        c = read(ROOT / "configs/runs" / (run_id+".json"))
        if c["protocol_sha256"] != digest(ROOT / "configs/protocol.json") or c["target_steps"] not in (2200, 2400):
            raise RuntimeError("Config target/provenance mismatch")
    atomic_json(ROOT / "audit/cpu_preflight.json", {"passed": True, "public_contracts": rows,
        "configurations": len(queue["queue"]), "private_U_read": False, "Test_inference": False,
        "source": code_hashes(), "created_utc": now()})
    print("CPU preflight complete: 20 data contracts /40 configs; no CUDA kernels")

if __name__ == "__main__":
    main()
