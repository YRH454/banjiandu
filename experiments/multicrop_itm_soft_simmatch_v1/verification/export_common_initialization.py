"""CPU-only export of exact registered step-zero values, never a trained best."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import sys
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from common import ROOT, read, digest, now, atomic_json, atomic_torch
from pair_model import PairITMModel
from base_train import common_digest

cfg = read(ROOT / "configs/runs/apple_001_softmatch_s20260825.json")
checkpoint = ROOT.parents[1] / "weights/ALBEF_4M.pth"
model = PairITMModel({**cfg["model"], "seed": cfg["seed"], "checkpoint": str(checkpoint)})
state = model.trainable_state()
legacy = hashlib.sha256(b"".join(v.numpy().tobytes() for _, v in sorted(state.items()))).hexdigest()
assert len(state) == 34 and legacy == cfg["legacy_common_tensor_sha256"], "CPU export does not match registered initial bytes"
path = ROOT / "assets/common_initialization.pt"
atomic_torch(path, state)
atomic_json(ROOT / "audit/common_initialization_export.json", {
    "passed": True, "cpu_only": True, "optimizer_updates": 0, "no_BCE_best_loaded": True,
    "path": "assets/common_initialization.pt", "sha256": digest(path), "bytes": path.stat().st_size,
    "legacy_common_tensor_sha256": legacy, "common_34_state_sha256": common_digest(state),
    "source_official_checkpoint_sha256": digest(checkpoint), "source_official_checkpoint_md5": model.load_report["checkpoint_md5"],
    "seed": cfg["seed"], "keys": list(state), "per_tensor_sha256": {
        k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()},
    "source_script_sha256": digest(__file__), "created_utc": now()})
print("EXACT_REGISTERED_INITIALIZATION_EXPORTED", legacy, digest(path))
