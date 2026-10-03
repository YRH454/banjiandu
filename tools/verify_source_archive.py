"""Verify the whitelisted public source snapshot without importing training code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "experiments/multicrop_itm_soft_simmatch_v1/source_manifest.json"


def verify(manifest_path: Path = DEFAULT) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format"] != "public_source_archive_v1":
        raise ValueError("Unknown archive format")
    seen = set()
    for entry in manifest["files"]:
        name = entry["repository_path"]
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or name in seen:
            raise ValueError("Unsafe or duplicate archive path")
        seen.add(name)
        if not path.is_file():
            raise ValueError(f"Missing source: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Source SHA256 mismatch: {name}")
    if not seen:
        raise ValueError("Empty archive")
    return len(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(f"Verified {verify(args.manifest)} public source files (no data/GPU access)")
