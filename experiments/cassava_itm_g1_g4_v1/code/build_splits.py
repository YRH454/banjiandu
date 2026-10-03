"""Freeze one five-class-stratified 80/10/10 split for all training seeds.

Usage: python code/build_splits.py
This reads the audited index, writes IDs only, and never changes source data.
Existing split files are immutable: a changed rerun fails instead of replacing them.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "image_index.jsonl"
AUDIT = ROOT / "audit" / "source_audit.json"
SNAPSHOT = ROOT / "data" / "source_snapshot.json"
SPLITS = ROOT / "data" / "splits"
SPLIT_SEED = 20261001
RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
CLASSES = ("CBB", "CBSD", "CGM", "CMD", "Healthy")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen split differs; refusing overwrite: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".split-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if path.exists():
            if path.read_bytes() != data:
                raise RuntimeError(f"Frozen split differs; refusing overwrite: {path}")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def allocate(group_sizes: dict[str, int], fraction: float, target: int) -> dict[str, int]:
    base = {name: math.floor(size * fraction) for name, size in group_sizes.items()}
    order = sorted(group_sizes, key=lambda name: (-(group_sizes[name] * fraction - base[name]), name))
    for name in order[:target - sum(base.values())]:
        base[name] += 1
    if sum(base.values()) != target:
        raise AssertionError("Stratified allocation total mismatch")
    return base


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if not audit["source_integrity_passed"] or audit["issue_counts"]:
        raise RuntimeError("Source audit has not passed")
    if audit["source_revision"] != snapshot["source_revision"]:
        raise RuntimeError("Source revision mismatch")
    if audit["manifest_sha256"] != snapshot["image_manifest_sha256"]:
        raise RuntimeError("Source manifest SHA-256 mismatch")
    if audit["captions_sha256"] != snapshot["captions_sha256"]:
        raise RuntimeError("Source caption SHA-256 mismatch")
    rows = [json.loads(line) for line in INDEX.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != audit["validated_rows"] or len(rows) != len({row["image_id"] for row in rows}):
        raise RuntimeError("Index count or ID uniqueness mismatch")
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        if row["class_code"] not in CLASSES:
            raise RuntimeError(f"Unexpected class: {row['class_code']}")
        groups[row["class_code"]].append(row)
    sizes = {name: len(groups[name]) for name in CLASSES}
    if sizes != audit["class_counts"]:
        raise RuntimeError("Index class counts differ from source audit")

    total = len(rows)
    targets = {"train": round(total * RATIOS["train"]),
               "validation": round(total * RATIOS["validation"])}
    targets["test"] = total - targets["train"] - targets["validation"]
    train_n = allocate(sizes, RATIOS["train"], targets["train"])
    validation_n = allocate(sizes, RATIOS["validation"], targets["validation"])
    rng = random.Random(SPLIT_SEED)
    partitions: dict[str, list[dict]] = {name: [] for name in RATIOS}
    for name in CLASSES:
        group = sorted(groups[name], key=lambda row: row["image_id"])
        rng.shuffle(group)
        a, b = train_n[name], validation_n[name]
        partitions["train"].extend(group[:a])
        partitions["validation"].extend(group[a:a + b])
        partitions["test"].extend(group[a + b:])

    split_ids = {name: {row["image_id"] for row in group} for name, group in partitions.items()}
    if any(split_ids[a] & split_ids[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Split overlap")
    if set.union(*split_ids.values()) != {row["image_id"] for row in rows}:
        raise AssertionError("Split coverage mismatch")

    files = {}
    summaries = {}
    for name, group in partitions.items():
        data = ("\n".join(sorted(split_ids[name])) + "\n").encode("utf-8")
        files[name] = {"file": f"{name}_ids.txt", "sha256": sha256(data), "count": len(group)}
        by_class = collections.Counter(row["class_code"] for row in group)
        by_subject = collections.Counter(row["subject"] for row in group)
        non_leaf = sum(by_subject[subject] for subject in ("root", "stem", "other"))
        summaries[name] = {"count": len(group),
                           "class_counts": {code: by_class[code] for code in CLASSES},
                           "subject_counts": dict(sorted(by_subject.items())),
                           "non_leaf_count": non_leaf}
        if len(group) != targets[name]:
            raise AssertionError(f"{name} count mismatch")
        write_immutable(SPLITS / files[name]["file"], data)

    manifest = {
        "status": "frozen_split_not_training_authorization",
        "source_repo": snapshot["repo_id"], "source_revision": snapshot["source_revision"],
        "manifest_sha256": snapshot["image_manifest_sha256"],
        "captions_sha256": snapshot["captions_sha256"],
        "image_index_sha256": file_sha256(INDEX),
        "split_seed": SPLIT_SEED, "training_seeds_independent": [20260825, 20260826, 20260827],
        "ratios": RATIOS, "stratification": "class_code; deterministic largest-remainder counts",
        "non_leaf_policy": "retained_and_tagged", "near_duplicate_screen_candidates_le4": audit["near_duplicate_screen"]["candidate_pairs"],
        "files": files, "summary": summaries,
        "checks": {"all_ids_covered": True, "no_split_overlap": True,
                   "source_audit_passed": True, "class_counts_match_source": True},
    }
    manifest_data = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_immutable(SPLITS / "split_manifest.json", manifest_data)
    print(json.dumps({"split_seed": SPLIT_SEED, "summary": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
