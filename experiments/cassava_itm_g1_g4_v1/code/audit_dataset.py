"""Read-only cassava source audit; outputs contain no caption text or images.

Usage: python code/audit_dataset.py
The source is never modified.  The JSON/JSONL outputs are generated locally.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"X:\PATH\datasets\EGOISTyrh\mushubanjiandu")
EXPECTED_REVISION = "860cac3804b0b9347ae513730615a8d060d478db"
EXPECTED_CLASSES = {"CBB": 1086, "CBSD": 2188, "CGM": 2381,
                    "Healthy": 2574, "CMD": 2500}
SUBJECTS = {"leaf", "whole-plant", "root", "stem", "other"}
FORBIDDEN_BLIND = {
    "bacterial_blight": r"\bbacterial\s+blight\b",
    "brown_streak_disease": r"\bbrown\s+streak\s+disease\b",
    "green_mottle_diagnosis": r"\bgreen\s+mottle\s+disease\b",
    "mosaic_disease": r"\bmosaic\s+disease\b",
    "pathogen": r"\b(?:virus|bacteri(?:a|al)|pathogen)\b",
    "explicit_diagnosis": r"\b(?:diagnosed|infected|diseased|healthy)\b",
}


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".audit-", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            stream.close()
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def dhash(image: Image.Image) -> int:
    pixels = list(ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS).getdata())
    result = 0
    for row in range(8):
        for column in range(8):
            result = (result << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return result


def near_duplicate_summary(rows: list[dict]) -> dict:
    """dHash is a candidate screen, never an automatic exclusion decision."""
    buckets: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    pairs: set[tuple[int, int]] = set()
    for index, row in enumerate(rows):
        value = int(row["dhash64"], 16)
        for band in range(8):
            key = (band, (value >> (8 * band)) & 255)
            for other in buckets[key]:
                pairs.add((other, index))
            buckets[key].append(index)
    counts = collections.Counter()
    examples = []
    for left, right in pairs:
        distance = bin(int(rows[left]["dhash64"], 16) ^ int(rows[right]["dhash64"], 16)).count("1")
        if distance > 4:
            continue
        counts[f"distance_le_{distance}"] += 1
        if rows[left]["class_code"] != rows[right]["class_code"]:
            counts["cross_class_distance_le_4"] += 1
        if len(examples) < 100:
            examples.append({"left": rows[left]["image_id"], "right": rows[right]["image_id"],
                             "distance": distance, "cross_class": rows[left]["class_code"] != rows[right]["class_code"]})
    return {"screen": "64-bit dHash Hamming <=4; candidates may be unrelated",
            "candidate_pairs": sum(v for k, v in counts.items() if k.startswith("distance_le_")),
            "cross_class_pairs": counts["cross_class_distance_le_4"],
            "exact_dhash_pairs": counts["distance_le_0"],
            "by_distance": {str(d): counts[f"distance_le_{d}"] for d in range(5)},
            "examples_max_100": sorted(examples, key=lambda r: (r["distance"], r["left"]))}


def tokenizer_summary(texts: list[str]) -> dict:
    snapshot = Path(r"X:\PATH\huggingface-cache\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594")
    if not snapshot.is_dir():
        return {"state": "tokenizer_unavailable", "required_limit": 256}
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(str(snapshot), local_files_only=True)
    lengths = []
    for begin in range(0, len(texts), 256):
        encoded = tokenizer(texts[begin:begin + 256], add_special_tokens=True,
                            truncation=False, padding=False, return_attention_mask=False)
        lengths.extend(map(len, encoded["input_ids"]))
    lengths.sort()
    return {"state": "measured", "tokenizer": "bert-base-uncased", "tokenizer_revision": snapshot.name,
            "limit_including_special_tokens": 256, "n": len(lengths),
            "min": lengths[0], "p50": lengths[len(lengths) // 2],
            "p95": lengths[int(len(lengths) * .95)], "max": lengths[-1],
            "over_256": sum(n > 256 for n in lengths),
            "over_256_fraction": sum(n > 256 for n in lengths) / len(lengths)}


def main() -> None:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"Source revision changed: {revision}")
    manifest_path = SOURCE / "image_manifest.jsonl"
    captions_path = SOURCE / "captions.json"
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    issues: list[dict] = []
    counts = collections.Counter()
    subject_by_class: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    blind_hits = collections.Counter()
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    file_hashes: dict[str, str] = {}
    pixel_hashes: dict[str, str] = {}
    blind_hashes: dict[str, str] = {}
    rows: list[dict] = []
    texts: list[str] = []

    def issue(kind: str, image_id: str, detail: str = "") -> None:
        counts[f"issue_{kind}"] += 1
        if len(issues) < 200:
            issues.append({"kind": kind, "image_id": image_id, "detail": detail})

    for item in manifest:
        image_id = Path(item["image_id"]).stem
        relpath = item["path"].replace("\\", "/")
        path = SOURCE / relpath
        if image_id in seen_ids:
            issue("duplicate_id", image_id)
        if relpath in seen_paths:
            issue("duplicate_path", image_id)
        seen_ids.add(image_id)
        seen_paths.add(relpath)
        if not relpath.startswith("images/") or ".." in Path(relpath).parts:
            issue("unsafe_path", image_id, relpath)
            continue
        if not path.is_file():
            issue("missing_image", image_id, relpath)
            continue
        actual_size = path.stat().st_size
        if actual_size != item["bytes"]:
            issue("size_mismatch", image_id, f"{actual_size} != {item['bytes']}")
        actual_hash = sha256_file(path)
        if actual_hash != item["sha256"]:
            issue("sha256_mismatch", image_id)
        if actual_hash in file_hashes:
            issue("exact_file_duplicate", image_id, file_hashes[actual_hash])
        else:
            file_hashes[actual_hash] = image_id
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                image_format = image.format
                view_hash = dhash(image)
        except Exception as exc:
            issue("image_decode", image_id, repr(exc))
            continue
        if (width, height) != (item["width"], item["height"]):
            issue("dimensions_mismatch", image_id, f"{width}x{height}")
        if image_format != "JPEG":
            issue("not_jpeg", image_id, str(image_format))
        caption = captions.get(image_id)
        if not isinstance(caption, dict):
            issue("missing_caption", image_id)
            continue
        if (caption.get("relpath") != relpath or caption.get("class_code") != item["class_code"]
                or caption.get("sha256") != actual_hash or caption.get("source_class") != item["source_class"]):
            issue("caption_identity_mismatch", image_id)
        subject = caption.get("subject")
        if subject not in SUBJECTS:
            issue("invalid_subject", image_id, str(subject))
        blind = caption.get("blind")
        guided = caption.get("guided")
        if not isinstance(blind, str) or not blind.strip():
            issue("empty_blind", image_id)
            continue
        blind_hash = hashlib.sha256(blind.encode("utf-8")).hexdigest()
        if blind_hash in blind_hashes:
            issue("duplicate_blind_caption", image_id, blind_hashes[blind_hash])
        else:
            blind_hashes[blind_hash] = image_id
        if not isinstance(guided, str) or not guided.strip():
            issue("empty_guided", image_id)
        if not caption.get("guided_parsed"):
            issue("guided_unparsed", image_id)
        for name, pattern in FORBIDDEN_BLIND.items():
            if re.search(pattern, blind, flags=re.IGNORECASE):
                blind_hits[name] += 1
        pixel_hash = caption.get("pixel_sha256")
        if pixel_hash:
            if pixel_hash in pixel_hashes:
                issue("same_pixel_hash", image_id, pixel_hashes[pixel_hash])
            else:
                pixel_hashes[pixel_hash] = image_id
        code = item["class_code"]
        counts[code] += 1
        subject_by_class[code][subject] += 1
        texts.append(blind)
        rows.append({"image_id": image_id, "relpath": relpath, "class_code": code,
                     "source_label_id": item["source_label_id"], "subject": subject,
                     "sha256": actual_hash, "blind_sha256": blind_hash,
                     "blind_characters": len(blind), "dhash64": f"{view_hash:016x}"})

    image_paths = {p.relative_to(SOURCE).as_posix() for p in (SOURCE / "images").rglob("*.jpg")}
    extra = image_paths - seen_paths
    missing = seen_paths - image_paths
    if extra:
        issue("unlisted_images", "", f"{len(extra)} paths; first {sorted(extra)[:5]}")
    if missing:
        issue("listed_but_absent", "", f"{len(missing)} paths; first {sorted(missing)[:5]}")
    extra_captions = set(captions) - seen_ids
    if extra_captions:
        issue("unlisted_captions", "", f"{len(extra_captions)} IDs; first {sorted(extra_captions)[:5]}")
    if {code: counts[code] for code in EXPECTED_CLASSES} != EXPECTED_CLASSES:
        issue("class_count", "", str({code: counts[code] for code in EXPECTED_CLASSES}))
    near = near_duplicate_summary(rows)
    tokens = tokenizer_summary(texts)
    report = {
        "source_repo": "EGOISTyrh/mushubanjiandu", "source_revision": revision,
        "source_path": str(SOURCE), "manifest_sha256": sha256_file(manifest_path),
        "captions_sha256": sha256_file(captions_path),
        "image_count_manifest": len(manifest), "image_count_disk": len(image_paths),
        "caption_count": len(captions), "validated_rows": len(rows),
        "class_counts": {code: counts[code] for code in EXPECTED_CLASSES},
        "subject_by_class": {code: dict(subject_by_class[code]) for code in EXPECTED_CLASSES},
        "non_leaf_count": sum(subject_by_class[c][s] for c in EXPECTED_CLASSES for s in ("root", "stem", "other")),
        "blind_hard_term_hits": dict(blind_hits), "token_lengths": tokens,
        "near_duplicate_screen": near, "issue_counts": {k: v for k, v in counts.items() if k.startswith("issue_")},
        "issue_examples_max_200": issues,
        "source_integrity_passed": not any(k.startswith("issue_") for k in counts),
    }
    index = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    atomic_bytes(ROOT / "data/image_index.jsonl", index)
    atomic_bytes(ROOT / "audit/source_audit.json", (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"validated_rows": len(rows), "issue_counts": report["issue_counts"],
                      "non_leaf_count": report["non_leaf_count"], "token_lengths": tokens,
                      "near_duplicate_candidates": near["candidate_pairs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
