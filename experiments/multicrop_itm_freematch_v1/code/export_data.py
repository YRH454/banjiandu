"""Export public frozen training contracts and their images, never private U/Test.

Only supervised columns required by ITM are exported. U tables stay byte-for-byte
identical; path mapping is in a separate public asset index. Sources are read-only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path
from common import ROOT, atomic_json, digest, read, now

ORIGINAL = Path("/path/to/albef/experiments")
SOURCES = {
    "apple": ("apple_itm_pairusa_random_v2", "apple_itm_usa_ot_ablation_v1", "data/manifest.json", 256),
    "cassava": ("cassava_itm_g1_g4_v1", "cassava_itm_g1_g4_v1", "data/u_manifest.json", 384),
    "rice": ("rice_itm_s1_s4_v1", "rice_itm_s1_s4_v1", "data/u_manifest.json", 400),
    "banana": ("banana_itm_s1_s4_v1", "banana_itm_s1_s4_v1", "data/u_manifest.json", 384),
}
BUDGETS = ("001", "005", "010", "020", "030")
L_FIELDS = ("image_id", "image_relpath", "positive_text", "negative_text",
            "source_text_sha256", "negative_text_sha256", "image_sha256")
U_FIELDS = ["pair_id", "image_id", "image_path", "text", "text_sha256"]
TOKENIZER = Path("/path/to/huggingface-cache/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")

def csv_rows(path, expected=None):
    if expected and digest(path) != expected:
        raise RuntimeError(f"Registered input changed: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    return fields, rows

def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def source_path(root, rel):
    rel = rel.replace("\\", "/")
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ValueError("Unsafe source relative path")
    candidate = root / rel
    if not candidate.is_file() and rel.startswith("raw/"):
        candidate = root / rel[4:]
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate

def export_crop(crop, tokenizer):
    l_name, u_name, u_manifest_rel, guard = SOURCES[crop]
    l_root, u_root = ORIGINAL / l_name, ORIGINAL / u_name
    manifest = read(l_root / "data/manifest.json")
    u_manifest = read(u_root / u_manifest_rel)
    image_root = Path(manifest["image_root"])
    destination = ROOT / "data" / crop
    destination.mkdir(parents=True, exist_ok=True)
    mapping, originals, contracts, all_text = {}, {}, {}, {}
    expected_image_hashes = {}
    def add_asset(image_id, rel, expected=None):
        actual = source_path(image_root, rel)
        portable = "assets/images/" + crop + "/" + actual.relative_to(image_root).as_posix()
        if image_id in mapping and mapping[image_id]["path"] != portable:
            raise ValueError(f"Non-unique image mapping {image_id}")
        mapping[image_id] = {"path": portable, "source_relpath": rel}
        originals[portable] = actual
        if expected:
            if portable in expected_image_hashes and expected_image_hashes[portable] != expected:
                raise ValueError("Conflicting image hashes")
            expected_image_hashes[portable] = expected
    def l_table(name, path, expected=None):
        _, rows = csv_rows(path, expected)
        ids = {r["image_id"] for r in rows}
        if len(ids) != len(rows):
            raise ValueError("Duplicate L/Validation images")
        donors = {r["image_id"]: r for r in rows}
        for row in rows:
            if text_hash(row["positive_text"]) != row["source_text_sha256"] or text_hash(row["negative_text"]) != row["negative_text_sha256"]:
                raise ValueError("Modified full caption")
            donor = donors[row["negative_source_image_id"]]
            if donor["image_id"] == row["image_id"] or donor["positive_text"] != row["negative_text"]:
                raise ValueError("Modified fixed other-case pairing")
            if row["review_status"] != "random_other_case_unverified":
                raise ValueError("Unexpected negative annotation protocol")
            add_asset(row["image_id"], row["image_relpath"], row["image_sha256"])
            for key in ("positive_text", "negative_text"):
                all_text[text_hash(row[key])] = row[key]
        target = destination / f"{name}.csv"
        if target.exists():
            raise FileExistsError(f"Refusing overwrite of exported table {target}")
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=L_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return ids, {"path": target.relative_to(ROOT).as_posix(), "sha256": digest(target),
                     "source_path": str(path), "source_sha256": digest(path), "anchors": len(rows),
                     "derivation": "whitelisted_ITM_columns_only_preserve_order_ids_text_hashes_pairing"}
    val_ids, validation = l_table("validation", l_root / manifest["files"]["validation"])
    if len(val_ids) != 400:
        raise ValueError("Frozen Validation must have 400 anchors")
    previous, train_ids = set(), None
    for budget in BUDGETS:
        registration = u_manifest["budgets"][budget]
        l_path = l_root / manifest["files"]["budgets"][budget]
        l_ids, l_contract = l_table(f"l_{budget}", l_path, registration["l_pairs_sha256"])
        if not previous <= l_ids or len(l_ids) != int(registration["l_count"]):
            raise ValueError("Frozen nested L contract changed")
        previous = l_ids
        u_path = u_root / registration["path"]
        fields, u_rows = csv_rows(u_path, registration["sha256"])
        if fields != U_FIELDS:
            raise ValueError("U public input must have exactly five allowed fields")
        if len({r["pair_id"] for r in u_rows}) != len(u_rows):
            raise ValueError("Duplicate U pair IDs")
        u_ids = {r["image_id"] for r in u_rows}
        if l_ids & u_ids or (l_ids | u_ids) & val_ids:
            raise ValueError("L/U/Validation image leakage")
        if train_ids is not None and train_ids != l_ids | u_ids:
            raise ValueError("Train pool changes across budgets")
        train_ids = l_ids | u_ids
        for row in u_rows:
            if text_hash(row["text"]) != row["text_sha256"]:
                raise ValueError("Modified U caption")
            add_asset(row["image_id"], row["image_path"])
            all_text[row["text_sha256"]] = row["text"]
        target = destination / f"u_{budget}.csv"
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(u_path, target)
        contracts[budget] = {"l": l_contract, "u": {"path": target.relative_to(ROOT).as_posix(),
            "sha256": digest(target), "source_path": str(u_path), "source_sha256": digest(u_path),
            "pairs": len(u_rows), "images": len(u_ids), "derivation": "identical_public_bytes"}}
    longest = max(len(ids) for ids in tokenizer(list(all_text.values()), truncation=False, padding=False)["input_ids"])
    if longest > guard:
        raise ValueError(f"Caption exceeds guard {longest}>{guard}; no truncation allowed")
    print(crop, "hashing", len(originals), "images", flush=True)
    for i, (portable, actual) in enumerate(sorted(originals.items())):
        h = digest(actual)
        if portable in expected_image_hashes and h != expected_image_hashes[portable]:
            raise ValueError(f"Image hash mismatch {actual}")
        for row in (v for v in mapping.values() if v["path"] == portable):
            row.update(sha256=h, bytes=actual.stat().st_size)
        if i % 2000 == 0:
            print(crop, "hashed", i, flush=True)
    index = destination / "assets.json"
    atomic_json(index, mapping)
    output = {"crop": crop, "created_utc": now(), "token_guard": guard, "max_observed_tokens": longest,
        "train_images": len(train_ids), "validation_images": len(val_ids), "validation": validation,
        "budgets": contracts, "asset_index": {"path": index.relative_to(ROOT).as_posix(), "sha256": digest(index)},
        "source_l_manifest_sha256": digest(l_root / "data/manifest.json"),
        "source_u_manifest_sha256": digest(u_root / u_manifest_rel), "private_u_read": False,
        "test_images_exported": False, "negative_policy": manifest["negative_policy"]}
    atomic_json(destination / "manifest.json", output)
    archive = ROOT / "staging" / f"{crop}_public_assets.tar"
    with tarfile.open(archive, "w") as tar:
        for path in sorted(destination.glob("*")):
            tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        for portable, actual in sorted(originals.items()):
            tar.add(actual, arcname=portable, recursive=False)
    atomic_json(ROOT / "audit" / f"export_{crop}.json", {"manifest_sha256": digest(destination / "manifest.json"),
        "archive": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size,
        "private_u_read": False, "test_inference": False, "images": len(originals), "created_utc": now()})
    print(crop, "export complete", archive.stat().st_size, flush=True)

def assets_archive():
    archive = ROOT / "staging/model_assets.tar"
    checkpoint = Path("/path/to/albef/weights/ALBEF_4M.pth")
    if digest(checkpoint, "md5") != "3c876d776a8e0ce61e2285fc9897f0b3":
        raise ValueError("Official checkpoint changed")
    entries = {"assets/ALBEF_4M.pth": checkpoint}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt"):
        entries["assets/tokenizer/" + name] = TOKENIZER / name
    with tarfile.open(archive, "w", dereference=True) as tar:
        for portable, path in entries.items():
            tar.add(path, arcname=portable, recursive=False)
    atomic_json(ROOT / "audit/model_assets.json", {"files": {k: {"sha256": digest(p), "bytes": p.stat().st_size} for k, p in entries.items()},
        "archive_sha256": digest(archive), "created_utc": now()})
    print("model assets complete", archive.stat().st_size, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", choices=list(SOURCES))
    parser.add_argument("--model-assets", action="store_true")
    args = parser.parse_args()
    if args.model_assets:
        assets_archive()
    if args.crop:
        from transformers import BertTokenizerFast
        export_crop(args.crop, BertTokenizerFast.from_pretrained(str(TOKENIZER), local_files_only=True))
