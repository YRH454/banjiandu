# -*- coding: utf-8 -*-
"""Prepare immutable, label-hidden U pairs for the approved OT ablation.

Only registered train_100.csv supplies caption text. Validation/test CSVs
are used solely for split IDs, neutral image paths, and file-hash checks.
Training must never import this module or open audit_private/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AB_ROOT = ROOT.parent / "apple_itm_pairusa_random_v2"
SOURCE = ROOT.parent / "apple_multimodal_ce_bce_v2" / "data"
TOKENIZER = Path(
    "C:/Users/Lenovo/.cache/huggingface/hub/models--bert-base-uncased/"
    "snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
)
SEED = 20260825
COUNTS = {"001": 133, "005": 668, "010": 1337, "020": 2674, "030": 4011}
FIELDS = ("pair_id", "image_id", "image_path", "text", "text_sha256")
PRIVATE_FIELDS = (
    "pair_id", "image_id", "caption_source_image_id", "construction_source_label",
    "source_type", "image_sha256", "caption_source_image_sha256", "text_sha256",
)
POLICY = "u_only_random_other_case_full_blind_caption_v1"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_seed(tag: str) -> int:
    return int(text_hash(f"{SEED}\0{tag}")[:16], 16)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows: list[dict[str, str]], fields: tuple[str, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def immutable_write(path: Path, value: bytes, verify_only: bool) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise RuntimeError(f"Existing artifact differs; refusing mutation: {path}")
        return
    if verify_only:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def unique_ids(rows: list[dict[str, str]], expected: int, name: str) -> set[str]:
    ids = {row["image_id"] for row in rows}
    if len(ids) != len(rows) or len(ids) != expected:
        raise ValueError(f"Incorrect/duplicate {name} count: {len(rows)}, {len(ids)}, expected {expected}")
    return ids


def image_path(row: dict[str, str]) -> Path:
    image_id = row["image_id"]
    expected = f"images/{image_id[:2]}/{image_id}"
    if row["image_relpath"].replace("\\", "/") != expected:
        raise ValueError(f"Non-neutral or unexpected image path: {image_id}")
    path = (SOURCE / expected).resolve()
    if SOURCE.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Image path missing/outside registered root: {image_id}")
    return path


def tokenize_training_captions(rows: list[dict[str, str]]) -> dict:
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(str(TOKENIZER), local_files_only=True)
    expected = {"pad_token_id": 0, "unk_token_id": 100, "cls_token_id": 101,
                "sep_token_id": 102, "mask_token_id": 103}
    if len(tokenizer) != 30522 or not tokenizer.do_lower_case:
        raise ValueError("Tokenizer is not the original bert-base-uncased vocabulary")
    if any(getattr(tokenizer, key) != value for key, value in expected.items()):
        raise ValueError("Tokenizer special IDs differ from registered ALBEF vocabulary")
    lengths = []
    over = []
    for start in range(0, len(rows), 128):
        chunk = rows[start:start + 128]
        result = tokenizer([row["positive_text"] for row in chunk], padding=False, truncation=False)
        for row, token_ids in zip(chunk, result["input_ids"]):
            lengths.append(len(token_ids))
            if len(token_ids) > 256:
                over.append({"image_id": row["image_id"], "tokens": len(token_ids)})
    if over:
        raise ValueError(f"Full captions exceed 256 tokens; no truncation allowed: {over[:10]}")
    return {"path": str(TOKENIZER), "vocabulary_sha256": sha256(TOKENIZER / "vocab.txt"),
            "n_train_captions": len(lengths), "special_tokens_included": True,
            "min_tokens": min(lengths), "max_tokens": max(lengths),
            "mean_tokens": sum(lengths) / len(lengths), "over_256": len(over),
            "truncation": False, "local_files_only": True}


def derange(rows: list[dict[str, str]], budget: str) -> tuple[list[int], int]:
    randomizer = random.Random(stable_seed(f"{POLICY}|derangement|{budget}"))
    count = len(rows)
    for attempt in range(1, 10001):
        donors = list(range(count))
        randomizer.shuffle(donors)
        if all(
            i != j
            and rows[i]["image_id"] != rows[j]["image_id"]
            and rows[i]["positive_text"] != rows[j]["positive_text"]
            and rows[i]["image_sha256"] != rows[j]["image_sha256"]
            for i, j in enumerate(donors)
        ):
            return donors, attempt
    raise RuntimeError(f"No valid U-only derangement found for {budget}")


def build_budget(rows: list[dict[str, str]], budget: str) -> tuple[list[dict], list[dict], dict]:
    rows = sorted(rows, key=lambda row: row["image_id"])
    donors, attempts = derange(rows, budget)
    hidden = []
    for index, donor in enumerate(donors):
        hidden.append((index, index, 1))
        hidden.append((index, donor, 0))
    randomizer = random.Random(stable_seed(f"{POLICY}|flat_shuffle|{budget}"))
    randomizer.shuffle(hidden)
    public_rows = []
    private_rows = []
    for position, (index, donor, source_label) in enumerate(hidden):
        anchor, caption = rows[index], rows[donor]
        # Depends only on a shuffled row's position: no label/donor encoded in ID.
        pair_id = text_hash(f"{SEED}|{POLICY}|opaque_id|{budget}|{position}")[:32]
        public_rows.append({"pair_id": pair_id, "image_id": anchor["image_id"],
                            "image_path": anchor["image_relpath"].replace("\\", "/"),
                            "text": caption["positive_text"],
                            "text_sha256": caption["source_text_sha256"]})
        private_rows.append({"pair_id": pair_id, "image_id": anchor["image_id"],
                             "caption_source_image_id": caption["image_id"],
                             "construction_source_label": str(source_label),
                             "source_type": "own_caption" if source_label else "random_other_case",
                             "image_sha256": anchor["image_sha256"],
                             "caption_source_image_sha256": caption["image_sha256"],
                             "text_sha256": caption["source_text_sha256"]})
    ids = {row["image_id"] for row in rows}
    public_lookup = {row["pair_id"]: row for row in public_rows}
    source_lookup = {row["image_id"]: row for row in rows}
    if len(public_lookup) != len(public_rows):
        raise ValueError("Nonunique opaque pair IDs")
    if Counter(row["image_id"] for row in public_rows) != Counter({key: 2 for key in ids}):
        raise ValueError("Incorrect two-pairs-per-image multiset")
    expected_captions = Counter(row["positive_text"] for row in rows)
    if Counter(row["text"] for row in public_rows) != Counter({key: value * 2 for key, value in expected_captions.items()}):
        raise ValueError("Caption multiset differs from twice the U originals")
    for label in ("0", "1"):
        private_subset = [row for row in private_rows if row["construction_source_label"] == label]
        if Counter(row["caption_source_image_id"] for row in private_subset) != Counter(ids):
            raise ValueError("Not every caption appears once per construction source")
        for metadata in private_subset:
            public = public_lookup[metadata["pair_id"]]
            donor = source_lookup[metadata["caption_source_image_id"]]
            anchor = source_lookup[metadata["image_id"]]
            if public["text"] != donor["positive_text"] or text_hash(public["text"]) != public["text_sha256"]:
                raise ValueError("Caption provenance/verbatim hash mismatch")
            if label == "0" and (
                donor["image_id"] == anchor["image_id"]
                or donor["positive_text"] == anchor["positive_text"]
                or donor["image_sha256"] == anchor["image_sha256"]
            ):
                raise ValueError("Invalid random-other-case negative")
    if any(tuple(row) != FIELDS for row in public_rows):
        raise ValueError("Public table schema contains unexpected fields")
    checks = {"n_images": len(rows), "n_pairs": len(public_rows), "derangement_attempts": attempts,
              "only_u_donors": True, "verbatim_text": True, "opaque_unique_pair_ids": True,
              "public_schema": list(FIELDS), "public_labels_or_donor_metadata": False,
              "two_pairs_per_image": True, "one_caption_per_source_category": True,
              "same_id_caption_or_image_hash_negatives": 0, "flat_table_shuffled": True,
              "construction_sources_balanced": True, "semantic_negative_guarantee": False,
              "semantic_status": "random_other_case_unverified",
              "all_source_labels_stored_only_in_audit_private": True}
    return public_rows, private_rows, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", help="Regenerate deterministically in memory and compare; never write")
    arguments = parser.parse_args()
    source_pair_path = AB_ROOT / "data/pairs/train_100.csv"
    registered_manifest_path = AB_ROOT / "data/manifest.json"
    registered_manifest = json.loads(registered_manifest_path.read_text(encoding="utf-8-sig"))
    if Path(registered_manifest["image_root"]).resolve() != SOURCE.resolve():
        raise ValueError("Registered image root differs")
    train = read_rows(source_pair_path)
    train_ids = unique_ids(train, 13373, "train pool pairs")
    source_pool_path = SOURCE / "train_pool_ids.csv"
    if unique_ids(read_rows(source_pool_path), 13373, "train pool IDs") != train_ids:
        raise ValueError("Original train pool and full A/B pair file disagree")
    validation_path, test_path = SOURCE / "validation.csv", SOURCE / "test.csv"
    validation, test = read_rows(validation_path), read_rows(test_path)
    validation_ids = unique_ids(validation, 1490, "validation IDs")
    test_ids = unique_ids(test, 3715, "test IDs")
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Split ID leakage")
    fixed_validation_path = AB_ROOT / "data/pairs/validation.csv"
    fixed_validation = read_rows(fixed_validation_path)
    fixed_ids = unique_ids(fixed_validation, 400, "fixed validation anchors")
    if not fixed_ids <= validation_ids:
        raise ValueError("Fixed validation anchors cross split")
    for row in train:
        if not row["positive_text"] or text_hash(row["positive_text"]) != row["source_text_sha256"]:
            raise ValueError(f"Original caption hash invalid: {row['image_id']}")
    print("Checking registered train/validation/test image hashes (test caption text is not accessed).", flush=True)
    all_rows = train + validation + test
    with ThreadPoolExecutor(max_workers=8) as workers:
        actual_hashes = list(workers.map(lambda row: sha256(image_path(row)), all_rows))
    image_hashes = dict(zip([row["image_id"] for row in all_rows], actual_hashes))
    for row in train + fixed_validation:
        if image_hashes[row["image_id"]] != row["image_sha256"]:
            raise ValueError(f"Registered image bytes differ: {row['image_id']}")
    train_hashes = {image_hashes[key] for key in train_ids}
    validation_hashes = {image_hashes[key] for key in validation_ids}
    test_hashes = {image_hashes[key] for key in test_ids}
    if train_hashes & validation_hashes or train_hashes & test_hashes or validation_hashes & test_hashes:
        raise ValueError("Cross-split identical image bytes")
    print("Tokenizing only registered training captions with the local ALBEF tokenizer.", flush=True)
    token_check = tokenize_training_captions(train)
    source_bindings = {
        str(source_pair_path): sha256(source_pair_path),
        str(registered_manifest_path): sha256(registered_manifest_path),
        str(source_pool_path): sha256(source_pool_path),
        str(validation_path): sha256(validation_path), str(test_path): sha256(test_path),
        str(fixed_validation_path): sha256(fixed_validation_path),
    }
    manifest = {"version": 1, "seed": SEED, "policy": POLICY,
                "task_protocol": "image_text_available_match_labels_limited",
                "image_root": str(SOURCE), "budgets": {}, "source_sha256": source_bindings,
                "public_schema": list(FIELDS), "train_pool_images": len(train),
                "caption_type": "blind", "caption_content": "complete_verbatim_original",
                "semantic_status": "random_other_case_unverified",
                "u_sampling": "flat_unlabeled_pair_list_without_source_balancing",
                "pair_id_policy": "sha256_of_shuffled_position_no_source_fields",
                "preparation_code_sha256": sha256(Path(__file__))}
    audit = {"passed": False, "scope": "prepare_and_structurally_verify_only_no_training",
             "seed": SEED, "policy": POLICY, "budgets": {}, "tokenizer": token_check,
             "splits": {"train": len(train_ids), "validation": len(validation_ids), "test": len(test_ids),
                        "fixed_validation": len(fixed_ids), "cross_split_id_overlap": 0,
                        "cross_split_image_hash_overlap": 0, "actual_image_files_hashed": len(all_rows),
                        "registered_train_and_fixed_validation_hashes_match": True},
             "caption_source": str(source_pair_path), "test_caption_text_accessed": False,
             "test_model_evaluation_performed": False, "original_source_files_modified": False,
             "source_sha256": source_bindings, "private_artifacts": {},
             "preparation_code_sha256": sha256(Path(__file__))}
    previous_l = set()
    for budget, count in COUNTS.items():
        l_path = AB_ROOT / f"data/pairs/train_{budget}.csv"
        l_rows = read_rows(l_path)
        l_ids = unique_ids(l_rows, count, f"L {budget}")
        if not previous_l <= l_ids or not l_ids <= train_ids:
            raise ValueError(f"Original nested labeled budgets invalid: {budget}")
        original_budget_ids = unique_ids(read_rows(SOURCE / f"budgets/train_{budget}.csv"), count, f"original L {budget}")
        if original_budget_ids != l_ids:
            raise ValueError(f"Original source budget and A/B budget differ: {budget}")
        previous_l = l_ids
        u_ids = train_ids - l_ids
        u_rows = [row for row in train if row["image_id"] in u_ids]
        if {image_hashes[key] for key in u_ids} & {image_hashes[key] for key in l_ids}:
            raise ValueError(f"L/U duplicate image hash overlap: {budget}")
        public, private, checks = build_budget(u_rows, budget)
        public_path = ROOT / f"data/u_pairs/u_{budget}.csv"
        private_path = ROOT / f"audit_private/u_{budget}_provenance.csv"
        public_data, private_data = csv_bytes(public, FIELDS), csv_bytes(private, PRIVATE_FIELDS)
        immutable_write(public_path, public_data, arguments.verify_only)
        immutable_write(private_path, private_data, arguments.verify_only)
        public_digest = hashlib.sha256(public_data).hexdigest()
        private_digest = hashlib.sha256(private_data).hexdigest()
        manifest["budgets"][budget] = {
            "path": public_path.relative_to(ROOT).as_posix(), "sha256": public_digest,
            "n_images": len(u_ids), "n_pairs": len(public), "l_count": count,
            "l_pairs_path": str(l_path), "l_pairs_sha256": sha256(l_path),
        }
        audit["private_artifacts"][budget] = {"path": private_path.relative_to(ROOT).as_posix(),
                                                 "sha256": private_digest}
        checks.update({"l_count": count, "u_is_exact_train_minus_l": True,
                       "l_u_id_overlap": 0, "l_u_image_hash_overlap": 0,
                       "validation_test_id_overlap": 0, "l_budget_nested": True,
                       "public_sha256": public_digest})
        audit["budgets"][budget] = checks
        print(f"Prepared {budget}: L={count}, U={len(u_ids)}, U pairs={len(public)}, sha256={public_digest}", flush=True)
    audit["passed"] = True
    audit["manifest_sha256"] = hashlib.sha256(json_bytes(manifest)).hexdigest()
    immutable_write(ROOT / "data/manifest.json", json_bytes(manifest), arguments.verify_only)
    immutable_write(ROOT / "audit/u_data_checks.json", json_bytes(audit), arguments.verify_only)
    print(json.dumps({"passed": True, "verify_only": arguments.verify_only,
                      "manifest": str(ROOT / "data/manifest.json"),
                      "max_tokens": token_check["max_tokens"],
                      "total_u_pairs_across_budgets": sum(value["n_pairs"] for value in manifest["budgets"].values())}), flush=True)


if __name__ == "__main__":
    main()
