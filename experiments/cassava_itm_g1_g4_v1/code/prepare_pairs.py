"""Freeze cassava L/validation random negatives and label-hidden U tables.

The source is read-only. Generated tables are immutable; a nonidentical rerun
fails. Only blind captions from Train/Validation are copied into output tables.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path

from transformers import BertTokenizerFast


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"X:\PATH\datasets\EGOISTyrh\mushubanjiandu")
TOKENIZER = Path(r"X:\PATH\huggingface-cache\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594")
SEED = 20260825
MAX_TOKENS = 384
BUDGETS = ("001", "005", "010", "020", "030", "100")
RATES = (0.01, 0.05, 0.10, 0.20, 0.30, 1.0)
CLASSES = ("CBB", "CBSD", "CGM", "CMD", "Healthy")
PAIR_FIELDS = ("image_id", "image_relpath", "positive_text", "negative_text",
               "negative_source_image_id", "negative_source_image_relpath", "edit_type",
               "review_status", "rule_id", "source_text_sha256", "negative_text_sha256",
               "image_sha256", "negative_source_image_sha256", "first_visible_budget",
               "donor_cohort", "candidate_reason", "class_code", "subject")
U_FIELDS = ("pair_id", "image_id", "image_path", "text", "text_sha256")
PRIVATE_FIELDS = ("pair_id", "image_id", "caption_source_image_id", "construction_source_label",
                  "image_sha256", "caption_source_image_sha256", "text_sha256")
POLICY = "random_other_case_full_blind_caption_cassava_v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(tag: str) -> int:
    return int(sha(f"{SEED}\0{tag}".encode())[:16], 16)


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Registered artifact differs; refusing overwrite: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def csv_bytes(rows: list[dict], fields: tuple[str, ...]) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def read_ids(name: str) -> list[str]:
    return (ROOT / "data" / "splits" / f"{name}_ids.txt").read_text(encoding="utf-8").splitlines()


def allocation(sizes: dict[str, int], target: int, rate: float) -> dict[str, int]:
    counts = {name: math.floor(sizes[name] * rate) for name in CLASSES}
    order = sorted(CLASSES, key=lambda name: (-(sizes[name] * rate - counts[name]), name))
    for name in order[:target - sum(counts.values())]:
        counts[name] += 1
    assert sum(counts.values()) == target
    return counts


def derangement(rows: list[dict], tag: str, text_key: str = "blind") -> tuple[list[int], int]:
    rng = random.Random(stable_seed("derangement/" + tag))
    n = len(rows)
    if n < 2:
        raise RuntimeError(f"Need two distinct cases for {tag}")
    for attempt in range(1, 10001):
        order = list(range(n))
        rng.shuffle(order)
        if all(i != j and rows[i][text_key] != rows[j][text_key]
               and rows[i]["sha256"] != rows[j]["sha256"] for i, j in enumerate(order)):
            return order, attempt
    raise RuntimeError(f"No valid full-caption derangement for {tag}")


def make_l_cohort(rows: list[dict], tag: str) -> list[dict]:
    rows = sorted(rows, key=lambda row: row["image_id"])
    order, _ = derangement(rows, "l/" + tag)
    result = []
    for i, j in enumerate(order):
        anchor, donor = rows[i], rows[j]
        result.append({"image_id": anchor["image_id"], "image_relpath": anchor["relpath"],
                       "positive_text": anchor["blind"], "negative_text": donor["blind"],
                       "negative_source_image_id": donor["image_id"],
                       "negative_source_image_relpath": donor["relpath"],
                       "edit_type": "random_other_case_full_caption",
                       "review_status": "random_other_case_unverified", "rule_id": POLICY,
                       "source_text_sha256": anchor["blind_sha256"],
                       "negative_text_sha256": donor["blind_sha256"],
                       "image_sha256": anchor["sha256"],
                       "negative_source_image_sha256": donor["sha256"],
                       "first_visible_budget": tag, "donor_cohort": tag,
                       "candidate_reason": "fixed other-case identity mismatch; semantic correctness unverified",
                       "class_code": anchor["class_code"], "subject": anchor["subject"]})
    return result


def check_l(rows: list[dict], expected: set[str]) -> None:
    by_id = {row["image_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise RuntimeError("L/validation ID coverage mismatch")
    if collections.Counter(row["positive_text"] for row in rows) != collections.Counter(row["negative_text"] for row in rows):
        raise RuntimeError("Text marginal differs between positive and negative labels")
    for row in rows:
        donor = by_id[row["negative_source_image_id"]]
        if (donor["image_id"] == row["image_id"] or donor["positive_text"] != row["negative_text"]
                or donor["image_sha256"] == row["image_sha256"]):
            raise RuntimeError("Invalid L random negative")
        if sha(row["positive_text"].encode()) != row["source_text_sha256"] or sha(row["negative_text"].encode()) != row["negative_text_sha256"]:
            raise RuntimeError("L caption source hash mismatch")


def make_u(rows: list[dict], tag: str) -> tuple[list[dict], list[dict]]:
    rows = sorted(rows, key=lambda row: row["image_id"])
    order, _ = derangement(rows, "u/" + tag)
    construction = [(i, i, 1) for i in range(len(rows))] + [(i, j, 0) for i, j in enumerate(order)]
    random.Random(stable_seed("u-flat/" + tag)).shuffle(construction)
    public, private = [], []
    for position, (i, j, label) in enumerate(construction):
        anchor, donor = rows[i], rows[j]
        pair_id = sha(f"{SEED}|{POLICY}|{tag}|{position}".encode())[:32]
        public.append({"pair_id": pair_id, "image_id": anchor["image_id"],
                       "image_path": anchor["relpath"], "text": donor["blind"],
                       "text_sha256": donor["blind_sha256"]})
        private.append({"pair_id": pair_id, "image_id": anchor["image_id"],
                        "caption_source_image_id": donor["image_id"],
                        "construction_source_label": str(label),
                        "image_sha256": anchor["sha256"],
                        "caption_source_image_sha256": donor["sha256"],
                        "text_sha256": donor["blind_sha256"]})
    if (collections.Counter(row["image_id"] for row in public)
            != collections.Counter({row["image_id"]: 2 for row in rows})):
        raise RuntimeError("U does not have exactly two pairs per image")
    if (collections.Counter(row["text_sha256"] for row in public)
            != collections.Counter({row["blind_sha256"]: 2 for row in rows})):
        raise RuntimeError("U caption marginal differs from twice its source")
    return public, private


def main() -> None:
    snapshot = json.loads((ROOT / "data/source_snapshot.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "audit/source_audit.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((ROOT / "data/splits/split_manifest.json").read_text(encoding="utf-8"))
    if (not audit["source_integrity_passed"] or snapshot["source_revision"] != audit["source_revision"]
            or audit["manifest_sha256"] != split_manifest["manifest_sha256"]
            or audit["captions_sha256"] != split_manifest["captions_sha256"]):
        raise RuntimeError("Source/split registration mismatch")
    index = {row["image_id"]: row for line in (ROOT / "data/image_index.jsonl").read_text(encoding="utf-8").splitlines()
             if line for row in [json.loads(line)]}
    train_ids, val_ids, test_ids = (set(read_ids(name)) for name in ("train", "validation", "test"))
    if (len(train_ids), len(val_ids), len(test_ids)) != (8583, 1073, 1073):
        raise RuntimeError("Frozen split counts changed")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids or set(index) != train_ids | val_ids | test_ids:
        raise RuntimeError("Frozen split ID boundary changed")
    for name, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids)):
        rows = [{"image_id": image_id} for image_id in sorted(ids)]
        write_once(ROOT / "data" / ("train_pool_ids.csv" if name == "train" else f"{name}.csv"), csv_bytes(rows, ("image_id",)))
    captions = json.loads((SOURCE / "captions.json").read_text(encoding="utf-8"))
    # Never emit Test caption text; remove the mapping before constructing tables.
    for image_id in test_ids:
        captions.pop(image_id, None)
    tokenizer = BertTokenizerFast.from_pretrained(str(TOKENIZER), local_files_only=True)
    if len(tokenizer) != 30522 or not tokenizer.do_lower_case:
        raise RuntimeError("ALBEF tokenizer identity mismatch")
    values = []
    for image_id in sorted(train_ids | val_ids):
        row, caption = dict(index[image_id]), captions[image_id]
        text = caption["blind"]
        if sha(text.encode()) != row["blind_sha256"] or caption["relpath"] != row["relpath"]:
            raise RuntimeError(f"Blind caption/index mismatch: {image_id}")
        row["blind"] = text
        values.append(row)
    lengths = []
    for start in range(0, len(values), 128):
        lengths.extend(map(len, tokenizer([row["blind"] for row in values[start:start + 128]],
                                          add_special_tokens=True, truncation=False)["input_ids"]))
    if max(lengths) > MAX_TOKENS:
        raise RuntimeError(f"Full blind caption exceeds approved {MAX_TOKENS} tokens")
    row_by_id = {row["image_id"]: row for row in values}
    grouped = {name: sorted([row for row in values if row["image_id"] in train_ids and row["class_code"] == name],
                            key=lambda row: row["image_id"]) for name in CLASSES}
    sizes = {name: len(grouped[name]) for name in CLASSES}
    for name in CLASSES:
        random.Random(stable_seed("l-class/" + name)).shuffle(grouped[name])
    manifest = {"version": 1, "seed": SEED, "max_text_tokens": MAX_TOKENS, "image_root": str(SOURCE),
                "negative_policy": POLICY, "files": {"budgets": {}, "validation": "data/pairs/validation.csv"},
                "source_counts": {}, "validation_anchors": 400,
                "caption_status": "verbatim blind; random-other-case semantic status unverified",
                "source_revision": snapshot["source_revision"], "split_manifest_sha256": file_sha(ROOT / "data/splits/split_manifest.json")}
    gate_hashes = {}
    u_manifest = {"version": 1, "seed": SEED, "policy": POLICY, "budgets": {},
                  "caption_type": "blind", "caption_content": "complete_verbatim_original",
                  "max_text_tokens": MAX_TOKENS, "public_schema": list(U_FIELDS),
                  "source_pair_manifest_sha256": None}
    previous_ids: set[str] = set()
    paired_by_id: dict[str, dict] = {}
    cohort_counts = {}
    for tag, rate in zip(BUDGETS, RATES):
        target = math.floor(len(train_ids) * rate)
        class_counts = allocation(sizes, target, rate)
        ids = {row["image_id"] for name in CLASSES for row in grouped[name][:class_counts[name]]}
        if len(ids) != target or not previous_ids <= ids:
            raise RuntimeError("L budget is not exactly nested")
        cohort = [row_by_id[image_id] for image_id in ids - previous_ids]
        new_pairs = make_l_cohort(cohort, tag)
        paired_by_id.update({row["image_id"]: row for row in new_pairs})
        l_rows = [paired_by_id[image_id] for image_id in sorted(ids)]
        check_l(l_rows, ids)
        relative = f"data/pairs/train_{tag}.csv"
        data = csv_bytes(l_rows, PAIR_FIELDS)
        write_once(ROOT / relative, data)
        gate_hashes[relative] = sha(data)
        manifest["files"]["budgets"][tag] = relative
        manifest["source_counts"][tag] = target
        write_once(ROOT / f"data/budgets/train_{tag}.csv",
                   csv_bytes([{"image_id": image_id} for image_id in sorted(ids)], ("image_id",)))
        cohort_counts[tag] = {"l": target, "new_cohort": len(cohort), "class_counts": class_counts}
        if tag != "100":
            u_ids = train_ids - ids
            u_rows, private = make_u([row_by_id[image_id] for image_id in u_ids], tag)
            u_relative = f"data/u_pairs/u_{tag}.csv"
            u_data, private_data = csv_bytes(u_rows, U_FIELDS), csv_bytes(private, PRIVATE_FIELDS)
            write_once(ROOT / u_relative, u_data)
            write_once(ROOT / f"audit_private/u_{tag}_provenance.csv", private_data)
            u_manifest["budgets"][tag] = {"path": u_relative, "sha256": sha(u_data),
                                          "n_images": len(u_ids), "n_pairs": len(u_rows),
                                          "l_count": target, "l_pairs_path": str(ROOT / relative),
                                          "l_pairs_sha256": sha(data), "private_sha256": sha(private_data)}
        previous_ids = ids
        print(json.dumps({"budget": tag, "L": target, "U": len(train_ids - ids)}, ensure_ascii=False), flush=True)
    if previous_ids != train_ids:
        raise RuntimeError("Full budget does not cover Train")
    selected = random.Random(stable_seed("validation-400-selection")).sample(sorted(val_ids), 400)
    val_rows = make_l_cohort([row_by_id[image_id] for image_id in selected], "validation400")
    check_l(val_rows, set(selected))
    val_data = csv_bytes(sorted(val_rows, key=lambda row: row["image_id"]), PAIR_FIELDS)
    write_once(ROOT / "data/pairs/validation.csv", val_data)
    gate_hashes["data/pairs/validation.csv"] = sha(val_data)
    u_manifest["source_pair_manifest_sha256"] = sha(json_bytes(manifest))
    write_once(ROOT / "data/manifest.json", json_bytes(manifest))
    write_once(ROOT / "data/u_manifest.json", json_bytes(u_manifest))
    write_once(ROOT / "audit/negative_quality_gate.json", json_bytes({
        "ready_for_training": True, "negative_policy": POLICY,
        "admitted_file_sha256": gate_hashes, "semantic_visual_approval": False,
        "semantic_false_negatives_possible": True, "formal_training_started": False,
        "scope": "structural pairing and source checks; no semantic gold standard"}))
    write_once(ROOT / "audit/pair_preparation.json", json_bytes({
        "passed": True, "source_revision": snapshot["source_revision"],
        "split_manifest_sha256": file_sha(ROOT / "data/splits/split_manifest.json"),
        "manifest_sha256": file_sha(ROOT / "data/manifest.json"),
        "u_manifest_sha256": file_sha(ROOT / "data/u_manifest.json"),
        "train_counts": cohort_counts, "validation_anchors": 400,
        "max_tokens_observed_train_and_validation": max(lengths),
        "over_384": sum(length > MAX_TOKENS for length in lengths),
        "full_blind_captions_unchanged": True, "test_captions_emitted": False,
        "random_semantic_status": "random_other_case_unverified"}))
    print(json.dumps({"prepared": True, "L_budgets": cohort_counts,
                      "validation": 400, "max_tokens": max(lengths)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
