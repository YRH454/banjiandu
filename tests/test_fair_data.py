"""Temporary synthetic contracts only; no real samples, assets, Torch or GPU."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fair_benchmark.data import BoundPairData, L_FIELDS, U_FIELDS, check_file, private_path
from fair_benchmark.spec import fingerprint, load_protocol, make_config


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


class SyntheticContract:
    """Intentionally non-image/model bytes: constructor validation, never training."""
    def __init__(self, root, budget=10):
        self.root = root
        self.cfg = make_config("banana", budget, "bce", 20260825)
        self.assets = {}
        self.l = self.labelled("synthetic_l", 16)
        self.val = self.labelled("synthetic_v", 400)
        self.u = []
        if budget != 100:
            for i in range(32):
                image_id = f"synthetic_u_{i}"
                meta = self.image(image_id)
                text = f"synthetic blind caption {i}"
                self.u.append(dict(pair_id=f"synthetic_pair_{i}", image_id=image_id,
                                   image_path=meta["path"], text=text, text_sha256=digest_bytes(text.encode())))
        model = self.entry("assets/ALBEF_4M.pth", b"synthetic invalid model, never loaded")
        tokenizer = self.entry("assets/tokenizer/vocab.txt", b"synthetic invalid tokenizer, never loaded")
        self.binding = dict(version="fair_private_inputs_v2", protocol_sha256=fingerprint(load_protocol()),
                            data_construction_seed=20260825, datasets={}, model_assets=[model, tokenizer])
        self.publish()

    def entry(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return dict(path=name, sha256=digest_bytes(value))

    def image(self, image_id):
        meta = self.entry(f"images/{image_id}.not_an_image", ("synthetic bytes " + image_id).encode())
        self.assets[image_id] = meta
        return meta

    def labelled(self, prefix, count):
        rows = []
        for i in range(count):
            image_id = f"{prefix}_{i}"
            meta = self.image(image_id)
            pos, neg = f"synthetic positive {image_id}", f"synthetic negative {image_id}"
            rows.append(dict(image_id=image_id, image_relpath=meta["path"], positive_text=pos, negative_text=neg,
                             source_text_sha256=digest_bytes(pos.encode()), negative_text_sha256=digest_bytes(neg.encode()),
                             image_sha256=meta["sha256"]))
        return rows

    def csv_entry(self, name, rows, fields):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        return dict(path=name, sha256=digest_bytes(path.read_bytes()))

    def publish(self, u_fields=U_FIELDS):
        l = self.csv_entry("synthetic_l.csv", self.l, L_FIELDS)
        val = self.csv_entry("synthetic_validation.csv", self.val, L_FIELDS)
        budget = {"l": l}
        if self.cfg["budget_percent"] != 100:
            budget["u"] = self.csv_entry("synthetic_u.csv", self.u, u_fields)
        assets = self.entry("synthetic_asset_index.json", json.dumps(self.assets).encode())
        self.manifest = dict(token_guard=384, asset_index=assets,
                             budgets={self.cfg["budget"]: budget}, validation=val)
        self.binding["datasets"]["banana"] = self.entry("synthetic_manifest.json", json.dumps(self.manifest).encode())

    def bind(self):
        return BoundPairData(self.root, self.cfg, self.binding)


class FairPrivateContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fair_synthetic_contract_")
        self.addCleanup(self.temp.cleanup)
        self.fixture = SyntheticContract(Path(self.temp.name))

    def test_valid_binding_is_shared_across_training_seeds(self):
        first = self.fixture.bind()
        cfg = copy.deepcopy(self.fixture.cfg)
        cfg = make_config(cfg["dataset"], cfg["budget_percent"], cfg["method"], 20260826)
        second = BoundPairData(self.fixture.root, cfg, self.fixture.binding)
        self.assertEqual(first.input_identity, second.input_identity)
        self.assertEqual((len(first.l), len(first.u), len(first.val)), (16, 32, 400))
        self.assertEqual(len(first.flatten(first.l)), 32)

    def test_paths_and_sha_cannot_escape_or_change(self):
        for name in ("../outside", str(self.fixture.root.resolve()), ""):
            with self.assertRaises(ValueError):
                private_path(self.fixture.root, name)
        entry = self.fixture.binding["datasets"]["banana"]
        changed = copy.deepcopy(entry); changed["sha256"] = "g" * 64
        with self.assertRaises(ValueError):
            check_file(self.fixture.root, changed)
        (self.fixture.root / entry["path"]).write_bytes(b"changed synthetic bytes")
        with self.assertRaisesRegex(ValueError, "registered SHA256"):
            self.fixture.bind()

    def test_forbidden_hidden_u_column_rejected(self):
        for row in self.fixture.u:
            row["hidden_label"] = "synthetic forbidden value"
        self.fixture.publish(u_fields=[*U_FIELDS, "hidden_label"])
        with self.assertRaisesRegex(ValueError, "Forbidden/missing columns"):
            self.fixture.bind()

    def test_extra_unheaded_or_missing_csv_cells_rejected(self):
        path = self.fixture.root / "synthetic_u.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        original = copy.deepcopy(rows)
        for change in ("extra", "missing"):
            rows = copy.deepcopy(original)
            if change == "extra":
                rows[1].append("synthetic forbidden cell")
            else:
                rows[1].pop()
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows(rows)
            self.fixture.manifest["budgets"]["010"]["u"]["sha256"] = digest_bytes(path.read_bytes())
            self.fixture.binding["datasets"]["banana"] = self.fixture.entry(
                "synthetic_manifest.json", json.dumps(self.fixture.manifest).encode())
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "CSV cells"):
                self.fixture.bind()

    def test_duplicate_ids_or_missing_u_logical_batch_rejected(self):
        original = copy.deepcopy(self.fixture.u)
        self.fixture.u[1]["pair_id"] = self.fixture.u[0]["pair_id"]
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "Duplicate pair IDs"):
            self.fixture.bind()
        self.fixture.u = original[:31]
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "insufficient logical batch"):
            self.fixture.bind()
        self.fixture.u = []
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "insufficient logical batch"):
            self.fixture.bind()

    def test_image_aliases_cannot_leak_into_u(self):
        row = self.fixture.u[0]
        meta = self.fixture.assets[self.fixture.l[0]["image_id"]]
        self.fixture.assets[row["image_id"]] = meta
        row["image_path"] = meta["path"]
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "Image aliases"):
            self.fixture.bind()

    def test_caption_asset_and_tokenizer_changes_rejected(self):
        self.fixture.l[0]["positive_text"] = "different synthetic caption"
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "blind caption changed"):
            self.fixture.bind()
        self.fixture.l[0]["source_text_sha256"] = digest_bytes(self.fixture.l[0]["positive_text"].encode())
        self.fixture.publish()
        path = self.fixture.root / self.fixture.assets[self.fixture.l[0]["image_id"]]["path"]
        original = path.read_bytes(); path.write_bytes(b"modified synthetic image")
        with self.assertRaisesRegex(ValueError, "registered SHA256"):
            self.fixture.bind()
        path.write_bytes(original)
        (self.fixture.root / "assets/tokenizer/unbound.txt").write_bytes(b"synthetic extra file")
        with self.assertRaisesRegex(ValueError, "tokenizer file"):
            self.fixture.bind()

    def test_full_label_binding_has_no_u_dependency(self):
        self.fixture.cfg = make_config("banana", 100, "bce", 20260825)
        self.fixture.publish()
        data = self.fixture.bind()
        self.assertEqual(data.u, [])
        self.assertNotIn("u", data.input_identity["budget"])

    def test_registered_config_data_seed_and_public_root_enforced(self):
        changed = copy.deepcopy(self.fixture.cfg); changed["target_steps"] = 4800
        with self.assertRaises(ValueError):
            BoundPairData(self.fixture.root, changed, self.fixture.binding)
        changed = copy.deepcopy(self.fixture.binding); changed["data_construction_seed"] = 20260826
        with self.assertRaisesRegex(ValueError, "input binding"):
            BoundPairData(self.fixture.root, self.fixture.cfg, changed)
        with self.assertRaisesRegex(ValueError, "outside the public repository"):
            BoundPairData(ROOT, self.fixture.cfg, self.fixture.binding)


if __name__ == "__main__":
    unittest.main(verbosity=2)
