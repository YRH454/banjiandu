"""Hash-bound private inputs, shared image views and blind-only U loading."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

from .references import ROOT, model_components
from .budget import ComputeLedger
from .evaluation import validate_test_selection, validate_teacher_accounting, validate_teacher_history
from .spec import fingerprint, is_sha256, load_protocol, positive_int, validate_config

L_FIELDS = ["image_id", "image_relpath", "positive_text", "negative_text",
            "source_text_sha256", "negative_text_sha256", "image_sha256"]
U_FIELDS = ["pair_id", "image_id", "image_path", "text", "text_sha256"]


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def private_path(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Private contract paths must be relative to the new input root")
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Input path escapes the private root")
    return path


def check_file(root, entry):
    if set(entry) != {"path", "sha256"} or not is_sha256(entry["sha256"]):
        raise ValueError("Real file path and trusted SHA256 are required")
    path = private_path(root, entry["path"])
    if file_digest(path) != entry["sha256"]:
        raise ValueError("Private input bytes differ from their registered SHA256")
    return path


class BoundPairData:
    """No data copying, pseudo labels in U, historical owners or old queue imports."""
    BINDING_FORMAT = "fair_private_inputs_v3"
    HOLDOUT_FORMAT = "fair_holdout_registration_v3"
    TEACHER_FORMAT = "fair_pairusa_targets_v3"
    load_protocol = staticmethod(load_protocol)
    validate_config = staticmethod(validate_config)
    validate_test_selection = staticmethod(validate_test_selection)

    def __init__(self, root, config, binding, protocol=None):
        self.root = Path(root).resolve()
        self.p = self.load_protocol() if protocol is None else copy.deepcopy(protocol)
        self.cfg = config = self.validate_config(copy.deepcopy(config), self.p)
        binding = copy.deepcopy(binding)
        if binding.get("version") != self.BINDING_FORMAT or binding.get("protocol_sha256") != fingerprint(self.p) or binding.get("data_construction_seed") != self.p["data_construction_seed"]:
            raise ValueError("A new trusted fair-v3 input binding or matching newer version is required")
        if self.root.is_relative_to(ROOT):
            raise ValueError("Keep actual data/assets/teacher/output workspace outside the public repository")
        crop, budget = config["dataset"], config["budget"]
        manifest_entry = binding["datasets"][crop]
        path = check_file(self.root, manifest_entry)
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        expected_guards = dict(apple=256, cassava=384, rice=400, banana=384)
        if self.manifest["token_guard"] != expected_guards[crop]:
            raise ValueError("Crop's full-caption guard differs")
        asset_entry = self.manifest["asset_index"]
        self.assets = json.loads(check_file(self.root, asset_entry).read_text(encoding="utf-8"))
        for entry in self.assets.values():
            check_file(self.root, dict(path=entry["path"], sha256=entry["sha256"]))
        def rows(entry, fields):
            path = check_file(self.root, entry)
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != fields:
                    raise ValueError("Forbidden/missing columns; U must not contain labels or provenance")
                values = list(reader)
            for row in values:
                if set(row) != set(fields) or any(not isinstance(row[k], str) or not row[k].strip() for k in fields):
                    raise ValueError("Extra/missing CSV cells or empty required fields")
                meta = self.assets.get(row["image_id"])
                if meta is None:
                    raise ValueError("Unregistered image ID")
                image_field = "image_path" if fields == U_FIELDS else "image_relpath"
                if private_path(self.root, row[image_field]) != private_path(self.root, meta["path"]):
                    raise ValueError("CSV image path and registered asset disagree")
                if fields == L_FIELDS and row["image_sha256"] != meta["sha256"]:
                    raise ValueError("L image SHA256 differs")
                keys = (("text", "text_sha256"),) if fields == U_FIELDS else (
                    ("positive_text", "source_text_sha256"), ("negative_text", "negative_text_sha256"))
                for text, hash_key in keys:
                    if hashlib.sha256(row[text].encode()).hexdigest() != row[hash_key]:
                        raise ValueError("Full blind caption changed")
            return values
        self._read_rows = rows
        b = self.manifest["budgets"][budget]
        self.l = rows(b["l"], L_FIELDS)
        self.u = rows(b["u"], U_FIELDS) if config["budget_percent"] != 100 else []
        self.val = rows(self.manifest["validation"], L_FIELDS)
        l_ids, u_ids, v_ids = ({x["image_id"] for x in group} for group in (self.l, self.u, self.val))
        if len(l_ids) != len(self.l) or len(v_ids) != 400 or len(self.val) != 400 or l_ids & u_ids or (l_ids | u_ids) & v_ids:
            raise ValueError("L/U/Validation membership/count changed")
        l_hashes, u_hashes, v_hashes = ({self.assets[k]["sha256"] for k in ids} for ids in (l_ids, u_ids, v_ids))
        if len(l_hashes) != len(l_ids) or len(v_hashes) != len(v_ids) or l_hashes & u_hashes or (l_hashes | u_hashes) & v_hashes:
            raise ValueError("Image aliases or duplicate bytes leak across L/U/Validation")
        if len({r["pair_id"] for r in self.u}) != len(self.u) or len(self.l) < 16 or (config["budget_percent"] != 100 and len(self.u) < 32):
            raise ValueError("Duplicate pair IDs or insufficient logical batch")
        self.test_anchors = positive_int(self.manifest["test_anchors"], "trusted Test anchor count")
        # Before training inspect only the unlabelled membership index and file
        # digest. Test caption/label rows are not exposed to the training API.
        check_file(self.root, self.manifest["test"])
        self._test_index = json.loads(check_file(self.root, self.manifest["test_index"]).read_text(encoding="utf-8"))
        if not isinstance(self._test_index, list) or len(self._test_index) != self.test_anchors:
            raise ValueError("Test index and registered count differ")
        for row in self._test_index:
            if set(row) != {"image_id", "image_path", "image_sha256"}:
                raise ValueError("Test membership index must not contain labels or captions")
            meta = self.assets.get(row["image_id"])
            if meta is None or row["image_sha256"] != meta["sha256"] or private_path(self.root, row["image_path"]) != private_path(self.root, meta["path"]):
                raise ValueError("Test index asset differs")
        test_ids = {r["image_id"] for r in self._test_index}
        test_hashes = {r["image_sha256"] for r in self._test_index}
        if len(test_ids) != self.test_anchors or len(test_hashes) != self.test_anchors or test_ids & (l_ids | u_ids | v_ids) or test_hashes & (l_hashes | u_hashes | v_hashes):
            raise ValueError("Test overlaps training/Validation by ID or image bytes")
        registration_entry = binding["holdout_registration"]
        registration = json.loads(check_file(self.root, registration_entry).read_text(encoding="utf-8"))
        expected_holdout = dict(dataset_manifest_sha256=manifest_entry["sha256"], validation_sha256=self.manifest["validation"]["sha256"],
                                test_sha256=self.manifest["test"]["sha256"], test_index_sha256=self.manifest["test_index"]["sha256"], test_anchors=self.test_anchors)
        registered = registration["datasets"][crop]
        if registration.get("format") != self.HOLDOUT_FORMAT or registration.get("protocol_sha256") != fingerprint(self.p) or registration.get("registered_before_training") is not True or registration.get("test_previously_unused_for_training_or_selection") is not True or registered.get("origin") not in ("official_holdout", "new_preregistered_holdout") or {k: registered[k] for k in expected_holdout} != expected_holdout:
            raise ValueError("A trusted, previously unused preregistered holdout is required; do not relabel Validation")
        self.holdout_identity = {**expected_holdout, "registration_sha256": registration_entry["sha256"]}
        self.test_contract_sha256 = fingerprint(self.holdout_identity)
        self._test_opened = False
        self._non_test_ids = l_ids | u_ids | v_ids
        model_assets = binding["model_assets"]
        if not model_assets:
            raise ValueError("Official model/tokenizer asset SHA256 binding is required")
        for entry in model_assets:
            check_file(self.root, entry)
        self.checkpoint_path = str(self.root / "assets/ALBEF_4M.pth")
        if "assets/ALBEF_4M.pth" not in {e["path"] for e in model_assets}:
            raise ValueError("Official ALBEF asset is not registered")
        tokenizer_files = {str(x.relative_to(self.root)).replace("\\", "/") for x in (self.root / "assets/tokenizer").rglob("*") if x.is_file()}
        if not tokenizer_files or not tokenizer_files.issubset({e["path"] for e in model_assets}):
            raise ValueError("Every tokenizer file must be hash bound")
        self.input_identity = dict(dataset_manifest=manifest_entry["sha256"], budget=b,
                                   validation=self.manifest["validation"], assets=asset_entry,
                                   model_assets=model_assets, data_construction_seed=20260825,
                                   l_anchors=len(self.l), holdout=self.holdout_identity)
        self.binding = binding
        self.teacher_cost_seconds, self.teacher_targets, self.teacher_identity = 0., None, None
        self.teacher_accounting, self.teacher_resources = {}, None
        self._cache = None

    @staticmethod
    def flatten(rows):
        return [dict(image_id=row["image_id"], pair_id=row["image_id"]+":"+kind,
                     text=row[key], label=label)
                for row in rows for kind, key, label in (("pos", "positive_text", 1.), ("neg", "negative_text", 0.))]

    def attach(self, model, device):
        # The old cache is a pure component; bind its root in an isolated module.
        import torch
        common = types.ModuleType("common")
        common.ROOT, common.digest = self.root, file_digest
        common.read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
        old_common = sys.modules.get("common")
        sys.modules["common"] = common
        path = ROOT / "experiments/multicrop_itm_mt_fixmatch_v1/code/data_backend.py"
        spec = importlib.util.spec_from_file_location("_fair_bound_cache", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        finally:
            if old_common is None:
                sys.modules.pop("common", None)
            else:
                sys.modules["common"] = old_common
        from albef_ssl.model import get_tokenizer
        self._cache = module.PrefixCache(model, get_tokenizer(str(self.root / "assets/tokenizer")),
                                        self.assets, self.manifest["token_guard"], self.cfg["seed"], device)
        self._fixed_seed = module.fixed_seed
        self.pairusa_loss = model_components().pair_usa_loss
        if self.cfg["uses_pairusa"]:
            entry = self.binding["teachers"][f"{self.cfg['dataset']}_{self.cfg['budget']}_s{self.cfg['seed']}"]
            artifact = torch.load(check_file(self.root, entry), map_location="cpu", weights_only=True)
            expected = dict(dataset=self.cfg["dataset"], budget=self.cfg["budget"], seed=self.cfg["seed"],
                            l_sha256=self.manifest["budgets"][self.cfg["budget"]]["l"]["sha256"],
                            validation_sha256=self.manifest["validation"]["sha256"], protocol_sha256=fingerprint(self.p))
            if artifact.get("format") != self.TEACHER_FORMAT or artifact["identity"] != expected or artifact["image_ids"] != [r["image_id"] for r in self.l] or artifact["test_evaluated"] is not False or artifact["training_labels_source"] != "this_cell_L_only":
                raise ValueError("Teacher belongs to another seed/input/budget/protocol")
            targets = artifact["targets"]
            if tuple(targets.shape) != (len(self.l), 2, 256) or not bool(torch.isfinite(targets).all()):
                raise ValueError("Complete canonical/flip teacher vectors are required")
            seconds = artifact["full_upstream_seconds"]
            if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math_is_finite_positive(seconds):
                raise ValueError("Teacher computation must not be silently treated as free")
            self.teacher_targets, self.teacher_cost_seconds = targets, float(seconds)
            self.teacher_identity = entry["sha256"]
            accounting = artifact["accounting"]
            fields = {"teacher_upstream_seconds", "teacher_successful_updates", "teacher_validation_calls", "teacher_l_forward_pairs",
                      "teacher_backward_pairs", "teacher_validation_forward_pairs", "teacher_descriptor_pairs"}
            if set(accounting) != fields or accounting["teacher_upstream_seconds"] != seconds or accounting["teacher_successful_updates"] != artifact["training_updates"] or accounting["teacher_validation_calls"] != len(artifact["validation_history"]) or accounting["teacher_validation_forward_pairs"] != 800*len(artifact["validation_history"]) or accounting["teacher_descriptor_pairs"] != 4*len(self.l)+800:
                raise ValueError("Complete measured teacher accounting differs")
            validate_teacher_history(artifact["validation_history"], self.p)
            validate_teacher_accounting(accounting, len(self.l), self.p)
            ledger = ComputeLedger(); ledger.add(**accounting)
            if any(accounting[k] <= 0 for k in fields):
                raise ValueError("Teacher operations must not be declared free")
            resources = artifact["resources"]
            if set(resources) != {"trainable_parameters", "descriptor_parameters", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"} or any(type(v) is not int or v <= 0 for v in resources.values()):
                raise ValueError("Measured teacher parameters and CUDA peak memory are required")
            self.teacher_accounting, self.teacher_resources = copy.deepcopy(accounting), copy.deepcopy(resources)
            self._teacher_index = {row["image_id"]: i for i, row in enumerate(self.l)}

    def batch(self, pairs, step, view, physical):
        if not self._test_opened and any(p["image_id"] not in self._non_test_ids for p in pairs):
            raise ValueError("Test rows remain locked until terminal selection")
        return self._cache.batch(pairs, step, view, physical)

    def teacher_vectors(self, selected_l, step):
        if self.teacher_targets is None:
            raise ValueError("New same-cell teacher targets are required")
        indices = [self._teacher_index[row["image_id"]] for row in selected_l]
        flips = [self._fixed_seed(self.cfg["seed"], step, "flip/" + row["image_id"]) & 1 for row in selected_l]
        return self.teacher_targets[indices, flips]

    def load_test(self, selection):
        if not isinstance(selection, dict) or type(selection.get("step")) is not int:
            raise ValueError("Complete terminal selection is required before Test")
        self.validate_test_selection(selection, self.cfg, self.p, selection["step"], test_hash=self.test_contract_sha256)
        if self._test_opened:
            raise ValueError("Test was already opened for this bound session")
        self._test_opened = True  # Failed access is not a free second selection opportunity.
        rows = self._read_rows(self.manifest["test"], L_FIELDS)
        actual = [dict(image_id=r["image_id"], image_path=r["image_relpath"], image_sha256=r["image_sha256"]) for r in rows]
        if actual != self._test_index:
            raise ValueError("Test caption rows differ from the preregistered unlabelled membership index")
        return rows


def math_is_finite_positive(value):
    import math
    return math.isfinite(value) and value > 0
