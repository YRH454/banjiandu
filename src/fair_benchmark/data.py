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
from .spec import fingerprint, is_sha256, load_protocol, validate_config

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
    def __init__(self, root, config, binding, protocol=None):
        self.root = Path(root).resolve()
        self.p = load_protocol() if protocol is None else copy.deepcopy(protocol)
        self.cfg = config = validate_config(copy.deepcopy(config), self.p)
        binding = copy.deepcopy(binding)
        if binding.get("version") != "fair_private_inputs_v2" or binding.get("protocol_sha256") != fingerprint(self.p) or binding.get("data_construction_seed") != self.p["data_construction_seed"]:
            raise ValueError("A new trusted fair-v2 input binding is required")
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
                                   model_assets=model_assets, data_construction_seed=20260825)
        self.binding = binding
        self.teacher_cost_seconds, self.teacher_targets, self.teacher_identity = 0., None, None
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
            if artifact.get("format") != "fair_pairusa_targets_v2" or artifact["identity"] != expected or artifact["image_ids"] != [r["image_id"] for r in self.l]:
                raise ValueError("Teacher belongs to another seed/input/budget/protocol")
            targets = artifact["targets"]
            if tuple(targets.shape) != (len(self.l), 2, 256) or not bool(torch.isfinite(targets).all()):
                raise ValueError("Complete canonical/flip teacher vectors are required")
            seconds = artifact["full_upstream_seconds"]
            if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math_is_finite_positive(seconds):
                raise ValueError("Teacher computation must not be silently treated as free")
            self.teacher_targets, self.teacher_cost_seconds = targets, float(seconds)
            self.teacher_identity = entry["sha256"]
            self._teacher_index = {row["image_id"]: i for i, row in enumerate(self.l)}

    def batch(self, pairs, step, view, physical):
        return self._cache.batch(pairs, step, view, physical)

    def teacher_vectors(self, selected_l, step):
        if self.teacher_targets is None:
            raise ValueError("New same-cell teacher targets are required")
        indices = [self._teacher_index[row["image_id"]] for row in selected_l]
        flips = [self._fixed_seed(self.cfg["seed"], step, "flip/" + row["image_id"]) & 1 for row in selected_l]
        return self.teacher_targets[indices, flips]


def math_is_finite_positive(value):
    import math
    return math.isfinite(value) and value > 0
