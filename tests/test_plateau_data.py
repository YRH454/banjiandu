"""Temporary synthetic input/holdout/teacher contracts, no real assets or GPU."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from test_fair_data import SyntheticContract
from test_plateau_benchmark import reach
from fair_benchmark.data import BoundPairData
from fair_benchmark.spec import fingerprint
from plateau_benchmark.budget import PhaseBudget
from plateau_benchmark.data import PlateauPairData
from plateau_benchmark.spec import load_protocol, make_config

try:
    import torch
except ImportError:
    torch = None


def upgrade(fixture, method="bce", stage="bce", budget=10, seed=20260825):
    fixture.cfg = make_config("banana", budget, method, seed, stage)
    fixture.publish()
    fixture.binding.update(version="fair_private_inputs_v4", protocol_sha256=fingerprint(load_protocol()))
    entry = fixture.binding["holdout_registration"]
    registration = json.loads((fixture.root/entry["path"]).read_text(encoding="utf-8"))
    registration.update(format="fair_holdout_registration_v4", protocol_sha256=fingerprint(load_protocol()))
    fixture.binding["holdout_registration"] = fixture.entry(entry["path"], json.dumps(registration).encode())
    return fixture


class PlateauPrivateInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="plateau_synthetic_only_")
        self.addCleanup(self.temp.cleanup)
        self.fixture = upgrade(SyntheticContract(Path(self.temp.name)))

    def bind(self):
        f = self.fixture
        return PlateauPairData(f.root, f.cfg, f.binding)

    def test_new_binding_same_data_three_seeds_and_phase_clock(self):
        first = self.bind()
        other_cfg = make_config("banana", 10, "ot", 20260826)
        other = PlateauPairData(self.fixture.root, other_cfg, self.fixture.binding)
        self.assertEqual(first.input_identity, other.input_identity)
        self.assertEqual(first._view_step(101), 101)
        self.assertEqual(other._view_step(101), 6101)
        with patch.object(BoundPairData, "batch", return_value="synthetic") as batch:
            other.batch([], 101, "weak", 16)
        self.assertEqual(batch.call_args.args[1], 6101)
        with patch.object(BoundPairData, "teacher_vectors", return_value="synthetic") as vectors:
            other.teacher_vectors([], 101)
        self.assertEqual(vectors.call_args.args[1], 6101)

    def test_v3_binding_and_holdout_contract_cannot_enter_v4(self):
        f = self.fixture
        changed = copy.deepcopy(f.binding); changed["version"] = "fair_private_inputs_v3"
        with self.assertRaises(ValueError):
            PlateauPairData(f.root, f.cfg, changed)
        entry = f.binding["holdout_registration"]
        reg = json.loads((f.root/entry["path"]).read_text(encoding="utf-8")); reg["format"] = "fair_holdout_registration_v3"
        f.binding["holdout_registration"] = f.entry(entry["path"], json.dumps(reg).encode())
        with self.assertRaisesRegex(ValueError, "previously unused"):
            self.bind()

    def test_test_locked_until_plateau_and_one_open_only(self):
        data = self.bind()
        with self.assertRaisesRegex(ValueError, "remain locked"):
            data.batch(data.flatten(self.fixture.test), 1, "weak", 16)
        budget = reach(PhaseBudget(data.cfg, len(data.l)))
        selection = dict(format="plateau_test_selection_v4", config_sha256=fingerprint(data.cfg), protocol_sha256=fingerprint(data.p),
                         checkpoint="terminal_ema", model="ema", step=budget.step, threshold=.5,
                         validation_metrics_sha256="6"*64, ema_state_sha256="7"*64,
                         test_contract_sha256=data.test_contract_sha256, budget_state=budget.state_dict(),
                         parent_receipt_sha256=None, adaptation_common_state_sha256=None)
        self.assertEqual(data.load_test(selection), self.fixture.test)
        with self.assertRaisesRegex(ValueError, "already opened"):
            data.load_test(selection)

    def test_label_or_image_alias_leak_rejected_and_100_no_u(self):
        f = self.fixture
        f.test = copy.deepcopy(f.val[:16]); upgrade(f)
        with self.assertRaisesRegex(ValueError, "Test overlaps"):
            self.bind()
        f = upgrade(SyntheticContract(Path(self.temp.name)), budget=100)
        self.assertEqual(PlateauPairData(f.root, f.cfg, f.binding).u, [])

    @unittest.skipIf(torch is None, "PyTorch absent")
    def test_teacher_artifact_uses_v4_identity_complete_cost_and_true_history_rules(self):
        f = upgrade(self.fixture, method="pairusa", stage="adaptation")
        entry = f.entry("synthetic_teacher.not_a_weight_file", b"synthetic bytes, mocked torch.load")
        f.binding["teachers"] = {"banana_010_s20260825": entry}
        l, epochs = len(f.l), 4
        # These mock values only exercise the artifact parser. They are not
        # measured teacher/GPU parameters and are never scientific results.
        accounting = dict(teacher_upstream_seconds=2., teacher_successful_updates=epochs,
                          teacher_validation_calls=epochs, teacher_l_forward_pairs=2*l*epochs+4*l,
                          teacher_backward_pairs=2*l*epochs, teacher_validation_forward_pairs=800*epochs,
                          teacher_descriptor_pairs=4*l+800)
        artifact = dict(format="fair_pairusa_targets_v4", identity=dict(dataset="banana", budget="010", seed=20260825,
                        l_sha256=f.manifest["budgets"]["010"]["l"]["sha256"], validation_sha256=f.manifest["validation"]["sha256"],
                        protocol_sha256=fingerprint(load_protocol())), image_ids=[r["image_id"] for r in f.l],
                        targets=torch.zeros(l, 2, 256), test_evaluated=False, training_labels_source="this_cell_L_only",
                        full_upstream_seconds=2., accounting=accounting, training_updates=epochs,
                        validation_history=[dict(epoch=i, validation_auroc=.8) for i in range(1, epochs+1)],
                        resources=dict(trainable_parameters=1, descriptor_parameters=1, peak_cuda_allocated_bytes=1, peak_cuda_reserved_bytes=1))
        transforms = types.ModuleType("torchvision.transforms"); transforms.RandAugment = lambda **_: None
        model_module = types.ModuleType("albef_ssl.model"); model_module.get_tokenizer = lambda _: None
        def attach(value):
            data = self.bind()
            with patch.dict(sys.modules, {"torchvision.transforms": transforms, "albef_ssl.model": model_module}), patch("torch.load", return_value=value), patch("fair_benchmark.data.model_components", return_value=types.SimpleNamespace(pair_usa_loss=None)):
                data.attach(None, "cpu")
            return data
        data = attach(artifact)
        self.assertEqual(data.teacher_identity, entry["sha256"])
        self.assertEqual(data.teacher_accounting, accounting)
        self.assertEqual(tuple(data.teacher_targets.shape), (16, 2, 256))
        for field, value in (("format", "fair_pairusa_targets_v3"), ("training_labels_source", "hidden_U_labels"), ("test_evaluated", True)):
            changed = copy.deepcopy(artifact); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                attach(changed)
        changed = copy.deepcopy(artifact); changed["accounting"]["teacher_descriptor_pairs"] = 0
        with self.assertRaises(ValueError):
            attach(changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
