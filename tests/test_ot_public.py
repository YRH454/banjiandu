"""Fixture-free CPU checks; no dataset, checkpoint, or training process needed."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OT_ROOT = ROOT / "experiments/apple_itm_usa_ot_ablation_v1"
sys.path.insert(0, str(OT_ROOT / "code"))
from ot_loss import OTConvergenceError, OTInputError, ot_weight, soft_ot_targets


class PublicOTNumericalChecks(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(20260825)
        self.anchors = torch.randn(32, 16, generator=self.generator)
        self.queries = torch.randn(32, 16, generator=self.generator)
        self.labels = torch.tensor([1.0, 0.0] * 16)

    def test_weight_schedule(self):
        self.assertEqual(ot_weight(100), 0.0)
        self.assertGreater(ot_weight(101), 0.0)
        self.assertAlmostEqual(ot_weight(200), 0.1)

    def test_balanced_finite_detached_targets(self):
        q, diagnostics = soft_ot_targets(self.anchors, self.labels, self.queries)
        self.assertEqual(tuple(q.shape), (32,))
        self.assertFalse(q.requires_grad)
        self.assertTrue(bool(torch.isfinite(q).all()))
        self.assertTrue(bool(((q >= 0) & (q <= 1)).all()))
        self.assertLessEqual(diagnostics["row_residual"], 1e-5)
        self.assertLessEqual(diagnostics["col_residual"], 1e-5)
        self.assertLessEqual(diagnostics["iterations"], 100)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(OTInputError):
            soft_ot_targets(torch.zeros_like(self.anchors), self.labels, self.queries)
        with self.assertRaises(OTInputError):
            soft_ot_targets(self.anchors, torch.full_like(self.labels, 0.5), self.queries)
        with self.assertRaises(OTConvergenceError):
            soft_ot_targets(self.anchors, self.labels, self.queries, max_iterations=1, tolerance=1e-12)

    def test_legacy_and_current_agree_on_fast_case(self):
        path = OT_ROOT / "archive/pre_recovery_solver/ot_loss.py"
        spec = importlib.util.spec_from_file_location("legacy_ot_public_check", path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        previous, old_diagnostics = legacy.soft_ot_targets(self.anchors, self.labels, self.queries)
        current, new_diagnostics = soft_ot_targets(self.anchors, self.labels, self.queries)
        torch.testing.assert_close(previous, current, rtol=0, atol=0)
        self.assertEqual(old_diagnostics["iterations"], new_diagnostics["iterations"])
        self.assertEqual(new_diagnostics["newton_steps"], 0)


if __name__ == "__main__":
    unittest.main()
