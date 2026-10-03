"""Public CPU checks using synthetic inputs only; never run a training entrypoint."""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/multicrop_itm_soft_simmatch_v1"


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = module_at("public_multicrop_metrics", EXP / "code/metrics.py")
renderer = module_at("public_report_renderer", ROOT / "tools/render_result_report.py")
archive = module_at("public_archive_verifier", ROOT / "tools/verify_source_archive.py")


def synthetic_summary():
    """Not measured experiment data: deliberately identical chance-level fixtures."""
    base = {
        "n": 800, "threshold": 0.5, "accuracy": 0.5, "precision": 0.5,
        "recall": 0.5, "negative_recall": 0.5, "f1": 0.5, "macro_f1": 0.5,
        "balanced_accuracy": 0.5, "auroc": 0.5, "average_precision": 0.5,
        "brier": 0.25, "ece_10": 0.0,
        "confusion_matrix": {"tn": 200, "fp": 200, "fn": 200, "tp": 200},
    }
    results = {}
    for dataset in renderer.DATASETS:
        for budget in renderer.BUDGETS:
            for method in renderer.METHODS:
                run = f"{dataset}_{budget:03d}_{method}_s20260825"
                target = 2200 if method == "softmatch" else 2400
                m = {**base, "threshold_0_5": dict(base), "step": 100,
                     "evaluation_model": "ema", "validation_anchors": 400,
                     "paired_accuracy": 0.5}
                results[run] = {"run_id": run, "state": "completed",
                    "successful_steps": target, "test_evaluated": False,
                    "best_step": 100, "best_metrics": m,
                    "wall_seconds_including_validation_and_checkpointing": 1000,
                    "training_step_seconds": 900,
                    "provenance": {"private": "RAW_PRIVATE_SENTINEL"}}
    return {"configurations": 40, "single_seed": 20260825, "no_Test": True,
            "equal_compute_claim": False, "results": results,
            "completed_utc": "2026-01-01T00:00:00+00:00"}


class PublicSourceTests(unittest.TestCase):
    def test_source_manifest(self):
        self.assertEqual(archive.verify(), 33)

    def test_archived_equations_and_cache_cpu(self):
        for entry in (EXP / "code/unit_tests.py",
                      EXP / "archive/speedup_20261002/test_perf_cache.py"):
            p = subprocess.run([sys.executable, str(entry)], cwd=ROOT,
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_new_python_syntax_without_execution(self):
        paths = list(EXP.rglob("*.py")) + list((ROOT / "tools").glob("*.py"))
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_public_protocol_grid_and_no_deployment(self):
        p = json.loads((EXP / "configs/protocol.public.json").read_text())
        self.assertNotIn("server", p)
        self.assertEqual(len(p["datasets"]) * len(p["budgets"]) * len(p["targets"]), 40)
        self.assertEqual(p["targets"], {"softmatch": 2200, "simmatch": 2400})
        self.assertEqual(p["seed"], 20260825)
        self.assertFalse(p["model"]["use_pairusa"])


class PublicMetricsTests(unittest.TestCase):
    def test_threshold_matches_exhaustive_reference(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        p = np.array([0.1, 0.1, 0.4, 0.7, 0.9, 0.9])
        candidates = np.unique(np.r_[0, 0.5, np.nextafter(p.max(), np.inf), p])
        values = [(metrics.binary_metrics(y, p, t)["macro_f1"], t) for t in candidates]
        maximum = max(v[0] for v in values)
        expected = min((t for score, t in values if abs(score-maximum) < 1e-12),
                       key=lambda t: (abs(t-0.5), t))
        self.assertEqual(metrics.select_threshold(y, p), expected)

    def test_fixed_threshold_confusion_counts(self):
        m = metrics.binary_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], 0.5)
        self.assertEqual(m["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["macro_f1"], 0.5)

    def test_invalid_labels_and_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            metrics.binary_metrics([0, 2], [0.1, 0.9])
        with self.assertRaises(ValueError):
            metrics.select_threshold([0, 1], [0.1, np.nan])


class PrivateReportBoundaryTests(unittest.TestCase):
    def test_render_uses_aggregate_whitelist_only(self):
        summary = synthetic_summary()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            rendered = renderer.render(path)
        self.assertIn('id="report-data"', rendered)
        self.assertIn("40 / 40", rendered)
        self.assertNotIn("RAW_PRIVATE_SENTINEL", rendered)
        self.assertNotIn('src="https://', rendered)

    def test_incomplete_grid_rejected(self):
        summary = synthetic_summary()
        summary["results"].pop(next(iter(summary["results"])))
        with self.assertRaises(ValueError):
            renderer.load_rows(summary)

    def test_inconsistent_confusion_metrics_rejected(self):
        m = synthetic_summary()["results"]["apple_001_softmatch_s20260825"]["best_metrics"]
        m["accuracy"] = 0.6
        with self.assertRaises(ValueError):
            renderer.validate_metrics(m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
