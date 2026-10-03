"""CPU-only publication checks. Synthetic tensors; no assets, GPU, training or SSH."""
from __future__ import annotations
import ast
import collections
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "docs/experiment_catalog_20261004.json"
MAN = ROOT / "docs/source_manifest_three_hosts_20261004.json"
CROPS = ("cassava_itm_g1_g4_v1", "rice_itm_s1_s4_v1", "banana_itm_s1_s4_v1")
MAIN = "multicrop_itm_mt_fixmatch_v1"
FREE = "multicrop_itm_freematch_v1"

def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

lookup = module_at("public_config_lookup", ROOT / "tools/find_experiment_config.py")
verifier = module_at("public_hash_verifier", ROOT / "tools/verify_source_archive.py")

class ThreeHostPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.catalog = json.loads(CAT.read_text(encoding="utf-8"))
        cls.rows = cls.catalog["runs"]
        cls.manifest = json.loads(MAN.read_text(encoding="utf-8"))

    def test_public_source_manifest(self):
        self.assertEqual(verifier.verify(MAN), 197)

    def test_unique_126_config_index_not_completed_claim(self):
        self.assertEqual(len(self.rows), 126)
        self.assertEqual(len({r["run_id"] for r in self.rows}), 126)
        self.assertTrue(self.catalog["not_a_completed_experiment_count"])
        self.assertEqual(self.catalog["excluded_external_user_managed_runs"], 40)

    def test_grid_counts_and_no_fullbudget_ot(self):
        counts = collections.Counter(r["experiment"] for r in self.rows)
        self.assertEqual(counts[MAIN], 40)
        self.assertEqual(counts[FREE], 20)
        for name in CROPS:
            self.assertEqual(counts[name], 22)
            hundred = [r for r in self.rows if r["experiment"] == name and r["budget"] == "100"]
            self.assertEqual({r["stage"] for r in hundred}, {"S1", "S2"})

    def test_configuration_paths_hash_lookup_and_targets(self):
        entries = {r["repository_path"]: r for r in self.manifest["files"]}
        for row in self.rows:
            with self.subTest(run=row["run_id"]):
                path = ROOT / row["public_config"]
                self.assertTrue(path.resolve().is_relative_to(ROOT.resolve()))
                cfg = json.loads(path.read_text())
                self.assertEqual(row["run_id"], cfg["run_id"])
                self.assertEqual(row["original_config_sha256"],
                                 entries[row["public_config"]]["original_sha256"])
                self.assertEqual(cfg["seed"], 20260825)
                self.assertFalse(cfg["publication"]["historical_fingerprint_reusable"])
                self.assertNotIn("launch_enabled", cfg)
                if row["experiment"] in (MAIN, FREE):
                    self.assertEqual(cfg["target_steps"],
                        {"meanteacher": 1800, "fixmatch": 2000, "freematch": 2600}[cfg["method"]])
                    self.assertEqual(cfg["training"]["logical_l_anchors"], 16)
                    self.assertEqual(cfg["training"]["logical_u_pairs"], 32)
                    self.assertTrue(cfg["evaluation"]["no_test_inference"])
                else:
                    self.assertEqual(cfg["training"]["max_steps"], 1600)

    def test_private_values_not_in_public_additions(self):
        forbidden = ("G:/", "G:\\", "C:/Users/", "C:\\Users\\",
                     "WIN-J", "jupyter-", "fj01-ssh", "10.50.",
                     "192.168.", "/root/rivermind-data", "id_ed25519")
        for entry in self.manifest["files"]:
            path = ROOT / entry["repository_path"]
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, text, str(path.relative_to(ROOT)))

    def test_configs_have_no_private_input_parent_bindings(self):
        for row in self.rows:
            cfg = json.loads((ROOT / row["public_config"]).read_text())
            for key in ("image_root", "pair_file", "u_file", "u_sha256",
                        "validation_file", "protocol_sha256", "host", "gpu_uuid"):
                self.assertNotIn(key, cfg)
            self.assertNotIn("checkpoint", cfg["model"])
            self.assertNotIn("tokenizer_path", cfg["model"])
            for block in (cfg.get("warmstart", {}), cfg["model"].get("warmstart", {})):
                self.assertTrue(set(block) <= {"parent_kind", "parent_run_id"})

    def test_python_syntax_without_training_import(self):
        for entry in self.manifest["files"]:
            path = ROOT / entry["repository_path"]
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_main_original_four_cpu_contracts(self):
        path = ROOT / "experiments" / MAIN / "code/unit_tests.py"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
        result = subprocess.run([sys.executable, "-B", str(path)],
            cwd=path.parent, env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_freematch_original_fourteen_CPU_reference_checks(self):
        path = ROOT / "experiments" / FREE / "code/test_freematch.py"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
        result = subprocess.run([sys.executable, "-B", str(path)],
            cwd=path.parent, env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_three_crop_OT_synthetic_finite_and_both_marginals(self):
        generator = torch.Generator().manual_seed(20260825)
        a = torch.randn(32, 16, generator=generator)
        q = torch.randn(32, 16, generator=generator)
        y = torch.tensor([0., 1.] * 16)
        for name in CROPS:
            ot = module_at("test_public_ot_" + name,
                           ROOT / "experiments" / name / "code/ot_loss.py")
            for tolerance in (1e-5, 1e-4):
                with self.subTest(crop=name, tolerance=tolerance):
                    targets, report = ot.soft_ot_targets(a, y, q, tolerance=tolerance)
                    self.assertTrue(torch.isfinite(targets).all())
                    self.assertFalse(targets.requires_grad)
                    self.assertLessEqual(report["row_residual"], tolerance)
                    self.assertLessEqual(report["col_residual"], tolerance)
                    self.assertLessEqual(report["iterations"], 100)

    def test_existing_ablation_core_identity(self):
        actual = hashlib.sha256((ROOT / "src/albef_ssl/model.py").read_bytes()).hexdigest()
        self.assertEqual(actual, "2f1784897dc4d94b12102bffb8c51537754f3f7c8e67c210d6d99ef98665b74b")

    def test_lookup_by_method_crop_budget_and_stage(self):
        rows = lookup.find(crop="banana", method="fixmatch", budget="10%")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "banana_010_fixmatch_s20260825")
        rows = lookup.find(crop="rice", stage="S4", budget="030")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "rice_itm_s4_l030_s20260825")

    def test_lookup_rejects_new_budget_and_does_not_include_external_scope(self):
        with self.assertRaises(ValueError):
            lookup.find(budget="12")
        self.assertEqual(lookup.find(method="simmatch"), [])
        self.assertEqual(lookup.find(method="softmatch"), [])

    def test_current_freematch_reference_execution_documented(self):
        p = json.loads((ROOT / "experiments" / FREE / "configs/protocol.public.json").read_text())
        self.assertEqual(p["freematch"]["ema_p"], .999)
        self.assertEqual(p["freematch"]["ent_loss_ratio"], .01)
        self.assertFalse(p["freematch"]["use_quantile"])
        self.assertFalse(p["freematch"]["clip_thresh"])
        self.assertTrue(p["model"]["activation_checkpointing"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
