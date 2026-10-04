"""Synthetic CPU-only fair-v2 controls; no data, checkpoints, GPU or SSH."""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fair_benchmark.budget import BudgetController, ComputeLedger
from fair_benchmark.report import summarize_results
from fair_benchmark.runner import FairTrainer
from fair_benchmark.spec import (ABLATION_METHODS, SEEDS, auxiliary_weights, load_protocol,
                                 fingerprint, lr_multiplier, make_config, make_plan, sample_indices, validate_config)


def validation(step, accuracy=.75, auc=.80, model="ema"):
    return dict(step=step, paired_accuracy=accuracy, auroc=auc, evaluation_model=model,
                validation_anchors=400, validation_pairs=800)


def reach(controller, end, score=.75):
    while controller.step < end and not controller.stop_reason:
        controller.commit_success(controller.step+1)
        if controller.evaluation_due:
            controller.observe_validation(validation(controller.step, score))


class SyntheticBackend:
    """In-memory controller test double, not a scientific model/GPU proof."""
    def __init__(self, cfg):
        self.cfg, self.step = cfg, 0
        self.simulation_only = True
        self.input_identity = {"synthetic_unit_test_only": "same_inputs"}
        self.teacher_identity = fingerprint(["synthetic_teacher", cfg["seed"]]) if cfg["uses_pairusa"] else None
        self.initial_common_hash = "1" * 64
        self.initial_cost = dict(setup_seconds=1., teacher_upstream_seconds=2. if cfg["uses_pairusa"] else 0.)

    def train_step(self):
        self.step += 1
        return dict(step=self.step, committed=True, cost=dict(attempts=1, failed_attempts=0,
                    l_pair_draws=32, u_pair_draws=0 if self.cfg["budget_percent"] == 100 else 32,
                    l_forward_pairs=32, backward_pairs=32, student_seconds=.01))

    def evaluate(self):
        return validation(self.step), validation(self.step, .99, .99, "student")

    def common_full_state_fingerprint(self):
        return "2" * 64

    def snapshot(self):
        return dict(step=self.step, synthetic_only=True)

    def restore(self, state):
        self.step = state["step"]


def synthetic_result(seed=SEEDS[0], method="bce", policy="fixed"):
    cfg = make_config("banana", 10, method, seed, policy)
    backend = SyntheticBackend(cfg)
    session = FairTrainer(cfg, backend, {"inputs": {"synthetic_unit_test_only": "same_inputs"}})
    while not session.budget.stop_reason:
        session.advance()
        if session.budget.evaluation_due:
            session.evaluate()
    return session.result()


class FairProtocolTests(unittest.TestCase):
    def test_three_training_seeds_separate_from_fixed_data_seed(self):
        p = load_protocol()
        self.assertEqual(p["training_seeds"], [20260825, 20260826, 20260827])
        self.assertEqual(p["data_construction_seed"], 20260825)

    def test_grid_unique_main_ablation_overlap_not_double_counted(self):
        p = make_plan()
        self.assertEqual(p["unique_student_configurations"], 564)
        self.assertEqual(p["main_configurations"], 420)
        self.assertEqual(p["ablation_configurations"], 264)
        self.assertEqual(len({r["run_id"] for r in p["runs"]}), 564)
        self.assertTrue(p["planned_not_completed"])
        self.assertFalse(p["equal_compute_claim"])
        self.assertTrue(all(not r["launch_enabled"] for r in p["runs"]))

    def test_all_methods_total_budget_and_no_s3_parent_or_extra_restart(self):
        p = load_protocol()
        self.assertFalse(p["ablation"]["s4_inherits_s3"])
        self.assertFalse(p["ablation"]["reset_optimizer_at_branch"])
        for cfg in make_plan()["runs"]:
            self.assertEqual(cfg["target_steps"], 3200)
            self.assertEqual(cfg["shared_bce_steps"], 1600 if cfg["method"] in ABLATION_METHODS else 0)
            self.assertNotIn("warmstart", cfg)

    def test_full_label_budget_has_no_ot_or_ssl_fake_group(self):
        for method in ("ot", "pairusa_ot", "fixmatch", "freematch"):
            with self.assertRaises(ValueError):
                make_config("apple", 100, method, SEEDS[0])
        self.assertEqual({r["method"] for r in make_plan()["runs"] if r["budget_percent"] == 100}, {"bce", "pairusa"})

    def test_method_independent_l_u_sampling_and_distinct_training_seeds(self):
        configs = [make_config("rice", 5, method, SEEDS[0]) for method in ("bce", "ot", "fixmatch", "simmatch")]
        draws = [sample_indices(c, 1701, 100, 200) for c in configs]
        self.assertTrue(all(v == draws[0] for v in draws))
        other = sample_indices(make_config("rice", 5, "bce", SEEDS[1]), 1701, 100, 200)
        self.assertNotEqual(draws[0], other)
        self.assertEqual(sample_indices(make_config("rice", 100, "bce", SEEDS[0]), 1, 100, 0)[1], [])

    def test_uniform_lr_horizon_and_ablation_auxiliary_delay(self):
        self.assertEqual(lr_multiplier(80), 1.)
        self.assertAlmostEqual(lr_multiplier(3200), 0.)
        cfg = make_config("banana", 10, "pairusa_ot", SEEDS[0])
        self.assertEqual(auxiliary_weights(cfg, 1600), dict(pairusa=0., ot=0.))
        self.assertEqual(auxiliary_weights(cfg, 1700), dict(pairusa=0., ot=0.))
        self.assertEqual(auxiliary_weights(cfg, 1800), dict(pairusa=.1, ot=.1))
        with self.assertRaises(ValueError):
            lr_multiplier(3201)

    def test_unknown_seed_or_modified_config_rejected(self):
        for seed in (42, True, 20260828):
            with self.assertRaises(ValueError):
                make_config("apple", 1, "bce", seed)
        cfg = make_config("banana", 10, "bce", SEEDS[0])
        cfg["target_steps"] = 4800
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_readonly_cli_needs_no_torch_and_generates_three_filtered_runs(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/fair_benchmark.py"), "plan",
                                 "--dataset", "banana", "--budget", "10", "--method", "pairusa_ot"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["selected_configurations"], 3)
        self.assertEqual([r["seed"] for r in plan["runs"]], list(SEEDS))


class FairBudgetTests(unittest.TestCase):
    def test_direct_controller_rejects_unregistered_protocol(self):
        p = load_protocol()
        p["training"]["max_successful_steps"] = 4800
        with self.assertRaises(ValueError):
            BudgetController(protocol=p)

    def test_fixed_budget_never_early_stops_and_has_32_equal_looks(self):
        c = BudgetController()
        reach(c, 3200)
        self.assertEqual(c.step, 3200)
        self.assertEqual(len(c.history), 32)
        self.assertEqual(c.stop_reason, "max_successful_steps")
        with self.assertRaises(ValueError):
            c.commit_success(3201)

    def test_validation_cannot_be_skipped_duplicated_or_use_student(self):
        c = BudgetController()
        for step in range(1, 101):
            c.commit_success(step)
        with self.assertRaises(ValueError):
            c.commit_success(101)
        with self.assertRaises(ValueError):
            c.observe_validation(validation(100, model="student"))
        c.observe_validation(validation(100))
        with self.assertRaises(ValueError):
            c.observe_validation(validation(100))
        with self.assertRaises(ValueError):
            c.commit_success(100)

    def test_adaptive_waits_until_after_warmup_and_resumes_patience(self):
        c = BudgetController("adaptive")
        reach(c, 2000)
        self.assertIsNone(c.stop_reason)
        self.assertEqual(c.stale, 4)
        replay = BudgetController("adaptive")
        replay.load_state_dict(c.state_dict())
        reach(c, 3200); reach(replay, 3200)
        self.assertEqual(c.step, 2400)
        self.assertEqual(c.stop_reason, "adaptive_validation_plateau")
        self.assertEqual(c.state_dict(), replay.state_dict())

    def test_auroc_tie_improvement_selects_best_but_not_patience_reset(self):
        c = BudgetController("adaptive")
        reach(c, 1600)
        for step in range(1601, 1701):
            c.commit_success(step)
        self.assertTrue(c.observe_validation(validation(1700, auc=.90)))
        self.assertEqual(c.best_step, 1700)
        self.assertEqual(c.stale, 1)

    def test_significant_primary_improvement_resets_only_registered_patience(self):
        c = BudgetController("adaptive")
        reach(c, 2000)
        for step in range(2001, 2101):
            c.commit_success(step)
        c.observe_validation(validation(2100, accuracy=.755))
        self.assertEqual(c.stale, 0)

    def test_tampered_reset_or_switched_policy_checkpoint_rejected(self):
        c = BudgetController("adaptive")
        reach(c, 2000)
        state = c.state_dict()
        state["stale"] = 0
        with self.assertRaises(ValueError):
            BudgetController("adaptive").load_state_dict(state)
        with self.assertRaises(ValueError):
            BudgetController("fixed").load_state_dict(c.state_dict())

    def test_invalid_or_nonfinite_validation_is_not_a_stop_signal(self):
        c = BudgetController("adaptive")
        for step in range(1, 101):
            c.commit_success(step)
        for score in (float("nan"), float("inf"), True, -1.):
            with self.assertRaises(ValueError):
                c.observe_validation(validation(100, score))
        self.assertEqual(c.history, [])

    def test_cost_ledger_retries_and_teacher_are_not_free(self):
        ledger = ComputeLedger()
        ledger.add(attempts=2, failed_attempts=1, student_seconds=3., teacher_upstream_seconds=5.)
        replay = ComputeLedger(); replay.load_state_dict(ledger.state_dict())
        self.assertEqual(replay.values, ledger.values)
        for value in (-1, float("nan"), True, .5):
            with self.assertRaises(ValueError):
                ledger.add(attempts=value)


class FairIntegrationTests(unittest.TestCase):
    def test_controller_resume_full_warmup_state_and_budget_match(self):
        cfg = make_config("banana", 10, "pairusa_ot", SEEDS[0], "adaptive")
        inputs = {"inputs": {"synthetic_unit_test_only": "same_inputs"}}
        session = FairTrainer(cfg, SyntheticBackend(cfg), inputs)
        while session.budget.step < 2000:
            session.advance()
            if session.budget.evaluation_due:
                session.evaluate()
        clone = FairTrainer(cfg, SyntheticBackend(cfg), inputs)
        clone.restore(session.snapshot())
        self.assertEqual(clone.snapshot(), session.snapshot())
        self.assertEqual(clone.warmup_fingerprint, "2" * 64)
        tampered = session.snapshot(); del tampered["warmup_common_full_state_sha256"]
        with self.assertRaises(ValueError):
            clone.restore(tampered)

    def test_results_three_seed_sample_std_and_missing_seed_rejection(self):
        rows = [synthetic_result(seed) for seed in SEEDS]
        report = summarize_results(rows)
        self.assertEqual(report["groups"][0]["seeds"], list(SEEDS))
        self.assertEqual(report["groups"][0]["metrics"]["paired_accuracy"], dict(mean=.75, sample_std=0.))
        with self.assertRaises(ValueError):
            summarize_results(rows[:2])
        with self.assertRaises(ValueError):
            summarize_results(rows+[rows[0]])

    def test_changed_input_and_warmup_cannot_enter_comparison(self):
        rows = [synthetic_result(seed, method) for seed in SEEDS for method in ("bce", "pairusa_ot")]
        summarize_results(rows)
        changed = copy.deepcopy(rows); changed[1]["provenance"]["inputs"] = {"synthetic_unit_test_only": "other"}
        with self.assertRaises(ValueError):
            summarize_results(changed)
        changed = copy.deepcopy(rows); changed[1]["warmup_common_full_state_sha256"] = "3" * 64
        with self.assertRaises(ValueError):
            summarize_results(changed)

    def test_adaptive_and_fixed_results_never_pooled_as_one_mean(self):
        rows = [synthetic_result(seed, policy=policy) for seed in SEEDS for policy in ("fixed", "adaptive")]
        report = summarize_results(rows)
        self.assertEqual({r["policy"] for r in report["groups"]}, {"fixed", "adaptive"})
        self.assertEqual(len(report["groups"]), 2)
        self.assertTrue(report["policies_not_pooled"])

    def test_backend_and_input_provenance_cannot_be_mislabelled(self):
        cfg = make_config("banana", 10, "bce", SEEDS[0])
        other = make_config("banana", 10, "fixmatch", SEEDS[0])
        inputs = {"inputs": {"synthetic_unit_test_only": "same_inputs"}}
        with self.assertRaises(ValueError):
            FairTrainer(cfg, SyntheticBackend(other), inputs)
        with self.assertRaises(ValueError):
            FairTrainer(cfg, SyntheticBackend(cfg), {"inputs": "different"})

    def test_teacher_reuse_and_synthetic_results_are_explicit(self):
        rows = [synthetic_result(seed, method) for seed in SEEDS for method in ("pairusa", "pairusa_ot")]
        self.assertTrue(summarize_results(rows)["simulation_only"])
        changed = copy.deepcopy(rows); changed[1]["teacher_targets_sha256"] = "3" * 64
        with self.assertRaises(ValueError):
            summarize_results(changed)
        changed = copy.deepcopy(rows); changed[1]["simulation_only"] = False
        with self.assertRaises(ValueError):
            summarize_results(changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
