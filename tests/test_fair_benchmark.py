"""Synthetic CPU-only fair-v3 controls; no data, checkpoints, GPU or SSH."""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fair_benchmark.budget import BudgetController, ComputeLedger
from fair_benchmark.evaluation import validate_teacher_accounting, validate_teacher_history
from fair_benchmark.report import summarize_results, _timing_comparable
from fair_benchmark.runner import FairTrainer
from fair_benchmark.spec import (ABLATION_METHODS, SEEDS, auxiliary_weights, load_protocol,
                                 fingerprint, lr_multiplier, make_config, make_plan, sample_indices, validate_config,
                                 validate_protocol)
from fair_test_helpers import (synthetic_holdout, synthetic_metrics, synthetic_teacher_accounting,
                               synthetic_teacher_resources)


def validation(step, accuracy=.75, auc=.80, model="ema"):
    return synthetic_metrics(step, accuracy, auc, model)


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
        self.holdout_identity = synthetic_holdout()
        self.input_identity = {"synthetic_unit_test_only": "same_inputs", "l_anchors": 16, "holdout": self.holdout_identity}
        self.test_contract_sha256, self.test_anchors = fingerprint(self.holdout_identity), 16
        self.test_consumed, self.test_forward_pairs = False, 0
        self.teacher_identity = fingerprint(["synthetic_teacher", cfg["seed"]]) if cfg["uses_pairusa"] else None
        self.initial_common_hash = "1" * 64
        self.initial_cost = {"setup_seconds": 1., **(synthetic_teacher_accounting() if cfg["uses_pairusa"] else {})}

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
        return dict(step=self.step, synthetic_only=True, test_consumed=self.test_consumed, test_forward_pairs=self.test_forward_pairs)

    def restore(self, state):
        self.step = state["step"]
        self.test_consumed, self.test_forward_pairs = state["test_consumed"], state["test_forward_pairs"]

    def ema_state_fingerprint(self):
        return fingerprint(["synthetic_terminal_ema", self.cfg, self.step])

    def evaluate_test(self, selection):
        if self.test_consumed:
            raise ValueError("Synthetic Test already consumed")
        self.test_consumed = True
        self.test_forward_pairs += 2*self.test_anchors
        return synthetic_metrics(self.step, split="test", anchors=self.test_anchors, threshold=selection["threshold"])

    def resource_usage(self):
        return dict(student_total_parameters=10, student_trainable_parameters=6, common_trainable_parameters=6,
                    method_specific_trainable_parameters=0, peak_cuda_allocated_bytes=None, peak_cuda_reserved_bytes=None,
                    teacher=synthetic_teacher_resources() if self.cfg["uses_pairusa"] else None)


def synthetic_session(seed=SEEDS[0], method="bce", policy="fixed", regime="native"):
    cfg = make_config("banana", 10, method, seed, policy, regime=regime)
    backend = SyntheticBackend(cfg)
    session = FairTrainer(cfg, backend, {"inputs": copy.deepcopy(backend.input_identity)})
    while not session.budget.stop_reason:
        session.advance()
        if session.budget.evaluation_due:
            session.evaluate()
    return session


def synthetic_result(seed=SEEDS[0], method="bce", policy="fixed", regime="native", tested=True):
    session = synthetic_session(seed, method, policy, regime)
    if tested:
        session.seal_for_test(); session.evaluate_test()
    return session.result()


class FairProtocolTests(unittest.TestCase):
    def test_three_training_seeds_separate_from_fixed_data_seed(self):
        p = load_protocol()
        self.assertEqual(p["training_seeds"], [20260825, 20260826, 20260827])
        self.assertEqual(p["data_construction_seed"], 20260825)

    def test_grid_unique_main_ablation_overlap_not_double_counted(self):
        p = make_plan()
        self.assertEqual(p["unique_student_configurations"], 684)
        self.assertEqual(p["main_configurations"], 420)
        self.assertEqual(p["ablation_configurations"], 264)
        self.assertEqual(p["warmup_control_configurations"], 168)
        self.assertEqual(len({r["run_id"] for r in p["runs"]}), 684)
        self.assertTrue(p["planned_not_completed"])
        self.assertFalse(p["equal_compute_claim"])
        self.assertTrue(all(not r["launch_enabled"] for r in p["runs"]))

    def test_all_methods_total_budget_and_no_s3_parent_or_extra_restart(self):
        p = load_protocol()
        self.assertFalse(p["ablation"]["s4_inherits_s3"])
        self.assertFalse(p["ablation"]["reset_optimizer_at_branch"])
        for cfg in make_plan()["runs"]:
            self.assertEqual(cfg["target_steps"], 3200)
            self.assertEqual(cfg["shared_bce_steps"], 1600 if cfg["regime"] == "shared_bce" else 0)
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

    def test_descriptive_policy_cannot_claim_a_different_execution(self):
        for section, key, value in (("evaluation", "threshold", "tune_on_test"),
                                    ("warmup_control", "ssl_internal_warmup_clock", "global_clock"),
                                    ("warmup_control", "simmatch_bank_initialization", "initial_EMA"),
                                    ("adaptive_supplement", "monitor", "training_loss"),
                                    ("fairness", "warmup_control_results_separate_from_main", 1),
                                    ("tuning", "prior_experiments_exist", False)):
            with self.subTest(section=section, key=key):
                changed = load_protocol(); changed[section][key] = value
                with self.assertRaises(ValueError):
                    validate_protocol(changed)


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
        inputs = {"inputs": copy.deepcopy(SyntheticBackend(cfg).input_identity)}
        session = FairTrainer(cfg, SyntheticBackend(cfg), inputs)
        while session.budget.step < 2000:
            session.advance()
            if session.budget.evaluation_due:
                session.evaluate()
        clone = FairTrainer(cfg, SyntheticBackend(cfg), inputs)
        clone.restore(session.snapshot())
        self.assertEqual(clone.snapshot()["backend"], session.snapshot()["backend"])
        self.assertEqual(clone.budget.state_dict(), session.budget.state_dict())
        self.assertEqual(clone.cost.values["resume_setup_seconds"], 1.)
        self.assertEqual(clone.cost.values["teacher_upstream_seconds"], session.cost.values["teacher_upstream_seconds"])
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
        self.assertEqual(len([g for g in report["groups"] if g["table"] == "main"]), 2)
        self.assertTrue(report["policies_and_tables_not_pooled"])

    def test_backend_and_input_provenance_cannot_be_mislabelled(self):
        cfg = make_config("banana", 10, "bce", SEEDS[0])
        other = make_config("banana", 10, "fixmatch", SEEDS[0])
        inputs = {"inputs": copy.deepcopy(SyntheticBackend(cfg).input_identity)}
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

    def test_missing_test_is_not_a_paper_main_result(self):
        rows = [synthetic_result(seed, tested=False) for seed in SEEDS]
        self.assertTrue(all(r["state"] == "awaiting_test" for r in rows))
        with self.assertRaisesRegex(ValueError, "Independent Test is missing"):
            summarize_results(rows)
        diagnostic = summarize_results(rows, view="validation_diagnostic")
        self.assertEqual(diagnostic["evaluation_split"], "validation_diagnostic_not_generalization")

    def test_test_access_requires_completed_training_and_explicit_seal(self):
        cfg = make_config("banana", 10, "bce", SEEDS[0])
        backend = SyntheticBackend(cfg)
        session = FairTrainer(cfg, backend, {"inputs": copy.deepcopy(backend.input_identity)})
        with self.assertRaisesRegex(ValueError, "not complete"):
            session.seal_for_test()
        done = synthetic_session()
        with self.assertRaisesRegex(ValueError, "Seal the terminal"):
            done.evaluate_test()
        self.assertFalse(done.test_attempted)

    def test_pre_module_validation_best_is_never_the_primary_checkpoint(self):
        cfg = make_config("banana", 10, "pairusa_ot", SEEDS[0])
        backend = SyntheticBackend(cfg)
        backend.evaluate = lambda: (validation(backend.step, .9 if backend.step <= 1600 else .6),
                                    validation(backend.step, model="student"))
        session = FairTrainer(cfg, backend, {"inputs": copy.deepcopy(backend.input_identity)})
        while not session.budget.stop_reason:
            session.advance()
            if session.budget.evaluation_due:
                session.evaluate()
        session.seal_for_test(); session.evaluate_test()
        result = session.result()
        self.assertEqual(result["validation_best_step"], 100)
        self.assertEqual(result["primary_step"], 3200)
        self.assertEqual(result["primary_metrics"]["paired_accuracy"], .75)

    def test_frozen_threshold_and_changed_ema_cannot_select_by_test(self):
        session = synthetic_session()
        sealed = session.seal_for_test()
        sealed["threshold"] = .9  # Returned copy cannot mutate the internal seal.
        self.assertEqual(session.test_selection["threshold"], .5)
        with patch.object(session.backend, "ema_state_fingerprint", return_value="8"*64):
            with self.assertRaisesRegex(ValueError, "changed after terminal"):
                session.evaluate_test()
        self.assertFalse(session.test_attempted)
        session.backend.evaluate_test = lambda selection: synthetic_metrics(3200, split="test", anchors=16, threshold=.9)
        with self.assertRaisesRegex(ValueError, "Test threshold differs"):
            session.evaluate_test()
        self.assertTrue(session.test_attempted)
        self.assertIsNone(session.test_metrics)

    def test_completed_test_intent_survives_resume_without_repeat(self):
        session = synthetic_session()
        session.seal_for_test(); session.evaluate_test()
        cfg = session.config
        backend = SyntheticBackend(cfg)
        clone = FairTrainer(cfg, backend, {"inputs": copy.deepcopy(backend.input_identity)})
        clone.restore(session.snapshot())
        with self.assertRaisesRegex(ValueError, "already consumed"):
            clone.evaluate_test()
        self.assertEqual(clone.result()["primary_metrics"], session.result()["primary_metrics"])
        self.assertEqual(clone.cost.values["test_forward_pairs"], 32)

    def test_failed_test_access_is_not_a_free_second_attempt(self):
        session = synthetic_session(); session.seal_for_test()
        def failed(selection):
            session.backend.test_consumed = True
            session.backend.test_forward_pairs += 4
            raise FloatingPointError("synthetic Test failure")
        session.backend.evaluate_test = failed
        with self.assertRaises(FloatingPointError):
            session.evaluate_test()
        self.assertEqual(session.result()["state"], "test_intent_pending_or_failed")
        self.assertEqual(session.cost.values["test_forward_pairs"], 4)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            session.evaluate_test()

    def test_real_path_requires_durable_intent_callback_not_a_gpu_proof(self):
        session = synthetic_session(); session.seal_for_test()
        session.backend.simulation_only = False  # Only exercise the API guard, no real GPU claim.
        with self.assertRaisesRegex(ValueError, "durable intent journal"):
            session.evaluate_test()
        self.assertFalse(session.test_attempted)

    def test_terminal_selection_and_old_version_checkpoint_rejected(self):
        session = synthetic_session(); session.seal_for_test()
        state = session.snapshot()
        cfg = session.config; backend = SyntheticBackend(cfg)
        clone = FairTrainer(cfg, backend, {"inputs": copy.deepcopy(backend.input_identity)})
        changed = copy.deepcopy(state); changed["test_selection"]["threshold"] = .8
        with self.assertRaisesRegex(ValueError, "terminal Validation"):
            clone.restore(changed)
        changed = copy.deepcopy(state); changed["format"] = "fair_itm_full_v2"
        with self.assertRaises(ValueError):
            clone.restore(changed)

    def test_tables_and_native_shared_bce_ssl_are_not_pooled(self):
        rows = [synthetic_result(seed, method, regime=regime) for seed in SEEDS
                for method, regime in (("bce", "native"), ("pairusa_ot", "native"),
                                       ("fixmatch", "native"), ("fixmatch", "shared_bce"))]
        report = summarize_results(rows)
        fix = [r for r in report["groups"] if r["method"] == "fixmatch"]
        self.assertEqual({r["table"] for r in fix}, {"main", "warmup_control"})
        self.assertEqual({r["regime"] for r in fix}, {"native", "shared_bce"})
        self.assertTrue(all(not c["complete_comparison"] for c in report["comparison_cells"]))

    def test_paired_three_seed_differences_and_full_cost_are_reported(self):
        rows = [synthetic_result(seed, method) for seed in SEEDS for method in ("bce", "pairusa_ot")]
        for row in rows:
            if row["method"] == "pairusa_ot":
                row["primary_metrics"]["paired_accuracy"] = dict(zip(SEEDS, (.8125, .75, .6875)))[row["seed"]]
        report = summarize_results(rows, table="main")
        difference = report["paired_comparisons"][0]["paired_differences"]["paired_accuracy"]
        self.assertEqual(difference["per_seed"], [.0625, 0., -.0625])
        self.assertEqual(difference["sample_std"], .0625)
        self.assertFalse(report["paired_comparisons"][0]["statistical_significance_claim"])
        full = next(g for g in report["groups"] if g["method"] == "pairusa_ot")
        self.assertEqual(full["teacher_validation_calls"], [1, 1, 1])
        self.assertEqual(full["student_validation_calls"], [32, 32, 32])
        self.assertFalse(full["timing_comparable"])

    def test_teacher_tuning_and_memory_cost_cannot_be_silently_omitted(self):
        rows = [synthetic_result(seed, "pairusa_ot") for seed in SEEDS]
        changed = copy.deepcopy(rows); changed[0]["compute"]["teacher_validation_calls"] = 0
        with self.assertRaisesRegex(ValueError, "teacher identity and operation costs"):
            summarize_results(changed)
        changed = copy.deepcopy(rows); changed[0]["tuning"]["trials_per_method"] = 10
        with self.assertRaisesRegex(ValueError, "tuning registration"):
            summarize_results(changed)
        changed = copy.deepcopy(rows); changed[0]["resources"]["peak_cuda_allocated_bytes"] = 1024
        with self.assertRaisesRegex(ValueError, "CPU simulations"):
            summarize_results(changed)

    def test_control_scope_is_fixed_and_has_no_hidden_extra_updates(self):
        plan = make_plan()
        controls = [r for r in plan["runs"] if "warmup_control" in r["roles"]]
        self.assertEqual({r["budget_percent"] for r in controls}, {1, 10})
        self.assertEqual({r["dataset"] for r in controls}, {"apple", "cassava", "rice", "banana"})
        self.assertTrue(all(r["target_steps"] == 3200 and r["shared_bce_steps"] == 1600 for r in controls))
        self.assertEqual(make_plan("adaptive")["unique_student_configurations"], 564)
        for policy, budget in (("fixed", 5), ("adaptive", 10)):
            with self.assertRaises(ValueError):
                make_config("banana", budget, "fixmatch", SEEDS[0], policy, regime="shared_bce")

    def test_sealed_selection_must_still_match_current_terminal_validation(self):
        for field in ("threshold", "validation_metrics_sha256"):
            session = synthetic_session(); session.seal_for_test()
            session.test_selection[field] = .8 if field == "threshold" else "9"*64
            with self.assertRaisesRegex(ValueError, "Frozen selection differs"):
                session.evaluate_test()
            self.assertFalse(session.test_attempted)
        session = synthetic_session(); session.seal_for_test()
        session.last_validation["ema"]["threshold"] = .8
        with self.assertRaisesRegex(ValueError, "Frozen selection differs"):
            session.seal_for_test()

    def test_failed_intent_persistence_blocks_access_and_cannot_be_refunded(self):
        session = synthetic_session(); session.seal_for_test()
        before, journal = session.snapshot(), []
        def failing_journal(receipt):
            journal.append(receipt)
            raise OSError("synthetic journal unavailable")
        with self.assertRaises(OSError):
            session.evaluate_test(persist_intent=failing_journal)
        self.assertTrue(session.test_attempted)
        self.assertFalse(session.backend.test_consumed)
        self.assertEqual(session.cost.values["test_forward_pairs"], 0)
        with self.assertRaisesRegex(ValueError, "cannot refund"):
            session.restore(before)
        cfg = session.config; backend = SyntheticBackend(cfg)
        clone = FairTrainer(cfg, backend, {"inputs": copy.deepcopy(backend.input_identity)})
        clone.restore(journal[0])
        with self.assertRaisesRegex(ValueError, "already consumed"):
            clone.evaluate_test()
        self.assertEqual(clone.result()["state"], "test_intent_pending_or_failed")

    def test_resume_rejects_test_access_without_intent_or_complete_forwards(self):
        session = synthetic_session(); session.seal_for_test()
        state = session.snapshot(); state["backend"]["test_consumed"] = True
        cfg = session.config; backend = SyntheticBackend(cfg)
        clone = FairTrainer(cfg, backend, {"inputs": copy.deepcopy(backend.input_identity)})
        with self.assertRaisesRegex(ValueError, "durable intent/accounting"):
            clone.restore(state)
        session.evaluate_test(); state = session.snapshot()
        state["backend"]["test_consumed"] = False
        state["backend"]["test_forward_pairs"] = state["compute"]["test_forward_pairs"] = 0
        with self.assertRaisesRegex(ValueError, "complete inference/durable intent"):
            clone.restore(state)

    def test_teacher_exact_counts_and_registered_stop_history(self):
        protocol = load_protocol()
        accounting = synthetic_teacher_accounting(17)
        validate_teacher_accounting(accounting, 17, protocol)
        for field in ("teacher_successful_updates", "teacher_backward_pairs", "teacher_l_forward_pairs",
                      "teacher_validation_forward_pairs", "teacher_descriptor_pairs"):
            changed = copy.deepcopy(accounting); changed[field] += 1
            with self.assertRaisesRegex(ValueError, "Teacher operation counts"):
                validate_teacher_accounting(changed, 17, protocol)
        history = [dict(epoch=i, validation_auroc=.8) for i in range(1, 5)]
        validate_teacher_history(history, protocol)
        for changed in (history[:2], history+[dict(epoch=5, validation_auroc=.8)],
                        [dict(epoch=1, validation_auroc=float("nan"))]):
            with self.assertRaises(ValueError):
                validate_teacher_history(changed, protocol)

    def test_timing_requires_cross_method_hardware_and_uncontended_execution(self):
        rows = [synthetic_result(seed, method) for seed in SEEDS for method in ("bce", "pairusa_ot")]
        with patch("fair_benchmark.report._timing_comparable", wraps=_timing_comparable) as checked:
            report = summarize_results(rows, table="main")
        self.assertTrue(any(len(call.args[0]) == 6 and {r["method"] for r in call.args[0]} == {"bce", "pairusa_ot"}
                            for call in checked.call_args_list))
        self.assertTrue(all(not g["timing_comparable"] for g in report["groups"]))
        # Metadata-only predicate fixtures, never claimed to be scientific results.
        timing = [dict(simulation_only=False, provenance=dict(hardware_sha256="1"*64,
                  environment_sha256="2"*64, serial_uncontended_execution=True)) for _ in range(6)]
        self.assertTrue(_timing_comparable(timing))
        timing[3]["provenance"]["hardware_sha256"] = "3"*64
        self.assertFalse(_timing_comparable(timing))
        timing[3]["provenance"]["hardware_sha256"] = "1"*64
        timing[5]["provenance"]["serial_uncontended_execution"] = False
        self.assertFalse(_timing_comparable(timing))

    def test_holdout_count_cannot_drift_between_contract_and_result(self):
        rows = [synthetic_result(seed) for seed in SEEDS]
        rows[0]["test_anchors"] = 32
        with self.assertRaisesRegex(ValueError, "Test membership contract changed"):
            summarize_results(rows)

if __name__ == "__main__":
    unittest.main(verbosity=2)
