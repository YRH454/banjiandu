"""CPU-only protocol/receipt controls; synthetic metrics are not experiments."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from fair_test_helpers import synthetic_metrics
from fair_benchmark.spec import SEEDS, fingerprint
from plateau_benchmark.budget import PhaseBudget
from plateau_benchmark.contracts import signed, validate_match_receipt, validate_test_selection
from plateau_benchmark.spec import (auxiliary_weights, load_protocol, lr_multiplier, make_config,
                                    make_plan, sample_indices, validate_config, validate_protocol)


def reach(budget, end=None, score=.75):
    while not budget.stop_reason and (end is None or budget.step < end):
        budget.commit_success(budget.step+1)
        if budget.evaluation_due:
            value = score(budget.step) if callable(score) else score
            budget.observe_validation(synthetic_metrics(budget.step, value))
    return budget


def match_receipt(method="ot", seed=SEEDS[0], l_anchors=64):
    cfg = make_config("banana", 10, method, seed)
    budget = reach(PhaseBudget(cfg, l_anchors))
    return signed(dict(format="plateau_match_v4", source_config=cfg, source_budget=budget.state_dict(),
                       parent_receipt_sha256="a"*64, input_identity_sha256="b"*64, l_anchors=l_anchors,
                       additional_steps=budget.step,
                       terminal_validation={m: synthetic_metrics(budget.step, model=m) for m in ("ema", "student")},
                       terminal_ema_sha256="c"*64, simulation_only=True, test_attempted=False))


class PlateauProtocolTests(unittest.TestCase):
    def test_unique_grid_has_three_seeds_parents_branches_and_matches(self):
        plan = make_plan()
        self.assertEqual((plan["logical_configurations"], plan["bce_parent_configurations"],
                          plan["adaptive_branch_configurations"], plan["matched_bce_configurations"]), (1128, 72, 564, 492))
        self.assertEqual((plan["main_configurations"], plan["ablation_configurations"]), (420, 264))
        self.assertEqual(len({r["run_id"] for r in plan["runs"]}), 1128)
        self.assertEqual(plan["training_seeds"], list(SEEDS))
        self.assertEqual(plan["data_construction_seed"], 20260825)
        self.assertTrue(all(not r["launch_enabled"] for r in plan["runs"]))

    def test_full_labels_exclude_u_ot_ssl_and_forbidden_matches(self):
        configs = [r for r in make_plan()["runs"] if r["budget_percent"] == 100]
        self.assertEqual({r["method"] for r in configs}, {"bce", "pairusa"})
        self.assertEqual({r["match_method"] for r in configs if r["stage"] == "matched_bce"}, {"pairusa"})
        for method in ("ot", "pairusa_ot", "fixmatch", "freematch"):
            with self.assertRaises(ValueError):
                make_config("banana", 100, method, SEEDS[0])

    def test_old_or_changed_protocol_config_is_not_resumable(self):
        for section, key, value in (("phases", "lr_floor_multiplier", 0.),
                                    ("phases", "second_stage_supervised_BCE_retained", False),
                                    ("tuning", "trials_per_method", 1)):
            p = load_protocol(); p[section][key] = value
            with self.assertRaises(ValueError):
                validate_protocol(p)
        cfg = make_config("banana", 10, "ot", SEEDS[0]); cfg["target_steps"] = 1600
        with self.assertRaises(ValueError):
            validate_config(cfg)
        with self.assertRaises(ValueError):
            make_config("banana", 10, "ot", SEEDS[0], "bce")

    def test_method_and_matching_independent_sampling_but_distinct_phases(self):
        configs = [make_config("banana", 10, m, SEEDS[0]) for m in ("bce", "ot", "fixmatch", "simmatch")]
        configs.append(make_config("banana", 10, "bce", SEEDS[0], "matched_bce", "ot"))
        draws = [sample_indices(c, 101, 64, 128) for c in configs]
        self.assertTrue(all(d == draws[0] for d in draws))
        parent = make_config("banana", 10, "bce", SEEDS[0], "bce")
        self.assertNotEqual(sample_indices(parent, 101, 64, 128), draws[0])
        other = make_config("banana", 10, "bce", SEEDS[1])
        self.assertNotEqual(sample_indices(other, 101, 64, 128), draws[0])

    def test_phase_lr_restarts_has_floor_and_matched_schedule_not_rescaled(self):
        a = make_config("banana", 10, "ot", SEEDS[0])
        b = make_config("banana", 10, "bce", SEEDS[0], "matched_bce", "ot")
        self.assertEqual(lr_multiplier(a, 1), 1/80)
        self.assertEqual(lr_multiplier(a, 4000), .1)
        self.assertEqual([lr_multiplier(a, s) for s in (1, 80, 100, 2400)], [lr_multiplier(b, s) for s in (1, 80, 100, 2400)])
        self.assertEqual(auxiliary_weights(a, 100)["ot"], 0.)
        self.assertAlmostEqual(auxiliary_weights(a, 200)["ot"], .1)

    def test_readonly_default_cli_no_torch_and_legacy_explicit(self):
        for flags, count in ((["plan", "--counts-only"], 1128), (["--version", "v3", "plan", "--counts-only"], 684)):
            script = "import runpy,sys;sys.modules['torch']=None;sys.argv="+repr([str(ROOT/"tools/fair_benchmark.py"), *flags])+";runpy.run_path(sys.argv[0],run_name='__main__')"
            r = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            output = json.loads(r.stdout)
            self.assertEqual(output.get("logical_configurations", output.get("unique_student_configurations")), count)


class PlateauBudgetTests(unittest.TestCase):
    def parent(self):
        return PhaseBudget(make_config("banana", 10, "bce", SEEDS[0], "bce"), 64)

    def test_bce_stops_at_validation_plateau_not_fixed_step(self):
        first = reach(self.parent())
        second = reach(self.parent(), score=lambda s: .75 if s < 1200 else .76)
        self.assertEqual((first.step, first.stop_reason), (1600, "validation_plateau"))
        self.assertEqual(second.step, 2000)
        self.assertFalse(first.evaluation_due)
        with self.assertRaises(ValueError):
            first.commit_success(1601)

    def test_improving_parent_hits_cap_not_fake_convergence(self):
        budget = reach(self.parent(), score=lambda s: .5+s*.00005)
        self.assertEqual((budget.step, budget.stop_reason), (6000, "phase_cap_exhausted"))

    def test_auroc_best_diagnostic_never_resets_primary_patience(self):
        budget = self.parent()
        while not budget.stop_reason:
            budget.commit_success(budget.step+1)
            if budget.evaluation_due:
                budget.observe_validation(synthetic_metrics(budget.step, auc=.5+budget.step/20000))
        self.assertEqual(budget.step, 1600)
        self.assertEqual(budget.best_step, 1600)

    def test_only_scheduled_ema_validation_and_no_skipped_milestone(self):
        budget = self.parent(); reach(budget, 99)
        with self.assertRaises(ValueError):
            budget.observe_validation(synthetic_metrics(99))
        budget.commit_success(100)
        with self.assertRaises(ValueError):
            budget.commit_success(101)
        for changes in ({"evaluation_model": "student"}, {"evaluation_split": "test"}, {"paired_accuracy": float("nan")}, {"validation_pairs": 16}, {"step": 99}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                budget.observe_validation({**synthetic_metrics(100), **changes})

    def test_all_branch_graces_cover_mt_and_sim_and_reset_patience(self):
        mins = []
        for method in ("bce", "pairusa", "ot", "pairusa_ot", "meanteacher", "fixmatch", "softmatch", "simmatch", "freematch"):
            budget = PhaseBudget(make_config("banana", 10, method, SEEDS[0]), 32000)
            mins.append(budget.minimum)
            self.assertEqual((budget.step, budget.stale, budget.significant_best), (0, 0, None))
        self.assertEqual(set(mins), {2100})
        with self.assertRaisesRegex(ValueError, "cannot cover"):
            PhaseBudget(make_config("banana", 10, "bce", SEEDS[0]), 64000)

    def test_resume_replays_history_and_rejects_forged_patience_grace_or_old(self):
        budget = reach(self.parent(), 1300)
        replay = self.parent(); replay.load_state_dict(budget.state_dict())
        self.assertEqual(reach(budget).state_dict(), reach(replay).state_dict())
        for name, value in (("stale", 0), ("minimum", 100), ("cap", 1600), ("format", "fair_budget_v3")):
            state = budget.state_dict(); state[name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                self.parent().load_state_dict(state)

    def test_matched_ignores_own_plateau_until_exact_source_stop(self):
        receipt = match_receipt()
        cfg = make_config("banana", 10, "bce", SEEDS[0], "matched_bce", "ot")
        budget = reach(PhaseBudget(cfg, 64, match_receipt=receipt))
        self.assertEqual(budget.step, receipt["additional_steps"])
        self.assertEqual(budget.stop_reason, "matched_source_steps")
        self.assertIsNone(budget.significant_best)

    def test_matched_receipt_rejects_arbitrary_steps_cross_seed_or_test(self):
        cfg = make_config("banana", 10, "bce", SEEDS[0], "matched_bce", "ot")
        original = match_receipt()
        for name, value in (("additional_steps", 100), ("test_attempted", True), ("l_anchors", 32)):
            changed = {k: copy.deepcopy(v) for k, v in original.items() if k != "receipt_sha256"}; changed[name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                PhaseBudget(cfg, 64, match_receipt=signed(changed))
        with self.assertRaises(ValueError):
            validate_match_receipt(match_receipt(seed=SEEDS[1]), cfg, 64)
        with self.assertRaises(ValueError):
            PhaseBudget(cfg, 64)

    def test_test_selection_requires_final_history_and_phase_source(self):
        cfg = make_config("banana", 10, "ot", SEEDS[0]); budget = reach(PhaseBudget(cfg, 64))
        selection = dict(format="plateau_test_selection_v4", config_sha256=fingerprint(cfg), protocol_sha256=fingerprint(load_protocol()),
                         checkpoint="terminal_ema", model="ema", step=budget.step, threshold=.5,
                         validation_metrics_sha256="a"*64, ema_state_sha256="b"*64, test_contract_sha256="c"*64,
                         budget_state=budget.state_dict(), parent_receipt_sha256="d"*64, adaptation_common_state_sha256="e"*64)
        validate_test_selection(selection, cfg, load_protocol(), budget.step)
        for key, value in (("model", "student"), ("parent_receipt_sha256", None), ("threshold", float("nan")), ("format", "fair_test_selection_v3")):
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_test_selection({**selection, key: value}, cfg, load_protocol(), budget.step)


if __name__ == "__main__":
    unittest.main(verbosity=2)
