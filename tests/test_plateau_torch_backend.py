"""Tiny CPU tensor replay plus fast synthetic controller integration.

FastControllerBackend deliberately simulates most updates. Its receipts/metrics
exercise contracts only, never certify ALBEF training, convergence or GPU use.
Boundary probes likewise do not constitute complete experimental trajectories.
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from fair_test_helpers import synthetic_metrics
from fair_benchmark.spec import SEEDS, fingerprint
from plateau_benchmark.spec import load_protocol, make_config
from plateau_benchmark.contracts import signed
from plateau_benchmark.report import _full_timing_comparable, summarize_results, validate_result

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from test_fair_torch_backend import TinyPair, SyntheticPairs
    from fair_benchmark.torch_backend import tree_digest
    from plateau_benchmark.runner import PlateauTrainer, train_until_stop
    from plateau_benchmark.torch_backend import PlateauBackend

    class PhasePairs(SyntheticPairs):
        def clock(self, step):
            return step if self.cfg["stage"] == "bce" else step+6000

        def batch(self, pairs, step, view, physical):
            return super().batch(pairs, self.clock(step), view, physical)

        def teacher_vectors(self, selected, step):
            return super().teacher_vectors(selected, self.clock(step))

    class FastControllerBackend(PlateauBackend):
        """Only the first update is real; the rest are synthetic state clocks."""
        score = .75

        def train_step(self):
            if self.step == 0:
                return super().train_step()
            self._assert_healthy()
            if self._test_consumed:
                raise ValueError("Test already consumed")
            self.step += 1
            return dict(step=self.step, committed=True, synthetic_controller_only=True,
                        cost=dict(attempts=1, failed_attempts=0, l_pair_draws=32,
                                  u_pair_draws=0 if self.cfg["budget_percent"] == 100 else 32,
                                  l_forward_pairs=32, backward_pairs=32, student_seconds=.001))

        def evaluate(self):
            score = self.score(self.step) if callable(self.score) else self.score
            return tuple(synthetic_metrics(self.step, score, model=m) for m in ("ema", "student"))

    def worker(method="bce", stage="adaptation", seed=SEEDS[0], parent=None, match=None, fast=False, budget=10):
        cfg = make_config("banana", budget, method, seed, stage, None if match is None else match["source_config"]["method"])
        cls = FastControllerBackend if fast else PlateauBackend
        backend = cls(cfg, PhasePairs(cfg), model=TinyPair(seed, cfg["uses_pairusa"]))
        return PlateauTrainer(cfg, backend, {"inputs": copy.deepcopy(backend.input_identity)}, parent=parent, match_receipt=match)

    def finish(session):
        while not session.budget.stop_reason:
            session.advance()
            if session.budget.evaluation_due:
                session.evaluate()
        return session

    def parent_worker(seed=SEEDS[0]):
        return finish(worker(stage="bce", seed=seed, fast=True))


@unittest.skipIf(torch is None, "PyTorch absent: isolated CPU environment required")
class PlateauTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.parent = parent_worker().export_parent()

    def test_all_nine_import_same_student_ema_rng_with_fresh_optimizer(self):
        hashes = []
        for method in ("bce", "pairusa", "ot", "pairusa_ot", "meanteacher", "fixmatch", "softmatch", "simmatch", "freematch"):
            with self.subTest(method=method):
                session = worker(method, parent=self.parent)
                backend = session.backend
                self.assertEqual(backend.step, 0)
                self.assertEqual(backend.physical, 16)
                self.assertFalse(backend.opt.state)
                for model, key in ((backend.model, "student"), (backend.ema, "ema")):
                    actual = backend._named_state(model, backend.common_names)
                    self.assertEqual(tree_digest(actual), tree_digest(self.parent["backend"][key]))
                hashes.append(backend.adaptation_common_hash)
                self.assertEqual(tree_digest(backend._rng()), tree_digest(self.parent["backend"]["rng"]))
        self.assertEqual(len(set(hashes)), 1)

    def test_all_nine_actual_tensor_updates_complete_state_replay(self):
        for method in ("bce", "pairusa", "ot", "pairusa_ot", "meanteacher", "fixmatch", "softmatch", "simmatch", "freematch"):
            with self.subTest(method=method):
                a = worker(method, parent=self.parent).backend
                if method in ("pairusa", "ot", "pairusa_ot"):
                    a.step = 200  # Synthetic active-loss boundary probe only.
                first = a.train_step(); saved = a.snapshot()
                a.train_step(); reference = a.snapshot()
                b = worker(method, parent=self.parent).backend
                b.restore(saved); b.train_step()
                self.assertEqual(tree_digest(reference), tree_digest(b.snapshot()))
                self.assertGreater(first["losses_raw"]["bce"], 0)
                self.assertEqual(first["losses_raw"]["bce"], first["losses_weighted"]["bce"])
                self.assertAlmostEqual(first["total_loss"], sum(first["losses_weighted"].values()))
                self.assertTrue(first["supervised_bce_retained"])

    def test_simmatch_primes_imported_parent_ema_only_on_first_step(self):
        session = worker("simmatch", parent=self.parent); backend = session.backend
        self.assertIsNone(backend.bank)
        self.assertEqual(backend.initial_cost["l_forward_pairs"], 0)
        cold = backend.snapshot()
        replay = worker("simmatch", parent=self.parent).backend; replay.restore(cold)
        self.assertIsNone(replay.bank)
        observed, expected = [], backend.ema_state_fingerprint()
        original = backend._prime_bank
        def prime():
            observed.append(backend.ema_state_fingerprint())
            return original()
        with patch.object(backend, "_prime_bank", side_effect=prime):
            record = backend.train_step()
        self.assertEqual(observed, [expected])
        self.assertIn("bank_initialization_seconds", record["cost"])
        self.assertEqual(record["cost"]["l_forward_pairs"], 128+64)
        self.assertEqual(record["instance_loss"], 0.)
        invalid = backend.snapshot(); invalid["algorithm"]["bank"] = None
        with self.assertRaises(ValueError):
            replay.restore(invalid)

    def test_freematch_and_mt_use_phase_local_clock_not_parent_steps(self):
        mt = worker("meanteacher", parent=self.parent)
        first, second = mt.advance(), mt.advance()
        self.assertEqual(first["unlabeled_weight"], 0.)
        self.assertEqual(second["unlabeled_weight"], 1/1600)
        self.assertEqual(first["lr_multiplier"], 1/80)
        free = worker("freematch", parent=self.parent)
        self.assertEqual(free.backend.algorithm.updates, 0)
        free.advance(); self.assertEqual(free.backend.algorithm.updates, 1)
        free.advance(); saved = free.snapshot()
        resumed = worker("freematch", parent=self.parent); resumed.restore(saved)
        self.assertEqual(resumed.backend.algorithm.updates, 2)

    def test_parent_import_requires_unconsumed_matching_complete_state(self):
        cfg = make_config("banana", 10, "ot", SEEDS[0])
        cold = PlateauBackend(cfg, PhasePairs(cfg), model=TinyPair(cfg["seed"]))
        with self.assertRaises(ValueError):
            cold.train_step()
        for key, value in (("student", {}), ("test_consumed", True), ("format", "fair_torch_state_v3")):
            altered = copy.deepcopy(self.parent); altered["backend"][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                worker("ot", parent=altered)
        with self.assertRaises(ValueError):
            worker("ot", seed=SEEDS[1], parent=self.parent)
        cold.import_parent(self.parent)
        with self.assertRaises(ValueError):
            cold.import_parent(self.parent)

    def test_same_step_oom_preserves_clock_and_full_retry_cost(self):
        s = worker("softmatch", parent=self.parent)
        s.backend.model.fail_once = True
        record = s.advance()
        self.assertEqual((s.budget.step, record["cost"]["attempts"], s.backend.physical), (1, 2, 8))
        self.assertEqual(s.cost.values["failed_attempts"], 1)

    def test_partial_commit_oom_poison_does_not_resume_live(self):
        s = worker(parent=self.parent)
        def partial():
            with torch.no_grad():
                s.backend.params[0].add_(1.)
            raise torch.cuda.OutOfMemoryError("synthetic partial commit")
        with patch.object(s.backend.opt, "step", side_effect=partial), self.assertRaises(RuntimeError):
            s.advance()
        self.assertEqual(s.budget.step, 0)
        with self.assertRaises(RuntimeError):
            s.snapshot()

    def test_common_batches_match_between_methods_and_matched_bce(self):
        a = worker("ot", parent=self.parent)
        b = worker("bce", parent=self.parent)
        lp = a.backend.data.flatten(a.backend.data.l[:16])
        self.assertEqual(tree_digest(a.backend.data.batch(lp, 101, "weak", 16)), tree_digest(b.backend.data.batch(lp, 101, "weak", 16)))
        parent_cfg = self.parent["receipt"]["config"]
        parent_data = PhasePairs(parent_cfg)
        self.assertNotEqual(tree_digest(a.backend.data.batch(lp, 101, "weak", 16)), tree_digest(parent_data.batch(lp, 101, "weak", 16)))


@unittest.skipIf(torch is None, "PyTorch absent")
class PlateauIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.bce_parent = parent_worker()
        cls.parent = cls.bce_parent.export_parent()

    def test_full_parent_checkpoint_and_plateau_receipt_survive_resume(self):
        saved = self.bce_parent.snapshot()
        resumed = worker(stage="bce", fast=True); resumed.restore(saved)
        self.assertEqual(resumed.budget.state_dict(), self.bce_parent.budget.state_dict())
        self.assertEqual(resumed.export_parent()["receipt"], self.parent["receipt"])
        changed = copy.deepcopy(saved); changed["format"] = "fair_itm_full_v3"
        with self.assertRaises(ValueError):
            resumed.restore(changed)

    def test_cap_without_plateau_never_exports_or_seals_test(self):
        s = worker(stage="bce", fast=True)
        s.backend.score = lambda step: .5+step/20000
        finish(s)
        self.assertEqual(s.budget.stop_reason, "phase_cap_exhausted")
        for action in (s.export_parent, s.seal_for_test):
            with self.assertRaises(ValueError):
                action()
        row = s.result()
        self.assertEqual(row["state"], "bce_cap_requires_review")
        validate_result(row)

    def test_branch_resets_patience_charges_parent_and_resume_costs(self):
        s = worker("ot", parent=self.parent, fast=True)
        self.assertEqual(s.budget.stale, 0)
        for _ in range(23):
            s.advance()
        saved = s.snapshot()
        resumed = worker("ot", parent=self.parent, fast=True); resumed.restore(saved)
        self.assertEqual(resumed.cost.values["attempts"], 23)
        self.assertGreaterEqual(resumed.cost.values["resume_setup_seconds"], 0)
        finish(resumed)
        row = resumed.result(); validate_result(row)
        self.assertEqual(row["additional_steps"], 2400)
        self.assertEqual(row["total_student_steps"], 4000)
        self.assertEqual(row["compute"]["attempts"], 4000)
        self.assertEqual(row["upstream_bce_compute"], self.parent["receipt"]["compute"])

    def test_matched_control_exact_source_steps_and_target_survives_resume(self):
        source = worker("ot", parent=self.parent, fast=True)
        source.backend.score = lambda step: .75 if step < 2000 else .76
        finish(source); receipt = source.export_match_receipt()
        self.assertEqual(receipt["additional_steps"], 2800)
        matched = worker(parent=self.parent, stage="matched_bce", match=receipt, fast=True)
        for _ in range(100):
            matched.advance()
        matched.evaluate()
        saved = matched.snapshot()
        resumed = worker(parent=self.parent, stage="matched_bce", match=receipt, fast=True); resumed.restore(saved)
        finish(resumed)
        self.assertEqual(resumed.budget.step, source.budget.step)
        self.assertEqual(resumed.budget.stop_reason, "matched_source_steps")
        validate_result(resumed.result())
        altered = copy.deepcopy(saved); altered["match_receipt"]["additional_steps"] = 1600
        with self.assertRaises(ValueError):
            resumed.restore(altered)

    def test_matching_target_cannot_switch_parent_even_same_cell(self):
        source = finish(worker("ot", parent=self.parent, fast=True)); match = source.export_match_receipt()
        altered = copy.deepcopy(self.parent)
        r = altered["receipt"]; r["compute"]["setup_seconds"] += 1
        altered["receipt"] = signed({k: v for k, v in r.items() if k != "receipt_sha256"})
        with self.assertRaisesRegex(ValueError, "parent differ"):
            worker(parent=altered, stage="matched_bce", match=match, fast=True)

    def test_test_intent_failure_and_old_snapshot_cannot_refund(self):
        s = finish(worker("ot", parent=self.parent, fast=True)); s.seal_for_test()
        saved = s.snapshot()
        def fail(_):
            raise OSError("synthetic journal failure")
        with self.assertRaises(OSError):
            s.evaluate_test(persist_intent=fail)
        self.assertEqual(s.backend.test_forward_pairs, 0)
        self.assertTrue(s.test_attempted)
        with self.assertRaises(ValueError):
            s.restore(saved)
        with self.assertRaises(ValueError):
            s.evaluate_test()

    def test_completed_test_locks_selection_blocks_training_matching_and_retest(self):
        s = finish(worker("ot", parent=self.parent, fast=True)); s.export_match_receipt()
        selection = s.seal_for_test(); before = s.snapshot()
        s.evaluate_test()
        row = s.result(); validate_result(row)
        self.assertEqual(row["primary_metrics"]["threshold"], selection["threshold"])
        for action in (s.advance, s.export_match_receipt, s.evaluate_test):
            with self.assertRaises(ValueError):
                action()
        with self.assertRaises(ValueError):
            s.restore(before)
        resumed = worker("ot", parent=self.parent, fast=True); resumed.restore(s.snapshot())
        with self.assertRaises(ValueError):
            resumed.evaluate_test()

    def test_report_separates_three_seed_attribution_and_marks_missing_methods(self):
        rows = []
        for seed in SEEDS:
            p = parent_worker(seed); bundle = p.export_parent()
            b = finish(worker(seed=seed, parent=bundle, fast=True))
            m = finish(worker("ot", seed=seed, parent=bundle, fast=True)); match = m.export_match_receipt()
            c = finish(worker(seed=seed, parent=bundle, stage="matched_bce", match=match, fast=True))
            for s in (p, b, m, c):
                s.seal_for_test(); s.evaluate_test(); rows.append(s.result())
        summary = summarize_results(rows)
        types = {c["contrast"] for c in summary["paired_contrasts"]}
        self.assertEqual(len(types), 4)
        paired = next(c for c in summary["paired_contrasts"] if c["equal_additional_steps"])
        self.assertEqual(paired["treatment_additional_steps"], paired["baseline_additional_steps"])
        self.assertTrue(all(not c["statistical_significance_claim"] for c in summary["paired_contrasts"]))
        self.assertTrue(all(c["complete_attribution_controls"] for c in summary["attribution_controls"]))
        self.assertTrue(summary["comparison_cells"][0]["missing_methods"])
        self.assertNotIn("provenance", json_safe(summary))
        self.assertTrue(all(not g["timing_comparable"] for g in summary["groups"]))
        with self.assertRaises(ValueError):
            summarize_results(rows[:-1])
        with self.assertRaises(ValueError):
            summarize_results(rows+[rows[0]])
        without_matches = [r for r in rows if r["stage"] != "matched_bce"]
        partial = summarize_results(without_matches)
        missing = next(c for c in partial["attribution_controls"] if c["method"] == "ot")
        self.assertIn("matched_BCE_for_this_method", missing["missing_controls"])

    def test_diagnostic_without_test_never_mislabels_generalization(self):
        rows = []
        for seed in SEEDS:
            p = parent_worker(seed); rows.append(p.result())
        with self.assertRaises(ValueError):
            summarize_results(rows)
        summary = summarize_results(rows, view="validation_diagnostic")
        self.assertIn("not_generalization", summary["evaluation_split"])

    def test_report_rejects_dropped_upstream_cost_and_forged_phase_zero(self):
        s = finish(worker("ot", parent=self.parent, fast=True)); row = s.result()
        bad = copy.deepcopy(row); bad["compute"] = bad["phase_compute"]
        with self.assertRaises(ValueError):
            validate_result(bad)
        bad = copy.deepcopy(row); bad["adaptation_common_state_sha256"] = None
        with self.assertRaises(ValueError):
            validate_result(bad)


    def test_explicit_loop_logs_validation_and_persists_complete_milestones(self):
        s = worker(stage="bce", fast=True)
        checkpoints, logs = [], []
        def save(snapshot):
            checkpoints.append((snapshot["budget"]["step"], snapshot["budget"]["stop_reason"]))
        result = train_until_stop(s, persist_checkpoint=save, log_record=lambda record: logs.append(record["kind"]))
        self.assertEqual(checkpoints[0], (0, None))
        self.assertEqual(checkpoints[-1], (1600, "validation_plateau"))
        self.assertEqual(logs.count("validation"), 16)
        self.assertEqual(logs.count("training"), 1600)
        validate_result(result)

    def test_worker_loop_resumes_pending_validation_and_requires_real_sinks(self):
        s = worker(stage="bce", fast=True)
        for _ in range(100):
            s.advance()
        logs = []
        train_until_stop(s, log_record=lambda row: logs.append(row["kind"]))
        self.assertEqual(logs[0], "validation")
        other = worker(stage="bce", fast=True); other.backend.simulation_only = False
        with self.assertRaisesRegex(ValueError, "private atomic"):
            train_until_stop(other)
        self.assertEqual(other.budget.step, 0)

    def test_parent_source_proof_cannot_be_changed_between_phases(self):
        altered = copy.deepcopy(self.parent)
        metadata = {k: v for k, v in altered["receipt"].items() if k != "receipt_sha256"}
        metadata["source_sha256"] = "a"*64; altered["receipt"] = signed(metadata)
        with self.assertRaisesRegex(ValueError, "code fingerprints"):
            worker("ot", parent=altered)

    def test_full_cost_timing_requires_parent_hardware_and_pipeline_evidence(self):
        provenance = dict(hardware_sha256="1"*64, environment_sha256="2"*64,
                          serial_uncontended_execution=True, full_pipeline_same_hardware_serial_execution=True)
        rows = [dict(simulation_only=False, provenance=copy.deepcopy(provenance), parent_receipt=None) for _ in range(2)]
        self.assertTrue(_full_timing_comparable(rows))
        rows[1]["parent_receipt"] = {"execution_identity": dict(hardware_sha256="3"*64, environment_sha256="2"*64, serial_uncontended_execution=True)}
        self.assertFalse(_full_timing_comparable(rows))
        rows[1]["parent_receipt"] = None
        rows[1]["provenance"].pop("full_pipeline_same_hardware_serial_execution")
        self.assertFalse(_full_timing_comparable(rows))


def json_safe(value):
    import json
    return json.dumps(value, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
