"""Synthetic CPU tests of all nine adapters, not actual ALBEF/GPU admission."""
from __future__ import annotations

import copy
import math
import sys
import unittest
from unittest.mock import patch
from fair_test_helpers import synthetic_holdout, synthetic_teacher_accounting, synthetic_teacher_resources
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None

if torch is not None:
    from fair_benchmark.spec import fingerprint, make_config, lr_multiplier
    from fair_benchmark.budget import BudgetController
    from fair_benchmark.references import source_module
    from fair_benchmark.torch_backend import TorchBackend, tree_digest

    class TinyPair(nn.Module):
        def __init__(self, seed, usa=False):
            super().__init__()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.lora_tail = nn.Linear(16, 768)
                self.match_head = nn.Linear(768, 1)
                self.student_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256))
            self.log_student_temperature = nn.Parameter(torch.tensor(math.log(.07)))
            if not usa:
                self.student_projection.requires_grad_(False)
                self.log_student_temperature.requires_grad_(False)
            self.fail_once = False

        def forward_pairs(self, images, text, mask):
            if self.fail_once and torch.is_grad_enabled():
                self.fail_once = False
                raise torch.cuda.OutOfMemoryError("synthetic single OOM; no actual GPU")
            raw = torch.tanh(self.lora_tail(images+text))
            return self.match_head(raw).squeeze(-1), raw

    class SyntheticPairs:
        def __init__(self, cfg):
            self.cfg = cfg
            self.holdout_identity = synthetic_holdout()
            self.input_identity = {"synthetic_pairs_only": "same_fixed_data", "l_anchors": 64, "holdout": self.holdout_identity}
            self.test_contract_sha256, self.test_anchors = fingerprint(self.holdout_identity), 16
            self.teacher_identity = fingerprint(["synthetic_teacher", cfg["seed"]]) if cfg["uses_pairusa"] else None
            self.l = [dict(image_id=f"synthetic_l_{i}", positive_text=f"synthetic_pos_{i}", negative_text=f"synthetic_neg_{i}") for i in range(64)]
            self.u = [] if cfg["budget_percent"] == 100 else [dict(image_id=f"synthetic_u_{i}", pair_id=f"synthetic_up_{i}", text=f"synthetic_blind_{i}") for i in range(128)]
            self.val = [dict(image_id=f"synthetic_val_{i}", positive_text=f"synthetic_vpos_{i}", negative_text=f"synthetic_vneg_{i}") for i in range(400)]
            self.teacher_cost_seconds = 2.
            self.teacher_accounting = synthetic_teacher_accounting(len(self.l)) if cfg["uses_pairusa"] else {}
            self.teacher_resources = synthetic_teacher_resources() if cfg["uses_pairusa"] else None

        def attach(self, model, device):
            self.device = device

        def load_test(self, selection):
            return [dict(image_id=f"synthetic_test_{i}", positive_text=f"synthetic_tpos_{i}", negative_text=f"synthetic_tneg_{i}") for i in range(16)]

        @staticmethod
        def flatten(rows):
            return [dict(image_id=r["image_id"], pair_id=r["image_id"]+":"+kind, text=r[key], label=label)
                    for r in rows for kind, key, label in (("pos", "positive_text", 1.), ("neg", "negative_text", 0.))]

        def batch(self, pairs, step, view, physical):
            def vector(key):
                generator = torch.Generator().manual_seed(int(fingerprint(key)[:16], 16))
                return torch.randn(16, generator=generator)
            images = torch.stack([vector([p["image_id"], step, view, self.cfg["seed"]]) for p in pairs])
            texts = torch.stack([vector(p["text"]) for p in pairs])
            return images, texts, torch.ones(len(pairs), 1)

        def teacher_vectors(self, selected, step):
            return torch.stack([torch.randn(256, generator=torch.Generator().manual_seed(
                int(fingerprint([r["image_id"], step, self.cfg["seed"]])[:16], 16))) for r in selected])

        @staticmethod
        def pairusa_loss(teacher, student, teacher_temperature, student_temperature):
            teacher = F.normalize(teacher.detach().float(), dim=-1)
            student = F.normalize(student.float(), dim=-1)
            diagonal = torch.eye(len(student), dtype=torch.bool)
            tp = F.softmax((teacher@teacher.T).masked_fill(diagonal, -1e4)/teacher_temperature, dim=-1)
            sl = F.log_softmax((student@student.T).masked_fill(diagonal, -1e4)/student_temperature, dim=-1)
            return F.kl_div(sl, tp, reduction="batchmean")

    def backend(method, seed=20260825, budget=10, regime="native"):
        cfg = make_config("banana", budget, method, seed, regime=regime)
        data = SyntheticPairs(cfg)
        return TorchBackend(cfg, data, model=TinyPair(seed, cfg["uses_pairusa"]))


@unittest.skipIf(torch is None, "PyTorch absent: use an isolated CPU test environment")
class FairTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_nine_methods_cold_full_state_replay(self):
        for method in ("bce", "pairusa", "ot", "pairusa_ot", "meanteacher", "fixmatch", "softmatch", "simmatch", "freematch"):
            with self.subTest(method=method):
                first = backend(method)
                if method in ("pairusa", "ot", "pairusa_ot"):
                    first.step = 1800  # synthetic active-phase probe, not formal successful updates
                first.train_step()
                saved = first.snapshot()
                first.train_step()
                reference = first.snapshot()
                replay = backend(method)
                replay.restore(saved)
                replay.train_step()
                self.assertEqual(tree_digest(reference), tree_digest(replay.snapshot()))

    def test_ablation_common_warmup_state_matches_despite_extra_heads(self):
        values = []
        initials = []
        for method in ("bce", "pairusa", "ot", "pairusa_ot"):
            runner = backend(method)
            initials.append(runner.initial_common_hash)
            runner.train_step(); runner.train_step()
            values.append(runner.common_full_state_fingerprint())
        self.assertEqual(len(set(initials)), 1)
        self.assertEqual(len(set(values)), 1)

    def test_usa_global_gradient_replay_is_microbatch_invariant(self):
        small = backend("pairusa"); small.step = 1800
        initial = small.snapshot()
        whole = backend("pairusa"); whole.restore(initial)
        whole.physical = 32  # test-only full logical-batch reference, not admitted physical policy
        a, b = small.train_step(), whole.train_step()
        self.assertAlmostEqual(a["pairusa_loss"], b["pairusa_loss"], places=6)
        for name in small.names:
            left = dict(small.model.named_parameters())[name]
            right = dict(whole.model.named_parameters())[name]
            # Gradient equivalence plus the existing 1e-6 parameter replay
            # criterion; near-zero Adam moments amplify sub-ulp GEMM differences.
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(left, right, rtol=0, atol=1e-6)

    def test_saf_extra_forward_and_teacher_initialization_cost_are_visible(self):
        free = backend("freematch")
        record = free.train_step()
        self.assertEqual(record["cost"]["u_forward_pairs"], 96)
        self.assertEqual(record["cost"]["backward_pairs"], 64)
        self.assertEqual(free.algorithm.updates, 1)
        usa = backend("pairusa")
        self.assertEqual(usa.initial_cost["teacher_upstream_seconds"], 2.)
        sim = backend("simmatch")
        self.assertEqual(sim.initial_cost["l_forward_pairs"], 128)

    def test_same_step_oom_rolls_back_soft_statistics_and_counts_retry(self):
        failed = backend("softmatch")
        failed.model.fail_once = True
        record = failed.train_step()
        self.assertEqual(record["step"], 1)
        self.assertEqual(failed.physical, 8)
        self.assertEqual(record["cost"]["attempts"], 2)
        self.assertEqual(record["cost"]["failed_attempts"], 1)
        clean = backend("softmatch"); clean.physical = 8
        clean.train_step()
        torch.testing.assert_close(failed.algorithm.mu, clean.algorithm.mu, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(failed.algorithm.p_model, clean.algorithm.p_model, rtol=1e-6, atol=1e-7)

    def test_ema_and_student_evaluated_under_same_contract(self):
        runner = backend("bce")
        runner.train_step()
        primary, diagnostic = runner.evaluate()
        self.assertEqual(primary["evaluation_model"], "ema")
        self.assertEqual(diagnostic["evaluation_model"], "student")
        self.assertEqual(primary["validation_pairs"], 800)
        self.assertEqual(primary["validation_anchors"], 400)
        self.assertTrue(runner.model.training)

    def test_incomplete_nonfinite_or_changed_seed_checkpoint_rejected(self):
        runner = backend("simmatch")
        runner.train_step()
        snapshot = runner.snapshot()
        changed = copy.deepcopy(snapshot); del changed["rng"]
        with self.assertRaises(ValueError):
            runner.restore(changed)
        changed = copy.deepcopy(snapshot); changed["algorithm"]["bank"].fill_(float("nan"))
        with self.assertRaises(ValueError):
            runner.restore(changed)
        with self.assertRaises(ValueError):
            backend("simmatch", seed=20260826).restore(snapshot)

    def test_changed_inputs_or_teacher_checkpoint_rejected(self):
        runner = backend("pairusa")
        snapshot = runner.snapshot()
        changed = copy.deepcopy(snapshot); changed["input_identity_sha256"] = "3" * 64
        with self.assertRaises(ValueError):
            runner.restore(changed)
        changed = copy.deepcopy(snapshot); changed["teacher_targets_sha256"] = "3" * 64
        with self.assertRaises(ValueError):
            runner.restore(changed)

    def test_partial_optimizer_oom_never_retries_or_saves_corrupt_state(self):
        runner = backend("bce")
        calls = []
        def partial_commit():
            calls.append(True)
            with torch.no_grad():
                runner.params[0].add_(1.)
            raise torch.cuda.OutOfMemoryError("synthetic partial optimizer failure; no GPU")
        with patch.object(runner.opt, "step", side_effect=partial_commit):
            with self.assertRaisesRegex(RuntimeError, "optimizer/EMA commit"):
                runner.train_step()
        self.assertEqual(len(calls), 1)
        self.assertEqual(runner.step, 0)
        self.assertEqual(runner.physical, 16)
        for action in (runner.train_step, runner.snapshot, runner.evaluate):
            with self.assertRaisesRegex(RuntimeError, "last complete checkpoint"):
                action()

    def test_full_label_bce_does_not_require_or_process_u(self):
        runner = backend("bce", budget=100)
        record = runner.train_step()
        self.assertEqual(record["cost"]["u_pair_draws"], 0)
        self.assertEqual(record["cost"]["u_forward_pairs"], 0)

    def test_shared_bce_all_nine_methods_match_and_do_not_use_ssl_early(self):
        initials, endings = [], []
        for method in ("bce", "pairusa", "ot", "pairusa_ot", "meanteacher", "fixmatch", "softmatch", "simmatch", "freematch"):
            with self.subTest(method=method):
                runner = backend(method, regime="shared_bce")
                initials.append(runner.initial_common_hash)
                if method == "simmatch":
                    self.assertIsNone(runner.bank)
                    self.assertEqual(runner.initial_cost["l_forward_pairs"], 0)
                first = runner.train_step(); runner.train_step()
                self.assertEqual(first["cost"]["u_forward_pairs"], 0)
                self.assertEqual(first["unlabeled_weight"], 0.)
                endings.append(runner.common_full_state_fingerprint())
                if method == "freematch":
                    self.assertEqual(runner.algorithm.updates, 0)
        self.assertEqual(len(set(initials)), 1)
        self.assertEqual(len(set(endings)), 1)

    def test_shared_bce_mt_uses_branch_local_ramp_without_lr_restart(self):
        runner = backend("meanteacher", regime="shared_bce")
        runner.step = 1600  # Synthetic phase-boundary probe, not a formal completed warmup.
        first = runner.train_step(); second = runner.train_step()
        self.assertEqual(first["ssl_phase_step"], 1)
        self.assertEqual(first["unlabeled_weight"], 0.)
        self.assertEqual(second["unlabeled_weight"], 1/640)
        self.assertEqual(first["lr_multiplier"], lr_multiplier(1601))

    def test_shared_bce_sim_bank_primes_current_ema_at_branch_and_replays(self):
        runner = backend("simmatch", regime="shared_bce")
        runner.train_step(); runner.train_step()
        warm = runner.snapshot()
        self.assertIsNone(warm["algorithm"]["bank"])
        cold = backend("simmatch", regime="shared_bce"); cold.restore(warm)
        self.assertEqual(tree_digest(warm), tree_digest(cold.snapshot()))
        runner.step = 1600  # Synthetic active-phase probe only.
        expected_ema = runner.ema_state_fingerprint()
        observed, prime_costs = [], []
        original_prime = runner._prime_bank
        def prime():
            observed.append(runner.ema_state_fingerprint())
            cost = original_prime()
            prime_costs.append(cost)
            return cost
        with patch.object(runner, "_prime_bank", side_effect=prime):
            record = runner.train_step()
        self.assertEqual(observed, [expected_ema])
        # A tiny CPU fixture can finish inside the Windows timer resolution.
        self.assertEqual(record["cost"]["bank_initialization_seconds"], prime_costs[0][0])
        self.assertEqual(record["cost"]["l_forward_pairs"], prime_costs[0][1]+64)
        self.assertIsNotNone(runner.bank)
        self.assertEqual(record["instance_loss"], 0.)
        saved = runner.snapshot(); runner.train_step()
        replay = backend("simmatch", regime="shared_bce"); replay.restore(saved); replay.train_step()
        self.assertEqual(tree_digest(runner.snapshot()), tree_digest(replay.snapshot()))

    def test_shared_bce_freematch_statistics_commit_only_in_ssl_phase(self):
        runner = backend("freematch", regime="shared_bce")
        runner.train_step()
        self.assertEqual(runner.algorithm.updates, 0)
        runner.step = 1600  # Synthetic SAT boundary probe, not a real 1600-step run.
        runner.train_step()
        self.assertEqual(runner.algorithm.updates, 1)
        saved = runner.snapshot(); runner.train_step()
        replay = backend("freematch", regime="shared_bce"); replay.restore(saved); replay.train_step()
        self.assertEqual(tree_digest(runner.snapshot()), tree_digest(replay.snapshot()))

    def test_test_uses_locked_validation_threshold_never_searches_on_test(self):
        runner = backend("bce"); runner.train_step()
        runner.step = 3200  # Synthetic inference-contract probe; no formal GPU completion.
        metrics = runner.evaluate()[0]
        budget = BudgetController()
        for step in range(1, 3201):
            budget.commit_success(step)
            if budget.evaluation_due:
                budget.observe_validation(dict(step=step, paired_accuracy=.75, auroc=.8,
                                               evaluation_model="ema", validation_anchors=400, validation_pairs=800))
        selection = dict(format="fair_test_selection_v3", config_sha256=fingerprint(runner.cfg), protocol_sha256=fingerprint(runner.p),
                         checkpoint="terminal_ema", model="ema", step=3200, threshold=metrics["threshold"],
                         validation_metrics_sha256=fingerprint(metrics), ema_state_sha256=runner.ema_state_fingerprint(),
                         test_contract_sha256=runner.test_contract_sha256, budget_state=budget.state_dict())
        before_test = runner.snapshot()
        math_module = source_module("experiments/multicrop_itm_mt_fixmatch_v1/code/metrics.py")
        with patch.object(math_module, "select_threshold", side_effect=AssertionError("No Test tuning")):
            result = runner.evaluate_test(selection)
        self.assertEqual(result["threshold"], metrics["threshold"])
        self.assertEqual(result["evaluation_split"], "test")
        self.assertEqual(runner.test_forward_pairs, 32)
        with self.assertRaisesRegex(ValueError, "cannot refund"):
            runner.restore(before_test)
        replay = backend("bce"); replay.restore(runner.snapshot())
        with self.assertRaisesRegex(ValueError, "consumed"):
            replay.evaluate_test(selection)

    def test_parameter_counts_are_measured_and_cpu_memory_not_fabricated(self):
        runner = backend("pairusa")
        resources = runner.resource_usage()
        self.assertEqual(resources["student_trainable_parameters"], sum(v.numel() for v in runner.model.parameters() if v.requires_grad))
        self.assertEqual(resources["student_total_parameters"], sum(v.numel() for v in runner.model.parameters()))
        self.assertGreater(resources["method_specific_trainable_parameters"], 0)
        self.assertIsNone(resources["peak_cuda_allocated_bytes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
