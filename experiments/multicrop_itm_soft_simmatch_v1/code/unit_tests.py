"""CPU contracts against explicit reference equations; no local GPU operations."""
import math
import tempfile
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from ssl_algorithms import SoftMatchStatistics, DistributionQueue, simmatch_targets, instance_ce, assert_nested_close
from losses import probabilities, weighted_bce
from data_backend import flatten_l, fixed_seed
from common import exclusive

class SSLContracts(unittest.TestCase):
    def test_softmatch_reference_gaussian(self):
        stats = SoftMatchStatistics(momentum=.9, n_sigma=2)
        z = torch.tensor([-3., -.01, .2, 4.])
        raw = torch.softmax(torch.stack((torch.zeros_like(z), z), 1), 1)
        pmodel = raw.mean(0)
        aligned = raw * (.5+1e-6)/(pmodel+1e-6)
        aligned /= aligned.sum(1, keepdim=True)
        conf = aligned.max(1).values
        mu = .9*.5+.1*conf.mean()
        var = .9*1.+.1*conf.var(unbiased=True)
        expected = torch.exp(-torch.clamp(conf-mu, max=0).square()/(2*var/4))
        y, weights, actual_aligned = stats.targets(z)
        torch.testing.assert_close(actual_aligned, aligned)
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(stats.mu, mu)
        torch.testing.assert_close(stats.var, var)
        self.assertEqual(y.tolist(), raw.argmax(1).float().tolist())

    def test_softmatch_raw_targets_not_DA_targets(self):
        stats = SoftMatchStatistics(momentum=1.)
        stats.p_model = torch.tensor([.01, .99])
        labels, weights, aligned = stats.targets(torch.tensor([.1, .2, .3, .4]))
        self.assertEqual(labels.tolist(), [1., 1., 1., 1.])
        self.assertEqual(aligned.argmax(1).tolist(), [0, 0, 0, 0])
        self.assertTrue(bool((weights <= 1).all()))

    def test_gaussian_continuous_not_binary(self):
        stats = SoftMatchStatistics(momentum=1., n_sigma=2)
        stats.p_model = torch.tensor([.5, .5])
        stats.mu, stats.var = torch.tensor(.9), torch.tensor(.01)
        _, weight, _ = stats.targets(torch.tensor([0., 1., 2., 4.]))
        self.assertGreater(float(weight[2]), float(weight[1]))
        self.assertGreater(float(weight[1]), float(weight[0]))
        self.assertEqual(float(weight[3]), 1.)

    def test_distribution_queue_exact_reference(self):
        q = DistributionQueue(3)
        probs = torch.tensor([[.1, .9], [.4, .6]])
        got = q.align(probs)
        reference = probs*(.5+1e-6)/(probs.mean(0)/3+1e-6)
        reference /= reference.sum(1, keepdim=True)
        torch.testing.assert_close(got, reference)
        self.assertEqual(q.pointer, 1)
        state = q.state_dict()
        q.align(probs.flip(1))
        q.load_state_dict(state)
        assert_nested_close(q.state_dict(), state)

    def test_simmatch_equations_and_gradient(self):
        w = F.normalize(torch.tensor([[1., .3], [.1, 1.]]), dim=1)
        bank = F.normalize(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]), dim=1)
        labels = torch.tensor([1, 0, 1, 0])
        probs = torch.tensor([[.2, .8], [.7, .3]])
        target, mask, inst, agg = simmatch_targets(probs, w, bank, labels, cutoff=.5)
        q = torch.softmax(w @ bank.t()/.5, dim=1)
        expected = q*probs[:, labels]
        expected /= expected.sum(1, keepdim=True)
        torch.testing.assert_close(inst, expected)
        torch.testing.assert_close(target, (.9*probs+.1*agg)[:, 1])
        torch.testing.assert_close(inst.sum(1), torch.ones(2))
        torch.testing.assert_close(agg.sum(1), torch.ones(2))
        s = w.clone().requires_grad_(True)
        instance_ce(s, bank, inst).mean().backward()
        self.assertTrue(bool(torch.isfinite(s.grad).all()))
        self.assertGreater(float(s.grad.abs().sum()), 0.)

    def test_pair_memory_same_image_opposite_labels(self):
        rows = [{"image_id": "same_image", "positive_text": "full positive", "negative_text": "full other-case"}]
        pairs = flatten_l(rows)
        self.assertEqual(len({p["pair_id"] for p in pairs}), 2)
        self.assertEqual([p["label"] for p in pairs], [1., 0.])
        self.assertEqual(pairs[0]["image_id"], pairs[1]["image_id"])

    def test_transaction_replay_soft_statistics(self):
        stats = SoftMatchStatistics()
        stats.targets(torch.tensor([1., 2., 3., 4.]))
        saved = stats.state_dict()
        a = stats.targets(torch.tensor([-2., -.1, .1, 2.]))
        after_a = stats.state_dict()
        stats.load_state_dict(saved)
        b = stats.targets(torch.tensor([-2., -.1, .1, 2.]))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert_nested_close(after_a, stats.state_dict())

    def test_all_U_normalization_and_detach(self):
        z = torch.zeros(4, requires_grad=True)
        teacher = torch.tensor([1., 0., 1., 0.], requires_grad=True)
        loss = weighted_bce(z, teacher, torch.tensor([1., 0., 0., 1.])).mean()
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertAlmostEqual(loss.item(), math.log(2)/2, places=6)

    def test_exclusive_lock_prevents_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder)/"gpu.lock"
            with exclusive(lock):
                with self.assertRaises(RuntimeError):
                    with exclusive(lock):
                        pass

if __name__ == "__main__":
    unittest.main(verbosity=2)
