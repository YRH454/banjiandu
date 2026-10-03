"""CPU-only contracts; no local CUDA/context allocation or experiment training."""
import unittest
import torch
from torch.nn import functional as F
from losses import probabilities, fixmatch_targets, mean_teacher_loss, weighted_bce
from data_backend import fixed_seed

class Contracts(unittest.TestCase):
    def test_binary_ce_equivalence(self):
        z = torch.tensor([-12., -.1, 0., 2., 20.], requires_grad=True)
        y = torch.tensor([0., 1., 0., 1., 1.])
        a = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        b = F.cross_entropy(torch.stack((torch.zeros_like(z), z), 1), y.long(), reduction="none")
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(probabilities(z), torch.softmax(torch.stack((torch.zeros_like(z), z), 1), 1))

    def test_fixmatch_threshold_and_gradient(self):
        weak = torch.tensor([-5., -1., 1., 5.], requires_grad=True)
        label, mask = fixmatch_targets(weak)
        self.assertEqual(label.tolist(), [0., 0., 1., 1.])
        self.assertEqual(mask.tolist(), [1., 0., 0., 1.])
        strong = torch.zeros(4, requires_grad=True)
        loss = weighted_bce(strong, label, mask).mean()
        loss.backward()
        self.assertAlmostEqual(loss.item(), .34657359, places=6)
        self.assertIsNone(weak.grad)
        self.assertEqual(strong.grad[1:3].tolist(), [0., 0.])

    def test_teacher_is_detached_not_student(self):
        teacher = torch.tensor([.9, .1], requires_grad=True)
        strong = torch.zeros(2, requires_grad=True)
        loss = mean_teacher_loss(strong, teacher).mean()
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(strong.grad.abs().sum()), 0)
        self.assertAlmostEqual(loss.item(), .16, places=6)

    def test_sampler_seed(self):
        self.assertEqual(fixed_seed(20260825, 1, "pair-sampler"), fixed_seed(20260825, 1, "pair-sampler"))
        self.assertNotEqual(fixed_seed(20260825, 1, "pair-sampler"), fixed_seed(20260825, 2, "pair-sampler"))

if __name__ == "__main__":
    unittest.main(verbosity=2)
