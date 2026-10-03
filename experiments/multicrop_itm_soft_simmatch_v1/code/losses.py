"""Binary reductions of the selected SSL mechanisms (all-U normalization)."""
import torch
from torch.nn import functional as F

def probabilities(logits):
    p = logits.float().sigmoid()
    return torch.stack((1-p, p), dim=-1)

def fixmatch_targets(weak_logits, cutoff=.95):
    probs = probabilities(weak_logits.detach())
    confidence, label = probs.max(-1)
    return label.float(), (confidence >= cutoff).float()

def mean_teacher_loss(strong_logits, teacher_probability):
    # Two-class probability MSE averaged over classes reduces to this scalar MSE.
    return (strong_logits.float().sigmoid() - teacher_probability.detach()).square()

def weighted_bce(strong_logits, target, weights):
    return F.binary_cross_entropy_with_logits(strong_logits.float(), target.detach(), reduction="none") * weights.detach()
