"""Pair-ITM reductions of SoftMatch and SimMatch, with resumable statistics.

Source mechanism reference: Microsoft USB 1ef4cbebcc0b368158315aeb425053858cf6c845.
No original image class/disease labels or private U fields are accepted.
"""
from __future__ import annotations
import copy
import torch
from torch.nn import functional as F
from losses import probabilities

class SoftMatchStatistics:
    def __init__(self, momentum=.999, n_sigma=2, variance_min=1e-12):
        self.momentum, self.n_sigma, self.variance_min = momentum, n_sigma, variance_min
        self.p_model = None
        self.mu = torch.tensor(.5)
        self.var = torch.tensor(1.)

    @torch.no_grad()
    def targets(self, weak_logits):
        raw = probabilities(weak_logits.detach())
        avg = raw.mean(0)
        self.p_model = avg if self.p_model is None else self.p_model.to(raw) * self.momentum + avg * (1-self.momentum)
        aligned = raw * (.5+1e-6) / (self.p_model+1e-6)
        aligned = aligned / aligned.sum(-1, keepdim=True)
        confidence = aligned.max(-1).values
        self.mu = self.mu.to(raw) * self.momentum + confidence.mean() * (1-self.momentum)
        self.var = self.var.to(raw) * self.momentum + confidence.var(unbiased=True) * (1-self.momentum)
        delta = (confidence-self.mu).clamp(max=0)
        weights = torch.exp(-delta.square() / (2*self.var.clamp_min(self.variance_min)/(self.n_sigma**2)))
        # SoftMatch alignment is ONLY for weighting; pseudo targets use raw logits.
        return raw.argmax(-1).float(), weights, aligned

    def state_dict(self):
        return {"p_model": None if self.p_model is None else self.p_model.detach().cpu().clone(),
                "mu": self.mu.detach().cpu().clone(), "var": self.var.detach().cpu().clone(),
                "momentum": self.momentum, "n_sigma": self.n_sigma, "variance_min": self.variance_min}

    def load_state_dict(self, value):
        if value["momentum"] != self.momentum or value["n_sigma"] != self.n_sigma or value["variance_min"] != self.variance_min:
            raise RuntimeError("SoftMatch statistics settings changed")
        self.p_model = None if value["p_model"] is None else value["p_model"].clone()
        self.mu, self.var = value["mu"].clone(), value["var"].clone()

class DistributionQueue:
    def __init__(self, length=256):
        self.values = torch.zeros(length, 2)
        self.pointer = 0
        self.length = length

    @torch.no_grad()
    def align(self, probs):
        self.values = self.values.to(probs)
        self.values[self.pointer] = probs.detach().mean(0)
        self.pointer = (self.pointer+1) % self.length
        aligned = probs * (.5+1e-6) / (self.values.mean(0)+1e-6)
        return aligned / aligned.sum(-1, keepdim=True)

    def state_dict(self):
        return {"values": self.values.detach().cpu().clone(), "pointer": self.pointer, "length": self.length}

    def load_state_dict(self, value):
        if value["length"] != self.length or value["values"].shape != (self.length, 2):
            raise RuntimeError("SimMatch distribution queue shape mismatch")
        self.values, self.pointer = value["values"].clone(), int(value["pointer"])

@torch.no_grad()
def simmatch_targets(aligned_probs, weak_features, bank, bank_labels, temperature=.5, alpha=.9, warmup=False, cutoff=.95):
    weak_q = F.softmax(weak_features.detach().float() @ bank.detach().float().t() / temperature, dim=-1)
    class_factor = aligned_probs.detach()[:, bank_labels]
    instance_target = weak_q * class_factor
    instance_target /= instance_target.sum(-1, keepdim=True).clamp_min(1e-12)
    aggregated = torch.zeros_like(aligned_probs).scatter_add_(1, bank_labels.expand(len(weak_q), -1), weak_q)
    refined = aligned_probs if warmup else alpha*aligned_probs + (1-alpha)*aggregated
    mask = (refined.max(-1).values >= cutoff).float()
    return refined[:, 1].detach(), mask.detach(), instance_target.detach(), aggregated

def instance_ce(strong_features, bank, targets, temperature=.5):
    log_q = F.log_softmax(strong_features.float() @ bank.detach().float().t() / temperature, dim=-1)
    return -(targets.detach()*log_q).sum(-1)

def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(v) for v in value)
    return copy.deepcopy(value)

def assert_nested_close(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    elif isinstance(a, dict):
        if set(a) != set(b):
            raise AssertionError("State key sets differ")
        for key in a:
            assert_nested_close(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError("State sequence lengths differ")
        for x, y in zip(a, b):
            assert_nested_close(x, y)
    elif a != b:
        raise AssertionError(f"State values differ: {a!r} != {b!r}")
