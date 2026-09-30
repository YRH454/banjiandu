"""CPU-only save/resume integration check for ``train_queue.run_student``.

It installs synthetic in-process model/cache stand-ins.  No dataset, caption
file, ALBEF weight, teacher, or GPU is touched.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import types
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
AUDIT = ROOT / "audit"
INTEGRATION = AUDIT / "integration_smoke"


class SyntheticInterrupt(RuntimeError):
    pass


class StubPairITMModel(nn.Module):
    """Small deterministic replacement matching the trainer's A-method API."""
    def __init__(self, config):
        super().__init__()
        self.lora_scale = nn.Parameter(torch.tensor(0.25))
        self.head = nn.Linear(1, 1)
        with torch.no_grad():
            self.head.weight.fill_(0.10)
            self.head.bias.fill_(-0.05)

    def forward_cached(self, image, pos_text, pos_mask, neg_text, neg_mask):
        score = self.lora_scale * image[:, :1].float() + self.head(image[:, :1].float())
        return torch.cat((score, -score), dim=1), score.expand(-1, 256)

    def trainable_state(self):
        return {name: parameter.detach().cpu().clone() for name, parameter in self.named_parameters()
                if parameter.requires_grad}

    def load_trainable_state(self, state):
        expected = set(self.trainable_state())
        if set(state) != expected:
            raise ValueError("Synthetic state key mismatch")
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.copy_(state[name].to(parameter))


class StubPairCache:
    def __init__(self, model, tokenizer, image_root):
        self.model = model

    def prime_images(self, rows, flips=True):
        return None

    def prime_text(self, texts):
        return None

    def pair_batch(self, rows, step, augment):
        # The row-specific stable feature keeps gradients nonzero while being independent of resume time.
        base = torch.tensor([[1.0 + int(row["image_id"])] for row in rows], dtype=torch.float32)
        text = torch.zeros((len(rows), 1, 1), dtype=torch.float32)
        mask = torch.ones((len(rows), 1), dtype=torch.long)
        return base, text, mask, text.clone(), mask.clone()


def cfg(run_id: str, steps: int = 2) -> dict:
    return {
        "run_id": run_id, "budget": "005", "method": "A_bce", "seed": 20260825,
        "pair_file": "synthetic_train.csv", "validation_file": "synthetic_validation.csv", "image_root": ".",
        "model": {"seed": 20260825},
        "training": {
            "max_steps": steps, "logical_positive_batch": 16, "physical_positive_batch": 16,
            "warmup_steps": 1, "checkpoint_every": 1, "eval_every": 1,
            "lora_lr": 1e-2, "head_lr": 1e-2, "weight_decay": 0.0, "max_grad_norm": 1.0,
            "pairusa_teacher_temp": 1.0, "pairusa_student_temp_init": 0.07,
            "pairusa_lambda": 0.1, "pairusa_start": 100, "pairusa_ramp_end": 200,
            "teacher_batch": 16, "teacher_lr": 2e-4, "teacher_max_epochs": 2, "teacher_patience": 1,
        },
    }


def rows(count: int) -> list[dict[str, str]]:
    return [{"image_id": str(index), "positive_text": f"positive {index}", "negative_text": f"negative {index}",
             "edit_type": "synthetic", "review_status": "synthetic"} for index in range(count)]


def same(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def main() -> None:
    sys.path.insert(0, str(CODE))
    import train_queue as queue

    # The import inside run_student is deliberately satisfied by this fake module.
    fake_pair_model = types.ModuleType("pair_model")
    fake_pair_model.PairITMModel = StubPairITMModel
    fake_pair_model.pair_usa_loss = lambda *args: torch.zeros((), device=args[1].device)
    sys.modules["pair_model"] = fake_pair_model

    # Preserve previous audit artifacts rather than deleting them on a rerun.
    INTEGRATION.mkdir(parents=True, exist_ok=False)
    queue.OUTPUTS = INTEGRATION
    queue.DEVICE = torch.device("cpu")
    queue.PairCache = StubPairCache
    train, validation = rows(16), rows(4)
    source = {"synthetic": "resume-control-flow-only"}

    original_atomic_torch = queue.atomic_torch
    interrupted = {"raised": False}

    def interrupt_after_first_last(path, value):
        original_atomic_torch(path, value)
        if path.name == "last.pt" and value.get("step") == 1 and not interrupted["raised"]:
            interrupted["raised"] = True
            raise SyntheticInterrupt("intentional interruption after durable step-1 checkpoint")

    queue.atomic_torch = interrupt_after_first_last
    resumed_cfg = cfg("resume_a")
    try:
        queue.run_student(resumed_cfg, train, validation, tokenizer=None, source_fp=source)
        raise AssertionError("Synthetic interruption did not occur")
    except SyntheticInterrupt:
        pass
    finally:
        queue.atomic_torch = original_atomic_torch

    step_one = torch.load(INTEGRATION / "resume_a" / "last.pt", map_location="cpu", weights_only=False)
    assert step_one["step"] == 1 and interrupted["raised"]
    resumed = queue.run_student(resumed_cfg, train, validation, tokenizer=None, source_fp=source)
    resumed_last = torch.load(INTEGRATION / "resume_a" / "last.pt", map_location="cpu", weights_only=False)
    assert resumed_last["step"] == 2 and resumed["state"] == "completed_validation"

    uninterrupted_cfg = cfg("uninterrupted_a")
    uninterrupted = queue.run_student(uninterrupted_cfg, train, validation, tokenizer=None, source_fp=source)
    uninterrupted_last = torch.load(INTEGRATION / "uninterrupted_a" / "last.pt", map_location="cpu", weights_only=False)
    assert uninterrupted_last["step"] == 2 and uninterrupted["state"] == "completed_validation"
    assert same(resumed_last["trainable_state"], uninterrupted_last["trainable_state"])
    assert same(resumed_last["optimizer"], uninterrupted_last["optimizer"])

    changed_rejected = False
    changed_cfg = cfg("resume_a", steps=3)
    try:
        queue.run_student(changed_cfg, train, validation, tokenizer=None, source_fp=source)
    except RuntimeError as exc:
        changed_rejected = "provenance" in str(exc).lower()
    assert changed_rejected

    report = {
        "scope": "CPU-only synthetic run_student save/resume integration; no dataset/model/GPU access",
        "passed": True,
        "interrupted_after_durable_step": 1,
        "resumed_to_step": int(resumed_last["step"]),
        "uninterrupted_to_step": int(uninterrupted_last["step"]),
        "resumed_state_equals_uninterrupted": True,
        "resumed_optimizer_equals_uninterrupted": True,
        "changed_provenance_rejected": True,
        "test_evaluation": "not performed",
    }
    queue.atomic_json(AUDIT / "resume_checks.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
