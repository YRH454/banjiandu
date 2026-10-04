"""Resumable success-only budget and validation-only early stopping."""
from __future__ import annotations

import copy
import math

from .spec import fingerprint, load_protocol, positive_int, validate_protocol


class BudgetController:
    def __init__(self, policy="fixed", protocol=None):
        self.protocol = load_protocol() if protocol is None else validate_protocol(copy.deepcopy(protocol))
        if policy not in ("fixed", "adaptive"):
            raise ValueError("Unknown budget policy")
        self.policy = policy
        self.step = 0
        self.history = []
        self.best_key = None
        self.best_step = None
        self.significant_best = None
        self.stale = 0
        self.stop_reason = None

    @property
    def cap(self):
        return self.protocol["training"]["max_successful_steps"]

    @property
    def evaluation_due(self):
        interval = self.protocol["evaluation"]["every_successful_steps"]
        return self.step > 0 and self.step % interval == 0 and (
            not self.history or self.history[-1]["step"] != self.step)

    def commit_success(self, step):
        positive_int(step, "committed step")
        if self.stop_reason or self.evaluation_due or step != self.step + 1 or step > self.cap:
            raise ValueError("Duplicate/out-of-order update, skipped validation, or exhausted budget")
        self.step = step

    def observe_validation(self, metrics):
        if not self.evaluation_due:
            raise ValueError("Validation must occur once on the common success-step schedule")
        if metrics.get("evaluation_model") != "ema" or metrics.get("step") != self.step:
            raise ValueError("Only this step's EMA validation may select or stop training")
        e = self.protocol["evaluation"]
        if metrics.get("validation_anchors") != e["validation_anchors"] or metrics.get("validation_pairs") != e["validation_pairs"]:
            raise ValueError("Validation membership/count contract changed")
        key = [metrics["paired_accuracy"], metrics["auroc"]]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in key):
            raise ValueError("Selection metrics must be finite probabilities")
        new_best = self.best_key is None or tuple(key) > tuple(self.best_key)
        if new_best:
            self.best_key, self.best_step = key, self.step
        self.history.append(dict(step=self.step, paired_accuracy=key[0], auroc=key[1]))
        stop = self.protocol["adaptive_supplement"]
        if self.policy == "adaptive" and self.step >= stop["min_successful_steps"]:
            if self.significant_best is None or key[0] - self.significant_best >= stop["min_delta"] - 1e-12:
                self.significant_best, self.stale = key[0], 0
            else:
                self.stale += 1
            if self.stale >= stop["patience_validations"]:
                self.stop_reason = "adaptive_validation_plateau"
        if self.step == self.cap:
            self.stop_reason = "max_successful_steps"
        return new_best

    def state_dict(self):
        return copy.deepcopy(dict(version="fair_budget_v2", policy=self.policy,
                                  protocol_sha256=fingerprint(self.protocol), step=self.step,
                                  history=self.history, best_key=self.best_key, best_step=self.best_step,
                                  significant_best=self.significant_best, stale=self.stale,
                                  stop_reason=self.stop_reason))

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("version") != "fair_budget_v2" or state.get("policy") != self.policy or state.get("protocol_sha256") != fingerprint(self.protocol):
            raise ValueError("Historical/changed budget checkpoint is not fair-v2 resumable")
        if type(state.get("step")) is not int or not 0 <= state["step"] <= self.cap:
            raise ValueError("Invalid successful-step count")
        replay = BudgetController(self.policy, self.protocol)
        interval = self.protocol["evaluation"]["every_successful_steps"]
        for i, row in enumerate(state.get("history", []), 1):
            if row["step"] != i * interval or row["step"] > state["step"] or replay.stop_reason:
                raise ValueError("Incomplete/out-of-order validation or training after early stop")
            replay.step = row["step"]
            replay.observe_validation({**row, "evaluation_model": "ema", "validation_anchors": 400, "validation_pairs": 800})
        last_validation = replay.history[-1]["step"] if replay.history else 0
        if state["step"] - last_validation > interval or (replay.stop_reason and state["step"] != replay.step):
            raise ValueError("Skipped validation or extra updates after termination")
        replay.step = state["step"]
        if replay.state_dict() != state:
            raise ValueError("Early-stop/best state does not match its validation history")
        self.__dict__.update(replay.__dict__)


class ComputeLedger:
    """Measured cost, not a fabricated FLOP count or equal-compute assertion."""
    FIELDS = ("student_seconds", "validation_seconds", "setup_seconds", "teacher_upstream_seconds",
              "bank_initialization_seconds", "attempts", "failed_attempts", "l_pair_draws", "u_pair_draws",
              "l_forward_pairs", "u_forward_pairs", "backward_pairs")

    def __init__(self):
        self.values = {k: 0.0 if k.endswith("seconds") else 0 for k in self.FIELDS}

    def add(self, **values):
        if set(values) - set(self.FIELDS):
            raise ValueError("Unknown computation accounting field")
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Costs must be finite and nonnegative")
            if not name.endswith("seconds") and type(value) is not int:
                raise ValueError("Visit/retry counts must be exact integers")
        for name, value in values.items():
            self.values[name] += value

    def state_dict(self):
        return copy.deepcopy(self.values)

    def load_state_dict(self, state):
        if set(state) != set(self.FIELDS):
            raise ValueError("Incomplete compute ledger")
        checked = ComputeLedger()
        checked.add(**state)
        self.values = checked.values
