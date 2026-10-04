"""Phase-local, replay-validated stopping; Test never supplies a stop signal."""
from __future__ import annotations

import copy
import math

from fair_benchmark.spec import fingerprint, positive_int
from .spec import validate_config, load_protocol


class PhaseBudget:
    def __init__(self, config, l_anchors, protocol=None, match_receipt=None):
        self.p = load_protocol() if protocol is None else copy.deepcopy(protocol)
        self.cfg = validate_config(copy.deepcopy(config), self.p)
        self.l_anchors = positive_int(l_anchors, "L anchors")
        if l_anchors < 16:
            raise ValueError("At least 16 L anchors required")
        self.match_receipt = copy.deepcopy(match_receipt)
        self.cap = config["target_steps"]
        phase = "bce" if config["stage"] == "bce" else "adaptation"
        settings = self.p["phases"][phase]
        self.minimum = settings["min_successful_steps"]
        if phase == "adaptation":
            self.minimum = max(self.minimum, math.ceil(l_anchors/16)+100)
            if self.minimum > self.cap:
                raise ValueError("Registered adaptation cap cannot cover common SimMatch grace; revise protocol, do not shorten grace")
        if config["stage"] == "matched_bce":
            from .contracts import validate_match_receipt
            validate_match_receipt(match_receipt, config, l_anchors, self.p)
            self.cap = match_receipt["additional_steps"]
        elif match_receipt is not None:
            raise ValueError("Only matched BCE may receive a matching receipt")
        self.patience, self.delta = settings["patience_validations"], settings["min_delta"]
        self.step, self.history, self.best_key, self.best_step = 0, [], None, None
        self.significant_best, self.stale, self.stop_reason = None, 0, None

    @property
    def evaluation_due(self):
        interval = self.p["evaluation"]["every_successful_steps"]
        return self.step > 0 and self.step % interval == 0 and (not self.history or self.history[-1]["step"] != self.step)

    def commit_success(self, step):
        positive_int(step, "successful phase step")
        if self.stop_reason or self.evaluation_due or step != self.step+1 or step > self.cap:
            raise ValueError("Duplicate/skipped/stopped phase update or pending Validation")
        self.step = step

    def observe_validation(self, metrics):
        if not self.evaluation_due or metrics.get("step") != self.step or metrics.get("evaluation_model") != "ema" or metrics.get("evaluation_split") != "validation" or metrics.get("validation_anchors") != 400 or metrics.get("validation_pairs") != 800:
            raise ValueError("Only scheduled EMA Validation may stop a phase")
        key = [metrics["paired_accuracy"], metrics["auroc"]]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in key):
            raise ValueError("Finite Validation probabilities required")
        best = self.best_key is None or tuple(key) > tuple(self.best_key)
        if best:
            self.best_key, self.best_step = key, self.step
        self.history.append(dict(step=self.step, paired_accuracy=key[0], auroc=key[1]))
        if self.cfg["stage"] != "matched_bce" and self.step >= self.minimum:
            if self.significant_best is None or key[0]-self.significant_best >= self.delta-1e-12:
                self.significant_best, self.stale = key[0], 0
            else:
                self.stale += 1
            if self.stale >= self.patience:
                self.stop_reason = "validation_plateau"
        if self.step == self.cap:
            self.stop_reason = "matched_source_steps" if self.cfg["stage"] == "matched_bce" else "phase_cap_exhausted"
        return best

    def state_dict(self):
        return copy.deepcopy(dict(format="plateau_phase_budget_v4", config_sha256=fingerprint(self.cfg),
                                  protocol_sha256=fingerprint(self.p), l_anchors=self.l_anchors,
                                  match_receipt=self.match_receipt, minimum=self.minimum, cap=self.cap,
                                  step=self.step, history=self.history, best_key=self.best_key, best_step=self.best_step,
                                  significant_best=self.significant_best, stale=self.stale, stop_reason=self.stop_reason))

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != set(self.state_dict()) or state.get("format") != "plateau_phase_budget_v4" or state.get("config_sha256") != fingerprint(self.cfg) or state.get("protocol_sha256") != fingerprint(self.p) or type(state.get("step")) is not int or not 0 <= state["step"] <= self.cap:
            raise ValueError("Changed/historical/incomplete phase budget")
        replay = PhaseBudget(self.cfg, self.l_anchors, self.p, self.match_receipt)
        every = self.p["evaluation"]["every_successful_steps"]
        for i, row in enumerate(state["history"], 1):
            if not isinstance(row, dict) or set(row) != {"step", "paired_accuracy", "auroc"} or type(row["step"]) is not int or row["step"] != i*every or row["step"] > state["step"] or replay.stop_reason:
                raise ValueError("Skipped/out-of-order Validation or updates after phase stop")
            replay.step = row["step"]
            replay.observe_validation({**row, "evaluation_model": "ema", "evaluation_split": "validation", "validation_anchors": 400, "validation_pairs": 800})
        last = replay.history[-1]["step"] if replay.history else 0
        if state["step"]-last > every or (replay.stop_reason and state["step"] != replay.step):
            raise ValueError("Missing Validation or extra training after phase termination")
        replay.step = state["step"]
        if replay.state_dict() != state:
            raise ValueError("Phase patience/best/grace/receipt differs from replayed history")
        self.__dict__.update(replay.__dict__)
