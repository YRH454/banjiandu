"""Shared success accounting, checkpoint selection and complete resume controls.

The backend owns student/EMA/optimizer/scaler/RNG/algorithm state. Historical
checkpoints or policies cannot be quietly relabelled as this new experiment.
"""
from __future__ import annotations

import copy
import time

from .budget import BudgetController, ComputeLedger
from .spec import fingerprint, is_sha256, load_protocol, validate_config


class FairTrainer:
    def __init__(self, config, backend, provenance, protocol=None):
        self.protocol = load_protocol() if protocol is None else copy.deepcopy(protocol)
        self.config = validate_config(copy.deepcopy(config), self.protocol)
        if backend.cfg != self.config or provenance.get("inputs") != backend.input_identity:
            raise ValueError("Backend configuration/private inputs and recorded provenance differ")
        if type(backend.simulation_only) is not bool or not is_sha256(backend.initial_common_hash):
            raise ValueError("Explicit simulation status and common initialization proof are required")
        if (config["uses_pairusa"] and not is_sha256(backend.teacher_identity)) or (not config["uses_pairusa"] and backend.teacher_identity is not None):
            raise ValueError("Same-cell teacher identity must be recorded for USA methods only")
        self.backend = backend
        self.provenance = copy.deepcopy(provenance)
        self.budget = BudgetController(config["policy"], self.protocol)
        self.cost = ComputeLedger()
        self.warmup_fingerprint = None
        self.cost.add(**backend.initial_cost)
        if backend.step != 0:
            raise ValueError("A new fair-v2 session must start from its zero-step state")

    def advance(self):
        if self.budget.stop_reason or self.budget.evaluation_due:
            raise ValueError("Evaluate the pending common milestone or finish this session first")
        record = self.backend.train_step()
        if record.get("committed") is not True or record.get("step") != self.budget.step + 1 or self.backend.step != record["step"]:
            raise ValueError("Only a confirmed single successful optimizer update consumes budget")
        self.cost.add(**record["cost"])
        self.budget.commit_success(record["step"])
        if self.config["shared_bce_steps"] and self.budget.step == self.config["shared_bce_steps"]:
            self.warmup_fingerprint = self.backend.common_full_state_fingerprint()
        return record

    def evaluate(self):
        if not self.budget.evaluation_due:
            raise ValueError("Off-schedule validation would change selection opportunities")
        start = time.monotonic()
        primary, diagnostic = self.backend.evaluate()
        is_best = self.budget.observe_validation(primary)
        self.cost.add(validation_seconds=time.monotonic() - start)
        return dict(ema=primary, student=diagnostic, selected_as_best=is_best)

    def snapshot(self):
        return dict(format="fair_itm_full_v2", config_sha256=fingerprint(self.config),
                    protocol_sha256=fingerprint(self.protocol), provenance=copy.deepcopy(self.provenance),
                    budget=self.budget.state_dict(), compute=self.cost.state_dict(),
                    warmup_common_full_state_sha256=self.warmup_fingerprint,
                    backend=self.backend.snapshot())

    def restore(self, payload):
        keys = {"format", "config_sha256", "protocol_sha256", "provenance", "budget", "compute", "warmup_common_full_state_sha256", "backend"}
        if set(payload) != keys or payload["format"] != "fair_itm_full_v2" or payload["config_sha256"] != fingerprint(self.config) or payload["protocol_sha256"] != fingerprint(self.protocol) or payload["provenance"] != self.provenance:
            raise ValueError("Incomplete/historical/changed-source checkpoint cannot resume fair-v2")
        budget = BudgetController(self.config["policy"], self.protocol)
        budget.load_state_dict(payload["budget"])
        cost = ComputeLedger()
        cost.load_state_dict(payload["compute"])
        if payload["backend"]["step"] != budget.step:
            raise ValueError("Backend and successful-update accounting differ")
        warmup = payload["warmup_common_full_state_sha256"]
        if self.config["shared_bce_steps"] and budget.step >= self.config["shared_bce_steps"]:
            if not is_sha256(warmup):
                raise ValueError("Controlled warmup endpoint proof is missing")
        elif warmup is not None:
            raise ValueError("Unexpected warmup endpoint")
        self.backend.restore(payload["backend"])
        self.budget, self.cost, self.warmup_fingerprint = budget, cost, warmup

    def result(self):
        if not self.budget.stop_reason or self.budget.evaluation_due or self.budget.best_step is None:
            raise ValueError("Training is not complete under its registered policy")
        if self.config["policy"] == "fixed" and self.budget.step != self.config["target_steps"]:
            raise ValueError("A short adaptive run cannot be presented as fixed-budget completion")
        return dict(version="pair_itm_fair_v2", run_id=self.config["run_id"], state="completed",
                    dataset=self.config["dataset"], budget=self.config["budget"],
                    method=self.config["method"], seed=self.config["seed"], policy=self.config["policy"],
                    successful_steps=self.budget.step, target_steps=self.config["target_steps"],
                    stop_reason=self.budget.stop_reason, validations=len(self.budget.history),
                    best_step=self.budget.best_step,
                    best_metrics=dict(paired_accuracy=self.budget.best_key[0], auroc=self.budget.best_key[1]),
                    evaluation_model="ema", test_evaluated=False,
                    simulation_only=self.backend.simulation_only,
                    teacher_targets_sha256=self.backend.teacher_identity,
                    initial_common_state_sha256=self.backend.initial_common_hash,
                    warmup_common_full_state_sha256=self.warmup_fingerprint,
                    protocol_sha256=fingerprint(self.protocol), config_sha256=fingerprint(self.config),
                    provenance=copy.deepcopy(self.provenance), compute=self.cost.state_dict(),
                    budget_state=self.budget.state_dict(), equal_compute_claim=False)
