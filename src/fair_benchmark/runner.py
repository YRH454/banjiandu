"""Shared success accounting, checkpoint selection and complete resume controls.

The backend owns student/EMA/optimizer/scaler/RNG/algorithm state. Historical
checkpoints or policies cannot be quietly relabelled as this new experiment.
"""
from __future__ import annotations

import copy
import time

from .budget import BudgetController, ComputeLedger
from .evaluation import validate_metrics, validate_test_selection
from .spec import VERSION, fingerprint, is_sha256, load_protocol, validate_config


class FairTrainer:
    STATE_FORMAT = "fair_itm_full_v3"
    SELECTION_FORMAT = "fair_test_selection_v3"
    VERSION = VERSION
    load_protocol = staticmethod(load_protocol)
    validate_config = staticmethod(validate_config)
    validate_test_selection = staticmethod(validate_test_selection)

    def _new_budget(self):
        return BudgetController(self.config["policy"], self.protocol)

    def __init__(self, config, backend, provenance, protocol=None):
        self.protocol = self.load_protocol() if protocol is None else copy.deepcopy(protocol)
        self.config = self.validate_config(copy.deepcopy(config), self.protocol)
        if backend.cfg != self.config or provenance.get("inputs") != backend.input_identity:
            raise ValueError("Backend configuration/private inputs and recorded provenance differ")
        if type(backend.simulation_only) is not bool or not is_sha256(backend.initial_common_hash):
            raise ValueError("Explicit simulation status and common initialization proof are required")
        if (config["uses_pairusa"] and not is_sha256(backend.teacher_identity)) or (not config["uses_pairusa"] and backend.teacher_identity is not None):
            raise ValueError("Same-cell teacher identity must be recorded for USA methods only")
        self.backend = backend
        self.provenance = copy.deepcopy(provenance)
        self.budget = self._new_budget()
        self.cost = ComputeLedger()
        self.warmup_fingerprint = None
        self.last_validation = None
        self.test_selection, self.test_metrics = None, None
        self.test_attempted = False
        self._saved_peaks = {"peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None}
        self.cost.add(**backend.initial_cost)
        if backend.step != 0:
            raise ValueError("A new fair-v3 session must start from its zero-step state")

    def advance(self):
        if self.budget.stop_reason or self.budget.evaluation_due or self.test_selection is not None:
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
        validate_metrics(primary, "validation", 400, self.budget.step)
        validate_metrics(diagnostic, "validation", 400, self.budget.step, model="student")
        is_best = self.budget.observe_validation(primary)
        self.cost.add(validation_seconds=time.monotonic() - start, validation_ema_forward_pairs=800,
                      validation_student_forward_pairs=800)
        self.last_validation = copy.deepcopy(dict(ema=primary, student=diagnostic))
        return dict(ema=primary, student=diagnostic, validation_best_diagnostic=is_best,
                    selected_as_primary_checkpoint=False)

    def _require_training_complete(self):
        if not self.budget.stop_reason or self.budget.evaluation_due or self.budget.best_step is None or self.last_validation is None:
            raise ValueError("Training and final scheduled Validation are not complete")
        if self.config["policy"] == "fixed" and self.budget.step != self.config["target_steps"]:
            raise ValueError("A short adaptive run cannot be presented as fixed-budget completion")
        if self.backend.step != self.budget.step or self.last_validation["ema"]["step"] != self.budget.step:
            raise ValueError("Terminal backend and Validation accounting disagree")
        for model in ("ema", "student"):
            validate_metrics(self.last_validation[model], "validation", 400, self.budget.step, model=model)
        if any(self.last_validation["ema"][k] != self.budget.history[-1][k] for k in ("paired_accuracy", "auroc")):
            raise ValueError("Terminal Validation and registered history differ")

    def _validate_frozen_selection(self):
        self.validate_test_selection(self.test_selection, self.config, self.protocol, self.budget.step,
                                self.backend.ema_state_fingerprint(), self.backend.test_contract_sha256)
        if self.test_selection["budget_state"] != self.budget.state_dict() or self.test_selection["validation_metrics_sha256"] != fingerprint(self.last_validation["ema"]) or self.test_selection["threshold"] != self.last_validation["ema"]["threshold"]:
            raise ValueError("Frozen selection differs from terminal Validation/budget")

    def seal_for_test(self):
        self._require_training_complete()
        if self.test_selection is None:
            self.test_selection = dict(format=self.SELECTION_FORMAT, config_sha256=fingerprint(self.config),
                                       protocol_sha256=fingerprint(self.protocol), checkpoint="terminal_ema", model="ema",
                                       step=self.budget.step, threshold=self.last_validation["ema"]["threshold"],
                                       validation_metrics_sha256=fingerprint(self.last_validation["ema"]),
                                       ema_state_sha256=self.backend.ema_state_fingerprint(),
                                       test_contract_sha256=self.backend.test_contract_sha256,
                                       budget_state=self.budget.state_dict())
        self._validate_frozen_selection()
        return copy.deepcopy(self.test_selection)

    def evaluate_test(self, persist_intent=None):
        self._require_training_complete()
        if self.test_selection is None:
            raise ValueError("Seal the terminal EMA and Validation threshold before Test")
        if self.test_attempted:
            raise ValueError("Test intent/access already consumed; do not repeat or select by Test")
        self._validate_frozen_selection()
        if not self.backend.simulation_only and not callable(persist_intent):
            raise ValueError("Real Test needs a private durable intent journal and unique-run lock")
        self.test_attempted = True
        if persist_intent is not None:
            persist_intent(self.snapshot())  # Caller atomically journals BEFORE opening Test.
        start, before = time.monotonic(), self.backend.test_forward_pairs
        try:
            metrics = self.backend.evaluate_test(copy.deepcopy(self.test_selection))
            validate_metrics(metrics, "test", self.backend.test_anchors, self.budget.step,
                             threshold=self.test_selection["threshold"])
            self.test_metrics = copy.deepcopy(metrics)
        finally:
            self.cost.add(test_seconds=time.monotonic()-start,
                          test_forward_pairs=self.backend.test_forward_pairs-before)
        return copy.deepcopy(self.test_metrics)

    def _resource_usage(self):
        resources = copy.deepcopy(self.backend.resource_usage())
        for name, saved in self._saved_peaks.items():
            if saved is not None:
                resources[name] = max(saved, resources[name])
        return resources

    def snapshot(self):
        return dict(format=self.STATE_FORMAT, config_sha256=fingerprint(self.config),
                    protocol_sha256=fingerprint(self.protocol), provenance=copy.deepcopy(self.provenance),
                    budget=self.budget.state_dict(), compute=self.cost.state_dict(),
                    warmup_common_full_state_sha256=self.warmup_fingerprint,
                    backend=self.backend.snapshot(), last_validation=copy.deepcopy(self.last_validation),
                    test_selection=copy.deepcopy(self.test_selection), test_attempted=self.test_attempted,
                    test_metrics=copy.deepcopy(self.test_metrics), resources=self._resource_usage())

    def restore(self, payload):
        keys = set(self.snapshot())
        if set(payload) != keys or payload["format"] != self.STATE_FORMAT or payload["config_sha256"] != fingerprint(self.config) or payload["protocol_sha256"] != fingerprint(self.protocol) or payload["provenance"] != self.provenance:
            raise ValueError("Incomplete/historical/changed-source checkpoint cannot resume fair-v3")
        if self.test_selection is not None and payload["test_selection"] != self.test_selection:
            raise ValueError("A sealed Test selection cannot be replaced in this session")
        if self.test_attempted and not payload["test_attempted"]:
            raise ValueError("Restoring an older snapshot cannot refund consumed Test intent")
        if self.test_metrics is not None and payload["test_metrics"] != self.test_metrics:
            raise ValueError("A completed Test result cannot be discarded on resume")
        budget = self._new_budget()
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
        last = payload["last_validation"]
        if budget.history:
            if not isinstance(last, dict) or set(last) != {"ema", "student"}:
                raise ValueError("Last scheduled Validation metrics are missing")
            for model in ("ema", "student"):
                validate_metrics(last[model], "validation", 400, budget.history[-1]["step"], model=model)
            if any(last["ema"][name] != budget.history[-1][name] for name in ("paired_accuracy", "auroc")):
                raise ValueError("Last Validation and budget history differ")
        elif last is not None:
            raise ValueError("Unexpected Validation before the first milestone")
        selection, metrics, attempted = payload["test_selection"], payload["test_metrics"], payload["test_attempted"]
        if type(attempted) is not bool or (selection is None and (attempted or metrics is not None)):
            raise ValueError("Invalid Test intent state")
        consumed, forwards = payload["backend"]["test_consumed"], payload["backend"]["test_forward_pairs"]
        if type(consumed) is not bool or type(forwards) is not int or not 0 <= forwards <= 2*self.backend.test_anchors or (not attempted and (consumed or forwards or cost.values["test_seconds"])) or (forwards and not consumed) or forwards != cost.values["test_forward_pairs"]:
            raise ValueError("Backend Test access and durable intent/accounting disagree")
        if selection is not None:
            self.validate_test_selection(selection, self.config, self.protocol, budget.step, test_hash=self.backend.test_contract_sha256)
            if selection["budget_state"] != budget.state_dict() or selection["validation_metrics_sha256"] != fingerprint(last["ema"]) or selection["threshold"] != last["ema"]["threshold"]:
                raise ValueError("Selection does not match the terminal Validation and budget")
        if metrics is not None:
            if not attempted or not consumed or forwards != 2*self.backend.test_anchors:
                raise ValueError("Test result without complete inference/durable intent")
            validate_metrics(metrics, "test", self.backend.test_anchors, budget.step, threshold=selection["threshold"])
        resources = payload["resources"]
        current_resources = self.backend.resource_usage()
        for name in ("student_total_parameters", "student_trainable_parameters", "common_trainable_parameters", "method_specific_trainable_parameters", "teacher"):
            if resources[name] != current_resources[name]:
                raise ValueError("Resource topology/teacher changed on resume")
        for name in self._saved_peaks:
            if self.backend.simulation_only:
                if resources[name] is not None:
                    raise ValueError("CPU simulations cannot claim real CUDA peaks")
            elif type(resources[name]) is not int or resources[name] <= 0:
                raise ValueError("Measured CUDA peaks are required")
        self.backend.restore(payload["backend"])
        if selection is not None and selection["ema_state_sha256"] != self.backend.ema_state_fingerprint():
            raise ValueError("Restored EMA differs from its frozen Test selection")
        if self.backend.test_forward_pairs != cost.values["test_forward_pairs"]:
            raise ValueError("Test forward accounting differs on resume")
        # A fresh worker's reconstruction is real cost. Charge it once, but do
        # not charge the physically reused upstream teacher a second time.
        initial = self.backend.initial_cost
        cost.add(resume_setup_seconds=initial.get("setup_seconds", 0.),
                 bank_initialization_seconds=initial.get("bank_initialization_seconds", 0.),
                 l_forward_pairs=initial.get("l_forward_pairs", 0))
        self.budget, self.cost, self.warmup_fingerprint = budget, cost, warmup
        self.last_validation, self.test_selection = copy.deepcopy(last), copy.deepcopy(selection)
        self.test_attempted, self.test_metrics = attempted, copy.deepcopy(metrics)
        self._saved_peaks = {name: resources[name] for name in self._saved_peaks}

    def result(self):
        self._require_training_complete()
        tested = self.test_metrics is not None
        return dict(version=self.VERSION, run_id=self.config["run_id"],
                    state="completed" if tested else ("test_intent_pending_or_failed" if self.test_attempted else "awaiting_test"),
                    dataset=self.config["dataset"], budget=self.config["budget"],
                    method=self.config["method"], seed=self.config["seed"], policy=self.config["policy"],
                    regime=self.config["regime"], roles=copy.deepcopy(self.config["roles"]),
                    successful_steps=self.budget.step, target_steps=self.config["target_steps"],
                    stop_reason=self.budget.stop_reason, validations=len(self.budget.history),
                    validation_best_step=self.budget.best_step,
                    validation_best_metrics=dict(paired_accuracy=self.budget.best_key[0], auroc=self.budget.best_key[1]),
                    terminal_validation=copy.deepcopy(self.last_validation["ema"]),
                    student_terminal_validation=copy.deepcopy(self.last_validation["student"]),
                    primary_checkpoint="terminal_ema", primary_step=self.budget.step, primary_split="independent_test",
                    primary_metrics=copy.deepcopy(self.test_metrics),
                    evaluation_model="ema", test_evaluated=tested, test_attempted=self.test_attempted,
                    test_selection=copy.deepcopy(self.test_selection), test_anchors=self.backend.test_anchors,
                    test_contract_sha256=self.backend.test_contract_sha256,
                    holdout_identity=copy.deepcopy(self.backend.holdout_identity),
                    simulation_only=self.backend.simulation_only,
                    teacher_targets_sha256=self.backend.teacher_identity,
                    initial_common_state_sha256=self.backend.initial_common_hash,
                    warmup_common_full_state_sha256=self.warmup_fingerprint,
                    protocol_sha256=fingerprint(self.protocol), config_sha256=fingerprint(self.config),
                    provenance=copy.deepcopy(self.provenance), compute=self.cost.state_dict(), resources=self._resource_usage(),
                    tuning=copy.deepcopy(self.protocol["tuning"]),
                    budget_state=self.budget.state_dict(), equal_compute_claim=False)
