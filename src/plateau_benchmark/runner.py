"""Plateau parents, independent adaptation budgets and matched-BCE replay.

The caller owns separate workers, private atomic checkpoint/receipt persistence,
group target freezing, real GPU admission and the durable one-shot Test journal.
No scheduler, SSH, training launch or private inputs are embedded here.
"""
from __future__ import annotations

import copy

from fair_benchmark.budget import ComputeLedger
from fair_benchmark.runner import FairTrainer
from fair_benchmark.spec import fingerprint, is_sha256
from fair_benchmark.torch_backend import tree_digest
from .budget import PhaseBudget
from .contracts import signed, validate_match_receipt, validate_parent_receipt, validate_test_selection
from .spec import VERSION, load_protocol, make_config, source_fingerprint, validate_config
from .torch_backend import PlateauBackend


def train_until_stop(session, persist_checkpoint=None, log_record=None):
    """Explicit private-worker API; callbacks persist outside the public repo.

    Calling this function starts training; the read-only CLI never calls it.
    Separate per-run workers and trusted real admission remain caller duties.
    """
    if not isinstance(session, PlateauTrainer):
        raise ValueError("A registered PlateauTrainer is required")
    if not session.backend.simulation_only and (not callable(persist_checkpoint) or not callable(log_record)):
        raise ValueError("Real training needs private atomic checkpoints and step/Validation logs")
    for callback in (persist_checkpoint, log_record):
        if callback is not None and not callable(callback):
            raise ValueError("Persistence callbacks must be callable")
    if session.test_selection is not None or session.test_attempted:
        raise ValueError("Training cannot resume after Test selection/access")
    if persist_checkpoint is not None:
        persist_checkpoint(session.snapshot())
    while not session.budget.stop_reason:
        if session.budget.evaluation_due:
            record = session.evaluate()
            if log_record is not None:
                log_record({"kind": "validation", **record})
        else:
            record = session.advance()
            if log_record is not None:
                log_record({"kind": "training", **record})
        if persist_checkpoint is not None and (session.budget.stop_reason or session.budget.step % session.protocol["training"]["checkpoint_every"] == 0):
            persist_checkpoint(session.snapshot())
    session._require_training_complete()
    return session.result()


class PlateauTrainer(FairTrainer):
    STATE_FORMAT = "plateau_itm_full_v4"
    SELECTION_FORMAT = "plateau_test_selection_v4"
    VERSION = VERSION
    load_protocol = staticmethod(load_protocol)
    validate_config = staticmethod(validate_config)
    validate_test_selection = staticmethod(validate_test_selection)

    def __init__(self, config, backend, provenance, protocol=None, parent=None, match_receipt=None):
        if not isinstance(backend, PlateauBackend):
            raise ValueError("PlateauTrainer requires the phase-aware Torch backend")
        p = load_protocol() if protocol is None else protocol
        validate_config(config, p)
        source = provenance.get("source_sha256")
        if not backend.simulation_only and (not is_sha256(source) or source != source_fingerprint()["source_sha256"]):
            raise ValueError("Real v4 training requires the current public code fingerprint in provenance")
        self.parent_receipt, self.match_receipt, self._parent_export = None, copy.deepcopy(match_receipt), None
        if config["stage"] == "bce":
            if parent is not None or match_receipt is not None:
                raise ValueError("A BCE parent must start cold without a branch receipt")
        else:
            if not isinstance(parent, dict) or "receipt" not in parent:
                raise ValueError("All second-stage methods require the same private BCE parent bundle")
            self.parent_receipt = validate_parent_receipt(parent["receipt"], config, len(backend.data.l), p)
            if self.parent_receipt["source_sha256"] != source:
                raise ValueError("Parent and branch code fingerprints differ")
            if config["stage"] == "matched_bce":
                validate_match_receipt(match_receipt, config, len(backend.data.l), p)
                if any(match_receipt[k] != self.parent_receipt[k] for k in ("input_identity_sha256", "simulation_only")) or match_receipt["parent_receipt_sha256"] != self.parent_receipt["receipt_sha256"]:
                    raise ValueError("Matching receipt and imported parent differ")
            elif match_receipt is not None:
                raise ValueError("Adaptive branches do not accept a matching target")
            # Construction/resume is not free; import is part of measured setup.
            backend.initial_cost["setup_seconds"] += backend.import_parent(parent)
        super().__init__(config, backend, provenance, p)

    def _new_budget(self):
        return PhaseBudget(self.config, len(self.backend.data.l), self.protocol, self.match_receipt)

    def evaluate(self):
        record = super().evaluate()
        if self.config["stage"] == "bce" and self.budget.stop_reason == "validation_plateau":
            self.export_parent()  # Freeze the complete stop state before another worker/Test.
        record.update(stop_reason=self.budget.stop_reason, phase_step=self.budget.step,
                      plateau_detected=self.budget.stop_reason == "validation_plateau")
        return record

    def export_parent(self):
        self._require_training_complete()
        if self.config["stage"] != "bce" or self.budget.stop_reason != "validation_plateau" or self.test_attempted or self.backend._test_consumed:
            raise ValueError("Only a BCE Validation plateau before Test may create branches")
        if self._parent_export is None:
            payload = self.backend.snapshot()
            receipt = signed(dict(format="plateau_parent_v4", config=copy.deepcopy(self.config),
                                  budget=self.budget.state_dict(), compute=self.cost.state_dict(),
                                  last_validation=copy.deepcopy(self.last_validation), resources=self._resource_usage(),
                                  l_anchors=len(self.backend.data.l), input_identity_sha256=fingerprint(self.backend.input_identity),
                                  initial_common_state_sha256=self.backend.initial_common_hash,
                                  simulation_only=self.backend.simulation_only, backend_state_sha256=tree_digest(payload),
                                  common_full_state_sha256=self.backend.common_full_state_fingerprint(),
                                  terminal_ema_sha256=self.backend.ema_state_fingerprint(),
                                  source_sha256=self.provenance.get("source_sha256"),
                                  execution_identity={k: self.provenance.get(k) for k in ("hardware_sha256", "environment_sha256", "serial_uncontended_execution")},
                                  test_attempted=False, test_consumed=False))
            validate_parent_receipt(receipt, protocol=self.protocol)
            self._parent_export = dict(receipt=receipt, backend=payload)
        return copy.deepcopy(self._parent_export)

    def export_match_receipt(self):
        self._require_training_complete()
        if self.config["stage"] != "adaptation" or self.config["method"] == "bce" or self.test_attempted or self.backend._test_consumed:
            raise ValueError("A non-BCE adaptive branch must stop before Test to define its matched control")
        receipt = signed(dict(format="plateau_match_v4", source_config=copy.deepcopy(self.config),
                              source_budget=self.budget.state_dict(), parent_receipt_sha256=self.parent_receipt["receipt_sha256"],
                              input_identity_sha256=fingerprint(self.backend.input_identity), l_anchors=len(self.backend.data.l),
                              additional_steps=self.budget.step, terminal_validation=copy.deepcopy(self.last_validation),
                              terminal_ema_sha256=self.backend.ema_state_fingerprint(),
                              simulation_only=self.backend.simulation_only, test_attempted=False))
        cfg = make_config(self.config["dataset"], self.config["budget_percent"], "bce", self.config["seed"],
                          "matched_bce", self.config["method"], self.protocol)
        validate_match_receipt(receipt, cfg, len(self.backend.data.l), self.protocol)
        return receipt

    def _validate_frozen_selection(self):
        super()._validate_frozen_selection()
        if self.test_selection["parent_receipt_sha256"] != self.backend.parent_receipt_sha256 or self.test_selection["adaptation_common_state_sha256"] != self.backend.adaptation_common_hash:
            raise ValueError("Frozen selection has a different parent/phase-zero state")

    def seal_for_test(self):
        self._require_training_complete()
        if self.config["stage"] == "bce":
            if self.budget.stop_reason != "validation_plateau":
                raise ValueError("A BCE cap is not a plateau reference; review the protocol before proceeding")
            if self._parent_export is None:
                self.export_parent()
        if self.test_selection is None:
            self.test_selection = dict(format=self.SELECTION_FORMAT, config_sha256=fingerprint(self.config),
                                       protocol_sha256=fingerprint(self.protocol), checkpoint="terminal_ema", model="ema",
                                       step=self.budget.step, threshold=self.last_validation["ema"]["threshold"],
                                       validation_metrics_sha256=fingerprint(self.last_validation["ema"]),
                                       ema_state_sha256=self.backend.ema_state_fingerprint(),
                                       test_contract_sha256=self.backend.test_contract_sha256,
                                       budget_state=self.budget.state_dict(),
                                       parent_receipt_sha256=self.backend.parent_receipt_sha256,
                                       adaptation_common_state_sha256=self.backend.adaptation_common_hash)
        self._validate_frozen_selection()
        return copy.deepcopy(self.test_selection)

    def snapshot(self):
        return {**super().snapshot(), "parent_receipt": copy.deepcopy(self.parent_receipt),
                "match_receipt": copy.deepcopy(self.match_receipt), "exported_parent": copy.deepcopy(self._parent_export)}

    def restore(self, payload):
        if payload.get("parent_receipt") != self.parent_receipt or payload.get("match_receipt") != self.match_receipt:
            raise ValueError("Checkpoint parent or matched target changed")
        exported = payload.get("exported_parent")
        if self._parent_export is not None and (exported is None or exported["receipt"] != self._parent_export["receipt"]):
            raise ValueError("A frozen BCE parent cannot be refunded or replaced")
        if exported is not None:
            if self.config["stage"] != "bce" or set(exported) != {"receipt", "backend"}:
                raise ValueError("Unexpected exported BCE parent")
            receipt = validate_parent_receipt(exported["receipt"], protocol=self.protocol)
            if receipt["config"] != self.config or receipt["budget"] != payload["budget"] or receipt["last_validation"] != payload["last_validation"] or tree_digest(exported["backend"]) != receipt["backend_state_sha256"] or receipt["terminal_ema_sha256"] != tree_digest(dict(ema=payload["backend"]["ema"], config_sha256=fingerprint(self.config), input_identity_sha256=fingerprint(self.backend.input_identity))):
                raise ValueError("Exported parent and resumed plateau state differ")
        super().restore(payload)
        if self.test_selection is not None:
            self._validate_frozen_selection()
        self._parent_export = copy.deepcopy(exported)

    def result(self):
        row = super().result()
        upstream = ComputeLedger().state_dict() if self.parent_receipt is None else copy.deepcopy(self.parent_receipt["compute"])
        total = ComputeLedger(); total.load_state_dict(upstream); total.add(**self.cost.state_dict())
        parent_steps = 0 if self.parent_receipt is None else self.parent_receipt["budget"]["step"]
        row.update(stage=self.config["stage"], match_method=self.config["match_method"],
                   parent_run_id=self.config["parent_run_id"], parent_receipt=copy.deepcopy(self.parent_receipt),
                   reference_receipt=None if self._parent_export is None else copy.deepcopy(self._parent_export["receipt"]),
                   match_receipt=copy.deepcopy(self.match_receipt),
                   adaptation_common_state_sha256=self.backend.adaptation_common_hash,
                   additional_steps=0 if self.config["stage"] == "bce" else self.budget.step,
                   total_student_steps=parent_steps+self.budget.step,
                   phase_compute=self.cost.state_dict(), upstream_bce_compute=upstream, compute=total.state_dict(),
                   parent_resources=None if self.parent_receipt is None else copy.deepcopy(self.parent_receipt["resources"]),
                   plateau_detected=self.budget.stop_reason == "validation_plateau",
                   capped_not_plateau=self.budget.stop_reason == "phase_cap_exhausted",
                   equal_steps_claim=False)
        if self.config["stage"] == "bce" and self.budget.stop_reason == "phase_cap_exhausted":
            row["state"] = "bce_cap_requires_review"
        return row
