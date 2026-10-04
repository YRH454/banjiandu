"""Replay-checked parent, matching and terminal Test receipts; no Torch import.

Hashes bind normal workflow records, not the trustworthiness of a private
operator or historical holdout declarations. Tensor verification lives in the
backend. Receipt metadata contains aggregate metrics only, never sample rows.
"""
from __future__ import annotations

import copy
import math

from fair_benchmark.budget import ComputeLedger
from fair_benchmark.evaluation import validate_metrics
from fair_benchmark.spec import fingerprint, is_sha256, positive_int
from .spec import validate_config, load_protocol, make_config


def signed(metadata):
    result = copy.deepcopy(metadata)
    result["receipt_sha256"] = fingerprint(metadata)
    return result


def _check_digest(receipt, fields, format_name):
    if not isinstance(receipt, dict) or set(receipt) != fields | {"receipt_sha256"} or receipt.get("format") != format_name:
        raise ValueError("Incomplete or historical phase receipt")
    if receipt["receipt_sha256"] != fingerprint({k: v for k, v in receipt.items() if k != "receipt_sha256"}):
        raise ValueError("Changed phase receipt SHA256")


def validate_phase_cost(cost, cfg, budget):
    ledger = ComputeLedger(); ledger.load_state_dict(cost)
    if cost["l_pair_draws"] != 32*budget.step or cost["u_pair_draws"] != (0 if cfg["budget_percent"] == 100 else 32*budget.step):
        raise ValueError("Phase sampling accounting differs")
    if cost["attempts"] != budget.step+cost["failed_attempts"] or any(cost[k] != 800*len(budget.history) for k in ("validation_ema_forward_pairs", "validation_student_forward_pairs")):
        raise ValueError("Phase retries or scheduled Validation were omitted")
    if cost["backward_pairs"] < 32*budget.step or cost["l_forward_pairs"] < 32*budget.step:
        raise ValueError("Phase supervised training operations were omitted")


def validate_terminal_validation(last, budget):
    if not budget.stop_reason or budget.evaluation_due or not budget.history or budget.history[-1]["step"] != budget.step:
        raise ValueError("Phase stop and final Validation are required")
    if not isinstance(last, dict) or set(last) != {"ema", "student"}:
        raise ValueError("Both terminal Validation diagnostics are required")
    for model in ("ema", "student"):
        validate_metrics(last[model], "validation", 400, budget.step, model=model)
    if any(last["ema"][k] != budget.history[-1][k] for k in ("paired_accuracy", "auroc")):
        raise ValueError("Terminal Validation differs from replayed stop history")


PARENT_FIELDS = {"format", "config", "budget", "compute", "last_validation", "resources", "l_anchors",
                 "input_identity_sha256", "initial_common_state_sha256", "simulation_only",
                 "backend_state_sha256", "common_full_state_sha256", "terminal_ema_sha256",
                 "test_attempted", "test_consumed", "source_sha256", "execution_identity"}


def validate_parent_receipt(receipt, branch_config=None, l_anchors=None, protocol=None):
    from .budget import PhaseBudget
    p = load_protocol() if protocol is None else protocol
    _check_digest(receipt, PARENT_FIELDS, "plateau_parent_v4")
    cfg = validate_config(receipt["config"], p)
    anchors = positive_int(receipt["l_anchors"], "parent L anchors")
    if cfg["stage"] != "bce" or cfg["method"] != "bce" or (l_anchors is not None and anchors != l_anchors):
        raise ValueError("Only same-cell BCE parents are accepted")
    if branch_config is not None:
        validate_config(branch_config, p)
        expected = make_config(branch_config["dataset"], branch_config["budget_percent"], "bce", branch_config["seed"], "bce", protocol=p)
        if branch_config["stage"] == "bce" or cfg != expected or branch_config["parent_run_id"] != cfg["run_id"]:
            raise ValueError("Parent belongs to another crop/budget/seed/stage")
    budget = PhaseBudget(cfg, anchors, p); budget.load_state_dict(receipt["budget"])
    if budget.stop_reason != "validation_plateau":
        raise ValueError("A safety cap is not a BCE plateau; automatic branching is forbidden")
    validate_terminal_validation(receipt["last_validation"], budget)
    validate_phase_cost(receipt["compute"], cfg, budget)
    if receipt["test_attempted"] is not False or receipt["test_consumed"] is not False or receipt["compute"]["test_forward_pairs"] or receipt["compute"]["test_seconds"] or any(v for k, v in receipt["compute"].items() if k.startswith("teacher_")):
        raise ValueError("Parent must be exported before any Test or module teacher use")
    if type(receipt["simulation_only"]) is not bool or any(not is_sha256(receipt[k]) for k in ("input_identity_sha256", "initial_common_state_sha256", "backend_state_sha256", "common_full_state_sha256", "terminal_ema_sha256")):
        raise ValueError("Parent initialization/input/full-state evidence is missing")
    if (not receipt["simulation_only"] and not is_sha256(receipt["source_sha256"])) or (receipt["source_sha256"] is not None and not is_sha256(receipt["source_sha256"])):
        raise ValueError("Real parent code-source evidence is required")
    execution = receipt["execution_identity"]
    if not isinstance(execution, dict) or set(execution) != {"hardware_sha256", "environment_sha256", "serial_uncontended_execution"} or any(v is not None and not is_sha256(v) for k, v in execution.items() if k.endswith("sha256")) or (execution["serial_uncontended_execution"] is not None and type(execution["serial_uncontended_execution"]) is not bool):
        raise ValueError("Invalid parent execution identity")
    return copy.deepcopy(receipt)


MATCH_FIELDS = {"format", "source_config", "source_budget", "parent_receipt_sha256", "input_identity_sha256",
                "l_anchors", "additional_steps", "terminal_validation", "terminal_ema_sha256",
                "simulation_only", "test_attempted"}


def validate_match_receipt(receipt, matched_config, l_anchors, protocol=None):
    from .budget import PhaseBudget
    p = load_protocol() if protocol is None else protocol
    validate_config(matched_config, p)
    _check_digest(receipt, MATCH_FIELDS, "plateau_match_v4")
    source = make_config(matched_config["dataset"], matched_config["budget_percent"], matched_config["match_method"], matched_config["seed"], protocol=p)
    if matched_config["stage"] != "matched_bce" or receipt["source_config"] != source or receipt["l_anchors"] != l_anchors or receipt["test_attempted"] is not False:
        raise ValueError("Matching target must come from the same module/cell/seed before Test")
    budget = PhaseBudget(source, l_anchors, p); budget.load_state_dict(receipt["source_budget"])
    validate_terminal_validation(receipt["terminal_validation"], budget)
    if type(receipt["additional_steps"]) is not int or receipt["additional_steps"] != budget.step or type(receipt["simulation_only"]) is not bool:
        raise ValueError("Matching target differs from source Validation termination")
    if any(not is_sha256(receipt[k]) for k in ("parent_receipt_sha256", "input_identity_sha256", "terminal_ema_sha256")):
        raise ValueError("Missing matching source evidence")
    return copy.deepcopy(receipt)


def validate_test_selection(selection, config, protocol, step, ema_hash=None, test_hash=None):
    from .budget import PhaseBudget
    fields = {"format", "config_sha256", "protocol_sha256", "checkpoint", "model", "step", "threshold",
              "validation_metrics_sha256", "ema_state_sha256", "test_contract_sha256", "budget_state",
              "parent_receipt_sha256", "adaptation_common_state_sha256"}
    validate_config(config, protocol)
    if not isinstance(selection, dict) or set(selection) != fields or selection["format"] != "plateau_test_selection_v4" or selection["config_sha256"] != fingerprint(config) or selection["protocol_sha256"] != fingerprint(protocol):
        raise ValueError("Changed/incomplete v4 terminal Test selection")
    if selection["checkpoint"] != "terminal_ema" or selection["model"] != "ema" or type(selection["step"]) is not int or selection["step"] != step:
        raise ValueError("Only the terminal EMA may access Test")
    state = selection["budget_state"]
    budget = PhaseBudget(config, state["l_anchors"], protocol, state["match_receipt"])
    budget.load_state_dict(state)
    if not budget.stop_reason or budget.evaluation_due or budget.step != step or (config["stage"] == "bce" and budget.stop_reason != "validation_plateau"):
        raise ValueError("Training/Validation not complete, or BCE reached only its cap")
    for name in ("validation_metrics_sha256", "ema_state_sha256", "test_contract_sha256"):
        if not is_sha256(selection[name]):
            raise ValueError("Frozen Test source proof is missing")
    for name in ("parent_receipt_sha256", "adaptation_common_state_sha256"):
        if (config["stage"] == "bce" and selection[name] is not None) or (config["stage"] != "bce" and not is_sha256(selection[name])):
            raise ValueError("Frozen branch-parent evidence is invalid")
    if state["match_receipt"] is not None and state["match_receipt"]["parent_receipt_sha256"] != selection["parent_receipt_sha256"]:
        raise ValueError("Matching target and branch have different parents")
    if (ema_hash is not None and selection["ema_state_sha256"] != ema_hash) or (test_hash is not None and selection["test_contract_sha256"] != test_hash):
        raise ValueError("EMA or holdout changed after sealing")
    t = selection["threshold"]
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 <= t <= math.nextafter(1., math.inf):
        raise ValueError("Invalid frozen Validation threshold")
    return copy.deepcopy(selection)
