"""Finite aggregate checks and immutable terminal-selection contracts; no GPU."""
from __future__ import annotations

import copy
import math

from .budget import BudgetController, ComputeLedger
from .spec import fingerprint, is_sha256, positive_int


def validate_binary_metrics(metrics, pairs, threshold=None):
    positive_int(pairs, "evaluation pairs")
    if metrics["n"] != pairs or type(metrics["n"]) is not int:
        raise ValueError("Evaluation pair count differs")
    value = metrics["threshold"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= math.nextafter(1., math.inf):
        raise ValueError("Threshold must be a finite registered probability boundary")
    if threshold is not None and value != threshold:
        raise ValueError("Test threshold differs from the frozen Validation threshold")
    cm = metrics["confusion_matrix"]
    if set(cm) != {"tn", "fp", "fn", "tp"} or any(type(v) is not int or v < 0 for v in cm.values()) or sum(cm.values()) != pairs:
        raise ValueError("Invalid aggregate confusion counts")
    if cm["tp"] + cm["fn"] != pairs // 2 or cm["tn"] + cm["fp"] != pairs // 2 or pairs % 2:
        raise ValueError("Evaluation must preserve one positive/negative pair per anchor")
    def divide(a, b):
        return a / b if b else 0.
    expected = dict(accuracy=(cm["tp"]+cm["tn"])/pairs,
                    macro_f1=.5*(divide(2*cm["tp"], 2*cm["tp"]+cm["fp"]+cm["fn"])
                                  + divide(2*cm["tn"], 2*cm["tn"]+cm["fp"]+cm["fn"])))
    for name in ("accuracy", "macro_f1", "auroc"):
        v = metrics[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError("Evaluation metrics must be finite probabilities")
        if name in expected and not math.isclose(v, expected[name], rel_tol=0, abs_tol=1e-9):
            raise ValueError("Metrics and confusion counts disagree")
    fingerprint(metrics)  # Reject NaN even in additional diagnostic fields.
    return metrics


def validate_metrics(metrics, split, anchors, step, model="ema", threshold=None):
    positive_int(anchors, "evaluation anchors")
    if metrics["evaluation_model"] != model or metrics["evaluation_split"] != split or metrics["step"] != step:
        raise ValueError("Evaluation model/split/step differs")
    if metrics[f"{split}_anchors"] != anchors or metrics[f"{split}_pairs"] != 2*anchors:
        raise ValueError("Evaluation membership/count differs")
    paired = metrics["paired_accuracy"]
    if isinstance(paired, bool) or not isinstance(paired, (int, float)) or not math.isfinite(paired) or not 0 <= paired <= 1:
        raise ValueError("Paired accuracy must be finite")
    if not math.isclose(paired*anchors, round(paired*anchors), rel_tol=0, abs_tol=1e-4+anchors*1e-7):
        raise ValueError("Paired accuracy does not correspond to an integer anchor count")
    validate_binary_metrics(metrics, 2*anchors, threshold)
    validate_binary_metrics(metrics["threshold_0_5"], 2*anchors, .5)
    if metrics["threshold_0_5"]["auroc"] != metrics["auroc"]:
        raise ValueError("Threshold-free AUROC differs across threshold reports")
    return metrics


def validate_test_selection(selection, config, protocol, step, ema_hash=None, test_hash=None):
    fields = {"format", "config_sha256", "protocol_sha256", "checkpoint", "model", "step", "threshold",
              "validation_metrics_sha256", "ema_state_sha256", "test_contract_sha256", "budget_state"}
    if not isinstance(selection, dict) or set(selection) != fields or selection["format"] != "fair_test_selection_v3" or selection["config_sha256"] != fingerprint(config) or selection["protocol_sha256"] != fingerprint(protocol):
        raise ValueError("Changed or incomplete terminal selection")
    if selection["checkpoint"] != "terminal_ema" or selection["model"] != "ema" or selection["step"] != step:
        raise ValueError("Only the terminal EMA may access Test")
    budget = BudgetController(config["policy"], protocol)
    budget.load_state_dict(selection["budget_state"])
    if not budget.stop_reason or budget.evaluation_due or budget.step != step:
        raise ValueError("Training and final scheduled Validation must complete before Test")
    for name in ("validation_metrics_sha256", "ema_state_sha256", "test_contract_sha256"):
        if not is_sha256(selection[name]):
            raise ValueError("Selection source proof is missing")
    if (ema_hash is not None and selection["ema_state_sha256"] != ema_hash) or (test_hash is not None and selection["test_contract_sha256"] != test_hash):
        raise ValueError("EMA/Test changed after terminal selection was frozen")
    threshold = selection["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1., math.inf):
        raise ValueError("Invalid frozen threshold")
    return copy.deepcopy(selection)


def validate_teacher_accounting(accounting, l_anchors, protocol):
    """One complete L epoch per Validation; include both descriptor/target views."""
    positive_int(l_anchors, "teacher L anchors")
    ledger = ComputeLedger()
    ledger.add(**{k: v for k, v in accounting.items() if k.startswith("teacher_")})
    epochs = positive_int(accounting["teacher_validation_calls"], "teacher Validation calls")
    if l_anchors < 16 or epochs > protocol["ablation"]["teacher_max_epochs"] or accounting["teacher_upstream_seconds"] <= 0:
        raise ValueError("Teacher epoch/data/time accounting differs")
    expected = dict(teacher_successful_updates=math.ceil(l_anchors/16)*epochs,
                    teacher_backward_pairs=2*l_anchors*epochs,
                    teacher_l_forward_pairs=2*l_anchors*epochs+4*l_anchors,
                    teacher_validation_forward_pairs=800*epochs,
                    teacher_descriptor_pairs=4*l_anchors+800)
    if any(accounting[k] != v for k, v in expected.items()):
        raise ValueError("Teacher operation counts do not cover complete L epochs and target preparation")


def validate_teacher_history(history, protocol):
    settings = protocol["ablation"]
    if not isinstance(history, list) or not 1 <= len(history) <= settings["teacher_max_epochs"]:
        raise ValueError("Incomplete teacher Validation history")
    best, stale = -math.inf, 0
    for epoch, row in enumerate(history, 1):
        if not isinstance(row, dict) or set(row) != {"epoch", "validation_auroc"} or type(row["epoch"]) is not int or row["epoch"] != epoch:
            raise ValueError("Teacher Validation epochs are out of order")
        value = row["validation_auroc"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Teacher Validation AUROC must be finite")
        if value > best:
            best, stale = value, 0
        else:
            stale += 1
        if stale >= settings["teacher_patience"] and epoch != len(history):
            raise ValueError("Teacher continued beyond its registered early stop")
    if len(history) < settings["teacher_max_epochs"] and stale < settings["teacher_patience"]:
        raise ValueError("Teacher ended before the registered cap/early stop")
