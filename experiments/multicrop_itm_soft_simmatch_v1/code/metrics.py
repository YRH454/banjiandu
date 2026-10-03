"""Validation-only threshold selection and matching/multilabel evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _divide(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def select_threshold(labels, probabilities) -> float:
    """Maximize validation macro-F1; ties prefer the threshold nearest 0.5."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all():
        raise ValueError("Invalid validation predictions")
    values = np.unique(p)
    candidates = np.unique(np.concatenate(([0.0, 0.5, np.nextafter(values[-1], np.inf)], values)))
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    positives_below = np.concatenate(([0], np.cumsum(ys)))
    cut = np.searchsorted(ps, candidates, side="left")
    fn = positives_below[cut].astype(float)
    tp = float(y.sum()) - fn
    tn = cut.astype(float) - fn
    fp = len(y) - cut - tp
    f1p = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    f1n = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn), where=(2 * tn + fp + fn) > 0)
    macro = (f1p + f1n) / 2
    best = np.flatnonzero(np.isclose(macro, macro.max(), rtol=0, atol=1e-12))
    idx = min(best, key=lambda i: (abs(candidates[i] - 0.5), candidates[i]))
    return float(candidates[idx])


def binary_metrics(labels, probabilities, threshold: float = 0.5) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if not len(y) or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid evaluation arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Evaluation labels must be public binary labels")
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    f1p = _divide(2 * tp, 2 * tp + fp + fn)
    f1n = _divide(2 * tn, 2 * tn + fp + fn)
    ece = 0.0
    for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        chosen = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if chosen.any():
            ece += float(chosen.mean() * abs(p[chosen].mean() - y[chosen].mean()))
    return {
        "n": len(y), "threshold": float(threshold), "accuracy": _divide(tp + tn, len(y)),
        "precision": _divide(tp, tp + fp), "recall": _divide(tp, tp + fn),
        "negative_recall": _divide(tn, tn + fp), "f1": f1p, "macro_f1": (f1p + f1n) / 2,
        "balanced_accuracy": (_divide(tp, tp + fn) + _divide(tn, tn + fp)) / 2,
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
        "brier": float(np.mean((p - y) ** 2)), "ece_10": ece,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def multilabel_metrics(targets, probabilities, classes, thresholds=None) -> dict:
    y, p = np.asarray(targets, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 2 or y.shape != p.shape or y.shape[1] != len(classes):
        raise ValueError("Invalid disease evaluation arrays")
    thresholds = list(thresholds) if thresholds is not None else [0.5] * len(classes)
    per_class = {name: binary_metrics(y[:, j], p[:, j], thresholds[j]) for j, name in enumerate(classes)}
    prediction = p >= np.asarray(thresholds)[None, :]
    tp = int(np.sum(prediction & (y == 1))); fp = int(np.sum(prediction & (y == 0)))
    fn = int(np.sum(~prediction & (y == 1)))
    aps = [v["average_precision"] for v in per_class.values() if v["average_precision"] is not None]
    return {"unique_images": len(y), "thresholds": thresholds,
            "micro_f1": _divide(2 * tp, 2 * tp + fp + fn),
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "mean_average_precision": float(np.mean(aps)) if aps else None,
            "per_class": per_class}
