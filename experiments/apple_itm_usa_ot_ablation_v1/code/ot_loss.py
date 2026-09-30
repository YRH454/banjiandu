# -*- coding: utf-8 -*-
"""Balanced entropic OT pseudo-targets for the registered ITM ablation.

This module has no model, dataset, or hidden U-source-label dependency.
The input CLS vectors are the current student's original fused CLS vectors.
"""

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


class OTInputError(ValueError):
    """An invalid numerical/input condition; no optimizer step is permitted."""

    def __init__(self, message: str, diagnostics: Dict[str, Any]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


class OTConvergenceError(RuntimeError):
    """The registered iteration limit was reached without both constraints."""

    def __init__(self, diagnostics: Dict[str, Any]):
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "Sinkhorn did not converge after {iterations} iterations: "
            "row_residual={row_residual:.9g}, "
            "col_residual={col_residual:.9g}, tolerance={tolerance:.9g}".format(
                **self.diagnostics
            )
        )


def ot_weight(step: int) -> float:
    """Weight for a 1-based successful update, independent of retry count."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if step <= 100:
        return 0.0
    if step >= 200:
        return 0.1
    return 0.05 * (1.0 - math.cos(math.pi * (step - 100) / 100.0))


def _fail(message: str, diagnostics: Dict[str, Any]) -> None:
    diagnostics.update(converged=False, invalid_input=True, error=message)
    raise OTInputError(message, diagnostics)


def _newton_dual_update(log_kernel: torch.Tensor, log_v: torch.Tensor):
    """One FP64 Newton update of the same balanced entropic OT dual.

    Eliminate row potentials analytically. With R=softmax(logK+v),
    phi(v)=mean(logsumexp(logK+v))-mean(v), grad=R.mean(0)-1/m,
    Hess=diag(R.mean(0))-R.T@R/n. Fix v[-1]=0 to remove the gauge.
    A backtracking Armijo search changes only the solver step, not epsilon,
    transport marginals, or the underlying objective. No extra Sinkhorn
    sweeps or hidden optimizer updates occur in this helper.
    """
    n_queries, n_anchors = log_kernel.shape
    v = log_v - log_v[-1]
    conditional = torch.log_softmax(log_kernel + v.unsqueeze(0), dim=1).exp()
    row_plan = conditional / n_queries
    column_mass = row_plan.sum(dim=0)
    gradient = column_mass - 1.0 / n_anchors
    hessian = torch.diag(column_mass) - conditional.T @ row_plan
    delta = torch.cat((
        torch.linalg.solve(hessian[:-1, :-1], -gradient[:-1]),
        torch.zeros(1, device=log_kernel.device, dtype=torch.float64),
    ))
    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError("OT Newton direction is non-finite")
    objective_before = torch.logsumexp(log_kernel + v.unsqueeze(0), dim=1).mean() - v.mean()
    slope = torch.dot(gradient, delta)
    if not bool(torch.isfinite(objective_before) & torch.isfinite(slope)) or float(slope) > 0.0:
        raise RuntimeError("OT Newton direction is not a finite descent direction")
    step_scale = 1.0
    for evaluation in range(1, 31):
        candidate = v + step_scale * delta
        objective_after = torch.logsumexp(log_kernel + candidate.unsqueeze(0), dim=1).mean() - candidate.mean()
        if bool(torch.isfinite(objective_after)) and bool(objective_after <= objective_before + 1e-4 * step_scale * slope):
            log_u = -math.log(n_queries) - torch.logsumexp(log_kernel + candidate.unsqueeze(0), dim=1)
            return log_u, candidate, {
                "line_search_evaluations": evaluation,
                "accepted_step_scale": step_scale,
                "dual_objective_before": float(objective_before),
                "dual_objective_after": float(objective_after),
                "directional_derivative": float(slope),
            }
        step_scale *= 0.5
    error = RuntimeError("OT Newton Armijo line search exhausted 30 evaluations")
    error.ot_line_search_evaluations = 30
    raise error


@torch.no_grad()
def soft_ot_targets(
    anchor_cls: torch.Tensor,
    anchor_labels: torch.Tensor,
    query_cls: torch.Tensor,
    epsilon: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return detached P(y=match) targets and JSON-serializable diagnostics.

    Uniform marginals are 1/n_query and 1/n_anchor. The first 50 iterations
    preserve the original log-Sinkhorn path. If still unconverged, remaining
    iterations use gauge-fixed, damped Newton updates of that same dual;
    these updates count toward max_iterations, which is not increased.
    Every ten iterations
    (and at the iteration limit), *the same* transport matrix is checked
    against both marginals. q uses that checked matrix after row division;
    no extra Sinkhorn update is made after the convergence check.

    Raises OTInputError for invalid/non-finite/zero-norm input and
    OTConvergenceError, with a .diagnostics dictionary, for nonconvergence.
    Neither exception is a signal to silently skip OT or count a step.
    """
    diagnostics: Dict[str, Any] = {
        "implementation": "balanced_log_sinkhorn_newton_after50_double_v2",
        "cost_dtype": "float32",
        "solver_dtype": "float64",
        "target_dtype": "float32",
        "check_every": 10,
        "iterations": 0,
        "converged": False,
        "solver_phase": "log_sinkhorn",
        "newton_start_after": 50,
        "sinkhorn_steps": 0,
        "newton_steps": 0,
        "newton_linear_solves": 0,
        "line_search_evaluations_total": 0,
        "line_search_max_evaluations_per_update": 30,
    }
    if not all(isinstance(x, torch.Tensor) for x in (anchor_cls, anchor_labels, query_cls)):
        _fail("CLS and labels must be tensors", diagnostics)
    if anchor_cls.ndim != 2 or query_cls.ndim != 2:
        _fail("anchor_cls and query_cls must be rank-2 tensors", diagnostics)
    if min(anchor_cls.shape) < 1 or min(query_cls.shape) < 1:
        _fail("CLS dimensions must be nonempty", diagnostics)
    if anchor_cls.shape[1] != query_cls.shape[1]:
        _fail("anchor and query feature dimensions must match", diagnostics)
    if anchor_cls.device != query_cls.device:
        _fail("anchor and query CLS must be on the same device", diagnostics)
    if not anchor_cls.is_floating_point() or not query_cls.is_floating_point():
        _fail("CLS vectors must have a floating dtype", diagnostics)
    if anchor_labels.ndim != 1 or len(anchor_labels) != len(anchor_cls):
        _fail("anchor_labels must be a vector matching anchor count", diagnostics)
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        _fail("max_iterations must be a positive integer", diagnostics)
    try:
        epsilon = float(epsilon)
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError):
        _fail("epsilon and tolerance must be finite positive numbers", diagnostics)
    if not math.isfinite(epsilon) or epsilon <= 0:
        _fail("epsilon must be finite and positive", diagnostics)
    if not math.isfinite(tolerance) or tolerance <= 0:
        _fail("tolerance must be finite and positive", diagnostics)

    diagnostics.update(
        epsilon=epsilon,
        tolerance=tolerance,
        max_iterations=max_iterations,
        n_anchors=len(anchor_cls),
        n_queries=len(query_cls),
        feature_dim=anchor_cls.shape[1],
    )
    with torch.autocast(device_type=anchor_cls.device.type, enabled=False):
        anchors = anchor_cls.detach().float()
        queries = query_cls.detach().float()
        labels = anchor_labels.detach().to(device=anchor_cls.device, dtype=torch.float32)
        if not all(bool(torch.isfinite(x).all()) for x in (anchors, queries, labels)):
            _fail("CLS and labels must be finite after FP32 conversion", diagnostics)
        if not bool(((labels == 0) | (labels == 1)).all()):
            _fail("anchor labels must be binary 0 or 1", diagnostics)
        # Cosine is undefined for a zero vector. Reject rather than creating
        # apparently valid uniform targets from missing/corrupt features.
        anchor_norms = torch.linalg.vector_norm(anchors, dim=1)
        query_norms = torch.linalg.vector_norm(queries, dim=1)
        if not bool(torch.isfinite(anchor_norms).all() & torch.isfinite(query_norms).all()):
            _fail("CLS norms overflowed FP32", diagnostics)
        if bool((anchor_norms <= 1e-12).any() | (query_norms <= 1e-12).any()):
            _fail("CLS vectors must have a nonzero norm greater than 1e-12", diagnostics)
        anchors = F.normalize(anchors, p=2, dim=1)
        queries = F.normalize(queries, p=2, dim=1)
        cost = 1.0 - (queries @ anchors.T).clamp(-1.0, 1.0)
        diagnostics.update(
            cost_min=float(cost.min().item()),
            cost_max=float(cost.max().item()),
            cost_mean=float(cost.mean().item()),
            cost_std=float(cost.std(unbiased=False).item()),
            anchor_positive_fraction=float(labels.mean().item()),
        )
        log_kernel = -cost.double() / epsilon
        if not bool(torch.isfinite(log_kernel).all()):
            _fail("cost divided by epsilon is not finite in FP64", diagnostics)
        n_queries, n_anchors = cost.shape
        log_a = -math.log(n_queries)
        log_b = -math.log(n_anchors)
        log_u = torch.zeros(n_queries, device=cost.device, dtype=torch.float64)
        log_v = torch.zeros(n_anchors, device=cost.device, dtype=torch.float64)
        history = []
        plan = None
        for iteration in range(1, max_iterations + 1):
            if iteration <= 50:
                log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
                log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
                diagnostics["sinkhorn_steps"] = iteration
            else:
                diagnostics["solver_phase"] = "damped_newton_same_ot_dual"
                diagnostics["newton_linear_solves"] += 1
                try:
                    log_u, log_v, newton_info = _newton_dual_update(log_kernel, log_v)
                except RuntimeError as exc:
                    failed_evaluations = getattr(exc, "ot_line_search_evaluations", 0)
                    diagnostics["line_search_evaluations_total"] += failed_evaluations
                    diagnostics.update(iterations=iteration, converged=False,
                                       numerical_failure=str(exc), attempted_update="damped_newton",
                                       failed_newton_line_search_evaluations=failed_evaluations)
                    raise OTConvergenceError(diagnostics) from exc
                diagnostics["newton_steps"] += 1
                diagnostics["line_search_evaluations_total"] += newton_info["line_search_evaluations"]
                diagnostics["last_newton_update"] = {"iteration": iteration, **newton_info}
            if iteration % 10 != 0 and iteration != max_iterations:
                continue
            plan = torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))
            if not bool(torch.isfinite(plan).all()):
                diagnostics["iterations"] = iteration
                _fail("transport matrix is non-finite", diagnostics)
            row_residual = float((plan.sum(dim=1) - 1.0 / n_queries).abs().max().item())
            col_residual = float((plan.sum(dim=0) - 1.0 / n_anchors).abs().max().item())
            history.append({
                "iteration": iteration,
                "row_residual": row_residual,
                "col_residual": col_residual,
            })
            diagnostics.update(
                iterations=iteration,
                last_checked_iteration=iteration,
                row_residual=row_residual,
                col_residual=col_residual,
                max_residual=max(row_residual, col_residual),
                residual_history=history,
                converged=max(row_residual, col_residual) <= tolerance,
            )
            if diagnostics["converged"]:
                break
        if not diagnostics["converged"]:
            raise OTConvergenceError(diagnostics)
        # This is the same P for which both residuals above passed.
        row_mass = plan.sum(dim=1, keepdim=True)
        if bool((row_mass <= 0).any()):
            _fail("transport matrix contains an empty row", diagnostics)
        q = ((plan / row_mass) @ labels.double()).float().detach()
        if not bool(torch.isfinite(q).all() & (q >= 0).all() & (q <= 1).all()):
            _fail("OT targets must be finite probabilities", diagnostics)
        q64 = q.double()
        entropy = -(
            q64 * q64.clamp_min(1e-15).log()
            + (1.0 - q64) * (1.0 - q64).clamp_min(1e-15).log()
        )
        quantile_levels = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
        diagnostics.update(
            q_mean=float(q64.mean().item()),
            q_std=float(q64.std(unbiased=False).item()),
            q_min=float(q64.min().item()),
            q_max=float(q64.max().item()),
            q_entropy=float(entropy.mean().item()),
            q_entropy_units="nats",
            q_near_half_fraction=float(((q64 - 0.5).abs() <= 0.05).double().mean().item()),
            q_near_half_halfwidth=0.05,
            q_quantile_levels=quantile_levels,
            q_quantiles=torch.quantile(q64, torch.tensor(quantile_levels, device=q.device, dtype=torch.float64)).tolist(),
            q_mean_prior_difference=float((q64.mean() - labels.double().mean()).abs().item()),
            q_requires_grad=q.requires_grad,
            target_matrix="same_checked_P_row_normalized",
        )
        return q, diagnostics
