"""Reject mixed policies, missing seeds, changed inputs and unmatched warmups."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from .budget import BudgetController, ComputeLedger
from .spec import SEEDS, fingerprint, is_sha256, load_protocol, make_config


def summarize_results(results, protocol=None):
    p = load_protocol() if protocol is None else protocol
    groups, cells, identities = defaultdict(list), defaultdict(list), set()
    for row in results:
        cfg = make_config(row["dataset"], int(row["budget"]), row["method"], row["seed"], row["policy"], p)
        if row["run_id"] in identities:
            raise ValueError("Duplicate logical run; replicas are not additional seeds")
        identities.add(row["run_id"])
        if row["version"] != p["version"] or row["run_id"] != cfg["run_id"] or row["state"] != "completed" or row["evaluation_model"] != "ema" or row["test_evaluated"] or row["equal_compute_claim"]:
            raise ValueError("Historical, incomplete, non-EMA, or misleading result")
        if row["protocol_sha256"] != fingerprint(p) or row["config_sha256"] != fingerprint(cfg):
            raise ValueError("Configuration/protocol changed")
        budget = BudgetController(cfg["policy"], p)
        budget.load_state_dict(row["budget_state"])
        if not budget.stop_reason or budget.step != row["successful_steps"] or row["stop_reason"] != budget.stop_reason or row["target_steps"] != cfg["target_steps"] or row["validations"] != len(budget.history) or row["best_step"] != budget.best_step:
            raise ValueError("Completion and validation accounting disagree")
        if row["best_metrics"] != dict(paired_accuracy=budget.best_key[0], auroc=budget.best_key[1]):
            raise ValueError("Reported best is not the registered validation selection")
        if cfg["policy"] == "fixed" and budget.step != cfg["target_steps"]:
            raise ValueError("Fixed-budget result stopped early")
        ledger = ComputeLedger()
        ledger.load_state_dict(row["compute"])
        if type(row["simulation_only"]) is not bool:
            raise ValueError("Results must distinguish synthetic tests from real training")
        if cfg["uses_pairusa"]:
            if not is_sha256(row["teacher_targets_sha256"]) or row["compute"]["teacher_upstream_seconds"] <= 0:
                raise ValueError("USA teacher identity and full measured cost are required")
        elif row["teacher_targets_sha256"] is not None or row["compute"]["teacher_upstream_seconds"]:
            raise ValueError("Unexpected USA teacher for this method")
        if row["compute"]["l_pair_draws"] != budget.step * 32:
            raise ValueError("L exposure and successful updates disagree")
        expected_u = 0 if cfg["budget_percent"] == 100 else budget.step * 32
        if row["compute"]["u_pair_draws"] != expected_u:
            raise ValueError("U sampling opportunities changed")
        if row["compute"]["attempts"] != budget.step + row["compute"]["failed_attempts"]:
            raise ValueError("Retries were dropped from computation accounting")
        key = (row["dataset"], row["budget"], row["method"], row["policy"])
        groups[key].append(row)
        cells[(row["dataset"], row["budget"], row["seed"], row["policy"])].append(row)
    if not groups:
        raise ValueError("No results; planned configurations are not completed experiments")
    simulation_flags = {r["simulation_only"] for rows in groups.values() for r in rows}
    if len(simulation_flags) != 1:
        raise ValueError("Synthetic control tests and real scientific runs must not be mixed")
    for rows in cells.values():
        for name in ("initial_common_state_sha256",):
            values = {r[name] for r in rows}
            if len(values) != 1 or any(not is_sha256(x) for x in values):
                raise ValueError("Methods do not share common model initialization within this seed")
        if len({fingerprint(r["provenance"]["inputs"]) for r in rows}) != 1:
            raise ValueError("L/U/Validation/model asset contracts differ within a comparison cell")
        warmups = {r["warmup_common_full_state_sha256"] for r in rows if r["method"] in ("bce", "pairusa", "ot", "pairusa_ot")}
        if len(warmups) > 1 or any(not is_sha256(x) for x in warmups):
            raise ValueError("Ablations did not share the same full fixed-step warmup state")
        teachers = {r["teacher_targets_sha256"] for r in rows if r["method"] in ("pairusa", "pairusa_ot")}
        if len(teachers) > 1:
            raise ValueError("USA ablations must reuse the same same-cell teacher targets")
    summary = []
    for key, rows in sorted(groups.items()):
        if sorted(r["seed"] for r in rows) != list(SEEDS):
            raise ValueError(f"Missing/extra training seed in {key}; require all three before aggregation")
        if len({fingerprint(r["provenance"]["inputs"]) for r in rows}) != 1:
            raise ValueError("Training seeds must not change the frozen data/model asset contract")
        values = {}
        for metric in ("paired_accuracy", "auroc"):
            samples = [r["best_metrics"][metric] for r in rows]
            if any(not math.isfinite(x) for x in samples):
                raise ValueError("Nonfinite seed metric")
            values[metric] = dict(mean=statistics.mean(samples), sample_std=statistics.stdev(samples))
        summary.append(dict(dataset=key[0], budget=key[1], method=key[2], policy=key[3],
                            seeds=list(SEEDS), metrics=values,
                            successful_steps=[r["successful_steps"] for r in sorted(rows, key=lambda x: x["seed"])],
                            full_accounted_seconds=[sum(v for k, v in r["compute"].items() if k.endswith("seconds")) for r in sorted(rows, key=lambda x: x["seed"])],
                            equal_compute_claim=False, evaluation_split="validation_not_Test"))
    return dict(version=p["version"], protocol_sha256=fingerprint(p),
                simulation_only=simulation_flags.pop(),
                seed_statistics="mean_and_sample_std_n3_not_a_significance_claim",
                policies_not_pooled=True, groups=summary)
