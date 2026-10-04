"""Independent-Test aggregation; keep main/control/adaptive views separate."""
from __future__ import annotations

import copy
import statistics
from collections import defaultdict

from .budget import BudgetController, ComputeLedger
from .evaluation import validate_metrics, validate_test_selection, validate_teacher_accounting
from .spec import SEEDS, fingerprint, is_sha256, load_protocol, make_config, positive_int


def validate_result(row, p):
    cfg = make_config(row["dataset"], int(row["budget"]), row["method"], row["seed"], row["policy"], p, row["regime"])
    if row["version"] != p["version"] or row["run_id"] != cfg["run_id"] or row["roles"] != cfg["roles"] or row["evaluation_model"] != "ema" or row["equal_compute_claim"]:
        raise ValueError("Historical, mislabelled comparison, non-EMA, or misleading result")
    if row["protocol_sha256"] != fingerprint(p) or row["config_sha256"] != fingerprint(cfg) or row["tuning"] != p["tuning"]:
        raise ValueError("Configuration/protocol/tuning registration changed")
    budget = BudgetController(cfg["policy"], p)
    budget.load_state_dict(row["budget_state"])
    if not budget.stop_reason or budget.evaluation_due or budget.step != row["successful_steps"] or row["stop_reason"] != budget.stop_reason or row["target_steps"] != cfg["target_steps"] or row["validations"] != len(budget.history) or row["validation_best_step"] != budget.best_step:
        raise ValueError("Completion and validation accounting disagree")
    if row["validation_best_metrics"] != dict(paired_accuracy=budget.best_key[0], auroc=budget.best_key[1]):
        raise ValueError("Validation-best diagnostic differs from registered history")
    if row["primary_checkpoint"] != "terminal_ema" or row["primary_step"] != budget.step or row["primary_split"] != "independent_test":
        raise ValueError("A pre-module Validation-best cannot be the primary checkpoint")
    validate_metrics(row["terminal_validation"], "validation", 400, budget.step)
    validate_metrics(row["student_terminal_validation"], "validation", 400, budget.step, model="student")
    if any(row["terminal_validation"][k] != budget.history[-1][k] for k in ("paired_accuracy", "auroc")):
        raise ValueError("Terminal Validation and history differ")
    ledger = ComputeLedger(); ledger.load_state_dict(row["compute"])
    cost = ledger.values
    inputs = row["provenance"]["inputs"]
    if not isinstance(inputs, dict) or positive_int(inputs.get("l_anchors"), "L anchors") < 16:
        raise ValueError("Recorded L membership count is missing")
    if type(row["simulation_only"]) is not bool or type(row["test_evaluated"]) is not bool or type(row["test_attempted"]) is not bool:
        raise ValueError("Explicit simulation/Test access status is required")
    if cost["l_pair_draws"] != budget.step*32 or cost["u_pair_draws"] != (0 if cfg["budget_percent"] == 100 else budget.step*32):
        raise ValueError("L/U sampling opportunities changed")
    if cost["attempts"] != budget.step+cost["failed_attempts"]:
        raise ValueError("Retries were dropped from computation accounting")
    if cost["validation_ema_forward_pairs"] != 800*len(budget.history) or cost["validation_student_forward_pairs"] != 800*len(budget.history):
        raise ValueError("Student/EMA Validation costs were dropped")
    teacher_fields = [k for k in cost if k.startswith("teacher_")]
    if cfg["uses_pairusa"]:
        if not is_sha256(row["teacher_targets_sha256"]) or any(cost[k] <= 0 for k in teacher_fields):
            raise ValueError("Complete teacher identity and operation costs are required")
        validate_teacher_accounting(cost, inputs["l_anchors"], p)
    elif row["teacher_targets_sha256"] is not None or any(cost[k] for k in teacher_fields):
        raise ValueError("Unexpected USA teacher for this method")
    resources = row["resources"]
    parameter_fields = ("student_total_parameters", "student_trainable_parameters", "common_trainable_parameters", "method_specific_trainable_parameters")
    if any(type(resources[k]) is not int or resources[k] < 0 for k in parameter_fields) or resources["common_trainable_parameters"] < 1 or resources["student_total_parameters"] < resources["student_trainable_parameters"] or resources["student_trainable_parameters"] != resources["common_trainable_parameters"]+resources["method_specific_trainable_parameters"]:
        raise ValueError("Invalid reported parameter accounting")
    for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        if row["simulation_only"]:
            if resources[key] is not None:
                raise ValueError("CPU simulations cannot claim GPU memory measurements")
        else:
            positive_int(resources[key], key)
    if cfg["uses_pairusa"]:
        teacher = resources["teacher"]
        for key in ("trainable_parameters", "descriptor_parameters"):
            positive_int(teacher[key], key)
        for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
            if not row["simulation_only"]:
                positive_int(teacher[key], key)
            elif teacher[key] is not None:
                raise ValueError("Synthetic teacher cannot claim CUDA measurements")
    elif resources["teacher"] is not None:
        raise ValueError("Unexpected teacher resource measurements")
    anchors = positive_int(row["test_anchors"], "Test anchors")
    holdout = row["holdout_identity"]
    if not isinstance(holdout, dict):
        raise ValueError("Recorded Test membership contract is missing")
    hash_fields = {"dataset_manifest_sha256", "validation_sha256", "test_sha256", "test_index_sha256", "registration_sha256"}
    expected_fields = hash_fields | {"test_anchors"}
    if row["simulation_only"] and holdout.get("synthetic_unit_test_only") is True:
        expected_fields.add("synthetic_unit_test_only")
    if set(holdout) != expected_fields or any(not is_sha256(holdout[k]) for k in hash_fields) or type(holdout["test_anchors"]) is not int or holdout["test_anchors"] != anchors or row["test_contract_sha256"] != fingerprint(holdout) or inputs.get("holdout") != holdout:
        raise ValueError("Recorded Test membership contract changed")
    selection = row["test_selection"]
    if selection is not None:
        validate_test_selection(selection, cfg, p, budget.step, test_hash=row["test_contract_sha256"])
        if selection["budget_state"] != budget.state_dict() or selection["validation_metrics_sha256"] != fingerprint(row["terminal_validation"]) or selection["threshold"] != row["terminal_validation"]["threshold"]:
            raise ValueError("Test selection differs from terminal Validation")
    if row["test_evaluated"]:
        if row["state"] != "completed" or not row["test_attempted"] or selection is None or cost["test_forward_pairs"] != 2*anchors:
            raise ValueError("Test completion and its full inference accounting differ")
        validate_metrics(row["primary_metrics"], "test", anchors, budget.step, threshold=selection["threshold"])
    elif row["primary_metrics"] is not None or row["state"] != ("test_intent_pending_or_failed" if row["test_attempted"] else "awaiting_test") or not 0 <= cost["test_forward_pairs"] <= 2*anchors or (not row["test_attempted"] and (cost["test_forward_pairs"] or cost["test_seconds"])):
        raise ValueError("Validation-only results must not claim independent-Test completion")
    if row["test_attempted"] and selection is None:
        raise ValueError("Test access without frozen selection")
    return cfg


def _timing_comparable(rows):
    timing = {(r["provenance"].get("hardware_sha256"), r["provenance"].get("environment_sha256")) for r in rows}
    return bool(rows) and not any(r["simulation_only"] for r in rows) and len(timing) == 1 and all(is_sha256(v) for v in next(iter(timing))) and all(r["provenance"].get("serial_uncontended_execution") is True for r in rows)


def summarize_results(results, protocol=None, view="test", table=None):
    p = load_protocol() if protocol is None else protocol
    if view not in ("test", "validation_diagnostic") or table not in (None, "main", "ablation", "warmup_control"):
        raise ValueError("Unknown summary view/table")
    groups, cells, identities, crops = defaultdict(list), defaultdict(list), set(), defaultdict(list)
    for row in results:
        cfg = validate_result(row, p)
        if row["run_id"] in identities:
            raise ValueError("Duplicate logical run; shared table roles are not extra replicas")
        identities.add(row["run_id"])
        if view == "test" and not row["test_evaluated"]:
            raise ValueError("Independent Test is missing; use validation_diagnostic explicitly, not a paper main table")
        cells[(row["dataset"], row["budget"], row["seed"], row["policy"])].append(row)
        crops[row["dataset"]].append(row)
        for role in cfg["roles"]:
            if table is None or role == table:
                groups[(role, row["dataset"], row["budget"], row["method"], row["policy"])].append(row)
    if not groups:
        raise ValueError("No matching results; plans are not completed experiments")
    flags = {r["simulation_only"] for rows in groups.values() for r in rows}
    if len(flags) != 1:
        raise ValueError("Synthetic controls and real scientific runs must not be mixed")
    for rows in crops.values():
        if len({fingerprint(r["holdout_identity"]) for r in rows}) != 1:
            raise ValueError("All budgets/seeds/regimes of a crop must use the same frozen Validation/Test")
    for rows in cells.values():
        initials = {r["initial_common_state_sha256"] for r in rows}
        if len(initials) != 1 or any(not is_sha256(x) for x in initials):
            raise ValueError("Methods do not share common initialization within this seed")
        if len({fingerprint(r["provenance"]["inputs"]) for r in rows}) != 1:
            raise ValueError("Comparison L/U/Validation/Test/model contracts differ")
        warmups = {r["warmup_common_full_state_sha256"] for r in rows if r["regime"] == "shared_bce"}
        if len(warmups) > 1 or any(not is_sha256(x) for x in warmups):
            raise ValueError("Shared-BCE branches did not share the full fixed-step warmup state")
        teachers = {r["teacher_targets_sha256"] for r in rows if r["method"] in ("pairusa", "pairusa_ot")}
        if len(teachers) > 1:
            raise ValueError("USA ablations must reuse the same same-cell teacher")
    summary, grouped_metrics = [], {}
    metric_field = "primary_metrics" if view == "test" else "terminal_validation"
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda x: x["seed"])
        if [r["seed"] for r in rows] != list(SEEDS):
            raise ValueError(f"Missing/extra training seed in {key}; require all three")
        if len({fingerprint(r["provenance"]["inputs"]) for r in rows}) != 1:
            raise ValueError("Training seeds changed the frozen data/model contracts")
        metrics, fixed_threshold = {}, {}
        for name in ("paired_accuracy", "auroc", "accuracy", "macro_f1"):
            values = [r[metric_field][name] for r in rows]
            metrics[name] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))
        for name in ("accuracy", "macro_f1"):
            values = [r[metric_field]["threshold_0_5"][name] for r in rows]
            fixed_threshold[name] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))
        item = dict(table=key[0], dataset=key[1], budget=key[2], method=key[3], policy=key[4], regime=rows[0]["regime"],
                    seeds=list(SEEDS), metrics=metrics, fixed_threshold_0_5=fixed_threshold,
                    thresholds_from_terminal_validation=[r["terminal_validation"]["threshold"] for r in rows],
                    successful_steps=[r["successful_steps"] for r in rows],
                    full_accounted_seconds=[sum(v for k, v in r["compute"].items() if k.endswith("seconds")) for r in rows],
                    computation_by_seed=[copy.deepcopy(r["compute"]) for r in rows],
                    resources_by_seed=[copy.deepcopy(r["resources"]) for r in rows],
                    validation_best_steps_diagnostic_only=[r["validation_best_step"] for r in rows],
                    student_validation_calls=[r["validations"] for r in rows],
                    teacher_validation_calls=[r["compute"]["teacher_validation_calls"] for r in rows],
                    equal_compute_claim=False)
        summary.append(item); grouped_metrics[key] = rows
    timing_cells = defaultdict(list)
    for key, rows in grouped_metrics.items():
        timing_cells[(*key[:3], key[4])].extend(rows)
    for item in summary:
        rows = timing_cells[(item["table"], item["dataset"], item["budget"], item["policy"])]
        item["timing_scope"] = "all_supplied_methods_and_seeds_in_this_table_cell"
        item["timing_comparable"] = len({r["method"] for r in rows}) > 1 and _timing_comparable(rows)
    comparisons = []
    for key, rows in grouped_metrics.items():
        baseline_key = (*key[:3], "bce", key[4])
        if key[3] == "bce" or baseline_key not in grouped_metrics:
            continue
        baseline, differences = grouped_metrics[baseline_key], {}
        for name in ("paired_accuracy", "auroc", "accuracy", "macro_f1"):
            values = [a[metric_field][name]-b[metric_field][name] for a, b in zip(rows, baseline)]
            differences[name] = dict(per_seed=values, mean=statistics.mean(values), sample_std=statistics.stdev(values))
        comparisons.append(dict(table=key[0], dataset=key[1], budget=key[2], policy=key[4], method=key[3], baseline="bce",
                                seeds=list(SEEDS), paired_differences=differences,
                                timing_comparable=_timing_comparable(rows+baseline), statistical_significance_claim=False))
    completion, cell_methods = [], defaultdict(set)
    for role, crop, budget, method, policy in grouped_metrics:
        cell_methods[(role, crop, budget, policy)].add(method)
    for (role, crop, budget, policy), methods in sorted(cell_methods.items()):
        expected = p["main_methods"] if role == "main" else (p["warmup_control"]["methods"] if role == "warmup_control" else (["bce", "pairusa"] if int(budget) == 100 else p["ablation_methods"]))
        missing = sorted(set(expected)-methods)
        completion.append(dict(table=role, dataset=crop, budget=budget, policy=policy, missing_methods=missing, complete_comparison=not missing))
    return dict(version=p["version"], protocol_sha256=fingerprint(p), simulation_only=flags.pop(),
                evaluation_split="independent_test" if view == "test" else "validation_diagnostic_not_generalization",
                checkpoint_selection="terminal_ema", validation_best_is_diagnostic_only=True,
                seed_statistics="mean_and_sample_std_n3_not_a_significance_claim",
                policies_and_tables_not_pooled=True, groups=summary, paired_comparisons=comparisons,
                comparison_cells=completion, scientific_validity_not_certified_by_this_summary=True)
