"""Separate plateau references, adaptive branches and matched-step contrasts."""
from __future__ import annotations

import copy
import statistics
from collections import defaultdict

from fair_benchmark.budget import ComputeLedger
from fair_benchmark.evaluation import validate_metrics, validate_teacher_accounting
from fair_benchmark.report import _timing_comparable
from fair_benchmark.spec import SEEDS, fingerprint, is_sha256, positive_int
from .budget import PhaseBudget
from .contracts import validate_match_receipt, validate_parent_receipt, validate_phase_cost, validate_terminal_validation, validate_test_selection
from .spec import load_protocol, make_config, validate_protocol


def _resources(resources, simulation, uses_teacher):
    fields = ("student_total_parameters", "student_trainable_parameters", "common_trainable_parameters", "method_specific_trainable_parameters")
    if any(type(resources[k]) is not int or resources[k] < 0 for k in fields) or resources["common_trainable_parameters"] < 1 or resources["student_total_parameters"] < resources["student_trainable_parameters"] or resources["student_trainable_parameters"] != resources["common_trainable_parameters"]+resources["method_specific_trainable_parameters"]:
        raise ValueError("Invalid common/method-specific parameter accounting")
    for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        if simulation:
            if resources[key] is not None:
                raise ValueError("Synthetic tests cannot claim CUDA memory")
        else:
            positive_int(resources[key], key)
    if not uses_teacher:
        if resources["teacher"] is not None:
            raise ValueError("Unexpected teacher resources")
    else:
        teacher = resources["teacher"]
        for k in ("trainable_parameters", "descriptor_parameters"):
            positive_int(teacher[k], k)
        for k in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
            if simulation:
                if teacher[k] is not None:
                    raise ValueError("Synthetic teacher cannot claim CUDA memory")
            else:
                positive_int(teacher[k], k)


def _full_timing_comparable(rows):
    if not _timing_comparable(rows) or any(r["provenance"].get("full_pipeline_same_hardware_serial_execution") is not True for r in rows):
        return False
    for row in rows:
        parent = row["parent_receipt"]
        if parent is not None:
            e, v = parent["execution_identity"], row["provenance"]
            if e["hardware_sha256"] != v["hardware_sha256"] or e["environment_sha256"] != v["environment_sha256"] or e["serial_uncontended_execution"] is not True:
                return False
    return True


def validate_result(row, protocol=None):
    p = load_protocol() if protocol is None else validate_protocol(protocol)
    cfg = make_config(row["dataset"], int(row["budget"]), row["method"], row["seed"], row["stage"], row["match_method"], p)
    if row["version"] != p["version"] or row["run_id"] != cfg["run_id"] or row["roles"] != cfg["roles"] or row["policy"] != cfg["policy"] or row["regime"] != cfg["regime"] or row["evaluation_model"] != "ema" or row["equal_compute_claim"] is not False or row["equal_steps_claim"] is not False:
        raise ValueError("Mislabelled/historical result or misleading fairness claim")
    if row["protocol_sha256"] != fingerprint(p) or row["config_sha256"] != fingerprint(cfg) or row["tuning"] != p["tuning"]:
        raise ValueError("Registered config/protocol/tuning differs")
    inputs = row["provenance"]["inputs"]
    l_anchors = positive_int(inputs["l_anchors"], "recorded L anchors")
    budget = PhaseBudget(cfg, l_anchors, p, row["match_receipt"])
    budget.load_state_dict(row["budget_state"])
    validate_terminal_validation(dict(ema=row["terminal_validation"], student=row["student_terminal_validation"]), budget)
    if row["successful_steps"] != budget.step or row["target_steps"] != cfg["target_steps"] or row["stop_reason"] != budget.stop_reason or row["validations"] != len(budget.history) or row["validation_best_step"] != budget.best_step or row["validation_best_metrics"] != dict(paired_accuracy=budget.best_key[0], auroc=budget.best_key[1]):
        raise ValueError("Phase completion and replayed history differ")
    if row["primary_checkpoint"] != "terminal_ema" or row["primary_step"] != budget.step or row["primary_split"] != "independent_test" or row["warmup_common_full_state_sha256"] is not None:
        raise ValueError("Only the v4 terminal EMA is a primary model")
    if row["plateau_detected"] is not (budget.stop_reason == "validation_plateau") or row["capped_not_plateau"] is not (budget.stop_reason == "phase_cap_exhausted"):
        raise ValueError("A cap must not be presented as a plateau")
    if any(type(row[k]) is not bool for k in ("simulation_only", "test_evaluated", "test_attempted")) or not is_sha256(row["initial_common_state_sha256"]):
        raise ValueError("Execution/Test/initialization proof is missing")
    if not row["simulation_only"] and not is_sha256(row["provenance"].get("source_sha256")):
        raise ValueError("Real result code-source evidence is missing")
    validate_phase_cost(row["phase_compute"], cfg, budget)
    cost = ComputeLedger(); cost.load_state_dict(row["phase_compute"])
    upstream = ComputeLedger(); upstream.load_state_dict(row["upstream_bce_compute"])
    combined = ComputeLedger(); combined.load_state_dict(upstream.state_dict()); combined.add(**cost.state_dict())
    if row["compute"] != combined.state_dict():
        raise ValueError("Full costs must include the entire BCE parent once per logical branch")
    parent = row["parent_receipt"]
    if cfg["stage"] == "bce":
        if parent is not None or row["parent_resources"] is not None or row["parent_run_id"] is not None or any(upstream.values.values()) or row["adaptation_common_state_sha256"] is not None or row["additional_steps"] != 0 or row["total_student_steps"] != budget.step:
            raise ValueError("Unexpected upstream adaptation state for BCE parent")
        ref = row["reference_receipt"]
        if budget.stop_reason == "validation_plateau":
            validate_parent_receipt(ref, protocol=p)
            if ref["config"] != cfg or ref["budget"] != budget.state_dict() or ref["input_identity_sha256"] != fingerprint(inputs) or ref["initial_common_state_sha256"] != row["initial_common_state_sha256"] or ref["simulation_only"] != row["simulation_only"]:
                raise ValueError("Frozen BCE reference receipt differs")
        elif ref is not None:
            raise ValueError("Capped BCE cannot supply a branch receipt")
    else:
        validate_parent_receipt(parent, cfg, l_anchors, p)
        if parent["source_sha256"] != row["provenance"].get("source_sha256"):
            raise ValueError("Branch and parent code-source versions differ")
        if row["reference_receipt"] is not None or parent["input_identity_sha256"] != fingerprint(inputs) or parent["initial_common_state_sha256"] != row["initial_common_state_sha256"] or parent["simulation_only"] != row["simulation_only"] or row["parent_run_id"] != cfg["parent_run_id"] or upstream.state_dict() != parent["compute"] or row["parent_resources"] != parent["resources"]:
            raise ValueError("Branch input/initialization/parent/cost differs")
        if not is_sha256(row["adaptation_common_state_sha256"]) or row["additional_steps"] != budget.step or row["total_student_steps"] != parent["budget"]["step"]+budget.step:
            raise ValueError("Missing phase-zero proof or additional-step accounting")
        _resources(parent["resources"], row["simulation_only"], False)
        if cfg["stage"] == "matched_bce":
            match = validate_match_receipt(row["match_receipt"], cfg, l_anchors, p)
            if match["parent_receipt_sha256"] != parent["receipt_sha256"] or match["input_identity_sha256"] != fingerprint(inputs) or match["simulation_only"] != row["simulation_only"] or match["additional_steps"] != budget.step:
                raise ValueError("Matched BCE did not use its source parent/exact additional steps")
    teacher_fields = [k for k in cost.values if k.startswith("teacher_")]
    if cfg["uses_pairusa"]:
        if not is_sha256(row["teacher_targets_sha256"]) or any(cost.values[k] <= 0 for k in teacher_fields):
            raise ValueError("Teacher identity/full upstream cost required")
        validate_teacher_accounting(cost.values, l_anchors, p)
    elif row["teacher_targets_sha256"] is not None or any(cost.values[k] for k in teacher_fields):
        raise ValueError("Unexpected teacher cost/identity")
    _resources(row["resources"], row["simulation_only"], cfg["uses_pairusa"])
    holdout, anchors = row["holdout_identity"], positive_int(row["test_anchors"], "Test anchors")
    hashes = {"dataset_manifest_sha256", "validation_sha256", "test_sha256", "test_index_sha256", "registration_sha256"}
    fields = hashes | {"test_anchors"}
    if row["simulation_only"] and holdout.get("synthetic_unit_test_only") is True:
        fields.add("synthetic_unit_test_only")
    if set(holdout) != fields or any(not is_sha256(holdout[k]) for k in hashes) or type(holdout["test_anchors"]) is not int or holdout["test_anchors"] != anchors or inputs["holdout"] != holdout or row["test_contract_sha256"] != fingerprint(holdout):
        raise ValueError("Trusted independent holdout contract differs")
    selection = row["test_selection"]
    if selection is not None:
        validate_test_selection(selection, cfg, p, budget.step, test_hash=row["test_contract_sha256"])
        if selection["budget_state"] != budget.state_dict() or selection["validation_metrics_sha256"] != fingerprint(row["terminal_validation"]) or selection["threshold"] != row["terminal_validation"]["threshold"] or selection["parent_receipt_sha256"] != (None if parent is None else parent["receipt_sha256"]) or selection["adaptation_common_state_sha256"] != row["adaptation_common_state_sha256"]:
            raise ValueError("Frozen Test selection and phase source differ")
    if row["test_evaluated"]:
        if row["state"] != "completed" or not row["test_attempted"] or selection is None or cost.values["test_forward_pairs"] != 2*anchors:
            raise ValueError("Incomplete one-shot Test inference")
        validate_metrics(row["primary_metrics"], "test", anchors, budget.step, threshold=selection["threshold"])
    else:
        state = "bce_cap_requires_review" if cfg["stage"] == "bce" and budget.stop_reason == "phase_cap_exhausted" else ("test_intent_pending_or_failed" if row["test_attempted"] else "awaiting_test")
        if row["state"] != state or row["primary_metrics"] is not None or not 0 <= cost.values["test_forward_pairs"] <= 2*anchors or (not row["test_attempted"] and (cost.values["test_forward_pairs"] or cost.values["test_seconds"])):
            raise ValueError("Validation-only/capped results must not claim Test completion")
    if row["test_attempted"] and selection is None:
        raise ValueError("Test intent without a frozen source")
    return cfg


def _distribution(values):
    return dict(per_seed=values, mean=statistics.mean(values), sample_std=statistics.stdev(values))


def summarize_results(results, protocol=None, view="test", table=None):
    p = load_protocol() if protocol is None else validate_protocol(protocol)
    if view not in ("test", "validation_diagnostic") or table not in (None, "main", "ablation", "bce_reference", "matched_bce"):
        raise ValueError("Unknown v4 summary view/table")
    groups, stages, cells, crops = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    seen, flags = set(), set()
    for row in results:
        cfg = validate_result(row, p)
        if row["run_id"] in seen:
            raise ValueError("Duplicate run; shared roles are not additional replicas")
        if view == "test" and not row["test_evaluated"]:
            raise ValueError("Test missing; explicitly use validation_diagnostic")
        seen.add(row["run_id"]); flags.add(row["simulation_only"])
        cells[(row["dataset"], row["budget"], row["seed"])].append(row)
        crops[row["dataset"]].append(row)
        stages[(row["stage"], row["dataset"], row["budget"], row["method"], row["match_method"])].append(row)
        for role in cfg["roles"]:
            if table is None or role == table:
                groups[(role, row["dataset"], row["budget"], row["method"], row["match_method"])].append(row)
    if not groups or len(flags) != 1:
        raise ValueError("No matching results, or real/synthetic experiments were mixed")
    for rows in crops.values():
        if len({fingerprint(r["holdout_identity"]) for r in rows}) != 1:
            raise ValueError("All crop budgets/seeds must share the frozen holdout")
    for rows in cells.values():
        if len({r["initial_common_state_sha256"] for r in rows}) != 1 or len({fingerprint(r["provenance"]["inputs"]) for r in rows}) != 1:
            raise ValueError("Compared methods differ in initialization/private inputs")
        parents = {r["parent_receipt"]["receipt_sha256"] for r in rows if r["parent_receipt"] is not None}
        parents.update(r["reference_receipt"]["receipt_sha256"] for r in rows if r["reference_receipt"] is not None)
        if len(parents) > 1 or len({r["adaptation_common_state_sha256"] for r in rows if r["stage"] != "bce"}) > 1:
            raise ValueError("Branches did not share the plateau parent and fresh phase-zero state")
        if len({r["teacher_targets_sha256"] for r in rows if r["method"] in ("pairusa", "pairusa_ot")}) > 1:
            raise ValueError("USA and full method must reuse the same teacher")
        by_id = {r["run_id"]: r for r in rows}
        for r in rows:
            match = r["match_receipt"]
            if match is None or match["source_config"]["run_id"] not in by_id:
                continue
            source = by_id[match["source_config"]["run_id"]]
            if source["budget_state"] != match["source_budget"] or source["terminal_validation"] != match["terminal_validation"]["ema"] or source["student_terminal_validation"] != match["terminal_validation"]["student"] or (source["test_selection"] is not None and source["test_selection"]["ema_state_sha256"] != match["terminal_ema_sha256"]):
                raise ValueError("Matched receipt differs from the supplied source result")
    for key, rows in stages.items():
        rows.sort(key=lambda r: r["seed"])
        if [r["seed"] for r in rows] != list(SEEDS) or len({fingerprint(r["provenance"]["inputs"]) for r in rows}) != 1:
            raise ValueError(f"Require all three seeds and fixed data contracts in {key}")
    metric_field = "primary_metrics" if view == "test" else "terminal_validation"
    summaries = []
    for key, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rows = sorted(rows, key=lambda r: r["seed"])
        summaries.append(dict(table=key[0], dataset=key[1], budget=key[2], method=key[3], match_method=key[4],
                              seeds=list(SEEDS), metrics={m: _distribution([r[metric_field][m] for r in rows]) for m in ("paired_accuracy", "auroc", "accuracy", "macro_f1")},
                              fixed_threshold_0_5={m: _distribution([r[metric_field]["threshold_0_5"][m] for r in rows]) for m in ("accuracy", "macro_f1")},
                              phase_steps=[r["successful_steps"] for r in rows], additional_steps=[r["additional_steps"] for r in rows],
                              total_student_steps=[r["total_student_steps"] for r in rows], stop_reasons=[r["stop_reason"] for r in rows],
                              full_accounted_seconds=[sum(v for k, v in r["compute"].items() if k.endswith("seconds")) for r in rows],
                              computation_by_seed=[copy.deepcopy(r["compute"]) for r in rows],
                              phase_computation_by_seed=[copy.deepcopy(r["phase_compute"]) for r in rows],
                              parent_computation_by_seed=[copy.deepcopy(r["upstream_bce_compute"]) for r in rows],
                              resources_by_seed=[copy.deepcopy(r["resources"]) for r in rows],
                              parent_resources_by_seed=[copy.deepcopy(r["parent_resources"]) for r in rows],
                              thresholds_from_terminal_validation=[r["terminal_validation"]["threshold"] for r in rows],
                              equal_steps_claim=False, equal_compute_claim=False))
    for item in summaries:
        rows = [r for key, group in groups.items() if key[:3] == (item["table"], item["dataset"], item["budget"]) for r in group]
        item["timing_comparable"] = len({(r["method"], r["match_method"]) for r in rows}) > 1 and _full_timing_comparable(rows)
        item["timing_scope"] = "all_supplied_methods_and_seeds_in_this_table_cell"
    contrasts, missing = [], []
    def compare(kind, module_rows, baseline_rows, equal_additional):
        contrasts.append(dict(contrast=kind, dataset=module_rows[0]["dataset"], budget=module_rows[0]["budget"],
                              method=module_rows[0]["method"], seeds=list(SEEDS), equal_additional_steps=equal_additional,
                              paired_differences={m: _distribution([a[metric_field][m]-b[metric_field][m] for a, b in zip(module_rows, baseline_rows)]) for m in ("paired_accuracy", "auroc", "accuracy", "macro_f1")},
                              treatment_additional_steps=[r["additional_steps"] for r in module_rows],
                              baseline_additional_steps=[r["additional_steps"] for r in baseline_rows],
                              timing_comparable=_full_timing_comparable(module_rows+baseline_rows), statistical_significance_claim=False))
    for key, rows in stages.items():
        stage, crop, budget, method, _ = key
        if stage != "adaptation":
            continue
        parent = stages.get(("bce", crop, budget, "bce", None))
        adaptive = stages.get(("adaptation", crop, budget, "bce", None))
        matched = stages.get(("matched_bce", crop, budget, "bce", method))
        lacking = []
        if parent:
            compare("BCE_continuation_gain" if method == "bce" else "module_total_gain_not_module_only", rows, parent, False)
        else:
            lacking.append("frozen_BCE_reference")
        if method != "bce":
            if adaptive:
                compare("adaptive_module_minus_adaptive_BCE_steps_may_differ", rows, adaptive, False)
            else:
                lacking.append("adaptive_BCE_continuation")
            if matched:
                compare("module_minus_matched_BCE_module_increment", rows, matched, True)
            else:
                lacking.append("matched_BCE_for_this_method")
        missing.append(dict(dataset=crop, budget=budget, method=method, missing_controls=lacking,
                            complete_attribution_controls=not lacking))
    comparison_cells = []
    for role, crop, budget in sorted({k[:3] for k in groups}):
        if role not in ("main", "ablation"):
            continue
        expected = p["main_methods"] if role == "main" else (["bce", "pairusa"] if int(budget) == 100 else p["ablation_methods"])
        supplied = {k[3] for k in groups if k[:3] == (role, crop, budget)}
        absent = sorted(set(expected)-supplied)
        comparison_cells.append(dict(table=role, dataset=crop, budget=budget, missing_methods=absent, complete_comparison=not absent))
    return dict(version=p["version"], protocol_sha256=fingerprint(p), simulation_only=flags.pop(),
                evaluation_split="independent_test" if view == "test" else "validation_diagnostic_not_generalization",
                checkpoint_selection="terminal_ema", groups=summaries, paired_contrasts=contrasts,
                attribution_controls=missing, comparison_cells=comparison_cells,
                seed_statistics="mean_and_sample_std_n3_not_a_significance_claim", tables_and_phases_not_pooled=True,
                scientific_validity_not_certified_by_this_summary=True)
