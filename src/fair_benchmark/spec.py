"""One source of truth for three-seed, equal-total-update ITM comparisons."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/pair_itm_fair_v3/configs/protocol.public.json"
VERSION = "pair_itm_fair_v3"
SEEDS = (20260825, 20260826, 20260827)
SSL_METHODS = ("meanteacher", "fixmatch", "softmatch", "simmatch", "freematch")
ABLATION_METHODS = ("bce", "pairusa", "ot", "pairusa_ot")


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_protocol(p):
    if p["version"] != VERSION or p["training_seeds"] != list(SEEDS):
        raise ValueError("fair-v3 requires the three registered training seeds; v2 is not resumable")
    if p["data_construction_seed"] != 20260825:
        raise ValueError("Training seeds must not regenerate splits or negative pairs")
    if p["datasets"] != ["apple", "cassava", "rice", "banana"]:
        raise ValueError("Unexpected crop scope")
    if p["main_budgets_percent"] != [1, 5, 10, 20, 30] or p["ablation_budgets_percent"] != [1, 5, 10, 20, 30, 100]:
        raise ValueError("Unexpected label-budget scope")
    if p["main_methods"] != ["bce", *SSL_METHODS, "pairusa_ot"] or p["ablation_methods"] != list(ABLATION_METHODS):
        raise ValueError("Unexpected method grid")
    t, a, e, stop = (p[k] for k in ("training", "ablation", "evaluation", "adaptive_supplement"))
    for k in ("max_successful_steps", "warmup_steps", "checkpoint_every", "max_same_step_attempts"):
        positive_int(t[k], k)
    total = t["max_successful_steps"]
    if total != 3200 or a["shared_bce_steps"] != 1600 or a["branch_steps"] != 1600:
        raise ValueError("The registered 1600+1600 student budget cannot drift per method")
    if a["shared_bce_steps"] + a["branch_steps"] != total:
        raise ValueError("Parent/warmup cost must be included in the total")
    if t["warmup_steps"] >= total or e["every_successful_steps"] != 100 or total % e["every_successful_steps"]:
        raise ValueError("Invalid LR or common validation schedule")
    if (t["logical_l_anchors"], t["logical_l_pairs"], t["logical_u_pairs"]) != (16, 32, 32):
        raise ValueError("Logical data exposure changed")
    if t["physical_pairs"] != 16 or t["oom_physical_fallback"] != [8, 4]:
        raise ValueError("Unexpected physical batching policy")
    if any(a[k] for k in ("reset_optimizer_at_branch", "reset_lr_at_branch", "s4_inherits_s3")):
        raise ValueError("Ablation branches must not acquire additional optimization stages")
    if a["parent_selection"] != "fixed_warmup_endpoint_never_parent_best":
        raise ValueError("Selected historical best is not a controlled warmup endpoint")
    if e["primary_model"] != "ema" or e["best_criterion"] != ["paired_accuracy", "auroc"] or e["test_inference"] is not True:
        raise ValueError("Selection model/metric/Test policy differs")
    if e["checkpoint_selection"] != "terminal_ema_fixed_3200_or_registered_adaptive_stop" or e["validation_best_usage"] != "diagnostic_only_never_primary_or_test_checkpoint" or e["primary_split"] != "independent_test" or e["test_never_selects_model_threshold_or_stopping"] is not True:
        raise ValueError("Primary results require a frozen terminal EMA and independent Test")
    if (e["validation_anchors"], e["validation_pairs"]) != (400, 800):
        raise ValueError("Unexpected validation contract")
    if stop["min_successful_steps"] != 1600 or stop["patience_validations"] != 8 or stop["min_delta"] != .005:
        raise ValueError("Adaptive policy must be frozen uniformly before training")
    if not 0 < stop["min_delta"] <= 1 or not math.isfinite(stop["min_delta"]):
        raise ValueError("Invalid early-stop delta")
    f = p["fairness"]
    if f["equal_compute_claim"] or f["equal_unlabeled_pair_visits_claim"] or f["historical_checkpoints_reusable"]:
        raise ValueError("Equal student steps do not imply equal computation/U use or old-checkpoint compatibility")
    if p["publication"]["launch_enabled"]:
        raise ValueError("A public plan must not pretend to be admitted for training")
    control = p["warmup_control"]
    if control["datasets"] != p["datasets"] or control["budgets_percent"] != [1, 10] or control["methods"] != p["main_methods"] or control["shared_bce_steps"] != 1600 or control["branch_steps"] != 1600 or control["reset_optimizer_or_lr"] or control["fixed_policy_only"] is not True:
        raise ValueError("Warmup control scope must be registered before observing new results")
    tuning = p["tuning"]
    if tuning["strategy"] != "frozen_settings_no_new_v3_hyperparameter_search" or any(type(tuning[k]) is not int or tuning[k] != 0 for k in ("trials_per_method", "new_data_pilot_runs_per_method", "test_uses_for_tuning")):
        raise ValueError("No new v3 tuning is registered; later search requires a new protocol")
    # These descriptive policies are implemented, not configurable labels.
    # Reject metadata that would claim a different scientific procedure.
    policies = {
        "training": {"warmup_steps": 80, "lr_schedule": "one_global_warmup_cosine_to_registered_cap"},
        "ablation": {
            "teacher_training_inputs": "this_cell_L_pairs_only_never_U_labels_or_Test",
            "teacher_reuse": "same_crop_budget_seed_and_teacher_fingerprint_for_USA_and_USA_OT",
            "teacher_cost": "full_upstream_cost_reported_per_method_even_when_physically_reused"},
        "warmup_control": {
            "selection": "registered_before_new_v3_runs_not_selected_by_observed_winners",
            "ssl_internal_warmup_clock": "branch_local_steps_and_1600_step_branch_horizon",
            "simmatch_bank_initialization": "current_EMA_after_shared_BCE_before_first_SSL_update",
            "unchanged_BCE_and_full_method_reused_not_rerun": True,
            "not_original_algorithm_standard_training": True},
        "evaluation": {
            "secondary_model": "student_validation_diagnostic_only",
            "threshold": "terminal_validation_max_macro_F1_locked_before_Test_plus_fixed_0.5",
            "test_access": "after_training_complete_and_terminal_selection_sealed_once",
            "test_count": "trusted_private_holdout_contract_not_validation_relabelled"},
        "adaptive_supplement": {
            "monitor": "paired_accuracy", "patience_starts_at_min_steps": True,
            "max_steps_and_lr_horizon_identical_to_fixed": True},
        "augmentation": {"same_views_for_same_cell_seed_successful_step": True},
        "tuning": {"prior_experiments_exist": True,
                   "later_search_requires_new_protocol_and_equal_registered_trial_budget": True},
        "fairness": {
            "fixed_equal_total_student_updates": True, "equal_compute_claim": False,
            "equal_unlabeled_pair_visits_claim": False, "historical_checkpoints_reusable": False,
            "adaptive_results_separate_from_fixed": True, "warmup_control_results_separate_from_main": True,
            "compute_comparison_requires_same_hardware_serial_execution": True,
            "same_hyperparameter_search_budget_per_method": True,
            "three_seed_mean_and_sample_std_complete_cells_only": True},
        "publication": {"launch_enabled": False,
                        "private_inputs_weights_teacher_and_real_admission_not_distributed": True,
                        "planned_configurations_not_completed_results": True}}
    for section, fields in policies.items():
        for key, expected in fields.items():
            actual = p[section].get(key)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f"Registered scientific policy differs: {section}.{key}")
    return p


def load_protocol(path=PROTOCOL_PATH):
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")))


def make_config(crop, budget, method, seed, policy="fixed", protocol=None, regime="native"):
    p = load_protocol() if protocol is None else validate_protocol(copy.deepcopy(protocol))
    if policy not in ("fixed", "adaptive") or seed not in SEEDS or type(seed) is not int:
        raise ValueError("Unregistered policy or training seed")
    if crop not in p["datasets"] or type(budget) is not int or budget not in p["ablation_budgets_percent"]:
        raise ValueError("Unregistered dataset/label budget")
    if method not in (*ABLATION_METHODS, *SSL_METHODS):
        raise ValueError("Unregistered method")
    if budget == 100 and method not in ("bce", "pairusa"):
        raise ValueError("100% labels: no U; SSL/OT is not an applicable comparison")
    if regime not in ("native", "shared_bce"):
        raise ValueError("Unknown training regime")
    if method in ABLATION_METHODS:
        regime = "shared_bce"
    control_cell = crop in p["warmup_control"]["datasets"] and budget in p["warmup_control"]["budgets_percent"]
    if method in SSL_METHODS and regime == "shared_bce" and (not control_cell or policy != "fixed"):
        raise ValueError("Shared-BCE SSL is a preregistered fixed-budget supplementary control only")
    roles = []
    if budget in p["main_budgets_percent"] and method in p["main_methods"] and (method in ABLATION_METHODS or regime == "native"):
        roles.append("main")
    if method in p["ablation_methods"]:
        roles.append("ablation")
    if control_cell and policy == "fixed" and regime == "shared_bce" and method in p["warmup_control"]["methods"]:
        roles.append("warmup_control")
    cfg = dict(version=p["version"], protocol_sha256=fingerprint(p), dataset=crop,
               budget_percent=budget, budget=f"{budget:03d}", method=method, seed=seed,
               data_construction_seed=p["data_construction_seed"], policy=policy,
               roles=roles, regime=regime, target_steps=p["training"]["max_successful_steps"],
               shared_bce_steps=p["ablation"]["shared_bce_steps"] if regime == "shared_bce" else 0,
               checkpoint_selection="terminal_ema", tuning_trials=0,
               initialization="new_official_ALBEF4M_seeded_common_state_no_historical_parent",
               launch_enabled=False, uses_pairusa=method in ("pairusa", "pairusa_ot"),
               uses_ot=method in ("ot", "pairusa_ot"))
    cfg["run_id"] = f"fair_v3_{crop}_{budget:03d}_{method}_{regime}_{policy}_s{seed}"
    return cfg


def make_plan(policy="fixed", protocol=None):
    p = load_protocol() if protocol is None else validate_protocol(copy.deepcopy(protocol))
    runs = []
    for crop in p["datasets"]:
        for budget in p["ablation_budgets_percent"]:
            methods = (*ABLATION_METHODS, *SSL_METHODS) if budget != 100 else ("bce", "pairusa")
            for seed in p["training_seeds"]:
                runs.extend(make_config(crop, budget, method, seed, policy, p) for method in methods)
                if policy == "fixed" and budget in p["warmup_control"]["budgets_percent"]:
                    runs.extend(make_config(crop, budget, method, seed, policy, p, "shared_bce") for method in SSL_METHODS)
    return dict(version=p["version"], policy=policy, protocol_sha256=fingerprint(p),
                training_seeds=p["training_seeds"], data_construction_seed=p["data_construction_seed"],
                unique_student_configurations=len(runs), main_configurations=sum("main" in r["roles"] for r in runs),
                ablation_configurations=sum("ablation" in r["roles"] for r in runs),
                warmup_control_configurations=sum("warmup_control" in r["roles"] for r in runs),
                shared_roles_not_extra_runs=True, planned_not_completed=True,
                equal_compute_claim=False, runs=runs)


def validate_config(cfg, protocol=None):
    p = load_protocol() if protocol is None else protocol
    expected = make_config(cfg["dataset"], cfg["budget_percent"], cfg["method"], cfg["seed"], cfg["policy"], p, cfg["regime"])
    if cfg != expected:
        raise ValueError("Configuration differs from the registered fair-v3 grid")
    return cfg


def sample_indices(cfg, step, n_l, n_u, protocol=None):
    """Method-independent draw stream; data construction remains a separate seed."""
    p = load_protocol() if protocol is None else protocol
    positive_int(step, "successful step")
    if step > p["training"]["max_successful_steps"] or n_l < 16:
        raise ValueError("Invalid sampling step or insufficient distinct L anchors")
    if cfg["budget_percent"] != 100 and n_u < 32:
        raise ValueError("Insufficient distinct U pairs")
    def draw(purpose, count, size):
        payload = [cfg["dataset"], cfg["budget"], cfg["seed"], step, purpose]
        seed = int(fingerprint(payload)[:16], 16)
        return random.Random(seed).sample(range(size), count)
    return draw("L", 16, n_l), [] if cfg["budget_percent"] == 100 else draw("U", 32, n_u)


def lr_multiplier(step, protocol=None):
    p = load_protocol() if protocol is None else protocol
    positive_int(step, "successful step")
    t = p["training"]
    if step > t["max_successful_steps"]:
        raise ValueError("Step exceeds the frozen LR horizon")
    if step <= t["warmup_steps"]:
        return step / t["warmup_steps"]
    progress = (step - t["warmup_steps"]) / (t["max_successful_steps"] - t["warmup_steps"])
    return .5 * (1 + math.cos(math.pi * progress))


def auxiliary_weights(cfg, step, protocol=None):
    p = load_protocol() if protocol is None else protocol
    positive_int(step, "successful step")
    if step > p["training"]["max_successful_steps"]:
        raise ValueError("Auxiliary schedule exceeds the frozen total budget")
    a = p["ablation"]
    branch_step = step - cfg["shared_bce_steps"]
    start, end = a["auxiliary_start_in_branch"], a["auxiliary_ramp_end_in_branch"]
    progress = max(0., min(1., (branch_step - start) / (end - start)))
    return dict(pairusa=a["pairusa_max_weight"] * progress if cfg["uses_pairusa"] else 0.,
                ot=a["ot_max_weight"] * .5 * (1 - math.cos(math.pi * progress)) if cfg["uses_ot"] else 0.)
