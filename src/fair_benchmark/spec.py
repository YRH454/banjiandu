"""One source of truth for three-seed, equal-total-update ITM comparisons."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/pair_itm_fair_v2/configs/protocol.public.json"
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
    if p["version"] != "pair_itm_fair_v2" or p["training_seeds"] != list(SEEDS):
        raise ValueError("fair-v2 requires the three registered training seeds")
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
    if e["primary_model"] != "ema" or e["best_criterion"] != ["paired_accuracy", "auroc"] or e["test_inference"]:
        raise ValueError("Selection model/metric/Test policy differs")
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
    return p


def load_protocol(path=PROTOCOL_PATH):
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")))


def make_config(crop, budget, method, seed, policy="fixed", protocol=None):
    p = load_protocol() if protocol is None else validate_protocol(copy.deepcopy(protocol))
    if policy not in ("fixed", "adaptive") or seed not in SEEDS or type(seed) is not int:
        raise ValueError("Unregistered policy or training seed")
    if crop not in p["datasets"] or type(budget) is not int or budget not in p["ablation_budgets_percent"]:
        raise ValueError("Unregistered dataset/label budget")
    if method not in (*ABLATION_METHODS, *SSL_METHODS):
        raise ValueError("Unregistered method")
    if budget == 100 and method not in ("bce", "pairusa"):
        raise ValueError("100% labels: no U; SSL/OT is not an applicable comparison")
    roles = []
    if budget in p["main_budgets_percent"] and method in p["main_methods"]:
        roles.append("main")
    if method in p["ablation_methods"]:
        roles.append("ablation")
    cfg = dict(version=p["version"], protocol_sha256=fingerprint(p), dataset=crop,
               budget_percent=budget, budget=f"{budget:03d}", method=method, seed=seed,
               data_construction_seed=p["data_construction_seed"], policy=policy,
               roles=roles, target_steps=p["training"]["max_successful_steps"],
               shared_bce_steps=p["ablation"]["shared_bce_steps"] if method in ABLATION_METHODS else 0,
               initialization="new_official_ALBEF4M_seeded_common_state_no_historical_parent",
               launch_enabled=False, uses_pairusa=method in ("pairusa", "pairusa_ot"),
               uses_ot=method in ("ot", "pairusa_ot"))
    cfg["run_id"] = f"fair_v2_{crop}_{budget:03d}_{method}_{policy}_s{seed}"
    return cfg


def make_plan(policy="fixed", protocol=None):
    p = load_protocol() if protocol is None else validate_protocol(copy.deepcopy(protocol))
    runs = []
    for crop in p["datasets"]:
        for budget in p["ablation_budgets_percent"]:
            methods = (*ABLATION_METHODS, *SSL_METHODS) if budget != 100 else ("bce", "pairusa")
            for seed in p["training_seeds"]:
                runs.extend(make_config(crop, budget, method, seed, policy, p) for method in methods)
    return dict(version=p["version"], policy=policy, protocol_sha256=fingerprint(p),
                training_seeds=p["training_seeds"], data_construction_seed=p["data_construction_seed"],
                unique_student_configurations=len(runs), main_configurations=sum("main" in r["roles"] for r in runs),
                ablation_configurations=sum("ablation" in r["roles"] for r in runs),
                shared_roles_not_extra_runs=True, planned_not_completed=True,
                equal_compute_claim=False, runs=runs)


def validate_config(cfg, protocol=None):
    p = load_protocol() if protocol is None else protocol
    expected = make_config(cfg["dataset"], cfg["budget_percent"], cfg["method"], cfg["seed"], cfg["policy"], p)
    if cfg != expected:
        raise ValueError("Configuration differs from the registered fair-v2 grid")
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
