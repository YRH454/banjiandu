"""Registered two-phase adaptation plans; no data, GPU or training launch."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from functools import lru_cache

from fair_benchmark.spec import ROOT, SEEDS, SSL_METHODS, ABLATION_METHODS, fingerprint, positive_int

VERSION = "pair_itm_plateau_v4"
PROTOCOL_PATH = ROOT / "experiments/pair_itm_plateau_v4/configs/protocol.public.json"
STAGES = ("bce", "adaptation", "matched_bce")


@lru_cache(maxsize=1)
def _registered():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("version") != VERSION or protocol != _registered():
        raise ValueError("Changed/old plateau protocol; register a new version before changing scientific rules")
    return protocol


def load_protocol():
    return copy.deepcopy(validate_protocol(_registered()))


def make_config(crop, budget, method, seed, stage="adaptation", match_method=None, protocol=None):
    p = load_protocol() if protocol is None else validate_protocol(protocol)
    if crop not in p["datasets"] or type(budget) is not int or budget not in p["ablation_budgets_percent"] or type(seed) is not int or seed not in SEEDS or stage not in STAGES:
        raise ValueError("Unregistered plateau cell/stage/seed")
    methods = (*ABLATION_METHODS, *SSL_METHODS) if budget != 100 else ("bce", "pairusa")
    if method not in methods or (stage != "adaptation" and method != "bce"):
        raise ValueError("Unregistered method; 100% has no U/OT/SSL")
    if (stage == "matched_bce" and (match_method not in methods or match_method == "bce")) or (stage != "matched_bce" and match_method is not None):
        raise ValueError("Matched BCE requires a non-BCE source method; other stages have no match target")
    roles = []
    if stage == "bce":
        roles = ["bce_reference"]
    elif stage == "matched_bce":
        roles = ["matched_bce"]
    else:
        if budget in p["main_budgets_percent"] and method in p["main_methods"]:
            roles.append("main")
        if method in ABLATION_METHODS:
            roles.append("ablation")
    stem = f"plateau_v4_{crop}_{budget:03d}"
    parent_id = f"{stem}_bce_bce_s{seed}"
    suffix = f"_for_{match_method}" if match_method else ""
    cap = p["phases"]["bce" if stage == "bce" else "adaptation"]["max_successful_steps"]
    return dict(version=VERSION, protocol_sha256=fingerprint(p), dataset=crop, budget_percent=budget,
                budget=f"{budget:03d}", seed=seed, data_construction_seed=20260825, method=method, stage=stage,
                match_method=match_method, policy="matched_steps" if stage == "matched_bce" else "validation_plateau",
                regime="BCE_plateau_then_supervised_plus_module", roles=roles,
                run_id=f"{stem}_{stage}_{method}{suffix}_s{seed}", parent_run_id=None if stage == "bce" else parent_id,
                target_steps=cap, shared_bce_steps=0, checkpoint_selection="terminal_ema", tuning_trials=0,
                uses_pairusa=method in ("pairusa", "pairusa_ot"), uses_ot=method in ("ot", "pairusa_ot"),
                launch_enabled=False)


def validate_config(cfg, protocol=None):
    p = load_protocol() if protocol is None else validate_protocol(protocol)
    expected = make_config(cfg["dataset"], cfg["budget_percent"], cfg["method"], cfg["seed"],
                           cfg["stage"], cfg["match_method"], p)
    if cfg != expected:
        raise ValueError("Configuration differs from the registered plateau-v4 plan")
    return cfg


def make_plan(protocol=None):
    p = load_protocol() if protocol is None else validate_protocol(protocol)
    runs = []
    for crop in p["datasets"]:
        for budget in p["ablation_budgets_percent"]:
            methods = (*ABLATION_METHODS, *SSL_METHODS) if budget != 100 else ("bce", "pairusa")
            for seed in SEEDS:
                runs.append(make_config(crop, budget, "bce", seed, "bce", protocol=p))
                runs.extend(make_config(crop, budget, method, seed, protocol=p) for method in methods)
                runs.extend(make_config(crop, budget, "bce", seed, "matched_bce", method, p)
                            for method in methods if method != "bce")
    return dict(version=VERSION, protocol_sha256=fingerprint(p), training_seeds=list(SEEDS),
                data_construction_seed=20260825, logical_configurations=len(runs),
                bce_parent_configurations=sum(r["stage"] == "bce" for r in runs),
                adaptive_branch_configurations=sum(r["stage"] == "adaptation" for r in runs),
                matched_bce_configurations=sum(r["stage"] == "matched_bce" for r in runs),
                main_configurations=sum("main" in r["roles"] for r in runs),
                ablation_configurations=sum("ablation" in r["roles"] for r in runs),
                planned_not_completed=True, matched_targets_unknown_until_source_validation_stop=True,
                matching_BCE_trajectories_may_be_physically_reused_only_with_full_receipts=True,
                equal_steps_claim=False, equal_compute_claim=False, runs=runs)


def sample_indices(cfg, step, n_l, n_u, protocol=None):
    positive_int(step, "phase step")
    if step > cfg["target_steps"] or n_l < 16 or (cfg["budget_percent"] != 100 and n_u < 32):
        raise ValueError("Invalid sampling budget or insufficient distinct anchors")
    phase = "bce" if cfg["stage"] == "bce" else "adaptation"
    def draw(kind, count, size):
        seed = int(fingerprint([cfg["dataset"], cfg["budget"], cfg["seed"], phase, step, kind])[:16], 16)
        return random.Random(seed).sample(range(size), count)
    return draw("L", 16, n_l), [] if cfg["budget_percent"] == 100 else draw("U", 32, n_u)


def lr_multiplier(cfg, step, protocol=None):
    p = load_protocol() if protocol is None else protocol
    positive_int(step, "phase LR step")
    if step > cfg["target_steps"]:
        raise ValueError("Phase safety cap exceeded")
    warm = p["phases"]["lr_warmup_steps"]
    if step <= warm:
        return step/warm
    progress = (step-warm)/(cfg["target_steps"]-warm)
    floor = p["phases"]["lr_floor_multiplier"]
    return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))


def auxiliary_weights(cfg, step, protocol=None):
    p = load_protocol() if protocol is None else protocol
    positive_int(step, "phase auxiliary step")
    if step > cfg["target_steps"]:
        raise ValueError("Phase auxiliary safety cap exceeded")
    a = p["ablation"]
    progress = max(0., min(1., (step-a["auxiliary_start_in_branch"])/
                          (a["auxiliary_ramp_end_in_branch"]-a["auxiliary_start_in_branch"])))
    return dict(pairusa=a["pairusa_max_weight"]*progress if cfg["uses_pairusa"] else 0.,
                ot=a["ot_max_weight"]*.5*(1-math.cos(math.pi*progress)) if cfg["uses_ot"] else 0.)


def source_fingerprint():
    from fair_benchmark.references import reference_paths
    paths = set(reference_paths())
    for directory in (ROOT/"src/fair_benchmark", ROOT/"src/plateau_benchmark"):
        paths.update(x.relative_to(ROOT).as_posix() for x in directory.glob("*.py"))
    paths.update((PROTOCOL_PATH.relative_to(ROOT).as_posix(), "tools/fair_benchmark.py"))
    files = [dict(path=f, sha256=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()) for f in sorted(paths)]
    return dict(format="plateau_source_fingerprint_v4", source_sha256=fingerprint(files), files=files,
                public_code_only_not_execution_admission=True)
