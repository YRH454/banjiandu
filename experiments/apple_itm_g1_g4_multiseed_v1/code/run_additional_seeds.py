"""Two additional, isolated training seeds for the registered Apple ITM ablation.

Preparation copies the *fixed* 20260825 pair tables.  Only model/training RNG
changes.  Nothing executes on import, and training requires explicit --run.
The independent B student is intentionally omitted; its per-budget USA teacher
is still trained, with its own result and provenance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
EXPERIMENTS = HERE.parent
OLD_AB = EXPERIMENTS / "apple_itm_pairusa_random_v2"
OLD_C = EXPERIMENTS / "apple_itm_pairusa_warmstart_v1"
OLD_OT = EXPERIMENTS / "apple_itm_usa_ot_ablation_v1"
SEEDS = (20260826, 20260827)
BUDGETS = ("005", "020", "001", "010", "030", "100")
OT_BUDGETS = BUDGETS[:-1]
DATA_SEED = 20260825
SEED_PLAN = HERE / "configs/seed_plan.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"Registered file differs: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_immutable(source: Path, target: Path) -> str:
    expected = digest(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
        try:
            shutil.copy2(source, temporary)
            if digest(temporary) != expected:
                raise RuntimeError(f"Copy failed integrity check: {source}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(target) != expected:
        raise RuntimeError(f"Fixed input changed: {target}")
    return expected


def roots(seed: int) -> tuple[Path, Path, Path, Path]:
    base = HERE / f"seed_{seed}"
    return base, base / "ab", base / "c", base / "ot"


def preparation_files() -> tuple[list[Path], list[Path]]:
    ab = [Path("data/manifest.json"), Path("audit/negative_quality_gate.json"),
          Path("audit/source_preflight.json"),
          Path("reports/执行登记_v5_随机异病例完整caption.md"),
          Path("data/pairs/validation.csv")]
    ab += [Path(f"data/pairs/train_{b}.csv") for b in BUDGETS]
    ot = [Path("data/manifest.json"), Path("audit/u_data_checks.json")]
    ot += [Path(f"data/u_pairs/u_{b}.csv") for b in OT_BUDGETS]
    return ab, ot


def prepare(seed: int) -> dict:
    """Copy data only; never import the model or allocate the GPU."""
    if seed not in SEEDS:
        raise ValueError(f"Only additional seeds {SEEDS} are registered")
    plan = read(SEED_PLAN)
    if (plan["additional_training_seeds"] != list(SEEDS)
            or plan["fixed_pair_construction_seed"] != DATA_SEED
            or plan["g1_g2_budget_order"] != list(BUDGETS)
            or plan["g3_g4_budget_order"] != list(OT_BUDGETS)
            or plan["student_steps_per_stage"] != 1600
            or plan["test_evaluation"] is not False):
        raise RuntimeError("Multi-seed registered plan changed")
    base, abroot, croot, otroot = roots(seed)
    ab_files, ot_files = preparation_files()
    copied = {}
    for relative in ab_files:
        copied[f"ab/{relative.as_posix()}"] = copy_immutable(OLD_AB / relative, abroot / relative)
    for relative in ot_files:
        copied[f"ot/{relative.as_posix()}"] = copy_immutable(OLD_OT / relative, otroot / relative)
    old_ab_manifest = read(OLD_AB / "data/manifest.json")
    old_ot_manifest = read(OLD_OT / "data/manifest.json")
    if old_ab_manifest["seed"] != DATA_SEED or old_ot_manifest["seed"] != DATA_SEED:
        raise RuntimeError("The fixed pair-table construction seed changed")
    for b in OT_BUDGETS:
        entry = old_ot_manifest["budgets"][b]
        if digest(abroot / f"data/pairs/train_{b}.csv") != entry["l_pairs_sha256"]:
            raise RuntimeError(f"OT and L pair tables disagree for {b}")
        if digest(otroot / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"OT U table differs for {b}")
    c_plan = read(OLD_C / "configs/plan.json")
    c_plan.update(parent_root=str(abroot), teacher="same_seed_teacher_without_independent_B_student")
    write_once(croot / "configs/plan.json", c_plan)
    ot_plan = read(OLD_OT / "configs/plan.json")
    ot_plan["seed"] = seed
    write_once(otroot / "configs/plan.json", ot_plan)
    receipt = {
        "training_seed": seed,
        "pair_table_construction_seed": DATA_SEED,
        "fixed_input_sha256": copied,
        "engine_sha256": {f"ab/{name}": digest(OLD_AB / "code" / name)
                          for name in ("train_queue.py", "pair_model.py", "prepare_random_pairs.py")}
                         | {f"ot/{name}": digest(OLD_OT / "code" / name)
                            for name in ("train_queue.py", "ot_loss.py", "prepare_u_pairs.py")}
                         | {"c/run_warmstart.py": digest(OLD_C / "code/run_warmstart.py")},
        "runner_sha256": digest(Path(__file__)),
        "seed_plan_sha256": digest(SEED_PLAN),
        "plan_sha256": {"c": digest(croot / "configs/plan.json"),
                        "ot": digest(otroot / "configs/plan.json")},
        "scope": "G1/G2 six budgets, G3/G4 five budgets; no independent B student, no test set",
    }
    write_once(base / "audit/preparation_receipt.json", receipt)
    return receipt


def check_prepared(seed: int) -> dict:
    base, abroot, croot, otroot = roots(seed)
    receipt = read(base / "audit/preparation_receipt.json")
    if receipt["training_seed"] != seed or receipt["runner_sha256"] != digest(Path(__file__)):
        raise RuntimeError("Seed runner/receipt mismatch")
    if receipt["seed_plan_sha256"] != digest(SEED_PLAN):
        raise RuntimeError("Registered multi-seed plan changed")
    for relative, expected in receipt["fixed_input_sha256"].items():
        if digest(base / relative) != expected:
            raise RuntimeError(f"Prepared input changed: {relative}")
    for relative, expected in receipt["engine_sha256"].items():
        source = {"ab": OLD_AB, "c": OLD_C, "ot": OLD_OT}[relative.split("/", 1)[0]]
        if digest(source / "code" / relative.split("/", 1)[1]) != expected:
            raise RuntimeError(f"Registered engine changed: {relative}")
    if read(croot / "configs/plan.json")["parent_root"] != str(abroot):
        raise RuntimeError("G2 parent root changed")
    if read(otroot / "configs/plan.json")["seed"] != seed:
        raise RuntimeError("OT plan seed changed")
    for branch, expected in receipt["plan_sha256"].items():
        if digest({"c": croot, "ot": otroot}[branch] / "configs/plan.json") != expected:
            raise RuntimeError(f"Registered {branch} plan changed")
    numerical = read(OLD_OT / "audit/ot_numerical_checks.json")
    if (not numerical.get("passed") or numerical.get("source_sha256", {}).get("ot_loss.py")
            != digest(OLD_OT / "code/ot_loss.py")):
        raise RuntimeError("OT solver is not bound to its passing numerical audit")
    return receipt


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def gpu_lock(seed: int):
    """Share the original A/B GPU lock with any still-running historical queue."""
    import msvcrt
    base = roots(seed)[0]
    base.mkdir(parents=True, exist_ok=True)
    streams = []
    try:
        for path in (OLD_AB / ".queue.lock", base / ".queue.lock"):
            stream = path.open("a+b")
            streams.append(stream)
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b" ")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        yield
    finally:
        for stream in reversed(streams):
            try:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            stream.close()


def load_engines(seed: int):
    _, abroot, _, otroot = roots(seed)
    sys.path.insert(0, str(OLD_AB / "code"))
    ab = load_module(f"apple_ab_seed_{seed}", OLD_AB / "code/train_queue.py")
    ab.ROOT, ab.OUTPUTS, ab.SEED = abroot, abroot / "outputs", seed
    # Fingerprint the immutable, historical engine; the new runner is appended below.
    ab.CODE = OLD_AB / "code"
    sys.path.insert(0, str(ab.CORE))
    ot = load_module(f"apple_ot_seed_{seed}", OLD_OT / "code/train_queue.py")
    ot.ROOT, ot.AB, ot.C_ROOT = otroot, abroot, roots(seed)[2]
    ot.OUTPUTS, ot.SEED, ot.parent = otroot / "outputs", seed, ab
    return ab, ot


def data_and_source(ab, seed: int) -> tuple[dict, dict]:
    manifest = ab.load_manifest(require_gate=True)
    ab.configure_files(manifest)
    source = ab.source_fingerprint(manifest)
    source["additional_seed_runner_sha256"] = digest(Path(__file__))
    source["training_seed"] = seed
    source["fixed_pair_construction_seed"] = DATA_SEED
    if ab.md5(Path(ab.model_config()["checkpoint"])) != ab.model_config()["checkpoint_md5_expected"]:
        raise RuntimeError("ALBEF pretrained checkpoint changed")
    return manifest, source


def pipeline(path: Path, run_ids: list[str], fingerprint: dict) -> dict:
    if path.exists():
        state = read(path)
        if state["run_ids"] != run_ids or state["source_fingerprint"] != fingerprint:
            raise RuntimeError(f"Queue identity changed: {path}")
        return state
    state = {"state": "starting", "run_ids": run_ids, "source_fingerprint": fingerprint,
             "active_run": None, "independent_test_evaluation": False}
    write_once(path, state)
    return state


def update_state(ab, path: Path, state: dict, **changes) -> None:
    state.update(changes, updated_utc=ab.now())
    ab.atomic_json(path, state)


def validated_rows(ab, manifest: dict, budget: str):
    train = ab.read_pairs(ab.ROOT / manifest["files"]["budgets"][budget])
    val = ab.read_pairs(ab.ROOT / manifest["files"]["validation"])
    if len(train) != ab.EXPECTED_COUNTS[budget] or len(val) != 400:
        raise RuntimeError(f"L/validation counts changed for {budget}")
    train_ids = {r["image_id"] for r in train}
    val_ids = {r["image_id"] for r in val}
    allowed_train = ab.source_ids(ab.V2_DATA / "budgets" / f"train_{budget}.csv")
    allowed_val = ab.source_ids(ab.V2_DATA / "validation.csv")
    if not train_ids <= allowed_train or not val_ids <= allowed_val or train_ids & val_ids:
        raise RuntimeError(f"Split leakage or boundary change for {budget}")
    return train, val


def run_g1_and_teacher(ab, seed: int, manifest: dict, source: dict, tokenizer) -> None:
    configs = ab.make_configs(manifest)
    by_key = {(c["budget"], c["method"]): c for c in configs}
    ids = [by_key[b, "A_bce"]["run_id"] for b in BUDGETS]
    path = ab.OUTPUTS / "pipeline_state.json"
    state = pipeline(path, ids, source)
    for b in BUDGETS:
        check_prepared(seed)
        a_cfg, teacher_cfg = by_key[b, "A_bce"], by_key[b, "B_bce_pairusa"]
        update_state(ab, path, state, state="running", active_run=a_cfg["run_id"])
        train, val = validated_rows(ab, manifest, b)
        ab.run_student(a_cfg, train, val, tokenizer, source)
        update_state(ab, path, state, state="running", active_run=f"teacher_l{b}_s{seed}")
        ab.train_teacher(teacher_cfg, train, val, tokenizer, source)
    update_state(ab, path, state, state="completed_validation", active_run=None)


def checked_parent(ab, seed: int, budget: str, source: dict) -> tuple[dict, dict, dict]:
    a_id = f"apple_itm_random_A_bce_l{budget}_s{seed}"
    a_dir = ab.OUTPUTS / a_id
    a_result = read(a_dir / "result.json")
    a_best = a_dir / "best.pt"
    if (a_result["state"] != "completed_validation"
            or a_result["provenance"]["base"]["source"] != source
            or a_result["provenance"]["base"]["run"] != read(ab.ROOT / "configs" / f"{a_id}.json")):
        raise RuntimeError(f"G1 result provenance differs for {budget}")
    teacher_dir = ab.OUTPUTS / f"teacher_l{budget}_s{seed}"
    teacher_result = read(teacher_dir / "result.json")
    teacher_prov = teacher_result["provenance"]
    if (teacher_result["state"] != "completed" or teacher_prov["source"] != source
            or teacher_prov["budget"] != budget
            or teacher_prov["train_pair_file_sha256"] != digest(ab.ROOT / f"data/pairs/train_{budget}.csv")
            or teacher_prov["validation_pair_file_sha256"] != digest(ab.ROOT / "data/pairs/validation.csv")):
        raise RuntimeError(f"USA teacher provenance differs for {budget}")
    files = {str(teacher_dir / name): digest(teacher_dir / name)
             for name in ("best.pt", "positive_targets.pt")}
    return a_result, {"path": str(a_best), "sha256": digest(a_best)}, {
        "dir": str(teacher_dir), "result_sha256": digest(teacher_dir / "result.json"), "artifacts": files}


def g2_config(ab, seed: int, budget: str, source: dict) -> dict:
    a_result, a_best, teacher = checked_parent(ab, seed, budget, source)
    b_id = f"apple_itm_random_B_bce_pairusa_l{budget}_s{seed}"
    cfg = copy.deepcopy(read(ab.ROOT / "configs" / f"{b_id}.json"))
    cfg["run_id"] = f"apple_itm_random_C_from_A_best_pairusa_l{budget}_s{seed}"
    cfg["method"] = "B_bce_pairusa_warmstart_A_best"
    cfg["variant"] = "G2_A_best_then_BCE_PairUSA"
    audit_path = roots(seed)[2] / "outputs" / cfg["run_id"] / "initialization_audit.json"
    cfg["warmstart"] = {
        "a_run_id": a_result["run_id"], "a_best_path": a_best["path"],
        "a_best_sha256": a_best["sha256"], "a_best_step": a_result["best_step"],
        "a_result_sha256": digest(ab.OUTPUTS / a_result["run_id"] / "result.json"),
        "teacher_artifacts": teacher["artifacts"], "teacher_result_sha256": teacher["result_sha256"],
        "source_fingerprint": source, "additional_steps": 1600,
        "optimizer_state": "fresh_for_new_phase", "independent_B_student": "not_trained"}
    cfg["model"]["warmstart"] = {"a_run_id": a_result["run_id"], "a_best_path": a_best["path"],
                                  "a_best_sha256": a_best["sha256"], "a_best_step": a_result["best_step"],
                                  "audit_path": str(audit_path)}
    if cfg["training"]["max_steps"] != 1600 or cfg["model"]["seed"] != seed:
        raise RuntimeError("G2 training definition changed")
    write_once(roots(seed)[2] / "configs" / f"{cfg['run_id']}.json", cfg)
    return cfg


def run_g2(ab, seed: int, manifest: dict, source: dict, tokenizer) -> None:
    import torch
    import pair_model
    croot = roots(seed)[2]
    configs = [g2_config(ab, seed, b, source) for b in BUDGETS]
    expected_configs = {cfg["budget"]: cfg for cfg in configs}
    identity = {"parent": source, "runner_sha256": digest(Path(__file__)),
                "plan_sha256": digest(croot / "configs/plan.json")}
    path = croot / "outputs/pipeline_state.json"
    state = pipeline(path, [c["run_id"] for c in configs], identity)
    original_model, original_teacher, original_outputs = pair_model.PairITMModel, ab.train_teacher, ab.OUTPUTS

    class ABestWarmStartedModel(original_model):
        def __init__(self, config):
            super().__init__(config)
            spec = config["warmstart"]
            if digest(Path(spec["a_best_path"])) != spec["a_best_sha256"]:
                raise RuntimeError("A-best checkpoint changed before G2 initialization")
            selected = torch.load(spec["a_best_path"], map_location="cpu", weights_only=False)
            parent_result = read(Path(spec["a_best_path"]).parent / "result.json")
            if (selected["provenance"] != parent_result["provenance"]
                    or selected["step"] != spec["a_best_step"]
                    or selected["validation"] != parent_result["best_validation"]):
                raise RuntimeError("A-best checkpoint/result mismatch")
            initial = self.trainable_state()
            a_state = selected["trainable_state"]
            extra = set(initial) - set(a_state)
            expected = {"student_projection.0.weight", "student_projection.0.bias",
                        "student_projection.1.weight", "student_projection.1.bias", "log_student_temperature"}
            if len(a_state) != 34 or extra != expected or set(a_state) - set(initial):
                raise RuntimeError("G2 34+5 trainable-key topology mismatch")
            fresh = {key: initial[key].clone() for key in extra}
            self.load_trainable_state({**initial, **a_state})
            loaded = self.trainable_state()
            if any(not torch.equal(loaded[k], value) for k, value in a_state.items()):
                raise RuntimeError("G2 failed exact A-best weight transfer")
            if any(not torch.equal(loaded[k], value) for k, value in fresh.items()):
                raise RuntimeError("G2 fresh USA weights changed during transfer")
            def tensor_hash(values):
                return hashlib.sha256(b"".join(v.contiguous().numpy().tobytes()
                    for _, v in sorted(values.items()))).hexdigest()
            write_once(Path(spec["audit_path"]), {
                "exact_copy": True, "copied_keys": 34, "fresh_keys": sorted(extra),
                "a_best_sha256": spec["a_best_sha256"], "a_best_step": selected["step"],
                "a_trainable_sha256": tensor_hash(a_state),
                "loaded_shared_sha256": tensor_hash({k: loaded[k] for k in a_state}),
                "fresh_usa_sha256": tensor_hash(fresh), "initialization": "A_best_weights_only"})

    def existing_teacher(cfg, train_rows, val_rows, _tokenizer, fingerprint):
        if fingerprint != source or expected_configs.get(cfg["budget"]) != cfg:
            raise RuntimeError("G2/teacher configuration changed")
        target = original_outputs / f"teacher_l{cfg['budget']}_s{seed}" / "best.pt"
        if digest(target) != cfg["warmstart"]["teacher_artifacts"][str(target)]:
            raise RuntimeError("G2 teacher changed")
        return target

    try:
        pair_model.PairITMModel = ABestWarmStartedModel
        ab.train_teacher, ab.OUTPUTS = existing_teacher, croot / "outputs"
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for cfg in configs:
            check_prepared(seed)
            update_state(ab, path, state, state="running", active_run=cfg["run_id"])
            train, val = validated_rows(ab, manifest, cfg["budget"])
            result = ab.run_student(cfg, train, val, tokenizer, source)
            audit = read(croot / "outputs" / cfg["run_id"] / "initialization_audit.json")
            if result["initial_shared_sha256"] != audit["a_trainable_sha256"]:
                raise RuntimeError("G2 initialization audit differs from trainer")
        update_state(ab, path, state, state="completed_validation", active_run=None)
    finally:
        pair_model.PairITMModel = original_model
        ab.train_teacher, ab.OUTPUTS = original_teacher, original_outputs


def check_u_data(otroot: Path, abroot: Path) -> dict:
    manifest = read(otroot / "data/manifest.json")
    gate = read(otroot / "audit/u_data_checks.json")
    if not gate["passed"] or gate["manifest_sha256"] != digest(otroot / "data/manifest.json"):
        raise RuntimeError("U structural audit does not bind the fixed manifest")
    if manifest["seed"] != DATA_SEED or manifest["preparation_code_sha256"] != digest(OLD_OT / "code/prepare_u_pairs.py"):
        raise RuntimeError("U construction protocol changed")
    for path, expected in manifest["source_sha256"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"U source changed: {path}")
    for b in OT_BUDGETS:
        entry = manifest["budgets"][b]
        if (digest(otroot / entry["path"]) != entry["sha256"]
                or digest(abroot / f"data/pairs/train_{b}.csv") != entry["l_pairs_sha256"]):
            raise RuntimeError(f"U/L fixed table changed for {b}")
    return manifest


def ot_configs(ab, seed: int, source: dict, manifest: dict) -> list[dict]:
    _, abroot, croot, otroot = roots(seed)
    plan = read(otroot / "configs/plan.json")
    if (plan["seed"] != seed or plan["budget_order"] != list(OT_BUDGETS)
            or plan["variants"] != ["G3", "G4"] or plan["additional_steps"] != 1600
            or plan["test_evaluation"] or plan["extra_BCE_continuation"]):
        raise RuntimeError("OT plan differs from registered G3/G4 protocol")
    configs = []
    for b in OT_BUDGETS:
        a_result, a_best, teacher = checked_parent(ab, seed, b, source)
        c_id = f"apple_itm_random_C_from_A_best_pairusa_l{b}_s{seed}"
        c_dir = croot / "outputs" / c_id
        c_result = read(c_dir / "result.json")
        c_cfg = read(croot / "configs" / f"{c_id}.json")
        if (c_result["state"] != "completed_validation"
                or c_result["provenance"]["base"]["source"] != source
                or c_result["provenance"]["base"]["run"] != c_cfg
                or c_cfg["warmstart"]["a_best_sha256"] != a_best["sha256"]):
            raise RuntimeError(f"G2 source differs for {b}")
        entry = manifest["budgets"][b]
        warm = {"a_best_path": a_best["path"], "a_best_sha256": a_best["sha256"],
                "a_best_step": a_result["best_step"], "a_run_id": a_result["run_id"],
                "c_run_id": c_id, "a_result_sha256": digest(abroot / "outputs" / a_result["run_id"] / "result.json"),
                "c_result_sha256": digest(c_dir / "result.json"),
                "c_initialization_audit": str(c_dir / "initialization_audit.json"),
                "c_initialization_sha256": digest(c_dir / "initialization_audit.json"),
                "teacher_artifacts": teacher["artifacts"],
                "teacher_result_sha256": teacher["result_sha256"]}
        b_id = f"apple_itm_random_B_bce_pairusa_l{b}_s{seed}"
        template = read(abroot / "configs" / f"{b_id}.json")
        for variant in ("G3", "G4"):
            model = copy.deepcopy(template["model"])
            model["use_pairusa"] = variant == "G4"
            model["fusion_chunk_size"] = plan["initial_physical_batch"]
            cfg = {"run_id": f"apple_itm_{variant}_from_A_best_{'ot' if variant == 'G3' else 'usa_ot'}_l{b}_s{seed}",
                   "variant": variant, "budget": b, "seed": seed,
                   "model": model, "training": copy.deepcopy(template["training"]),
                   "image_root": manifest["image_root"],
                   "pair_file": str(abroot / f"data/pairs/train_{b}.csv"),
                   "validation_file": str(abroot / "data/pairs/validation.csv"),
                   "u_file": str(otroot / entry["path"]), "u_sha256": entry["sha256"],
                   "u_n_images": entry["n_images"], "u_n_pairs": entry["n_pairs"],
                   "warmstart": warm, "ot": plan["ot"], "u_strong": plan["u_strong"],
                   "u_weak": plan["u_weak"], "unlabeled_batch": 32,
                   "max_cpu_prefix_cache_gib": plan["max_cpu_prefix_cache_gib"],
                   "caption_protocol": plan["caption_protocol"], "test_evaluation": False}
            write_once(otroot / "configs" / f"{cfg['run_id']}.json", cfg)
            configs.append(cfg)
    return configs


def run_ot(ab, ot, seed: int, source: dict) -> None:
    import torch
    _, abroot, croot, otroot = roots(seed)
    manifest = check_u_data(otroot, abroot)
    configs = ot_configs(ab, seed, source, manifest)
    fp = {"additional_seed_runner_sha256": digest(Path(__file__)),
          "ab_source_fingerprint": source,
          "ot_plan_sha256": digest(otroot / "configs/plan.json"),
          "u_manifest_sha256": digest(otroot / "data/manifest.json"),
          "u_gate_sha256": digest(otroot / "audit/u_data_checks.json"),
          "ot_engine_sha256": digest(OLD_OT / "code/train_queue.py"),
          "ot_solver_sha256": digest(OLD_OT / "code/ot_loss.py"),
          "g2_results": {b: digest(croot / "outputs" / f"apple_itm_random_C_from_A_best_pairusa_l{b}_s{seed}" / "result.json")
                         for b in OT_BUDGETS}}
    path = otroot / "outputs/pipeline_state.json"
    state = pipeline(path, [c["run_id"] for c in configs], fp)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for cfg in configs:
        check_prepared(seed)
        warm = cfg["warmstart"]
        for artifact, expected in (
            (warm["a_best_path"], warm["a_best_sha256"]),
            (abroot / "outputs" / warm["a_run_id"] / "result.json", warm["a_result_sha256"]),
            (croot / "outputs" / warm["c_run_id"] / "result.json", warm["c_result_sha256"]),
            (warm["c_initialization_audit"], warm["c_initialization_sha256"]),
            (abroot / "outputs" / f"teacher_l{cfg['budget']}_s{seed}" / "result.json",
             warm["teacher_result_sha256"]),
            *warm["teacher_artifacts"].items(),
        ):
            if digest(Path(artifact)) != expected:
                raise RuntimeError(f"OT parent artifact changed: {artifact}")
        update_state(ab, path, state, state="running", active_run=cfg["run_id"])
        ot.run_one(cfg, fp)
    update_state(ab, path, state, state="completed_validation", active_run=None)


def run(seed: int) -> None:
    import torch
    check_prepared(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Registered ITM ablation requires CUDA")
    try:
        with gpu_lock(seed):
            ab, ot = load_engines(seed)
            manifest, source = data_and_source(ab, seed)
            from albef_ssl.model import get_tokenizer
            tokenizer = get_tokenizer(ab.model_config()["tokenizer_path"])
            torch.set_num_threads(8)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
            run_g1_and_teacher(ab, seed, manifest, source, tokenizer)
            run_g2(ab, seed, manifest, source, tokenizer)
            run_ot(ab, ot, seed, source)
    except BaseException as exc:
        path = roots(seed)[0] / "audit/launcher_failure.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seed": seed, "error": repr(exc),
                                    "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Copy/check fixed inputs; never train")
    mode.add_argument("--check", action="store_true", help="Read-only prepared-input check")
    mode.add_argument("--run", action="store_true", help="Explicitly start/resume the full seed queue")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.seed), ensure_ascii=False, indent=2))
    elif args.check:
        print(json.dumps(check_prepared(args.seed), ensure_ascii=False, indent=2))
    else:
        run(args.seed)


if __name__ == "__main__":
    main()
