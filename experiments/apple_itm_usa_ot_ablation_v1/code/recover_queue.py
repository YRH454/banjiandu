# -*- coding: utf-8 -*-
"""Explicit, audited continuation across the equivalent OT solver repair.

Default --check is read-only. Only --run wraps the active checkpoint and
starts the remaining six registered runs. Historical results stay untouched.
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
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
OUTPUTS = ROOT / "outputs"
BACKUP = ROOT / "audit/recovery_backup_20260930"
SNAPSHOT = ROOT / "audit/recovery_source_snapshot_20260930.json"
MIGRATION = ROOT / "audit/recovery_migration_20260930.json"
VERIFICATION = ROOT / "audit/recovery_verification_20260930.json"
ACTIVE = "apple_itm_G3_from_A_best_ot_l001_s20260825"
ORIGINAL_LAST_SHA = "b7b29b0dc50c4817367317c8c9a1064839cf06ba162207831e83bbcf97296ac7"
ORIGINAL_BEST_SHA = "0e6e2a7364913c9327bb8f9ae66d18e2074ccaa79fa6853dc98f370da0809e35"
LEGACY_SOLVER_SHA = "741b151bbda45ec84d9f4ce575c8632fe81cf4b3344d5e727e191f16a7c6171b"
RECOVERY_CODE = ("recover_queue.py", "prepare_recovery_migration.py", "verify_recovery.py", "verify_training.py")
COMPLETED_ORDER = (
    "apple_itm_G3_from_A_best_ot_l005_s20260825",
    "apple_itm_G4_from_A_best_usa_ot_l005_s20260825",
    "apple_itm_G3_from_A_best_ot_l020_s20260825",
    "apple_itm_G4_from_A_best_usa_ot_l020_s20260825",
)
PINNED_COMPLETED = {
    COMPLETED_ORDER[0]: {
        "best.pt": "4aa34e395a23c773c38a0b7827eeae9e76961040313667937220fa5c449708ec",
        "last.pt": "b52dfe3376f07cb5ceb97992fe0d4ec135528c36edd4b39d4bedd1c02b6996d1",
        "result.json": "194f8205c2235296585dc820cfe43bd0a143722bf31ee381167a842ebc16533b",
        "status.json": "9e98c0c6fa0bc7f7cfd918a263a0979d46c95cefcfb262d090104f4e5041d2aa",
        "train.jsonl": "545e08bcb20dc37dc6ada9a461c402eeb9cad4c1a31296a496d7295b31bb0fb1",
        "validation.jsonl": "1249393354b55f49288183685bcdf8a14e7fa40d450fe0968e2834f9ab9ef36e",
        "best_validation_predictions.csv": "3527b5e879543cd9c293624561f4942d2166db1271c6eca183e8947b94151c2c",
        "initialization_audit.json": "4396c2e586d95f426ac0cdd29005b25f3fb2afc2dc950a32b753feb1f9c65cb3",
    },
    COMPLETED_ORDER[1]: {
        "best.pt": "6a2b49f8bf1ead3767f565a79e6db9636d0f2e140f6396692deb13b2b96f5200",
        "last.pt": "67371b3aba2a67edb790613eb20017c5215bd8720e615c7e02ecb07c77e5e428",
        "result.json": "7636219dcd883f1098bc4ec3860c4f42d32592a240dba44b68c1c95f864883bd",
        "status.json": "8185bdcafe864bdd7e9d44e391959c0ff2a0f7db01049fbe34152721c259560a",
        "train.jsonl": "a0450d78e3ab78e009f0dcb60f2c094e4baba1b6b50e4eca3fceaec06dd07f13",
        "validation.jsonl": "6a19ca958043555d62d73bef8ab269038d3b7cb1c980c3e50dfb3ec759fd097f",
        "best_validation_predictions.csv": "73a6cade83d1a66426ffa3e30e66fcab94b99908a7520a8367f548aaa428992a",
        "initialization_audit.json": "f9f4614f60561c6d1029a08007b986e9488bcc62b165622b22cc619eeefde641",
    },
    COMPLETED_ORDER[2]: {
        "best.pt": "da1b95adff24c3b3fda9c83d1c71a0d0f778cec87b30ffadbdfa07a53bf60821",
        "last.pt": "31d35c20302c70b5d653f765f463cee8121db7b6379cd9ea172056476993e9ca",
        "result.json": "51f62d24040cecd37724951db24c69324b3466a4a7e80d40fd7415782744f3ba",
        "status.json": "0971aaf5d16eb97409d679642e29893166781c272ea62a96fc9ac94e87225a05",
        "train.jsonl": "66a44cfdec0d7bf6772a2a27c79e5a29ad99be7a672d82fed19c5385972e2151",
        "validation.jsonl": "1022dfc90b2dbead9b409a2ab05859ee113b6babb89388b535c31d7fd70cecad",
        "best_validation_predictions.csv": "76ebdfa587ab369f8ba36121586a0fb4a62afd2b6749e9c66cf92306bddbf594",
        "initialization_audit.json": "0d0b7ec013b9d8cff126f1cbad1b96385fc556b9d1a09e06e232ed8a68086b07",
    },
    COMPLETED_ORDER[3]: {
        "best.pt": "5a1e4f223729acbb4d23c38f4edc60ec21f699538d1ea3fa52a8a91a3ab1d9dc",
        "last.pt": "3968f1131a7b5fb93bb28b51dee19dc88833ae5b5d2893998168794d36cdc8cb",
        "result.json": "ea70896bd341f3074cb5eed1a01f8611b60cbdb111c34f5865aac94fa6d4c86b",
        "status.json": "8a410d9ad206f261b16fd8342b86b8f58e17861bb99bca41c2c53aa65088f5a4",
        "train.jsonl": "e299dc6cd5f7d69dac282694b6ca570c8153e019111bbd2bce96dbcad1cd00ed",
        "validation.jsonl": "c954ca988cd94e572734a021c30626786cf1b16a45848cdbe1966106d02c1b84",
        "best_validation_predictions.csv": "935a3ed532e3c6b58e4e2e8b5809ccd7925c0f2fdb43effa4a29474023e474f5",
        "initialization_audit.json": "00db4e833a5dae07537d211f332a2d8be6893a3825485cbfd93a03a1e53e47e5",
    },
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Immutable recovery artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def checked_copy(source, target, expected=None):
    source, target = Path(source), Path(target)
    expected = expected or sha256(source)
    if sha256(source) != expected:
        raise RuntimeError(f"Backup source differs: {source}")
    if target.exists():
        if sha256(target) != expected:
            raise RuntimeError(f"Backup target differs: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if sha256(target) != expected:
        raise RuntimeError(f"Backup verification failed: {target}")
    return expected


def load_queue():
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    spec = importlib.util.spec_from_file_location("recovery_parent_training_queue", CODE / "train_queue.py")
    queue = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = queue
    spec.loader.exec_module(queue)
    return queue


def assert_equal_tree(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.cpu(), b.cpu()):
            raise RuntimeError(f"Non-provenance tensor changed: {path}")
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or not np.array_equal(a, b):
            raise RuntimeError(f"RNG array changed: {path}")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise RuntimeError(f"State dictionary changed: {path}")
        for key in a:
            assert_equal_tree(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, type(a)) or len(a) != len(b):
            raise RuntimeError(f"State sequence changed: {path}")
        for index, (first, second) in enumerate(zip(a, b)):
            assert_equal_tree(first, second, f"{path}/{index}")
    elif a != b:
        raise RuntimeError(f"State scalar changed: {path}")


def verify_completed(legacy_fp):
    diagnostics = {}
    for run_id in COMPLETED_ORDER:
        directory = OUTPUTS / run_id
        actual_files = {p.name for p in directory.iterdir() if p.is_file()}
        if actual_files != set(PINNED_COMPLETED[run_id]):
            raise RuntimeError(f"Unexpected artifact set in historical complete run: {run_id}")
        for name, expected in PINNED_COMPLETED[run_id].items():
            if sha256(directory / name) != expected:
                raise RuntimeError(f"Historical complete artifact changed: {run_id}/{name}")
        result = read_json(directory / "result.json")
        if result["state"] != "completed_validation" or result["final_step"] != 1600 or result["provenance"]["source"] != legacy_fp:
            raise RuntimeError("Historical result source/complete state differs")
        iterations = []
        for line in (directory / "train.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["diagnostics"]["active"]:
                iterations.append(row["diagnostics"]["iterations"])
        if not iterations or max(iterations) > 40:
            raise RuntimeError("Historical completed run may have crossed the unchanged first-50 solver path")
        diagnostics[run_id] = {"files": len(actual_files), "final_step": 1600,
                               "max_ot_iterations": max(iterations), "old_source_preserved": True}
    return diagnostics


def recovery_fingerprint(queue, migration):
    base = queue.fingerprint()
    if base != migration["new_base_fingerprint"]:
        raise RuntimeError("Repaired training sources changed after migration registration")
    fp = dict(base)
    for relative, expected in migration["recovery_code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Recovery implementation changed: {relative}")
        fp[relative] = expected
    fp[MIGRATION.relative_to(ROOT).as_posix()] = sha256(MIGRATION)
    return fp


def verified_inputs(queue):
    snapshot, migration = read_json(SNAPSHOT), read_json(MIGRATION)
    if sha256(SNAPSHOT) != migration["snapshot_sha256"]:
        raise RuntimeError("Recovery source snapshot changed")
    for relative, expected in snapshot["backups"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Archived original artifact changed: {relative}")
    if snapshot["completed_artifacts"] != PINNED_COMPLETED or migration["legacy_fingerprint"] != snapshot["legacy_fingerprint"]:
        raise RuntimeError("Recovery historical allowlist changed")
    completed = verify_completed(snapshot["legacy_fingerprint"])
    for relative, expected in migration["evidence_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Recovery numerical/source evidence changed: {relative}")
    _, configs, base_fp = queue.prepare()
    if base_fp != migration["new_base_fingerprint"]:
        raise RuntimeError("Source fingerprints differ after preparing immutable configurations")
    if [c["run_id"] for c in configs] != snapshot["run_ids"] or tuple(snapshot["run_ids"][:4]) != COMPLETED_ORDER:
        raise RuntimeError("Registered queue order changed")
    fp = recovery_fingerprint(queue, migration)
    return snapshot, migration, configs, fp, completed


def wrap_active_checkpoint(queue, cfg, fp, snapshot):
    path = OUTPUTS / ACTIVE / "last.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    provenance = {"source": fp, "run": cfg, "test_evaluation": False}
    if checkpoint["provenance"] == provenance:
        queue.validate_resume_checkpoint(checkpoint, provenance, checkpoint["trainable_state"])
        receipt_path = ROOT / "audit/recovery_checkpoint_migration_receipt_20260930.json"
        if not receipt_path.is_file():
            raise RuntimeError("Recovered checkpoint is missing its required migration receipt")
        receipt = read_json(receipt_path)
        if (receipt.get("source_sha256") != ORIGINAL_LAST_SHA or receipt.get("target_provenance") != provenance
                or receipt.get("source_provenance") != snapshot["active"]["original_provenance"]
                or receipt.get("durable_step") != 100 or receipt.get("all_nonprovenance_state_exact") is not True
                or receipt.get("migration_manifest_sha256") != sha256(MIGRATION)
                or sha256(BACKUP / "active/last.pt") != ORIGINAL_LAST_SHA):
            raise RuntimeError("Recovery receipt/archived source lineage does not match")
        if checkpoint["step"] == 100:
            lineage = checkpoint.get("recovery_lineage", {})
            if (sha256(path) != receipt.get("migrated_sha256")
                    or lineage.get("source_checkpoint_sha256") != ORIGINAL_LAST_SHA
                    or lineage.get("source_provenance") != snapshot["active"]["original_provenance"]
                    or lineage.get("migration_manifest_sha256") != sha256(MIGRATION)):
                raise RuntimeError("Initial wrapped last100 differs from its migration receipt")
        return {"already_migrated": True, "durable_step": checkpoint["step"], "sha256": sha256(path)}
    if sha256(path) != ORIGINAL_LAST_SHA or checkpoint["step"] != 100 or checkpoint["provenance"]["source"] != snapshot["legacy_fingerprint"]:
        raise RuntimeError("Only the exact registered legacy step100 may be migrated")
    original_provenance = copy.deepcopy(checkpoint["provenance"])
    if original_provenance["run"] != cfg or original_provenance["test_evaluation"] is not False:
        raise RuntimeError("Checkpoint configuration changed")
    queue.validate_resume_checkpoint(checkpoint, original_provenance, checkpoint["trainable_state"])
    checked_copy(path, BACKUP / "active/last.pt", ORIGINAL_LAST_SHA)
    changed = copy.deepcopy(checkpoint)
    changed["provenance"] = provenance
    changed["recovery_lineage"] = {"source_checkpoint_sha256": ORIGINAL_LAST_SHA,
        "source_provenance": original_provenance, "source_durable_step": 100,
        "migration_manifest_sha256": sha256(MIGRATION), "scope": "metadata_only_exact_state_preserved"}
    queue.validate_resume_checkpoint(changed, provenance, checkpoint["trainable_state"])
    assert_equal_tree({k: v for k, v in checkpoint.items() if k != "provenance"},
                      {k: v for k, v in changed.items() if k not in {"provenance", "recovery_lineage"}})
    temporary = path.with_name("last.numeric_recovery_pending.pt")
    if temporary.exists():
        assert_equal_tree(changed, torch.load(temporary, map_location="cpu", weights_only=False))
    else:
        queue.atomic_torch(temporary, changed)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("New wrapped checkpoint failed ZIP CRC")
    readback = torch.load(temporary, map_location="cpu", weights_only=False)
    assert_equal_tree(changed, readback)
    new_sha = sha256(temporary)
    receipt = {"source_sha256": ORIGINAL_LAST_SHA, "migrated_sha256": new_sha,
               "durable_step": 100, "all_nonprovenance_state_exact": True,
               "source_provenance": original_provenance, "target_provenance": provenance,
               "migration_manifest_sha256": sha256(MIGRATION)}
    immutable_json(ROOT / "audit/recovery_checkpoint_migration_receipt_20260930.json", receipt)
    os.replace(temporary, path)
    return receipt


def archive_failure(run_id):
    if run_id in COMPLETED_ORDER or Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise RuntimeError("Failure archiving may only address a nonhistorical registered run directory")
    path = OUTPUTS / run_id / "failure.json"
    if not path.exists():
        return None
    digest = sha256(path)
    target = path.parent / "recovery_backups" / f"archived_failure_{digest}.json"
    checked_copy(path, target, digest)
    path.unlink()  # Exact checked backup exists; prevents obsolete failure being reported as active.
    return {"run_id": run_id, "path": str(target.relative_to(ROOT)), "sha256": digest}


def run_recovery(queue):
    with queue.queue_lock(), queue.parent.prevent_system_sleep():
        snapshot, migration, configs, fp, preserved = verified_inputs(queue)
        verification = read_json(VERIFICATION)
        if not verification.get("passed") or verification.get("fingerprint") != fp:
            raise RuntimeError("Exact recovered source fingerprint lacks passing real GPU recovery verification")
        state_path = OUTPUTS / "pipeline_state.json"
        old_state = read_json(state_path)
        if old_state["fingerprint"] not in (snapshot["legacy_fingerprint"], fp):
            raise RuntimeError("Pipeline is neither the pinned legacy nor the exact recovery source")
        if old_state["run_ids"] != snapshot["run_ids"]:
            raise RuntimeError("Pipeline run order changed")
        if old_state["fingerprint"] == snapshot["legacy_fingerprint"] and sha256(state_path) != snapshot["pipeline_sha256"]:
            raise RuntimeError("Legacy pipeline changed since the registered second failure")
        active_cfg = next(cfg for cfg in configs if cfg["run_id"] == ACTIVE)
        receipt = wrap_active_checkpoint(queue, active_cfg, fp, snapshot)
        state = {"state": "recovering", "active_run": ACTIVE, "run_ids": snapshot["run_ids"],
                 "fingerprint": fp, "legacy_fingerprint": snapshot["legacy_fingerprint"],
                 "preserved_legacy_completed": list(COMPLETED_ORDER), "pid": os.getpid(),
                 "started_utc": old_state["started_utc"], "recovery_started_utc": old_state.get("recovery_started_utc", queue.now()),
                 "updated_utc": queue.now(), "test_evaluation": False,
                 "recovery_verification_sha256": sha256(VERIFICATION), "checkpoint_migration": receipt,
                 "archived_failures": list(old_state.get("archived_failures", []))}
        queue.atomic_json(state_path, state)
        torch.manual_seed(queue.SEED)
        torch.cuda.manual_seed_all(queue.SEED)
        torch.backends.cudnn.benchmark = True
        try:
            for cfg in configs:
                if cfg["run_id"] in COMPLETED_ORDER:
                    print(f"Preserved historical completed result without rewriting: {cfg['run_id']}", flush=True)
                    continue
                archived_failure = archive_failure(cfg["run_id"])
                if archived_failure is not None and archived_failure not in state["archived_failures"]:
                    state["archived_failures"].append(archived_failure)
                state.update(state="running", active_run=cfg["run_id"], updated_utc=queue.now())
                queue.atomic_json(state_path, state)
                queue.run_one(cfg, fp)
                verify_completed(snapshot["legacy_fingerprint"])
                queue.report()
            state.update(state="completed_validation", active_run=None, completed_utc=queue.now(), updated_utc=queue.now())
            queue.atomic_json(state_path, state)
            queue.report()
        except BaseException as error:
            state.update(state="failed", error=repr(error), updated_utc=queue.now())
            queue.atomic_json(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Explicitly migrate active provenance and run the six remaining registrations")
    parser.add_argument("--check", action="store_true", help="Validate immutable migration inputs only (default)")
    args = parser.parse_args()
    queue = load_queue()
    if args.run:
        run_recovery(queue)
    else:
        snapshot, migration, configs, fp, completed = verified_inputs(queue)
        print(json.dumps({"inputs_valid": True, "historical_runs_preserved": completed,
                          "remaining_runs": [c["run_id"] for c in configs if c["run_id"] not in COMPLETED_ORDER],
                          "fingerprint": fp, "formal_execution_started": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
