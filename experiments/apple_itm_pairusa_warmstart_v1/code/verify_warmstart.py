"""CPU-only engineering checks for the isolated A-best warm-start driver.

The real constructor is checked with the genuine 5% A checkpoint.  Resume is
checked through the unchanged registered run_student using a tiny CPU model
and synthetic pairs/teacher targets, not by running a scientific experiment.
All writes stay under this branch's audit directory.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import traceback

# Prevent this verification process from allocating on the training GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_failure(call, description, contains=None):
    try:
        call()
    except (RuntimeError, ValueError) as exc:
        if contains is not None:
            require(contains in str(exc), f"Unexpected {description} error: {exc}")
        return {"passed": True, "message": str(exc)}
    raise AssertionError(f"Expected rejection was not raised: {description}")


def equal_nested(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(equal_nested(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_nested(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def verify_real_constructor(driver, backend, scratch):
    import pair_model
    original_model = pair_model.PairITMModel
    original_teacher, original_outputs = backend.train_teacher, backend.OUTPUTS
    source = driver.verify_parent_source()
    cfg = driver.make_config("005", source)
    checkpoint = torch.load(cfg["warmstart"]["a_best_path"], map_location="cpu", weights_only=True)
    constructor_cfg = copy.deepcopy(cfg["model"])
    constructor_cfg.update(use_pairusa=True)
    constructor_cfg["warmstart"]["audit_path"] = str(scratch / "real_initialization_audit.json")
    try:
        cls = driver.install_warmstart(backend)
        model = cls(constructor_cfg)
        require(all(p.device.type == "cpu" for p in model.parameters()), "Real warm-start model left CPU")
        actual = model.trainable_state()
        a_state = checkpoint["trainable_state"]
        require(len(a_state) == 34 and len(actual) == 39, "Unexpected real A/C topology")
        require(set(actual) - set(a_state) == driver.EXTRA_KEYS, "Unexpected fresh USA key set")
        require(all(torch.equal(actual[k], v) for k, v in a_state.items()), "A parameters were not inherited exactly")
        proof = driver.read(constructor_cfg["warmstart"]["audit_path"])
        require(proof["a_trainable_sha256"] == proof["loaded_shared_sha256"], "Shared initialization hash differs")
        expected_a_hash = "0fd86473f75ed286bf7d8509c88c98c93804c3aaaa77754dce887b56ca5e57af"
        require(proof["a_trainable_sha256"] == expected_a_hash, "5% A selected weight hash differs from independent audit")

        # Reproduce only the original constructor's small seed-isolated heads,
        # rather than constructing a second 850+ MB ALBEF CPU model.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(constructor_cfg["seed"]) + 303)
            nn.Linear(768, 1)
            fresh_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256))
        require(all(torch.equal(actual[f"student_projection.{k}"], v) for k, v in fresh_projection.state_dict().items()),
                "USA projection does not equal the registered fresh initialization")
        require(abs(float(model.student_temperature()) - 0.07) < 1e-7, "Fresh student temperature differs")
        require(not any(p.requires_grad for block in model.visual_encoder.blocks[:10] for p in block.parameters()),
                "Cached visual prefix unexpectedly trainable")
        require(not any(p.requires_grad for layer in model.text_encoder.encoder.layer[:6] for p in layer.parameters()),
                "Cached text prefix unexpectedly trainable")

        bad_missing = dict(a_state)
        bad_missing.pop(next(iter(bad_missing)))
        bad_key = dict(a_state)
        bad_key["wrong_parameter"] = bad_key.pop(next(iter(bad_key)))
        shape_key = next(k for k, v in a_state.items() if v.ndim > 0)
        bad_shape = {**a_state, shape_key: a_state[shape_key][:-1]}
        bad_nan_value = a_state[shape_key].clone()
        bad_nan_value.flatten()[0] = float("nan")
        bad_nan = {**a_state, shape_key: bad_nan_value}
        rejected = {
            "missing_key": expect_failure(lambda: driver.merge_a_into_student(model, bad_missing), "missing A key"),
            "unexpected_key": expect_failure(lambda: driver.merge_a_into_student(model, bad_key), "unexpected A key"),
            "shape_mismatch": expect_failure(lambda: driver.merge_a_into_student(model, bad_shape), "A shape mismatch"),
            "nonfinite_parameter": expect_failure(lambda: driver.merge_a_into_student(model, bad_nan), "nonfinite A parameter"),
        }
        require(all(torch.equal(model.trainable_state()[k], v) for k, v in a_state.items()), "Rejected merge changed valid model")
        require(not torch.cuda.is_initialized(), "CUDA initialized during the CPU constructor test")
        return {"passed": True, "device": "cpu", "a_best_step": checkpoint["step"],
                "a_checkpoint_sha256": driver.digest(cfg["warmstart"]["a_best_path"]),
                "copied_key_count": 34, "fresh_key_count": 5,
                "fresh_projection_exact": True, "frozen_prefixes_not_trainable": True,
                "initial_shared_sha256": proof["a_trainable_sha256"], "rejections": rejected}
    finally:
        pair_model.PairITMModel = original_model
        backend.train_teacher, backend.OUTPUTS = original_teacher, original_outputs


class TinyPairModel(nn.Module):
    """Deterministic CPU surrogate retaining BCE plus real Pair-USA gradients."""
    def __init__(self, config):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(713)
            self.lora_weight = nn.Parameter(torch.randn(4, 8) * 0.03)
            self.match_head = nn.Linear(8, 1)
            self.student_projection = nn.Linear(8, 256)
            self.log_student_temperature = nn.Parameter(torch.tensor(-2.659260036932778))

    def forward_cached(self, image, positive, positive_mask, negative, negative_mask):
        pos = torch.tanh((image + positive) @ self.lora_weight)
        neg = torch.tanh((image + negative) @ self.lora_weight)
        return torch.cat((self.match_head(pos), self.match_head(neg)), dim=-1), self.student_projection(pos)

    def student_temperature(self):
        return self.log_student_temperature.exp().clamp_min(0.01)

    def trainable_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters()}

    def load_trainable_state(self, state):
        self.load_state_dict(state, strict=True)


class TinyPairCache:
    def __init__(self, model, tokenizer, image_root):
        pass

    def prime_images(self, rows, flips=True):
        pass

    def prime_text(self, texts):
        pass

    def pair_batch(self, rows, step, augment):
        index = torch.tensor([int(row["image_id"]) for row in rows], dtype=torch.float32)
        image = torch.stack((index / 17, torch.sin(index), torch.cos(index), torch.ones_like(index)), dim=1)
        positive = torch.stack((torch.sin(index / 3), index / 19, torch.cos(index / 2), index / 23), dim=1)
        negative = positive.flip(1) - 0.1
        mask = torch.ones((len(rows), 4), dtype=torch.long)
        return image, positive, mask, negative, mask


def verify_actual_run_student_resume(driver, backend, scratch):
    import pair_model
    original_model = pair_model.PairITMModel
    saved = {key: getattr(backend, key) for key in ("DEVICE", "PairCache", "train_teacher", "OUTPUTS", "atomic_torch")}
    source = {"synthetic_cpu_fixture": "warmstart-resume-v1"}
    rows = [{"image_id": str(i), "positive_text": f"synthetic positive {i}", "negative_text": f"synthetic negative {(i+1)%16}"} for i in range(16)]
    teacher_dir = scratch / "synthetic_teacher"
    teacher_dir.mkdir()
    generator = torch.Generator().manual_seed(193)
    target_path, teacher_path = teacher_dir / "positive_targets.pt", teacher_dir / "best.pt"
    torch.save({"synthetic_cpu_fixture": True}, teacher_path)
    torch.save({"provenance": {"source": source}, "image_ids": [r["image_id"] for r in rows],
                "view_ids": ["canonical", "horizontal_flip"], "positive_vectors": torch.randn(16, 2, 256, generator=generator)}, target_path)
    cfg = {
        "run_id": "synthetic_C_resume_cpu_only", "method": "B_bce_pairusa_warmstart_A_best", "budget": "005",
        "model": {}, "image_root": str(scratch), "warmstart": {"synthetic_source_a_hash": "unchanged"},
        "training": {"max_steps": 4, "warmup_steps": 1, "lora_lr": 1e-3, "head_lr": 2e-3,
                     "weight_decay": 0.01, "max_grad_norm": 1.0, "checkpoint_every": 1, "eval_every": 2,
                     "pairusa_start": 0, "pairusa_ramp_end": 1, "pairusa_lambda": 0.1, "pairusa_teacher_temp": 1.0}}
    full_root, resume_root, bad_root = (scratch / name for name in ("uninterrupted", "interrupted_then_resumed", "changed_provenance"))
    class DeliberateInterruption(RuntimeError):
        pass
    try:
        pair_model.PairITMModel = TinyPairModel
        backend.DEVICE = torch.device("cpu")
        backend.PairCache = TinyPairCache
        backend.train_teacher = lambda *args, **kwargs: teacher_path
        backend.OUTPUTS = full_root
        backend.run_student(cfg, rows, rows, None, source)
        full_checkpoint = torch.load(full_root / cfg["run_id"] / "last.pt", map_location="cpu", weights_only=False)

        backend.OUTPUTS = resume_root
        original_save = saved["atomic_torch"]
        def interrupt_after_saved_step_two(path, payload):
            original_save(path, payload)
            if Path(path).name == "last.pt" and payload.get("step") == 2:
                raise DeliberateInterruption("CPU fixture interruption after durable step 2")
        backend.atomic_torch = interrupt_after_saved_step_two
        try:
            backend.run_student(cfg, rows, rows, None, source)
            raise AssertionError("Interruption injection never occurred")
        except DeliberateInterruption:
            pass
        interrupted_checkpoint = torch.load(resume_root / cfg["run_id"] / "last.pt", map_location="cpu", weights_only=False)
        require(interrupted_checkpoint["step"] == 2, "Interruption was not after step 2")
        backend.atomic_torch = original_save

        # Same on-disk branch checkpoint, but a changed scientific warm-start
        # binding: the unchanged real run_student must refuse to restore it.
        original_save(bad_root / cfg["run_id"] / "last.pt", interrupted_checkpoint)
        backend.OUTPUTS = bad_root
        changed_cfg = copy.deepcopy(cfg)
        changed_cfg["warmstart"]["synthetic_source_a_hash"] = "changed"
        rejection = expect_failure(lambda: backend.run_student(changed_cfg, rows, rows, None, source),
                                   "changed C checkpoint provenance", "Student checkpoint provenance changed")

        backend.OUTPUTS = resume_root
        resumed_result = backend.run_student(cfg, rows, rows, None, source)
        resumed = torch.load(resume_root / cfg["run_id"] / "last.pt", map_location="cpu", weights_only=False)
        require(resumed["step"] == 4, "Recovery did not advance past interrupted step 2")
        for key in ("trainable_state", "optimizer", "scaler", "best_step", "best_key", "provenance"):
            require(equal_nested(full_checkpoint[key], resumed[key]), f"Resumed {key} differs from uninterrupted execution")
        records = [json.loads(line) for line in (resume_root / cfg["run_id"] / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        require([item["step"] for item in records] == [1, 2, 3, 4], "Resume repeated/skipped a committed step")
        require(all(item["lambda"] == 0.1 and item["pairusa"] > 0 for item in records), "Real Pair-USA did not execute on every synthetic step")
        changed_completed = copy.deepcopy(cfg)
        changed_completed["warmstart"]["synthetic_source_a_hash"] = "changed"
        completed_rejection = expect_failure(lambda: backend.run_student(changed_completed, rows, rows, None, source),
                                             "changed completed C provenance", "Completed student provenance changed")
        require(not torch.cuda.is_initialized(), "CUDA initialized during CPU resume verification")
        return {"passed": True, "device": "cpu", "scope": "unchanged_parent_run_student_with_tiny_model_and_synthetic_teacher",
                "completed_steps": resumed["step"], "interrupted_after_step": 2,
                "state_and_optimizer_bitwise_equal": True, "all_recorded_steps": [item["step"] for item in records],
                "real_pairusa_loss_enabled": True, "pairusa_losses": [item["pairusa"] for item in records],
                "checkpoint_provenance_rejection": rejection, "completed_provenance_rejection": completed_rejection,
                "initial_shared_sha256": resumed_result["initial_shared_sha256"]}
    finally:
        pair_model.PairITMModel = original_model
        for key, value in saved.items():
            setattr(backend, key, value)


def main():
    spec = importlib.util.spec_from_file_location("warmstart_driver_verification", ROOT / "code/run_warmstart.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    identity = driver.branch_fingerprint()
    scratch = Path(tempfile.mkdtemp(prefix="verify-cpu-", dir=ROOT / "audit"))
    result = {"passed": False, "branch_fingerprint": identity, "device": "cpu", "scratch_directory": str(scratch),
              "verification_code_sha256": driver.digest(Path(__file__)), "started_utc": driver.now(),
              "scope": "real_A_weight_transfer_plus_synthetic_training_resume_not_scientific_validation"}
    try:
        backend = driver.load_backend()
        result["real_warmstart"] = verify_real_constructor(driver, backend, scratch)
        result["synthetic_resume"] = verify_actual_run_student_resume(driver, backend, scratch)
        require(driver.branch_fingerprint() == identity, "Branch changed during verification")
        require(not torch.cuda.is_initialized(), "Verification must never initialize CUDA")
        result.update(passed=True, completed_utc=driver.now(), cuda_initialized=False)
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc(), completed_utc=driver.now())
        driver.atomic_json(ROOT / "audit/verification.json", result)
        raise
    driver.atomic_json(ROOT / "audit/verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
