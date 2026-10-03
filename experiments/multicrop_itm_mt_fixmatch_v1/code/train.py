"""Independent single-GPU Mean Teacher/FixMatch pair-ITM training.

No import of legacy queues. All complete resumptions are fingerprint checked.
CPU --check has no CUDA access; GPU --gate is isolated engineering, no evaluation.
"""
from __future__ import annotations
import argparse
import copy
import gc
import importlib.metadata
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT, read, digest, canonical, code_hashes, now, atomic_json, atomic_torch, exclusive
from data_backend import PrefixCache, fixed_seed, flatten_l, chunks, load_contract
from losses import fixmatch_targets, mean_teacher_loss, weighted_bce
from pair_model import PairITMModel
from albef_ssl.model import get_tokenizer
from metrics import select_threshold, binary_metrics

SUPPORTED = ("meanteacher", "fixmatch")

def versions():
    packages = ("torch", "torchvision", "numpy", "Pillow", "transformers", "timm", "scikit-learn", "huggingface-hub")
    return {"python": sys.version, "executable": sys.executable,
            "packages": {p: importlib.metadata.version(p) for p in packages}}

def config_for(run_id):
    if any(x in run_id for x in ("/", "\\", "..")):
        raise ValueError("Unsafe run ID")
    cfg = read(ROOT / "configs/runs" / (run_id + ".json"))
    if cfg["run_id"] != run_id or cfg["method"] not in SUPPORTED or not cfg["launch_enabled"]:
        raise RuntimeError("Method is not implemented/launch enabled")
    protocol = read(ROOT / "configs/protocol.json")
    if digest(ROOT / "configs/protocol.json") != cfg["protocol_sha256"]:
        raise RuntimeError("Protocol fingerprint mismatch")
    if cfg["target_steps"] != protocol["targets"][cfg["method"]] or cfg["seed"] != 20260825:
        raise RuntimeError("Registered target/seed mismatch")
    return cfg

def provenance(cfg, manifest):
    model_meta = read(ROOT / "audit/model_assets.json")
    # Every checkpoint/tokenizer byte is checked before model construction/resume.
    for path, meta in model_meta["files"].items():
        if digest(ROOT / path) != meta["sha256"]:
            raise RuntimeError("Model/tokenizer asset changed: " + path)
    return {"config": canonical(cfg), "protocol": cfg["protocol_sha256"],
            "dataset_manifest": digest(ROOT / "data" / cfg["dataset"] / "manifest.json"),
            "inputs": manifest["budgets"][cfg["budget"]], "validation": manifest["validation"],
            "assets": manifest["asset_index"], "model_assets": model_meta["files"],
            "source": code_hashes(), "environment": versions()}

def common_digest(state):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}

def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])

class Runner:
    def __init__(self, cfg, contract, prov):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA-visible GPU required; no DDP")
        self.cfg, self.prov = cfg, prov
        self.manifest, self.assets, self.l, self.u, self.val = contract
        self.device = torch.device("cuda:0")
        torch.set_num_threads(4)
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        model_cfg = {**cfg["model"], "seed": cfg["seed"], "checkpoint": str(ROOT / "assets/ALBEF_4M.pth"),
                     "fusion_chunk_size": cfg["training"]["physical_pairs"]}
        self.model = PairITMModel(model_cfg).to(self.device).train()
        initial = self.model.trainable_state()
        if len(initial) != 34 or any("student_projection" in k or "temperature" in k for k in initial):
            raise RuntimeError("Common trainable topology is not 34 keys")
        if not torch.equal(self.model.match_head.weight.detach(), (self.model.itm_head.weight[1]-self.model.itm_head.weight[0]).unsqueeze(0)):
            raise RuntimeError("Scalar ITM initialization mismatch")
        self.initial_hash = common_digest(initial)
        self.ema = copy.deepcopy(self.model).requires_grad_(False).eval()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        lora = [p for n, p in self.model.named_parameters() if p.requires_grad and "lora_" in n]
        head = [p for n, p in self.model.named_parameters() if p.requires_grad and "match_head" in n]
        if len(lora) + len(head) != len(self.params):
            raise RuntimeError("Unexpected optimizer parameters")
        t = cfg["training"]
        self.opt = torch.optim.AdamW([{"params": lora, "lr": t["lora_lr"], "base_lr": t["lora_lr"]},
            {"params": head, "lr": t["head_lr"], "base_lr": t["head_lr"]}], weight_decay=t["weight_decay"])
        self.scaler = torch.amp.GradScaler("cuda", init_scale=t["amp_initial_scale"], growth_interval=2000)
        self.cache = PrefixCache(self.model, get_tokenizer(str(ROOT / "assets/tokenizer")), self.assets,
                                 self.manifest["token_guard"], cfg["seed"], self.device)
        self.step, self.physical = 0, t["physical_pairs"]
        self.best_key, self.best_step, self.best_metrics = [-1., -1.], 0, None
        self.elapsed_seconds, self.u_seen = 0., set()
        self.wall_seconds, self.wall_started = 0., time.monotonic()

    def ema_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.ema.named_parameters() if n in self.names}

    def load_ema(self, state):
        if set(state) != set(self.names):
            raise RuntimeError("EMA topology mismatch")
        with torch.no_grad():
            for n, p in self.ema.named_parameters():
                if n in state:
                    p.copy_(state[n].to(p))

    def snapshot(self):
        return {"format": "pair_ssl_full_v1", "provenance": self.prov, "step": self.step,
                "target_steps": self.cfg["target_steps"], "trainable_state": self.model.trainable_state(),
                "ema": self.ema_state(), "optimizer": copy.deepcopy(self.opt.state_dict()),
                "scaler": copy.deepcopy(self.scaler.state_dict()), "rng": rng_state(),
                "physical": self.physical, "initial_state_sha256": self.initial_hash,
                "best_key": self.best_key, "best_step": self.best_step, "best_metrics": self.best_metrics,
                "elapsed_seconds": self.elapsed_seconds, "u_seen": sorted(self.u_seen),
                "wall_seconds": self.wall_seconds + time.monotonic() - self.wall_started,
                "updated_utc": now()}

    def load(self, payload):
        required = ("format", "provenance", "step", "target_steps", "trainable_state", "ema", "optimizer", "scaler", "rng", "initial_state_sha256", "best_key", "best_step", "best_metrics", "elapsed_seconds", "u_seen")
        if any(k not in payload for k in required) or payload["format"] != "pair_ssl_full_v1":
            raise RuntimeError("Incomplete checkpoint is not a resumable state")
        if payload["provenance"] != self.prov or payload["initial_state_sha256"] != self.initial_hash:
            raise RuntimeError("Resume fingerprints do not match")
        if payload["target_steps"] != self.cfg["target_steps"]:
            raise RuntimeError("Target changed on resume")
        self.model.load_trainable_state(payload["trainable_state"])
        self.load_ema(payload["ema"])
        self.opt.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.step, self.physical = int(payload["step"]), int(payload["physical"])
        self.model.fusion_chunk_size = self.physical
        self.ema.fusion_chunk_size = self.physical
        self.best_key, self.best_step, self.best_metrics = payload["best_key"], payload["best_step"], payload["best_metrics"]
        self.elapsed_seconds, self.u_seen = payload["elapsed_seconds"], set(payload["u_seen"])
        self.wall_seconds, self.wall_started = payload.get("wall_seconds", 0.), time.monotonic()
        restore_rng(payload["rng"])

    @torch.no_grad()
    def predict(self, model, batch, amp=True):
        outputs = []
        for x in chunks(batch, self.physical):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                outputs.append(model.forward_pairs(*tuple(v if amp or not v.is_floating_point() else v.float() for v in x))[0].float())
        return torch.cat(outputs)

    def train_step(self):
        step = self.step + 1
        seed = self.cfg["seed"]
        chosen_l = random.Random(fixed_seed(seed, step, "pair-sampler")).sample(self.l, 16)
        chosen_u = random.Random(fixed_seed(seed, step, "u-sampler")).sample(self.u, 32)
        lp = flatten_l(chosen_l)
        targets_l = torch.tensor([p["label"] for p in lp], device=self.device)
        timer = time.monotonic()
        attempts = 0
        fp32 = False
        while True:
            attempts += 1
            if attempts > 12:
                raise RuntimeError("Repeated same-step failures; stopping without counting update")
            self.opt.zero_grad(set_to_none=True)
            self.model.fusion_chunk_size = self.ema.fusion_chunk_size = self.physical
            lb = uw = us = weak = z = loss = raw_loss = inputs = target_u = mask = None
            try:
                lb = self.cache.batch(lp, step, "weak", self.physical)
                uw = self.cache.batch(chosen_u, step, "weak", self.physical)
                us = self.cache.batch(chosen_u, step, "strong", self.physical)
                teacher = self.ema if self.cfg["method"] == "meanteacher" else self.model
                weak = self.predict(teacher, uw, amp=not fp32)
                if self.cfg["method"] == "meanteacher":
                    target_u, mask = weak.sigmoid().detach(), torch.ones(32, device=self.device)
                    ramp = min(1., (step-1) / (self.cfg["method_settings"]["unsup_warm_up"] * self.cfg["target_steps"]))
                else:
                    target_u, mask = fixmatch_targets(weak, self.cfg["method_settings"]["p_cutoff"])
                    ramp = 1.
                warmup = self.cfg["training"]["warmup_steps"]
                factor = step / warmup if step <= warmup else .5 * (1 + math.cos(math.pi * (step-warmup) / (self.cfg["target_steps"]-warmup)))
                for g in self.opt.param_groups:
                    g["lr"] = g["base_lr"] * factor
                sup_total, unsup_total = 0., 0.
                for i, x in enumerate(chunks(lb, self.physical)):
                    begin = i * self.physical
                    inputs = tuple(v.float() if fp32 and v.is_floating_point() else v for v in x)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
                        z, _ = self.model.forward_pairs(*inputs)
                        loss = F.binary_cross_entropy_with_logits(z.float(), targets_l[begin:begin+len(z)], reduction="sum") / 32
                    self.scaler.scale(loss).backward() if not fp32 else loss.backward()
                    sup_total += float(loss.detach())
                for i, x in enumerate(chunks(us, self.physical)):
                    begin = i * self.physical
                    inputs = tuple(v.float() if fp32 and v.is_floating_point() else v for v in x)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
                        z, _ = self.model.forward_pairs(*inputs)
                        tar, weights = target_u[begin:begin+len(z)], mask[begin:begin+len(z)]
                        values = mean_teacher_loss(z, tar) if self.cfg["method"] == "meanteacher" else weighted_bce(z, tar, weights)
                        raw_loss = values.sum() / 32
                        loss = raw_loss * ramp * self.cfg["training"]["lambda_u"]
                    self.scaler.scale(loss).backward() if not fp32 else loss.backward()
                    unsup_total += float(raw_loss.detach())
                if not fp32:
                    self.scaler.unscale_(self.opt)
                finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.params)
                if not finite:
                    if fp32:
                        raise FloatingPointError("Nonfinite FP32 gradient")
                    scale = self.scaler.get_scale()
                    self.scaler.update(new_scale=max(1., scale / 2))
                    fp32 = scale <= 16
                    del lb, uw, us, weak
                    continue
                norm = torch.nn.utils.clip_grad_norm_(self.params, self.cfg["training"]["max_grad_norm"])
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Nonfinite clipping norm")
                if fp32:
                    self.opt.step()
                else:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                with torch.no_grad():
                    teacher_params = dict(self.ema.named_parameters())
                    m = self.cfg["training"]["ema_decay"]
                    for n, p in self.model.named_parameters():
                        if p.requires_grad:
                            teacher_params[n].mul_(m).add_(p, alpha=1-m)
                self.step = step
                self.u_seen.update(p["pair_id"] for p in chosen_u)
                seconds = time.monotonic()-timer
                self.elapsed_seconds += seconds
                return {"step": step, "sup_loss": sup_total, "unsup_loss": unsup_total,
                        "lambda_u_effective": ramp * self.cfg["training"]["lambda_u"],
                        "util_ratio": float(mask.mean()), "pseudo_positive_fraction": float((target_u >= .5).float().mean()),
                        "seconds": seconds, "physical": self.physical, "scale": self.scaler.get_scale(),
                        "fp32_retry": fp32, "attempts": attempts, "l_pair_visits": step*32,
                        "u_pair_visits": step*32, "unique_u_pairs": len(self.u_seen), "updated_utc": now()}
            except torch.cuda.OutOfMemoryError:
                self.opt.zero_grad(set_to_none=True)
                lb = uw = us = weak = z = loss = raw_loss = inputs = target_u = mask = None
                gc.collect()
                torch.cuda.empty_cache()
                if self.physical <= 4:
                    raise
                self.physical //= 2
                # Reset scaler stage when OOM follows a partially constructed pass.
                state = self.scaler.state_dict()
                self.scaler = torch.amp.GradScaler("cuda")
                self.scaler.load_state_dict(state)

    @torch.no_grad()
    def evaluate(self):
        self.ema.eval()
        probabilities, labels, ordered = [], [], []
        for begin in range(0, len(self.val), 8):
            rows = self.val[begin:begin+8]
            pairs = flatten_l(rows)
            batch = self.cache.batch(pairs, self.step, "validation", self.physical)
            logits = self.predict(self.ema, batch)
            probabilities.extend(logits.sigmoid().cpu().tolist())
            labels.extend(p["label"] for p in pairs)
            ordered.extend(float(a > b) for a, b in logits.reshape(-1, 2).cpu().tolist())
        threshold = select_threshold(labels, probabilities)
        metrics = binary_metrics(labels, probabilities, threshold)
        metrics["threshold_0_5"] = binary_metrics(labels, probabilities, .5)
        metrics["paired_accuracy"] = float(np.mean(ordered))
        metrics["validation_anchors"] = 400
        metrics["evaluation_model"] = "ema"
        metrics["step"] = self.step
        return metrics, {"labels": labels, "probabilities": probabilities, "threshold": threshold,
                         "image_ids": [r["image_id"] for r in self.val], "step": self.step, "split": "validation"}

def check(run_id):
    cfg = config_for(run_id)
    contract = load_contract(cfg["dataset"], cfg["budget"])
    prov = provenance(cfg, contract[0])
    last = ROOT / "outputs" / run_id / "last.pt"
    resumed_step = None
    if last.exists():
        payload = torch.load(last, map_location="cpu", weights_only=False)
        if payload.get("provenance") != prov or payload.get("format") != "pair_ssl_full_v1":
            raise RuntimeError("Complete last.pt fingerprints do not match")
        if not all(k in payload for k in ("optimizer", "scaler", "rng", "ema", "trainable_state")):
            raise RuntimeError("Incomplete last checkpoint")
        resumed_step = payload["step"]
    return cfg, contract, prov, resumed_step

def gate(run_id, gpu_uuid):
    cfg, contract, prov, _ = check(run_id)
    out = ROOT / "audit/gpu_gates" / run_id
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "passed.json"
    if receipt.exists():
        previous = read(receipt)
        if previous.get("provenance") == prov and previous.get("gpu_uuid") == gpu_uuid and previous.get("passed"):
            print("Previously passed identical gate; not rerunning")
            return
        raise RuntimeError("Existing gate belongs to another fingerprint; preserve and re-register version")
    start = time.monotonic()
    runner = Runner(cfg, contract, prov)
    a = runner.train_step()
    snapshot = runner.snapshot()
    atomic_torch(out / "engineering_last1.pt", snapshot)
    a2 = runner.train_step()
    reference = runner.snapshot()
    runner.load(torch.load(out / "engineering_last1.pt", map_location="cpu", weights_only=False))
    b2 = runner.train_step()
    replay = runner.snapshot()
    max_diff = max(float((reference["trainable_state"][n]-replay["trainable_state"][n]).abs().max()) for n in runner.names)
    ema_diff = max(float((reference["ema"][n]-replay["ema"][n]).abs().max()) for n in runner.names)
    if max_diff > 1e-6 or ema_diff > 1e-6 or reference["step"] != replay["step"] or reference["scaler"] != replay["scaler"]:
        raise RuntimeError("Full resume replay failed")
    for i, (x, y) in enumerate(zip(reference["optimizer"]["state"].values(), replay["optimizer"]["state"].values())):
        for name in x:
            if torch.is_tensor(x[name]):
                torch.testing.assert_close(x[name], y[name], rtol=1e-6, atol=1e-7)
    common_path = ROOT / "audit/common_initialization.json"
    with exclusive(ROOT / "locks/init.lock"):
        if common_path.exists():
            if read(common_path)["common_34_state_sha256"] != runner.initial_hash:
                raise RuntimeError("Methods do not share identical 34-key initialization")
        else:
            atomic_json(common_path, {"common_34_state_sha256": runner.initial_hash, "keys": runner.names,
                "official_md5": runner.model.load_report["checkpoint_md5"], "load_report": runner.model.load_report,
                "no_BCE_best_loaded": True, "created_utc": now()})
    atomic_json(receipt, {"passed": True, "provenance": prov, "gpu_uuid": gpu_uuid,
        "common_34_state_sha256": runner.initial_hash, "engineering_only": True, "validation_or_test_run": False,
        "physical": runner.physical, "logical_L_pairs": 32, "logical_U_pairs": 32,
        "one": a, "two": a2, "replay_two": b2, "replay_max_parameter_difference": max_diff,
        "replay_max_ema_difference": ema_diff, "optimizer_scaler_replay_verified": True,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "seconds": time.monotonic()-start,
        "created_utc": now()})
    print("GPU gate passed", receipt, flush=True)

def run(run_id, gpu_uuid):
    cfg, contract, prov, resumed = check(run_id)
    gate_receipt = ROOT / "audit/gpu_gates" / run_id / "passed.json"
    if not gate_receipt.exists():
        raise RuntimeError("Real GPU/resume gate required")
    passed = read(gate_receipt)
    if not passed.get("passed") or passed["provenance"] != prov or passed["gpu_uuid"] != gpu_uuid:
        raise RuntimeError("Gate fingerprints/GPU mismatch")
    out = ROOT / "outputs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        result = read(out / "result.json")
        if result["provenance"] != prov or result["successful_steps"] != cfg["target_steps"]:
            raise RuntimeError("Existing result is not this completed run")
        print("Already complete; nothing restarted")
        return
    runner = Runner(cfg, contract, prov)
    if resumed is not None:
        runner.load(torch.load(out / "last.pt", map_location="cpu", weights_only=False))
    else:
        atomic_torch(out / "last.pt", runner.snapshot())
    atomic_json(out / "status.json", {"state": "running", "step": runner.step, "target_steps": cfg["target_steps"],
                "pid": os.getpid(), "gpu_uuid": gpu_uuid, "provenance": prov, "updated_utc": now()})
    session_log = out / ("train_" + time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}.jsonl")
    with session_log.open("a", encoding="utf-8", buffering=1) as log:
        while runner.step < cfg["target_steps"]:
            record = runner.train_step()
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
            print(json.dumps(record, allow_nan=False), flush=True)
            evaluated = runner.step % cfg["training"]["eval_every"] == 0 or runner.step == cfg["target_steps"]
            if evaluated:
                metrics, predictions = runner.evaluate()
                atomic_json(out / f"validation_{runner.step:04d}.json", metrics)
                atomic_json(out / f"predictions_validation_{runner.step:04d}.json", predictions)
                key = [metrics["paired_accuracy"], metrics["auroc"]]
                if tuple(key) > tuple(runner.best_key):
                    runner.best_key, runner.best_step, runner.best_metrics = key, runner.step, metrics
                    atomic_torch(out / "best.pt", runner.snapshot())
            if evaluated or runner.step % cfg["training"]["checkpoint_every"] == 0:
                atomic_torch(out / "last.pt", runner.snapshot())
            atomic_json(out / "status.json", {"state": "running", "step": runner.step,
                "target_steps": cfg["target_steps"], "best_step": runner.best_step, "pid": os.getpid(),
                "gpu_uuid": gpu_uuid, "physical": runner.physical, "updated_utc": now()})
    result = {"run_id": run_id, "state": "completed", "successful_steps": runner.step,
        "best_step": runner.best_step, "best_metrics": runner.best_metrics, "provenance": prov,
        "last_sha256": digest(out / "last.pt"), "best_sha256": digest(out / "best.pt"),
        "training_step_seconds": runner.elapsed_seconds, "logical_l_pair_visits": runner.step*32,
        "wall_seconds_including_validation_and_checkpointing": runner.wall_seconds + time.monotonic() - runner.wall_started,
        "logical_u_pair_visits": runner.step*32, "unique_u_pairs": len(runner.u_seen),
        "test_evaluated": False, "single_seed": cfg["seed"], "equal_compute_claim": False,
        "initial_state_sha256": runner.initial_hash, "completed_utc": now()}
    atomic_json(out / "result.json", result)
    atomic_json(out / "status.json", {"state": "completed", "step": runner.step,
        "target_steps": cfg["target_steps"], "best_step": runner.best_step, "updated_utc": now()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--gate", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    try:
        if args.check:
            cfg, contract, prov, step = check(args.run_id)
            print(json.dumps({"passed": True, "run_id": args.run_id, "resume_step": step, "provenance": prov}, ensure_ascii=False))
        else:
            if not args.gpu_uuid:
                raise RuntimeError("GPU UUID registration required")
            with exclusive(ROOT / "locks" / (args.gpu_uuid + ".lock")), exclusive(ROOT / "locks" / (args.run_id + ".lock")):
                gate(args.run_id, args.gpu_uuid) if args.gate else run(args.run_id, args.gpu_uuid)
    except BaseException:
        traceback.print_exc()
        atomic_json(ROOT / "audit/failures" / (args.run_id + f"_{os.getpid()}.json"),
                    {"run_id": args.run_id, "mode": "check" if args.check else "gate" if args.gate else "run",
                     "error": traceback.format_exc(), "utc": now(), "pid": os.getpid()})
        raise
