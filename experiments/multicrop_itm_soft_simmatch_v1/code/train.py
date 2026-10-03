"""Linux independent SoftMatch/SimMatch continuous runs, same pair-ITM contract."""
from __future__ import annotations
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import gc
import hashlib
import json
import math
import random
import time
import traceback
import torch
from torch import nn
from torch.nn import functional as F
import base_train as base
from common import ROOT, read, digest, canonical, now, atomic_json, atomic_torch, exclusive
from data_backend import chunks, flatten_l, fixed_seed, load_contract
from losses import probabilities, weighted_bce
from ssl_algorithms import SoftMatchStatistics, DistributionQueue, simmatch_targets, instance_ce, cpu_state, assert_nested_close

SUPPORTED = ("softmatch", "simmatch")

def config_for(run_id):
    if any(c in run_id for c in ("/", "\\", "..")):
        raise ValueError("Unsafe run ID")
    cfg = read(ROOT / "configs/runs" / (run_id + ".json"))
    protocol = read(ROOT / "configs/protocol.json")
    if cfg["run_id"] != run_id or cfg["method"] not in SUPPORTED or not cfg["launch_enabled"]:
        raise RuntimeError("This server is assigned SoftMatch/SimMatch only")
    if cfg["protocol_sha256"] != digest(ROOT / "configs/protocol.json"):
        raise RuntimeError("Protocol changed")
    if cfg["target_steps"] != protocol["targets"][cfg["method"]] or cfg["seed"] != 20260825:
        raise RuntimeError("Registered method target/seed changed")
    return cfg

class Runner(base.Runner):
    def __init__(self, cfg, contract, prov):
        super().__init__(cfg, contract, prov)
        common = self.model.trainable_state()
        legacy = hashlib.sha256(b"".join(v.numpy().tobytes() for _, v in sorted(common.items()))).hexdigest()
        if legacy != cfg["legacy_common_tensor_sha256"]:
            raise RuntimeError("The 34 common initial tensors differ from the original BCE reference")
        self.legacy_hash = legacy
        self.bank = None
        self.bank_priming_seconds = 0.
        self.bank_priming_pairs = 0
        self.warmup_steps = 0
        self.pair_to_index = {}
        if cfg["method"] == "softmatch":
            s = cfg["method_settings"]
            self.algorithm = SoftMatchStatistics(s["ema_p"], s["n_sigma"], s["variance_min_numerical_guard"])
        else:
            s = cfg["method_settings"]
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(cfg["seed"] + s["projector_seed_offset"])
                projection = nn.Sequential(nn.Linear(768, 768), nn.ReLU(inplace=False), nn.Linear(768, s["proj_size"]))
            self.model.sim_projection = projection.to(self.device)
            self.ema.sim_projection = copy.deepcopy(projection).to(self.device).requires_grad_(False).eval()
            self.params = [p for p in self.model.parameters() if p.requires_grad]
            self.names = [n for n, p in self.model.named_parameters() if p.requires_grad]
            self.opt.add_param_group({"params": list(self.model.sim_projection.parameters()),
                                     "lr": s["projector_lr"], "base_lr": s["projector_lr"]})
            if len(self.names) != 38:
                raise RuntimeError("SimMatch requires 34 common +4 projection tensors, not USA39")
            self.algorithm = DistributionQueue(s["da_len"])
            pairs = flatten_l(self.l)
            self.pair_to_index = {p["pair_id"]: i for i, p in enumerate(pairs)}
            if len(self.pair_to_index) != 2*len(self.l):
                raise RuntimeError("Pair memory keys collide")
            self.bank_labels = torch.tensor([int(p["label"]) for p in pairs], dtype=torch.long, device=self.device)
            self.warmup_steps = math.ceil(len(self.l) / cfg["training"]["logical_l_anchors"])

    @torch.no_grad()
    def predict_features(self, model, batch, amp=True):
        zs, features = [], []
        for x in chunks(batch, self.physical):
            values = tuple(v.float() if not amp and v.is_floating_point() else v for v in x)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                z, raw = model.forward_pairs(*values)
                projected = F.normalize(model.sim_projection(raw).float(), dim=-1)
            zs.append(z.float())
            features.append(projected.float())
        return torch.cat(zs), torch.cat(features)

    @torch.no_grad()
    def prime_bank(self):
        if self.cfg["method"] != "simmatch" or self.bank is not None:
            return
        start = time.monotonic()
        pairs = flatten_l(self.l)
        features = []
        original_physical = self.physical
        for begin in range(0, len(pairs), 32):
            while True:
                try:
                    # Unaugmented public labeled-TRAIN pairs, NOT Validation.
                    batch = self.cache.batch(pairs[begin:begin+32], 0, "validation", self.physical)
                    _, f = self.predict_features(self.ema, batch)
                    features.append(f.detach().cpu())
                    break
                except torch.cuda.OutOfMemoryError:
                    if self.physical <= 4:
                        raise
                    self.physical //= 2
                    self.model.fusion_chunk_size = self.ema.fusion_chunk_size = self.physical
                    gc.collect()
                    torch.cuda.empty_cache()
        self.bank = torch.cat(features).to(self.device)
        if self.bank.shape != (len(pairs), self.cfg["method_settings"]["proj_size"]) or not bool(torch.isfinite(self.bank).all()):
            raise RuntimeError("Labeled pair bank priming failed")
        self.bank_priming_seconds = time.monotonic()-start
        self.bank_priming_pairs = len(pairs)
        print(json.dumps({"bank_primed_public_L_pairs": len(pairs), "optimizer_updates": 0,
                          "bank_priming_seconds": self.bank_priming_seconds, "physical": self.physical}), flush=True)

    def algorithm_state(self):
        result = {"method": self.cfg["method"], "statistics": self.algorithm.state_dict(),
                  "instance_warmup_steps": self.warmup_steps, "bank_priming_seconds": self.bank_priming_seconds,
                  "bank_priming_pairs": self.bank_priming_pairs, "legacy_common_tensor_sha256": self.legacy_hash}
        if self.cfg["method"] == "simmatch":
            result.update(bank=None if self.bank is None else self.bank.detach().cpu().clone(),
                          bank_labels=self.bank_labels.detach().cpu().clone(), pair_to_index=self.pair_to_index)
        return cpu_state(result)

    def load_algorithm_state(self, state):
        if state["method"] != self.cfg["method"] or state["legacy_common_tensor_sha256"] != self.legacy_hash or state["instance_warmup_steps"] != self.warmup_steps:
            raise RuntimeError("Algorithm-state contract changed")
        self.algorithm.load_state_dict(state["statistics"])
        self.bank_priming_seconds, self.bank_priming_pairs = state["bank_priming_seconds"], state["bank_priming_pairs"]
        if self.cfg["method"] == "simmatch":
            if state["pair_to_index"] != self.pair_to_index or not torch.equal(state["bank_labels"], self.bank_labels.cpu()):
                raise RuntimeError("Pair-ID memory registry changed")
            if state["bank"] is None:
                self.bank = None
            else:
                if state["bank"].shape != (len(self.pair_to_index), self.cfg["method_settings"]["proj_size"]):
                    raise RuntimeError("Pair bank shape changed")
                self.bank = state["bank"].to(self.device)

    def snapshot(self):
        payload = super().snapshot()
        payload.update(algorithm_state=self.algorithm_state(), algorithm_checkpoint_version="soft_sim_full_v1")
        return payload

    def load(self, payload):
        if payload.get("algorithm_checkpoint_version") != "soft_sim_full_v1" or "algorithm_state" not in payload:
            raise RuntimeError("Missing SoftMatch statistics/SimMatch bank state; not a full resume")
        super().load(payload)
        self.load_algorithm_state(payload["algorithm_state"])

    def train_step(self):
        self.prime_bank()
        step = self.step+1
        seed = self.cfg["seed"]
        lrows = random.Random(fixed_seed(seed, step, "pair-sampler")).sample(self.l, 16)
        urows = random.Random(fixed_seed(seed, step, "u-sampler")).sample(self.u, 32)
        lp = flatten_l(lrows)
        label_l = torch.tensor([p["label"] for p in lp], device=self.device)
        original_state = self.algorithm_state()
        timer, attempts, fp32 = time.monotonic(), 0, False
        while True:
            attempts += 1
            if attempts > 12:
                self.load_algorithm_state(original_state)
                raise RuntimeError("Same-step AMP/OOM retry exhausted")
            self.load_algorithm_state(original_state)
            self.opt.zero_grad(set_to_none=True)
            self.model.fusion_chunk_size = self.ema.fusion_chunk_size = self.physical
            lb = uw = us = weak = raw = z = loss = raw_loss = inputs = target_u = mask = None
            weak_features = l_ema_features = instance_target = values = projected = instance_values = None
            try:
                lb = self.cache.batch(lp, step, "weak", self.physical)
                uw = self.cache.batch(urows, step, "weak", self.physical)
                us = self.cache.batch(urows, step, "strong", self.physical)
                if self.cfg["method"] == "softmatch":
                    weak = self.predict(self.model, uw, amp=not fp32)
                    target_u, mask, _ = self.algorithm.targets(weak)
                    instance_active = False
                else:
                    weak, weak_features = self.predict_features(self.ema, uw, amp=not fp32)
                    aligned = self.algorithm.align(probabilities(weak.detach()))
                    instance_active = step > self.warmup_steps
                    s = self.cfg["method_settings"]
                    target_u, mask, instance_target, _ = simmatch_targets(aligned, weak_features, self.bank,
                        self.bank_labels, s["T"], s["smoothing_alpha"], not instance_active, s["p_cutoff"])
                    _, l_ema_features = self.predict_features(self.ema, lb, amp=not fp32)
                warmup = self.cfg["training"]["warmup_steps"]
                factor = step/warmup if step <= warmup else .5*(1+math.cos(math.pi*(step-warmup)/(self.cfg["target_steps"]-warmup)))
                for g in self.opt.param_groups:
                    g["lr"] = g["base_lr"]*factor
                sup_total, unsup_total, instance_total = 0., 0., 0.
                for i, x in enumerate(chunks(lb, self.physical)):
                    begin = i*self.physical
                    inputs = tuple(v.float() if fp32 and v.is_floating_point() else v for v in x)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
                        z, raw = self.model.forward_pairs(*inputs)
                        loss = F.binary_cross_entropy_with_logits(z.float(), label_l[begin:begin+len(z)], reduction="sum")/32
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite supervised loss")
                    loss.backward() if fp32 else self.scaler.scale(loss).backward()
                    sup_total += float(loss.detach())
                for i, x in enumerate(chunks(us, self.physical)):
                    begin = i*self.physical
                    inputs = tuple(v.float() if fp32 and v.is_floating_point() else v for v in x)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
                        z, raw = self.model.forward_pairs(*inputs)
                        values = weighted_bce(z, target_u[begin:begin+len(z)], mask[begin:begin+len(z)])
                        raw_loss = values.sum()/32
                        loss = raw_loss*self.cfg["training"]["lambda_u"]
                        if self.cfg["method"] == "simmatch":
                            projected = F.normalize(self.model.sim_projection(raw).float(), dim=-1)
                            with torch.autocast("cuda", enabled=False):
                                instance_values = instance_ce(projected, self.bank, instance_target[begin:begin+len(z)], self.cfg["method_settings"]["T"])
                                iloss = instance_values.sum()/32
                            # Keep zero-gradient projection graph during warmup, as USB does.
                            loss = loss + iloss*self.cfg["method_settings"]["in_loss_ratio"]*float(instance_active)
                            instance_total += float(iloss.detach())*float(instance_active)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite semi-supervised loss")
                    loss.backward() if fp32 else self.scaler.scale(loss).backward()
                    unsup_total += float(raw_loss.detach())
                if not fp32:
                    self.scaler.unscale_(self.opt)
                finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.params)
                if not finite:
                    if fp32:
                        raise FloatingPointError("Nonfinite FP32 gradient")
                    old = self.scaler.get_scale()
                    self.scaler.update(new_scale=max(1., old/2))
                    fp32 = old <= 16
                    lb = uw = us = weak = raw = z = loss = raw_loss = inputs = None
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
                    emap = dict(self.ema.named_parameters())
                    momentum = self.cfg["training"]["ema_decay"]
                    for name, p in self.model.named_parameters():
                        if p.requires_grad:
                            emap[name].mul_(momentum).add_(p, alpha=1-momentum)
                    if self.cfg["method"] == "simmatch":
                        indexes = torch.tensor([self.pair_to_index[p["pair_id"]] for p in lp], device=self.device)
                        self.bank[indexes] = l_ema_features.detach()
                self.step = step
                self.u_seen.update(p["pair_id"] for p in urows)
                seconds = time.monotonic()-timer
                self.elapsed_seconds += seconds
                result = {"step": step, "sup_loss": sup_total, "unsup_loss": unsup_total, "instance_loss": instance_total,
                    "instance_active": instance_active, "util_ratio": float(mask.mean()),
                    "pseudo_positive_fraction": float((target_u >= .5).float().mean()), "seconds": seconds,
                    "physical": self.physical, "scale": self.scaler.get_scale(), "fp32_retry": fp32, "attempts": attempts,
                    "l_pair_visits": step*32, "u_pair_visits": step*32, "unique_u_pairs": len(self.u_seen), "updated_utc": now()}
                if self.cfg["method"] == "softmatch":
                    result.update(confidence_mu=float(self.algorithm.mu), confidence_var=float(self.algorithm.var),
                                  min_weight=float(mask.min()), max_weight=float(mask.max()))
                else:
                    result.update(memory_pair_count=len(self.pair_to_index), instance_warmup_steps=self.warmup_steps)
                return result
            except torch.cuda.OutOfMemoryError:
                self.opt.zero_grad(set_to_none=True)
                lb = uw = us = weak = raw = z = loss = raw_loss = inputs = target_u = mask = None
                weak_features = l_ema_features = instance_target = values = projected = instance_values = None
                self.load_algorithm_state(original_state)
                gc.collect()
                torch.cuda.empty_cache()
                if self.physical <= 4:
                    raise
                self.physical //= 2
                state = self.scaler.state_dict()
                self.scaler = torch.amp.GradScaler("cuda")
                self.scaler.load_state_dict(state)

def check(run_id):
    cfg, contract, prov, step = base.check(run_id)
    last = ROOT / "outputs" / run_id / "last.pt"
    if last.exists():
        payload = torch.load(last, map_location="cpu", weights_only=False)
        if payload.get("algorithm_checkpoint_version") != "soft_sim_full_v1" or "algorithm_state" not in payload:
            raise RuntimeError("Last checkpoint lacks complete algorithm state")
    return cfg, contract, prov, step

def gate(run_id, gpu_uuid):
    cfg, contract, prov, _ = check(run_id)
    out = ROOT / "audit/gpu_gates" / run_id
    receipt = out / "passed.json"
    if receipt.exists():
        previous = read(receipt)
        if previous.get("passed") and previous["provenance"] == prov and previous["gpu_uuid"] == gpu_uuid:
            return
        raise RuntimeError("Existing gate has different fingerprint; preserve previous version")
    out.mkdir(parents=True, exist_ok=True)
    timer = time.monotonic()
    runner = Runner(cfg, contract, prov)
    runner.prime_bank()
    phase_offset = runner.warmup_steps if cfg["method"] == "simmatch" else 0
    # Engineer-only active-phase gate, NOT formal successful-update accounting.
    runner.step = phase_offset
    first = runner.train_step()
    first_payload = runner.snapshot()
    atomic_torch(out / "engineering_first.pt", first_payload)
    second = runner.train_step()
    reference = runner.snapshot()
    initial_hash, legacy_hash, peak = runner.initial_hash, runner.legacy_hash, torch.cuda.max_memory_allocated()
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    replay_runner = Runner(cfg, contract, prov)
    replay_runner.load(torch.load(out / "engineering_first.pt", map_location="cpu", weights_only=False))
    replay_step = replay_runner.train_step()
    replay = replay_runner.snapshot()
    for key in ("trainable_state", "ema", "optimizer", "scaler", "rng", "algorithm_state"):
        if key == "rng":
            # NumPy state has ndarray; compare its tensor-compatible representation.
            a, b = reference[key], replay[key]
            assert a["python"] == b["python"] and a["numpy"][0] == b["numpy"][0]
            assert (a["numpy"][1] == b["numpy"][1]).all() and a["numpy"][2:] == b["numpy"][2:]
            torch.testing.assert_close(a["torch"], b["torch"], rtol=0, atol=0)
            assert_nested_close(a["cuda"], b["cuda"])
        else:
            assert_nested_close(cpu_state(reference[key]), cpu_state(replay[key]))
    max_diff = max(float((reference["trainable_state"][k]-replay["trainable_state"][k]).abs().max()) for k in reference["trainable_state"])
    if cfg["method"] == "simmatch" and not (second["instance_active"] and second["instance_loss"] > 0):
        raise RuntimeError("SimMatch instance loss was not exercised")
    atomic_json(receipt, {"passed": True, "provenance": prov, "gpu_uuid": gpu_uuid,
        "engineering_only": True, "formal_successful_steps_added": 0, "no_Validation_Test": True,
        "actual_engineering_updates": 2, "replayed_updates": 1, "engineering_phase_offset": phase_offset,
        "first": first, "second": second, "replay": replay_step,
        "max_parameter_replay_difference": max_diff, "full_optimizer_scaler_rng_ema_algorithm_state_replay": True,
        "common_34_state_sha256": initial_hash, "legacy_common_tensor_sha256": legacy_hash,
        "trainable_keys": len(replay_runner.names), "cuda_peak_allocated_bytes": max(peak, torch.cuda.max_memory_allocated()),
        "bank_priming_pairs": replay_runner.bank_priming_pairs, "seconds": time.monotonic()-timer, "created_utc": now()})
    print("Real GPU + cold full-state replay gate passed", run_id, flush=True)

def verify_result(run_id):
    cfg, contract, prov, _ = check(run_id)
    out = ROOT / "outputs" / run_id
    result, status = read(out / "result.json"), read(out / "status.json")
    if result["provenance"] != prov or result["successful_steps"] != cfg["target_steps"] or status["state"] != "completed" or status["step"] != cfg["target_steps"]:
        raise RuntimeError("Completion accounting mismatch")
    for name in ("last", "best"):
        path = out / (name + ".pt")
        if digest(path) != result[name + "_sha256"]:
            raise RuntimeError("Completed checkpoint hash changed")
        p = torch.load(path, map_location="cpu", weights_only=False)
        if p["provenance"] != prov or p.get("algorithm_checkpoint_version") != "soft_sim_full_v1" or "algorithm_state" not in p:
            raise RuntimeError("Completed checkpoint is incomplete")
        if name == "last" and p["step"] != cfg["target_steps"]:
            raise RuntimeError("Last does not cover the target")
    return result

# Reuse only the registered scheduling/evaluation/checkpoint loop; extend its runner.
base.config_for = config_for
base.Runner = Runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-uuid")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--gate", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify-result", action="store_true")
    args = parser.parse_args()
    try:
        if args.check:
            _, _, _, step = check(args.run_id)
            print(json.dumps({"passed": True, "run_id": args.run_id, "resume_step": step}))
        elif args.verify_result:
            print(json.dumps(verify_result(args.run_id)))
        else:
            expected_uuid = read(ROOT / "configs/protocol.json")["server"]["gpu_uuid"]
            if args.gpu_uuid != expected_uuid:
                raise RuntimeError("GPU allocation mismatch")
            with exclusive(ROOT / "locks" / (args.gpu_uuid + ".lock")), exclusive(ROOT / "locks" / (args.run_id + ".lock")):
                if args.gate:
                    gate(args.run_id, args.gpu_uuid)
                else:
                    base.run(args.run_id, args.gpu_uuid)
    except BaseException:
        traceback.print_exc()
        atomic_json(ROOT / "audit/failures" / (args.run_id + f"_{os.getpid()}.json"), {
            "run_id": args.run_id, "mode": "check" if args.check else "gate" if args.gate else "verify" if args.verify_result else "run",
            "error": traceback.format_exc(), "utc": now(), "pid": os.getpid()})
        raise
