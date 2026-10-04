"""One training engine for the controlled ablations and five SSL mechanisms.

Reuses the archived mathematical components, not their deployment/queue code.
CPU injection is for synthetic tests only; real model construction needs CUDA,
private official assets, input contracts and independent GPU admission.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .references import model_components, numerical_components, source_module
from .evaluation import validate_test_selection
from .spec import auxiliary_weights, fingerprint, load_protocol, lr_multiplier, sample_indices, validate_config


def tree_digest(value):
    h = hashlib.sha256()
    def walk(x):
        if torch.is_tensor(x):
            v = x.detach().cpu().contiguous()
            h.update(str((tuple(v.shape), str(v.dtype))).encode())
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.shape, str(x.dtype))).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode()); walk(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for v in x:
                walk(v)
        else:
            h.update(repr(x).encode())
    walk(value)
    return h.hexdigest()


def finite_tree(value):
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return not np.issubdtype(value.dtype, np.floating) or bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


class TorchBackend:
    def __init__(self, config, data, protocol=None, model=None):
        started = time.monotonic()
        self.p = load_protocol() if protocol is None else copy.deepcopy(protocol)
        self.cfg = config = validate_config(copy.deepcopy(config), self.p)
        if data.cfg != config:
            raise ValueError("Training configuration and bound private inputs differ")
        self.input_identity = copy.deepcopy(data.input_identity)
        self.holdout_identity = copy.deepcopy(data.holdout_identity)
        self.test_contract_sha256, self.test_anchors = data.test_contract_sha256, data.test_anchors
        self._test_consumed, self.test_forward_pairs = False, 0
        self.data, self.method, self.step = data, config["method"], 0
        self._poisoned = False
        self.simulation_only = model is not None
        if model is None:
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("One registered CUDA-visible GPU is required for real ALBEF training")
            self.device = torch.device("cuda:0")
            torch.cuda.reset_peak_memory_stats(self.device)
            random.seed(config["seed"]); np.random.seed(config["seed"])
            torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            model_cfg = {**self.p["model"], "checkpoint": data.checkpoint_path,
                         "seed": config["seed"], "use_pairusa": config["uses_pairusa"],
                         "fusion_chunk_size": self.p["training"]["physical_pairs"]}
            model = model_components().PairITMModel(model_cfg).to(self.device)
        else:
            self.device = next(model.parameters()).device
            if self.device.type != "cpu":
                raise ValueError("Injected models are restricted to CPU synthetic tests")
        self.model = model.train()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        self.common_names = [n for n, v in model.named_parameters() if v.requires_grad
                             and not any(s in n for s in ("student_projection", "temperature", "sim_projection"))]
        if not self.simulation_only and len(self.common_names) != 34:
            raise RuntimeError("Common LoRA/ITM topology changed")
        self.initial_common_hash = tree_digest(self._named_state(self.model, self.common_names))
        if self.method == "simmatch":
            s = self.p["algorithms"]["simmatch"]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config["seed"] + s["projector_seed_offset"])
                self.model.sim_projection = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, s["proj_size"])).to(self.device)
        self.ema = copy.deepcopy(self.model).requires_grad_(False).eval()
        self.names = [n for n, v in self.model.named_parameters() if v.requires_grad]
        self.params = [v for v in self.model.parameters() if v.requires_grad]
        t = self.p["training"]
        groups = []
        for selected, rate in (("lora", t["lora_lr"]), ("head", t["head_lr"])):
            params = [v for n, v in self.model.named_parameters() if v.requires_grad and "sim_projection" not in n
                      and (("lora_" in n) == (selected == "lora"))]
            if params:
                groups.append(dict(params=params, lr=rate, base_lr=rate))
        if self.method == "simmatch":
            rate = self.p["algorithms"]["simmatch"]["projector_lr"]
            groups.append(dict(params=list(self.model.sim_projection.parameters()), lr=rate, base_lr=rate))
        self.opt = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])
        self.scaler = torch.amp.GradScaler("cuda", init_scale=t["amp_initial_scale"], growth_interval=2000,
                                          enabled=self.device.type == "cuda")
        self.physical = t["physical_pairs"]
        self.parts = numerical_components()
        self.algorithm = None
        if self.method == "softmatch":
            s = self.p["algorithms"]["softmatch"]
            self.algorithm = self.parts["soft_sim"].SoftMatchStatistics(s["ema_p"], s["n_sigma"], s["variance_min"])
        elif self.method == "simmatch":
            self.algorithm = self.parts["soft_sim"].DistributionQueue(self.p["algorithms"]["simmatch"]["da_len"])
        elif self.method == "freematch":
            self.algorithm = self.parts["free"].FreeMatchState(self.p["algorithms"]["freematch"]["ema_p"], self.device)
        self.bank, self.bank_labels, self.pair_to_index = None, None, {}
        self.instance_warmup = math.ceil(len(data.l) / 16) if self.method == "simmatch" else 0
        self._cost = None
        data.attach(self.model, self.device)
        self.teacher_identity = data.teacher_identity if config["uses_pairusa"] else None
        self.teacher_resources = copy.deepcopy(data.teacher_resources) if config["uses_pairusa"] else None
        if self.method == "simmatch":
            pairs = self.data.flatten(self.data.l)
            self.pair_to_index = {p["pair_id"]: i for i, p in enumerate(pairs)}
            if len(self.pair_to_index) != len(pairs):
                raise ValueError("L pair-ID memory keys collide")
            self.bank_labels = torch.tensor([p["label"] for p in pairs], dtype=torch.long, device=self.device)
        bank_seconds, bank_pairs = self._prime_bank() if self.method == "simmatch" and not config["shared_bce_steps"] else (0., 0)
        self._synchronize()
        self.initial_cost = dict(setup_seconds=max(0., time.monotonic() - started - bank_seconds),
                                 bank_initialization_seconds=bank_seconds, l_forward_pairs=bank_pairs,
                                 **(data.teacher_accounting if config["uses_pairusa"] else {}))

    @staticmethod
    def _named_state(model, names):
        return {n: v.detach().cpu().clone() for n, v in model.named_parameters() if n in names}

    def _assert_healthy(self):
        if self._poisoned:
            raise RuntimeError("Optimizer/EMA commit failed; use a fresh worker and last complete checkpoint")

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _amp(self, enabled=True):
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=enabled and self.device.type == "cuda")

    def _pieces(self, batch):
        for start in range(0, len(batch[0]), self.physical):
            yield start, tuple(v[start:start + self.physical] for v in batch)

    def _forward(self, model, batch, which, amp=True):
        if self._cost is not None:
            self._cost[which + "_forward_pairs"] += len(batch[0])
        inputs = tuple(v.float() if not amp and v.is_floating_point() else v for v in batch)
        with self._amp(amp):
            return model.forward_pairs(*inputs)

    @torch.no_grad()
    def _predict(self, model, batch, which, amp=True, projected=False):
        logits, features = [], []
        for _, chunk in self._pieces(batch):
            z, raw = self._forward(model, chunk, which, amp)
            logits.append(z.float())
            if projected:
                with self._amp(amp):
                    raw = F.normalize(model.sim_projection(raw).float(), dim=-1)
            features.append(raw.float())
        return torch.cat(logits), torch.cat(features)

    def _prime_bank(self):
        start = time.monotonic()
        pairs = self.data.flatten(self.data.l)
        self.pair_to_index = {p["pair_id"]: i for i, p in enumerate(pairs)}
        if len(self.pair_to_index) != len(pairs):
            raise ValueError("L pair-ID memory keys collide")
        self.bank_labels = torch.tensor([p["label"] for p in pairs], dtype=torch.long, device=self.device)
        self.bank = torch.cat([self._predict(self.ema, self.data.batch(pairs[i:i+32], 0, "weak", self.physical),
                                              "l", projected=True)[1] for i in range(0, len(pairs), 32)])
        self._synchronize()
        return time.monotonic() - start, len(pairs)

    def _algorithm_state(self):
        value = None if self.algorithm is None else self.algorithm.state_dict()
        return dict(statistics=copy.deepcopy(value), bank=None if self.bank is None else self.bank.detach().cpu().clone(),
                    bank_labels=None if self.bank_labels is None else self.bank_labels.detach().cpu().clone(),
                    pair_to_index=copy.deepcopy(self.pair_to_index), instance_warmup=self.instance_warmup)

    def _restore_algorithm(self, state, checkpoint_step=None):
        if state["instance_warmup"] != self.instance_warmup or state["pair_to_index"] != self.pair_to_index:
            raise ValueError("SimMatch memory membership/warmup changed")
        if self.algorithm is not None:
            if self.method == "freematch":
                self.algorithm.load(state["statistics"])
            else:
                self.algorithm.load_state_dict(state["statistics"])
        elif state["statistics"] is not None:
            raise ValueError("Unexpected algorithm statistics")
        if self.method == "simmatch":
            should_be_ready = not self.cfg["shared_bce_steps"] or (checkpoint_step is not None and checkpoint_step > self.cfg["shared_bce_steps"])
            if checkpoint_step is not None and (state["bank"] is not None) != should_be_ready:
                raise ValueError("SimMatch memory must initialize at the registered SSL phase boundary")
            if not torch.equal(state["bank_labels"], self.bank_labels.cpu()) or (state["bank"] is not None and state["bank"].shape != (len(self.pair_to_index), self.p["algorithms"]["simmatch"]["proj_size"])):
                raise ValueError("Complete SimMatch pair memory is required")
            self.bank = None if state["bank"] is None else state["bank"].to(self.device).clone()
        elif state["bank"] is not None or state["bank_labels"] is not None:
            raise ValueError("Unexpected memory bank")

    def _usa_gradient(self, lbatch, selected_l, step, amp):
        projections = []
        with torch.no_grad():
            for start, chunk in self._pieces(lbatch):
                _, raw = self._forward(self.model, chunk, "l", amp)
                keep = (torch.arange(len(raw), device=self.device) + start) % 2 == 0
                with self._amp(amp):
                    projections.append(self.model.student_projection(raw[keep]).float())
        projected = torch.cat(projections).detach().requires_grad_(True)
        temperature = self.model.log_student_temperature.detach().float().requires_grad_(True)
        targets = self.data.teacher_vectors(selected_l, step).to(self.device)
        # Import only the safe model/loss module; no legacy teacher/queue initializer.
        loss_fn = self.data.pairusa_loss
        value = loss_fn(targets, projected, self.p["ablation"]["pairusa_teacher_temperature"], temperature.exp().clamp_min(.01))
        gradient, temperature_gradient = torch.autograd.grad(value, (projected, temperature))
        if not finite_tree((value, gradient, temperature_gradient)):
            raise FloatingPointError("Nonfinite global Pair-USA gradient")
        return projected.detach(), gradient.detach(), temperature_gradient.detach(), value.detach()

    def _backward(self, loss, pairs, amp):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite registered loss; no silent loss skipping")
        self._cost["backward_pairs"] += pairs
        self.scaler.scale(loss).backward() if amp else loss.backward()

    def train_step(self):
        self._assert_healthy()
        if self._test_consumed:
            raise ValueError("Training must not continue after Test access")
        step = self.step + 1
        if step > self.cfg["target_steps"]:
            raise ValueError("Student budget exhausted")
        li, ui = sample_indices(self.cfg, step, len(self.data.l), len(self.data.u), self.p)
        selected_l = [self.data.l[i] for i in li]
        selected_u = [self.data.u[i] for i in ui]
        lp = self.data.flatten(selected_l)
        labels = torch.tensor([p["label"] for p in lp], device=self.device)
        weights = auxiliary_weights(self.cfg, step, self.p)
        phase_step = step - self.cfg["shared_bce_steps"]
        ssl_active = self.method in ("meanteacher", "fixmatch", "softmatch", "simmatch", "freematch") and phase_step > 0
        use_u = ssl_active or weights["ot"] > 0
        start = time.monotonic()
        self._cost = dict(attempts=0, failed_attempts=0, l_forward_pairs=0, u_forward_pairs=0, backward_pairs=0)
        amp = self.device.type == "cuda"
        try:
            if ssl_active and self.method == "simmatch" and self.bank is None:
                bank_seconds, _ = self._prime_bank()
                self._cost["bank_initialization_seconds"] = bank_seconds
            original_algorithm = self._algorithm_state()
            for attempt in range(1, self.p["training"]["max_same_step_attempts"] + 1):
                commit_started = False
                # Release every previous attempt's activation/gradient surrogate
                # before allocating a new physical batch, especially after OOM.
                weak = features = strong = anchors = usa = projected = surrogate = None
                z = raw = loss = values = unweighted = instance = indices = None
                target = mask = proposed = fairness_gradient = instance_target = l_ema = None
                self._cost["attempts"] += 1
                self._restore_algorithm(original_algorithm)
                self.opt.zero_grad(set_to_none=True)
                self.model.fusion_chunk_size = self.ema.fusion_chunk_size = self.physical
                lbatch = uw = us = None
                try:
                    lbatch = self.data.batch(lp, step, "weak", self.physical)
                    target = mask = proposed = fairness_gradient = instance_target = l_ema = None
                    fairness = torch.tensor(0., device=self.device)
                    ramp = 1.
                    if use_u:
                        uw = self.data.batch(selected_u, step, "weak", self.physical)
                        us = self.data.batch(selected_u, step, "strong", self.physical)
                        teacher = self.ema if self.method in ("meanteacher", "simmatch") else self.model
                        weak, features = self._predict(teacher, uw, "u", amp, projected=self.method == "simmatch")
                        if self.method == "meanteacher":
                            target, mask = weak.sigmoid(), torch.ones(32, device=self.device)
                            horizon = self.cfg["target_steps"] - self.cfg["shared_bce_steps"]
                            ramp = min(1., (phase_step-1) / (self.p["algorithms"]["meanteacher"]["unsup_warm_up"] * horizon))
                        elif self.method == "fixmatch":
                            target, mask = self.parts["losses"].fixmatch_targets(weak, self.p["algorithms"]["fixmatch"]["p_cutoff"])
                        elif self.method == "softmatch":
                            target, mask, _ = self.algorithm.targets(weak)
                        elif self.method == "simmatch":
                            s = self.p["algorithms"]["simmatch"]
                            aligned = self.algorithm.align(self.parts["losses"].probabilities(weak))
                            target, mask, instance_target, _ = self.parts["soft_sim"].simmatch_targets(
                                aligned, features, self.bank, self.bank_labels, s["T"], s["smoothing_alpha"],
                                phase_step <= self.instance_warmup, s["p_cutoff"])
                            _, l_ema = self._predict(self.ema, lbatch, "l", amp, projected=True)
                        elif self.method == "freematch":
                            target, mask, proposed, _ = self.algorithm.propose(weak)
                            strong, _ = self._predict(self.model, us, "u", amp)
                            fairness, fairness_gradient = self.parts["free"].saf_value_and_gradient(strong, mask, proposed)
                        else:
                            _, anchors = self._predict(self.model, lbatch, "l", amp)
                            a = self.p["ablation"]
                            target, _ = self.parts["ot"].soft_ot_targets(anchors, labels, features,
                                epsilon=a["ot_epsilon"], max_iterations=a["ot_max_iterations"], tolerance=a["ot_both_marginal_tolerance"])
                            mask = torch.ones(32, device=self.device)
                            ramp = weights["ot"]
                    usa = self._usa_gradient(lbatch, selected_l, step, amp) if weights["pairusa"] > 0 else None
                    for group in self.opt.param_groups:
                        group["lr"] = group["base_lr"] * lr_multiplier(step, self.p)
                    sup_total, unsup_total, instance_total = 0., 0., 0.
                    for begin, chunk in self._pieces(lbatch):
                        z, raw = self._forward(self.model, chunk, "l", amp)
                        loss = F.binary_cross_entropy_with_logits(z.float(), labels[begin:begin+len(z)], reduction="sum") / 32
                        sup_total += float(loss.detach())
                        if usa is not None:
                            keep = (torch.arange(len(z), device=self.device) + begin) % 2 == 0
                            with self._amp(amp):
                                projected = self.model.student_projection(raw[keep]).float()
                            first = begin // 2
                            torch.testing.assert_close(projected, usa[0][first:first+len(projected)], rtol=0, atol=0)
                            surrogate = (projected * usa[1][first:first+len(projected)]).sum()
                            if begin == 0:
                                surrogate = surrogate + self.model.log_student_temperature * usa[2]
                            loss = loss + weights["pairusa"] * surrogate
                        self._backward(loss, len(z), amp)
                    if use_u:
                        for begin, chunk in self._pieces(us):
                            z, raw = self._forward(self.model, chunk, "u", amp)
                            values = (self.parts["losses"].mean_teacher_loss(z, target[begin:begin+len(z)])
                                      if self.method == "meanteacher" else
                                      self.parts["losses"].weighted_bce(z, target[begin:begin+len(z)], mask[begin:begin+len(z)]))
                            unweighted = values.sum() / 32
                            loss = unweighted * ramp * self.p["training"]["lambda_u"]
                            unsup_total += float(unweighted.detach())
                            if self.method == "simmatch":
                                with self._amp(amp):
                                    projected = F.normalize(self.model.sim_projection(raw).float(), dim=-1)
                                s = self.p["algorithms"]["simmatch"]
                                instance = self.parts["soft_sim"].instance_ce(projected, self.bank, instance_target[begin:begin+len(z)], s["T"]).sum()/32
                                active = float(phase_step > self.instance_warmup)
                                loss = loss + s["in_loss_ratio"] * active * instance
                                instance_total += float(instance.detach()) * active
                            if self.method == "freematch":
                                torch.testing.assert_close(z.float(), strong[begin:begin+len(z)], rtol=0, atol=0)
                                loss = loss + self.p["algorithms"]["freematch"]["ent_loss_ratio"] * (z.float()*fairness_gradient[begin:begin+len(z)]).sum()
                            self._backward(loss, len(z), amp)
                    if amp:
                        self.scaler.unscale_(self.opt)
                    if not all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in self.params):
                        if not amp:
                            raise FloatingPointError("Nonfinite FP32 gradients")
                        old_scale = self.scaler.get_scale()
                        self.scaler.update(new_scale=max(1., old_scale / 2))
                        amp = old_scale > 16
                        self._cost["failed_attempts"] += 1
                        continue
                    norm = torch.nn.utils.clip_grad_norm_(self.params, self.p["training"]["max_grad_norm"])
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError("Nonfinite clipping norm")
                    # An optimizer/EMA failure may have partially mutated live
                    # state. Never retry it as if it were a pre-update OOM.
                    commit_started = True
                    try:
                        if amp:
                            self.scaler.step(self.opt); self.scaler.update()
                        else:
                            self.opt.step()
                        with torch.no_grad():
                            ema_params = dict(self.ema.named_parameters())
                            decay = self.p["training"]["ema_decay"]
                            for n, v in self.model.named_parameters():
                                if v.requires_grad:
                                    ema_params[n].mul_(decay).add_(v, alpha=1-decay)
                            if ssl_active and self.method == "simmatch":
                                indices = torch.tensor([self.pair_to_index[p["pair_id"]] for p in lp], device=self.device)
                                self.bank[indices] = l_ema
                        if ssl_active and self.method == "freematch":
                            self.algorithm.commit(proposed)
                        self._synchronize()
                    except BaseException:
                        self._poisoned = True
                        raise
                    self.step = step
                    cost = {**self._cost, "student_seconds": max(0., time.monotonic()-start-self._cost.get("bank_initialization_seconds", 0.)),
                            "l_pair_draws": 32, "u_pair_draws": len(selected_u)}
                    return dict(step=step, committed=True, cost=cost, physical_pairs=self.physical,
                                supervised_loss=sup_total, unlabeled_loss=unsup_total,
                                pairusa_loss=0. if usa is None else float(usa[3]),
                                saf_loss=float(fairness), instance_loss=instance_total,
                                auxiliary_weights=weights, lr_multiplier=lr_multiplier(step, self.p),
                                ssl_phase_step=phase_step if ssl_active else None,
                                unlabeled_weight=ramp if use_u else 0.,
                                fp32_retry=not amp and self.device.type == "cuda")
                except torch.cuda.OutOfMemoryError as error:
                    if commit_started:
                        raise RuntimeError("OOM during optimizer/EMA commit; preserve last complete checkpoint, do not retry live state") from error
                    self._cost["failed_attempts"] += 1
                    self.opt.zero_grad(set_to_none=True)
                    lbatch = uw = us = None
                    weak = features = strong = anchors = usa = projected = surrogate = None
                    z = raw = loss = values = unweighted = instance = indices = None
                    target = mask = proposed = fairness_gradient = instance_target = l_ema = None
                    gc.collect(); torch.cuda.empty_cache()
                    if self.physical <= 4:
                        raise
                    self.physical //= 2
                    state = self.scaler.state_dict()
                    self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
                    self.scaler.load_state_dict(state)
            raise RuntimeError("Repeated same-step failures; preserve evidence, do not count a successful update")
        finally:
            self._cost = None

    @torch.no_grad()
    def evaluate(self):
        self._assert_healthy()
        metrics_module = source_module("experiments/multicrop_itm_mt_fixmatch_v1/code/metrics.py")
        out = []
        for model, name in ((self.ema, "ema"), (self.model, "student")):
            was_training = model.training
            model.eval()
            try:
                logits = torch.cat([self._predict(model, self.data.batch(self.data.flatten(self.data.val[i:i+8]),
                                     self.step, "validation", self.physical), "l")[0].cpu()
                                    for i in range(0, len(self.data.val), 8)])
            finally:
                model.train(was_training)
            labels = [1, 0] * len(self.data.val)
            probabilities = logits.sigmoid().numpy()
            threshold = metrics_module.select_threshold(labels, probabilities)
            metrics = metrics_module.binary_metrics(labels, probabilities, threshold)
            metrics.update(threshold_0_5=metrics_module.binary_metrics(labels, probabilities, .5),
                           paired_accuracy=float((logits.reshape(-1, 2)[:, 0] > logits.reshape(-1, 2)[:, 1]).float().mean()),
                           evaluation_model=name, evaluation_split="validation", step=self.step, validation_anchors=len(self.data.val),
                           validation_pairs=len(labels))
            out.append(metrics)
        return tuple(out)

    def ema_state_fingerprint(self):
        self._assert_healthy()
        return tree_digest(dict(ema=self._named_state(self.ema, self.names), config_sha256=fingerprint(self.cfg),
                                input_identity_sha256=fingerprint(self.input_identity)))

    def resource_usage(self):
        common = sum(v.numel() for n, v in self.model.named_parameters() if n in self.common_names)
        trainable = sum(v.numel() for v in self.model.parameters() if v.requires_grad)
        return dict(student_total_parameters=sum(v.numel() for v in self.model.parameters()),
                    student_trainable_parameters=trainable, common_trainable_parameters=common,
                    method_specific_trainable_parameters=trainable-common,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None,
                    teacher=copy.deepcopy(self.teacher_resources))

    @torch.no_grad()
    def evaluate_test(self, selection):
        self._assert_healthy()
        validate_test_selection(selection, self.cfg, self.p, self.step,
                                self.ema_state_fingerprint(), self.test_contract_sha256)
        if self._test_consumed:
            raise ValueError("Test access was already consumed")
        self._test_consumed = True
        rows = self.data.load_test(selection)
        metrics_module = source_module("experiments/multicrop_itm_mt_fixmatch_v1/code/metrics.py")
        self.ema.eval()
        pieces = []
        for i in range(0, len(rows), 8):
            pairs = self.data.flatten(rows[i:i+8])
            batch = self.data.batch(pairs, self.step, "validation", self.physical)
            self.test_forward_pairs += len(pairs)
            pieces.append(self._predict(self.ema, batch, "l")[0].cpu())
        logits = torch.cat(pieces)
        probabilities, labels = logits.sigmoid().numpy(), [1, 0]*len(rows)
        # No threshold search, student comparison or model selection on Test.
        metrics = metrics_module.binary_metrics(labels, probabilities, selection["threshold"])
        metrics.update(threshold_0_5=metrics_module.binary_metrics(labels, probabilities, .5),
                       paired_accuracy=float((logits.reshape(-1, 2)[:, 0] > logits.reshape(-1, 2)[:, 1]).float().mean()),
                       evaluation_model="ema", evaluation_split="test", step=self.step,
                       test_anchors=len(rows), test_pairs=len(labels))
        return metrics

    def common_full_state_fingerprint(self):
        self._assert_healthy()
        optimizer = {n: copy.deepcopy(self.opt.state[v]) for n, v in self.model.named_parameters() if n in self.common_names}
        return tree_digest(dict(student=self._named_state(self.model, self.common_names),
                                ema=self._named_state(self.ema, self.common_names), optimizer=optimizer,
                                scaler=self.scaler.state_dict(), rng=self._rng(), step=self.step,
                                physical=self.physical))

    def _rng(self):
        return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [])

    def snapshot(self):
        self._assert_healthy()
        return dict(format="fair_torch_state_v3", config_sha256=fingerprint(self.cfg), step=self.step,
                    student=self._named_state(self.model, self.names), ema=self._named_state(self.ema, self.names),
                    optimizer=copy.deepcopy(self.opt.state_dict()), scaler=copy.deepcopy(self.scaler.state_dict()),
                    rng=self._rng(), physical=self.physical, algorithm=self._algorithm_state(),
                    initial_common_state_sha256=self.initial_common_hash, simulation_only=self.simulation_only,
                    input_identity_sha256=fingerprint(self.input_identity), teacher_targets_sha256=self.teacher_identity,
                    test_consumed=self._test_consumed, test_forward_pairs=self.test_forward_pairs)

    def restore(self, payload):
        expected = set(self.snapshot())
        if set(payload) != expected or payload["format"] != "fair_torch_state_v3" or payload["config_sha256"] != fingerprint(self.cfg) or payload["initial_common_state_sha256"] != self.initial_common_hash or payload["simulation_only"] != self.simulation_only or payload["input_identity_sha256"] != fingerprint(self.input_identity) or payload["teacher_targets_sha256"] != self.teacher_identity:
            raise ValueError("Backend fingerprint/initialization/complete-state mismatch")
        if type(payload["step"]) is not int or not 0 <= payload["step"] <= self.cfg["target_steps"] or payload["physical"] not in (16, 8, 4) or not finite_tree(payload):
            raise ValueError("Invalid/nonfinite complete state")
        if type(payload["test_consumed"]) is not bool or type(payload["test_forward_pairs"]) is not int or not 0 <= payload["test_forward_pairs"] <= 2*self.test_anchors or (payload["test_forward_pairs"] and not payload["test_consumed"]):
            raise ValueError("Invalid saved Test access state")
        if self._test_consumed and (not payload["test_consumed"] or payload["test_forward_pairs"] < self.test_forward_pairs):
            raise ValueError("Restoring an older snapshot cannot refund consumed Test access")
        if set(payload["algorithm"]) != set(self._algorithm_state()):
            raise ValueError("Incomplete method-specific state")
        if self.method == "freematch" and payload["algorithm"]["statistics"]["updates"] != max(0, payload["step"]-self.cfg["shared_bce_steps"]):
            raise ValueError("SAT statistics must commit once per successful optimizer update")
        for model, key in ((self.model, "student"), (self.ema, "ema")):
            current = dict(model.named_parameters())
            if set(payload[key]) != set(self.names) or any(current[n].shape != v.shape for n, v in payload[key].items()):
                raise ValueError("Trainable/EMA topology differs")
        if set(payload["rng"]) != {"python", "numpy", "torch", "cuda"} or len(payload["rng"]["cuda"]) != (1 if self.device.type == "cuda" else 0):
            raise ValueError("Complete RNG state is required")
        with torch.no_grad():
            for model, key in ((self.model, "student"), (self.ema, "ema")):
                for n, v in model.named_parameters():
                    if n in self.names:
                        v.copy_(payload[key][n].to(v))
        self.opt.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self._restore_algorithm(payload["algorithm"], payload["step"])
        self.step, self.physical = payload["step"], payload["physical"]
        self._test_consumed, self.test_forward_pairs = payload["test_consumed"], payload["test_forward_pairs"]
        random.setstate(payload["rng"]["python"]); np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
