"""Isolated, unadmitted FreeMatch execution candidates; original source unchanged.
Retain the exact physical16 strong graphs instead of a redundant forward.
No formal launcher, no shortened caption, no new optimizer/statistics steps.
"""
import copy,gc,json,math,os,random,time
from collections import OrderedDict
import torch
from torch.nn import functional as F
import train
from data_backend import fixed_seed,flatten_l,chunks
from losses import weighted_bce
from freematch_algorithm import saf_value_and_gradient

class SpeedRunner(train.Runner):
    def __init__(self,cfg,contract,prov,checkpointing=True,decoded_capacity=256):
        super().__init__(cfg,contract,prov)
        self.actual_activation_checkpointing=checkpointing
        self.model.activation_checkpointing=checkpointing
        self.model.text_encoder.config.gradient_checkpointing=checkpointing
        self.decoded_capacity=decoded_capacity
        self.decoded=OrderedDict()
        reference_decode=self.cache.decode
        def decode(image_id):
            if image_id not in self.decoded:
                self.decoded[image_id]=reference_decode(image_id)
                if len(self.decoded)>decoded_capacity:self.decoded.popitem(last=False)
            self.decoded.move_to_end(image_id)
            return self.decoded[image_id].copy()
        if decoded_capacity:self.cache.decode=decode

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
            lb = uw = us = weak = z = loss = raw_loss = inputs = target_u = mask = strong = fairness = fairness_grad = proposed = thresholds = strong_graphs = None
            try:
                lb = self.cache.batch(lp, step, "weak", self.physical)
                uw = self.cache.batch(chosen_u, step, "weak", self.physical)
                us = self.cache.batch(chosen_u, step, "strong", self.physical)
                weak = self.predict(self.model, uw, amp=not fp32)
                target_u, mask, proposed, thresholds = self.algorithm.propose(weak)
                strong_graphs = []
                for x in chunks(us, self.physical):
                    inputs = tuple(v.float() if fp32 and v.is_floating_point() else v for v in x)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
                        strong_graphs.append(self.model.forward_pairs(*inputs)[0])
                strong = torch.cat([v.detach().float() for v in strong_graphs])
                fairness, fairness_grad = saf_value_and_gradient(strong, mask, proposed)
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
                        z = strong_graphs[i]
                        strong_graphs[i] = None
                        tar, weights = target_u[begin:begin+len(z)], mask[begin:begin+len(z)]
                        torch.testing.assert_close(z.float(), strong[begin:begin+len(z)], rtol=0, atol=0)
                        values = weighted_bce(z, tar, weights)
                        raw_loss = values.sum() / 32
                        loss = raw_loss * self.cfg["training"]["lambda_u"]
                        loss = loss + self.cfg["method_settings"]["ent_loss_ratio"] * (z.float()*fairness_grad[begin:begin+len(z)]).sum()
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
                self.algorithm.commit(proposed)
                self.step = step
                self.u_seen.update(p["pair_id"] for p in chosen_u)
                seconds = time.monotonic()-timer
                self.elapsed_seconds += seconds
                return {"step": step, "sup_loss": sup_total, "unsup_loss": unsup_total,
                        "saf_loss":float(fairness),"threshold_global":float(self.algorithm.time_p),
                        "threshold_negative":float(thresholds[0]),"threshold_positive":float(thresholds[1]),
                        "algorithm_successful_updates":self.algorithm.updates,
                        "extra_strong_forward_pair_visits":0,
                        "total_loss":sup_total+self.cfg["training"]["lambda_u"]*unsup_total+self.cfg["method_settings"]["ent_loss_ratio"]*float(fairness),
                        "lambda_u_effective": ramp * self.cfg["training"]["lambda_u"],
                        "util_ratio": float(mask.mean()), "pseudo_positive_fraction": float((target_u >= .5).float().mean()),
                        "seconds": seconds, "physical": self.physical, "scale": self.scaler.get_scale(),
                        "fp32_retry": fp32, "attempts": attempts, "l_pair_visits": step*32,
                        "u_pair_visits": step*32, "unique_u_pairs": len(self.u_seen), "updated_utc": now()}
            except torch.cuda.OutOfMemoryError:
                self.opt.zero_grad(set_to_none=True)
                lb = uw = us = weak = z = loss = raw_loss = inputs = target_u = mask = strong = fairness = fairness_grad = proposed = thresholds = strong_graphs = None
                gc.collect()
                torch.cuda.empty_cache()
                if self.physical <= 4:
                    raise
                self.physical //= 2
                # Reset scaler stage when OOM follows a partially constructed pass.
                state = self.scaler.state_dict()
                self.scaler = torch.amp.GradScaler("cuda")
                self.scaler.load_state_dict(state)
