"""Explicit, separately charged per-cell Pair-USA relation-teacher preparation.

Returns a new private artifact; caller must preserve/hash/register it outside
Git. Does not import an old queue, reuse an old teacher, or generate admission.
"""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .data import private_path
from .references import model_components, source_module
from .spec import fingerprint, load_protocol, validate_config


def prepare_teacher_targets(config, data, protocol=None):
    p = load_protocol() if protocol is None else protocol
    validate_config(config, p)
    if data.cfg != config or fingerprint(data.p) != fingerprint(p):
        raise ValueError("Teacher configuration and bound private inputs differ")
    if not config["uses_pairusa"] or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("An explicitly authorized same-cell teacher needs one CUDA-visible GPU")
    started = time.monotonic()
    device = torch.device("cuda:0")
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    parts = model_components()
    model_cfg = {**p["model"], "checkpoint": data.checkpoint_path, "seed": seed}
    descriptor = parts.FrozenALBEFDescriptor(model_cfg).to(device).eval()
    from albef_ssl.model import get_tokenizer
    tokenizer = get_tokenizer(str(data.root / "assets/tokenizer"))
    mean = torch.tensor([.48145466, .4578275, .40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.26862954, .26130258, .27577711], device=device).view(1, 3, 1, 1)
    @torch.no_grad()
    def descriptors(rows, flip=False):
        out = []
        for begin in range(0, len(rows), 16):
            group = rows[begin:begin+16]
            images = []
            for row in group:
                with Image.open(private_path(data.root, data.assets[row["image_id"]]["path"])) as image:
                    image = image.convert("RGB").resize((384, 384), Image.Resampling.BICUBIC)
                    if flip:
                        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    images.append(np.asarray(image).copy())
            pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device).float()/255.
            texts = [row[k] for row in group for k in ("positive_text", "negative_text")]
            encoded = tokenizer(texts, truncation=False, padding=True, return_tensors="pt")
            if encoded["input_ids"].shape[1] > data.manifest["token_guard"]:
                raise ValueError("Refuse to truncate a teacher caption")
            with torch.autocast("cuda", dtype=torch.float16):
                image_vectors = descriptor.encode_image((pixels-mean)/std).repeat_interleave(2, dim=0)
                text_vectors = descriptor.encode_text(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            out.append(descriptor.relation_features(image_vectors, text_vectors).float().cpu())
        return torch.cat(out)
    train_x = descriptors(data.l)
    val_x = descriptors(data.val)
    flipped_x = descriptors(data.l, True)
    del descriptor
    torch.cuda.empty_cache()
    teacher = parts.PairRelationTeacher(seed).to(device)
    settings = p["ablation"]
    opt = torch.optim.AdamW(teacher.parameters(), lr=settings["teacher_lr"], weight_decay=p["training"]["weight_decay"])
    labels = torch.tensor([1., 0.] * len(data.l))
    metric = source_module("experiments/multicrop_itm_mt_fixmatch_v1/code/metrics.py")
    history, best, best_auc, stale, updates = [], None, -math.inf, 0, 0
    for epoch in range(1, settings["teacher_max_epochs"]+1):
        teacher.train()
        order_seed = int(fingerprint([config["dataset"], config["budget"], seed, epoch, "teacher_L_order"])[:16], 16)
        order = torch.randperm(len(data.l), generator=torch.Generator().manual_seed(order_seed))
        for begin in range(0, len(order), 16):
            anchors = order[begin:begin+16]
            indices = torch.stack((2*anchors, 2*anchors+1), dim=1).flatten()
            _, logits = teacher(train_x[indices].to(device))
            loss = F.binary_cross_entropy_with_logits(logits.float(), labels[indices].to(device))
            opt.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite teacher loss; preserve evidence, no skipped update")
            loss.backward()
            if any(v.grad is not None and not bool(torch.isfinite(v.grad).all()) for v in teacher.parameters()):
                raise FloatingPointError("Nonfinite teacher gradient")
            norm = torch.nn.utils.clip_grad_norm_(teacher.parameters(), p["training"]["max_grad_norm"])
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite teacher clipping norm")
            opt.step(); updates += 1
        teacher.eval()
        with torch.no_grad():
            probabilities = teacher(val_x.to(device))[1].sigmoid().float().cpu().numpy()
        auc = metric.binary_metrics([1, 0]*len(data.val), probabilities)["auroc"]
        history.append(dict(epoch=epoch, validation_auroc=auc))
        if auc > best_auc:
            best_auc, stale, best = auc, 0, copy.deepcopy(teacher.state_dict())
        else:
            stale += 1
        if stale >= settings["teacher_patience"]:
            break
    teacher.load_state_dict(best); teacher.eval()
    @torch.no_grad()
    def positives(features):
        return torch.cat([teacher(features[i:i+256].to(device))[0].float().cpu()
                          for i in range(0, len(features), 256)])[::2]
    targets = torch.stack((positives(train_x), positives(flipped_x)), dim=1)
    probe = F.normalize(targets[:16, 0], dim=-1)
    off_diagonal = (probe@probe.T)[~torch.eye(len(probe), dtype=torch.bool)]
    if not bool(torch.isfinite(targets).all()) or float(off_diagonal.var()) <= 1e-12:
        raise RuntimeError("Teacher relations collapsed; do not silently admit this artifact")
    return dict(format="fair_pairusa_targets_v2",
                identity=dict(dataset=config["dataset"], budget=config["budget"], seed=seed,
                              l_sha256=data.manifest["budgets"][config["budget"]]["l"]["sha256"],
                              validation_sha256=data.manifest["validation"]["sha256"], protocol_sha256=fingerprint(p)),
                image_ids=[r["image_id"] for r in data.l], targets=targets,
                teacher_state={k: v.cpu() for k, v in best.items()}, training_updates=updates,
                validation_history=history, full_upstream_seconds=time.monotonic()-started,
                training_labels_source="this_cell_L_only", test_evaluated=False)
