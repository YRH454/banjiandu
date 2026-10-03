"""ALBEF image--text matching student and the frozen Pair-USA teacher pieces.

The student deliberately keeps the official two-logit ``itm_head`` intact and
frozen.  ``match_head`` is a scalar view of it, initialized as ``z1 - z0``;
this makes binary BCE numerically identical to the official ITM decision at
step zero while allowing the task head to adapt.  Pair-USA operates only on
the positive branch returned by :meth:`PairITMModel.forward_cached`.

The relation teacher is intentionally a separate object.  It uses another,
fully frozen copy of the original ALBEF image/text encoders, so a changing
student LoRA adapter can never change teacher targets.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from albef_ssl.model import ALBEFModel  # noqa: E402


def _checked_model_config(config: Mapping) -> dict:
    """Translate the registered flat ITM configuration to the core model."""
    required = ("checkpoint", "seed", "image_size")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"PairITMModel requires configuration keys: {missing}")
    vision_layers = list(config.get("vision_lora_layers", (10, 11)))
    cross_layers = list(config.get("cross_lora_layers", range(6, 12)))
    if vision_layers != [10, 11]:
        raise ValueError("Registered visual LoRA layers must be ViT blocks [10, 11]")
    if cross_layers != list(range(6, 12)):
        raise ValueError("Registered fusion LoRA layers must be BERT cross-attention layers 6--11")
    if int(config.get("image_size", 384)) != 384:
        raise ValueError("This registered experiment uses image_size=384")
    return {
        "checkpoint": str(config["checkpoint"]),
        "seed": int(config["seed"]),
        "image_size": int(config["image_size"]),
        "enable_disease_head": False,
        "enable_pair_projectors": False,
        "enable_sclip": False,
        "vision_lora_rank": int(config.get("vision_lora_rank", 4)),
        "vision_lora_alpha": float(config.get("vision_lora_alpha", 8)),
        "last_vision_layers": 2,
        "cross_lora_rank": int(config.get("cross_lora_rank", 8)),
        "cross_lora_alpha": float(config.get("cross_lora_alpha", 16)),
    }


def _set_dropout_eval(module: nn.Module) -> None:
    """Keep frozen ALBEF's pretraining dropout deterministic during adaptation."""
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.eval()


class PairITMModel(ALBEFModel):
    """LoRA-adapted ALBEF matcher with cacheable frozen prefixes.

    ``forward`` and ``forward_cached`` return ``(logits, positive_vectors)``.
    ``logits[:, 0]`` belongs to the true caption and ``logits[:, 1]`` to the
    same-image counterfactual caption.  The second output has shape ``[B,256]``
    and is the only student representation used in Pair-USA.
    """

    relation_dim = 256

    def __init__(self, config: Mapping):
        self.registered_config = dict(config)
        super().__init__(_checked_model_config(config))
        self.use_pairusa = bool(config.get("use_pairusa", True))
        self.fusion_chunk_size = int(config.get("fusion_chunk_size", 8))
        self.activation_checkpointing = bool(config.get("activation_checkpointing", True))
        if self.fusion_chunk_size < 1:
            raise ValueError("fusion_chunk_size must be positive")

        # The official head is retained for audits but never trained in this
        # task.  A scalar logit initialized as z1-z0 is BCE-equivalent.
        self.itm_head.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config["seed"]) + 303)
            self.match_head = nn.Linear(768, 1)
            self.student_projection = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, self.relation_dim),
            )
        with torch.no_grad():
            self.match_head.weight.copy_(self.itm_head.weight[1] - self.itm_head.weight[0])
            self.match_head.bias.copy_(self.itm_head.bias[1] - self.itm_head.bias[0])
        initial_temperature = float(config.get("initial_student_temperature", config.get("student_temperature", 0.07)))
        if initial_temperature <= 0:
            raise ValueError("initial_student_temperature must be positive")
        self.log_student_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature), dtype=torch.float32))
        if not self.use_pairusa:
            self.student_projection.requires_grad_(False)
            self.log_student_temperature.requires_grad_(False)

        # xBERT already implements non-reentrant per-layer checkpointing when
        # this flag and the module's training mode are both true.
        self.text_encoder.config.gradient_checkpointing = self.activation_checkpointing
        self.load_report.update(
            task="same-image counterfactual image-text matching",
            task_head="scalar logit initialized as official itm_head logit[1]-logit[0]",
            pairusa_enabled=self.use_pairusa,
            pairusa_relation_dim=self.relation_dim,
            pairusa_student_temperature_initial=initial_temperature,
            fusion_chunk_size=self.fusion_chunk_size,
            activation_checkpointing=self.activation_checkpointing,
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def train(self, mode: bool = True):
        """Enable checkpointing for LoRA layers while keeping dropout disabled."""
        super().train(mode)
        # ViT has no nonzero registered dropout.  xBERT needs training=True to
        # activate its existing checkpoint path, then its dropout modules are
        # independently held in eval mode for exact cache/full equivalence.
        _set_dropout_eval(self)
        self.itm_head.eval()
        return self

    def image_prefix(self, pixel_values: Tensor) -> Tensor:
        """Frozen patch embedding plus ViT blocks 0--9, shared by pos/neg text."""
        vision = self.visual_encoder
        with torch.no_grad():
            image = vision.patch_embed(pixel_values)
            image = torch.cat((vision.cls_token.expand(image.shape[0], -1, -1), image), dim=1)
            image = vision.pos_drop(image + vision.pos_embed[:, : image.shape[1], :])
            for block in vision.blocks[:10]:
                image = block(image)
        return image

    def text_prefix(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Frozen BERT text layers 0--5, cacheable by caption hash."""
        with torch.no_grad():
            return self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                mode="text",
                use_cache=False,
            ).last_hidden_state

    def _image_tail(self, image_prefix: Tensor) -> Tensor:
        image = image_prefix
        for block in self.visual_encoder.blocks[10:]:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                image = activation_checkpoint(block, image, use_reentrant=False)
            else:
                image = block(image)
        return self.visual_encoder.norm(image)

    def _fuse(self, image: Tensor, text_prefix: Tensor, attention_mask: Tensor) -> Tensor:
        """Run fusion in chunks while retaining one differentiable visual tail."""
        if image.shape[0] != text_prefix.shape[0] or image.shape[0] != attention_mask.shape[0]:
            raise ValueError("Each cached image, caption prefix, and mask batch must align")
        pieces: list[Tensor] = []
        for start in range(0, image.shape[0], self.fusion_chunk_size):
            stop = min(start + self.fusion_chunk_size, image.shape[0])
            image_part = image[start:stop]
            mask_part = attention_mask[start:stop]
            image_mask = torch.ones(image_part.shape[:2], dtype=mask_part.dtype, device=image_part.device)
            output = self.text_encoder(
                encoder_embeds=text_prefix[start:stop],
                attention_mask=mask_part,
                encoder_hidden_states=image_part,
                encoder_attention_mask=image_mask,
                return_dict=True,
                mode="fusion",
                use_cache=False,
            )
            pieces.append(output.last_hidden_state[:, 0, :])
        return torch.cat(pieces, dim=0)

    def forward_cached(
        self,
        image_prefix: Tensor,
        positive_text_prefix: Tensor,
        positive_attention_mask: Tensor,
        negative_text_prefix: Tensor,
        negative_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Score true and counterfactual captions from frozen cache tensors."""
        batch_size = image_prefix.shape[0]
        if any(value.shape[0] != batch_size for value in (positive_text_prefix, positive_attention_mask,
                                                           negative_text_prefix, negative_attention_mask)):
            raise ValueError("Positive and negative cached tensors must use the image batch size")
        image = self._image_tail(image_prefix)
        positive_cls = self._fuse(image, positive_text_prefix, positive_attention_mask)
        negative_cls = self._fuse(image, negative_text_prefix, negative_attention_mask)
        logits = torch.cat((self.match_head(positive_cls), self.match_head(negative_cls)), dim=-1)
        return logits, self.student_projection(positive_cls)

    def forward(
        self,
        pixel_values: Tensor,
        positive_input_ids: Tensor,
        positive_attention_mask: Tensor,
        negative_input_ids: Tensor,
        negative_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Full forward; equivalent to prefix caching in eval mode."""
        image = self.image_prefix(pixel_values)
        positive = self.text_prefix(positive_input_ids, positive_attention_mask)
        negative = self.text_prefix(negative_input_ids, negative_attention_mask)
        return self.forward_cached(image, positive, positive_attention_mask, negative, negative_attention_mask)

    def student_temperature(self) -> Tensor:
        return self.log_student_temperature.exp().clamp_min(0.01)

    def forward_pairs(self, image_prefix: Tensor, text_prefix: Tensor, mask: Tensor):
        """Aligned binary ITM pairs; no access to caption provenance/hidden labels."""
        features = self._fuse(self._image_tail(image_prefix), text_prefix, mask)
        return self.match_head(features).squeeze(-1), features

    def trainable_state(self) -> dict[str, Tensor]:
        return {name: parameter.detach().cpu().clone() for name, parameter in self.named_parameters()
                if parameter.requires_grad}

    def load_trainable_state(self, state: Mapping[str, Tensor]) -> None:
        expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if set(state) != expected:
            missing = sorted(expected - set(state))
            unexpected = sorted(set(state) - expected)
            raise ValueError(f"Checkpoint trainable state differs; missing={missing[:3]}, unexpected={unexpected[:3]}")
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name in state:
                    value = state[name]
                    if parameter.shape != value.shape:
                        raise ValueError(f"Shape mismatch for {name}: {parameter.shape} versus {value.shape}")
                    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


class FrozenALBEFDescriptor(nn.Module):
    """Frozen original ALBEF encoders for precomputing teacher pair features."""

    def __init__(self, config: Mapping):
        super().__init__()
        self.backbone = ALBEFModel(_checked_model_config(config))
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        # A teacher descriptor must never inherit the student's training mode.
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode_image(self, pixel_values: Tensor) -> Tensor:
        image = self.backbone.visual_encoder(pixel_values)
        return F.normalize(self.backbone.vision_proj(image[:, 0, :]).float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.backbone.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            mode="text",
            use_cache=False,
        )
        return F.normalize(self.backbone.text_proj(output.last_hidden_state[:, 0, :]).float(), dim=-1)

    @staticmethod
    def relation_features(image_vectors: Tensor, text_vectors: Tensor) -> Tensor:
        if image_vectors.shape != text_vectors.shape or image_vectors.ndim != 2 or image_vectors.shape[1] != 256:
            raise ValueError("Pair teacher relation features require aligned [B,256] image and text vectors")
        image_vectors = F.normalize(image_vectors.float(), dim=-1)
        text_vectors = F.normalize(text_vectors.float(), dim=-1)
        return torch.cat((image_vectors, text_vectors, image_vectors + text_vectors,
                          (image_vectors - text_vectors).abs(), image_vectors * text_vectors), dim=-1)


class PairRelationTeacher(nn.Module):
    """Trainable relation MLP/head over frozen 1,280-D teacher descriptors."""

    relation_dim = 256
    descriptor_dim = relation_dim * 5

    def __init__(self, seed: int = 20260825):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 404)
            self.teacher_relation = nn.Sequential(
                nn.LayerNorm(self.descriptor_dim),
                nn.Linear(self.descriptor_dim, self.relation_dim),
                nn.GELU(),
                nn.LayerNorm(self.relation_dim),
            )
            self.teacher_head = nn.Linear(self.relation_dim, 1)

    def forward(self, descriptors: Tensor) -> tuple[Tensor, Tensor]:
        if descriptors.ndim != 2 or descriptors.shape[1] != self.descriptor_dim:
            raise ValueError("PairRelationTeacher expects [B,1280] frozen ALBEF relation descriptors")
        vectors = self.teacher_relation(descriptors)
        return vectors, self.teacher_head(vectors).squeeze(-1)


def pair_usa_loss(
    teacher_vectors: Tensor,
    student_vectors: Tensor,
    teacher_temperature: float | Tensor,
    student_temperature: Tensor | float,
) -> Tensor:
    """FP32 ``KL(P_teacher || Q_student)`` over non-self positive-pair relations."""
    if teacher_vectors.shape != student_vectors.shape or teacher_vectors.ndim != 2:
        raise ValueError("Teacher and student Pair-USA vectors must have the same [B,D] shape")
    batch_size = teacher_vectors.shape[0]
    if batch_size < 2:
        return student_vectors.sum() * 0.0
    device_type = "cuda" if student_vectors.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        # Detaching here is intentional even if callers precomputed targets in
        # no_grad: it protects the teacher against accidental coupling.
        teacher = F.normalize(teacher_vectors.detach().float(), dim=-1)
        student = F.normalize(student_vectors.float(), dim=-1)
        teacher_similarity = teacher @ teacher.transpose(0, 1)
        student_similarity = student @ student.transpose(0, 1)
        diagonal = torch.eye(batch_size, dtype=torch.bool, device=student.device)
        teacher_similarity = teacher_similarity.masked_fill(diagonal, -1e4)
        student_similarity = student_similarity.masked_fill(diagonal, -1e4)
        teacher_probs = F.softmax(teacher_similarity / float(teacher_temperature), dim=-1)
        temperature = torch.as_tensor(student_temperature, device=student.device, dtype=torch.float32).clamp_min(0.01)
        student_log_probs = F.log_softmax(student_similarity / temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
