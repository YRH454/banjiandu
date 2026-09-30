"""Official ALBEF 4M encoders and ITM head, with explicit LoRA adaptation.

The vendored architecture is salesforce/ALBEF commit
b9727e43c3040491774d1b22cc27718aa7772fac. All pretrained core tensors must load
before adapters or optional task heads are created. No BLIP substitutions.
"""
from __future__ import annotations

from functools import partial
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizerFast

from .vendor.albef.vit import VisionTransformer, interpolate_pos_embed
from .vendor.albef.xbert import BertConfig, BertModel

DEFAULT_CHECKPOINT = "/data/workspace/models/ALBEF/ALBEF_4M.pth"
DEFAULT_TOKENIZER = "/data/workspace/models/ALBEF/bert-base-uncased"
OFFICIAL_4M_MD5 = "3c876d776a8e0ce61e2285fc9897f0b3"


def get_tokenizer(path=None):
    tokenizer = BertTokenizerFast.from_pretrained(path or DEFAULT_TOKENIZER, local_files_only=True)
    expected = {"pad_token_id": 0, "unk_token_id": 100, "cls_token_id": 101,
                "sep_token_id": 102, "mask_token_id": 103}
    if len(tokenizer) != 30522 or any(getattr(tokenizer, key) != value for key, value in expected.items()):
        raise ValueError("ALBEF 4M requires the unmodified 30522-token bert-base-uncased vocabulary")
    if not tokenizer.do_lower_case:
        raise ValueError("ALBEF 4M tokenizer must lowercase text")
    return tokenizer


class LoRALinear(nn.Module):
    """Frozen original linear layer plus a zero-initialized rank-r residual."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank <= 0: raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.scale = float(alpha) / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


class LoRAQKV(nn.Module):
    """Adapt ViT query/value only; the key projection remains frozen."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank <= 0 or base.out_features != 3 * base.in_features:
            raise ValueError("Expected positive LoRA rank and fused square QKV")
        self.base = base
        self.base.requires_grad_(False)
        self.width = base.in_features
        self.scale = float(alpha) / rank
        self.lora_q_a = nn.Parameter(torch.empty(rank, self.width))
        self.lora_q_b = nn.Parameter(torch.zeros(self.width, rank))
        self.lora_v_a = nn.Parameter(torch.empty(rank, self.width))
        self.lora_v_b = nn.Parameter(torch.zeros(self.width, rank))
        nn.init.kaiming_uniform_(self.lora_q_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_v_a, a=math.sqrt(5))

    def forward(self, x):
        q = F.linear(F.linear(x, self.lora_q_a), self.lora_q_b) * self.scale
        v = F.linear(F.linear(x, self.lora_v_a), self.lora_v_b) * self.scale
        return self.base(x) + torch.cat((q, torch.zeros_like(q), v), dim=-1)


def _md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ALBEFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = dict(config)
        seed = int(config.get("seed", 20260825))
        self.image_size = int(config.get("image_size", 384))
        if self.image_size % 16: raise ValueError("ALBEF image size must be a multiple of 16")
        self.enable_disease_head = bool(config.get("enable_disease_head", False))
        self.enable_pair_projectors = bool(config.get("enable_pair_projectors", False))
        self.enable_sclip = bool(config.get("enable_sclip", False))
        self.local_layer = int(config.get("local_layer", 6))
        if not 6 <= self.local_layer <= 11: raise ValueError("local_layer must identify a fusion layer, 6 through 11")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.visual_encoder = VisionTransformer(
                img_size=self.image_size, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            bert_config = BertConfig.from_json_file(str(Path(__file__).parent / "vendor/albef/bert_config.json"))
            bert_config.use_cache = False
            self.text_encoder = BertModel(bert_config, add_pooling_layer=False)
            self.vision_proj = nn.Linear(768, 256)
            self.text_proj = nn.Linear(768, 256)
            self.itm_head = nn.Linear(768, 2)
            self.load_report = self._load_official(config.get("checkpoint", DEFAULT_CHECKPOINT))
            self.requires_grad_(False)
            # Shared adapters are initialized before every optional task head.
            # Optional flags therefore cannot perturb common initialization.
            cross_rank = int(config.get("cross_lora_rank", 8))
            vision_rank = int(config.get("vision_lora_rank", 4))
            vision_layers = int(config.get("last_vision_layers", 2))
            if not 1 <= vision_layers <= 12: raise ValueError("last_vision_layers must be in [1,12]")
            for layer in self.text_encoder.encoder.layer[bert_config.fusion_layer:]:
                attention = layer.crossattention.self
                for name in ("query", "value"):
                    setattr(attention, name, LoRALinear(getattr(attention, name), cross_rank,
                                                       config.get("cross_lora_alpha", 2 * cross_rank)))
            for block in self.visual_encoder.blocks[-vision_layers:]:
                block.attn.qkv = LoRAQKV(block.attn.qkv, vision_rank, config.get("vision_lora_alpha", 2 * vision_rank))
            self.itm_head.requires_grad_(True)
            text_rank = int(config.get("text_lora_rank", 0)) if self.enable_sclip else 0
            if text_rank:
                text_alpha = float(config.get("text_lora_alpha", 2 * text_rank))
                text_layers = int(config.get("text_lora_layers", bert_config.fusion_layer))
                if not 1 <= text_layers <= bert_config.fusion_layer:
                    raise ValueError("text_lora_layers must be within the unimodal BERT layers")
                for layer in self.text_encoder.encoder.layer[:text_layers]:
                    attention = layer.attention.self
                    for name in ("query", "value"):
                        setattr(attention, name, LoRALinear(getattr(attention, name), text_rank, text_alpha))
            if self.enable_sclip:
                self.vision_proj.requires_grad_(True)
                self.text_proj.requires_grad_(True)
        if self.enable_disease_head:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 101)
                self.disease_head = nn.Linear(768, 6)
        if self.enable_pair_projectors:
            dim = int(config.get("projection_dim", 256))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 202)
                self.global_projector = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, dim))
                self.local_projector = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, dim))
        self.load_report.update(
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            trainable_parameter_names=[n for n, p in self.named_parameters() if p.requires_grad],
            cross_lora_rank=cross_rank, cross_lora_alpha=float(config.get("cross_lora_alpha", 2 * cross_rank)),
            vision_lora_rank=vision_rank, vision_lora_alpha=float(config.get("vision_lora_alpha", 2 * vision_rank)),
            last_vision_layers=vision_layers, lora_dropout=0.0,
            sclip_enabled=self.enable_sclip,
            text_lora_rank=text_rank,
            text_lora_alpha=float(config.get("text_lora_alpha", 2 * text_rank)) if text_rank else 0.0,
            text_lora_layers=int(config.get("text_lora_layers", bert_config.fusion_layer)) if text_rank else 0,
            pretrained_projection_heads_trainable=self.enable_sclip,
            disease_feature="visual_encoder CLS" if self.enable_disease_head else None,
            local_feature=f"multimodal BERT layer {self.local_layer} output CLS; diagnostic, not patch-word alignment" if self.enable_pair_projectors else None)

    def _load_official(self, checkpoint):
        path = Path(checkpoint)
        if not path.is_file(): raise FileNotFoundError(f"Missing checkpoint: {path}")
        digest = _md5(path)
        if digest != OFFICIAL_4M_MD5: raise ValueError("Checkpoint is not the verified official ALBEF 4M payload")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = payload["model"]
        own = self.state_dict()
        selected, ignored, unexpected = {}, [], []
        for key, value in state.items():
            mapped = key.replace("text_encoder.bert.", "text_encoder.", 1) if key.startswith("text_encoder.bert.") else key
            if mapped in own:
                selected[mapped] = value
            elif (key.startswith(("visual_encoder_m.", "text_encoder_m.", "vision_proj_m.", "text_proj_m.", "text_encoder.cls."))
                  or key in {"image_queue", "text_queue", "queue_ptr", "idx_queue", "temp"}):
                ignored.append(key)
            else:
                unexpected.append(key)
        missing = sorted(set(own) - set(selected))
        if missing or unexpected:
            raise ValueError(f"ALBEF core checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        original_position_shape = list(selected["visual_encoder.pos_embed"].shape)
        selected["visual_encoder.pos_embed"] = interpolate_pos_embed(selected["visual_encoder.pos_embed"], self.visual_encoder)
        bad_shapes = [key for key in own if own[key].shape != selected[key].shape]
        if bad_shapes: raise ValueError(f"ALBEF core tensor shape mismatch: {bad_shapes}")
        self.load_state_dict(selected, strict=True)
        report = {"architecture": "official ALBEF ViT-B/16 + fusion BERT", "checkpoint": str(path),
                  "checkpoint_bytes": path.stat().st_size, "checkpoint_md5": digest,
                  "official_commit": "b9727e43c3040491774d1b22cc27718aa7772fac",
                  "checkpoint_keys": len(state), "loaded_core_keys": len(selected),
                  "missing_core_keys": [], "unexpected_keys": [], "ignored_pretraining_keys": sorted(ignored),
                  "original_position_shape": original_position_shape,
                  "adapted_position_shape": list(own["visual_encoder.pos_embed"].shape),
                  "pretrained_itm_head": True, "pretrained_projections": True}
        del payload, state, selected
        return report

    def forward(self, pixel_values, input_ids, attention_mask):
        image = self.visual_encoder(pixel_values)
        image_attention = torch.ones(image.shape[:2], dtype=attention_mask.dtype, device=image.device)
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   encoder_hidden_states=image, encoder_attention_mask=image_attention,
                                   return_dict=True, mode="multi_modal", use_cache=False,
                                   output_hidden_states=self.enable_pair_projectors)
        fused_cls = output.last_hidden_state[:, 0, :]
        result = {"logits": self.itm_head(fused_cls)}
        if self.enable_disease_head: result["disease_logits"] = self.disease_head(image[:, 0, :])
        if self.enable_pair_projectors:
            result["global_vector"] = self.global_projector(fused_cls)
            result["local_vector"] = self.local_projector(output.hidden_states[self.local_layer + 1][:, 0, :])
        return result

    def encode_image(self, pixel_values, return_cls=False):
        """Return normalized ALBEF image embeddings for ITC/S-CLIP losses."""
        image = self.visual_encoder(pixel_values)
        cls = image[:, 0, :]
        embedding = F.normalize(self.vision_proj(cls).float(), dim=-1)
        return (embedding, cls) if return_cls else embedding

    def encode_text(self, input_ids, attention_mask):
        """Return normalized unimodal text embeddings from layers before fusion."""
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   return_dict=True, mode="text", use_cache=False)
        return F.normalize(self.text_proj(output.last_hidden_state[:, 0, :]).float(), dim=-1)
