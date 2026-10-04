"""Real two-phase adapters over the shared Torch engine, no launch or takeover."""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch

from fair_benchmark.spec import fingerprint
from fair_benchmark.torch_backend import TorchBackend, finite_tree, tree_digest
from .contracts import validate_parent_receipt, validate_test_selection
from .data import PlateauPairData
from .spec import auxiliary_weights, load_protocol, lr_multiplier, sample_indices, validate_config


class PlateauBackend(TorchBackend):
    STATE_FORMAT = "plateau_torch_state_v4"
    load_protocol = staticmethod(load_protocol)
    validate_config = staticmethod(validate_config)
    validate_test_selection = staticmethod(validate_test_selection)

    def __init__(self, config, data, protocol=None, model=None):
        if model is None and not isinstance(data, PlateauPairData):
            raise ValueError("Real v4 training requires v4 hash-bound data")
        self.parent_receipt_sha256 = self.adaptation_common_hash = None
        super().__init__(config, data, protocol, model)

    def _bank_ready_at(self, step):
        return self.method == "simmatch" and step > 0

    def _sample_indices(self, step):
        return sample_indices(self.cfg, step, len(self.data.l), len(self.data.u), self.p)

    def _lr_multiplier(self, step):
        return lr_multiplier(self.cfg, step, self.p)

    def _auxiliary_weights(self, step):
        return auxiliary_weights(self.cfg, step, self.p)

    def common_full_state_fingerprint(self):
        self._assert_healthy()
        # Reading a defaultdict with [] would create empty optimizer entries;
        # phase-zero evidence must be a read-only observation of fresh state.
        optimizer = {n: copy.deepcopy(self.opt.state.get(v, {})) for n, v in self.model.named_parameters() if n in self.common_names}
        return tree_digest(dict(student=self._named_state(self.model, self.common_names),
                                ema=self._named_state(self.ema, self.common_names), optimizer=optimizer,
                                scaler=self.scaler.state_dict(), rng=self._rng(), step=self.step,
                                physical=self.physical))

    def import_parent(self, bundle):
        self._assert_healthy()
        if self.step or self.parent_receipt_sha256 is not None or self._test_consumed or self.opt.state:
            raise ValueError("Parent import is allowed exactly once in a fresh branch")
        if not isinstance(bundle, dict) or set(bundle) != {"receipt", "backend"}:
            raise ValueError("Parent receipt plus complete private tensor state required")
        receipt = validate_parent_receipt(bundle["receipt"], self.cfg, len(self.data.l), self.p)
        payload = bundle["backend"]
        if tree_digest(payload) != receipt["backend_state_sha256"] or payload.get("format") != self.STATE_FORMAT or payload.get("config_sha256") != fingerprint(receipt["config"]) or payload.get("step") != receipt["budget"]["step"]:
            raise ValueError("Parent tensors/full-state do not match the plateau receipt")
        if receipt["input_identity_sha256"] != fingerprint(self.input_identity) or receipt["initial_common_state_sha256"] != self.initial_common_hash or receipt["simulation_only"] != self.simulation_only or payload["test_consumed"] or payload["test_forward_pairs"] or not finite_tree(payload):
            raise ValueError("Parent inputs/initialization/execution/Test state differs")
        if payload["parent_receipt_sha256"] is not None or payload["adaptation_common_state_sha256"] is not None or payload["teacher_targets_sha256"] is not None:
            raise ValueError("A module branch cannot be substituted for BCE")
        if tree_digest(dict(ema=payload["ema"], config_sha256=payload["config_sha256"], input_identity_sha256=payload["input_identity_sha256"])) != receipt["terminal_ema_sha256"]:
            raise ValueError("Parent terminal EMA identity differs")
        if set(payload["rng"]) != {"python", "numpy", "torch", "cuda"} or len(payload["rng"]["cuda"]) != (1 if self.device.type == "cuda" else 0):
            raise ValueError("Complete same-environment parent RNG is required")
        for model, key in ((self.model, "student"), (self.ema, "ema")):
            current = dict(model.named_parameters())
            if set(payload[key]) != set(self.common_names) or any(current[n].shape != v.shape for n, v in payload[key].items()):
                raise ValueError("Parent common student/EMA topology differs")
        start = time.monotonic()
        with torch.no_grad():
            for model, key in ((self.model, "student"), (self.ema, "ema")):
                for n, v in model.named_parameters():
                    if n in self.common_names:
                        v.copy_(payload[key][n].to(v))
        # All branches reset optimizer/scaler/LR/physical batch equally. New
        # heads/statistics stay at their independently seeded cold state.
        random.setstate(payload["rng"]["python"]); np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        self.parent_receipt_sha256 = receipt["receipt_sha256"]
        self.adaptation_common_hash = self.common_full_state_fingerprint()
        self._synchronize()
        return max(0., time.monotonic()-start)

    def train_step(self):
        if self.cfg["stage"] != "bce" and self.parent_receipt_sha256 is None:
            raise ValueError("Adaptation/matched BCE requires the registered plateau parent")
        record = super().train_step()
        record.update(stage=self.cfg["stage"], phase_step=self.step, supervised_bce_retained=True)
        record["total_loss"] = sum(record["losses_weighted"].values())
        return record

    def snapshot(self):
        return {**super().snapshot(), "parent_receipt_sha256": self.parent_receipt_sha256,
                "adaptation_common_state_sha256": self.adaptation_common_hash}

    def restore(self, payload):
        if payload.get("parent_receipt_sha256") != self.parent_receipt_sha256 or payload.get("adaptation_common_state_sha256") != self.adaptation_common_hash:
            raise ValueError("Cannot change the branch parent or phase-zero state on resume")
        if self.cfg["stage"] != "bce" and self.parent_receipt_sha256 is None:
            raise ValueError("Import the trusted parent before resuming a branch")
        super().restore(payload)
