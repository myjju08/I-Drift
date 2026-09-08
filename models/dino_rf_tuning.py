"""Offline real/fake tuning of two DINO bottlenecks with a frozen real anchor."""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from models.ssl_resnet import _load_backbone_state, derive_ssl_map_features


TRAINABLE_BLOCKS = ("layer3.5", "layer4.2")
PRESERVATION_MAPS = ("layer3", "layer4_blk2", "layer4")


def atomic_torch_save(payload, destination: str | Path) -> None:
    """Replace the destination with a new inode, preserving retained hardlinks."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tuning-", suffix=".pt", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def module_fingerprint(module: nn.Module) -> str:
    """Hash parameter/buffer values independently of serialization and device."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class _ConditionalHead(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(channels, 1)
        self.embedding = nn.Embedding(num_classes, channels)
        self.scale = math.sqrt(channels)

    def forward(self, feature, labels):
        pooled = feature.float().mean(dim=(2, 3))
        return self.linear(pooled).squeeze(1) + (
            pooled * self.embedding(labels)
        ).sum(dim=1) / self.scale


class DinoRealFakeTuner(nn.Module):
    """Own student, immutable teacher, two conditional heads, and fixed scales.

    Student training is confined to non-BN parameters in layer3.5/layer4.2.
    All BatchNorm modules use original running statistics, including in train
    mode. Inputs are always detached: this module never trains a generator.
    """
    def __init__(
        self, backbone: nn.Module, *, num_classes: int = 1000,
        stage_channels=(1024, 2048), head_seed: int = 43,
    ):
        super().__init__()
        self.teacher = copy.deepcopy(backbone).eval().requires_grad_(False)
        self.student = copy.deepcopy(backbone).eval()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(head_seed))
            self.heads = nn.ModuleDict({
                "layer3": _ConditionalHead(int(stage_channels[0]), num_classes),
                "layer4": _ConditionalHead(int(stage_channels[1]), num_classes),
            })
        self.feature_keys = tuple(
            name + suffix for name in PRESERVATION_MAPS
            for suffix in ("", "_mean", "_std", "_mean_2", "_mean_4", "_std_2", "_std_4")
        )
        # layer3 and its terminal block (blk6) are two logical drift entries.
        weights = [2.0 if key.startswith("layer3") else 1.0 for key in self.feature_keys]
        self.register_buffer("feature_weights", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer("feature_scales", torch.ones(len(self.feature_keys)))
        self.register_buffer("scales_calibrated", torch.tensor(False))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self._configure_trainable()

    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        backbone = resnet50(weights=None)
        backbone.fc = nn.Identity()
        backbone.load_state_dict(_load_backbone_state(path, "dino_resnet50"), strict=True)
        return cls(backbone, **kwargs)

    def _configure_trainable(self):
        self.student.requires_grad_(False)
        for name in TRAINABLE_BLOCKS:
            self.student.get_submodule(name).requires_grad_(True)
        for module in self.student.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval().requires_grad_(False)
        self.teacher.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        for module in self.student.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def _maps(self, backbone, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected NCHW raw RGB inputs")
        x = ((images.detach().float() + 1.0) * 0.5 - self.image_mean) / self.image_std
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        maps = {"layer3": F.avg_pool2d(x, 2)}
        for index, block in enumerate(backbone.layer4, start=1):
            x = block(x)
            if index == 2:
                maps["layer4_blk2"] = F.avg_pool2d(x, 2)
        maps["layer4"] = F.avg_pool2d(x, 2)
        return maps

    @staticmethod
    def _features(maps):
        features = {}
        for name in PRESERVATION_MAPS:
            features.update(derive_ssl_map_features(
                name, maps[name], patch_mean_size=(2, 4), patch_std_size=(2, 4),
                use_std=True, use_mean=True,
            ))
        return features

    @torch.no_grad()
    def scale_statistics(self, real_images):
        """Teacher-only TRAIN-real sums/counts; caller controls microbatching."""
        features = self._features(self._maps(self.teacher, real_images))
        sums = torch.stack([features[key].double().square().sum() for key in self.feature_keys])
        counts = torch.tensor([features[key].numel() for key in self.feature_keys], device=sums.device, dtype=torch.float64)
        return sums, counts

    @torch.no_grad()
    def set_scales(self, sums, counts, floor=1.0e-4):
        if not torch.isfinite(sums).all() or not torch.isfinite(counts).all() or not (counts > 0).all():
            raise ValueError("Invalid fixed teacher-scale statistics")
        self.feature_scales.copy_((sums / counts).clamp_min(float(floor) ** 2).sqrt().float())
        self.scales_calibrated.fill_(True)

    def forward(self, real, fake, labels, *, preservation_weight=10.0, head_only=False, return_scores=False):
        if real.shape != fake.shape or labels.shape != (real.shape[0],):
            raise ValueError("Expected aligned, same-class real/fake pairs")
        if not bool(self.scales_calibrated):
            raise RuntimeError("Calibrate fixed scales on original-teacher TRAIN reals first")
        count = real.shape[0]
        images = torch.cat((real.detach(), fake.detach()))
        if head_only:
            with torch.no_grad():
                maps = self._maps(self.student, images)
        else:
            maps = self._maps(self.student, images)
        all_labels = torch.cat((labels, labels)).long()
        logits = torch.stack([head(maps[name], all_labels) for name, head in self.heads.items()]).mean(dim=0)
        real_logits, fake_logits = logits[:count].float(), logits[count:].float()
        bce = 0.5 * (F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean())
        student_features = self._features({key: value[:count] for key, value in maps.items()})
        with torch.no_grad():
            teacher_features = self._features(self._maps(self.teacher, real))
        errors = torch.stack([
            (student_features[key].float() - teacher_features[key].float()).square().mean()
            / self.feature_scales[index].square()
            for index, key in enumerate(self.feature_keys)
        ])
        preservation = (errors * self.feature_weights).sum() / self.feature_weights.sum()
        loss = bce + float(preservation_weight) * preservation
        with torch.no_grad():
            ratios = torch.stack([
                student_features[key].float().square().mean().sqrt()
                / teacher_features[key].float().square().mean().sqrt().clamp_min(1.0e-8)
                for key in self.feature_keys
            ])
        output = {
            "loss": loss, "bce": bce.detach(), "preservation": preservation.detach(),
            "real_logit": real_logits.detach().mean(), "fake_logit": fake_logits.detach().mean(),
            "balanced_accuracy": 0.5 * ((real_logits.detach() > 0).float().mean() + (fake_logits.detach() < 0).float().mean()),
            "real_rms_ratio": (ratios * self.feature_weights).sum() / self.feature_weights.sum(),
        }
        if return_scores:
            output["real_scores"] = real_logits.detach()
            output["fake_scores"] = fake_logits.detach()
        return output

    def tuning_state_dict(self):
        return {
            "schema_version": 1, "trainable_blocks": list(TRAINABLE_BLOCKS),
            "feature_keys": list(self.feature_keys), "student": self.student.state_dict(),
            "heads": self.heads.state_dict(), "feature_scales": self.feature_scales.detach().clone(),
            "feature_weights": self.feature_weights.detach().clone(),
            "scales_calibrated": bool(self.scales_calibrated),
        }

    def load_tuning_state_dict(self, state: Mapping):
        if state.get("schema_version") != 1 or state.get("trainable_blocks") != list(TRAINABLE_BLOCKS) or state.get("feature_keys") != list(self.feature_keys):
            raise ValueError("Incompatible DINO tuning state")
        self.student.load_state_dict(state["student"], strict=True)
        self.heads.load_state_dict(state["heads"], strict=True)
        self.feature_scales.copy_(state["feature_scales"])
        self.feature_weights.copy_(state["feature_weights"])
        self.scales_calibrated.fill_(bool(state["scales_calibrated"]))
        self._configure_trainable()

    def export_backbone(self, path):
        """Flat keys compatible with SSLResNetFeatureExtractor's strict loader."""
        atomic_torch_save({key: value.detach().cpu() for key, value in self.student.state_dict().items()}, path)

    @torch.no_grad()
    def frozen_state_report(self):
        allowed = {name for name, parameter in self.student.named_parameters() if parameter.requires_grad}
        original = self.teacher.state_dict()
        changed = [name for name, value in self.student.state_dict().items() if not torch.equal(value, original[name])]
        frozen_changed = sorted(set(changed).difference(allowed))
        bn_keys = set()
        for name, module in self.student.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                bn_keys.update(name + "." + key for key in module.state_dict())
        return {
            "trainable_parameter_names": sorted(allowed),
            "trainable_parameter_count": sum(parameter.numel() for parameter in self.student.parameters() if parameter.requires_grad),
            "changed_trainable_keys": sorted(set(changed).intersection(allowed)),
            "frozen_tensor_count": len(original) - len(allowed),
            "changed_frozen_keys": frozen_changed,
            "all_frozen_tensors_unchanged": not frozen_changed,
            "bn_parameters_and_buffers_unchanged": not bool(set(changed).intersection(bn_keys)),
        }
