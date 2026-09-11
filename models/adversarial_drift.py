"""Raw-image adversarial supervision and frozen learned drift features.

All modes train the same conditional discriminator with logistic loss.  The
``raw_gan`` generator consumes its frozen target logits; ``feature_drift``
consumes only its frozen target features; ``mixed`` consumes both from one
shared target forward. Generator loss composition belongs
to the trainer.  No normalization layer or spectral-normalization buffer can
change the target while real, generated, negative, and historical samples pass
through it.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


FEATURE_STAGES = ("stage2", "stage3", "stage4")


def logistic_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Mean real and fake logistic losses; callers detach fake *images*."""
    return F.softplus(-real_logits.float()).mean() + F.softplus(fake_logits.float()).mean()


def logistic_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss, retaining the gradient into its images."""
    return F.softplus(-fake_logits.float()).mean()


def _spatial_tokens(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        return F.adaptive_avg_pool2d(features.float(), (4, 4)).flatten(2).transpose(1, 2)
    if features.ndim == 3:
        if features.shape[1] == 16:
            return features.float()
        side = math.isqrt(features.shape[1])
        if side * side != features.shape[1]:
            raise ValueError("Spatial structure tokens must form a square grid")
        maps = features.float().transpose(1, 2).reshape(features.shape[0], features.shape[2], side, side)
        return F.adaptive_avg_pool2d(maps, (4, 4)).flatten(2).transpose(1, 2)
    if features.ndim == 2:
        return features.float().unsqueeze(1).expand(-1, 16, -1)
    raise ValueError(f"Expected NC, NTC, or NCHW features, got {tuple(features.shape)}")


def _stage_feature(features: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
    if stage in features:
        return features[stage]
    alias = "layer" + stage[-1]
    if alias in features:
        return features[alias]
    raise KeyError(f"Missing teacher/student feature {stage} (or {alias})")


def within_class_structure_loss(
    student_features: Mapping[str, torch.Tensor],
    teacher_features: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Match absolute within-class real/real distances at all drift scales.

    The teacher is always detached.  Both branches use the same 4x4 spatial
    positions and token normalization as the generator drift coordinates.
    Student stages 2/3 match frozen DINO stage 3; student stage 4 matches DINO
    stage 4.  Each loss averages over unordered, distinct-image, same-class
    pairs and spatial positions, then the three scales receive equal weight.
    Unlike a correlation loss, affine changes of cosine distances incur a
    penalty.  Absolute feature magnitude is intentionally removed by the
    token normalization also used in the generator drift.
    """
    if labels.ndim != 1:
        raise ValueError("Structure labels must be one dimensional")
    mask = torch.triu(labels[:, None].eq(labels[None, :]), diagonal=1)
    losses = []
    for stage in FEATURE_STAGES:
        teacher_stage = "stage3" if stage in {"stage2", "stage3"} else "stage4"
        student = _spatial_tokens(_stage_feature(student_features, stage))
        teacher = _spatial_tokens(_stage_feature(teacher_features, teacher_stage).detach())
        if student.shape[0] != labels.numel() or teacher.shape[0] != labels.numel():
            raise ValueError("Structure features and labels must describe the same real images")
        teacher = teacher.to(device=student.device, dtype=torch.float32)
        with torch.autocast(device_type=student.device.type, enabled=False):
            student = F.normalize(student.float(), dim=-1, eps=1e-8)
            teacher = F.normalize(teacher, dim=-1, eps=1e-8)
            student = student.transpose(0, 1)
            teacher = teacher.transpose(0, 1)
            student_distance = 1.0 - student @ student.transpose(1, 2)
            teacher_distance = 1.0 - teacher @ teacher.transpose(1, 2)
            selected = (student_distance - teacher_distance).square()[:, mask.to(student.device)]
            # A zero connected to the online graph also supports tiny smoke
            # batches with no repeated labels, without an empty-mean NaN.
            losses.append(selected.mean() if selected.numel() else student.sum() * 0.0)
    return torch.stack(losses).mean()


class _ProjectionHead(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.unconditional = nn.Linear(channels, 1)
        self.embedding = nn.Embedding(num_classes, channels)
        self.scale = math.sqrt(channels)

    def forward(self, pooled: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.unconditional(pooled).squeeze(1) + (
            self.embedding(labels) * pooled
        ).sum(dim=1) / self.scale


class ConditionalMultiScaleDiscriminator(nn.Module):
    """Small projection discriminator on raw RGB in training space [-1, 1].

    A full-resolution 3x3 stem precedes 2x average pooling.  Four stride-two
    blocks then produce 64/32/16/8 grids for 256-pixel inputs.  The latter three
    scales each have a conditional projection head; their logits are averaged
    before logistic loss.  Exposed 4x4 tokens have unit RMS across channels.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, base_channels: int = 32) -> None:
        super().__init__()
        if min(in_channels, num_classes, base_channels) <= 0:
            raise ValueError("Discriminator dimensions must be positive")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        stem_channels = max(4, self.base_channels // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, stem_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        widths = [self.base_channels * (2 ** index) for index in range(4)]
        blocks = []
        previous = stem_channels
        for width in widths:
            blocks.append(nn.Sequential(
                nn.Conv2d(previous, width, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(width, width, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
            ))
            previous = width
        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleDict({
            stage: _ProjectionHead(widths[index + 1], self.num_classes)
            for index, stage in enumerate(FEATURE_STAGES)
        })

    def _maps(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError(f"Expected NCHW RGB images, got {tuple(images.shape)}")
        if min(images.shape[-2:]) < 32:
            raise ValueError("Discriminator inputs must be at least 32x32")
        hidden = F.avg_pool2d(self.stem(images.float()), kernel_size=2)
        maps = {}
        for index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden)
            if index >= 2:
                maps[f"stage{index}"] = hidden
        return maps

    @staticmethod
    def _feature_tokens(maps: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result = {}
        for stage, feature in maps.items():
            tokens = F.adaptive_avg_pool2d(feature.float(), (4, 4)).flatten(2).transpose(1, 2)
            result[stage] = F.normalize(tokens, dim=-1, eps=1e-8) * math.sqrt(tokens.shape[-1])
        return result

    def features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._feature_tokens(self._maps(images))

    def forward(self, images: torch.Tensor, labels: torch.Tensor, return_features: bool = False):
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError("Discriminator needs one integer class label per image")
        maps = self._maps(images)
        logits = torch.stack([
            self.heads[stage](feature.mean(dim=(2, 3)), labels.long())
            for stage, feature in maps.items()
        ]).mean(dim=0)
        if return_features:
            return logits, self._feature_tokens(maps)
        return logits


class AdversarialDriftSystem:
    """Online discriminator, optimizer, and immutable-per-G-step target.

    Call ``discriminator_step`` only after the generator backward/update.  It
    detaches both image branches, averages discriminator gradients across
    ranks, rejects non-finite updates collectively, and then updates the target.
    Manual gradient synchronization avoids DDP reducer interactions with lazy
    second-order R1 and an optional structure-loss branch.
    """
    def __init__(
        self, device, mode: str = "raw_gan", in_channels: int = 3,
        num_classes: int = 1000, base_channels: int = 32, lr: float = 1e-4,
        adam_b1: float = 0.0, adam_b2: float = 0.99, ema_decay: Optional[float] = None,
        r1_gamma: float = 1.0, r1_interval: int = 16, structure_weight: float = 1.0,
        d_chunk_size: int = 8, g_chunk_size: int = 16, seed: int = 43,
        max_grad_norm: float = 0.0, fuse_grad_reduce: bool = False,
    ) -> None:
        if mode not in {"raw_gan", "feature_drift", "mixed"}:
            raise ValueError(f"Unknown adversarial mode {mode!r}")
        ema_decay = (0.0 if mode == "raw_gan" else 0.99) if ema_decay is None else float(ema_decay)
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1)")
        if min(int(r1_interval), int(d_chunk_size), int(g_chunk_size)) <= 0:
            raise ValueError("R1 interval and chunk sizes must be positive")
        for name, value in {"lr": lr, "r1_gamma": r1_gamma, "structure_weight": structure_weight, "max_grad_norm": max_grad_norm}.items():
            if not math.isfinite(float(value)) or float(value) < 0 or (name == "lr" and value == 0):
                raise ValueError(f"Invalid {name}: {value}")
        if not (0 <= adam_b1 < 1 and 0 <= adam_b2 < 1):
            raise ValueError("Adam betas must lie in [0, 1)")
        self.mode = mode
        self.ema_decay = ema_decay
        self.r1_gamma = float(r1_gamma)
        self.r1_interval = int(r1_interval)
        self.structure_weight = float(structure_weight)
        self.d_chunk_size = int(d_chunk_size)
        self.g_chunk_size = int(g_chunk_size)
        self.max_grad_norm = float(max_grad_norm)
        self.fuse_grad_reduce = bool(fuse_grad_reduce)
        self.updates = 0
        self._settings = {
            "mode": mode, "in_channels": int(in_channels), "num_classes": int(num_classes),
            "base_channels": int(base_channels), "lr": float(lr), "adam_b1": float(adam_b1),
            "adam_b2": float(adam_b2), "ema_decay": ema_decay, "r1_gamma": self.r1_gamma,
            "r1_interval": self.r1_interval, "structure_weight": self.structure_weight,
            "d_chunk_size": self.d_chunk_size, "g_chunk_size": self.g_chunk_size,
            "seed": int(seed), "max_grad_norm": self.max_grad_norm,
        }
        self._config_fields = {}
        # Preserve settings compatibility for checkpoints from the original
        # modes, whose per-parameter synchronization remains the default.
        if self.fuse_grad_reduce:
            self._settings["fuse_grad_reduce"] = True
        # Seed only the CPU generator, and restore it after CPU construction.
        # Unlike torch.manual_seed(), this never changes any CUDA RNG state.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(int(seed))
            self.online = ConditionalMultiScaleDiscriminator(in_channels, num_classes, base_channels)
        self.online = self.online.to(device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            for parameter in self.online.parameters():
                dist.broadcast(parameter.data, src=0)
        self.target = copy.deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=float(lr), betas=(float(adam_b1), float(adam_b2)),
        )

    @classmethod
    def from_config(cls, cfg: Mapping, device):
        if not bool(cfg.get("adversarial_mode", "")):
            return None
        arguments = {
            name: cfg.get("adversarial_" + name, default)
            for name, default in {
                "mode": "raw_gan", "base_channels": 32, "lr": 1e-4,
                "adam_b1": 0.0, "adam_b2": 0.99, "ema_decay": None,
                "r1_gamma": 1.0, "r1_interval": 16, "structure_weight": 1.0,
                "d_chunk_size": 8, "g_chunk_size": 16, "seed": 43, "max_grad_norm": 0.0,
                "fuse_grad_reduce": False,
            }.items()
        }
        system = cls(device, num_classes=int(cfg.get("num_classes", 1000)), in_channels=int(cfg.get("in_channels", 3)), **arguments)
        # Include trainer-owned weights/sampling policy in resume validation.
        system._config_fields = copy.deepcopy({key: value for key, value in cfg.items() if key.startswith("adversarial_")})
        return system

    @property
    def device(self):
        return next(self.online.parameters()).device

    def target_logits(self, images: torch.Tensor, labels: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
        self.target.eval()
        chunk_size = self.g_chunk_size if chunk_size is None else int(chunk_size)
        if chunk_size <= 0 or images.shape[0] == 0:
            raise ValueError("Target inference needs a positive chunk size and nonempty images")
        with torch.autocast(device_type=images.device.type, enabled=False):
            return torch.cat([
                self.target(images[start:start + chunk_size].float(), labels[start:start + chunk_size])
                for start in range(0, images.shape[0], chunk_size)
            ])

    def target_features(self, images: torch.Tensor, labels=None, chunk_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        del labels  # The projection head conditions logits, not the feature maps.
        self.target.eval()
        chunk_size = self.g_chunk_size if chunk_size is None else int(chunk_size)
        if chunk_size <= 0 or images.shape[0] == 0:
            raise ValueError("Target inference needs a positive chunk size and nonempty images")
        parts = {stage: [] for stage in FEATURE_STAGES}
        with torch.autocast(device_type=images.device.type, enabled=False):
            for start in range(0, images.shape[0], chunk_size):
                features = self.target.features(images[start:start + chunk_size].float())
                for stage, values in features.items():
                    parts[stage].append(values)
        return {stage: torch.cat(values) for stage, values in parts.items()}

    def target_logits_and_features(
        self, images: torch.Tensor, labels: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Share the expensive image encoding between both mixed G losses.

        The target is frozen, but this method retains the image gradient for
        both outputs. The same immutable snapshot also encodes drift particles.
        """
        self.target.eval()
        chunk_size = self.g_chunk_size if chunk_size is None else int(chunk_size)
        if chunk_size <= 0 or images.shape[0] == 0:
            raise ValueError("Target inference needs a positive chunk size and nonempty images")
        logits_parts = []
        feature_parts = {stage: [] for stage in FEATURE_STAGES}
        with torch.autocast(device_type=images.device.type, enabled=False):
            for start in range(0, images.shape[0], chunk_size):
                logits, features = self.target(
                    images[start:start + chunk_size].float(),
                    labels[start:start + chunk_size], return_features=True,
                )
                logits_parts.append(logits)
                for stage, values in features.items():
                    feature_parts[stage].append(values)
        return torch.cat(logits_parts), {
            stage: torch.cat(values) for stage, values in feature_parts.items()
        }

    def _all_finite(self, local_finite: torch.Tensor) -> bool:
        flag = local_finite.detach().to(device=self.device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    @torch.no_grad()
    def update_target(self) -> None:
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            if self.ema_decay == 0.0:
                target.copy_(online)
            else:
                target.lerp_(online, 1.0 - self.ema_decay)
        for target, online in zip(self.target.buffers(), self.online.buffers()):
            target.copy_(online)
        self.target.eval().requires_grad_(False)

    def discriminator_step(
        self, real_images: torch.Tensor, fake_images: torch.Tensor,
        real_labels: torch.Tensor, fake_labels: torch.Tensor, step: int,
        teacher_real_features: Optional[Mapping[str, torch.Tensor]] = None,
        chunk_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor | float]:
        if min(real_images.shape[0], fake_images.shape[0]) <= 0:
            raise ValueError("Discriminator requires nonempty real and fake batches")
        if real_labels.shape != (real_images.shape[0],) or fake_labels.shape != (fake_images.shape[0],):
            raise ValueError("Discriminator image counts must match their labels")
        if self.mode in {"feature_drift", "mixed"} and self.structure_weight > 0 and teacher_real_features is None:
            raise ValueError("Feature-drift structure preservation requires detached real DINO features")
        chunk_size = self.d_chunk_size if chunk_size is None else int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("Discriminator chunk size must be positive")
        self.online.train()
        self.optimizer.zero_grad(set_to_none=True)
        r1_due = self.r1_gamma > 0 and int(step) % self.r1_interval == 0
        real_logits_parts, fake_logits_parts, r1_parts = [], [], []
        feature_parts = {stage: [] for stage in FEATURE_STAGES}
        with torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, real_images.shape[0], chunk_size):
                real = real_images[start:start + chunk_size].detach().float().requires_grad_(r1_due)
                logits, features = self.online(real, real_labels[start:start + chunk_size], return_features=True)
                real_logits_parts.append(logits)
                if self.mode in {"feature_drift", "mixed"} and self.structure_weight > 0:
                    for stage, values in features.items():
                        feature_parts[stage].append(values)
                if r1_due:
                    gradient = torch.autograd.grad(logits.sum(), real, create_graph=True, retain_graph=True)[0]
                    r1_parts.append(gradient.square().flatten(1).sum(dim=1))
            for start in range(0, fake_images.shape[0], chunk_size):
                fake_logits_parts.append(self.online(
                    fake_images[start:start + chunk_size].detach().float(),
                    fake_labels[start:start + chunk_size],
                ))
            real_logits = torch.cat(real_logits_parts)
            fake_logits = torch.cat(fake_logits_parts)
            logistic = logistic_discriminator_loss(real_logits, fake_logits)
            r1 = torch.cat(r1_parts).mean() if r1_due else logistic.new_zeros(())
            r1_weighted = r1 * (self.r1_gamma * self.r1_interval / 2.0)
            structure = logistic.new_zeros(())
            if self.mode in {"feature_drift", "mixed"} and self.structure_weight > 0:
                structure = within_class_structure_loss(
                    {stage: torch.cat(values) for stage, values in feature_parts.items()},
                    teacher_real_features, real_labels,
                )
            loss = logistic + r1_weighted + self.structure_weight * structure
        def metric_value(value):
            # The trainer packs tensor metrics for a single logging transfer.
            # Mixed runs avoid premature CUDA host synchronization per scalar.
            return value.detach() if self.mode == "mixed" else float(value.detach())

        metrics = {
            "adversarial/d_loss": metric_value(loss),
            "adversarial/d_logistic_loss": metric_value(logistic),
            "adversarial/d_real_logit": metric_value(real_logits.detach().mean()),
            "adversarial/d_fake_logit": metric_value(fake_logits.detach().mean()),
            "adversarial/r1": metric_value(r1),
            "adversarial/r1_weighted": metric_value(r1_weighted),
            "adversarial/r1_applied": float(r1_due),
            "adversarial/structure_loss": metric_value(structure),
            "adversarial/d_updated": 0.0,
        }
        loss_finite = self._all_finite(torch.isfinite(loss))
        metrics["adversarial/d_loss_finite"] = float(loss_finite)
        if not loss_finite:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite adversarial discriminator loss on at least one rank")
        loss.backward()
        parameters = list(self.online.parameters())
        finite = torch.stack([
            torch.isfinite(parameter.grad).all()
            for parameter in parameters if parameter.grad is not None
        ]).all()
        grad_finite = self._all_finite(finite)
        metrics["adversarial/d_grad_finite"] = float(grad_finite)
        if not grad_finite:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite adversarial discriminator gradients on at least one rank")
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            for parameter in parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
            if self.fuse_grad_reduce:
                # All discriminator parameters are dense fp32 on one device.
                # One collective avoids launching one NCCL operation per leaf.
                packed = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
                packed.div_(world_size)
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                offset = 0
                for parameter in parameters:
                    size = parameter.numel()
                    parameter.grad.copy_(packed[offset:offset + size].view_as(parameter))
                    offset += size
                del packed
            else:
                for parameter in parameters:
                    parameter.grad.div_(world_size)
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.max_grad_norm if self.max_grad_norm > 0 else float("inf"),
        )
        if not self._all_finite(torch.isfinite(grad_norm)):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite synchronized discriminator gradient norm on at least one rank")
        metrics["adversarial/d_grad_norm"] = metric_value(grad_norm)
        self.optimizer.step()
        self.update_target()
        self.updates += 1
        metrics["adversarial/d_updated"] = 1.0
        metrics["adversarial/d_updates"] = float(self.updates)
        return metrics

    def state_dict(self) -> dict:
        return {
            "version": 1, "settings": copy.deepcopy(self._settings),
            "config": copy.deepcopy(self._config_fields), "updates": self.updates,
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        if state.get("version") != 1:
            raise ValueError("Unsupported adversarial checkpoint version")
        if state.get("settings") != self._settings or state.get("config", {}) != self._config_fields:
            raise ValueError("Adversarial checkpoint architecture, mode, or training configuration changed")
        required = {"online", "target", "optimizer", "updates"}
        if not required.issubset(state):
            raise ValueError(f"Incomplete adversarial checkpoint: missing {required - set(state)}")
        self.online.load_state_dict(state["online"], strict=True)
        self.target.load_state_dict(state["target"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])
        self.target.eval().requires_grad_(False)
