#!/usr/bin/env python3
"""Capture DINO kernel statistics and verify an existing calibrated profile.

The training loss normalizes each feature's L2 distances by its weighted mean
before applying temperature. Capture reports those normalized distances and
mutual-affinity ESS from a fresh generator's first batch; it does not fit or
approve new production temperatures. Existing DINO profiles retain their pinned
historical calibration artifact, verified by the same guard as training.

Both commands require an already calibrated direct-RGB DINO config. Outputs
must be outside this checkout. No encoder other than DINO is instantiated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_imagenet_gen as train_module  # noqa: E402
from drifting_core.imagenet_loss import _cdist_batched, _ratio_of_means  # noqa: E402


_STAGES = ("stage3", "stage4")


class _CalibrationComplete(RuntimeError):
    pass


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _feature_checkpoint_provenance(cfg: Dict[str, Any]) -> Dict[str, Any]:
    extractor = train_module.resolve_feature_extractor_name(cfg)
    checkpoint_value = cfg.get("feature_checkpoint")
    if not checkpoint_value:
        raise ValueError(f"{extractor} calibration requires a feature checkpoint")
    checkpoint = Path(str(checkpoint_value)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "feature_extractor": extractor,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": _sha256_file(checkpoint),
    }


def _raw_dataset_provenance(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not bool(cfg.get("require_complete_raw_imagenet", False)):
        return None
    root = Path(str(cfg.get("imagenet_path", ""))).resolve()
    manifest_path = root / "raw_imagenet_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Raw ImageNet calibration manifest is missing: {manifest_path}"
        )
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("complete") is not True:
        raise RuntimeError(f"Raw ImageNet manifest is incomplete: {manifest_path}")
    return {
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "mapping_sha256": manifest.get("mapping", {}).get(
            "official_mapping_sha256"
        ),
        "expected": manifest.get("expected"),
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor metadata and exact bytes, including bfloat16 tensors."""
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _ess(values: torch.Tensor, dim: int) -> torch.Tensor:
    mass = values.sum(dim=dim)
    square_mass = values.square().sum(dim=dim)
    return mass.square() / square_mass.clamp_min(1e-20)


def _quantiles(values: torch.Tensor) -> Dict[str, float]:
    values = values.float().reshape(-1)
    qs = torch.tensor((0.1, 0.25, 0.5, 0.75, 0.9), device=values.device)
    result = torch.quantile(values, qs).cpu().tolist()
    return {name: float(value) for name, value in zip(("p10", "p25", "p50", "p75", "p90"), result)}


def _mutual_ess(
    normalized_with_self_mask: torch.Tensor,
    *,
    split_idx: int,
    temperature: float,
) -> Dict[str, float]:
    logits = -normalized_with_self_mask / float(temperature)
    affinity = (
        F.softmax(logits, dim=2) * F.softmax(logits, dim=1)
    ).clamp_min(1e-6).sqrt()
    repulsive = affinity[:, :, :split_idx]
    positive = affinity[:, :, split_idx:]
    return {
        "pos_row": float(_ess(positive, dim=2).mean().item()),
        "pos_col": float(_ess(positive, dim=1).mean().item()),
        "repulsive_row": float(_ess(repulsive, dim=2).mean().item()),
        "repulsive_col": float(_ess(repulsive, dim=1).mean().item()),
    }


@torch.no_grad()
def _capture_curves(
    *,
    gen_feats: Dict[str, torch.Tensor],
    pos_feats: Dict[str, torch.Tensor],
    neg_feats: Optional[Dict[str, torch.Tensor]],
    B: int,
    G: int,
    P: int,
    N: int,
    weight_neg: Optional[torch.Tensor],
    r_values: Iterable[float],
    multipliers: Iterable[float],
    max_token_rows: int,
) -> Dict[str, Any]:
    device = next(iter(gen_feats.values())).device
    r_values = tuple(float(value) for value in r_values)
    multipliers = tuple(float(value) for value in multipliers)
    features: Dict[str, Any] = {}

    for name, gen_feature in gen_feats.items():
        stage = train_module._feature_loss_group(name)
        if stage not in _STAGES or name not in pos_feats:
            continue
        pos_feature = pos_feats[name]
        token_count = int(gen_feature.shape[1])
        feature_dim = int(gen_feature.shape[2])
        gen_bt = rearrange(
            gen_feature.detach().float(),
            "(b g) t d -> (b t) g d",
            b=B,
            g=G,
        )
        pos_bt = rearrange(
            pos_feature.detach().float(),
            "(b p) t d -> (b t) p d",
            b=B,
            p=P,
        )
        if neg_feats is not None and name in neg_feats and N > 0:
            neg_bt = rearrange(
                neg_feats[name].detach().float(),
                "(b n) t d -> (b t) n d",
                b=B,
                n=N,
            )
        else:
            neg_bt = gen_bt.new_zeros(gen_bt.shape[0], 0, feature_dim)

        targets = torch.cat((gen_bt, neg_bt, pos_bt), dim=1)
        gen_weights = gen_bt.new_ones(B, G)
        pos_weights = gen_bt.new_ones(B, P)
        neg_weights = (
            weight_neg.detach().float()
            if weight_neg is not None
            else gen_bt.new_ones(B, N)
        )
        target_weights = repeat(
            torch.cat((gen_weights, neg_weights, pos_weights), dim=1),
            "b m -> (b t) m",
            t=token_count,
        )
        distances = _cdist_batched(gen_bt, targets)
        scale = _ratio_of_means(
            distances * target_weights.unsqueeze(1),
            target_weights,
            use_global_stats=True,
        )
        normalized = distances / scale.clamp_min(1e-3)

        self_mask = F.pad(
            torch.eye(G, device=device, dtype=torch.bool),
            (0, N + P),
        ).unsqueeze(0)
        split_idx = G + N

        repulsive_mask = (~self_mask[:, :, :split_idx]).expand(
            normalized.shape[0], -1, -1
        )
        generated_nonself_mask = (~self_mask[:, :, :G]).expand(
            normalized.shape[0], -1, -1
        )
        distribution = {
            "positive": _quantiles(normalized[:, :, split_idx:]),
            "generated_nonself": _quantiles(
                normalized[:, :, :G][generated_nonself_mask]
            ),
            "repulsive_nonself": _quantiles(
                normalized[:, :, :split_idx][repulsive_mask]
            ),
        }
        if N > 0:
            distribution["real_negative"] = _quantiles(
                normalized[:, :, G:split_idx]
            )

        if normalized.shape[0] > max_token_rows:
            row_indices = torch.linspace(
                0,
                normalized.shape[0] - 1,
                steps=max_token_rows,
                device=device,
            ).round().long()
            normalized = normalized.index_select(0, row_indices)
        normalized_with_self_mask = normalized + self_mask.float() * 100.0

        curves: Dict[str, Any] = {}
        for multiplier in multipliers:
            multiplier_key = f"{multiplier:.10g}"
            curves[multiplier_key] = {
                f"{r_value:.10g}": _mutual_ess(
                    normalized_with_self_mask,
                    split_idx=split_idx,
                    temperature=r_value * multiplier,
                )
                for r_value in r_values
            }
        features[name] = {
            "stage": stage,
            "tokens": token_count,
            "dimension": feature_dim,
            "raw_weighted_mean_distance": float(scale.item()),
            "normalized_distance_quantiles": distribution,
            "curves": curves,
        }
        del distances, normalized, targets, gen_bt, pos_bt, neg_bt

    if {value["stage"] for value in features.values()} != set(_STAGES):
        raise ValueError("DINO capture requires nonempty stage3 and stage4 statistics")
    return {
        "definition": (
            "actual mutual affinity sqrt(softmax_target(-d/(scale*R*m)) * "
            "softmax_query(-d/(scale*R*m))); ESS matched after feature-wise scale"
        ),
        "R_list": list(r_values),
        "multipliers": list(multipliers),
        "features": features,
    }


def _run_capture(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    cfg = _load_verified_dino_config(config_path)
    if (not math.isfinite(args.grid_min) or not math.isfinite(args.grid_max)
            or args.grid_min <= 0 or args.grid_max < args.grid_min
            or args.grid_steps < 1 or args.max_token_rows < 1):
        raise ValueError("Capture requires 0 < grid-min <= grid-max and positive counts")
    output_path = _external_output_path(args.output)
    workdir_path = _external_output_path(args.workdir)
    cfg.update(
        {
            "use_wandb": False,
            "console_log": True,
            "eval_at_start": False,
            "eval_per_step": 1_000_000_000,
            "eval_per_generated_epochs": 0.0,
            "save_per_step": 1_000_000_000,
            "save_per_generated_epochs": 0.0,
            "total_generated_epochs": 0.0,
            "train_max_step_exclusive": 1,
            "adversarial_mode": "none",
            "historical_gen_replay": False,
            "feature_use_remat": False,
            # Verify the unmodified production config first, then collect
            # statistics without a training update or evaluation.
            "temperature_calibration_status": "calibration_capture",
            # Populate a representative class bank instead of calibrating on
            # the first 128 cached samples (which yields almost all repeated
            # positives for P=64).
            "push_per_step": 8192,
            "loader_batch_size": 512,
        }
    )
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite an existing capture: {output_path}")
    if workdir_path.exists() and (
        not workdir_path.is_dir() or any(workdir_path.iterdir())
    ):
        raise RuntimeError(
            f"Temperature capture workdir must be new or empty: {workdir_path}"
        )
    # Preserve the exact input config independently of its mutable source path.
    config_snapshot_path = output_path.with_name(
        f"{output_path.stem}_config.yaml"
    )
    if config_snapshot_path.exists():
        if config_snapshot_path.read_bytes() != config_bytes:
            raise RuntimeError(
                f"Refusing to replace a different capture config snapshot: "
                f"{config_snapshot_path}"
            )
    else:
        _atomic_write_bytes(config_snapshot_path, config_bytes)
    dataset_provenance = _raw_dataset_provenance(cfg)
    feature_provenance = _feature_checkpoint_provenance(cfg)
    multiplier_values = torch.exp(
        torch.linspace(
            math.log(float(args.grid_min)),
            math.log(float(args.grid_max)),
            int(args.grid_steps),
            dtype=torch.float64,
        )
    ).tolist()
    multiplier_values.append(1.0)
    multiplier_values = sorted(set(round(float(value), 12) for value in multiplier_values))

    original_train_step = train_module.train_step

    def calibration_step(
        generator,
        feature_extractor,
        optimizer,
        labels,
        pos_samples,
        neg_samples,
        device,
        step,
        step_cfg,
        **unused,
    ):
        del optimizer, step, unused
        B = int(labels.shape[0])
        P = int(pos_samples.shape[1])
        N = int(neg_samples.shape[1])
        G = int(step_cfg.get("gen_per_label", 32))
        cfg_scales = train_module.sample_cfg(
            B,
            float(step_cfg.get("cfg_min", 1.0)),
            float(step_cfg.get("cfg_max", 4.0)),
            float(step_cfg.get("neg_cfg_pw", 3.0)),
            float(step_cfg.get("no_cfg_frac", 0.0)),
            device,
        )
        raw_generator = generator.module if hasattr(generator, "module") else generator
        amp_feature = train_module._feature_use_bf16(feature_extractor)
        amp_generator = train_module._gen_use_bf16(generator)
        activation_kwargs = step_cfg["activation_kwargs"]

        all_real = torch.cat(
            (
                pos_samples.reshape(B * P, *pos_samples.shape[2:]),
                neg_samples.reshape(B * N, *neg_samples.shape[2:]),
            ),
            dim=0,
        )
        with torch.no_grad(), train_module._amp_ctx(amp_feature):
            real_features = feature_extractor.get_activations(
                all_real.to(device),
                **activation_kwargs,
            )
        pos_features = {name: value[: B * P] for name, value in real_features.items()}
        neg_features = {name: value[B * P :] for name, value in real_features.items()}

        expanded_labels = repeat(labels, "b -> (b g)", g=G)
        expanded_cfg = repeat(cfg_scales, "b -> (b g)", g=G)
        with torch.no_grad(), train_module._amp_ctx(amp_generator):
            generated = raw_generator(
                expanded_labels,
                cfg_scale=expanded_cfg,
                train=True,
            )["samples"]
        with torch.no_grad(), train_module._amp_ctx(amp_feature):
            generated_features = feature_extractor.get_activations(
                generated,
                **activation_kwargs,
            )

        weight_neg = (
            (cfg_scales - 1.0).unsqueeze(1).expand(-1, N)
            * float(G - 1)
            / float(max(1, N))
        )
        payload = _capture_curves(
            gen_feats=generated_features,
            pos_feats=pos_features,
            neg_feats=neg_features,
            B=B,
            G=G,
            P=P,
            N=N,
            weight_neg=weight_neg,
            r_values=step_cfg.get("R_list", (0.2, 0.05, 0.02)),
            multipliers=multiplier_values,
            max_token_rows=int(args.max_token_rows),
        )
        payload.update(
            {
                "diagnostic_only": True,
                "production_temperatures_fitted": False,
                "profile_verification": "training_guard_passed_before_capture_overrides",
                "inherited_layer_temperature_multipliers": (
                    train_module._resolve_layer_temperature_multipliers(step_cfg)
                ),
                "capture_overrides": {
                    "adversarial_mode": "none", "historical_gen_replay": False,
                    "push_per_step": 8192, "loader_batch_size": 512,
                    "feature_use_remat": False,
                },
                "config": str(config_snapshot_path),
                "source_config": str(config_path),
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "dataset_provenance": dataset_provenance,
                "feature_provenance": feature_provenance,
                "requested_seed": args.seed,
                "effective_seed": int(step_cfg["seed"]),
                "feature_extractor": train_module.resolve_feature_extractor_name(step_cfg),
                "generator_geometry": {
                    "input_size": int(step_cfg["input_size"]),
                    "in_channels": int(step_cfg["in_channels"]),
                    "out_channels": int(step_cfg["out_channels"]),
                    "patch_size": int(step_cfg["patch_size"]),
                    "image_tokens": (
                        int(step_cfg["input_size"]) // int(step_cfg["patch_size"])
                    ) ** 2,
                    "class_tokens": int(step_cfg.get("n_cls_tokens", 0)),
                },
                "batch": {"B": B, "G": G, "P": P, "N": N},
                "cfg_scales": [float(value) for value in cfg_scales.cpu().tolist()],
                "tensor_sha256": {
                    "labels": _tensor_sha256(labels),
                    "positive_samples": _tensor_sha256(pos_samples),
                    "negative_samples": _tensor_sha256(neg_samples),
                    "generated_samples": _tensor_sha256(generated),
                    "cfg_scales": _tensor_sha256(cfg_scales),
                },
                "max_token_rows_per_feature": int(args.max_token_rows),
            }
        )
        _atomic_write_bytes(
            output_path, (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        )
        print(f"[feature-temperature] captured {payload['feature_extractor']} -> {output_path}", flush=True)
        raise _CalibrationComplete

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("temperature capture must run as a single process")
    train_module.train_step = calibration_step
    try:
        rank, world_size, device = train_module.setup_distributed()
        if world_size != 1:
            raise RuntimeError("temperature capture must run as a single process")
        workdir_path.mkdir(parents=True, exist_ok=True)
        train_module.train_gen(cfg, str(workdir_path), rank, world_size, device)
        raise RuntimeError("Training returned without completing the diagnostic capture")
    except _CalibrationComplete:
        pass
    finally:
        train_module.train_step = original_train_step
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_finalize_symmetric(args: argparse.Namespace) -> None:
    """Legacy JSON-only finalizer retained for historical provenance tests.

    This is not a CLI mode and does not run a reference encoder or refit taus.
    """
    forward_path = Path(args.forward).resolve()
    reverse_path = Path(args.reverse).resolve()
    forward = json.loads(forward_path.read_text(encoding="utf-8"))
    reverse = json.loads(reverse_path.read_text(encoding="utf-8"))
    if (
        int(forward.get("schema_version", 0)) != 2
        or forward.get("calibration_kind")
        != "symmetric_raw_imagenet_dino_moco"
        or set((forward.get("symmetric_profiles") or {})) != {"dino", "moco"}
    ):
        raise ValueError("Forward artifact is not a symmetric DINO/MoCo fit")
    if int(reverse.get("schema_version", 0)) != 2 or set(
        (reverse.get("candidates") or {})
    ) != {"dino"}:
        raise ValueError("Reverse artifact must contain exactly the DINO candidate")
    if forward.get("dataset_provenance") != reverse.get("dataset_provenance"):
        raise ValueError("Forward and reverse fits use different dataset provenance")
    if any(
        forward.get(key) != reverse.get(key)
        for key in (
            "groups_considered",
            "quantiles",
            "capture_protocol",
            "min_distance",
            "huber_tuning",
            "trim_fraction",
        )
    ):
        raise ValueError("Forward and reverse fit protocols differ")
    forward_dino_captures = forward.get("references") or {}
    forward_moco_captures = (
        (forward.get("candidates") or {}).get("moco") or {}
    ).get("captures") or {}
    reverse_moco_captures = reverse.get("references") or {}
    reverse_dino_captures = (
        (reverse.get("candidates") or {}).get("dino") or {}
    ).get("captures") or {}
    if (
        forward_dino_captures != reverse_dino_captures
        or forward_moco_captures != reverse_moco_captures
    ):
        raise ValueError(
            "Forward and reverse fits must use the same DINO/MoCo captures"
        )

    max_log_error = float(args.max_log_reciprocity_error)
    if not math.isfinite(max_log_error) or max_log_error < 0.0:
        raise ValueError("--max-log-reciprocity-error must be finite and >= 0")
    stage_validation: Dict[str, Any] = {}
    for stage in _STAGES:
        forward_ratio = float(
            forward["candidates"]["moco"]["stages"][stage][
                "selected_multiplier"
            ]
        )
        reverse_ratio = float(
            reverse["candidates"]["dino"]["stages"][stage][
                "selected_multiplier"
            ]
        )
        log_error = abs(math.log(forward_ratio * reverse_ratio))
        if log_error > max_log_error:
            raise RuntimeError(
                f"{stage} forward/reverse calibration is not reciprocal: "
                f"q_forward={forward_ratio} q_reverse={reverse_ratio} "
                f"abs_log_product={log_error} > {max_log_error}"
            )
        stage_validation[stage] = {
            "moco_over_dino": forward_ratio,
            "dino_over_moco": reverse_ratio,
            "product": forward_ratio * reverse_ratio,
            "abs_log_product": log_error,
        }

    final = json.loads(json.dumps(forward))
    final["schema_version"] = 3
    final["reciprocity_validation"] = {
        "verified": True,
        "reverse_artifact": str(reverse_path),
        "reverse_artifact_sha256": _sha256_file(reverse_path),
        "max_log_reciprocity_error": max_log_error,
        "stages": stage_validation,
    }
    output_path = Path(args.output).resolve()
    _atomic_write_bytes(
        output_path, (json.dumps(final, indent=2) + "\n").encode("utf-8")
    )
    print(f"[feature-distance-symmetric] finalized -> {output_path}", flush=True)


def _external_output_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Calibration outputs must be outside the code checkout")
    return path


def _load_verified_dino_config(config_path: Path) -> Dict[str, Any]:
    cfg = train_module.load_yaml_config(str(config_path))
    train_module._validate_dino_only_config(cfg)
    if not bool(cfg.get("require_raw_temperature_calibration", False)):
        raise ValueError("A pinned raw-ImageNet DINO temperature profile is required")
    if cfg.get("temperature_calibration_status") != "ready":
        raise ValueError("An already calibrated ready DINO config is required")
    train_module._validate_raw_temperature_calibration(cfg)
    return cfg


def _run_verify_artifact(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    output_path = _external_output_path(args.output) if args.output else None
    if output_path is not None and output_path.exists():
        raise RuntimeError(f"Refusing to overwrite verification: {output_path}")
    cfg = _load_verified_dino_config(config_path)
    payload = {
        "verified": True,
        "verification": "unchanged_production_training_guard",
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "feature_extractor": train_module.resolve_feature_extractor_name(cfg),
        "feature_checkpoint": cfg["feature_checkpoint"],
        "calibration_artifact": cfg["temperature_calibration_artifact"],
        "calibration_artifact_sha256": cfg["temperature_calibration_artifact_sha256"],
        "layer_temperature_profile": cfg["layer_temperature_profile"],
        "layer_temperature_multipliers": (
            train_module._resolve_layer_temperature_multipliers(cfg)
        ),
        "temperature_calibration_inheritance_manifest": cfg.get(
            "temperature_calibration_inheritance_manifest"
        ),
        "temperature_calibration_inheritance_manifest_sha256": cfg.get(
            "temperature_calibration_inheritance_manifest_sha256"
        ),
        "production_temperatures_fitted": False,
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    if output_path is not None:
        _atomic_write_bytes(output_path, encoded.encode("utf-8"))
    print(encoded, end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser(
        "verify-artifact", help="Verify and reuse the pinned DINO profile without GPU work"
    )
    verify.add_argument("--config", required=True)
    verify.add_argument("--output", help="Optional new JSON report outside the checkout")
    verify.set_defaults(func=_run_verify_artifact)

    capture = subparsers.add_parser(
        "capture", help="Capture DINO-only diagnostic statistics; no new production fit"
    )
    capture.add_argument("--config", required=True)
    capture.add_argument("--workdir", required=True, help="New directory outside the checkout")
    capture.add_argument("--output", required=True, help="New JSON outside the checkout")
    capture.add_argument("--max-token-rows", type=int, default=64)
    capture.add_argument("--seed", type=int, default=None)
    capture.add_argument("--grid-min", type=float, default=0.25)
    capture.add_argument("--grid-max", type=float, default=4.0)
    capture.add_argument("--grid-steps", type=int, default=81)
    capture.set_defaults(func=_run_capture)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
