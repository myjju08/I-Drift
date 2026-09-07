#!/usr/bin/env python3
"""Match DINO/MoCo reverse-kernel temperatures to an MAE reference.

The training loss first divides every feature's L2 distances by that feature's
weighted mean distance.  Consequently, raw activation norms are not a valid
temperature calibration signal.  This tool captures the *normalized* distance
grid from an otherwise-real first S4 training batch and fits stage-wise
temperature multipliers by matching the actual exponential mutual-affinity
ESS at every configured R.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
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
_ESS_METRICS = ("pos_row", "pos_col", "repulsive_row", "repulsive_col")


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
    checkpoint_value = cfg.get("feature_checkpoint") or cfg.get("mae_checkpoint")
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
    cfg = train_module.load_yaml_config(str(config_path))
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
            "feature_adapter": False,
            "feature_gan": False,
            "feature_use_remat": False,
            "mae_use_remat": False,
            # A production raw-pixel config remains launch-locked until the
            # resulting multi-seed fit is applied. Captures themselves must be
            # able to execute with the pending unit profile.
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
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite an existing capture: {output_path}")
    workdir_path = Path(args.workdir).resolve()
    if workdir_path.exists() and (
        not workdir_path.is_dir() or any(workdir_path.iterdir())
    ):
        raise RuntimeError(
            f"Temperature capture workdir must be new or empty: {workdir_path}"
        )
    # Preserve the exact pre-tau input config beside every capture. Production
    # YAMLs are intentionally updated after the fit, so their live path is not
    # a stable reproducibility record.
    config_snapshot_path = output_path.with_name(
        f"{output_path.stem}_config_pre_tau.yaml"
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
        amp_feature = train_module._mae_use_bf16(feature_extractor)
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

    train_module.train_step = calibration_step
    rank, world_size, device = train_module.setup_distributed()
    if world_size != 1:
        raise RuntimeError("temperature capture must run as a single process")
    workdir_path.mkdir(parents=True, exist_ok=True)
    try:
        train_module.train_gen(cfg, args.workdir, rank, world_size, device)
    except _CalibrationComplete:
        pass
    finally:
        train_module.train_step = original_train_step
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _matched_error(
    reference: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    stage: str,
    multiplier_key: str,
) -> tuple[float, int]:
    differences = []
    reference_features = reference["features"]
    candidate_features = candidate["features"]
    for name in sorted(set(reference_features) & set(candidate_features)):
        ref_feature = reference_features[name]
        candidate_feature = candidate_features[name]
        if ref_feature["stage"] != stage or candidate_feature["stage"] != stage:
            continue
        ref_curves = ref_feature["curves"]
        ref_one_key = min(ref_curves, key=lambda value: abs(float(value) - 1.0))
        for r_key, ref_metrics in ref_curves[ref_one_key].items():
            candidate_metrics = candidate_feature["curves"][multiplier_key][r_key]
            for metric_name in _ESS_METRICS:
                ref_value = max(float(ref_metrics[metric_name]), 1e-12)
                candidate_value = max(float(candidate_metrics[metric_name]), 1e-12)
                differences.append(math.log(candidate_value / ref_value))
    if not differences:
        raise ValueError(f"No matched {stage} feature/R/ESS statistics")
    rmse = math.sqrt(sum(value * value for value in differences) / len(differences))
    return rmse, len(differences)


def _run_fit(args: argparse.Namespace) -> None:
    reference = json.loads(Path(args.reference).read_text())
    fitted: Dict[str, Any] = {}
    for raw_candidate in args.candidate:
        if "=" not in raw_candidate:
            raise ValueError("--candidate must use NAME=JSON_PATH")
        name, raw_path = raw_candidate.split("=", 1)
        candidate = json.loads(Path(raw_path).read_text())
        stage_results: Dict[str, Any] = {}
        for stage in _STAGES:
            choices = []
            first_feature = next(iter(candidate["features"].values()))
            for multiplier_key in first_feature["curves"]:
                error, terms = _matched_error(
                    reference,
                    candidate,
                    stage=stage,
                    multiplier_key=multiplier_key,
                )
                choices.append(
                    {
                        "multiplier": float(multiplier_key),
                        "log_ess_rmse": error,
                        "matched_terms": terms,
                    }
                )
            best = min(choices, key=lambda item: item["log_ess_rmse"])
            stage_results[stage] = {"selected": best, "grid": choices}
            print(
                f"[feature-temperature] {name} {stage} multiplier="
                f"{best['multiplier']:.8g} log_ess_rmse={best['log_ess_rmse']:.6f}",
                flush=True,
            )
        fitted[name] = {
            "capture": str(Path(raw_path).resolve()),
            "stages": stage_results,
        }
    payload = {
        "reference": str(Path(args.reference).resolve()),
        "objective": "matched feature-key/R positive+repulsive row+column mutual-affinity log-ESS RMSE",
        "candidates": fitted,
    }
    output_path = Path(args.output).resolve()
    _atomic_write_bytes(
        output_path, (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    )
    print(f"[feature-temperature] fit -> {output_path}", flush=True)


def _distance_log_ratios(
    reference: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    stage: str,
    min_distance: float,
) -> list[float]:
    """Return log ratios of the actual normalized kernel-distance quantiles."""
    ratios: list[float] = []
    reference_features = reference["features"]
    candidate_features = candidate["features"]
    for name in sorted(set(reference_features) & set(candidate_features)):
        ref_feature = reference_features[name]
        candidate_feature = candidate_features[name]
        if ref_feature["stage"] != stage or candidate_feature["stage"] != stage:
            continue
        ref_distributions = ref_feature["normalized_distance_quantiles"]
        candidate_distributions = candidate_feature["normalized_distance_quantiles"]
        for group in ("positive", "repulsive_nonself"):
            for quantile in ("p25", "p50", "p75", "p90"):
                reference_value = float(ref_distributions[group][quantile])
                candidate_value = float(candidate_distributions[group][quantile])
                # Very small generated-to-generated distances are numerical
                # near-ties, not a useful scale signal for d/tau.
                if reference_value <= min_distance or candidate_value <= min_distance:
                    continue
                ratios.append(math.log(candidate_value / reference_value))
    if not ratios:
        raise ValueError(f"No matched normalized-distance statistics for {stage}")
    return ratios


def _run_fit_distance(args: argparse.Namespace) -> None:
    references = [json.loads(Path(path).read_text()) for path in args.reference]
    grouped_candidates: Dict[str, list[tuple[str, Dict[str, Any]]]] = {}
    for raw_candidate in args.candidate:
        if "=" not in raw_candidate:
            raise ValueError("--candidate must use NAME=JSON_PATH")
        name, raw_path = raw_candidate.split("=", 1)
        grouped_candidates.setdefault(name, []).append(
            (raw_path, json.loads(Path(raw_path).read_text()))
        )

    fitted: Dict[str, Any] = {}
    for name, candidate_entries in grouped_candidates.items():
        if len(candidate_entries) != len(references):
            raise ValueError(
                f"{name} has {len(candidate_entries)} captures but "
                f"{len(references)} references were supplied"
            )
        stage_results: Dict[str, Any] = {}
        for stage in _STAGES:
            all_ratios: list[float] = []
            per_capture = []
            for reference, (candidate_path, candidate) in zip(
                references, candidate_entries
            ):
                ratios = _distance_log_ratios(
                    reference,
                    candidate,
                    stage=stage,
                    min_distance=float(args.min_distance),
                )
                capture_log_multiplier = sum(ratios) / len(ratios)
                per_capture.append(
                    {
                        "candidate": str(Path(candidate_path).resolve()),
                        "multiplier": math.exp(capture_log_multiplier),
                        "matched_terms": len(ratios),
                    }
                )
                all_ratios.extend(ratios)
            log_multiplier = sum(all_ratios) / len(all_ratios)
            multiplier = math.exp(log_multiplier)
            postfit_rmse = math.sqrt(
                sum((value - log_multiplier) ** 2 for value in all_ratios)
                / len(all_ratios)
            )
            stage_results[stage] = {
                "selected_multiplier": multiplier,
                "postfit_log_ratio_rmse": postfit_rmse,
                "matched_terms": len(all_ratios),
                "per_capture": per_capture,
            }
            print(
                f"[feature-distance-temperature] {name} {stage} "
                f"multiplier={multiplier:.10g} "
                f"postfit_log_rmse={postfit_rmse:.6f}",
                flush=True,
            )
        fitted[name] = {"stages": stage_results}

    payload = {
        "definition": (
            "Fit m so matched quantiles of the actual pre-kernel exponent "
            "input (d/weighted_mean_d)/(R*m) have the same scale as MAE. "
            "R cancels in the ratio; the geometric least-squares optimum is used."
        ),
        "references": [str(Path(path).resolve()) for path in args.reference],
        "quantiles": ["p25", "p50", "p75", "p90"],
        "groups": ["positive", "repulsive_nonself"],
        "min_distance": float(args.min_distance),
        "candidates": fitted,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[feature-distance-temperature] fit -> {output_path}", flush=True)


def _huber_log_location(values: list[float], tuning: float = 1.5) -> tuple[float, float]:
    """Robust center and MAD scale for log distance ratios."""
    if not values:
        raise ValueError("Cannot fit an empty set of log ratios")
    location = statistics.median(values)
    scale = 0.0
    for _ in range(50):
        scale = max(
            1.4826 * statistics.median(abs(value - location) for value in values),
            1e-6,
        )
        cap = float(tuning) * scale
        weights = [
            min(1.0, cap / max(abs(value - location), 1e-12))
            for value in values
        ]
        updated = sum(
            weight * value for weight, value in zip(weights, values)
        ) / sum(weights)
        if abs(updated - location) < 1e-12:
            location = updated
            break
        location = updated
    return location, scale


def _run_fit_distance_robust(args: argparse.Namespace) -> None:
    """Fit d/tau while rejecting numerically degenerate generated near-ties."""
    quantiles = ("p10", "p25", "p50", "p75", "p90")
    groups = ("positive", "generated_nonself", "real_negative")

    references: Dict[int, tuple[str, Dict[str, Any]]] = {}
    for path in args.reference:
        capture = json.loads(Path(path).read_text())
        seed = int(capture["effective_seed"])
        if seed in references:
            raise ValueError(f"Duplicate reference seed: {seed}")
        references[seed] = (path, capture)

    def capture_protocol(capture: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "generator_geometry": capture.get("generator_geometry"),
            "batch": capture.get("batch"),
            "R_list": capture.get("R_list"),
            "max_token_rows_per_feature": capture.get(
                "max_token_rows_per_feature"
            ),
            "feature_layout": {
                name: {
                    key: feature.get(key)
                    for key in ("stage", "tokens", "dimension")
                }
                for name, feature in sorted((capture.get("features") or {}).items())
            },
        }

    reference_protocols = {
        json.dumps(
            capture_protocol(capture), sort_keys=True, separators=(",", ":")
        )
        for _, capture in references.values()
    }
    if len(reference_protocols) != 1:
        raise ValueError("Reference capture protocol changed between seeds")
    reference_capture_protocol = capture_protocol(next(iter(references.values()))[1])

    dataset_provenance: Optional[Dict[str, Any]] = None
    provenance_payloads = [
        capture.get("dataset_provenance") for _, capture in references.values()
    ]
    if any(value is not None for value in provenance_payloads):
        if any(value is None for value in provenance_payloads):
            raise ValueError("Reference captures mix datasets with/without provenance")
        canonical = {
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in provenance_payloads
        }
        if len(canonical) != 1:
            raise ValueError("Reference captures use different dataset provenance")
        dataset_provenance = provenance_payloads[0]

    grouped_candidates: Dict[str, Dict[int, tuple[str, Dict[str, Any]]]] = {}
    for raw_candidate in args.candidate:
        if "=" not in raw_candidate:
            raise ValueError("--candidate must use NAME=JSON_PATH")
        name, path = raw_candidate.split("=", 1)
        capture = json.loads(Path(path).read_text())
        seed = int(capture["effective_seed"])
        if seed in grouped_candidates.setdefault(name, {}):
            raise ValueError(f"Duplicate {name} seed: {seed}")
        grouped_candidates[name][seed] = (path, capture)

    fitted: Dict[str, Any] = {}
    for name, candidates in grouped_candidates.items():
        if set(candidates) != set(references):
            raise ValueError(
                f"{name} seeds {sorted(candidates)} do not match references "
                f"{sorted(references)}"
            )
        for seed, (_, candidate) in candidates.items():
            if candidate.get("dataset_provenance") != dataset_provenance:
                raise ValueError(
                    f"{name} seed {seed}: dataset provenance differs from reference"
                )
            if capture_protocol(candidate) != reference_capture_protocol:
                raise ValueError(
                    f"{name} seed {seed}: capture batch/R/feature layout differs"
                )
        hash_validation: Dict[str, Any] = {}
        for seed, (_, reference) in references.items():
            _, candidate = candidates[seed]
            if candidate["batch"] != reference["batch"]:
                raise ValueError(f"{name} seed {seed}: batch settings differ")
            matched_hashes = {}
            for hash_name, reference_hash in reference["tensor_sha256"].items():
                candidate_hash = candidate["tensor_sha256"].get(hash_name)
                matched_hashes[hash_name] = candidate_hash == reference_hash
                if candidate_hash != reference_hash:
                    raise ValueError(
                        f"{name} seed {seed}: {hash_name} tensor hash differs"
                    )
            hash_validation[str(seed)] = matched_hashes

        stage_results: Dict[str, Any] = {}
        for stage in _STAGES:
            accepted: list[Dict[str, Any]] = []
            counts = {
                group: {"accepted": 0, "excluded_below_min_distance": 0}
                for group in groups
            }
            for seed, (_, reference) in references.items():
                _, candidate = candidates[seed]
                for feature_name in sorted(
                    set(reference["features"]) & set(candidate["features"])
                ):
                    ref_feature = reference["features"][feature_name]
                    candidate_feature = candidate["features"][feature_name]
                    if (
                        ref_feature["stage"] != stage
                        or candidate_feature["stage"] != stage
                    ):
                        continue
                    ref_distributions = ref_feature["normalized_distance_quantiles"]
                    candidate_distributions = candidate_feature[
                        "normalized_distance_quantiles"
                    ]
                    for group in groups:
                        for quantile in quantiles:
                            reference_value = float(ref_distributions[group][quantile])
                            candidate_value = float(
                                candidate_distributions[group][quantile]
                            )
                            if min(reference_value, candidate_value) < float(
                                args.min_distance
                            ):
                                counts[group]["excluded_below_min_distance"] += 1
                                continue
                            counts[group]["accepted"] += 1
                            accepted.append(
                                {
                                    "seed": seed,
                                    "feature": feature_name,
                                    "group": group,
                                    "quantile": quantile,
                                    "log_ratio": math.log(
                                        candidate_value / reference_value
                                    ),
                                }
                            )

            log_ratios = [entry["log_ratio"] for entry in accepted]
            location, mad_scale = _huber_log_location(
                log_ratios, tuning=float(args.huber_tuning)
            )
            sorted_ratios = sorted(log_ratios)
            trim_count = int(float(args.trim_fraction) * len(sorted_ratios))
            trimmed = (
                sorted_ratios[trim_count:-trim_count]
                if trim_count > 0
                else sorted_ratios
            )
            group_results = {}
            for group in groups:
                values = [
                    entry["log_ratio"]
                    for entry in accepted
                    if entry["group"] == group
                ]
                if values:
                    group_location, group_scale = _huber_log_location(
                        values, tuning=float(args.huber_tuning)
                    )
                    group_results[group] = {
                        "huber_multiplier": math.exp(group_location),
                        "median_multiplier": math.exp(statistics.median(values)),
                        "log_mad_scale": group_scale,
                        **counts[group],
                    }
                else:
                    group_results[group] = {**counts[group], "status": "all_excluded"}

            per_seed = {}
            for seed in sorted(references):
                values = [
                    entry["log_ratio"]
                    for entry in accepted
                    if entry["seed"] == seed
                ]
                seed_location, _ = _huber_log_location(
                    values, tuning=float(args.huber_tuning)
                )
                per_seed[str(seed)] = math.exp(seed_location)

            multiplier = math.exp(location)
            r_values = [float(value) for value in next(iter(references.values()))[1]["R_list"]]
            stage_results[stage] = {
                "selected_multiplier": multiplier,
                "effective_tau": [value * multiplier for value in r_values],
                "huber_tuning": float(args.huber_tuning),
                "log_mad_scale": mad_scale,
                "median_multiplier": math.exp(statistics.median(log_ratios)),
                "trimmed_geometric_mean_multiplier": math.exp(
                    statistics.fmean(trimmed)
                ),
                "accepted_terms": len(log_ratios),
                "per_seed_multiplier": per_seed,
                "groups": group_results,
            }
            print(
                f"[feature-distance-robust] {name} {stage} "
                f"multiplier={multiplier:.10g} accepted={len(log_ratios)}",
                flush=True,
            )

        fitted[name] = {
            "captures": {
                str(seed): str(Path(path).resolve())
                for seed, (path, _) in candidates.items()
            },
            "input_hash_validation": hash_validation,
            "stages": stage_results,
        }

    payload = {
        "schema_version": 2,
        "definition": (
            "Robust Huber location of log(candidate/reference) for the actual "
            "normalized kernel distance d/weighted_mean_d. Values below the "
            "minimum distance are excluded before fitting because their d/(R*m) "
            "is a generated near-tie rather than a stable temperature signal."
        ),
        "references": {
            str(seed): str(Path(path).resolve())
            for seed, (path, _) in references.items()
        },
        "groups_considered": list(groups),
        "quantiles": list(quantiles),
        "min_distance": float(args.min_distance),
        "huber_tuning": float(args.huber_tuning),
        "trim_fraction": float(args.trim_fraction),
        "capture_protocol": reference_capture_protocol,
        "dataset_provenance": dataset_provenance,
        "candidates": fitted,
    }
    symmetric_reference_name = str(
        getattr(args, "symmetric_reference_name", "") or ""
    ).strip()
    if symmetric_reference_name:
        if len(fitted) != 1:
            raise ValueError(
                "--symmetric-reference-name requires exactly one candidate name"
            )
        candidate_name = next(iter(fitted))
        if candidate_name == symmetric_reference_name:
            raise ValueError("Symmetric reference and candidate names must differ")
        reference_profile = {"default": 1.0}
        candidate_profile = {"default": 1.0}
        for stage in _STAGES:
            ratio = float(
                fitted[candidate_name]["stages"][stage]["selected_multiplier"]
            )
            reference_profile[stage] = ratio ** -0.5
            candidate_profile[stage] = ratio ** 0.5
        reference_feature_provenance = {
            json.dumps(
                capture.get("feature_provenance"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for _, capture in references.values()
        }
        candidate_entries = grouped_candidates[candidate_name]
        candidate_feature_provenance = {
            json.dumps(
                capture.get("feature_provenance"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for _, capture in candidate_entries.values()
        }
        if len(reference_feature_provenance) != 1 or len(
            candidate_feature_provenance
        ) != 1:
            raise ValueError("Feature checkpoint provenance changed between seeds")
        reference_provenance = next(iter(references.values()))[1].get(
            "feature_provenance"
        )
        candidate_provenance = next(iter(candidate_entries.values()))[1].get(
            "feature_provenance"
        )
        payload.update(
            {
                "calibration_kind": "symmetric_raw_imagenet_dino_moco",
                "required_seeds": sorted(references),
                "centering": "geometric_mean_one",
                "fixed_input_feature_multipliers": {
                    "global": 1.0,
                    "norm_x": 1.0,
                    "reason": (
                        "global and norm_x are computed directly from the same raw "
                        "RGB input tensor before either frozen encoder"
                    ),
                },
                "encoder_provenance": {
                    symmetric_reference_name: reference_provenance,
                    candidate_name: candidate_provenance,
                },
                "symmetric_profiles": {
                    symmetric_reference_name: reference_profile,
                    candidate_name: candidate_profile,
                },
                "capture_validation": {
                    symmetric_reference_name: {
                        str(seed): {
                            "capture": str(Path(path).resolve()),
                            "config_sha256": capture.get("config_sha256"),
                            "tensor_sha256": capture.get("tensor_sha256"),
                        }
                        for seed, (path, capture) in references.items()
                    },
                    candidate_name: {
                        str(seed): {
                            "capture": str(Path(path).resolve()),
                            "config_sha256": capture.get("config_sha256"),
                            "tensor_sha256": capture.get("tensor_sha256"),
                        }
                        for seed, (path, capture) in candidate_entries.items()
                    },
                },
            }
        )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[feature-distance-robust] fit -> {output_path}", flush=True)


def _run_finalize_symmetric(args: argparse.Namespace) -> None:
    """Pin the forward fit only after a reverse-fit reciprocity check."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--config", required=True)
    capture.add_argument("--workdir", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--max-token-rows", type=int, default=64)
    capture.add_argument("--seed", type=int, default=None)
    capture.add_argument("--grid-min", type=float, default=0.25)
    capture.add_argument("--grid-max", type=float, default=4.0)
    capture.add_argument("--grid-steps", type=int, default=81)
    capture.set_defaults(func=_run_capture)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--reference", required=True)
    fit.add_argument("--candidate", action="append", required=True)
    fit.add_argument("--output", required=True)
    fit.set_defaults(func=_run_fit)

    fit_distance = subparsers.add_parser("fit-distance")
    fit_distance.add_argument("--reference", action="append", required=True)
    fit_distance.add_argument("--candidate", action="append", required=True)
    fit_distance.add_argument("--min-distance", type=float, default=1e-4)
    fit_distance.add_argument("--output", required=True)
    fit_distance.set_defaults(func=_run_fit_distance)

    fit_distance_robust = subparsers.add_parser("fit-distance-robust")
    fit_distance_robust.add_argument("--reference", action="append", required=True)
    fit_distance_robust.add_argument("--candidate", action="append", required=True)
    fit_distance_robust.add_argument("--min-distance", type=float, default=0.02)
    fit_distance_robust.add_argument("--huber-tuning", type=float, default=1.5)
    fit_distance_robust.add_argument("--trim-fraction", type=float, default=0.1)
    fit_distance_robust.add_argument(
        "--symmetric-reference-name",
        default="",
        help=(
            "When set, require one candidate and emit geometric-center profiles "
            "for this reference name and the candidate name."
        ),
    )
    fit_distance_robust.add_argument("--output", required=True)
    fit_distance_robust.set_defaults(func=_run_fit_distance_robust)

    finalize_symmetric = subparsers.add_parser("finalize-symmetric")
    finalize_symmetric.add_argument("--forward", required=True)
    finalize_symmetric.add_argument("--reverse", required=True)
    finalize_symmetric.add_argument("--output", required=True)
    finalize_symmetric.add_argument(
        "--max-log-reciprocity-error", type=float, default=1e-10
    )
    finalize_symmetric.set_defaults(func=_run_finalize_symmetric)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
