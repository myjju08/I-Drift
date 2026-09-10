"""Offline ImageNet-256 evaluation with official-style reference stats.

This script:
1. Loads the EMA checkpoint.
2. Generates class-conditional 256x256 samples with multi-GPU sampling.
3. Writes a transient ADM-style sample archive (.npz with arr_0.npy).
4. Computes FID, IS, precision, and recall in one execution.

Training-time eval in train_imagenet_gen.py intentionally stays lightweight and
unchanged. This script is for full offline evaluation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from numpy.lib import format as npformat
import torch
import torch.nn as nn
from tqdm import tqdm

# Make repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.imagenet_generator import build_ditgen_from_config  # noqa: E402
from train_imagenet_gen import _amp_ctx, _gen_use_bf16, load_yaml_config  # noqa: E402
from utils import EMA  # noqa: E402
from vae_imagenet import _load_vae  # noqa: E402


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64, copy=False)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _compute_inception_score_from_logits(logits: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    rng = np.random.RandomState(2020)
    logits = np.asarray(logits)
    logits = logits[rng.permutation(logits.shape[0]), :]
    probs = _softmax_np(logits)
    n = probs.shape[0]
    split_size = n // splits
    if split_size <= 0:
        raise ValueError(f"Need at least {splits} samples to compute Inception Score, got {n}")
    probs = probs[: split_size * splits]
    scores = []
    for split in range(splits):
        part = probs[split * split_size : (split + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    arr = np.asarray(scores, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _read_exact(fp, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = fp.read(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {len(data)}")
    return bytes(data)


@dataclass
class MomentAccumulator:
    dim: int

    def __post_init__(self) -> None:
        self.count = 0
        self.sum = np.zeros((self.dim,), dtype=np.float64)
        self.sum_outer = np.zeros((self.dim, self.dim), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected (N,{self.dim}) array, got {x.shape}")
        x64 = x.astype(np.float64, copy=False)
        self.count += x64.shape[0]
        self.sum += x64.sum(axis=0)
        self.sum_outer += x64.T @ x64

    def mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("Need at least two samples to compute covariance.")
        mean = self.sum / self.count
        cov = (self.sum_outer - self.count * np.outer(mean, mean)) / (self.count - 1)
        return mean, cov


class NpzArrayWriter:
    def __init__(self, path: Path, shape: Tuple[int, ...], dtype: np.dtype) -> None:
        self.path = path
        self.shape = tuple(int(x) for x in shape)
        self.dtype = np.dtype(dtype)
        self.count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.zf = zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
        self.fp = self.zf.open("arr_0.npy", "w", force_zip64=True)
        header = {
            "descr": npformat.dtype_to_descr(self.dtype),
            "fortran_order": False,
            "shape": self.shape,
        }
        npformat.write_array_header_2_0(self.fp, header)

    def write(self, batch: np.ndarray) -> None:
        batch = np.ascontiguousarray(batch, dtype=self.dtype)
        if batch.shape[1:] != self.shape[1:]:
            raise ValueError(f"Expected trailing shape {self.shape[1:]}, got {batch.shape[1:]}")
        self.count += int(batch.shape[0])
        self.fp.write(batch.tobytes(order="C"))

    def close(self, verify_count: bool = True) -> None:
        try:
            self.fp.close()
        finally:
            self.zf.close()
        if verify_count and self.count != self.shape[0]:
            raise ValueError(f"Wrote {self.count} samples, expected {self.shape[0]}")

    def __enter__(self) -> "NpzArrayWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(verify_count=(exc_type is None))


class NpzArrayReader:
    def __init__(self, npz_path: str, key: str = "arr_0") -> None:
        self.zf = zipfile.ZipFile(npz_path, "r")
        member = key if key.endswith(".npy") else f"{key}.npy"
        self.fp = self.zf.open(member, "r")
        version = npformat.read_magic(self.fp)
        if version == (1, 0):
            shape, fortran_order, dtype = npformat.read_array_header_1_0(self.fp)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = npformat.read_array_header_2_0(self.fp)
        else:
            raise ValueError(f"Unsupported .npy version {version} in {npz_path}:{member}")
        if fortran_order:
            raise ValueError(f"Fortran-order arrays are not supported: {npz_path}:{member}")
        self.shape = tuple(int(x) for x in shape)
        self.dtype = np.dtype(dtype)
        self.row_size = int(np.prod(self.shape[1:], dtype=np.int64))
        self.row_bytes = self.row_size * self.dtype.itemsize
        self.index = 0

    def __iter__(self) -> "NpzArrayReader":
        return self

    def iter_batches(self, batch_size: int) -> Iterator[np.ndarray]:
        batch_size = max(1, int(batch_size))
        while self.index < self.shape[0]:
            take = min(batch_size, self.shape[0] - self.index)
            raw = _read_exact(self.fp, take * self.row_bytes)
            arr = np.frombuffer(raw, dtype=self.dtype).reshape((take, *self.shape[1:])).copy()
            self.index += take
            yield arr

    def close(self) -> None:
        try:
            self.fp.close()
        finally:
            self.zf.close()

    def __enter__(self) -> "NpzArrayReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class VaeDecodeModule(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.vae = _load_vae(device)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        images = self.vae.decode(latents.float() / 0.18215).sample
        return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


class PixelDecodeModule(nn.Module):
    """Map direct RGB generator outputs from training space to [0, 1]."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "Pixel-space evaluation expects BCHW RGB generator outputs, "
                f"got {tuple(images.shape)}"
            )
        return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


class InceptionFeatureBundle(nn.Module):
    """PyTorch-GPU analogue of the official Drifting JAX Inception path.

    torch-fidelity's ``inception-v3-compat`` uses the TensorFlow-converted
    Inception weights, TF-compatible bilinear resize to 299x299, and the same
    ``(x - 128) / 128`` input normalization used by the official release.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        try:
            from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        except ImportError as exc:
            raise ImportError(
                "torch-fidelity is required for official PyTorch-GPU evaluation. "
                "Install with `pip install torch-fidelity`."
            ) from exc
        self.extractor = FeatureExtractorInceptionV3(
            name="inception-v3-compat",
            features_list=["2048", "logits_unbiased"],
        ).to(device)
        self.extractor.eval()

    def forward(self, images_u8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.extractor(images_u8)
        if isinstance(feats, torch.Tensor):
            feats = (feats,)
        if len(feats) != 2:
            raise RuntimeError(f"Unexpected inception output arity: {len(feats)}")
        pool3, logits = feats
        return pool3.float(), logits.float()


def _load_ref_stats(npz_path: str) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        keys = set(data.files)
        if {"ref_mu", "ref_sigma"}.issubset(keys):
            mu_key, sigma_key = "ref_mu", "ref_sigma"
        elif {"mu", "sigma"}.issubset(keys):
            mu_key, sigma_key = "mu", "sigma"
        else:
            raise KeyError(
                f"Reference stats must contain ref_mu/ref_sigma or mu/sigma: {npz_path}"
            )
        stats = {
            "mu": np.asarray(data[mu_key], dtype=np.float64).copy(),
            "sigma": np.asarray(data[sigma_key], dtype=np.float64).copy(),
        }
        if {"mu_s", "sigma_s"}.issubset(keys):
            stats["mu_s"] = np.asarray(data["mu_s"], dtype=np.float64).copy()
            stats["sigma_s"] = np.asarray(data["sigma_s"], dtype=np.float64).copy()
        return stats


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    del eps
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)
    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean shape mismatch: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariance shape mismatch: {sigma1.shape} vs {sigma2.shape}")
    diff = mu1 - mu2
    tr_covmean = np.sum(
        np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2)).astype("complex128")).real
    )
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def _make_balanced_eval_labels(n_samples: int, num_classes: int, seed: int) -> np.ndarray:
    reps = int(math.ceil(float(n_samples) / float(num_classes)))
    labels = np.tile(np.arange(num_classes, dtype=np.int64), reps)[:n_samples]
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    return labels


def _load_imagenet_val_labels(cfg: dict) -> np.ndarray:
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for --label_source val. "
            "Set LABEL_SOURCE=balanced to use the old class-balanced labels."
        ) from exc

    imagenet_path = str(cfg.get("imagenet_path") or os.environ.get("IMAGENET_PATH", ""))
    cache_path = str(cfg.get("cache_path") or os.environ.get("IMAGENET_CACHE_PATH", ""))
    use_cache = bool(cfg.get("use_cache", False))

    if use_cache and cache_path:
        split_root = os.path.join(cache_path, "val")
        if os.path.isdir(split_root):
            ds = datasets.DatasetFolder(root=split_root, loader=str, extensions=(".pt",))
            return np.asarray([target for _, target in ds.samples], dtype=np.int64)

    split_root = os.path.join(imagenet_path, "val")
    if not os.path.isdir(split_root):
        raise FileNotFoundError(
            "ImageNet val split not found for official val-label evaluation: "
            f"{split_root}. Set imagenet_path/IMAGENET_PATH or use LABEL_SOURCE=balanced."
        )
    ds = datasets.ImageFolder(root=split_root)
    return np.asarray([target for _, target in ds.samples], dtype=np.int64)


def _make_eval_labels(
    n_samples: int,
    num_classes: int,
    seed: int,
    cfg: dict,
    label_source: str,
) -> np.ndarray:
    if label_source == "balanced":
        return _make_balanced_eval_labels(n_samples, num_classes=num_classes, seed=seed)
    if label_source not in ("val", "official_val"):
        raise ValueError(
            f"Unknown label_source={label_source!r}; expected 'official_val', 'val', or 'balanced'."
        )

    labels = _load_imagenet_val_labels(cfg)
    if labels.size == 0:
        raise ValueError("No ImageNet validation labels were found.")
    if label_source == "official_val":
        # Official Drifting inference builds the val loader with
        # DistributedSampler(..., shuffle=True) and epoch0_sampler() calls
        # sampler.set_epoch(0). For a single eval process this is exactly
        # torch.randperm(N) with DistributedSampler's default seed=0.
        generator = torch.Generator()
        generator.manual_seed(0)
        indices = torch.randperm(int(labels.shape[0]), generator=generator).numpy()
        labels = labels[indices]
    if n_samples <= labels.shape[0]:
        return labels[:n_samples].astype(np.int64, copy=False)
    reps = int(math.ceil(float(n_samples) / float(labels.shape[0])))
    return np.tile(labels, reps)[:n_samples].astype(np.int64, copy=False)


def _load_ema_state_dict_compat(ema: EMA, state_dict: dict) -> None:
    """Load EMA weights, trimming the old extra null-class row when present."""
    target_state = ema.shadow.state_dict()
    key = "class_embed.weight"
    if key in state_dict and key in target_state:
        src = state_dict[key]
        dst = target_state[key]
        if (
            torch.is_tensor(src)
            and torch.is_tensor(dst)
            and src.ndim == 2
            and dst.ndim == 2
            and src.shape != dst.shape
            and src.shape[0] == dst.shape[0] + 1
            and src.shape[1:] == dst.shape[1:]
        ):
            state_dict = dict(state_dict)
            state_dict[key] = src[: dst.shape[0]]
            print(
                "[eval] trimmed legacy null class embedding "
                f"{tuple(src.shape)} -> {tuple(state_dict[key].shape)}"
            )
    ema.load_state_dict(state_dict)


def _build_generator(cfg: dict, ckpt_path: str, device: torch.device) -> Tuple[nn.Module, int]:
    raw_cfg = cfg.get("_raw", {})
    gen_raw = build_ditgen_from_config(raw_cfg.get("model", cfg), raw_cfg.get("dataset", cfg)).to(device)
    ema = EMA(gen_raw, decay=float(cfg.get("ema_decay", 0.999)))
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "ema" not in state:
        raise KeyError(f"Checkpoint does not contain EMA weights: {ckpt_path}")
    _load_ema_state_dict_compat(ema, state["ema"])
    step_loaded = int(state.get("step", -1))
    shadow = ema.shadow.to(device).eval()
    del gen_raw, ema, state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return shadow, step_loaded


def _maybe_dataparallel(module: nn.Module) -> nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(module, device_ids=list(range(torch.cuda.device_count())))
    return module


@torch.no_grad()
def _sample_and_extract(
    generator: nn.Module,
    decoder: nn.Module,
    feature_model: InceptionFeatureBundle,
    device: torch.device,
    cfg_scale: float,
    labels_np: np.ndarray,
    batch_size: int,
    sample_npz_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    use_bf16 = _gen_use_bf16(generator)
    total = int(labels_np.shape[0])
    pool3_chunks = []
    logits_chunks = []
    pool3_stats = MomentAccumulator(2048)
    with NpzArrayWriter(sample_npz_path, shape=(total, 256, 256, 3), dtype=np.uint8) as writer:
        for start in tqdm(range(0, total, max(1, batch_size)), desc="sample", unit="batch"):
            end = min(total, start + max(1, batch_size))
            labels = torch.from_numpy(labels_np[start:end]).long().to(device, non_blocking=True)
            with _amp_ctx(use_bf16):
                out = generator(labels, cfg_scale=cfg_scale, train=False)
            latents = out["samples"]
            pixels = decoder(latents)
            # Match official Drifting _to_uint8: nan_to_num, multiply by 255,
            # clip, then cast to uint8 (truncate/floor for non-negative values).
            pixels = torch.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)
            images_u8 = (
                pixels.mul(255.0)
                .clamp(0.0, 255.0)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
                .cpu()
                .numpy()
            )
            writer.write(images_u8)

            images_t = torch.from_numpy(images_u8).permute(0, 3, 1, 2).contiguous().to(
                device, non_blocking=True
            )
            pool3_t, logits_t = feature_model(images_t)
            pool3_np = pool3_t.cpu().numpy().astype(np.float32, copy=False)
            logits_np = logits_t.cpu().numpy().astype(np.float32, copy=False)

            pool3_chunks.append(pool3_np.copy())
            logits_chunks.append(logits_np.copy())
            pool3_stats.update(pool3_np)

            del labels, out, latents, pixels, images_t, pool3_t, logits_t
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    sample_pool3 = np.concatenate(pool3_chunks, axis=0)
    sample_logits = np.concatenate(logits_chunks, axis=0)
    sample_pool3_mu_sigma = pool3_stats.mean_cov()
    del pool3_chunks, logits_chunks, pool3_stats
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sample_pool3, sample_logits, sample_pool3_mu_sigma


@torch.no_grad()
def _extract_ref_pool3(
    ref_npz: str,
    feature_model: InceptionFeatureBundle,
    device: torch.device,
    batch_size: int,
    max_images: Optional[int] = None,
) -> np.ndarray:
    chunks = []
    remaining = None if max_images is None or int(max_images) <= 0 else int(max_images)
    with NpzArrayReader(ref_npz, key="arr_0") as reader:
        for batch in tqdm(reader.iter_batches(batch_size), desc="ref", unit="batch"):
            if remaining is not None:
                if remaining <= 0:
                    break
                batch = batch[:remaining]
                remaining -= int(batch.shape[0])
            images_t = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(
                device, non_blocking=True
            )
            pool3_t, _ = feature_model(images_t)
            chunks.append(pool3_t.cpu().numpy().astype(np.float32, copy=False).copy())
            del images_t, pool3_t
    if not chunks:
        raise ValueError(f"No reference images were read from {ref_npz}")
    ref_pool3 = np.concatenate(chunks, axis=0)
    del chunks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ref_pool3


def _pairwise_squared_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1).unsqueeze(0)
    d = a2 + b2 - 2.0 * (a @ b.t())
    return d.clamp_min_(0.0)


def _pairwise_squared_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1)[None, :]
    d = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


def _compute_manifold_radii(
    feats: np.ndarray,
    device: torch.device,
    nhood_size: int,
    row_batch_size: int,
    col_batch_size: int,
) -> np.ndarray:
    n = feats.shape[0]
    k = nhood_size + 1
    radii = np.empty((n,), dtype=np.float64)
    if device.type == "cuda":
        feats_t = torch.from_numpy(feats).to(device=device, dtype=torch.float64)
        for rs in tqdm(range(0, n, row_batch_size), desc="pr-radii", unit="batch", leave=False):
            re = min(n, rs + row_batch_size)
            a = feats_t[rs:re]
            knn = None
            for cs in range(0, n, col_batch_size):
                ce = min(n, cs + col_batch_size)
                d = _pairwise_squared_torch(a, feats_t[cs:ce])
                if knn is None:
                    knn = d if d.shape[1] <= k else torch.topk(d, k=k, dim=1, largest=False).values
                else:
                    knn = torch.topk(torch.cat([knn, d], dim=1), k=k, dim=1, largest=False).values
                del d
            radii[rs:re] = knn[:, nhood_size].cpu().numpy()
            del a, knn
            torch.cuda.empty_cache()
        del feats_t
        torch.cuda.empty_cache()
        return radii

    for rs in tqdm(range(0, n, row_batch_size), desc="pr-radii", unit="batch", leave=False):
        re = min(n, rs + row_batch_size)
        a = feats[rs:re]
        knn = None
        for cs in range(0, n, col_batch_size):
            ce = min(n, cs + col_batch_size)
            d = _pairwise_squared_np(a, feats[cs:ce])
            if knn is None:
                knn = d if d.shape[1] <= k else np.partition(d, kth=k - 1, axis=1)[:, :k]
            else:
                knn = np.partition(np.concatenate([knn, d], axis=1), kth=k - 1, axis=1)[:, :k]
        radii[rs:re] = np.sort(knn, axis=1)[:, nhood_size]
    return radii


def _compute_membership(
    eval_feats: np.ndarray,
    ref_feats: np.ndarray,
    ref_radii: np.ndarray,
    device: torch.device,
    row_batch_size: int,
    col_batch_size: int,
) -> np.ndarray:
    hits = np.zeros((eval_feats.shape[0],), dtype=bool)
    if device.type == "cuda":
        eval_t = torch.from_numpy(eval_feats).to(device=device, dtype=torch.float64)
        ref_t = torch.from_numpy(ref_feats).to(device=device, dtype=torch.float64)
        ref_radii_t = torch.from_numpy(ref_radii).to(device=device, dtype=torch.float64)
        for rs in tqdm(range(0, eval_feats.shape[0], row_batch_size), desc="pr-membership", unit="batch", leave=False):
            re = min(eval_feats.shape[0], rs + row_batch_size)
            q = eval_t[rs:re]
            hit = torch.zeros((re - rs,), dtype=torch.bool, device=device)
            for cs in range(0, ref_feats.shape[0], col_batch_size):
                ce = min(ref_feats.shape[0], cs + col_batch_size)
                d = _pairwise_squared_torch(q, ref_t[cs:ce])
                hit |= (d <= ref_radii_t[cs:ce].unsqueeze(0)).any(dim=1)
                del d
                if bool(hit.all()):
                    break
            hits[rs:re] = hit.cpu().numpy()
            del q, hit
            torch.cuda.empty_cache()
        del eval_t, ref_t, ref_radii_t
        torch.cuda.empty_cache()
        return hits

    for rs in tqdm(range(0, eval_feats.shape[0], row_batch_size), desc="pr-membership", unit="batch", leave=False):
        re = min(eval_feats.shape[0], rs + row_batch_size)
        q = eval_feats[rs:re]
        hit = np.zeros((re - rs,), dtype=bool)
        for cs in range(0, ref_feats.shape[0], col_batch_size):
            ce = min(ref_feats.shape[0], cs + col_batch_size)
            d = _pairwise_squared_np(q, ref_feats[cs:ce])
            hit |= (d <= ref_radii[cs:ce][None, :]).any(axis=1)
            if bool(hit.all()):
                break
        hits[rs:re] = hit
    return hits


def _compute_precision_recall(
    sample_pool3: np.ndarray,
    ref_pool3: np.ndarray,
    device: torch.device,
    nhood_size: int,
    row_batch_size: int,
    col_batch_size: int,
) -> Tuple[float, float]:
    print("[eval] Computing precision/recall manifolds...")
    ref_radii = _compute_manifold_radii(
        ref_pool3, device=device, nhood_size=nhood_size, row_batch_size=row_batch_size, col_batch_size=col_batch_size
    )
    sample_radii = _compute_manifold_radii(
        sample_pool3,
        device=device,
        nhood_size=nhood_size,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
    )
    sample_in_ref = _compute_membership(
        sample_pool3,
        ref_pool3,
        ref_radii,
        device=device,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
    )
    ref_in_sample = _compute_membership(
        ref_pool3,
        sample_pool3,
        sample_radii,
        device=device,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
    )
    precision = float(sample_in_ref.mean())
    recall = float(ref_in_sample.mean())
    return precision, recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ImageNet-256 official-style eval.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cfg_scale", type=float, required=True)
    parser.add_argument("--eval_ref_npz", default="", help="Legacy combined reference npz for FID and PR.")
    parser.add_argument("--fid_ref_npz", default="", help="Reference stats npz containing ref_mu/ref_sigma or mu/sigma.")
    parser.add_argument("--pr_ref_npz", default="", help="Reference image npz containing arr_0 for precision/recall.")
    parser.add_argument("--pr_ref_count", type=int, default=10000, help="Number of PR reference images to use; <=0 uses all.")
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256, help="Total generation batch across visible GPUs.")
    parser.add_argument("--metrics_batch_size", type=int, default=128, help="Feature extraction batch size.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label_source",
        choices=("official_val", "val", "balanced"),
        default="official_val",
        help=(
            "official_val matches Drifting's epoch-0 shuffled val sampler; "
            "val keeps sorted ImageNet val order; balanced keeps the old class-balanced shuffle."
        ),
    )
    parser.add_argument("--out", required=True, help="JSON output path.")
    parser.add_argument(
        "--sample_npz",
        default="",
        help="Override transient sample archive path. Default: <out>.samples.npz",
    )
    parser.add_argument(
        "--keep_sample_npz",
        action="store_true",
        help="Keep the generated sample archive after metrics are written.",
    )
    parser.add_argument("--pr_nhood", type=int, default=3)
    parser.add_argument("--pr_row_batch_size", type=int, default=1024)
    parser.add_argument("--pr_col_batch_size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml_config(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_npz_path = Path(args.sample_npz) if args.sample_npz else out_path.with_suffix(".samples.npz")
    if sample_npz_path.suffix != ".npz":
        sample_npz_path = sample_npz_path.with_suffix(".npz")

    fid_ref_npz = args.fid_ref_npz or args.eval_ref_npz
    pr_ref_npz = args.pr_ref_npz or args.eval_ref_npz
    if not fid_ref_npz:
        raise ValueError("Set --fid_ref_npz or legacy --eval_ref_npz")
    if not pr_ref_npz:
        raise ValueError("Set --pr_ref_npz or legacy --eval_ref_npz")
    pr_ref_count = None if int(args.pr_ref_count) <= 0 else int(args.pr_ref_count)

    print(f"[eval] device={device}")
    print(f"[eval] visible_gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] cfg_scale={args.cfg_scale}")
    print(f"[eval] n_samples={args.n_samples}")
    print(f"[eval] batch_size={args.batch_size}")
    print(f"[eval] seed={args.seed}")
    print(f"[eval] label_source={args.label_source}")
    print(f"[eval] metrics_batch_size={args.metrics_batch_size}")
    print(f"[eval] fid_ref_npz={fid_ref_npz}")
    print(f"[eval] pr_ref_npz={pr_ref_npz}")
    print(f"[eval] pr_ref_count={pr_ref_count if pr_ref_count is not None else 'all'}")
    print(f"[eval] sample_npz={sample_npz_path}")

    if not os.path.isfile(fid_ref_npz):
        raise FileNotFoundError(f"FID reference stats not found: {fid_ref_npz}")
    if not os.path.isfile(pr_ref_npz):
        raise FileNotFoundError(f"Precision/recall reference batch not found: {pr_ref_npz}")

    num_classes = int(cfg.get("num_classes", 1000))
    labels_np = _make_eval_labels(
        args.n_samples,
        num_classes=num_classes,
        seed=args.seed,
        cfg=cfg,
        label_source=args.label_source,
    )

    generator, step_loaded = _build_generator(cfg, args.ckpt, device)
    generator = _maybe_dataparallel(generator)
    if bool(cfg.get("use_latent", True)):
        decoder_kind = "sd_vae"
        decoder_module: nn.Module = VaeDecodeModule(device)
    else:
        decoder_kind = "direct_pixel"
        decoder_module = PixelDecodeModule()
    print(f"[eval] generator_output={decoder_kind}")
    decoder = _maybe_dataparallel(decoder_module.to(device).eval())
    feature_model = InceptionFeatureBundle(device).to(device).eval()
    ref_stats = _load_ref_stats(fid_ref_npz)

    print("[eval] Sampling images and extracting sample activations...")
    sample_pool3, sample_logits, (sample_mu, sample_sigma) = _sample_and_extract(
        generator=generator,
        decoder=decoder,
        feature_model=feature_model,
        device=device,
        cfg_scale=float(args.cfg_scale),
        labels_np=labels_np,
        batch_size=int(args.batch_size),
        sample_npz_path=sample_npz_path,
    )
    is_mean, is_std = _compute_inception_score_from_logits(sample_logits)
    del sample_logits

    print("[eval] Extracting reference pool3 activations for precision/recall...")
    ref_pool3 = _extract_ref_pool3(
        ref_npz=pr_ref_npz,
        feature_model=feature_model,
        device=device,
        batch_size=int(args.metrics_batch_size),
        max_images=pr_ref_count,
    )

    del feature_model, generator, decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fid = _frechet_distance(ref_stats["mu"], ref_stats["sigma"], sample_mu, sample_sigma)
    precision, recall = _compute_precision_recall(
        sample_pool3=sample_pool3,
        ref_pool3=ref_pool3,
        device=device,
        nhood_size=int(args.pr_nhood),
        row_batch_size=int(args.pr_row_batch_size),
        col_batch_size=int(args.pr_col_batch_size),
    )

    elapsed = time.time() - t_start
    results = {
        "ckpt": args.ckpt,
        "config": args.config,
        "step": step_loaded,
        "cfg_scale": float(args.cfg_scale),
        "n_samples": int(args.n_samples),
        "sample_npz": str(sample_npz_path),
        "label_source": args.label_source,
        "fid_ref_npz": fid_ref_npz,
        "pr_ref_npz": pr_ref_npz,
        "pr_ref_count": 0 if pr_ref_count is None else int(pr_ref_count),
        "fid": float(fid),
        "is_mean": float(is_mean),
        "is_std": float(is_std),
        "isc_mean": float(is_mean),
        "isc_std": float(is_std),
        "precision": float(precision),
        "recall": float(recall),
        "elapsed_sec": float(elapsed),
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[eval] FID       = {fid:.6f}")
    print(f"[eval] IS        = {is_mean:.6f} +/- {is_std:.6f}")
    print(f"[eval] Precision = {precision:.6f}")
    print(f"[eval] Recall    = {recall:.6f}")
    print(f"[eval] JSON      = {out_path}")

    if not args.keep_sample_npz and sample_npz_path.exists():
        sample_npz_path.unlink()
        print(f"[eval] removed sample archive: {sample_npz_path}")


if __name__ == "__main__":
    main()
