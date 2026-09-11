"""ImageNet DataLoader helpers for drift-model-imagenet."""
from __future__ import annotations

import multiprocessing
import os
import random
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _center_crop(img: Image.Image, size: int) -> Image.Image:
    """ADM-style center crop."""
    while min(*img.size) >= 2 * size:
        img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
    scale = size / min(*img.size)
    img = img.resize(tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC)
    arr = np.array(img)
    cy = (arr.shape[0] - size) // 2
    cx = (arr.shape[1] - size) // 2
    return Image.fromarray(arr[cy:cy + size, cx:cx + size])


class _LatentCacheDataset(datasets.DatasetFolder):
    """ImageFolder-style dataset that loads pre-encoded VAE latent .pt files."""

    def __init__(self, root: str):
        super().__init__(root=root, loader=str, extensions=(".pt",))

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        data = torch.load(path, map_location="cpu", weights_only=False)
        moments = data["moments"] if torch.rand(1).item() < 0.5 else data["moments_flip"]
        return np.asarray(moments), target


class _NpyFlatLatentDataset(torch.utils.data.Dataset):
    """Flat-index npy latent cache dataset.

    Expected layout::

        cache_root/
            imagenet256_features/{idx}.npy   # shape (1, 4, 32, 32) float32
            imagenet256_labels/{idx}.npy     # shape (1,) int64

    Indices are contiguous integers 0 .. N-1.
    """

    def __init__(self, cache_root: str):
        self.feat_dir = os.path.join(cache_root, "imagenet256_features")
        self.lbl_dir  = os.path.join(cache_root, "imagenet256_labels")
        if not os.path.isdir(self.feat_dir):
            raise FileNotFoundError(f"Features dir not found: {self.feat_dir}")
        if not os.path.isdir(self.lbl_dir):
            raise FileNotFoundError(f"Labels dir not found: {self.lbl_dir}")
        # Build sorted index list from feature dir
        self.indices = sorted(
            int(f[:-4]) for f in os.listdir(self.feat_dir) if f.endswith(".npy")
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        feat = np.load(os.path.join(self.feat_dir, f"{idx}.npy"))   # (1, 4, 32, 32)
        lbl  = np.load(os.path.join(self.lbl_dir,  f"{idx}.npy"))   # (1,)
        feat = feat.squeeze(0)   # (4, 32, 32)
        label = int(lbl.flat[0])
        return feat, label


def _resolve_npy_flat_split(cache_path: str, split: str) -> str:
    """Resolve either ``cache/<split>/...`` or a direct split directory."""
    split_root = os.path.join(cache_path, split)
    for candidate in (split_root, cache_path):
        if (
            os.path.isdir(os.path.join(candidate, "imagenet256_features"))
            and os.path.isdir(os.path.join(candidate, "imagenet256_labels"))
        ):
            return candidate
    raise FileNotFoundError(
        f"Flat latent cache split not found for split={split!r} under {cache_path}. "
        "Expected imagenet256_features/ and imagenet256_labels/ either in "
        f"{split_root} or directly in {cache_path}."
    )


class _ConcurrencyLimitedImageLoader:
    """Run the physical image open/decode under a process-shared semaphore.

    ImageFolder still has the configured number of DataLoader workers, so the
    sampler, worker seeds, transforms, and output ordering are unchanged.  Only
    the number of workers allowed to issue a raw image read/decode at once is
    capped.  The semaphore is created in the rank process and inherited by its
    DataLoader workers.
    """

    def __init__(self, semaphore) -> None:
        if semaphore is None:
            raise ValueError("A shared semaphore is required")
        self._semaphore = semaphore

    def __call__(self, path: str) -> Image.Image:
        with self._semaphore:
            return datasets.folder.default_loader(path)


def create_raw_image_io_semaphore(max_concurrency: int):
    """Create one raw-image I/O gate to share across loaders in a rank."""
    max_concurrency = int(max_concurrency)
    if max_concurrency < 0:
        raise ValueError("raw image I/O concurrency must be non-negative")
    if max_concurrency == 0:
        return None
    return multiprocessing.BoundedSemaphore(max_concurrency)


def _build_transforms(
    resolution: int,
    use_aug: bool,
    split: str,
    *,
    return_uint8: bool = False,
):
    tensor_transform = (
        transforms.PILToTensor()
        if return_uint8
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    )
    if use_aug and split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            tensor_transform,
        ])
    operations = [transforms.Lambda(lambda img: _center_crop(img, resolution))]
    # Match the cached training stream's random orientation while keeping the
    # validation reference deterministic across FID evaluations.
    if split == "train":
        operations.append(transforms.RandomHorizontalFlip())
    operations.append(tensor_transform)
    return transforms.Compose(operations)


def _worker_init_fn(worker_id: int, rank: int = 0) -> None:
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_imagenet_split(
    *,
    imagenet_path: str,
    resolution: int = 256,
    batch_size: int = 256,
    split: str = "train",
    use_aug: bool = False,
    use_latent: bool = False,
    use_cache: bool = False,
    cache_path: str = "",
    cache_format: str = "pt_imagefolder",   # "pt_imagefolder" | "npy_flat"
    vae_variant: str = "mse",
    latent_mmap_cache_path: str = "",
    latent_decoder_path: str = "",
    latent_scaling_factor: float = 0.18215,
    num_classes: int = 1000,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    latent_device: Optional[torch.device | str] = None,
    vae_model_id: Optional[str] = None,
    vae_revision: Optional[str] = None,
    return_uint8: bool = False,
    raw_image_io_concurrency: int = 0,
    raw_image_io_semaphore=None,
) -> Tuple[DataLoader, callable, callable]:
    """Create an ImageNet DataLoader with preprocess/postprocess functions.

    Returns:
        (loader, preprocess_fn, postprocess_fn)
        - preprocess_fn: (images, labels) batch → {"images": BCHW, "labels": B}
        - postprocess_fn: generated latents/pixels → pixel images in [0, 1]
    """
    from vae_imagenet import resolve_vae_model_id

    vae_model_id = resolve_vae_model_id(vae_model_id, vae_variant)
    # Explicit local-cache/decoder paths opt into byte-preserving latent inputs.
    # The existing online-VAE and .pt cache interfaces keep their defaults.
    exact_latent_inputs = bool(latent_mmap_cache_path or latent_decoder_path)
    if exact_latent_inputs:
        if not (use_latent and use_cache):
            raise ValueError("Exact latent inputs require use_latent=true and use_cache=true")
        if cache_format != "npy_flat" or resolution != 256:
            raise ValueError("Exact latent inputs require cache_format='npy_flat' and resolution=256")
        if not cache_path:
            raise ValueError("cache_path must identify the original flat latent source")
        if use_aug:
            raise ValueError("Exact latent inputs contain no declared augmentations; use_aug must be false")

    if return_uint8 and (split != "train" or use_latent or use_cache):
        raise ValueError(
            "return_uint8 is only valid for a direct raw training split "
            "(split='train', use_latent=false, use_cache=false)."
        )

    raw_image_io_concurrency = int(raw_image_io_concurrency)
    if raw_image_io_concurrency < 0:
        raise ValueError("raw_image_io_concurrency must be non-negative")
    if raw_image_io_concurrency > 0 and (use_latent or use_cache):
        raise ValueError(
            "raw_image_io_concurrency is only valid for direct raw ImageFolder inputs"
        )
    if raw_image_io_semaphore is not None and raw_image_io_concurrency <= 0:
        raise ValueError(
            "raw_image_io_semaphore requires raw_image_io_concurrency > 0"
        )

    if exact_latent_inputs and split == "train":
        from train.latent_data import FlatNpyLatentDataset, MmapLatentDataset

        if latent_mmap_cache_path:
            ds = MmapLatentDataset(
                latent_mmap_cache_path, source_root=cache_path, num_classes=num_classes,
            )
        else:
            ds = FlatNpyLatentDataset(cache_path, num_classes=num_classes)
    elif use_cache and not exact_latent_inputs:
        if not cache_path:
            raise ValueError(
                "cache_path must be set when use_cache=True. "
                "Set `cache_path` in config or IMAGENET_CACHE_PATH."
            )
        if cache_format == "npy_flat":
            ds = _NpyFlatLatentDataset(
                cache_root=_resolve_npy_flat_split(cache_path, split)
            )
        else:
            split_root = os.path.join(cache_path, split)
            if not os.path.isdir(split_root):
                raise FileNotFoundError(
                    f"Latent cache split not found: {split_root}. "
                    "Expected cache root with train/ and val/."
                )
            ds = _LatentCacheDataset(root=split_root)
    else:
        if not imagenet_path:
            raise ValueError(
                "imagenet_path must be set when use_cache=False. "
                "Set `imagenet_path` in config or IMAGENET_PATH."
            )
        split_root = os.path.join(imagenet_path, split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(
                f"ImageNet split not found: {split_root}. "
                "Set `imagenet_path` (or IMAGENET_PATH) to a directory containing train/ and val/."
            )
        tf = _build_transforms(
            resolution,
            use_aug=use_aug,
            split=split,
            return_uint8=return_uint8,
        )
        image_loader = None
        if raw_image_io_concurrency > 0:
            semaphore = raw_image_io_semaphore
            if semaphore is None:
                semaphore = create_raw_image_io_semaphore(raw_image_io_concurrency)
            image_loader = _ConcurrencyLimitedImageLoader(semaphore)
        ds = datasets.ImageFolder(
            root=split_root,
            transform=tf,
            **({"loader": image_loader} if image_loader is not None else {}),
        )

    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=(split == "train"))

    prefetch_factor = int(prefetch_factor)
    if num_workers > 0 and prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive when num_workers > 0")
    keep_workers = (
        num_workers > 0
        if persistent_workers is None
        else bool(persistent_workers) and num_workers > 0
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None and split == "train"),
        drop_last=(split == "train"),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        pin_memory=pin_memory,
        persistent_workers=keep_workers,
        worker_init_fn=lambda wid: _worker_init_fn(wid, rank),
    )

    if exact_latent_inputs:
        from train.latent_decoder import LatentDecoderPostprocessor

        def preprocess_fn(batch):
            images, label = batch
            if not isinstance(images, torch.Tensor):
                images = torch.from_numpy(np.asarray(images))
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            return {"images": images.float(), "labels": label}

        # Flat npy files contain training latents only. Validation above keeps
        # the original sorted raw RGB ImageFolder and deterministic center crop;
        # only generated latents are decoded, with no VAE encoder calls.
        postprocess_fn = LatentDecoderPostprocessor(
            model_path=latent_decoder_path or vae_model_id,
            scaling_factor=latent_scaling_factor,
        )
        return loader, preprocess_fn, postprocess_fn

    if use_latent or use_cache:
        from vae_imagenet import get_vae_enc_dec

        if use_cache:
            def preprocess_fn(batch):
                cached, label = batch
                if isinstance(cached, np.ndarray):
                    cached = torch.from_numpy(cached)
                if isinstance(label, np.ndarray):
                    label = torch.from_numpy(label)
                return {"images": cached.float(), "labels": label}
        else:
            _enc_state: dict = {}

            def preprocess_fn(batch, device=None):
                if "enc" not in _enc_state:
                    _dev = device or latent_device
                    if _dev is None:
                        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    _dev = torch.device(_dev)
                    _enc_state["enc"], _ = get_vae_enc_dec(
                        _dev,
                        model_id=vae_model_id,
                        revision=vae_revision,
                    )
                    _enc_state["device"] = _dev
                images, label = batch
                if not isinstance(images, torch.Tensor):
                    images = torch.from_numpy(np.array(images))
                if isinstance(label, np.ndarray):
                    label = torch.from_numpy(label)
                images = images.float().to(_enc_state["device"], non_blocking=True)
                latents = _enc_state["enc"](images)
                return {"images": latents, "labels": label}

        _dec_state: dict = {}

        def postprocess_fn(latents: torch.Tensor) -> torch.Tensor:
            # Decoder parameters are fp32 by default. Cached latents can be fp16,
            # so align dtype/device before decode to avoid conv dtype mismatch.
            if _dec_state.get("device") != latents.device or "dec" not in _dec_state:
                from vae_imagenet import get_vae_enc_dec
                _, _dec_state["dec"] = get_vae_enc_dec(
                    latents.device,
                    model_id=vae_model_id,
                    revision=vae_revision,
                )
                _dec_state["device"] = latents.device

            decode_in = latents.to(
                device=_dec_state["device"],
                dtype=torch.float32,
                non_blocking=True,
            )
            pixels = _dec_state["dec"](decode_in)
            return ((pixels + 1) / 2).clamp(0, 1)

        return loader, preprocess_fn, postprocess_fn

    def preprocess_fn(batch):
        images, label = batch
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(np.array(images))
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        return {
            "images": images if return_uint8 else images.float(),
            "labels": label,
        }

    def postprocess_fn(images: torch.Tensor) -> torch.Tensor:
        return ((images + 1) / 2).clamp(0, 1)

    return loader, preprocess_fn, postprocess_fn


def infinite_sampler(loader: DataLoader, start_step: int = 0) -> Iterator:
    """Yield batches indefinitely, skipping the first start_step batches."""
    epoch = start_step // len(loader)
    skip  = start_step % len(loader)
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    while True:
        for i, batch in enumerate(loader):
            if skip > 0 and i < skip:
                continue
            yield batch
        skip = 0
        epoch += 1
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
