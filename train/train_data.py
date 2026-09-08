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
    # Preserve training orientation randomness and deterministic validation.
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
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    return_uint8: bool = False,
    raw_image_io_concurrency: int = 0,
    raw_image_io_semaphore=None,
) -> Tuple[DataLoader, callable, callable]:
    """Create an ImageNet DataLoader with preprocess/postprocess functions.

    Returns:
        (loader, preprocess_fn, postprocess_fn)
        - preprocess_fn: (images, labels) batch → {"images": BCHW, "labels": B}
        - postprocess_fn: generated RGB pixels → pixel images in [0, 1]
    """
    if use_latent or use_cache:
        raise ValueError("ImageNet loading supports direct raw RGB inputs only")
    if return_uint8 and split != "train":
        raise ValueError(
            "return_uint8 is only valid for a direct raw training split "
            "(split='train', use_latent=false, use_cache=false)."
        )

    raw_image_io_concurrency = int(raw_image_io_concurrency)
    if raw_image_io_concurrency < 0:
        raise ValueError("raw_image_io_concurrency must be non-negative")
    if raw_image_io_semaphore is not None and raw_image_io_concurrency <= 0:
        raise ValueError(
            "raw_image_io_semaphore requires raw_image_io_concurrency > 0"
        )

    if not imagenet_path:
        raise ValueError(
            "imagenet_path must be set for raw RGB loading. "
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
