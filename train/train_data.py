"""ImageNet DataLoader helpers for drift-model-imagenet."""
from __future__ import annotations

import multiprocessing
import os
import random
from glob import glob
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

    def __init__(self, root: str, random_flip: bool = True):
        super().__init__(root=root, loader=str, extensions=(".pt",))
        self.random_flip = bool(random_flip)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        data = torch.load(path, map_location="cpu", weights_only=False)
        required = {"moments", "moments_flip"}
        if not isinstance(data, dict) or not required.issubset(data):
            raise ValueError(
                f"Invalid PT latent cache sample {path}: expected keys "
                f"{sorted(required)}"
            )
        use_flip = self.random_flip and torch.rand(1).item() >= 0.5
        key = "moments_flip" if use_flip else "moments"
        moments = np.asarray(data[key])
        if moments.shape != (4, 32, 32):
            raise ValueError(
                f"Invalid PT latent shape in {path}:{key}: {moments.shape}; "
                "expected (4, 32, 32)"
            )
        return moments, target


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


class _HFParquetLatentDataset(torch.utils.data.Dataset):
    """ImageNet VAE latents backed by local Hugging Face cache shards.

    The usual layout contains ``train-*.parquet`` source files and lets
    :func:`datasets.load_dataset` reuse/build its Arrow cache.  A staging-only
    layout may instead contain the already materialized
    ``parquet-train-*.arrow`` shards directly; this is byte-identical while
    avoiding a second copy of the Parquet sources on another server.
    """

    def __init__(self, cache_root: str, random_flip: bool = True):
        try:
            from datasets import Dataset, concatenate_datasets, load_dataset
        except ImportError as exc:
            raise ImportError(
                "cache_format='hf_parquet' requires the 'datasets' package."
            ) from exc

        shards = sorted(glob(os.path.join(cache_root, "train-*.parquet")))
        if shards:
            self.dataset = load_dataset(
                "parquet",
                data_files={"train": shards},
                split="train",
                cache_dir=os.path.join(cache_root, ".arrow_cache"),
            )
        else:
            arrow_shards = sorted(
                glob(os.path.join(cache_root, "parquet-train-*.arrow"))
            )
            if not arrow_shards:
                raise FileNotFoundError(
                    "No train-*.parquet or parquet-train-*.arrow shards found "
                    f"in: {cache_root}"
                )
            self.dataset = concatenate_datasets(
                [Dataset.from_file(path) for path in arrow_shards]
            )
        required = {"latent_mean", "latent_mean_flip", "label"}
        missing = required.difference(self.dataset.column_names)
        if missing:
            raise ValueError(f"HF latent cache is missing columns: {sorted(missing)}")
        self.dataset.set_format(
            type="numpy",
            columns=["latent_mean", "latent_mean_flip", "label"],
        )
        self.random_flip = bool(random_flip)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        use_flip = self.random_flip and torch.rand(1).item() >= 0.5
        key = "latent_mean_flip" if use_flip else "latent_mean"
        latent = np.asarray(row[key], dtype=np.float32).reshape(4, 32, 32)
        return latent, int(row["label"])


def infer_latent_cache_format(cache_root: str) -> str:
    """Infer one supported latent-cache layout from its filesystem tree."""
    if os.path.isdir(os.path.join(cache_root, "train")):
        return "pt_imagefolder"
    if glob(os.path.join(cache_root, "train-*.parquet")) or glob(
        os.path.join(cache_root, "parquet-train-*.arrow")
    ):
        return "hf_parquet"
    if os.path.isdir(os.path.join(cache_root, "imagenet256_features")):
        return "npy_flat"
    raise ValueError(
        "Could not infer latent cache format from "
        f"{cache_root}. Expected train/<class>/*.pt, train-*.parquet, "
        "parquet-train-*.arrow, or imagenet256_features/."
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
    cache_format: str = "pt_imagefolder",   # "auto" | "pt_imagefolder" | "npy_flat" | "hf_parquet"
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    random_flip: bool = True,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    latent_device: Optional[torch.device | str] = None,
    vae_model_id: str = "stabilityai/sd-vae-ft-mse",
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

    if use_cache:
        if not cache_path:
            raise ValueError(
                "cache_path must be set when use_cache=True. "
                "Set `cache_path` in config or IMAGENET_CACHE_PATH."
            )
        if cache_format == "auto":
            cache_format = infer_latent_cache_format(cache_path)
        if cache_format == "npy_flat":
            if split != "train":
                raise ValueError(
                    "cache_format='npy_flat' currently contains only the train split."
                )
            # Flat-index .npy layout: cache_path/imagenet256_features & imagenet256_labels
            # (no train/val subdirectory — full train set only)
            ds = _NpyFlatLatentDataset(cache_root=cache_path)
        elif cache_format == "hf_parquet":
            if split != "train":
                raise ValueError(
                    "cache_format='hf_parquet' currently contains only the train split."
                )
            ds = _HFParquetLatentDataset(
                cache_root=cache_path,
                random_flip=random_flip,
            )
        elif cache_format == "pt_imagefolder":
            split_root = os.path.join(cache_path, split)
            if not os.path.isdir(split_root):
                raise FileNotFoundError(
                    f"Latent cache split not found: {split_root}. "
                    "Expected cache root with train/ and val/."
                )
            ds = _LatentCacheDataset(
                root=split_root,
                random_flip=random_flip,
            )
        else:
            raise ValueError(
                f"Unknown cache_format={cache_format!r}; expected one of "
                "'auto', 'pt_imagefolder', 'npy_flat', or 'hf_parquet'."
            )
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

    should_shuffle = (split == "train") if shuffle is None else bool(shuffle)
    should_drop_last = (split == "train") if drop_last is None else bool(drop_last)

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=should_shuffle,
        )

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
        shuffle=(sampler is None and should_shuffle),
        drop_last=should_drop_last,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        pin_memory=pin_memory,
        persistent_workers=keep_workers,
        worker_init_fn=lambda wid: _worker_init_fn(wid, rank),
    )

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
