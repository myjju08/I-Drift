#!/usr/bin/env python3
"""Build a resumable JPEG pixel cache from the ImageNet SD-VAE cache.

The source layout is expected to be::

    latent_root/{train,val}/<class>/<sample>.pt

and the output mirrors it as::

    output_root/{train,val}/<class>/<sample>.jpg

Each ``.pt`` file may contain both ``moments`` and ``moments_flip``.  This
builder intentionally decodes only ``moments``; pixel-space augmentation can
apply a horizontal flip after loading the JPEG.

Examples::

    torchrun --standalone --nproc_per_node=4 \
      scripts/build_pixel_cache_from_latents.py \
      --latent-root data/imagenet/latent_cache_256 \
      --output-root data/imagenet/pixel_cache_256_vae_q95 \
      --splits train val --batch-size 64 --jpeg-quality 95

    # Resume the same command. Non-empty destination files are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Iterator, Sequence

import torch
import torch.distributed as dist
from tqdm import tqdm
from torchvision.io import encode_jpeg


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode an ImageNet latent cache into a sharded JPEG cache."
    )
    parser.add_argument(
        "--latent-root",
        type=Path,
        required=True,
        help="Cache root containing train/ and/or val/ class directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root for mirrored JPEG files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
        help="Splits to build (default: train val).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="VAE decode batch size per rank (default: 64).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="torchvision CUDA JPEG quality in [1, 100] (default: 95).",
    )
    parser.add_argument(
        "--model-id",
        default="stabilityai/sd-vae-ft-mse",
        help="Diffusers AutoencoderKL model id.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face model revision.",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=8,
        help="Threads per rank used to load small .pt files (default: 8).",
    )
    parser.add_argument(
        "--write-workers",
        type=int,
        default=8,
        help="Threads per rank used for atomic JPEG writes (default: 8).",
    )
    parser.add_argument(
        "--max-files-per-rank",
        type=int,
        default=0,
        help="Debug-only per-rank limit; 0 processes the full shard.",
    )
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=0,
        help=(
            "Balanced pilot-cache limit applied independently to every class "
            "and requested split; 0 processes every source file."
        ),
    )
    args = parser.parse_args()

    # Preserve user order while removing duplicates.
    args.splits = tuple(dict.fromkeys(args.splits))
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if args.load_workers <= 0 or args.write_workers <= 0:
        parser.error("--load-workers and --write-workers must be positive")
    if args.max_files_per_rank < 0:
        parser.error("--max-files-per-rank must be non-negative")
    if args.max_files_per_class < 0:
        parser.error("--max-files-per-class must be non-negative")
    return args


def _distributed_context() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VAE decoding and CUDA JPEG encoding")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _class_directories(latent_root: Path, split: str) -> list[Path]:
    split_root = latent_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Latent-cache split does not exist: {split_root}")
    with os.scandir(split_root) as entries:
        names = sorted(entry.name for entry in entries if entry.is_dir())
    if not names:
        raise RuntimeError(f"No class directories found under {split_root}")
    return [split_root / name for name in names]


def _iter_latent_files(
    latent_root: Path,
    splits: Sequence[str],
    max_files_per_class: int = 0,
) -> Iterator[Path]:
    """Yield source paths in a deterministic order without retaining 1.3M Paths."""
    for split in splits:
        for class_dir in _class_directories(latent_root, split):
            with os.scandir(class_dir) as entries:
                names = sorted(
                    entry.name
                    for entry in entries
                    if entry.is_file() and entry.name.endswith(".pt")
                )
            if max_files_per_class:
                names = names[:max_files_per_class]
            for name in names:
                yield class_dir / name


def _count_sources(
    latent_root: Path,
    splits: Sequence[str],
    max_files_per_class: int = 0,
) -> int:
    return sum(
        1
        for _ in _iter_latent_files(
            latent_root, splits, max_files_per_class=max_files_per_class
        )
    )


def _destination_for(source: Path, latent_root: Path, output_root: Path) -> Path:
    return (output_root / source.relative_to(latent_root)).with_suffix(".jpg")


def _is_complete_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _load_moments(path: Path) -> torch.Tensor:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load latent file {path}: {exc}") from exc
    if not isinstance(state, dict) or "moments" not in state:
        raise KeyError(f"Latent file lacks a 'moments' tensor: {path}")
    moments = torch.as_tensor(state["moments"])
    if tuple(moments.shape) != (4, 32, 32):
        raise ValueError(
            f"Expected moments shape (4, 32, 32), got {tuple(moments.shape)} in {path}"
        )
    if not moments.is_floating_point():
        raise TypeError(f"Expected floating-point moments in {path}, got {moments.dtype}")
    return moments.contiguous()


def _atomic_write_bytes(destination: Path, payload: bytes, rank: int) -> None:
    # Destination parents are normally prepared once per batch. Keep this here
    # as a safety net for a manually modified/nested source cache.
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-r{rank}-p{os.getpid()}"
    )
    try:
        with open(temporary, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, destination)
    finally:
        # A failed write must not look resumable. os.replace removes the temp
        # path on success; unlink handles only an interrupted/failed write.
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _decode_encode_write_batch(
    source_paths: Sequence[Path],
    destination_paths: Sequence[Path],
    *,
    vae: torch.nn.Module,
    scaling_factor: float,
    device: torch.device,
    jpeg_quality: int,
    rank: int,
    load_pool: concurrent.futures.ThreadPoolExecutor,
    write_pool: concurrent.futures.ThreadPoolExecutor,
) -> None:
    moments = list(load_pool.map(_load_moments, source_paths))
    latent_batch = torch.stack(moments).to(
        device=device, dtype=torch.float32, non_blocking=True
    )

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        decoded = vae.decode(
            latent_batch / scaling_factor, return_dict=False
        )[0]

    # JPEG is an 8-bit pixel representation, so explicitly use the same
    # saturating [-1, 1] -> [0, 255] conversion used by image evaluation.
    pixels_u8 = (
        decoded.float()
        .add(1.0)
        .mul(127.5)
        .round_()
        .clamp_(0.0, 255.0)
        .to(torch.uint8)
        .contiguous()
    )
    encoded_cuda = encode_jpeg(
        list(pixels_u8.unbind(0)), quality=int(jpeg_quality)
    )
    if not isinstance(encoded_cuda, list):
        encoded_cuda = [encoded_cuda]
    if len(encoded_cuda) != len(destination_paths):
        raise RuntimeError(
            f"CUDA JPEG encoder returned {len(encoded_cuda)} buffers for "
            f"{len(destination_paths)} images"
        )

    # Copy only compressed byte streams to host memory. At Q95 this is about
    # 30 KiB/image for this cache, versus 192 KiB/image for raw RGB.
    payloads = [encoded.cpu().numpy().tobytes() for encoded in encoded_cuda]
    list(
        write_pool.map(
            lambda item: _atomic_write_bytes(item[0], item[1], rank),
            zip(destination_paths, payloads),
        )
    )


def _split_status(
    latent_root: Path,
    output_root: Path,
    split: str,
    max_files_per_class: int = 0,
) -> dict[str, int | bool]:
    source_count = 0
    output_count = 0
    for source in _iter_latent_files(
        latent_root,
        (split,),
        max_files_per_class=max_files_per_class,
    ):
        source_count += 1
        if _is_complete_file(_destination_for(source, latent_root, output_root)):
            output_count += 1
    return {
        "source_count": source_count,
        "output_count": output_count,
        "missing_count": source_count - output_count,
        "complete": output_count == source_count,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-p{os.getpid()}")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    args = _parse_args()
    latent_root = args.latent_root.resolve()
    output_root = args.output_root.resolve()
    if latent_root == output_root:
        raise ValueError("--output-root must differ from --latent-root")

    rank, world_size, local_rank, device = _distributed_context()
    started_at = dt.datetime.now(dt.timezone.utc)

    source_total = _count_sources(
        latent_root,
        args.splits,
        max_files_per_class=args.max_files_per_class,
    )
    if source_total <= 0:
        raise RuntimeError(
            f"No .pt files found for splits {args.splits} under {latent_root}"
        )
    local_total = max(0, (source_total - rank + world_size - 1) // world_size)
    if args.max_files_per_rank:
        local_progress_total = min(local_total, args.max_files_per_rank)
    else:
        local_progress_total = local_total

    # Import after CUDA rank selection so each process constructs exactly one
    # frozen decoder on its assigned device.
    from vae_imagenet import load_vae

    vae = load_vae(
        device=device,
        model_id=args.model_id,
        revision=args.revision,
    )
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

    if rank == 0:
        print(
            "[pixel-cache] "
            f"source={latent_root} output={output_root} "
            f"splits={','.join(args.splits)} files={source_total:,} "
            f"world_size={world_size} batch_per_rank={args.batch_size} "
            f"jpeg_quality={args.jpeg_quality} model={args.model_id} "
            f"scaling_factor={scaling_factor:g}",
            flush=True,
        )

    processed = 0
    built = 0
    skipped = 0
    pending_sources: list[Path] = []
    pending_destinations: list[Path] = []

    progress = tqdm(
        total=local_progress_total,
        desc=f"pixel-cache rank {rank}",
        position=local_rank,
        dynamic_ncols=True,
        leave=True,
    )
    with (
        concurrent.futures.ThreadPoolExecutor(
            max_workers=args.load_workers,
            thread_name_prefix=f"latent-load-r{rank}",
        ) as load_pool,
        concurrent.futures.ThreadPoolExecutor(
            max_workers=args.write_workers,
            thread_name_prefix=f"jpeg-write-r{rank}",
        ) as write_pool,
    ):
        for global_index, source in enumerate(
            _iter_latent_files(
                latent_root,
                args.splits,
                max_files_per_class=args.max_files_per_class,
            )
        ):
            if global_index % world_size != rank:
                continue
            if args.max_files_per_rank and processed >= args.max_files_per_rank:
                break
            processed += 1
            destination = _destination_for(source, latent_root, output_root)
            if _is_complete_file(destination):
                skipped += 1
                progress.update(1)
                continue

            pending_sources.append(source)
            pending_destinations.append(destination)
            if len(pending_sources) < args.batch_size:
                continue

            _decode_encode_write_batch(
                pending_sources,
                pending_destinations,
                vae=vae,
                scaling_factor=scaling_factor,
                device=device,
                jpeg_quality=args.jpeg_quality,
                rank=rank,
                load_pool=load_pool,
                write_pool=write_pool,
            )
            built += len(pending_sources)
            progress.update(len(pending_sources))
            pending_sources.clear()
            pending_destinations.clear()

        if pending_sources:
            _decode_encode_write_batch(
                pending_sources,
                pending_destinations,
                vae=vae,
                scaling_factor=scaling_factor,
                device=device,
                jpeg_quality=args.jpeg_quality,
                rank=rank,
                load_pool=load_pool,
                write_pool=write_pool,
            )
            built += len(pending_sources)
            progress.update(len(pending_sources))

    progress.close()

    totals = torch.tensor(
        [processed, built, skipped], dtype=torch.int64, device=device
    )
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        # No rank may publish a complete manifest until every atomic JPEG write
        # from every other rank is visible.
        dist.barrier(device_ids=[local_rank])
    processed_total, built_total, skipped_total = (
        int(value) for value in totals.cpu().tolist()
    )

    if rank == 0:
        limited_run = bool(args.max_files_per_rank or args.max_files_per_class)
        split_status = {
            split: _split_status(
                latent_root,
                output_root,
                split,
                max_files_per_class=args.max_files_per_class,
            )
            for split in args.splits
        }
        complete_requested = (
            not limited_run
            and all(bool(status["complete"]) for status in split_status.values())
        )
        finished_at = dt.datetime.now(dt.timezone.utc)
        manifest = {
            "schema_version": 1,
            "source_latent_root": str(latent_root),
            "output_pixel_root": str(output_root),
            "requested_splits": list(args.splits),
            "source_key": "moments",
            "source_flip_key_decoded": False,
            "vae_model_id": args.model_id,
            "vae_revision": args.revision,
            "vae_scaling_factor": scaling_factor,
            "decode_autocast_dtype": "bfloat16",
            "pixel_conversion": "clip(round((decoded + 1) * 127.5), 0, 255)",
            "image_format": "jpeg",
            "jpeg_backend": "torchvision.io.encode_jpeg_cuda",
            "jpeg_quality": args.jpeg_quality,
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size,
            "source_count_requested": source_total,
            "processed_this_invocation": processed_total,
            "built_this_invocation": built_total,
            "skipped_existing_this_invocation": skipped_total,
            "limited_run": limited_run,
            "max_files_per_rank": args.max_files_per_rank,
            "max_files_per_class": args.max_files_per_class,
            "split_status": split_status,
            "complete_requested": complete_requested,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": (finished_at - started_at).total_seconds(),
        }
        manifest_path = output_root / "_pixel_cache_manifest.json"
        _atomic_write_json(manifest_path, manifest)
        print(
            "[pixel-cache] finished "
            f"processed={processed_total:,} built={built_total:,} "
            f"skipped={skipped_total:,} complete={complete_requested}; "
            f"manifest={manifest_path}",
            flush=True,
        )

    # Keep nonzero ranks alive until rank 0 has atomically published the
    # completion manifest.
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
