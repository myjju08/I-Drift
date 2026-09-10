#!/usr/bin/env python3
"""Packed, resumable ImageNet SD-VAE sample cache and compatible trainer loader.

No training or CUDA work happens on import. ``prepare`` and ``finalize`` are CPU
operations; only an explicit ``build`` command loads the VAE on the chosen GPU.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

EXPECTED_COUNTS = {"train": 1281167, "val": 50000}
VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
SCHEMA = 1


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


@contextmanager
def _lock(path, *, wait=False):
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}") from exc
        yield


def _load_index(root):
    root = Path(root)
    data = json.loads((root / "index_manifest.json").read_text())
    if data.get("schema") != SCHEMA or data.get("expected_counts") != EXPECTED_COUNTS:
        raise RuntimeError("Cache index schema or required ImageNet counts differ")
    for split, count in EXPECTED_COUNTS.items():
        info = data["splits"][split]
        if info["count"] != count or len(info["classes"]) != 1000:
            raise RuntimeError(f"Incomplete ImageNet index: {split}")
        for name, expected in info["index_sha256"].items():
            if _sha(root / split / name) != expected:
                raise RuntimeError(f"Cache index changed: {split}/{name}")
    return data


def prepare(raw_root, cache_root, chunk_size=2048):
    """Index the complete raw ImageFolder once, preserving its sorted order."""
    from torchvision.datasets import ImageFolder

    raw_root, root = Path(raw_root).resolve(), Path(cache_root).resolve()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root / ".prepare.lock"):
        if (root / "index_manifest.json").exists():
            previous = _load_index(root)
            if previous["raw_root"] != str(raw_root) or previous["chunk_size"] != chunk_size:
                raise RuntimeError("Existing index uses different source or chunk size")
            return previous
        result = {"schema": SCHEMA, "created_utc": _now(), "raw_root": str(raw_root),
                  "expected_counts": EXPECTED_COUNTS, "chunk_size": chunk_size, "splits": {}}
        reference_classes = None
        for split, expected in EXPECTED_COUNTS.items():
            dataset = ImageFolder(str(raw_root / split), loader=str)
            if len(dataset) != expected or len(dataset.classes) != 1000:
                raise RuntimeError(f"Incomplete {split}: {len(dataset)} images, {len(dataset.classes)} classes")
            if reference_classes is not None and dataset.classes != reference_classes:
                raise RuntimeError("Train/validation class mapping differs")
            reference_classes = dataset.classes
            dest = root / split
            (dest / "chunks").mkdir(parents=True, exist_ok=True)
            labels = np.asarray(dataset.targets, dtype=np.int64)
            np.save(dest / "labels.npy", labels, allow_pickle=False)
            offsets = np.zeros(len(dataset) + 1, dtype=np.int64)
            with open(dest / "paths.txt", "wb") as stream:
                for i, (source, _) in enumerate(dataset.samples):
                    relative = Path(source).relative_to(raw_root / split).as_posix()
                    if "\n" in relative:
                        raise RuntimeError("Newline in ImageNet path")
                    stream.write(relative.encode("utf-8") + b"\n")
                    offsets[i + 1] = stream.tell()
            np.save(dest / "path_offsets.npy", offsets, allow_pickle=False)
            result["splits"][split] = {
                "count": len(dataset), "classes": dataset.classes,
                "class_to_idx": dataset.class_to_idx,
                "index_sha256": {name: _sha(dest / name) for name in
                                  ("labels.npy", "paths.txt", "path_offsets.npy")},
                "class_counts": np.bincount(labels, minlength=1000).tolist(),
            }
            print(json.dumps({"event": "indexed", "split": split, "count": len(dataset)}), flush=True)
        _json(root / "index_manifest.json", result)
        return result


def _chunks(index, split):
    size, count = index["chunk_size"], index["splits"][split]["count"]
    return [(number, start, min(start + size, count))
            for number, start in enumerate(range(0, count, size))]


def _chunk_paths(root, split, number):
    base = Path(root) / split / "chunks" / f"chunk_{number:06d}"
    return base.with_suffix(".npy"), base.with_suffix(".json")


def _valid_chunk(root, split, item, recipe_sha, verify_hash=False):
    number, start, end = item
    path, done = _chunk_paths(root, split, number)
    if not done.exists():
        return False
    marker = json.loads(done.read_text())
    if marker.get("recipe_sha256") != recipe_sha or marker.get("range") != [start, end]:
        raise RuntimeError(f"Chunk identity differs: {done}")
    if not path.is_file() or path.stat().st_size != marker["bytes"]:
        raise RuntimeError(f"Completed chunk missing/truncated: {path}")
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.shape != (end - start, 2, 4, 32, 32) or data.dtype != np.float32:
        raise RuntimeError(f"Wrong latent shape/dtype: {path}")
    if verify_hash and _sha(path) != marker["sha256"]:
        raise RuntimeError(f"Completed chunk checksum differs: {path}")
    return True


class _RawIndexedDataset(Dataset):
    def __init__(self, root, split, index):
        from torchvision import transforms
        from train.train_data import _center_crop

        self.root = Path(root) / split
        self.raw_root = Path(index["raw_root"]) / split
        self.offsets = np.load(self.root / "path_offsets.npy", mmap_mode="r")
        self.count = index["splits"][split]["count"]
        self.fd = None
        self.transform = transforms.Compose([
            transforms.Lambda(lambda image: _center_crop(image, 256)),
            transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        from torchvision.datasets.folder import default_loader

        if self.fd is None:
            self.fd = os.open(self.root / "paths.txt", os.O_RDONLY)
        start, stop = int(self.offsets[item]), int(self.offsets[item + 1])
        relative = os.pread(self.fd, stop - start, start).decode("utf-8").rstrip("\n")
        return self.transform(default_loader(self.raw_root / relative)), item


class _ChunkBatches:
    def __init__(self, chunks, batch_size):
        self.chunks, self.batch_size = chunks, batch_size

    def __iter__(self):
        for _, start, end in self.chunks:
            for pos in range(start, end, self.batch_size):
                yield list(range(pos, min(pos + self.batch_size, end)))

    def __len__(self):
        return sum((end - start + self.batch_size - 1) // self.batch_size
                   for _, start, end in self.chunks)


def build(cache_root, vae_path, rank=0, world_size=1, device="cuda:0",
          batch_size=32, num_workers=4, seed=43, max_chunks=0):
    """Encode disjoint chunk numbers, restarting safely after completed chunks."""
    from diffusers import AutoencoderKL

    root, vae_path = Path(cache_root).resolve(), Path(vae_path).resolve()
    if not (0 <= rank < world_size) or batch_size <= 0 or num_workers < 0:
        raise ValueError("Invalid rank, world_size, batch_size, or num_workers")
    index = _load_index(root)
    weight_files = sorted(p for p in vae_path.rglob("*")
                          if p.is_file() and (p.suffix in {".safetensors", ".bin"} or p.name == "config.json"))
    if not weight_files or not any(p.suffix in {".safetensors", ".bin"} for p in weight_files):
        raise RuntimeError(f"Local pinned VAE weights missing: {vae_path}")
    recipe = {"schema": SCHEMA, "index_sha256": _sha(root / "index_manifest.json"),
              "vae_revision": VAE_REVISION,
              "vae_sha256": {p.relative_to(vae_path).as_posix(): _sha(p) for p in weight_files},
              "seed": seed, "batch_size": batch_size, "dtype": "float32", "scale": .18215,
              "posterior": "sample", "views": ["clean", "horizontal_flip"],
              "preprocessing": "official RGB -> ADM center crop256 -> float32 [-1,1]",
              "rng": "dedicated CUDA generator, seed=(seed+1000003*split_id+chunk_id) modulo2**63",
              "cudnn_allow_tf32": True, "matmul_allow_tf32": False,
              "cudnn_benchmark": False}
    # All workers use the same recipe; CPU/GPU allocations and world_size are operational only.
    with _lock(root / ".recipe.lock", wait=True):
        if (root / "encoding_recipe.json").exists():
            if json.loads((root / "encoding_recipe.json").read_text()) != recipe:
                raise RuntimeError("Existing cache was encoded with a different immutable recipe")
        else:
            _json(root / "encoding_recipe.json", recipe)
    recipe_sha = _sha(root / "encoding_recipe.json")
    torch.set_num_threads(1)
    # Match the original vae_imagenet.build_latent_cache process defaults:
    # FP32 tensors, TF32-eligible cuDNN convolutions, IEEE FP32 matmuls, no
    # convolution benchmarking. These flags are pinned in the cache recipe.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(device)
    vae = AutoencoderKL.from_pretrained(str(vae_path), local_files_only=True,
                                       torch_dtype=torch.float32).eval().to(device)
    vae.requires_grad_(False)
    completed_this_process = 0
    progress_path = root / f"progress_rank{rank:02d}.json"
    with _lock(root / f".build_rank{rank:02d}.lock"):
        for split_id, split in enumerate(EXPECTED_COUNTS):
            assigned = [item for item in _chunks(index, split) if item[0] % world_size == rank]
            pending = [item for item in assigned
                       if not _valid_chunk(root, split, item, recipe_sha, verify_hash=True)]
            if max_chunks:
                pending = pending[:max(0, max_chunks - completed_this_process)]
            if not pending:
                continue
            dataset = _RawIndexedDataset(root, split, index)
            loader = DataLoader(dataset, batch_sampler=_ChunkBatches(pending, batch_size),
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=num_workers > 0,
                                prefetch_factor=2 if num_workers else None)
            batches = iter(loader)
            for item in pending:
                number, start, end = item
                path, done = _chunk_paths(root, split, number)
                with _lock(done.with_suffix(".lock")):
                    partial = path.with_name(path.name + f".partial.{os.getpid()}")
                    array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                                      shape=(end - start, 2, 4, 32, 32))
                    rng = torch.Generator(device=device)
                    rng.manual_seed((seed + 1000003 * split_id + number) % (2 ** 63))
                    for pos in range(start, end, batch_size):
                        images, indices = next(batches)
                        stop = min(pos + batch_size, end)
                        if indices.tolist() != list(range(pos, stop)):
                            raise RuntimeError("Raw loader order differs from indexed chunk")
                        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                        with torch.no_grad():
                            for view in (0, 1):
                                pixels = images if view == 0 else images.flip(-1)
                                latent = vae.encode(pixels).latent_dist.sample(generator=rng) * .18215
                                if tuple(latent.shape) != (stop - pos, 4, 32, 32) or not torch.isfinite(latent).all():
                                    raise RuntimeError("VAE produced nonfinite or wrongly shaped latent samples")
                                array[pos - start:stop - start, view] = latent.cpu().numpy()
                        del images, latent, pixels
                    array.flush()
                    del array
                    os.replace(partial, path)
                    marker = {"schema": SCHEMA, "range": [start, end], "split": split,
                              "recipe_sha256": recipe_sha, "bytes": path.stat().st_size,
                              "sha256": _sha(path), "completed_utc": _now()}
                    _json(done, marker)
                completed_this_process += 1
                progress = {"rank": rank, "world_size": world_size, "split": split,
                            "chunk": number, "images_in_chunk": end - start,
                            "completed_this_process": completed_this_process, "updated_utc": _now()}
                _json(progress_path, progress)
                print(json.dumps(progress), flush=True)
            del batches, loader, dataset
            if max_chunks and completed_this_process >= max_chunks:
                break


def finalize(cache_root):
    """Publish completion only after every required chunk passes its checksum."""
    root = Path(cache_root)
    with _lock(root / ".finalize.lock"):
        index = _load_index(root)
        recipe_sha = _sha(root / "encoding_recipe.json")
        chunks = []
        for split in EXPECTED_COUNTS:
            for item in _chunks(index, split):
                if not _valid_chunk(root, split, item, recipe_sha, verify_hash=True):
                    raise RuntimeError(f"Cannot finalize partial cache: {split} chunk {item[0]}")
                path, done = _chunk_paths(root, split, item[0])
                chunks.append({"path": path.relative_to(root).as_posix(),
                               "marker": done.relative_to(root).as_posix(),
                               "marker_sha256": _sha(done), "bytes": path.stat().st_size})
        manifest = {"schema": SCHEMA, "complete": True, "completed_utc": _now(),
                    "expected_counts": EXPECTED_COUNTS,
                    "index_sha256": _sha(root / "index_manifest.json"),
                    "recipe_sha256": recipe_sha, "chunks": chunks,
                    "latent_shape": [2, 4, 32, 32], "dtype": "float32"}
        _json(root / "complete_manifest.json", manifest)
        return manifest


def validate_complete(cache_root):
    """Fail closed on partial, changed, missing, or truncated cache artifacts."""
    root = Path(cache_root)
    manifest_path = root / "complete_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Cache is not complete: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("complete") is not True
            or manifest.get("expected_counts") != EXPECTED_COUNTS):
        raise RuntimeError("Invalid complete cache manifest")
    if (_sha(root / "index_manifest.json") != manifest["index_sha256"]
            or _sha(root / "encoding_recipe.json") != manifest["recipe_sha256"]):
        raise RuntimeError("Completed cache identity changed")
    index = _load_index(root)
    expected_paths = {_chunk_paths(root, split, item[0])[0].relative_to(root).as_posix()
                      for split in EXPECTED_COUNTS for item in _chunks(index, split)}
    if {chunk["path"] for chunk in manifest["chunks"]} != expected_paths:
        raise RuntimeError("Completed cache does not cover every indexed image")
    for chunk in manifest["chunks"]:
        path, marker = root / chunk["path"], root / chunk["marker"]
        if (not path.is_file() or path.stat().st_size != chunk["bytes"]
                or _sha(marker) != chunk["marker_sha256"]):
            raise RuntimeError(f"Completed cache chunk changed: {path}")
    return index


class PackedLatentDataset(Dataset):
    """Read the same clean/flip draw as the original .pt DatasetFolder loader."""
    def __init__(self, cache_root, split, *, validated_index=None, max_open_chunks=128):
        self.root, self.split = Path(cache_root), split
        self.index = validated_index if validated_index is not None else validate_complete(self.root)
        info = self.index["splits"][split]
        self.classes, self.class_to_idx = info["classes"], info["class_to_idx"]
        self.targets = np.load(self.root / split / "labels.npy", mmap_mode="r")
        self.count, self.chunk_size = info["count"], self.index["chunk_size"]
        self.maps = OrderedDict()
        self.max_open_chunks = max_open_chunks

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        number, offset = divmod(int(item), self.chunk_size)
        if number not in self.maps:
            path, _ = _chunk_paths(self.root, self.split, number)
            self.maps[number] = np.load(path, mmap_mode="r", allow_pickle=False)
            if len(self.maps) > self.max_open_chunks:
                self.maps.popitem(last=False)
        self.maps.move_to_end(number)
        view = 0 if torch.rand(1).item() < .5 else 1
        # A writable copy prevents collate from exposing readonly mmap storage.
        return np.array(self.maps[number][offset, view], copy=True), int(self.targets[item])


def make_create_imagenet_split(original_create, cache_root):
    """Return a drop-in trainer hook; only use_cache=True is intercepted."""
    def create(**kwargs):
        if not kwargs.get("use_cache", False):
            return original_create(**kwargs)
        from train.train_data import _worker_init_fn
        from vae_imagenet import get_vae_enc_dec

        if not kwargs.get("use_latent", False):
            raise ValueError("Packed MAE cache requires use_latent=True")
        root = Path(cache_root)
        requested = kwargs.get("cache_path")
        if requested and Path(requested).resolve() != root.resolve():
            raise ValueError("Requested cache_path differs from the pinned packed cache")
        split = kwargs.get("split", "train")
        dataset = PackedLatentDataset(root, split)
        rank, world_size = kwargs.get("rank", 0), kwargs.get("world_size", 1)
        sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                       shuffle=split == "train")
                   if kwargs.get("distributed", False) else None)
        workers = kwargs.get("num_workers", 8)
        persistent = kwargs.get("persistent_workers")
        loader = DataLoader(dataset, batch_size=kwargs.get("batch_size", 256),
                            shuffle=sampler is None and split == "train",
                            drop_last=split == "train", sampler=sampler, num_workers=workers,
                            prefetch_factor=kwargs.get("prefetch_factor", 2) if workers else None,
                            pin_memory=kwargs.get("pin_memory", True),
                            persistent_workers=workers > 0 if persistent is None else bool(persistent) and workers > 0,
                            worker_init_fn=functools.partial(_worker_init_fn, rank=rank))

        def preprocess(batch):
            images, labels = batch
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            return {"images": images.float(), "labels": labels}

        decoder = {}

        def postprocess(latents):
            if decoder.get("device") != latents.device:
                _, decoder["decode"] = get_vae_enc_dec(
                    latents.device, model_id=kwargs.get("vae_model_id", "stabilityai/sd-vae-ft-mse"),
                    revision=kwargs.get("vae_revision"))
                decoder["device"] = latents.device
            pixels = decoder["decode"](latents.to(dtype=torch.float32, non_blocking=True))
            return ((pixels + 1) / 2).clamp(0, 1)

        return loader, preprocess, postprocess
    return create


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("prepare")
    index.add_argument("--raw-root", required=True)
    index.add_argument("--cache-root", required=True)
    index.add_argument("--chunk-size", type=int, default=2048)
    encode = commands.add_parser("build")
    encode.add_argument("--cache-root", required=True)
    encode.add_argument("--vae-path", required=True)
    encode.add_argument("--rank", type=int, default=0)
    encode.add_argument("--world-size", type=int, default=1)
    encode.add_argument("--device", default="cuda:0")
    encode.add_argument("--batch-size", type=int, default=32)
    encode.add_argument("--num-workers", type=int, default=4)
    encode.add_argument("--seed", type=int, default=43)
    encode.add_argument("--max-chunks", type=int, default=0)
    finish = commands.add_parser("finalize")
    finish.add_argument("--cache-root", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--cache-root", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = {"prepare": prepare, "build": build, "finalize": finalize,
              "verify": validate_complete}[command](**args)
    if result is not None:
        print(json.dumps({"command": command, "ok": True, "cache_root": args["cache_root"]}), flush=True)


if __name__ == "__main__":
    main()
