"""Byte-preserving loading of the existing flat ImageNet latent files.

No VAE encoding, rescaling, dtype conversion, or augmentation is performed.
The source's numeric row order includes any padded/repeated rows it contains.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


LATENT_SHAPE = (4, 32, 32)
FEATURE_DIRECTORY = "imagenet256_features"
LABEL_DIRECTORY = "imagenet256_labels"
LATENT_CACHE_RECIPE = "flat-npy-numeric-order-float32-bitwise-v1"


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _directory_signature(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_mtime_ns]


def _file_signature(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def _validate_relocation(cache_path, source_root, metadata):
    """Validate an explicitly relocated, fully checksum-verified mmap copy.

    The original packing manifest stays unchanged. The relocation certificate
    binds its payload hashes to local file identities and a small collection
    of authentic source rows used for the existing startup bitwise audit.
    """
    path = cache_path / "relocation.json"
    certificate = json.loads(path.read_text())
    rows = sorted(set(np.linspace(0, int(metadata["count"]) - 1,
                                  min(32, int(metadata["count"])), dtype=np.int64).tolist()))
    expected = {
        "schema_version": 1,
        "kind": "verified_exact_latent_cache_relocation",
        "original_metadata_sha256": _sha256(cache_path / "metadata.json"),
        "original_source_root": metadata["source_root"],
        "source_evidence_root": source_root,
        "source_evidence_rows": rows,
        "source_evidence_is_full_dataset": False,
    }
    if any(certificate.get(key) != value for key, value in expected.items()):
        raise ValueError("Relocated latent cache provenance does not match its manifest/source evidence")
    files = certificate.get("files", {})
    if set(files) != {"latents.npy", "labels.npy", "source_stats.npy"}:
        raise ValueError("Relocated latent cache requires all three verified payloads")
    for name, evidence in files.items():
        if (evidence.get("sha256") != metadata["files"][name]["sha256"]
                or evidence.get("signature") != _file_signature(cache_path / name)):
            raise ValueError(f"Relocated latent payload changed after full checksum verification: {name}")
    return certificate


def flat_latent_count(source_root):
    """Verify both explicitly selected source directories have all numeric IDs."""
    source_root = Path(source_root).resolve(strict=True)
    counts = []
    for name in (FEATURE_DIRECTORY, LABEL_DIRECTORY):
        directory = source_root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Latent source directory missing: {directory}")
        indices = []
        with os.scandir(directory) as entries:
            for entry in entries:
                stem, extension = os.path.splitext(entry.name)
                if extension != ".npy" or not stem.isdecimal() or str(int(stem)) != stem:
                    raise ValueError(f"Unexpected flat latent source entry: {entry.path}")
                if not entry.is_file():
                    raise ValueError(f"Flat latent source entry is not a file: {entry.path}")
                indices.append(int(stem))
        indices.sort()
        if not indices or any(index != expected for expected, index in enumerate(indices)):
            raise ValueError(f"Latent IDs must be contiguous 0..N-1: {directory}")
        counts.append(len(indices))
    if counts[0] != counts[1]:
        raise ValueError(f"Feature/label source counts differ: {counts}")
    return counts[0]


def _read_source_array(path, *, shape, dtype):
    with Path(path).open("rb") as handle:
        before = os.fstat(handle.fileno())
        value = np.load(handle, allow_pickle=False)
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Latent source changed while being read: {path}")
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"Invalid latent source {path}: shape={value.shape}, dtype={value.dtype}; expected {shape}/{dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite values in latent source: {path}")
    return value, (before.st_size, before.st_mtime_ns)


def read_flat_latent_row(source_root, index, *, num_classes=1000):
    root = Path(source_root)
    feature, feature_stat = _read_source_array(
        root / FEATURE_DIRECTORY / f"{index}.npy", shape=(1, *LATENT_SHAPE), dtype=np.dtype("float32"),
    )
    label, label_stat = _read_source_array(
        root / LABEL_DIRECTORY / f"{index}.npy", shape=(1,), dtype=np.dtype("int64"),
    )
    label = int(label[0])
    if not 0 <= label < num_classes:
        raise ValueError(f"Label outside [0,{num_classes}): row={index}, label={label}")
    return feature[0], label, (*feature_stat, *label_stat)


class FlatNpyLatentDataset(Dataset):
    """Direct source loader, chiefly for audits and comparison with the mmap."""

    def __init__(self, source_root, *, num_classes=1000):
        self.root = str(Path(source_root).resolve(strict=True))
        self.num_classes = int(num_classes)
        self.count = flat_latent_count(self.root)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        feature, label, _ = read_flat_latent_row(self.root, index, num_classes=self.num_classes)
        return torch.from_numpy(feature), label


class MmapLatentDataset(Dataset):
    """Read-only exact FP32 cache with a completed, source-bound manifest."""

    def __init__(self, cache_path, *, source_root, num_classes=1000):
        self.cache_path = Path(cache_path).resolve(strict=True)
        self.root = str(Path(source_root).resolve(strict=True))
        metadata_path = self.cache_path / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Completed latent mmap metadata missing; incomplete caches cannot be used")
        self.metadata = json.loads(metadata_path.read_text())
        metadata = self.metadata
        if (metadata.get("schema_version") != 1 or metadata.get("complete") is not True
                or metadata.get("recipe") != LATENT_CACHE_RECIPE
                or metadata.get("num_classes") != int(num_classes)):
            raise ValueError("Latent mmap manifest is incomplete or source/recipe/classes do not match")
        self.count = int(metadata["count"])
        self.num_classes = int(num_classes)
        if self.count <= 0 or len(metadata.get("label_counts", [])) != self.num_classes:
            raise ValueError("Invalid latent mmap count or label inventory")
        self.relocation = None
        if (self.cache_path / "relocation.json").is_file():
            self.relocation = _validate_relocation(self.cache_path, self.root, metadata)
            source_directories = self.relocation.get("source_directories", {})
        else:
            if metadata.get("source_root") != self.root:
                raise ValueError("Latent mmap source does not match its original manifest")
            source_directories = metadata["source_directories"]
        for directory in (FEATURE_DIRECTORY, LABEL_DIRECTORY):
            if _directory_signature(Path(self.root) / directory) != source_directories.get(directory):
                raise ValueError(f"Latent source directory inventory changed: {directory}")
        self._features = None
        self._labels = None
        for name, shape, dtype in (
            ("latents.npy", (self.count, *LATENT_SHAPE), np.dtype("float32")),
            ("labels.npy", (self.count,), np.dtype("int64")),
            ("source_stats.npy", (self.count, 4), np.dtype("int64")),
        ):
            path = self.cache_path / name
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous
                    or path.stat().st_size != metadata["files"][name]["bytes"]):
                raise ValueError(f"Latent mmap shape/dtype/size mismatch: {name}")
            del array
        # Small manifests are hashed at each startup; the ~20 GiB payload hash
        # is recorded at build time and can be verified independently if needed.
        for name in ("labels.npy", "source_stats.npy"):
            if _sha256(self.cache_path / name) != metadata["files"][name]["sha256"]:
                raise ValueError(f"Latent mmap inventory checksum mismatch: {name}")
        labels = np.load(self.cache_path / "labels.npy", mmap_mode="r", allow_pickle=False)
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError("Latent mmap labels out of range")
        if np.bincount(labels, minlength=self.num_classes).tolist() != metadata["label_counts"]:
            raise ValueError("Latent mmap label counts do not match metadata")
        stats = np.load(self.cache_path / "source_stats.npy", mmap_mode="r", allow_pickle=False)
        for index in sorted(set(np.linspace(0, self.count - 1, min(32, self.count), dtype=np.int64).tolist())):
            current = []
            for directory in (FEATURE_DIRECTORY, LABEL_DIRECTORY):
                stat = (Path(self.root) / directory / f"{index}.npy").stat()
                current.extend((stat.st_size, stat.st_mtime_ns))
            if current != stats[index].tolist():
                raise ValueError(f"Latent source sample changed since packing: row {index}")

    def __len__(self):
        return self.count

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = None
        state["_labels"] = None
        return state

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if self._features is None:
            self._features = np.load(self.cache_path / "latents.npy", mmap_mode="r", allow_pickle=False)
            self._labels = np.load(self.cache_path / "labels.npy", mmap_mode="r", allow_pickle=False)
        # Own a writable row; DataLoader/callers cannot mutate shared storage.
        return torch.from_numpy(self._features[index].copy()), int(self._labels[index])
