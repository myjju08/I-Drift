"""ArrayMemoryBank — class-wise ring buffer for real/fake samples.

PyTorch port of the official JAX ArrayMemoryBank.
Stores samples as numpy arrays (CPU); returns torch tensors on demand.
"""
from __future__ import annotations

import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class ArrayMemoryBank:
    """Per-class ring buffer that stores image/latent samples.

    Used during generator training to maintain a pool of real images
    (positive bank) and unconditioned images (negative bank).

    Args:
        num_classes: Number of distinct labels (1000 for ImageNet).
        max_size:    Maximum stored samples per class.
        dtype:       NumPy dtype for raw storage (default float32).
        storage_mode: ``"raw"`` preserves the original behavior.  The explicit
            ``"pixel_uint8"`` mode stores RGB pixels as uint8 and reconstructs
            the usual ImageNet ``[-1, 1]`` float32 normalization on sampling.
            It accepts either source uint8 CHW tensors directly or pixels that
            came from ``ToTensor`` plus ``Normalize(0.5, 0.5)``.  Both ingress
            paths store and sample the exact same source bytes.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        max_size: int = 64,
        dtype=np.float32,
        storage_mode: str = "raw",
    ):
        self.num_classes   = int(num_classes)
        self.max_size      = int(max_size)
        self.dtype         = np.dtype(dtype)
        self.storage_mode  = str(storage_mode).strip().lower()
        if self.storage_mode not in {"raw", "pixel_uint8"}:
            raise ValueError(
                "storage_mode must be 'raw' or 'pixel_uint8', got "
                f"{storage_mode!r}."
            )
        self.storage_dtype = (
            np.dtype(np.uint8)
            if self.storage_mode == "pixel_uint8"
            else self.dtype
        )
        self.bank: Optional[np.ndarray] = None
        self.feature_shape: Optional[Tuple[int, ...]] = None
        self.ptr   = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)

    # ------------------------------------------------------------------
    def _init_bank(self, sample_shape: Tuple[int, ...]) -> None:
        self.feature_shape = tuple(sample_shape)
        if self.storage_mode == "pixel_uint8" and (
            len(self.feature_shape) != 3 or self.feature_shape[0] != 3
        ):
            raise ValueError(
                "pixel_uint8 storage expects CHW RGB samples, got shape "
                f"{self.feature_shape}."
            )
        self.bank = np.zeros(
            (self.num_classes, self.max_size, *self.feature_shape),
            dtype=self.storage_dtype,
        )

    def _encode_samples(self, samples: np.ndarray) -> np.ndarray:
        """Convert caller values into the configured CPU storage format."""
        if self.storage_mode == "raw":
            return np.asarray(samples, dtype=self.dtype)

        source = np.asarray(samples)
        if source.dtype == np.uint8:
            # Raw ImageNet's opt-in loader already has the bank's native
            # representation.  Preserve it byte-for-byte instead of expanding
            # to float32 only to invert the normalization below.
            return source

        pixels = np.asarray(source, dtype=np.float32)
        if not np.isfinite(pixels).all():
            raise ValueError("pixel_uint8 samples must contain only finite values.")
        # Permit a tiny floating-point tolerance at the endpoints, but reject
        # accidentally unnormalised [0, 255] inputs rather than silently
        # corrupting them with clipping.
        tolerance = 1.0e-5
        pixel_min = float(pixels.min()) if pixels.size else 0.0
        pixel_max = float(pixels.max()) if pixels.size else 0.0
        if pixel_min < -1.0 - tolerance or pixel_max > 1.0 + tolerance:
            raise ValueError(
                "pixel_uint8 samples must be normalized to [-1, 1], got "
                f"range [{pixel_min:.6g}, {pixel_max:.6g}]."
            )
        # Invert ToTensor()->Normalize(0.5, 0.5).  For ordinary 8-bit source
        # images this recovers the source bytes exactly; arbitrary generated
        # floats are quantized to the nearest 8-bit pixel.
        unit = np.clip(pixels, -1.0, 1.0) * np.float32(0.5) + np.float32(0.5)
        return np.rint(unit * np.float32(255.0)).astype(np.uint8)

    def _decode_samples(
        self,
        samples: np.ndarray,
        device: Optional[torch.device] = None,
        *,
        copy_samples: bool = True,
    ) -> torch.Tensor:
        """Decode samples, copying caller-owned arrays unless ownership is known."""
        tensor = torch.from_numpy(samples.copy() if copy_samples else samples)
        if device is not None:
            # Keep the host-to-device copy compressed for pixel banks; the
            # float expansion happens only after the uint8 tensor reaches the
            # requested device.
            tensor = tensor.to(device)
        if self.storage_mode == "pixel_uint8":
            # Match torchvision's operation order: ToTensor divides uint8 by
            # 255, then Normalize subtracts 0.5 and divides by 0.5.
            tensor = tensor.float().div_(255.0).sub_(0.5).div_(0.5)
        return tensor

    # ------------------------------------------------------------------
    def add(
        self,
        samples: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
    ) -> None:
        """Insert samples into per-class ring buffers.

        Args:
            samples: (N, *feature_shape) — can be a Tensor or ndarray.
            labels:  (N,) integer class labels.
        """
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        samples = self._encode_samples(samples)
        labels  = np.asarray(labels).astype(np.int32)

        if self.bank is None:
            self._init_bank(samples.shape[1:])

        for i in range(len(labels)):
            lbl = int(labels[i])
            idx = int(self.ptr[lbl])
            self.bank[lbl, idx] = samples[i]
            self.ptr[lbl]   = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1

    # ------------------------------------------------------------------
    def sample(
        self,
        labels: torch.Tensor | np.ndarray,
        n_samples: int,
        device: Optional[torch.device] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Sample stored entries for each label.

        Args:
            labels:    (B,) integer class labels.
            n_samples: Number of samples to draw per label.
            device:    Target torch device for the returned tensor.

        Returns:
            Tensor of shape (B, n_samples, *feature_shape).
        """
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy().astype(np.int32)
        else:
            labels_np = np.asarray(labels).astype(np.int32)

        B = labels_np.shape[0]
        sample_indices = np.empty((B, n_samples), dtype=np.int32)

        for i in range(B):
            lbl   = int(labels_np[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros(n_samples, dtype=np.int32)
            else:
                choice = np.random.choice if rng is None else rng.choice
                sample_indices[i] = choice(
                    valid, n_samples, replace=(valid < n_samples)
                )

        # bank[labels, indices] => (B, n_samples, *feature_shape)
        out = self.bank[labels_np[:, None], sample_indices]
        # NumPy advanced indexing already allocates an independent uint8 array.
        # torch.from_numpy retains that allocation until the tensor dies, so
        # another host copy is unnecessary before the CUDA transfer/expansion.
        # Keep raw-mode and direct _decode_samples callers' copy behavior.
        return self._decode_samples(
            out, device=device, copy_samples=self.storage_mode != "pixel_uint8"
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.count.sum())

    def is_ready(self, min_per_class: int = 1) -> bool:
        """True if every class has at least `min_per_class` samples."""
        return bool((self.count >= min_per_class).all())

    def save_npz(self, path: str | Path) -> None:
        """Persist the bank without Python pickles.

        Historical generated replay uses one frozen per-rank snapshot.  Keeping
        it outside the model checkpoint avoids inflating every EMA checkpoint.
        """
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("Cannot save an empty MemoryBank.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            bank=self.bank,
            ptr=self.ptr,
            count=self.count,
            storage_mode=np.asarray(self.storage_mode),
        )

    def load_npz(self, path: str | Path) -> None:
        """Restore a snapshot written by :meth:`save_npz`."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as state:
            bank = np.asarray(state["bank"])
            ptr = np.asarray(state["ptr"], dtype=np.int32)
            count = np.asarray(state["count"], dtype=np.int32)
            stored_mode = (
                str(state["storage_mode"].item())
                if "storage_mode" in state
                else "raw"
            )
        if stored_mode != self.storage_mode:
            raise ValueError(
                f"Snapshot storage_mode {stored_mode!r} does not match bank "
                f"storage_mode {self.storage_mode!r}."
            )
        expected_prefix = (self.num_classes, self.max_size)
        if bank.ndim < 2 or tuple(bank.shape[:2]) != expected_prefix:
            raise ValueError(
                f"Snapshot bank shape {bank.shape} does not match {expected_prefix}."
            )
        if ptr.shape != (self.num_classes,) or count.shape != (self.num_classes,):
            raise ValueError("Snapshot pointer/count shapes do not match num_classes.")
        self.bank = np.asarray(bank, dtype=self.storage_dtype)
        self.feature_shape = tuple(self.bank.shape[2:])
        self.ptr = ptr.copy()
        self.count = count.copy()


class CompressedPixelMemoryBank(ArrayMemoryBank):
    """Exact ``pixel_uint8`` ring bank with per-image lossless compression.

    The logical bank and sampling algorithm intentionally match
    :class:`ArrayMemoryBank`.  Only the physical representation differs: each
    post-transform CHW uint8 image is filtered with a reversible horizontal
    delta predictor and stored as an independent Zstandard frame.  Incompressible
    images fall back to their original bytes. Zstandard frames carry a content
    checksum, and raw fallbacks carry an explicit CRC32, so corrupt slots fail
    closed instead of silently changing training pixels.

    Independent frames retain random slot access without touching the ImageNet
    files again.  This is important for direct-pixel training: a file-backed or
    source-reference bank would trade RAM pressure for hundreds of extra JPEG
    decodes per rank and step.
    """

    _RAW_TAG = 0
    _ZSTD_DELTA_X_TAG = 1

    def __init__(
        self,
        num_classes: int = 1000,
        max_size: int = 64,
        *,
        compression_level: int = 1,
        codec_workers: int = 1,
    ):
        super().__init__(
            num_classes=num_classes,
            max_size=max_size,
            storage_mode="pixel_uint8",
        )
        try:
            import zstandard as zstd
        except ImportError as exc:  # pragma: no cover - exercised in deployment guard
            raise RuntimeError(
                "CompressedPixelMemoryBank requires the 'zstandard' package. "
                "Install requirements.txt before launching the experiment."
            ) from exc
        self.compression_level = int(compression_level)
        self.codec_workers = int(codec_workers)
        if self.codec_workers < 1:
            raise ValueError("codec_workers must be at least 1")
        self._zstd = zstd
        self._codec_state = threading.local()
        self._executor = self._new_executor()
        self.payload_bytes = 0
        self.raw_equivalent_bytes = 0
        self.last_add_seconds = 0.0
        self.last_sample_seconds = 0.0

    def _new_executor(self) -> Optional[ThreadPoolExecutor]:
        if self.codec_workers <= 1:
            return None
        return ThreadPoolExecutor(
            max_workers=self.codec_workers,
            thread_name_prefix="pixel-bank-codec",
        )

    def suspend_codec_workers(self) -> None:
        """Join codec threads before a non-persistent DataLoader forks."""
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

    def resume_codec_workers(self) -> None:
        """Recreate the lazy codec pool after evaluation worker startup."""
        if self.codec_workers > 1 and self._executor is None:
            self._executor = self._new_executor()

    def _thread_compressor(self):
        compressor = getattr(self._codec_state, "compressor", None)
        if compressor is None:
            compressor = self._zstd.ZstdCompressor(
                level=self.compression_level, write_checksum=True
            )
            self._codec_state.compressor = compressor
        return compressor

    def _thread_decompressor(self):
        decompressor = getattr(self._codec_state, "decompressor", None)
        if decompressor is None:
            decompressor = self._zstd.ZstdDecompressor()
            self._codec_state.decompressor = decompressor
        return decompressor

    def _init_bank(self, sample_shape: Tuple[int, ...]) -> None:
        self.feature_shape = tuple(sample_shape)
        if len(self.feature_shape) != 3 or self.feature_shape[0] != 3:
            raise ValueError(
                "pixel_uint8 storage expects CHW RGB samples, got shape "
                f"{self.feature_shape}."
            )
        # Object references cost about 1 MiB for the production 128k slots;
        # image payloads are allocated only when each logical slot is written.
        self.bank = np.empty((self.num_classes, self.max_size), dtype=object)
        self.bank.fill(None)

    @staticmethod
    def _delta_x(sample: np.ndarray) -> np.ndarray:
        filtered = np.empty_like(sample)
        filtered[..., 0] = sample[..., 0]
        np.subtract(sample[..., 1:], sample[..., :-1], out=filtered[..., 1:])
        return filtered

    def _compress_sample(self, sample: np.ndarray) -> bytes:
        sample = np.ascontiguousarray(sample, dtype=np.uint8)
        filtered = self._delta_x(sample)
        compressed = self._thread_compressor().compress(filtered.tobytes(order="C"))
        if len(compressed) >= sample.nbytes:
            raw = sample.tobytes(order="C")
            checksum = zlib.crc32(raw) & 0xFFFFFFFF
            return (
                bytes((self._RAW_TAG,))
                + checksum.to_bytes(4, byteorder="little")
                + raw
            )
        return bytes((self._ZSTD_DELTA_X_TAG,)) + compressed

    def _decode_payload(self, payload: bytes, out: np.ndarray) -> None:
        expected_bytes = int(out.nbytes)
        if not payload:
            raise RuntimeError("Compressed memory-bank slot is empty or corrupt.")
        tag = payload[0]
        if tag == self._RAW_TAG:
            if len(payload) < 5:
                raise RuntimeError("Raw memory-bank slot is missing its checksum.")
            stored_checksum = int.from_bytes(payload[1:5], byteorder="little")
            decoded = payload[5:]
            if len(decoded) != expected_bytes:
                raise RuntimeError(
                    "Raw memory-bank slot has the wrong decoded length: "
                    f"expected={expected_bytes} actual={len(decoded)}."
                )
            actual_checksum = zlib.crc32(decoded) & 0xFFFFFFFF
            if actual_checksum != stored_checksum:
                raise RuntimeError("Raw memory-bank slot checksum mismatch.")
            np.copyto(out, np.frombuffer(decoded, dtype=np.uint8).reshape(out.shape))
            return
        if tag != self._ZSTD_DELTA_X_TAG:
            raise RuntimeError(f"Unknown compressed memory-bank codec tag {tag}.")
        try:
            decoded = self._thread_decompressor().decompress(
                payload[1:], max_output_size=expected_bytes
            )
        except Exception as exc:
            raise RuntimeError("Could not decompress memory-bank slot.") from exc
        if len(decoded) != expected_bytes:
            raise RuntimeError(
                "Compressed memory-bank slot has the wrong decoded length: "
                f"expected={expected_bytes} actual={len(decoded)}."
            )
        filtered = np.frombuffer(decoded, dtype=np.uint8).reshape(out.shape)
        np.add.accumulate(filtered, axis=-1, dtype=np.uint8, out=out)

    def add(
        self,
        samples: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
    ) -> None:
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        samples = self._encode_samples(samples)
        labels = np.asarray(labels).astype(np.int32)
        if self.bank is None:
            self._init_bank(samples.shape[1:])
        elif tuple(samples.shape[1:]) != self.feature_shape:
            raise ValueError(
                "Memory-bank sample shape changed after initialization: "
                f"expected={self.feature_shape} actual={tuple(samples.shape[1:])}."
            )
        assert self.bank is not None

        started = time.perf_counter()
        if self._executor is None:
            payloads = [self._compress_sample(samples[i]) for i in range(len(labels))]
        else:
            payloads = list(
                self._executor.map(
                    self._compress_sample,
                    (samples[i] for i in range(len(labels))),
                )
            )
        sample_bytes = int(np.prod(self.feature_shape, dtype=np.int64))
        for i, payload in enumerate(payloads):
            lbl = int(labels[i])
            idx = int(self.ptr[lbl])
            previous = self.bank[lbl, idx]
            self.bank[lbl, idx] = payload
            self.payload_bytes += len(payload) - (
                len(previous) if previous is not None else 0
            )
            if previous is None:
                self.raw_equivalent_bytes += sample_bytes
            self.ptr[lbl] = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1
        self.last_add_seconds = time.perf_counter() - started

    def _decode_task(self, task: Tuple[bytes, np.ndarray]) -> None:
        payload, destination = task
        self._decode_payload(payload, destination)

    def sample(
        self,
        labels: torch.Tensor | np.ndarray,
        n_samples: int,
        device: Optional[torch.device] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy().astype(np.int32)
        else:
            labels_np = np.asarray(labels).astype(np.int32)

        batch_size = labels_np.shape[0]
        sample_indices = np.empty((batch_size, n_samples), dtype=np.int32)
        # Keep this loop and choice call order identical to ArrayMemoryBank.
        for i in range(batch_size):
            lbl = int(labels_np[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros(n_samples, dtype=np.int32)
            else:
                choice = np.random.choice if rng is None else rng.choice
                sample_indices[i] = choice(
                    valid, n_samples, replace=(valid < n_samples)
                )

        out = np.empty(
            (batch_size, n_samples, *self.feature_shape), dtype=np.uint8
        )
        started = time.perf_counter()
        # Duplicate draws are common while a class is filling. Decode the first
        # occurrence once and copy its exact bytes into later output positions.
        decoded_slots: Dict[Tuple[int, int], Tuple[int, int]] = {}
        decode_tasks = []
        duplicate_positions = []
        for i in range(batch_size):
            lbl = int(labels_np[i])
            for j in range(n_samples):
                idx = int(sample_indices[i, j])
                key = (lbl, idx)
                source_position = decoded_slots.get(key)
                if source_position is not None:
                    duplicate_positions.append(((i, j), source_position))
                    continue
                payload = self.bank[lbl, idx]
                if payload is None:
                    if int(self.count[lbl]) <= 0:
                        out[i, j].fill(0)
                        decoded_slots[key] = (i, j)
                        continue
                    raise RuntimeError(
                        f"Memory-bank slot label={lbl} index={idx} is uninitialized."
                    )
                decode_tasks.append((payload, out[i, j]))
                decoded_slots[key] = (i, j)

        if self._executor is None:
            for task in decode_tasks:
                self._decode_task(task)
        else:
            list(self._executor.map(self._decode_task, decode_tasks))
        for destination, source in duplicate_positions:
            np.copyto(out[destination], out[source])
        self.last_sample_seconds = time.perf_counter() - started

        tensor = torch.from_numpy(out)
        if device is not None:
            tensor = tensor.to(device)
        return tensor.float().div_(255.0).sub_(0.5).div_(0.5)

    @property
    def compression_ratio(self) -> float:
        if self.payload_bytes <= 0:
            return 1.0
        return float(self.raw_equivalent_bytes) / float(self.payload_bytes)

    def save_npz(self, path: str | Path) -> None:
        raise NotImplementedError(
            "CompressedPixelMemoryBank snapshots are not used by the positive "
            "training bank; use the dense bank for persisted historical replay."
        )

    def load_npz(self, path: str | Path) -> None:
        raise NotImplementedError(
            "CompressedPixelMemoryBank snapshots are not supported."
        )


__all__ = ["ArrayMemoryBank", "CompressedPixelMemoryBank"]
