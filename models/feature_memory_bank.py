"""CPU feature-map ring bank for frozen teacher encoders.

The ordinary :class:`memory_bank.ArrayMemoryBank` stores VAE latents in NumPy
arrays.  DINO/MoCo training can avoid repeatedly decoding those real latents by
storing the frozen teacher's selected raw feature maps instead.  PyTorch owns
the storage here because NumPy has no native bfloat16 dtype; keeping bfloat16
on CPU preserves the teacher output bits exactly across add/sample round trips.

This class deliberately stores *raw maps*, not derived mean/std/patch
statistics.  Those statistics can be reconstructed after sampling while the
larger VAE + ResNet forward is reused.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict, Optional, Tuple

import numpy as np
import torch


MapShapes = Mapping[str, Sequence[int]]
FeatureMaps = Mapping[str, torch.Tensor]


class FeatureMemoryBank:
    """Per-class CPU ring buffer for dictionaries of feature maps.

    Every map shares the same class/slot coordinates and therefore the same
    ``ptr`` and ``count`` arrays.  Storage is allocated lazily on the first
    :meth:`add`, unless map metadata was supplied to the constructor (metadata
    is still validated on the first add).

    Args:
        num_classes: Number of label-specific ring buffers.
        max_size: Maximum entries per class.
        map_shapes: Optional expected ``{map_name: sample_shape}`` metadata.
        dtype: CPU storage dtype.  ``torch.bfloat16`` is the exact-cache mode
            used for BF16 teacher outputs.
    """

    def __init__(
        self,
        num_classes: int,
        max_size: int,
        *,
        map_shapes: Optional[MapShapes] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, got {dtype!r}")

        self.dtype = dtype
        self.ptr = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)
        self.bank: Dict[str, torch.Tensor] = {}
        self._map_names: Tuple[str, ...] = ()
        self._map_shapes: Dict[str, Tuple[int, ...]] = {}

        if map_shapes is not None:
            names, shapes = self._normalize_metadata(map_shapes)
            self._map_names = names
            self._map_shapes = shapes

    @staticmethod
    def _normalize_metadata(
        map_shapes: MapShapes,
    ) -> Tuple[Tuple[str, ...], Dict[str, Tuple[int, ...]]]:
        if not isinstance(map_shapes, Mapping) or not map_shapes:
            raise ValueError("map_shapes must be a non-empty mapping")

        names = tuple(str(name) for name in map_shapes)
        if any(not name for name in names):
            raise ValueError("feature map names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("feature map names must be unique")

        shapes: Dict[str, Tuple[int, ...]] = {}
        for normalized_name, original_name in zip(names, map_shapes):
            raw_shape = map_shapes[original_name]
            if isinstance(raw_shape, torch.Size):
                shape = tuple(raw_shape)
            else:
                try:
                    shape = tuple(int(dim) for dim in raw_shape)
                except TypeError as exc:
                    raise TypeError(
                        f"shape for map {normalized_name!r} must be a sequence"
                    ) from exc
            if not shape or any(dim <= 0 for dim in shape):
                raise ValueError(
                    f"shape for map {normalized_name!r} must contain only "
                    f"positive dimensions, got {shape}"
                )
            shapes[normalized_name] = shape
        return names, shapes

    @property
    def map_names(self) -> Tuple[str, ...]:
        """Map names in their stable insertion order."""
        return self._map_names

    @property
    def map_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """A copy of the per-sample shape metadata."""
        return dict(self._map_shapes)

    def _metadata_from_features(
        self,
        features: FeatureMaps,
    ) -> Tuple[Tuple[str, ...], Dict[str, Tuple[int, ...]], int]:
        if not isinstance(features, Mapping) or not features:
            raise ValueError("features must be a non-empty mapping")

        names = tuple(str(name) for name in features)
        if any(not name for name in names):
            raise ValueError("feature map names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("feature map names must be unique")

        shapes: Dict[str, Tuple[int, ...]] = {}
        batch_size: Optional[int] = None
        for normalized_name, original_name in zip(names, features):
            value = features[original_name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"feature map {normalized_name!r} must be a torch.Tensor"
                )
            if value.ndim < 2:
                raise ValueError(
                    f"feature map {normalized_name!r} must have shape "
                    "(batch, ...sample_shape)"
                )
            if batch_size is None:
                batch_size = int(value.shape[0])
            elif int(value.shape[0]) != batch_size:
                raise ValueError("all feature maps must have the same batch size")
            shape = tuple(int(dim) for dim in value.shape[1:])
            if any(dim <= 0 for dim in shape):
                raise ValueError(
                    f"feature map {normalized_name!r} has invalid sample shape {shape}"
                )
            shapes[normalized_name] = shape

        assert batch_size is not None
        return names, shapes, batch_size

    def _validate_or_initialize(
        self,
        features: FeatureMaps,
    ) -> Tuple[Tuple[str, ...], int]:
        names, shapes, batch_size = self._metadata_from_features(features)

        if self._map_names:
            if names != self._map_names:
                raise ValueError(
                    "feature map names/order do not match bank metadata: "
                    f"expected {self._map_names}, got {names}"
                )
            if shapes != self._map_shapes:
                raise ValueError(
                    "feature map shapes do not match bank metadata: "
                    f"expected {self._map_shapes}, got {shapes}"
                )
        else:
            self._map_names = names
            self._map_shapes = shapes

        if not self.bank:
            self.bank = {
                name: torch.empty(
                    (self.num_classes, self.max_size, *self._map_shapes[name]),
                    dtype=self.dtype,
                    device="cpu",
                )
                for name in self._map_names
            }
        return names, batch_size

    @staticmethod
    def _labels_numpy(
        labels: torch.Tensor | np.ndarray | Sequence[int],
    ) -> np.ndarray:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().to(device="cpu").numpy()
        result = np.asarray(labels)
        if result.ndim != 1:
            raise ValueError(f"labels must be one-dimensional, got {result.shape}")
        if not np.issubdtype(result.dtype, np.integer):
            raise TypeError("labels must have an integer dtype")
        return result.astype(np.int64, copy=False)

    def _validate_label_range(self, labels: np.ndarray) -> None:
        if labels.size == 0:
            return
        minimum = int(labels.min())
        maximum = int(labels.max())
        if minimum < 0 or maximum >= self.num_classes:
            raise IndexError(
                f"labels must be in [0, {self.num_classes}), got "
                f"min={minimum}, max={maximum}"
            )

    def add(
        self,
        features: FeatureMaps,
        labels: torch.Tensor | np.ndarray | Sequence[int],
    ) -> None:
        """Insert a batch into the class-wise rings.

        Inputs are detached and converted to CPU ``self.dtype`` once per map.
        If they are already CPU tensors of that dtype, copying into and back out
        of the bank preserves every bit.
        """
        names, batch_size = self._validate_or_initialize(features)
        labels_np = self._labels_numpy(labels)
        if labels_np.shape[0] != batch_size:
            raise ValueError(
                f"labels batch size {labels_np.shape[0]} does not match "
                f"feature batch size {batch_size}"
            )
        self._validate_label_range(labels_np)
        if batch_size == 0:
            return

        cpu_features = {
            normalized_name: features[original_name]
            .detach()
            .to(device="cpu", dtype=self.dtype)
            for normalized_name, original_name in zip(names, features)
        }

        # Preserve stream order for repeated labels and ring wraparound.
        for sample_index, raw_label in enumerate(labels_np):
            label = int(raw_label)
            slot = int(self.ptr[label])
            for name in self._map_names:
                self.bank[name][label, slot].copy_(cpu_features[name][sample_index])
            self.ptr[label] = (slot + 1) % self.max_size
            if self.count[label] < self.max_size:
                self.count[label] += 1

    def sample(
        self,
        labels: torch.Tensor | np.ndarray | Sequence[int],
        n_samples: int,
        *,
        device: Optional[torch.device | str] = None,
        rng: Optional[np.random.Generator] = None,
        return_indices: bool = False,
    ) -> Dict[str, torch.Tensor] | Tuple[Dict[str, torch.Tensor], np.ndarray]:
        """Sample feature-map dictionaries for each requested class.

        Returns tensors shaped ``(B, n_samples, *sample_shape)``.  Passing an
        isolated NumPy generator makes sampling deterministic without consuming
        global NumPy RNG state.
        """
        if not self.bank or not self._map_names:
            raise RuntimeError("FeatureMemoryBank is empty; call add() first")
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        labels_np = self._labels_numpy(labels)
        self._validate_label_range(labels_np)
        empty_labels = sorted(
            {int(label) for label in labels_np if self.count[int(label)] <= 0}
        )
        if empty_labels:
            raise RuntimeError(
                f"cannot sample empty class buffers: {empty_labels[:20]}"
            )

        sample_indices = np.empty(
            (int(labels_np.shape[0]), n_samples), dtype=np.int64
        )
        choice = np.random.choice if rng is None else rng.choice
        for row, raw_label in enumerate(labels_np):
            valid = int(self.count[int(raw_label)])
            sample_indices[row] = choice(
                valid,
                n_samples,
                replace=(valid < n_samples),
            )

        labels_index = torch.from_numpy(labels_np).long().unsqueeze(1)
        sample_index = torch.from_numpy(sample_indices).long()
        sampled = {
            name: self.bank[name][labels_index, sample_index]
            for name in self._map_names
        }
        if device is not None:
            sampled = {
                name: value.to(device=device) for name, value in sampled.items()
            }
        if return_indices:
            return sampled, sample_indices.copy()
        return sampled

    def __len__(self) -> int:
        return int(self.count.sum())

    def is_ready(self, min_per_class: int = 1) -> bool:
        """Return whether every class holds at least ``min_per_class`` entries."""
        min_per_class = int(min_per_class)
        if min_per_class < 0:
            raise ValueError("min_per_class must be non-negative")
        return bool((self.count >= min_per_class).all())


__all__ = ["FeatureMemoryBank"]
