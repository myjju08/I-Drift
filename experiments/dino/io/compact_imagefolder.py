"""Packed ImageFolder metadata without changing its loader or sample stream.

The original ImageFolder still discovers and orders files. Replacing its Python
sample records before worker startup avoids touching a different shared Python
object header for every path in every forked worker. Pixel I/O and transforms
remain methods of the original dataset instance.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import operator
import os
import sys

import numpy as np


class PackedImageSamples(Sequence):
    def __init__(self, samples, *, root=''):
        count = len(samples)
        prefix = os.fspath(root).rstrip(os.sep) + os.sep if root else ''
        # Preserve unusual roots/paths verbatim instead of normalizing them.
        if prefix and any(not path.startswith(prefix) for path, _ in samples):
            prefix = ''
        offsets = np.empty(count + 1, dtype=np.uint64)
        targets = np.empty(count, dtype=np.int32)
        blob = bytearray()
        digest = hashlib.sha256()
        estimated_python_bytes = sys.getsizeof(samples)
        offsets[0] = 0
        for index, sample in enumerate(samples):
            path, label = sample
            if not isinstance(path, str):
                raise TypeError('ImageFolder paths must be strings')
            label = operator.index(label)
            if not np.iinfo(np.int32).min <= label <= np.iinfo(np.int32).max:
                raise ValueError('ImageFolder target exceeds int32 range')
            encoded = path[len(prefix):].encode('utf-8', errors='surrogatepass')
            blob.extend(encoded)
            offsets[index + 1] = len(blob)
            targets[index] = label
            # Length framing makes this an unambiguous ordered path/target hash.
            full_path = path.encode('utf-8', errors='surrogatepass')
            digest.update(len(full_path).to_bytes(8, 'little'))
            digest.update(full_path)
            digest.update(label.to_bytes(8, 'little', signed=True))
            estimated_python_bytes += sys.getsizeof(sample) + sys.getsizeof(path)
        self.prefix = prefix
        self._blob = bytes(blob)
        self._offsets = offsets
        self.targets = targets
        self._offsets.flags.writeable = False
        self.targets.flags.writeable = False
        self.report = {
            'sample_count': count, 'ordered_paths_and_targets_sha256': digest.hexdigest(),
            'packed_payload_bytes': len(self._blob) + offsets.nbytes + targets.nbytes,
            'original_samples_estimated_python_bytes': estimated_python_bytes,
            'path_prefix_bytes': len(prefix.encode('utf-8', errors='surrogatepass')),
            'order_source': 'the original ImageFolder.samples sequence',
            'paths_labels_pixels_and_transforms_unchanged': True,
        }

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError('ImageFolder sample index out of range')
        start, stop = int(self._offsets[index]), int(self._offsets[index + 1])
        path = self.prefix + self._blob[start:stop].decode('utf-8', errors='surrogatepass')
        return path, int(self.targets[index])


def compact_imagefolder_metadata(dataset):
    """Replace only metadata, before iter(loader) starts worker processes."""
    if isinstance(dataset.samples, PackedImageSamples):
        return dataset.compact_metadata_report
    samples = dataset.samples
    packed = PackedImageSamples(samples, root=dataset.root)
    if len(dataset.targets) != len(packed):
        raise ValueError('ImageFolder samples and targets have different lengths')
    if not np.array_equal(np.asarray(dataset.targets, dtype=np.int64), packed.targets):
        raise ValueError('ImageFolder samples and targets disagree')
    dataset.samples = packed
    dataset.imgs = packed
    dataset.targets = packed.targets
    dataset.compact_metadata_report = dict(packed.report)
    return dataset.compact_metadata_report


def make_compact_split_factory(original_create_split, *, train_prefetch_factor=1):
    """Wrap the original factory; preserve worker count, seed function and order.

Set train_prefetch_factor=None to retain the original queue depth as well.
The default reduces only the queued training batches from two to one per worker.
"""
    if train_prefetch_factor is not None and (
        type(train_prefetch_factor) is not int or train_prefetch_factor < 1
    ):
        raise ValueError('train_prefetch_factor must be a positive integer or None')
    reports = []

    def factory(*args, **kwargs):
        if args:
            raise TypeError('The original create_imagenet_split uses keyword-only arguments')
        active = dict(kwargs)
        raw = not active.get('use_cache', False) and not active.get('use_latent', False)
        if (raw and active.get('split', 'train') == 'train'
                and active.get('num_workers', 8) > 0 and train_prefetch_factor is not None):
            active['prefetch_factor'] = train_prefetch_factor
        loader, preprocess, postprocess = original_create_split(**active)
        if raw:
            # DataLoader construction does not spawn workers; its first iterator
            # does. Reject accidental use after worker startup explicitly.
            if getattr(loader, '_iterator', None) is not None:
                raise RuntimeError('Cannot compact a DataLoader after workers have started')
            report = dict(compact_imagefolder_metadata(loader.dataset),
                          split=active.get('split', 'train'),
                          num_workers=loader.num_workers,
                          prefetch_factor=loader.prefetch_factor,
                          persistent_workers=loader.persistent_workers)
            loader.dataset.compact_metadata_report = report
            reports.append(report)
        return loader, preprocess, postprocess

    factory.compact_metadata_reports = reports
    return factory
