#!/usr/bin/env python3
"""Exact bounded whole-file PIL loader used by the active DINO runtime."""
from __future__ import annotations

import io
import os
from pathlib import Path
import stat

MAX_FILE_BYTES = 8 * 1024 ** 2


class WholeFilePILLoader:
    """Use the same PIL RGB decode after bounded, sequential reads of one file.

The original callable handles accimage, nonregular/oversize files, and failures
in the read optimization. Decode exceptions still come from PIL. No image cache,
read-ahead advice, transform, RNG call, or image-backend selection is introduced.
"""

    def __init__(self, original_loader, *, max_file_bytes=MAX_FILE_BYTES):
        from torchvision.datasets.folder import default_loader, pil_loader
        if original_loader not in (default_loader, pil_loader):
            raise ValueError('Only the original torchvision default/PIL loaders are supported')
        if type(max_file_bytes) is not int or max_file_bytes <= 0:
            raise ValueError('max_file_bytes must be a positive integer')
        self.original_loader = original_loader
        self.max_file_bytes = max_file_bytes

    def __call__(self, path):
        from PIL import Image
        from torchvision import get_image_backend
        from torchvision.datasets.folder import default_loader
        if self.original_loader is default_loader and get_image_backend() == 'accimage':
            return self.original_loader(path)
        try:
            payload = self._read_regular_file(path)
        except OSError:
            # Preserve the original error/fallback behavior, rather than turn a
            # recoverable fast-read failure into a new training failure.
            return self.original_loader(path)
        if payload is None:
            return self.original_loader(path)
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert('RGB')

    def _read_regular_file(self, path):
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= self.max_file_bytes:
                return None
            chunks, received = [], 0
            while received < before.st_size:
                if len(chunks) >= 64:
                    # Bound fragment bookkeeping too; unusual short-read
                    # streams fall back to the original buffered loader.
                    return None
                chunk = os.read(descriptor, before.st_size - received)
                if not chunk:
                    return None
                chunks.append(chunk)
                received += len(chunk)
            after = os.fstat(descriptor)
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                return None
            return b''.join(chunks)
        finally:
            os.close(descriptor)


def make_wholefile_split_factory(original_create_split, *, max_file_bytes=MAX_FILE_BYTES):
    """Replace only a newly created raw ImageFolder's supported image reader."""
    reports = []

    def factory(*args, **kwargs):
        from torchvision.datasets.folder import default_loader, pil_loader
        loader, preprocess, postprocess = original_create_split(*args, **kwargs)
        raw = not kwargs.get('use_cache', False) and not kwargs.get('use_latent', False)
        if raw:
            if getattr(loader, '_iterator', None) is not None:
                raise RuntimeError('Cannot replace an image reader after worker startup')
            dataset = loader.dataset
            original = getattr(dataset, 'loader', None)
            applied = original in (default_loader, pil_loader)
            if applied:
                dataset.loader = WholeFilePILLoader(original, max_file_bytes=max_file_bytes)
            report = {'split': kwargs.get('split', 'train'), 'reader_applied': applied,
                      'max_file_bytes': max_file_bytes, 'accimage_path_delegated': True,
                      'custom_loader_preserved': not applied,
                      'dataset_class': type(dataset).__module__+'.'+type(dataset).__qualname__,
                      'transforms_sampler_labels_worker_init_unchanged': True}
            dataset.wholefile_reader_report = report
            reports.append(report)
        return loader, preprocess, postprocess

    factory.wholefile_reader_reports = reports
    return factory
