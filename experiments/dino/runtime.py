"""Lossless I/O from the active DINO run, with package-local dependencies.

The arena sizing, codec settings, bank routing, loader ordering and train queue
depth match ``io_optimization/train_fastio_ablation.py`` as used on 2026-09-09.
The runtime never resumes another run's W&B identity or edits live services.
"""
from __future__ import annotations

import json
import time

RAM_BUDGET_BYTES = 13 * 1024 ** 3
CODEC_BATCH_SIZE = 128
INITIAL_GENERATOR_SHA256 = {
    0: 'f2d9f6522592733bd50c5d995bfb507f09bff965f7bfd397fddc336b383aa31c',
    1: 'b4300af0b38959d2e642bc3a8d719087a48f6daceba3d49ad3ed167677266ce9',
}
STORAGE_DESCRIPTION = ('lossless color-XY Zstandard uint8 in fixed packed native RAM arenas; '
                       '13 GiB arena+index capacity per rank; no disk spill; original samples and RNG')


class StageTimings:
    """Host wall timing only: no tensor operations, synchronization or RNG calls."""
    def __init__(self):
        self.step = None
        self.values = {}

    def reset(self, step):
        self.step, self.values = step, {}

    def add(self, key, seconds):
        self.values[key] = self.values.get(key, 0.0) + seconds


class TimedLoader:
    def __init__(self, loader, timings):
        self.loader, self.timings = loader, timings

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        started = time.perf_counter()
        iterator = iter(self.loader)
        self.timings.add('loader_iterator_setup_seconds', time.perf_counter() - started)
        return TimedIterator(iterator, self.timings)


class TimedIterator:
    def __init__(self, iterator, timings):
        self.iterator, self.timings = iterator, timings

    def __iter__(self):
        return self

    def __next__(self):
        started = time.perf_counter()
        try:
            return next(self.iterator)
        finally:
            self.timings.add('loader_next_seconds', time.perf_counter() - started)


def install_compact_timed_loader(trainer, timings):
    from .io.compact_imagefolder import make_compact_split_factory
    from .io.wholefile_reader import make_wholefile_split_factory
    reader = make_wholefile_split_factory(trainer.create_imagenet_split)
    compact = make_compact_split_factory(reader, train_prefetch_factor=1)
    compact.wholefile_reader_reports = reader.wholefile_reader_reports

    def create_split(**kwargs):
        loader, preprocess, postprocess = compact(**kwargs)
        print('[FastIO loader] ' + json.dumps({
            'split': kwargs.get('split', 'train'), 'rank': kwargs.get('rank', 0),
            'actual_prefetch_factor': loader.prefetch_factor, 'num_workers': loader.num_workers,
            'compact_metadata': True,
            'wholefile_reader': getattr(loader.dataset, 'wholefile_reader_report', None),
        }), flush=True)
        if kwargs.get('split', 'train') != 'train':
            return loader, preprocess, postprocess

        def timed_preprocess(*args, **options):
            started = time.perf_counter()
            try:
                return preprocess(*args, **options)
            finally:
                timings.add('preprocess_seconds', time.perf_counter() - started)

        return TimedLoader(loader, timings), timed_preprocess, postprocess

    trainer.create_imagenet_split = create_split
    return compact


def make_memory_bank_factory(workdir, rank, codec_workers=8, *,
                             ram_budget_bytes=RAM_BUDGET_BYTES, decode_backend='native'):
    """Replace only the 1000-class, 128-slot positive uint8 bank, as in the live run."""
    import numpy as np
    from memory_bank import ArrayMemoryBank
    from .io.packed_ram_bank import PackedRAMColorMemoryBank
    if type(codec_workers) is not int or codec_workers <= 0:
        raise ValueError('codec_workers must be positive')
    if int(rank) not in (0, 1):
        raise ValueError('The reproduced DINO runtime requires exactly two ranks')
    compressed_banks = []

    def factory(num_classes=1000, max_size=64, dtype=np.float32, storage_mode='raw'):
        if (int(num_classes), int(max_size), str(storage_mode).strip().lower()) == (1000, 128, 'pixel_uint8'):
            bank = PackedRAMColorMemoryBank(
                num_classes=num_classes, max_size=max_size, compression_level=1,
                codec_workers=codec_workers, predictor='xy', decode_backend=decode_backend,
                codec_batch_size=CODEC_BATCH_SIZE, ram_budget_bytes=ram_budget_bytes,
            )
            compressed_banks.append(bank)
            return bank
        return ArrayMemoryBank(num_classes, max_size, dtype, storage_mode)

    factory.compressed_banks = compressed_banks
    return factory


def check_factory(decode_backend='native'):
    """Small CPU trace checks ring wrap, empty classes, decoded bytes and RNG."""
    import numpy as np
    import torch
    from memory_bank import ArrayMemoryBank, CompressedPixelMemoryBank
    factory = make_memory_bank_factory(None, 0, 2, ram_budget_bytes=16 * 1024 ** 2,
                                       decode_backend=decode_backend)
    compressed = factory(1000, 128, storage_mode='pixel_uint8')
    dense = ArrayMemoryBank(1000, 128, storage_mode='pixel_uint8')
    if not isinstance(compressed, CompressedPixelMemoryBank):
        raise AssertionError('Positive factory route changed')
    if (type(factory(1, 1000, storage_mode='pixel_uint8')) is not ArrayMemoryBank
            or type(factory(1000, 16, storage_mode='raw')) is not ArrayMemoryBank):
        raise AssertionError('A negative or historical bank was replaced')
    try:
        source = np.random.default_rng(20260908)
        labels = np.array([0] * 133 + [1, 2, 1], dtype=np.int64)
        pixels = source.integers(0, 256, size=(len(labels), 3, 2, 3), dtype=np.uint8)
        for start, end in ((0, 79), (79, len(labels))):
            dense.add(pixels[start:end], labels[start:end])
            compressed.add(pixels[start:end], labels[start:end])
        if not (np.array_equal(dense.ptr, compressed.ptr) and np.array_equal(dense.count, compressed.count)):
            raise AssertionError('Ring pointers or class counts changed')
        for count in (1, 4, 129):
            left_rng, right_rng = np.random.default_rng(count), np.random.default_rng(count)
            selected = np.array([0, 1, 2, 999], dtype=np.int64)
            if not torch.equal(dense.sample(selected, count, rng=left_rng),
                               compressed.sample(selected, count, rng=right_rng)):
                raise AssertionError('Decoded sample values changed')
            if left_rng.bit_generator.state != right_rng.bit_generator.state:
                raise AssertionError('Sampling RNG changed')
        stats = compressed.storage_stats
        if stats['ram_allocated_capacity_bytes'] > 16 * 1024 ** 2 or stats['disk_entries'] != 0:
            raise AssertionError('Bank allocation exceeded its bound or spilled to disk')
        return {'passed': True, 'ring_wrap': True, 'empty_class': True,
                'samples_and_rng_identical': True, 'storage_stats': stats}
    finally:
        compressed.close()


def check_initial_generator(trainer, cfg):
    import torch
    from models.dino_rf_tuning import module_fingerprint
    report = {}
    with torch.random.fork_rng(devices=[]):
        for rank in (0, 1):
            torch.manual_seed(43 + rank)
            generator = trainer.build_ditgen_from_config(cfg['_raw']['model'], cfg['_raw']['dataset'])
            fingerprint = module_fingerprint(generator)
            del generator
            if fingerprint != INITIAL_GENERATOR_SHA256[rank]:
                raise AssertionError(f'Initial generator rank {rank} differs from the original run')
            report[str(rank)] = {'seed': 43 + rank, 'sha256': fingerprint, 'matches_source': True}
    return report
