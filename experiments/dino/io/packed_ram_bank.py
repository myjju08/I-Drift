"""Exact color-Zstd positive bank in a fixed anonymous RAM arena.

The native retained allocation (arena, page indices, free stack, slot lengths,
class pointers/counts) is bounded by ``ram_budget_bytes``. Input/output tensors,
bounded transient encoded frames, codec workspaces and Python bookkeeping are
additional memory. No file, disk overflow, allocator trim, or runtime controls
are used. Incompressible contents can exceed the admitted capacity: raise rather
than silently discard a sample or change the training distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
import time

import numpy as np
import torch

from .color_compressed_memory_bank import ColorCompressedPixelMemoryBank


@dataclass(frozen=True, slots=True)
class _PackedRef:
    slot: int
    length: int

    def __len__(self):
        return self.length


class _SlotTable:
    """Keep original sample() dispatch without persistent Python payload objects."""
    def __init__(self, owner):
        self.owner = owner
        self.shape = (owner.num_classes, owner.max_size)

    def __getitem__(self, key):
        label, index = key
        owner = self.owner
        owner._check_owner()
        if not (0 <= int(label) < owner.num_classes and 0 <= int(index) < owner.max_size):
            raise IndexError('Packed bank slot outside table')
        slot = int(label) * owner.max_size + int(index)
        length = int(owner._lengths[slot])
        return _PackedRef(slot, length) if length else None

    @property
    def flat(self):
        for label in range(self.shape[0]):
            for index in range(self.shape[1]):
                yield self[label, index]


class PackedRAMColorMemoryBank(ColorCompressedPixelMemoryBank):
    """One process owns a fixed page arena; only transient frames use malloc.

    Pages may be noncontiguous. Each logical record has a compact fixed-capacity
    page index, so ring overwrite never requires moving unrelated records or
    compacting the arena. Reuse an overwritten record's pages before allocating
    more. The original sample method retains draw order, duplicate decoding,
    empty-class behavior and uint8-to-float normalization exactly.

    add()/sample()/close() are sequential trainer operations; simultaneous calls
    from independent caller threads are not supported (same as the source bank).
    Internal codec workers run only while that operation is active.
    """
    def __init__(self, *args, ram_budget_bytes=13*1024**3, page_bytes=4096,
                 predictor='xy', decode_backend='numpy', codec_batch_size=None, **kwargs):
        if type(ram_budget_bytes) is not int or ram_budget_bytes <= 0:
            raise ValueError('ram_budget_bytes must be a positive integer')
        if type(page_bytes) is not int or page_bytes < 64 or page_bytes & (page_bytes-1):
            raise ValueError('page_bytes must be a power of two at least64')
        if decode_backend not in {'numpy', 'native'}:
            raise ValueError('decode_backend must be numpy or native')
        if codec_batch_size is not None and (type(codec_batch_size) is not int
                                            or not 1 <= codec_batch_size <= 128):
            raise ValueError('codec_batch_size must be an integer from1through128 or None')
        self.decode_backend = decode_backend
        self._native_decode = None
        self._native_filter = None
        if decode_backend == 'native':
            from .native_color_decoder import native_decode_payload, filter_color_xy
            self._native_decode = native_decode_payload
            self._native_filter = filter_color_xy
        self.ram_budget_bytes = ram_budget_bytes
        self.page_bytes = page_bytes
        self._owner_pid = os.getpid()
        self._closed = False
        self._arena_mapping = None
        self._pages = None
        self._page_ids = None
        self._lengths = None
        self._free_pages = None
        self._free_count = 0
        self._page_count = 0
        self._pages_per_record = 0
        self._max_payload_bytes = 0
        self._entries = 0
        self.metadata_bytes = 0
        self.arena_capacity_bytes = 0
        self.arena_dontfork = False
        super().__init__(*args, predictor=predictor, **kwargs)
        # Explicit larger windows amortize executor barriers while retaining a
        # hard transient bound: at most128 encoded RGB256 frames (~24MiB).
        self.codec_batch_size = (min(128, max(1, self.codec_workers*2))
                                 if codec_batch_size is None else codec_batch_size)
        if self.num_classes <= 0 or self.max_size <= 0:
            self.close()
            raise ValueError('num_classes and max_size must be positive')

    def _check_owner(self):
        if os.getpid() != self._owner_pid:
            raise RuntimeError('Packed bank cannot be accessed from a forked process')
        if self._closed:
            raise RuntimeError('Packed bank is closed')

    def _filter(self, sample):
        if self._native_filter is not None and self.predictor == 'xy':
            return self._native_filter(sample)
        return super()._filter(sample)

    def _init_bank(self, sample_shape):
        self._check_owner()
        if len(sample_shape) != 3 or sample_shape[0] != 3 or min(sample_shape) <= 0:
            raise ValueError(f'Expected nonempty CHW RGB shape, got {sample_shape}')
        if self.bank is not None:
            raise RuntimeError('Packed bank is already initialized')
        raw_bytes = int(np.prod(sample_shape, dtype=np.int64))
        if raw_bytes+5 > np.iinfo(np.uint32).max:
            raise ValueError('Packed slot length exceeds uint32 capacity')
        max_payload = raw_bytes+5
        slots = self.num_classes*self.max_size
        pages_per_record = (max_payload+self.page_bytes-1)//self.page_bytes
        fixed_metadata = slots*pages_per_record*4 + slots*4 + self.ptr.nbytes+self.count.nbytes
        page_count = min(slots*pages_per_record,
                         (self.ram_budget_bytes-fixed_metadata)//(self.page_bytes+4))
        if page_count <= 0 or page_count > np.iinfo(np.uint32).max:
            raise ValueError('RAM budget cannot represent the packed arena and metadata')
        arena_bytes = page_count*self.page_bytes
        mapping = mmap.mmap(-1, arena_bytes, flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
        try:
            # Training/eval DataLoader children never access this bank. Prevent
            # inheriting its huge VMA and accidental COW/worker accounting.
            if hasattr(mmap, 'MADV_DONTFORK'):
                mapping.madvise(mmap.MADV_DONTFORK)
                self.arena_dontfork = True
            pages = np.ndarray((page_count, self.page_bytes), dtype=np.uint8, buffer=mapping)
            page_ids = np.empty((slots, pages_per_record), dtype=np.uint32)
            lengths = np.zeros(slots, dtype=np.uint32)
            free_pages = np.arange(page_count, dtype=np.uint32)
        except BaseException:
            # ndarray has no retained memoryview; clear it before closing.
            if 'pages' in locals():
                del pages
            mapping.close()
            raise
        self.feature_shape = tuple(sample_shape)
        self._max_payload_bytes = max_payload
        self._pages_per_record = pages_per_record
        self._page_count = page_count
        self._free_count = page_count
        self._arena_mapping, self._pages = mapping, pages
        self._page_ids, self._lengths, self._free_pages = page_ids, lengths, free_pages
        self.metadata_bytes = fixed_metadata+free_pages.nbytes
        self.arena_capacity_bytes = arena_bytes
        self.bank = _SlotTable(self)

    def _store_payload(self, slot, payload):
        self._check_owner()
        if not isinstance(payload, bytes) or not 0 < len(payload) <= self._max_payload_bytes:
            raise RuntimeError('Invalid packed codec frame length')
        old_length = int(self._lengths[slot])
        old_pages = (old_length+self.page_bytes-1)//self.page_bytes
        needed = (len(payload)+self.page_bytes-1)//self.page_bytes
        extra = max(0, needed-old_pages)
        if extra > self._free_count:
            raise MemoryError('Packed bank RAM capacity exhausted: '
                              f'need_extra_pages={extra} free_pages={self._free_count}; '
                              'record and ring pointer retained; disk overflow is disabled')
        if extra:
            new_free = self._free_count-extra
            self._page_ids[slot, old_pages:needed] = self._free_pages[new_free:self._free_count]
            self._free_count = new_free
        ids = self._page_ids[slot, :needed]
        source = np.frombuffer(payload, dtype=np.uint8)
        full, tail = divmod(len(payload), self.page_bytes)
        if full:
            self._pages[ids[:full]] = source[:full*self.page_bytes].reshape(full, self.page_bytes)
        if tail:
            self._pages[int(ids[full]), :tail] = source[full*self.page_bytes:]
        if old_pages > needed:
            released = old_pages-needed
            self._free_pages[self._free_count:self._free_count+released] = self._page_ids[slot, needed:old_pages]
            self._free_count += released
        self._lengths[slot] = len(payload)
        self.payload_bytes += len(payload)-old_length
        if not old_length:
            self._entries += 1
            self.raw_equivalent_bytes += self._max_payload_bytes-5

    def _read_payload(self, reference):
        self._check_owner()
        if not isinstance(reference, _PackedRef) or not 0 <= reference.slot < len(self._lengths):
            raise RuntimeError('Invalid packed slot reference')
        if not 0 < reference.length <= self._max_payload_bytes or int(self._lengths[reference.slot]) != reference.length:
            raise RuntimeError('Stale or corrupt packed slot reference')
        needed = (reference.length+self.page_bytes-1)//self.page_bytes
        ids = self._page_ids[reference.slot, :needed]
        if np.any(ids >= self._page_count):
            raise RuntimeError('Packed slot page index is corrupt')
        # One native gather per frame; no Python loop over image pages. The
        # codec accepts the view without another persistent payload allocation.
        gathered = self._pages[ids]
        return memoryview(gathered).cast('B')[:reference.length]

    def _decode_payload(self, payload, out):
        self._check_owner()
        if isinstance(payload, _PackedRef):
            payload = self._read_payload(payload)
        if self._native_decode is not None:
            return self._native_decode(self, payload, out)
        return super()._decode_payload(payload, out)

    def add(self, samples, labels):
        self._check_owner()
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        samples = self._encode_samples(samples)
        labels = np.asarray(labels).astype(np.int32)
        if labels.ndim != 1 or len(samples) != len(labels):
            raise ValueError('Expected one label per sample')
        if np.any(labels < 0) or np.any(labels >= self.num_classes):
            raise ValueError('Label outside memory-bank class range')
        if self.bank is None:
            self._init_bank(samples.shape[1:])
        elif tuple(samples.shape[1:]) != self.feature_shape:
            raise ValueError('Memory-bank sample shape changed after initialization')
        started = time.perf_counter()
        window = self.codec_batch_size
        for start in range(0, len(labels), window):
            stop = min(start+window, len(labels))
            if self._executor is None:
                payloads = [self._compress_sample(samples[i]) for i in range(start, stop)]
            else:
                payloads = list(self._executor.map(self._compress_sample, (samples[i] for i in range(start, stop))))
            for offset, payload in enumerate(payloads):
                label = int(labels[start+offset])
                index = int(self.ptr[label])
                self._store_payload(label*self.max_size+index, payload)
                self.ptr[label] = (index+1)%self.max_size
                if self.count[label] < self.max_size:
                    self.count[label] += 1
        self.last_add_seconds = time.perf_counter()-started

    def sample(self, *args, **kwargs):
        self._check_owner()
        return super().sample(*args, **kwargs)

    def resume_codec_workers(self):
        self._check_owner()
        return super().resume_codec_workers()

    @property
    def storage_stats(self):
        used = self._page_count-self._free_count
        return {'ram_budget_bytes': self.ram_budget_bytes,
                'ram_allocated_capacity_bytes': self.arena_capacity_bytes+self.metadata_bytes,
                'arena_capacity_bytes': self.arena_capacity_bytes,
                'metadata_bytes': self.metadata_bytes,
                'ram_payload_bytes': self.payload_bytes, 'disk_payload_bytes': 0,
                'ram_entries': self._entries, 'disk_entries': 0,
                'payload_bytes': self.payload_bytes, 'raw_equivalent_bytes': self.raw_equivalent_bytes,
                'pages_used': used, 'page_bytes': self.page_bytes,
                'page_slack_bytes': used*self.page_bytes-self.payload_bytes,
                'free_capacity_bytes': self._free_count*self.page_bytes,
                'spill_read_bytes': 0, 'spill_write_bytes': 0,
                'arena_dontfork': self.arena_dontfork, 'closed': self._closed}

    def close(self):
        if self._closed:
            return
        self._check_owner()
        self.suspend_codec_workers()
        self.bank = None
        self._pages = None
        self._page_ids = self._lengths = self._free_pages = None
        if self._arena_mapping is not None:
            self._arena_mapping.close()
            self._arena_mapping = None
        self._closed = True


__all__ = ['PackedRAMColorMemoryBank']
