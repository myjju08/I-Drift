"""Lossless byte-exact RGB bank with reversible color and spatial decorrelation."""
from __future__ import annotations

import zlib

import numpy as np

from memory_bank import CompressedPixelMemoryBank


class ColorCompressedPixelMemoryBank(CompressedPixelMemoryBank):
    """Keep all original bank/RNG semantics; encode independent checksummed slots.

    uint8 modular arithmetic is intentional: (R,G,B)→(G,R−G,B−G). The x
    predictor stores horizontal differences; xy additionally differences rows.
    Each inverse uses matching uint8 cumulative sums, with exact overflow.
    """
    _COLOR_X_TAG = 2
    _COLOR_XY_TAG = 3

    def __init__(self, *args, predictor='x', **kwargs):
        if predictor not in {'x', 'xy'}:
            raise ValueError('predictor must be x or xy')
        self.predictor = predictor
        super().__init__(*args, **kwargs)

    def _filter(self, sample):
        color = np.empty_like(sample)
        color[0] = sample[1]
        np.subtract(sample[0], sample[1], out=color[1])
        np.subtract(sample[2], sample[1], out=color[2])
        filtered = self._delta_x(color)
        if self.predictor == 'xy':
            vertical = np.empty_like(filtered)
            vertical[:, 0] = filtered[:, 0]
            np.subtract(filtered[:, 1:], filtered[:, :-1], out=vertical[:, 1:])
            filtered = vertical
        return filtered

    def _compress_sample(self, sample):
        sample = np.ascontiguousarray(sample, dtype=np.uint8)
        compressed = self._thread_compressor().compress(self._filter(sample).tobytes(order='C'))
        if len(compressed) >= sample.nbytes:
            raw = sample.tobytes(order='C')
            return bytes((self._RAW_TAG,)) + (zlib.crc32(raw)&0xFFFFFFFF).to_bytes(4, 'little') + raw
        tag = self._COLOR_X_TAG if self.predictor == 'x' else self._COLOR_XY_TAG
        return bytes((tag,)) + compressed

    def _decode_payload(self, payload, out):
        if not payload or payload[0] not in {self._COLOR_X_TAG, self._COLOR_XY_TAG}:
            return super()._decode_payload(payload, out)
        try:
            decoded = self._thread_decompressor().decompress(payload[1:], max_output_size=out.nbytes)
        except Exception as error:
            raise RuntimeError('Could not decompress color memory-bank slot') from error
        if len(decoded) != out.nbytes:
            raise RuntimeError('Color memory-bank slot decoded length mismatch')
        filtered = np.frombuffer(decoded, dtype=np.uint8).reshape(out.shape)
        if payload[0] == self._COLOR_XY_TAG:
            np.add.accumulate(filtered, axis=-2, dtype=np.uint8, out=out)
            np.add.accumulate(out, axis=-1, dtype=np.uint8, out=out)
        else:
            np.add.accumulate(filtered, axis=-1, dtype=np.uint8, out=out)
        green = out[0].copy()
        np.add(out[1], green, out=out[0])
        np.add(out[2], green, out=out[2])
        out[1] = green
