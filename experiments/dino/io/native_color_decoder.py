"""Optional exact integer decoder; shared object is built and pinned before use."""
from pathlib import Path
import ctypes
import numpy as np

LIBRARY = Path(__file__).with_name('native_color_decode.so')
_library = ctypes.CDLL(str(LIBRARY))
_decode = _library.idrift_color_xy_decode
_decode.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)
_decode.restype = None
_encode = _library.idrift_color_xy_encode
_encode.argtypes = _decode.argtypes
_encode.restype = None


def filter_color_xy(sample):
    if not (isinstance(sample, np.ndarray) and sample.dtype == np.uint8
            and sample.ndim == 3 and sample.shape[0] == 3 and min(sample.shape) > 0
            and sample.flags.c_contiguous):
        raise ValueError('Native filter requires contiguous CHW uint8 RGB input')
    out = np.empty_like(sample)
    _encode(ctypes.c_void_p(sample.ctypes.data), ctypes.c_void_p(out.ctypes.data),
            sample.shape[1], sample.shape[2])
    return out


def inverse_color_xy(decoded, out):
    if not (out.dtype == np.uint8 and out.ndim == 3 and out.shape[0] == 3
            and out.flags.c_contiguous and out.flags.writeable and min(out.shape) > 0):
        raise ValueError('Native decoder requires writable contiguous CHW uint8 RGB output')
    if not isinstance(decoded, bytes) or len(decoded) != out.nbytes:
        raise ValueError('Native decoder input must be a complete immutable decoded byte frame')
    _decode(ctypes.cast(ctypes.c_char_p(decoded), ctypes.c_void_p),
            ctypes.c_void_p(out.ctypes.data), out.shape[1], out.shape[2])


def native_decode_payload(self, payload, out):
    from .color_compressed_memory_bank import ColorCompressedPixelMemoryBank
    if not payload or payload[0] != self._COLOR_XY_TAG:
        return ColorCompressedPixelMemoryBank._decode_payload(self, payload, out)
    try:
        decoded = self._thread_decompressor().decompress(payload[1:], max_output_size=out.nbytes)
    except Exception as error:
        raise RuntimeError('Could not decompress color memory-bank slot') from error
    inverse_color_xy(decoded, out)
