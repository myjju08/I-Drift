#!/usr/bin/env python3
"""Stream and safely extract the official ILSVRC-2012 ImageNet archives.

The 147.9 GB training archive is an *outer* tar containing one tar per WNID.
This program never stores that outer archive.  It first builds a resumable
index by fetching only the 512-byte outer-tar headers, then downloads nested
class-tar payload ranges in parallel.  Each class is committed atomically, so a
process restart skips committed classes and resumes the remaining exact ranges.

The validation archive is also streamed rather than stored.  The official
devkit ``meta.mat`` and validation ground truth determine the WNID directory
for every validation JPEG.

Only files below ``--output-root`` and ``--state-root`` are created. Existing
non-matching final files/directories are never deleted or silently replaced.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
from typing import BinaryIO, Iterator, Mapping, Optional, Sequence
import urllib.error
import urllib.request
import urllib.response


OFFICIAL_TRAIN_URL = (
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
)
OFFICIAL_VAL_URL = (
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
)
OFFICIAL_DEVKIT_URL = (
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
)
OFFICIAL_TRAIN_BYTES = 147_897_477_120
OFFICIAL_VAL_BYTES = 6_744_924_160
OFFICIAL_VAL_MD5 = "29b22e2961454d5413ddabcf34fc5622"
OFFICIAL_TRAIN_FILES = 1_281_167
OFFICIAL_VAL_FILES = 50_000
OFFICIAL_CLASSES = 1_000

_WNID_RE = re.compile(r"^n\d{8}$")
_TRAIN_TAR_RE = re.compile(r"^(n\d{8})\.tar$")
_VAL_IMAGE_RE = re.compile(r"^ILSVRC2012_val_(\d{8})\.(?:JPEG|jpeg|jpg)$")
_JPEG_SUFFIXES = {".jpeg", ".jpg"}
_BLOCK_SIZE = 512
_READ_CHUNK = 1024 * 1024


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Range-resumably stream/extract official raw ImageNet train/val "
            "without saving the outer archives."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination containing ImageFolder-style train/ and val/.",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help="Resume/marker directory (default: OUTPUT/.idrift_raw_state).",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("devkit", "train", "val"),
        default=("devkit", "train", "val"),
        help="Parts to process; val always requires an available devkit mapping.",
    )
    parser.add_argument("--train-url", default=OFFICIAL_TRAIN_URL)
    parser.add_argument("--val-url", default=OFFICIAL_VAL_URL)
    parser.add_argument(
        "--val-local-archive",
        type=Path,
        default=None,
        help=(
            "Optional fully downloaded validation tar. It must match the "
            "configured byte count and MD5; the official remote is still "
            "probed for source provenance."
        ),
    )
    parser.add_argument(
        "--expected-val-archive-md5",
        default=OFFICIAL_VAL_MD5,
    )
    parser.add_argument("--devkit-url", default=OFFICIAL_DEVKIT_URL)
    parser.add_argument(
        "--expected-train-archive-bytes",
        type=int,
        default=OFFICIAL_TRAIN_BYTES,
    )
    parser.add_argument(
        "--expected-val-archive-bytes",
        type=int,
        default=OFFICIAL_VAL_BYTES,
    )
    parser.add_argument(
        "--expected-train-files", type=int, default=OFFICIAL_TRAIN_FILES
    )
    parser.add_argument(
        "--expected-val-files", type=int, default=OFFICIAL_VAL_FILES
    )
    parser.add_argument(
        "--expected-classes", type=int, default=OFFICIAL_CLASSES
    )
    parser.add_argument(
        "--reference-class-root",
        type=Path,
        default=None,
        help=(
            "Optional existing ImageFolder split whose sorted WNIDs must match "
            "the raw/devkit mapping exactly."
        ),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--http-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument(
        "--train-workers",
        type=int,
        default=16,
        help="Parallel exact-Range class downloads (default: 16).",
    )
    parser.add_argument(
        "--val-checkpoint-every",
        type=int,
        default=100,
        help="Persist a validation Range offset every N committed images.",
    )
    parser.add_argument(
        "--reserve-bytes",
        type=int,
        default=10 * 1024**3,
        help="Disk space to leave unused after the conservative extraction estimate.",
    )
    parser.add_argument(
        "--allow-nonstandard-counts-for-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug-max-train-classes",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug-max-val-files",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    args.parts = tuple(dict.fromkeys(args.parts))
    args.output_root = args.output_root.resolve()
    args.state_root = (
        args.state_root.resolve()
        if args.state_root is not None
        else args.output_root / ".idrift_raw_state"
    )
    if args.val_local_archive is not None:
        args.val_local_archive = args.val_local_archive.resolve()
        if "val" not in args.parts:
            parser.error("--val-local-archive requires val in --parts")
    args.expected_val_archive_md5 = str(
        args.expected_val_archive_md5
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", args.expected_val_archive_md5):
        parser.error("--expected-val-archive-md5 must be 32 lowercase hex digits")
    if args.output_root == args.state_root:
        parser.error("--state-root must not equal --output-root")
    if args.timeout <= 0 or args.http_retries <= 0 or args.retry_delay < 0:
        parser.error("HTTP timeout/retry values must be positive")
    if args.train_workers <= 0:
        parser.error("--train-workers must be positive")
    if args.val_checkpoint_every <= 0:
        parser.error("--val-checkpoint-every must be positive")
    if args.reserve_bytes < 0:
        parser.error("--reserve-bytes must be non-negative")
    for name in (
        "expected_train_archive_bytes",
        "expected_val_archive_bytes",
        "expected_train_files",
        "expected_val_files",
        "expected_classes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.debug_max_train_classes < 0 or args.debug_max_val_files < 0:
        parser.error("debug limits must be non-negative")

    official_counts = (
        args.expected_train_archive_bytes == OFFICIAL_TRAIN_BYTES
        and args.expected_val_archive_bytes == OFFICIAL_VAL_BYTES
        and args.expected_train_files == OFFICIAL_TRAIN_FILES
        and args.expected_val_files == OFFICIAL_VAL_FILES
        and args.expected_classes == OFFICIAL_CLASSES
    )
    if not official_counts and not args.allow_nonstandard_counts_for_test:
        parser.error(
            "Non-official sizes/counts require --allow-nonstandard-counts-for-test"
        )
    return args


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write_bytes(path, encoded)


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _request(
    url: str,
    *,
    timeout: float,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> urllib.response.addinfourl:
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "I-Drift-raw-ImageNet-streamer/1.0",
    }
    if end is not None and start is None:
        raise ValueError("A Range end requires a Range start")
    if start is not None:
        if start < 0 or (end is not None and end < start):
            raise ValueError(f"Invalid byte range {start}-{end}")
        headers["Range"] = (
            f"bytes={int(start)}-" if end is None else f"bytes={int(start)}-{int(end)}"
        )
    request = urllib.request.Request(url, headers=headers, method="GET")
    return urllib.request.urlopen(request, timeout=timeout)


def _parse_content_range(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if match is None:
        raise ValueError(f"Invalid Content-Range header: {value!r}")
    start, end, total = (int(part) for part in match.groups())
    if start < 0 or end < start or total <= end:
        raise ValueError(f"Inconsistent Content-Range header: {value!r}")
    return start, end, total


def _probe_remote(
    url: str,
    *,
    timeout: float,
    retries: int,
    retry_delay: float,
    expected_bytes: Optional[int] = None,
) -> dict[str, object]:
    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            with _request(url, timeout=timeout, start=0, end=0) as response:
                status = int(getattr(response, "status", response.getcode()))
                content_range = response.headers.get("Content-Range", "")
                if status != 206 or not content_range:
                    raise RuntimeError(
                        f"Server must support Range GET with HTTP 206; got {status} "
                        f"Content-Range={content_range!r} for {url}"
                    )
                range_start, range_end, total = _parse_content_range(content_range)
                if (range_start, range_end) != (0, 0):
                    raise RuntimeError(
                        f"Range probe returned {range_start}-{range_end}, expected 0-0"
                    )
                if expected_bytes is not None and total != expected_bytes:
                    raise RuntimeError(
                        f"Remote size mismatch for {url}: {total}, expected {expected_bytes}"
                    )
                # Consume one byte so broken Range proxies fail during probing.
                if not response.read(1):
                    raise RuntimeError(f"Empty Range response from {url}")
                return {
                    "url": url,
                    "total_bytes": total,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "content_type": response.headers.get("Content-Type"),
                    "range_supported": True,
                }
        except BaseException as exc:
            last_error = exc
            if attempt == retries:
                break
            print(
                f"[http] probe attempt {attempt}/{retries} failed for {url}: {exc}; "
                f"retrying in {retry_delay:g}s",
                flush=True,
            )
            time.sleep(retry_delay)
    raise RuntimeError(f"Could not probe {url} after {retries} attempts") from last_error


def _open_range(
    remote: Mapping[str, object],
    start: int,
    *,
    timeout: float,
    end: Optional[int] = None,
) -> urllib.response.addinfourl:
    response = _request(
        str(remote["url"]), timeout=timeout, start=start, end=end
    )
    status = int(getattr(response, "status", response.getcode()))
    content_range = response.headers.get("Content-Range", "")
    if status != 206:
        response.close()
        raise RuntimeError(
            f"Resume requires HTTP 206 at byte {start}; server returned {status}"
        )
    range_start, range_end, total = _parse_content_range(content_range)
    expected_end = total - 1 if end is None else end
    if (
        range_start != start
        or range_end != expected_end
        or total != int(remote["total_bytes"])
    ):
        response.close()
        raise RuntimeError(
            f"Wrong Range response {content_range!r}; expected {start}-{expected_end}/"
            f"{remote['total_bytes']}"
        )
    return response


def _validate_remote_state(state: Optional[dict], remote: Mapping[str, object]) -> None:
    if state is None:
        return
    prior = state.get("remote")
    if not isinstance(prior, dict):
        raise ValueError("Resume state lacks remote identity")
    for key in ("url", "total_bytes"):
        if prior.get(key) != remote.get(key):
            raise RuntimeError(
                f"Remote identity changed for {key}: {prior.get(key)!r} -> "
                f"{remote.get(key)!r}"
            )
    if prior.get("etag") and remote.get("etag") and prior["etag"] != remote["etag"]:
        raise RuntimeError("Remote ETag changed; refusing byte-offset resume")


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"Unexpected EOF with {remaining} of {count} bytes remaining")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _drain(stream: BinaryIO, count: int) -> None:
    remaining = count
    while remaining:
        chunk = stream.read(min(_READ_CHUNK, remaining))
        if not chunk:
            raise EOFError(f"Unexpected EOF while draining {remaining} bytes")
        remaining -= len(chunk)


class _BoundedHashReader(io.RawIOBase):
    """Expose exactly one outer-tar payload while hashing consumed bytes."""

    def __init__(self, source: BinaryIO, size: int):
        self.source = source
        self.remaining = int(size)
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        if size is None or size < 0:
            size = min(_READ_CHUNK, self.remaining)
        else:
            size = min(size, self.remaining)
        data = self.source.read(size)
        if not data:
            raise EOFError(
                f"Unexpected EOF inside bounded tar payload; {self.remaining} remain"
            )
        self.remaining -= len(data)
        self.bytes_read += len(data)
        self.digest.update(data)
        return data

    def drain(self) -> None:
        while self.remaining:
            self.read(min(_READ_CHUNK, self.remaining))

    @property
    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def _tar_header(block: bytes) -> Optional[tarfile.TarInfo]:
    if len(block) != _BLOCK_SIZE:
        raise ValueError(f"Tar header must be {_BLOCK_SIZE} bytes")
    if block == b"\0" * _BLOCK_SIZE:
        return None
    try:
        return tarfile.TarInfo.frombuf(
            block, encoding="utf-8", errors="surrogateescape"
        )
    except tarfile.TarError as exc:
        raise RuntimeError(f"Invalid/checksum-failed outer tar header: {exc}") from exc


def _safe_basename(name: str) -> str:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise RuntimeError(f"Unsafe archive path: {name!r}")
    if len(path.parts) != 1:
        raise RuntimeError(f"Nested archive paths are not allowed: {name!r}")
    result = path.name
    if "/" in result or "\\" in result:
        raise RuntimeError(f"Unsafe archive basename: {name!r}")
    return result


def _copy_tar_member_atomic(
    source: BinaryIO,
    destination: Path,
    expected_size: int,
    *,
    allow_replace: bool,
    require_jpeg: bool,
) -> str:
    if destination.exists():
        if destination.is_file() and destination.stat().st_size == expected_size:
            _drain(source, expected_size)
            return "existing"
        if not allow_replace:
            _drain(source, expected_size)
            raise RuntimeError(
                f"Existing final file does not match archive member: {destination}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    remaining = expected_size
    first = b""
    marker_tail = b""
    file_tail = b""
    saw_eoi = False
    try:
        with open(temporary, "wb") as handle:
            while remaining:
                chunk = source.read(min(_READ_CHUNK, remaining))
                if not chunk:
                    raise EOFError(
                        f"Unexpected EOF extracting {destination}; {remaining} remain"
                    )
                if len(first) < 8:
                    first = (first + chunk)[:8]
                marker_scan = marker_tail + chunk
                if b"\xff\xd9" in marker_scan:
                    saw_eoi = True
                marker_tail = marker_scan[-1:]
                file_tail = (file_tail + chunk)[-12:]
                handle.write(chunk)
                remaining -= len(chunk)
        is_jpeg = first.startswith(b"\xff\xd8") and saw_eoi
        # The official ILSVRC2012 train archive contains one valid PNG named
        # n02105855_2933.JPEG.  Keep its official name and exact bytes: PIL and
        # torchvision dispatch by the file signature, not by this suffix.
        is_png = (
            first == b"\x89PNG\r\n\x1a\n"
            and file_tail == b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
        )
        if require_jpeg and not (is_jpeg or is_png):
            raise RuntimeError(
                f"JPEG/PNG marker validation failed for {destination}"
            )
        os.replace(temporary, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return "written"


def _directory_file_sizes(directory: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise RuntimeError(f"Unexpected non-regular entry in class dir: {entry.path}")
            if entry.name.startswith(".") and entry.name.endswith(".partial"):
                raise RuntimeError(f"Stale partial file requires inspection: {entry.path}")
            result[entry.name] = entry.stat(follow_symlinks=False).st_size
    return result


def _extract_nested_train_class(
    payload: _BoundedHashReader,
    *,
    wnid: str,
    train_root: Path,
) -> tuple[int, dict[str, int]]:
    partial_root = train_root / ".partial"
    staging = partial_root / wnid
    final = train_root / wnid
    if staging.exists() and final.exists():
        raise RuntimeError(f"Both staging and final class directories exist for {wnid}")
    validate_only = final.exists()
    work = final if validate_only else staging
    if work.exists() and (work.is_symlink() or not work.is_dir()):
        raise RuntimeError(f"Class destination is not a real directory: {work}")
    if not work.exists():
        work.mkdir(parents=True, exist_ok=False)

    expected: dict[str, int] = {}
    with tarfile.open(fileobj=payload, mode="r|") as nested:
        for member in nested:
            if member.isdir():
                # Official class tars are flat; harmless '.' directory headers
                # may still be present in repacks.
                _safe_basename(member.name.rstrip("/"))
                continue
            if not member.isreg():
                raise RuntimeError(
                    f"Unsafe/non-regular member in {wnid}.tar: {member.name!r} "
                    f"type={member.type!r}"
                )
            basename = _safe_basename(member.name)
            if Path(basename).suffix.lower() not in _JPEG_SUFFIXES:
                raise RuntimeError(f"Non-JPEG member in {wnid}.tar: {basename}")
            if not basename.startswith(wnid + "_"):
                raise RuntimeError(
                    f"Train image {basename!r} does not match enclosing WNID {wnid}"
                )
            if basename in expected:
                raise RuntimeError(f"Duplicate member {basename!r} in {wnid}.tar")
            expected[basename] = int(member.size)
            member_stream = nested.extractfile(member)
            if member_stream is None:
                raise RuntimeError(f"Could not read regular member {basename}")
            if validate_only:
                destination = work / basename
                if not destination.is_file() or destination.stat().st_size != member.size:
                    _drain(member_stream, int(member.size))
                    raise RuntimeError(
                        f"Pre-existing final class {wnid} is incomplete/mismatched at "
                        f"{destination}; refusing to modify it"
                    )
                _drain(member_stream, int(member.size))
            else:
                _copy_tar_member_atomic(
                    member_stream,
                    work / basename,
                    int(member.size),
                    allow_replace=True,
                    require_jpeg=True,
                )

    payload.drain()
    actual = _directory_file_sizes(work)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        wrong = sorted(
            name for name in set(expected) & set(actual) if expected[name] != actual[name]
        )[:5]
        raise RuntimeError(
            f"Class {wnid} validation failed: expected={len(expected)} "
            f"actual={len(actual)} missing={missing} extra={extra} wrong_size={wrong}"
        )
    if not expected:
        raise RuntimeError(f"Class {wnid} contained no JPEGs")
    if not validate_only:
        final.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(final)
    return len(expected), expected


def _read_range_with_retry(
    remote: Mapping[str, object],
    start: int,
    end: int,
    *,
    timeout: float,
    retries: int,
    retry_delay: float,
    label: str,
) -> bytes:
    """Fetch one exact inclusive byte range and reject short/overlong replies."""
    expected = end - start + 1
    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            with _open_range(remote, start, end=end, timeout=timeout) as response:
                data = _read_exact(response, expected)
                if response.read(1):
                    raise RuntimeError(f"Server returned excess data for {start}-{end}")
                return data
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            print(
                f"[{label}] Range {start}-{end} attempt {attempt}/{retries} failed: "
                f"{exc}; retrying in {retry_delay:g}s",
                flush=True,
            )
            time.sleep(retry_delay)
    raise RuntimeError(
        f"Could not fetch exact Range {start}-{end} for {label}"
    ) from last_error


def _validate_train_index(
    state: Mapping[str, object],
    remote: Mapping[str, object],
    expected_wnids: set[str],
) -> list[dict]:
    _validate_remote_state(dict(state), remote)
    if state.get("complete") is not True:
        raise RuntimeError("Outer train index is not complete")
    raw_entries = state.get("entries")
    if not isinstance(raw_entries, list):
        raise RuntimeError("Outer train index has no entry list")
    entries: list[dict] = []
    expected_header = 0
    seen: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise RuntimeError("Invalid outer train index entry")
        entry = dict(raw)
        wnid = str(entry.get("wnid", ""))
        header_offset = int(entry.get("header_offset", -1))
        payload_offset = int(entry.get("payload_offset", -1))
        size = int(entry.get("size", -1))
        next_offset = int(entry.get("next_offset", -1))
        calculated_next = payload_offset + size + (-size) % _BLOCK_SIZE
        if (
            not _WNID_RE.fullmatch(wnid)
            or wnid in seen
            or header_offset != expected_header
            or payload_offset != header_offset + _BLOCK_SIZE
            or size <= 0
            or next_offset != calculated_next
            or next_offset > int(remote["total_bytes"])
        ):
            raise RuntimeError(f"Invalid outer train index entry: {entry}")
        seen.add(wnid)
        entries.append(entry)
        expected_header = next_offset
    if seen != expected_wnids:
        raise RuntimeError(
            f"Outer train WNIDs differ from devkit: "
            f"missing={sorted(expected_wnids-seen)[:5]} extra={sorted(seen-expected_wnids)[:5]}"
        )
    archive_end = int(state.get("archive_end_offset", -1))
    if archive_end != expected_header + 2 * _BLOCK_SIZE:
        raise RuntimeError(
            f"Invalid outer train end offset {archive_end}; expected "
            f"{expected_header + 2 * _BLOCK_SIZE}"
        )
    return entries


def _build_train_outer_index(
    *,
    state_root: Path,
    remote: Mapping[str, object],
    timeout: float,
    retries: int,
    retry_delay: float,
    expected_wnids: set[str],
) -> list[dict]:
    """Index an outer tar using only one exact 512-byte request per header."""
    path = state_root / "train_outer_index.json"
    state = _load_json(path)
    _validate_remote_state(state, remote)
    if state is None:
        state = {
            "schema_version": 2,
            "remote": dict(remote),
            "entries": [],
            "next_header_offset": 0,
            "zero_blocks": 0,
            "complete": False,
            "updated_at_utc": _utc_now(),
        }
    if state.get("complete") is True:
        entries = _validate_train_index(state, remote, expected_wnids)
        print(f"[train-index] reused {len(entries)} indexed class ranges", flush=True)
        return entries

    raw_entries = state.get("entries")
    if not isinstance(raw_entries, list):
        raise RuntimeError("Corrupt partial train index: entries is not a list")
    entries = [dict(item) for item in raw_entries]
    offset = int(state.get("next_header_offset", 0))
    zero_blocks = int(state.get("zero_blocks", 0))
    seen = {str(entry.get("wnid")) for entry in entries}
    if len(seen) != len(entries):
        raise RuntimeError("Corrupt partial train index: duplicate WNID")
    last_member_end = int(entries[-1]["next_offset"]) if entries else 0
    if offset != last_member_end + zero_blocks * _BLOCK_SIZE:
        raise RuntimeError("Corrupt partial train index: initial offset mismatch")

    total = int(remote["total_bytes"])
    print(
        f"[train-index] resume header={offset:,}; indexed={len(entries)}/"
        f"{len(expected_wnids)}",
        flush=True,
    )
    while offset + _BLOCK_SIZE <= total:
        header_offset = offset
        block = _read_range_with_retry(
            remote,
            header_offset,
            header_offset + _BLOCK_SIZE - 1,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            label="train-index",
        )
        info = _tar_header(block)
        if info is None:
            zero_blocks += 1
            offset += _BLOCK_SIZE
            state["next_header_offset"] = offset
            state["zero_blocks"] = zero_blocks
            state["updated_at_utc"] = _utc_now()
            _atomic_write_json(path, state)
            if zero_blocks < 2:
                continue
            trailing = total - offset
            if trailing > 1024 * 1024:
                raise RuntimeError(
                    f"Suspicious {trailing:,}-byte tail after outer tar end markers"
                )
            if trailing:
                tail = _read_range_with_retry(
                    remote,
                    offset,
                    total - 1,
                    timeout=timeout,
                    retries=retries,
                    retry_delay=retry_delay,
                    label="train-index-tail",
                )
                if tail.strip(b"\0"):
                    raise RuntimeError("Non-zero bytes follow outer tar end markers")
            state["complete"] = True
            state["archive_end_offset"] = offset
            state["updated_at_utc"] = _utc_now()
            _atomic_write_json(path, state)
            break
        if zero_blocks:
            raise RuntimeError("Non-zero header follows outer tar end marker")
        if not info.isreg():
            raise RuntimeError(
                f"Unexpected outer train member type={info.type!r}: {info.name!r}"
            )
        basename = _safe_basename(info.name)
        match = _TRAIN_TAR_RE.fullmatch(basename)
        if match is None:
            raise RuntimeError(f"Unexpected outer train member: {info.name!r}")
        wnid = match.group(1)
        if wnid in seen:
            raise RuntimeError(f"Duplicate outer train class {wnid}")
        size = int(info.size)
        payload_offset = header_offset + _BLOCK_SIZE
        next_offset = payload_offset + size + (-size) % _BLOCK_SIZE
        if size <= 0 or next_offset > total:
            raise RuntimeError(
                f"Invalid outer member bounds for {wnid}: size={size} next={next_offset}"
            )
        entry = {
            "wnid": wnid,
            "header_offset": header_offset,
            "payload_offset": payload_offset,
            "size": size,
            "next_offset": next_offset,
        }
        entries.append(entry)
        seen.add(wnid)
        offset = next_offset
        state["entries"] = entries
        state["next_header_offset"] = offset
        state["updated_at_utc"] = _utc_now()
        _atomic_write_json(path, state)
        if len(entries) % 25 == 0 or len(entries) == len(expected_wnids):
            print(
                f"[train-index] indexed={len(entries)}/{len(expected_wnids)} "
                f"next_header={offset:,}",
                flush=True,
            )

    if state.get("complete") is not True:
        raise RuntimeError("Train outer index reached remote EOF before two zero blocks")
    return _validate_train_index(state, remote, expected_wnids)


def _file_sizes_sha256(file_sizes: Mapping[str, int]) -> str:
    digest = hashlib.sha256()
    for name in sorted(file_sizes):
        digest.update(name.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(int(file_sizes[name])).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_train_marker(
    marker: Mapping[str, object], entry: Mapping[str, object], train_root: Path
) -> dict:
    wnid = str(entry["wnid"])
    for key, expected in (
        ("wnid", wnid),
        ("outer_header_offset", int(entry["header_offset"])),
        ("outer_member_size", int(entry["size"])),
    ):
        if marker.get(key) != expected:
            raise RuntimeError(
                f"Class marker/index mismatch for {wnid}: {key}={marker.get(key)!r}, "
                f"expected {expected!r}"
            )
    if marker.get("complete") is not True:
        raise RuntimeError(f"Incomplete class marker for {wnid}")
    class_dir = train_root / wnid
    if class_dir.is_symlink() or not class_dir.is_dir():
        raise RuntimeError(f"Committed marker lacks final class directory: {class_dir}")
    actual = _directory_file_sizes(class_dir)
    if len(actual) != int(marker.get("image_count", -1)):
        raise RuntimeError(f"Committed class file-count changed for {wnid}")
    actual_digest = _file_sizes_sha256(actual)
    recorded_digest = marker.get("file_sizes_sha256")
    if recorded_digest is not None and recorded_digest != actual_digest:
        raise RuntimeError(f"Committed class filename/size fingerprint changed for {wnid}")
    result = dict(marker)
    if recorded_digest is None:
        # Upgrade markers written by the older sequential implementation.
        result["file_sizes_sha256"] = actual_digest
    return result


def _extract_train_range(
    *,
    entry: Mapping[str, object],
    marker_root: Path,
    train_root: Path,
    remote: Mapping[str, object],
    timeout: float,
    retries: int,
    retry_delay: float,
) -> dict:
    wnid = str(entry["wnid"])
    marker_path = marker_root / f"{wnid}.json"
    prior = _load_json(marker_path)
    if prior is not None:
        marker = _validate_train_marker(prior, entry, train_root)
        if marker != prior:
            _atomic_write_json(marker_path, marker)
        return marker

    start = int(entry["payload_offset"])
    size = int(entry["size"])
    end = start + size - 1
    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            with _open_range(remote, start, end=end, timeout=timeout) as response:
                payload = _BoundedHashReader(response, size)
                image_count, file_sizes = _extract_nested_train_class(
                    payload, wnid=wnid, train_root=train_root
                )
                if response.read(1):
                    raise RuntimeError(f"Server returned excess class bytes for {wnid}")
                marker = {
                    "schema_version": 2,
                    "complete": True,
                    "wnid": wnid,
                    "outer_header_offset": int(entry["header_offset"]),
                    "outer_member_size": size,
                    "payload_range": [start, end],
                    "next_offset": int(entry["next_offset"]),
                    "image_count": image_count,
                    "nested_tar_sha256": payload.hexdigest,
                    "file_sizes_sha256": _file_sizes_sha256(file_sizes),
                    "completed_at_utc": _utc_now(),
                }
                _atomic_write_json(marker_path, marker)
                return marker
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            print(
                f"[train:{wnid}] attempt {attempt}/{retries} failed: {exc}; "
                f"restarting its class Range in {retry_delay:g}s",
                flush=True,
            )
            time.sleep(retry_delay)
    raise RuntimeError(f"Could not extract class {wnid} after {retries} attempts") from last_error


def _write_train_progress(
    path: Path,
    remote: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
    markers: Mapping[str, Mapping[str, object]],
    *,
    expected_files: int,
) -> dict:
    classes = len(markers)
    images = sum(int(marker["image_count"]) for marker in markers.values())
    payload_bytes = sum(
        int(entry["size"]) for entry in entries if str(entry["wnid"]) in markers
    )
    all_classes = classes == len(entries)
    state = {
        "schema_version": 2,
        "remote": dict(remote),
        "index_complete": True,
        "classes_completed": classes,
        "images_completed": images,
        "payload_bytes_completed": payload_bytes,
        "archive_complete": bool(all_classes and images == expected_files),
        "updated_at_utc": _utc_now(),
    }
    _atomic_write_json(path, state)
    return state


def _stream_train(
    *,
    output_root: Path,
    state_root: Path,
    remote: Mapping[str, object],
    timeout: float,
    retries: int,
    retry_delay: float,
    workers: int,
    expected_wnids: set[str],
    expected_files: int,
    debug_max_classes: int,
) -> dict:
    train_root = output_root / "train"
    marker_root = state_root / "train_classes"
    state_path = state_root / "train_progress.json"
    train_root.mkdir(parents=True, exist_ok=True)
    marker_root.mkdir(parents=True, exist_ok=True)
    entries = _build_train_outer_index(
        state_root=state_root,
        remote=remote,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        expected_wnids=expected_wnids,
    )

    by_wnid = {str(entry["wnid"]): entry for entry in entries}
    markers: dict[str, dict] = {}
    for path in sorted(marker_root.glob("n????????.json")):
        wnid = path.stem
        if wnid not in by_wnid:
            raise RuntimeError(f"Class marker is absent from outer index: {path}")
        loaded = _load_json(path)
        if loaded is None:
            continue
        marker = _validate_train_marker(loaded, by_wnid[wnid], train_root)
        if marker != loaded:
            _atomic_write_json(path, marker)
        markers[wnid] = marker

    state = _write_train_progress(
        state_path, remote, entries, markers, expected_files=expected_files
    )
    if state["archive_complete"]:
        try:
            (train_root / ".partial").rmdir()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise RuntimeError(
                "Completed train archive still has non-empty staging data"
            ) from exc
        print("[train] all indexed classes already committed", flush=True)
        return state

    pending = [entry for entry in entries if str(entry["wnid"]) not in markers]
    if debug_max_classes:
        pending = pending[:debug_max_classes]
    print(
        f"[train] exact-Range workers={min(workers, max(1, len(pending)))} "
        f"pending_this_run={len(pending)} committed={len(markers)}/{len(entries)}",
        flush=True,
    )
    failures: list[tuple[str, BaseException]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {
                executor.submit(
                    _extract_train_range,
                    entry=entry,
                    marker_root=marker_root,
                    train_root=train_root,
                    remote=remote,
                    timeout=timeout,
                    retries=retries,
                    retry_delay=retry_delay,
                ): entry
                for entry in pending
            }
            for future in as_completed(futures):
                entry = futures[future]
                wnid = str(entry["wnid"])
                try:
                    markers[wnid] = future.result()
                except BaseException as exc:
                    failures.append((wnid, exc))
                    print(f"[train:{wnid}] FAILED: {exc}", file=sys.stderr, flush=True)
                    continue
                state = _write_train_progress(
                    state_path, remote, entries, markers, expected_files=expected_files
                )
                print(
                    f"[train] committed {wnid}: {markers[wnid]['image_count']:,} images; "
                    f"classes={state['classes_completed']}/{len(entries)} "
                    f"images={state['images_completed']:,}/{expected_files:,}",
                    flush=True,
                )
    state = _write_train_progress(
        state_path, remote, entries, markers, expected_files=expected_files
    )
    if failures:
        names = ", ".join(wnid for wnid, _ in failures[:10])
        raise RuntimeError(f"{len(failures)} train class Range(s) failed: {names}") from failures[0][1]
    if debug_max_classes and len(markers) < len(entries):
        print(f"[train] debug stop after {len(pending)} new classes", flush=True)
        return state
    if len(markers) != len(entries):
        raise RuntimeError(f"Train class count mismatch: {len(markers)} != {len(entries)}")
    if int(state["images_completed"]) != expected_files:
        raise RuntimeError(
            f"Train file count mismatch: {state['images_completed']} != {expected_files}"
        )
    if state["archive_complete"] is not True:
        raise RuntimeError("Train progress failed to reach complete state")
    try:
        (train_root / ".partial").rmdir()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise RuntimeError(
            "Completed train archive still has non-empty staging data"
        ) from exc
    return state


def _download_devkit(
    *,
    state_root: Path,
    url: str,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> tuple[Path, Path]:
    devkit_root = state_root / "devkit"
    meta_path = devkit_root / "meta.mat"
    groundtruth_path = devkit_root / "ILSVRC2012_validation_ground_truth.txt"
    if meta_path.is_file() and groundtruth_path.is_file():
        return meta_path, groundtruth_path

    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[devkit] streaming {url}", flush=True)
            with _request(url, timeout=timeout, start=None) as response:
                with tarfile.open(fileobj=response, mode="r|gz") as archive:
                    found: set[str] = set()
                    for member in archive:
                        if not member.isreg():
                            continue
                        basename = PurePosixPath(member.name.replace("\\", "/")).name
                        if basename not in {
                            "meta.mat",
                            "ILSVRC2012_validation_ground_truth.txt",
                        }:
                            continue
                        # We never honor the archive path, only an allowlisted basename.
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuntimeError(f"Could not read devkit member {member.name}")
                        data = source.read()
                        if len(data) != member.size:
                            raise EOFError(f"Short devkit member {member.name}")
                        destination = (
                            meta_path if basename == "meta.mat" else groundtruth_path
                        )
                        _atomic_write_bytes(destination, data)
                        found.add(basename)
            if meta_path.is_file() and groundtruth_path.is_file():
                return meta_path, groundtruth_path
            raise RuntimeError(f"Devkit lacked required files; found={sorted(found)}")
        except BaseException as exc:
            last_error = exc
            if attempt == retries:
                break
            print(
                f"[devkit] attempt {attempt}/{retries} failed: {exc}; "
                f"retrying in {retry_delay:g}s",
                flush=True,
            )
            time.sleep(retry_delay)
    raise RuntimeError(f"Could not stream official devkit from {url}") from last_error


def _matlab_text(value: object) -> str:
    import numpy as np

    array = np.asarray(value)
    if array.size == 1:
        item = array.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return "".join(str(item) for item in array.reshape(-1))


def _load_devkit_mapping(
    meta_path: Path,
    groundtruth_path: Path,
    *,
    expected_classes: int,
    expected_val_files: int,
) -> tuple[dict[int, str], list[int]]:
    import numpy as np
    from scipy.io import loadmat

    payload = loadmat(meta_path, squeeze_me=True, struct_as_record=False)
    if "synsets" not in payload:
        raise RuntimeError(f"Devkit meta lacks 'synsets': {meta_path}")
    mapping: dict[int, str] = {}
    for synset in np.atleast_1d(payload["synsets"]).reshape(-1):
        try:
            class_id = int(np.asarray(synset.ILSVRC2012_ID).reshape(-1)[0])
            wnid = _matlab_text(synset.WNID).strip()
        except Exception as exc:
            raise RuntimeError("Unsupported official meta.mat synset structure") from exc
        if 1 <= class_id <= expected_classes:
            if not _WNID_RE.fullmatch(wnid):
                raise RuntimeError(f"Invalid WNID {wnid!r} for class id {class_id}")
            if class_id in mapping:
                raise RuntimeError(f"Duplicate ImageNet class id {class_id}")
            mapping[class_id] = wnid
    if len(mapping) != expected_classes or set(mapping) != set(
        range(1, expected_classes + 1)
    ):
        raise RuntimeError(
            f"Devkit mapping must contain ids 1..{expected_classes}; got {len(mapping)}"
        )
    if len(set(mapping.values())) != expected_classes:
        raise RuntimeError("Devkit WNIDs are not unique")

    with open(groundtruth_path, "r", encoding="ascii") as handle:
        labels = [int(line.strip()) for line in handle if line.strip()]
    if len(labels) != expected_val_files:
        raise RuntimeError(
            f"Validation ground truth has {len(labels)} rows, expected {expected_val_files}"
        )
    unknown = sorted(set(labels) - set(mapping))
    if unknown:
        raise RuntimeError(f"Validation ground truth references unknown ids: {unknown[:10]}")
    return mapping, labels


def _initial_val_state(state_path: Path, remote: Mapping[str, object]) -> dict:
    state = _load_json(state_path)
    _validate_remote_state(state, remote)
    if state is None:
        state = {
            "schema_version": 1,
            "remote": dict(remote),
            "next_offset": 0,
            "images_completed": 0,
            "archive_complete": False,
            "updated_at_utc": _utc_now(),
        }
    return state


def _stream_val(
    *,
    output_root: Path,
    state_root: Path,
    remote: Mapping[str, object],
    timeout: float,
    id_to_wnid: Mapping[int, str],
    groundtruth: Sequence[int],
    expected_files: int,
    checkpoint_every: int,
    debug_max_files: int,
    local_archive: Optional[Path] = None,
) -> dict:
    val_root = output_root / "val"
    state_path = state_root / "val_progress.json"
    val_root.mkdir(parents=True, exist_ok=True)
    state = _initial_val_state(state_path, remote)
    if bool(state.get("archive_complete")):
        print("[val] resume state already marks archive complete", flush=True)
        return state

    start = int(state["next_offset"])
    total = int(remote["total_bytes"])
    if not 0 <= start < total:
        raise RuntimeError(f"Invalid val resume offset {start} for size {total}")
    since_checkpoint = 0
    committed_this_run = 0
    checkpoint_offset = start
    checkpoint_images = int(state["images_completed"])
    print(
        f"[val] Range resume byte={start:,}/{total:,} "
        f"images>={state['images_completed']}/{expected_files}",
        flush=True,
    )
    if local_archive is None:
        stream_context = _open_range(remote, start, timeout=timeout)
    else:
        local_stream = local_archive.open("rb")
        local_stream.seek(start)
        stream_context = local_stream
        print(
            f"[val] reading verified local archive from byte {start:,}: "
            f"{local_archive}",
            flush=True,
        )
    with stream_context as response:
        offset = start
        zero_blocks = 0
        while offset < total:
            header_offset = offset
            header = _read_exact(response, _BLOCK_SIZE)
            offset += _BLOCK_SIZE
            info = _tar_header(header)
            if info is None:
                zero_blocks += 1
                if zero_blocks >= 2:
                    state["archive_complete"] = True
                    state["next_offset"] = offset
                    state["images_completed"] = max(
                        int(state["images_completed"]), checkpoint_images
                    )
                    state["updated_at_utc"] = _utc_now()
                    _atomic_write_json(state_path, state)
                    break
                continue
            if zero_blocks:
                raise RuntimeError("Non-zero val header followed an archive end block")
            size = int(info.size)
            padding = (-size) % _BLOCK_SIZE
            next_offset = offset + size + padding
            if info.isdir():
                _drain(response, size + padding)
                offset = next_offset
                continue
            if not info.isreg():
                raise RuntimeError(
                    f"Unexpected validation tar member type={info.type!r}: {info.name!r}"
                )
            basename = _safe_basename(info.name)
            match = _VAL_IMAGE_RE.fullmatch(basename)
            if match is None:
                raise RuntimeError(f"Unexpected validation member: {basename!r}")
            image_index = int(match.group(1))
            if not 1 <= image_index <= expected_files:
                raise RuntimeError(f"Validation image index out of range: {basename}")
            wnid = id_to_wnid[int(groundtruth[image_index - 1])]
            destination = val_root / wnid / basename
            _copy_tar_member_atomic(
                response,
                destination,
                size,
                allow_replace=False,
                require_jpeg=True,
            )
            _drain(response, padding)
            offset = next_offset
            committed_this_run += 1
            since_checkpoint += 1
            checkpoint_images += 1
            checkpoint_offset = next_offset
            if since_checkpoint >= checkpoint_every:
                state["next_offset"] = checkpoint_offset
                state["images_completed"] = checkpoint_images
                state["updated_at_utc"] = _utc_now()
                _atomic_write_json(state_path, state)
                since_checkpoint = 0
                print(
                    f"[val] committed through {basename}; "
                    f"images>={checkpoint_images:,}/{expected_files:,} "
                    f"next_byte={checkpoint_offset:,}",
                    flush=True,
                )
            if debug_max_files and committed_this_run >= debug_max_files:
                state["next_offset"] = checkpoint_offset
                state["images_completed"] = checkpoint_images
                state["updated_at_utc"] = _utc_now()
                _atomic_write_json(state_path, state)
                print(f"[val] debug stop after {committed_this_run} files", flush=True)
                return state

    if not state.get("archive_complete"):
        raise RuntimeError("Validation HTTP stream ended before archive end markers")
    return state


def _sorted_wnid_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    result: list[str] = []
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                raise RuntimeError(f"Unexpected non-directory split entry: {entry.path}")
            if not _WNID_RE.fullmatch(entry.name):
                raise RuntimeError(f"Unexpected non-WNID split directory: {entry.path}")
            result.append(entry.name)
    return sorted(result)


def _verify_train_tree(
    train_root: Path, expected_wnids: set[str], expected_files: int
) -> tuple[int, dict[str, int]]:
    wnids = _sorted_wnid_dirs(train_root)
    if set(wnids) != expected_wnids:
        raise RuntimeError(
            f"Train WNID set mismatch: missing={sorted(expected_wnids-set(wnids))[:5]} "
            f"extra={sorted(set(wnids)-expected_wnids)[:5]}"
        )
    per_class: dict[str, int] = {}
    total = 0
    for wnid in wnids:
        count = 0
        with os.scandir(train_root / wnid) as entries:
            for entry in entries:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise RuntimeError(f"Unsafe train entry: {entry.path}")
                if Path(entry.name).suffix.lower() not in _JPEG_SUFFIXES:
                    raise RuntimeError(f"Non-JPEG train file: {entry.path}")
                if not entry.name.startswith(wnid + "_"):
                    raise RuntimeError(f"Train filename/WNID mismatch: {entry.path}")
                if entry.stat(follow_symlinks=False).st_size <= 0:
                    raise RuntimeError(f"Empty train file: {entry.path}")
                count += 1
        per_class[wnid] = count
        total += count
    if total != expected_files:
        raise RuntimeError(f"Train file count {total} != expected {expected_files}")
    return total, per_class


def _verify_val_tree(
    val_root: Path,
    expected_wnids: set[str],
    id_to_wnid: Mapping[int, str],
    groundtruth: Sequence[int],
    expected_files: int,
) -> tuple[int, dict[str, int]]:
    wnids = _sorted_wnid_dirs(val_root)
    if set(wnids) != expected_wnids:
        raise RuntimeError(
            f"Val WNID set mismatch: missing={sorted(expected_wnids-set(wnids))[:5]} "
            f"extra={sorted(set(wnids)-expected_wnids)[:5]}"
        )
    seen: set[int] = set()
    per_class: dict[str, int] = {}
    for wnid in wnids:
        count = 0
        with os.scandir(val_root / wnid) as entries:
            for entry in entries:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise RuntimeError(f"Unsafe val entry: {entry.path}")
                match = _VAL_IMAGE_RE.fullmatch(entry.name)
                if match is None:
                    raise RuntimeError(f"Unexpected val filename: {entry.path}")
                index = int(match.group(1))
                if index in seen or not 1 <= index <= expected_files:
                    raise RuntimeError(f"Duplicate/out-of-range val image: {entry.path}")
                expected_wnid = id_to_wnid[int(groundtruth[index - 1])]
                if wnid != expected_wnid:
                    raise RuntimeError(
                        f"Val ground-truth mismatch for {entry.name}: {wnid} != {expected_wnid}"
                    )
                if entry.stat(follow_symlinks=False).st_size <= 0:
                    raise RuntimeError(f"Empty val file: {entry.path}")
                seen.add(index)
                count += 1
        per_class[wnid] = count
    if len(seen) != expected_files or seen != set(range(1, expected_files + 1)):
        raise RuntimeError(f"Val file/index count {len(seen)} != {expected_files}")
    return len(seen), per_class


def _mapping_sha256(id_to_wnid: Mapping[int, str]) -> str:
    canonical = json.dumps(
        {str(key): id_to_wnid[key] for key in sorted(id_to_wnid)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _validate_local_val_archive(
    path: Path, *, expected_bytes: int, expected_md5: str
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Local validation archive is missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Local validation archive size mismatch: "
            f"{actual_bytes:,} != {expected_bytes:,}"
        )
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    actual_md5 = digest.hexdigest()
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"Local validation archive MD5 mismatch: "
            f"{actual_md5} != {expected_md5}"
        )
    print(
        f"[val] verified local archive bytes={actual_bytes:,} md5={actual_md5}",
        flush=True,
    )
    return {
        "path": str(path),
        "bytes": actual_bytes,
        "md5": actual_md5,
        "verified": True,
    }


def _write_manifest(
    *,
    output_root: Path,
    state_root: Path,
    args: argparse.Namespace,
    id_to_wnid: Mapping[int, str],
    train_remote: Optional[Mapping[str, object]],
    val_remote: Optional[Mapping[str, object]],
) -> dict:
    expected_wnids = set(id_to_wnid.values())
    train_state = _load_json(state_root / "train_progress.json") or {}
    val_state = _load_json(state_root / "val_progress.json") or {}
    train_ready = bool(train_state.get("archive_complete"))
    val_ready = bool(val_state.get("archive_complete"))
    train_verification: Optional[dict] = None
    val_verification: Optional[dict] = None
    if train_ready:
        total, per_class = _verify_train_tree(
            output_root / "train", expected_wnids, args.expected_train_files
        )
        train_verification = {
            "complete": True,
            "class_count": len(per_class),
            "file_count": total,
            "per_class_file_count": per_class,
        }
    if val_ready:
        _, groundtruth_path = _download_devkit(
            state_root=state_root,
            url=args.devkit_url,
            timeout=args.timeout,
            retries=args.http_retries,
            retry_delay=args.retry_delay,
        )
        _, groundtruth = _load_devkit_mapping(
            state_root / "devkit" / "meta.mat",
            groundtruth_path,
            expected_classes=args.expected_classes,
            expected_val_files=args.expected_val_files,
        )
        total, per_class = _verify_val_tree(
            output_root / "val",
            expected_wnids,
            id_to_wnid,
            groundtruth,
            args.expected_val_files,
        )
        val_verification = {
            "complete": True,
            "class_count": len(per_class),
            "file_count": total,
            "per_class_file_count": per_class,
        }

    sorted_wnids = sorted(expected_wnids)
    reference_verification: Optional[dict] = None
    if args.reference_class_root is not None:
        reference = _sorted_wnid_dirs(args.reference_class_root.resolve())
        if reference != sorted_wnids:
            raise RuntimeError(
                f"Reference ImageFolder mapping differs: {args.reference_class_root}"
            )
        reference_verification = {
            "path": str(args.reference_class_root.resolve()),
            "class_count": len(reference),
            "sorted_wnids_identical": True,
        }

    complete = bool(train_verification and val_verification)
    manifest = {
        "schema_version": 1,
        "complete": complete,
        "created_at_utc": _utc_now(),
        "output_root": str(output_root),
        "state_root": str(state_root),
        "source": {
            "train": dict(train_remote) if train_remote is not None else None,
            "val": dict(val_remote) if val_remote is not None else None,
            "devkit_url": args.devkit_url,
            "val_local_archive": (
                {
                    "path": str(args.val_local_archive),
                    "bytes": args.expected_val_archive_bytes,
                    "md5": args.expected_val_archive_md5,
                    "verified_before_extraction": True,
                }
                if args.val_local_archive is not None
                else None
            ),
        },
        "expected": {
            "classes": args.expected_classes,
            "train_files": args.expected_train_files,
            "val_files": args.expected_val_files,
        },
        "train": train_verification
        or {
            "complete": False,
            "classes_committed": int(train_state.get("classes_completed", 0)),
            "images_committed": int(train_state.get("images_completed", 0)),
        },
        "val": val_verification
        or {
            "complete": False,
            "images_checkpointed": int(val_state.get("images_completed", 0)),
        },
        "mapping": {
            "official_imagenet_id_to_wnid": {
                str(key): id_to_wnid[key] for key in sorted(id_to_wnid)
            },
            "official_mapping_sha256": _mapping_sha256(id_to_wnid),
            "imagefolder_class_to_index": {
                wnid: index for index, wnid in enumerate(sorted_wnids)
            },
            "train_val_devkit_wnids_identical": complete,
            "reference": reference_verification,
        },
    }
    _atomic_write_json(output_root / "raw_imagenet_manifest.json", manifest)
    return manifest


def _progress_offset(path: Path) -> int:
    state = _load_json(path)
    return int(state.get("next_offset", 0)) if state else 0


def _train_payload_completed(path: Path) -> int:
    state = _load_json(path)
    return int(state.get("payload_bytes_completed", 0)) if state else 0


def _preflight_space(
    *,
    output_root: Path,
    state_root: Path,
    parts: Sequence[str],
    train_remote: Optional[Mapping[str, object]],
    val_remote: Optional[Mapping[str, object]],
    reserve_bytes: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    remaining = 0
    if "train" in parts and train_remote is not None:
        remaining += max(
            0,
            int(train_remote["total_bytes"])
            - _train_payload_completed(state_root / "train_progress.json"),
        )
    if "val" in parts and val_remote is not None:
        remaining += max(
            0,
            int(val_remote["total_bytes"])
            - _progress_offset(state_root / "val_progress.json"),
        )
    required = remaining + reserve_bytes
    free = shutil.disk_usage(output_root).free
    print(
        f"[space] free={free:,} conservative_remaining={remaining:,} "
        f"reserve={reserve_bytes:,} required={required:,}",
        flush=True,
    )
    if free < required:
        raise RuntimeError(
            f"Insufficient free space: {free:,} bytes available, {required:,} required. "
            "The downloader will not store either outer archive; free space is "
            "needed for extracted JPEGs plus the configured reserve."
        )


def main() -> None:
    args = _parse_args()
    output_root: Path = args.output_root
    state_root: Path = args.state_root
    output_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    train_remote: Optional[dict[str, object]] = None
    val_remote: Optional[dict[str, object]] = None
    if "train" in args.parts:
        train_remote = _probe_remote(
            args.train_url,
            timeout=args.timeout,
            retries=args.http_retries,
            retry_delay=args.retry_delay,
            expected_bytes=args.expected_train_archive_bytes,
        )
    if "val" in args.parts:
        val_remote = _probe_remote(
            args.val_url,
            timeout=args.timeout,
            retries=args.http_retries,
            retry_delay=args.retry_delay,
            expected_bytes=args.expected_val_archive_bytes,
        )
    _preflight_space(
        output_root=output_root,
        state_root=state_root,
        parts=args.parts,
        train_remote=train_remote,
        val_remote=val_remote,
        reserve_bytes=args.reserve_bytes,
    )

    # Devkit is tiny and required for mapping validation even when only train
    # extraction is requested. Reuse atomically cached allowlisted files.
    meta_path, groundtruth_path = _download_devkit(
        state_root=state_root,
        url=args.devkit_url,
        timeout=args.timeout,
        retries=args.http_retries,
        retry_delay=args.retry_delay,
    )
    id_to_wnid, groundtruth = _load_devkit_mapping(
        meta_path,
        groundtruth_path,
        expected_classes=args.expected_classes,
        expected_val_files=args.expected_val_files,
    )
    _atomic_write_json(
        state_root / "class_mapping.json",
        {
            "schema_version": 1,
            "id_to_wnid": {str(k): id_to_wnid[k] for k in sorted(id_to_wnid)},
            "mapping_sha256": _mapping_sha256(id_to_wnid),
            "created_at_utc": _utc_now(),
        },
    )

    if "train" in args.parts:
        assert train_remote is not None
        _stream_train(
            output_root=output_root,
            state_root=state_root,
            remote=train_remote,
            timeout=args.timeout,
            retries=args.http_retries,
            retry_delay=args.retry_delay,
            workers=args.train_workers,
            expected_wnids=set(id_to_wnid.values()),
            expected_files=args.expected_train_files,
            debug_max_classes=args.debug_max_train_classes,
        )
    if "val" in args.parts:
        assert val_remote is not None
        if args.val_local_archive is not None:
            _validate_local_val_archive(
                args.val_local_archive,
                expected_bytes=args.expected_val_archive_bytes,
                expected_md5=args.expected_val_archive_md5,
            )
        _stream_val(
            output_root=output_root,
            state_root=state_root,
            remote=val_remote,
            timeout=args.timeout,
            id_to_wnid=id_to_wnid,
            groundtruth=groundtruth,
            expected_files=args.expected_val_files,
            checkpoint_every=args.val_checkpoint_every,
            debug_max_files=args.debug_max_val_files,
            local_archive=args.val_local_archive,
        )

    manifest = _write_manifest(
        output_root=output_root,
        state_root=state_root,
        args=args,
        id_to_wnid=id_to_wnid,
        train_remote=train_remote,
        val_remote=val_remote,
    )
    print(
        f"[done] manifest={output_root / 'raw_imagenet_manifest.json'} "
        f"complete={manifest['complete']}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "[interrupted] atomic files and class-boundary Range state were preserved; "
            "rerun the identical command to resume.",
            file=sys.stderr,
            flush=True,
        )
        raise
