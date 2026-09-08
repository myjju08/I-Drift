#!/usr/bin/env python3
"""Mirror authenticated ImageNet train JPEGs into tmpfs without changing live paths.

The original complete manifest, outer-tar index and class markers are read-only.
Each network Range is streamed by the original extractor, compared with the
original class hashes, then separately marked verified. This command never
renames the live dataset, deletes the original, or changes training settings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import threading
import time

if __package__:
    from . import download_extract_raw_imagenet as extract
else:
    import download_extract_raw_imagenet as extract


_GOLD_FIELDS = (
    "wnid", "complete", "outer_header_offset", "outer_member_size",
    "payload_range", "next_offset", "image_count", "nested_tar_sha256",
    "file_sizes_sha256",
)


def _read_json_digest(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Expected a regular provenance file: {path}")
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value, hashlib.sha256(data).hexdigest()


def _real_directory(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise RuntimeError(f"Refusing non-directory or symlink destination: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _validate_roots(source_root: Path, ram_root: Path) -> tuple[Path, Path]:
    source = source_root.resolve(strict=True)
    # Restrict writable output to a dedicated, direct tmpfs child. Do not resolve
    # a symlink before rejecting it, and never permit /dev/shm itself as output.
    ram = Path(os.path.abspath(ram_root))
    if ram.parent != Path("/dev/shm") or not re.fullmatch(
        r"idrift_raw_imagenet_[A-Za-z0-9_-]+", ram.name
    ):
        raise ValueError("--ram-root must be /dev/shm/idrift_raw_imagenet_<name>")
    if ram.is_symlink() or (ram.exists() and not ram.is_dir()):
        raise RuntimeError(f"Unsafe RAM root: {ram}")
    if source == ram or ram.is_relative_to(source) or source.is_relative_to(ram):
        raise ValueError("Source and RAM roots must be disjoint")
    # Refuse an unexpected /dev/shm mount rather than filling container storage.
    shm_is_tmpfs = any(
        line.split()[4] == "/dev/shm" and " - tmpfs " in line
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
    )
    if not shm_is_tmpfs:
        raise RuntimeError("/dev/shm must be a tmpfs mount")
    return source, ram


def _validate_gold(gold: dict, entry: dict) -> None:
    expected = {
        "wnid": entry["wnid"], "complete": True,
        "outer_header_offset": entry["header_offset"],
        "outer_member_size": entry["size"], "next_offset": entry["next_offset"],
        "payload_range": [entry["payload_offset"], entry["payload_offset"] + entry["size"] - 1],
    }
    for key, value in expected.items():
        if gold.get(key) != value:
            raise RuntimeError(f"Gold marker/index mismatch for {entry['wnid']}: {key}")
    if type(gold.get("image_count")) is not int or gold["image_count"] <= 0:
        raise RuntimeError(f"Invalid gold image count for {entry['wnid']}")
    for key in ("nested_tar_sha256", "file_sizes_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(gold.get(key, ""))):
            raise RuntimeError(f"Gold marker lacks a valid {key}: {entry['wnid']}")


def _compare_gold(candidate: dict, gold: dict) -> None:
    for key in _GOLD_FIELDS:
        if candidate.get(key) != gold.get(key):
            raise RuntimeError(f"Mirror/gold mismatch for {gold['wnid']}: {key}")


def _load_worker_affinity(path: Path | None, *, workers: int, executor_kind: str) -> tuple[list[list[int]] | None, str | None]:
    if path is None:
        return None, None
    if executor_kind != "process":
        raise ValueError("--worker-affinity-json requires --executor process")
    data = Path(path).read_bytes()
    plan = json.loads(data)
    if not isinstance(plan, list) or len(plan) != workers:
        raise ValueError("Worker affinity JSON must contain exactly one CPU array per worker")
    allowed = set(os.sched_getaffinity(0))
    for index, cpus in enumerate(plan):
        if (not isinstance(cpus, list) or not cpus
                or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
                or len(set(cpus)) != len(cpus)):
            raise ValueError(f"Worker affinity entry {index} must be a nonempty array of distinct nonnegative CPU integers")
        unavailable = set(cpus) - allowed
        if unavailable:
            raise ValueError(f"Worker affinity entry {index} includes CPUs outside the parent affinity: {sorted(unavailable)}")
    return plan, hashlib.sha256(data).hexdigest()


def _initialize_worker_affinity(plan: list[list[int]], counter) -> None:
    # ProcessPool initializers run once in each newly forked CPU-only worker.
    # A shared counter assigns exactly one planned mask; task scheduling order
    # remains the executor's responsibility and cannot change image contents.
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    if index >= len(plan):
        raise RuntimeError("Process pool created more workers than the affinity plan")
    os.sched_setaffinity(0, set(plan[index]))
    actual = sorted(os.sched_getaffinity(0))
    if actual != sorted(plan[index]):
        raise RuntimeError(f"Worker {index} CPU affinity did not match its plan")
    print(json.dumps({"event": "mirror_worker_affinity", "worker_index": index,
                      "pid": os.getpid(), "cpus": actual}, sort_keys=True), flush=True)


def _load_source(source: Path) -> tuple[dict, str, str, list[dict], dict[str, tuple[dict, str]]]:
    manifest, manifest_sha = _read_json_digest(source / "raw_imagenet_manifest.json")
    if manifest.get("complete") is not True or manifest.get("schema_version") != 1:
        raise RuntimeError("Source ImageNet manifest must be verified complete schema 1")
    if Path(str(manifest.get("output_root", ""))).resolve() != source:
        raise RuntimeError("Source manifest root mismatch")
    expected = manifest.get("expected", {})
    for key, value in (("classes", extract.OFFICIAL_CLASSES),
                       ("train_files", extract.OFFICIAL_TRAIN_FILES),
                       ("val_files", extract.OFFICIAL_VAL_FILES)):
        if expected.get(key) != value:
            raise RuntimeError(f"Source manifest canonical count mismatch: {key}")
    mapping = manifest.get("mapping", {})
    class_map = mapping.get("imagefolder_class_to_index", {})
    if not isinstance(class_map, dict) or len(class_map) != extract.OFFICIAL_CLASSES:
        raise RuntimeError("Source manifest lacks the full canonical class mapping")
    wnids = sorted(class_map)
    if (class_map != {name: i for i, name in enumerate(wnids)}
            or any(not extract._WNID_RE.fullmatch(name) for name in wnids)
            or mapping.get("train_val_devkit_wnids_identical") is not True
            or set(mapping.get("official_imagenet_id_to_wnid", {}).values()) != set(wnids)):
        raise RuntimeError("Source canonical class mapping is inconsistent")
    train = manifest.get("train", {})
    per_class = train.get("per_class_file_count", {})
    if (train.get("complete") is not True or train.get("file_count") != extract.OFFICIAL_TRAIN_FILES
            or train.get("class_count") != extract.OFFICIAL_CLASSES or set(per_class) != set(wnids)):
        raise RuntimeError("Source train split is not verified complete")
    state_root = Path(str(manifest.get("state_root", ""))).resolve(strict=True)
    if not state_root.is_relative_to(source) or state_root == source:
        raise RuntimeError("Source state_root must be strictly below the source root")
    index, index_sha = _read_json_digest(state_root / "train_outer_index.json")
    remote = manifest.get("source", {}).get("train", {})
    if remote.get("url") != extract.OFFICIAL_TRAIN_URL or remote.get("total_bytes") != extract.OFFICIAL_TRAIN_BYTES:
        raise RuntimeError("Source must be the exact official ImageNet train archive")
    entries = extract._validate_train_index(index, remote, set(wnids))
    gold = {}
    for entry in entries:
        name = entry["wnid"]
        marker, marker_sha = _read_json_digest(state_root / "train_classes" / f"{name}.json")
        _validate_gold(marker, entry)
        if marker["image_count"] != per_class[name]:
            raise RuntimeError(f"Gold marker/manifest image count mismatch: {name}")
        gold[name] = (marker, marker_sha)
    if sum(item[0]["image_count"] for item in gold.values()) != extract.OFFICIAL_TRAIN_FILES:
        raise RuntimeError("Gold markers do not cover the canonical training image count")
    return manifest, manifest_sha, index_sha, entries, gold


def _validate_verified(marker: dict, *, entry: dict, gold: dict, gold_sha: str,
                       manifest_sha: str, train_root: Path) -> dict:
    _compare_gold(marker, gold)
    if (marker.get("mirror_verified") is not True
            or marker.get("source_marker_sha256") != gold_sha
            or marker.get("source_manifest_sha256") != manifest_sha):
        raise RuntimeError(f"RAM marker provenance mismatch: {entry['wnid']}")
    return extract._validate_train_marker(marker, entry, train_root)


def _mirror_class(*, entry: dict, gold: dict, gold_sha: str, manifest_sha: str,
                  ram: Path, remote: dict, timeout: float, retries: int,
                  retry_delay: float) -> dict:
    name = entry["wnid"]
    train_root = ram / "train"
    verified_path = ram / ".verified_markers" / f"{name}.json"
    candidate_path = ram / ".candidate_markers" / f"{name}.json"
    for path in (verified_path, candidate_path):
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlink marker: {path}")
    if verified_path.exists():
        marker, _ = _read_json_digest(verified_path)
        return _validate_verified(marker, entry=entry, gold=gold, gold_sha=gold_sha,
                                  manifest_sha=manifest_sha, train_root=train_root)
    if (train_root / name).exists() and not candidate_path.exists():
        raise RuntimeError(f"Uncommitted RAM class {name} has no candidate marker; inspect this RAM-only orphan before retrying")
    candidate = extract._extract_train_range(
        entry=entry, marker_root=ram / ".candidate_markers", train_root=train_root,
        remote=remote, timeout=timeout, retries=retries, retry_delay=retry_delay,
    )
    _compare_gold(candidate, gold)
    extract._validate_train_marker(candidate, entry, train_root)
    result = dict(candidate, mirror_verified=True, source_marker_sha256=gold_sha,
                  source_manifest_sha256=manifest_sha, verified_at_utc=extract._utc_now())
    extract._atomic_write_json(verified_path, result)
    return result


def _preflight_space(ram: Path, entries: list[dict], gold: dict, verified: dict,
                     reserve_gib: float) -> dict:
    remaining = [entry for entry in entries if entry["wnid"] not in verified]
    # Tar payload is an upper bound for JPEG bytes. Add conservative tmpfs page
    # rounding (one 4 KiB page/image) and directory/marker overhead on top.
    payload = sum(entry["size"] for entry in remaining)
    overhead = sum(gold[entry["wnid"]][0]["image_count"] * 4096 + 65536 for entry in remaining)
    reserve = math.ceil(reserve_gib * 1024**3)
    free = shutil.disk_usage(ram).free
    required = payload + overhead + reserve
    if free < required:
        raise RuntimeError(f"Insufficient tmpfs for the entire remaining mirror plus reserve: free={free:,} required={required:,} remaining_tar_upper={payload:,} overhead={overhead:,} reserve={reserve:,}")
    return {"free_bytes": free, "remaining_tar_upper_bytes": payload,
            "file_overhead_upper_bytes": overhead, "reserve_bytes": reserve,
            "required_free_bytes": required}


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    source, ram = _validate_roots(args.source_root, args.ram_root)
    progress_path = Path(os.path.abspath(args.progress_json)) if args.progress_json else ram / "progress.json"
    # Progress output is user-selected but must never overwrite source data,
    # provenance, or any file within the newly mirrored JPEG tree.
    resolved_progress = progress_path.resolve(strict=False)
    if (resolved_progress.is_relative_to(source) or resolved_progress.is_relative_to(ram / "train")
            or resolved_progress in {ram / "mirror_manifest.json", ram / ".mirror.lock"}
            or any(resolved_progress.is_relative_to(ram / p) for p in (".candidate_markers", ".verified_markers"))):
        raise ValueError("Unsafe --progress-json destination")
    _real_directory(ram)
    lock_path = ram / ".mirror.lock"
    if lock_path.is_symlink():
        raise RuntimeError("Refusing symlink lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(args, source, ram, progress_path, started)


def _run_locked(args: argparse.Namespace, source: Path, ram: Path,
                progress_path: Path, started: float) -> dict:
    executor_kind = getattr(args, "executor", "thread")
    if executor_kind not in ("thread", "process"):
        raise ValueError("executor must be thread or process")
    if executor_kind == "process" and threading.active_count() != 1:
        raise RuntimeError("The fork process executor requires a single-threaded standalone CPU parent")
    affinity_path = getattr(args, "worker_affinity_json", None)
    affinity_plan, affinity_sha = _load_worker_affinity(
        affinity_path, workers=args.workers, executor_kind=executor_kind)
    manifest, manifest_sha, index_sha, entries, gold = _load_source(source)
    for directory in (ram / "train", ram / "train" / ".partial",
                      ram / ".candidate_markers", ram / ".verified_markers"):
        _real_directory(directory)
    by_name = {entry["wnid"]: entry for entry in entries}
    verified = {}
    for path in sorted((ram / ".verified_markers").iterdir()):
        if path.name.startswith(".") and path.name.endswith(".partial"):
            continue  # An interrupted atomic marker write is never committed.
        if path.suffix != ".json" or path.stem not in by_name:
            raise RuntimeError(f"Unexpected verified marker: {path}")
        marker, _ = _read_json_digest(path)
        original, sha = gold[path.stem]
        verified[path.stem] = _validate_verified(marker, entry=by_name[path.stem], gold=original,
            gold_sha=sha, manifest_sha=manifest_sha, train_root=ram / "train")
    budget = _preflight_space(ram, entries, gold, verified, args.reserve_shm_gib)
    pending = [entry for entry in entries if entry["wnid"] not in verified]
    if args.max_classes:
        pending = pending[:max(0, args.max_classes - len(verified))]
    remote = manifest["source"]["train"]
    if pending:
        observed = extract._probe_remote(extract.OFFICIAL_TRAIN_URL, timeout=args.timeout,
            retries=args.retries, retry_delay=args.retry_delay, expected_bytes=extract.OFFICIAL_TRAIN_BYTES)
        extract._validate_remote_state({"remote": remote}, observed)
        remote = observed

    def publish(status: str, *, error: str | None = None) -> dict:
        elapsed = time.monotonic() - started
        state = {"schema_version": 1, "complete": status == "complete", "status": status,
            "source_root": str(source), "ram_root": str(ram),
            "source_manifest_sha256": manifest_sha, "source_index_sha256": index_sha,
            "verified_classes": len(verified),
            "verified_images": sum(marker["image_count"] for marker in verified.values()),
            "verified_tar_payload_bytes": sum(by_name[name]["size"] for name in verified),
            "elapsed_seconds": elapsed, "updated_at_utc": extract._utc_now(),
            "requested_workers": args.workers, "executor": executor_kind,
            "max_classes": args.max_classes,
            "preflight_space": budget}
        if error is not None:
            state["error"] = error
        if affinity_plan is not None:
            state["worker_affinity_json"] = str(Path(affinity_path).resolve())
            state["worker_affinity_sha256"] = affinity_sha
            state["worker_affinity_plan"] = affinity_plan
        extract._atomic_write_json(progress_path, state)
        print(json.dumps(state, sort_keys=True), flush=True)
        return state

    publish("running")
    try:
        # Keep only workers futures live; do not queue all 1000 Range downloads.
        executor_type = ProcessPoolExecutor if executor_kind == "process" else ThreadPoolExecutor
        executor_kwargs = {"max_workers": args.workers}
        if executor_kind == "process":
            # This standalone downloader imports no CUDA/torch and forks before
            # creating any parent threads. Avoid repeated cold module reads;
            # HTTP sockets are opened separately inside each class worker.
            process_context = multiprocessing.get_context("fork")
            executor_kwargs["mp_context"] = process_context
            if affinity_plan is not None:
                executor_kwargs["initializer"] = _initialize_worker_affinity
                executor_kwargs["initargs"] = (affinity_plan, process_context.Value("i", 0))
        with executor_type(**executor_kwargs) as executor:
            iterator = iter(pending)
            active = {}
            def submit_next() -> None:
                entry = next(iterator, None)
                if entry is None:
                    return
                marker, sha = gold[entry["wnid"]]
                future = executor.submit(_mirror_class, entry=entry, gold=marker, gold_sha=sha,
                    manifest_sha=manifest_sha, ram=ram, remote=remote, timeout=args.timeout,
                    retries=args.retries, retry_delay=args.retry_delay)
                active[future] = entry["wnid"]
            for _ in range(args.workers):
                submit_next()
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    name = active.pop(future)
                    verified[name] = future.result()
                    publish("running")
                    submit_next()
        complete = set(verified) == set(by_name)
        if complete:
            partial = ram / "train" / ".partial"
            if partial.exists():
                if partial.is_symlink():
                    raise RuntimeError("Unsafe partial directory")
                partial.rmdir()  # Succeeds only when extraction staging is empty.
            count, per_class = extract._verify_train_tree(ram / "train", set(by_name), extract.OFFICIAL_TRAIN_FILES)
            if per_class != manifest["train"]["per_class_file_count"]:
                raise RuntimeError("Completed mirror class counts differ from source")
            mirror = {"schema_version": 1, "complete": True, "source_root": str(source),
                "ram_root": str(ram), "train_root": str(ram / "train"),
                "source_manifest_sha256": manifest_sha, "source_index_sha256": index_sha,
                "class_count": len(per_class), "file_count": count,
                "per_class_file_count": per_class, "mapping": manifest["mapping"],
                "source": manifest["source"]["train"], "completed_at_utc": extract._utc_now(),
                "verification": "Every nested tar SHA256 and filename/size SHA256 matched original gold markers; original dataset unchanged",
                "live_path_switched": False}
            extract._atomic_write_json(ram / "mirror_manifest.json", mirror)
        return publish("complete" if complete else "pilot_complete")
    except BaseException as exc:
        publish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--ram-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--executor", choices=("thread", "process"), default="thread",
                        help="Process mode bypasses Python GIL contention in the standalone CPU downloader")
    parser.add_argument("--worker-affinity-json", type=Path,
                        help="Optional process-worker CPU arrays, one array per worker; all CPUs must be allowed for the parent")
    parser.add_argument("--max-classes", type=int, default=0, help="Cap total verified classes; zero mirrors all 1000")
    parser.add_argument("--reserve-shm-gib", type=float, default=20)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64 or not 0 <= args.max_classes <= extract.OFFICIAL_CLASSES:
        parser.error("workers must be 1..64; max-classes must be 0..1000")
    if (not math.isfinite(args.reserve_shm_gib) or args.reserve_shm_gib < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0 or args.retries < 1
            or not math.isfinite(args.retry_delay) or args.retry_delay < 0):
        parser.error("Invalid reserve or HTTP retry settings")
    run(args)


if __name__ == "__main__":
    main()
