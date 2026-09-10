#!/usr/bin/env python3
"""Explicitly activate, inspect, or restore the authenticated RAM ImageNet tree.

Production CLI paths are fixed. Activation is never automatic. The watcher only
restores an already active RAM path when its tree or completion manifest goes
missing, for example after container stop/start clears tmpfs. No dataset files
are deleted, and the original disk directory is retained for atomic rollback.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import threading

if __package__:
    from . import mirror_raw_imagenet_to_ram as mirror
else:
    import mirror_raw_imagenet_to_ram as mirror


@dataclass(frozen=True)
class StoragePaths:
    source_root: Path
    ram_root: Path
    recovery_root: Path
    backup_name: str = ".train.disk_backup_20260908"

    @property
    def train(self) -> Path:
        return self.source_root / "train"

    @property
    def backup(self) -> Path:
        return self.source_root / self.backup_name

    @property
    def ram_train(self) -> Path:
        return self.ram_root / "train"

    @property
    def ram_manifest(self) -> Path:
        return self.ram_root / "mirror_manifest.json"

    @property
    def record(self) -> Path:
        return self.recovery_root / "raw_imagenet_storage_activation.json"


PRODUCTION = StoragePaths(
    Path("/workspace/I-Drift/data/imagenet/raw_ilsvrc2012"),
    Path("/dev/shm/idrift_raw_imagenet_20260908"),
    Path("/workspace/I-Drift/runs/throughput_recovery_20260908"),
)


def _kind(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "missing"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    return "other"


def _check_paths(paths: StoragePaths) -> None:
    for path in (paths.source_root, paths.ram_root, paths.recovery_root):
        if not path.is_absolute() or path != Path(os.path.abspath(path)):
            raise RuntimeError(f"Storage path must be absolute and normalized: {path}")
        # Resolving the existing parent also rejects intermediate symlinks. The
        # RAM directory itself may legitimately disappear at container restart.
        if path.parent.resolve(strict=True) != path.parent:
            raise RuntimeError(f"Storage path has a symlink parent: {path}")
    if _kind(paths.source_root) != "directory":
        raise RuntimeError("Original source root must remain a real directory")
    if _kind(paths.ram_root) not in ("directory", "missing"):
        raise RuntimeError("RAM root must be a real directory or absent")
    if _kind(paths.recovery_root) not in ("directory", "missing"):
        raise RuntimeError("Recovery root must be a real directory or absent")
    if (Path(paths.backup_name).name != paths.backup_name or paths.backup_name in ("", ".", "..", "train")
            or not paths.backup_name.startswith(".train.disk_backup_")):
        raise RuntimeError("Unsafe disk backup name")
    roots = (paths.source_root, paths.ram_root, paths.recovery_root)
    if any(a.is_relative_to(b) for i, a in enumerate(roots) for j, b in enumerate(roots) if i != j):
        raise RuntimeError("Source, RAM and recovery roots must be disjoint")


@contextmanager
def _locked(paths: StoragePaths):
    _check_paths(paths)
    paths.recovery_root.mkdir(exist_ok=True)
    lock = paths.recovery_root / "raw_imagenet_storage.lock"
    if _kind(lock) not in ("missing", "file"):
        raise RuntimeError("Refusing non-regular storage lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _identity(path: Path) -> dict:
    st = path.lstat()
    return {"device": st.st_dev, "inode": st.st_ino}


def _mount_identity(paths: StoragePaths) -> dict:
    """Identify this mount lifetime; inode/device numbers are not durable IDs."""
    def unescape(value: str) -> str:
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)

    candidates = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields, filesystem = before.split(), after.split()
        mountpoint = Path(unescape(fields[4]))
        if paths.source_root == mountpoint or paths.source_root.is_relative_to(mountpoint):
            candidates.append((len(mountpoint.parts), fields, filesystem))
    if not candidates:
        raise RuntimeError("Cannot identify original dataset filesystem mount")
    _, fields, filesystem = max(candidates, key=lambda item: item[0])
    return {"schema_version": 1,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "mount_namespace_inode": Path("/proc/self/ns/mnt").stat().st_ino,
            "source_root_device": paths.source_root.stat().st_dev,
            "mount_id": int(fields[0]), "mount_device": fields[2],
            "mount_root": unescape(fields[3]), "mount_point": unescape(fields[4]),
            "filesystem_type": filesystem[0], "mount_source": unescape(filesystem[1])}


def _gold_markers_digest(gold: dict) -> str:
    encoded = json.dumps({name: value[1] for name, value in gold.items()},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rename_exchange(left: Path, right: Path) -> None:
    # Exchange the directory entries themselves; never resolve symlink operands.
    # Both are siblings on one mounted filesystem; the link target may be tmpfs.
    if left.parent != right.parent:
        raise RuntimeError("Atomic storage exchange requires sibling paths")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError("libc renameat2 is required; no non-atomic fallback") from exc
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(left), -100, os.fsencode(right), 2) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"Atomic RENAME_EXCHANGE failed: {os.strerror(error)}; no fallback was attempted")


def _inspect(paths: StoragePaths) -> dict:
    train_kind, backup_kind = _kind(paths.train), _kind(paths.backup)
    target = str(paths.ram_train)
    train_link = os.readlink(paths.train) if train_kind == "symlink" else None
    backup_link = os.readlink(paths.backup) if backup_kind == "symlink" else None
    if train_kind == "directory" and (
        backup_kind == "missing" or (backup_kind == "symlink" and backup_link == target)
    ):
        mode = "disk"
    elif train_kind == "symlink" and train_link == target and backup_kind == "directory":
        mode = "ram"
    else:
        raise RuntimeError(f"Unexpected storage layout: train={train_kind}:{train_link}, backup={backup_kind}:{backup_link}")
    ram_kind, manifest_kind = _kind(paths.ram_train), _kind(paths.ram_manifest)
    if ram_kind not in ("directory", "missing") or manifest_kind not in ("file", "missing"):
        raise RuntimeError("Unexpected RAM train or completion-manifest type")
    return {"schema_version": 1, "mode": mode, "source_root": str(paths.source_root),
            "train_path": str(paths.train), "disk_backup_path": str(paths.backup),
            "ram_train_path": str(paths.ram_train), "train_kind": train_kind,
            "backup_kind": backup_kind, "ram_train_present": ram_kind == "directory",
            "ram_completion_manifest_present": manifest_kind == "file",
            "needs_restore": mode == "ram" and (ram_kind == "missing" or manifest_kind == "missing")}


def _publish(paths: StoragePaths, state: dict, *, action: str) -> dict:
    report = dict(state, action=action, checked_at_utc=mirror.extract._utc_now())
    path = paths.recovery_root / "raw_imagenet_storage_status.json"
    if _kind(path) not in ("missing", "file"):
        raise RuntimeError("Refusing non-regular storage status file")
    mirror.extract._atomic_write_json(path, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def _validate_complete_mirror(paths: StoragePaths) -> dict:
    original, original_sha, index_sha, entries, gold = mirror._load_source(paths.source_root)
    manifest, manifest_sha = mirror._read_json_digest(paths.ram_manifest)
    expected = {"schema_version": 1, "complete": True,
                "source_root": str(paths.source_root), "ram_root": str(paths.ram_root),
                "train_root": str(paths.ram_train), "source_manifest_sha256": original_sha,
                "source_index_sha256": index_sha,
                "class_count": mirror.extract.OFFICIAL_CLASSES,
                "file_count": mirror.extract.OFFICIAL_TRAIN_FILES,
                "per_class_file_count": original["train"]["per_class_file_count"],
                "mapping": original["mapping"], "source": original["source"]["train"]}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"RAM completion manifest mismatch: {key}")
    verified_root = paths.ram_root / ".verified_markers"
    if _kind(verified_root) != "directory":
        raise RuntimeError("RAM verified markers must be a real directory")
    names = {entry["wnid"] for entry in entries}
    if set(path.name for path in paths.ram_train.iterdir()) != names:
        raise RuntimeError("RAM train class directory names differ from the canonical mapping")
    marker_names = {path.name for path in verified_root.iterdir()
                    if not (path.name.startswith(".") and path.name.endswith(".partial"))}
    if marker_names != {f"{name}.json" for name in names}:
        raise RuntimeError("RAM verified marker set is incomplete or unexpected")
    total = 0
    for entry in entries:
        name = entry["wnid"]
        marker, _ = mirror._read_json_digest(verified_root / f"{name}.json")
        original_marker, original_marker_sha = gold[name]
        checked = mirror._validate_verified(marker, entry=entry, gold=original_marker,
            gold_sha=original_marker_sha, manifest_sha=original_sha, train_root=paths.ram_train)
        total += checked["image_count"]
    if total != mirror.extract.OFFICIAL_TRAIN_FILES:
        raise RuntimeError("RAM verified markers do not cover all training images")
    return {"source_manifest_sha256": original_sha, "source_index_sha256": index_sha,
            "source_gold_markers_sha256": _gold_markers_digest(gold),
            "ram_mirror_manifest_sha256": manifest_sha, "verified_classes": len(entries),
            "verified_images": total}


def _load_record(paths: StoragePaths) -> dict | None:
    if _kind(paths.record) == "missing":
        return None
    record, _ = mirror._read_json_digest(paths.record)
    for key, expected in (("source_root", str(paths.source_root)),
                          ("ram_root", str(paths.ram_root)),
                          ("disk_backup_path", str(paths.backup))):
        if record.get(key) != expected:
            raise RuntimeError(f"Storage activation record mismatch: {key}")
    return record


def _write_record(paths: StoragePaths, record: dict) -> None:
    if _kind(paths.record) not in ("missing", "file"):
        raise RuntimeError("Refusing non-regular storage activation record")
    mirror.extract._atomic_write_json(paths.record, record)


def _validate_original_after_remount(paths: StoragePaths, disk_path: Path, record: dict) -> dict:
    """Reauthenticate the retained directory without reading image payloads.

    This deliberately costs one complete filename/size scan after a detected
    remount. It is not run in the normal watcher loop or on the original mount.
    It detects missing/extra files and size changes, not same-size corruption.
    """
    _, manifest_sha, index_sha, entries, gold = mirror._load_source(paths.source_root)
    observed = {"source_manifest_sha256": manifest_sha, "source_index_sha256": index_sha,
                "source_gold_markers_sha256": _gold_markers_digest(gold)}
    pinned = record.get("verification") or {}
    for key, value in observed.items():
        if pinned.get(key) != value:
            raise RuntimeError(f"Original disk remount provenance mismatch: {key}")
    names = {entry["wnid"] for entry in entries}
    if _kind(disk_path) != "directory" or set(path.name for path in disk_path.iterdir()) != names:
        raise RuntimeError("Original disk remount class directory set mismatch")
    total = 0
    for entry in entries:
        checked = mirror.extract._validate_train_marker(gold[entry["wnid"]][0], entry, disk_path)
        total += checked["image_count"]
    if total != mirror.extract.OFFICIAL_TRAIN_FILES:
        raise RuntimeError("Original disk remount image count mismatch")
    return dict(observed, verified_classes=len(entries), verified_images=total,
                method="Pinned source metadata and all original class filename/size fingerprints; no JPEG payload reads",
                checked_at_utc=mirror.extract._utc_now())


def _check_original_identity(paths: StoragePaths, disk_path: Path) -> None:
    record = _load_record(paths)
    if record is None:
        return
    identity = _identity(disk_path)
    mount = _mount_identity(paths)
    previous_mount = record.get("original_mount_identity")
    if previous_mount is None:
        # An old record cannot prove whether a mount change occurred. Never
        # weaken its original inode guard or silently bless a different tree.
        if record.get("original_disk_identity") != identity:
            raise RuntimeError("Original disk directory inode differs from legacy activation record; remount provenance unavailable")
        return
    if previous_mount == mount:
        if record.get("original_disk_identity") != identity:
            raise RuntimeError("Original disk directory inode differs from activation record on the same mount")
        return
    verification = _validate_original_after_remount(paths, disk_path, record)
    _write_record(paths, dict(record, original_disk_identity=identity,
        original_mount_identity=mount, remount_validation=dict(verification,
            previous_mount_identity=previous_mount, current_mount_identity=mount)))


def activate(paths: StoragePaths = PRODUCTION) -> dict:
    with _locked(paths):
        before = _inspect(paths)
        if before["mode"] == "ram":
            _check_original_identity(paths, paths.backup)
            verification = _validate_complete_mirror(paths)
            return _publish(paths, dict(before, verification=verification), action="already_active")
        _check_original_identity(paths, paths.train)
        verification = _validate_complete_mirror(paths)
        identity = _identity(paths.train)
        record = {"schema_version": 1, "state": "prepared", "source_root": str(paths.source_root),
                  "ram_root": str(paths.ram_root), "disk_backup_path": str(paths.backup),
                  "original_disk_identity": identity, "original_mount_identity": _mount_identity(paths),
                  "verification": verification,
                  "prepared_at_utc": mirror.extract._utc_now()}
        # Publish intent before the namespace mutation; a crash after exchange
        # still leaves the original inode identifiable for the next restore.
        _write_record(paths, record)
        if _kind(paths.backup) == "missing":
            paths.backup.symlink_to(paths.ram_train, target_is_directory=True)
        _rename_exchange(paths.train, paths.backup)
        if _identity(paths.backup) != identity or _inspect(paths)["mode"] != "ram":
            raise RuntimeError("Post-exchange storage invariant failed; original backup retained")
        _write_record(paths, dict(record, state="active", activated_at_utc=mirror.extract._utc_now()))
        return _publish(paths, dict(_inspect(paths), verification=verification), action="activated")


def _restore_locked(paths: StoragePaths, *, reason: str) -> dict:
    before = _inspect(paths)
    if before["mode"] == "disk":
        _check_original_identity(paths, paths.train)
        return _publish(paths, before, action="already_on_disk")
    _check_original_identity(paths, paths.backup)
    identity = _identity(paths.backup)
    _rename_exchange(paths.train, paths.backup)
    if _identity(paths.train) != identity or _inspect(paths)["mode"] != "disk":
        raise RuntimeError("Post-restore storage invariant failed; no files were deleted")
    record = _load_record(paths)
    if record is not None:
        _write_record(paths, dict(record, state="restored", restored_at_utc=mirror.extract._utc_now(),
                                 restore_reason=reason))
    return _publish(paths, dict(_inspect(paths), restore_reason=reason), action="restored")


def restore(paths: StoragePaths = PRODUCTION) -> dict:
    with _locked(paths):
        return _restore_locked(paths, reason="explicit_restore")


def status(paths: StoragePaths = PRODUCTION) -> dict:
    with _locked(paths):
        return _publish(paths, _inspect(paths), action="status")


def watch_once(paths: StoragePaths = PRODUCTION, previous: dict | None = None) -> dict:
    with _locked(paths):
        current = _inspect(paths)
        if current["needs_restore"]:
            _restore_locked(paths, reason="RAM tree or completion manifest missing")
            return _inspect(paths)
        if current != previous:
            _publish(paths, current, action="watch")
        return current


def ensure(paths: StoragePaths = PRODUCTION) -> dict:
    """Synchronous pre-launch fallback check; never activates a RAM mirror."""
    return watch_once(paths)


def watch(paths: StoragePaths = PRODUCTION) -> None:
    stopped = threading.Event()
    def stop(_signum, _frame):
        stopped.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous = None
    # Initial restoration happens before any five-second wait. Supervisor must
    # allow this initial check to finish before launching dependent training.
    while not stopped.is_set():
        previous = watch_once(paths, previous)
        stopped.wait(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("activate", "status", "restore", "watch", "ensure"))
    args = parser.parse_args()
    # No CLI overrides for paths or verification: test fixtures use Python APIs.
    {"activate": activate, "status": status, "restore": restore, "watch": watch, "ensure": ensure}[args.action]()


if __name__ == "__main__":
    main()
