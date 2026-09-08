from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts import manage_raw_imagenet_storage as storage


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    paths = storage.StoragePaths(tmp_path / "source", tmp_path / "ram", tmp_path / "recovery")
    paths.source_root.mkdir()
    paths.ram_root.mkdir()
    paths.recovery_root.mkdir()
    name, filename = "n01440764", "n01440764_1.JPEG"
    for root in (paths.train, paths.ram_train):
        (root / name).mkdir(parents=True)
        (root / name / filename).write_bytes(b"identical original JPEG bytes")
    file_sizes = {filename: (paths.ram_train / name / filename).stat().st_size}
    entry = {"wnid": name, "header_offset": 0, "payload_offset": 512,
             "size": 10240, "next_offset": 10752}
    gold = {"wnid": name, "complete": True, "image_count": 1,
            "outer_header_offset": 0, "outer_member_size": 10240,
            "payload_range": [512, 10751], "next_offset": 10752,
            "nested_tar_sha256": "a" * 64,
            "file_sizes_sha256": storage.mirror.extract._file_sizes_sha256(file_sizes)}
    original = {"train": {"per_class_file_count": {name: 1}},
                "mapping": {"imagefolder_class_to_index": {name: 0}},
                "source": {"train": {"url": "test-official-url"}}}
    original_sha, index_sha, gold_sha = "b" * 64, "c" * 64, "d" * 64
    # Original manifest/index/gold parsing is covered by mirror helper tests;
    # retain the real RAM marker, layout and fingerprint checks here.
    monkeypatch.setattr(storage.mirror, "_load_source", lambda source: (
        original, original_sha, index_sha, [entry], {name: (gold, gold_sha)}))
    monkeypatch.setattr(storage.mirror.extract, "OFFICIAL_CLASSES", 1)
    monkeypatch.setattr(storage.mirror.extract, "OFFICIAL_TRAIN_FILES", 1)
    manifest = {"schema_version": 1, "complete": True, "source_root": str(paths.source_root),
        "ram_root": str(paths.ram_root), "train_root": str(paths.ram_train),
        "source_manifest_sha256": original_sha, "source_index_sha256": index_sha,
        "class_count": 1, "file_count": 1, "per_class_file_count": {name: 1},
        "mapping": original["mapping"], "source": original["source"]["train"]}
    paths.ram_manifest.write_text(json.dumps(manifest))
    verified_root = paths.ram_root / ".verified_markers"
    verified_root.mkdir()
    verified = dict(gold, mirror_verified=True, source_manifest_sha256=original_sha,
                    source_marker_sha256=gold_sha, verified_at_utc="2026-09-08")
    (verified_root / f"{name}.json").write_text(json.dumps(verified))
    return paths, name, filename


def test_activate_open_handle_rollback_and_reactivation(prepared):
    paths, name, filename = prepared
    original_identity = storage._identity(paths.train)
    original_file = paths.train / name / filename
    with original_file.open("rb") as held:
        result = storage.activate(paths)
        assert result["mode"] == "ram" and result["action"] == "activated"
        assert paths.train.is_symlink() and not paths.backup.is_symlink()
        assert storage._identity(paths.backup) == original_identity
        assert held.read() == original_file.read_bytes()
        assert storage.activate(paths)["action"] == "already_active"
    restored = storage.restore(paths)
    assert restored["mode"] == "disk" and paths.backup.is_symlink()
    assert storage._identity(paths.train) == original_identity
    assert storage.restore(paths)["action"] == "already_on_disk"
    assert storage.activate(paths)["mode"] == "ram"
    assert storage._identity(paths.backup) == original_identity


@pytest.mark.parametrize("field,value", [
    ("complete", False), ("source_manifest_sha256", "0" * 64),
    ("source_index_sha256", "0" * 64), ("class_count", 2),
    ("file_count", 2), ("mapping", {}),
])
def test_activation_gate_rejects_bad_completion_before_namespace_change(prepared, field, value):
    paths, _, _ = prepared
    manifest = json.loads(paths.ram_manifest.read_text())
    manifest[field] = value
    paths.ram_manifest.write_text(json.dumps(manifest))
    original = storage._identity(paths.train)
    with pytest.raises(RuntimeError, match=f"manifest mismatch: {field}"):
        storage.activate(paths)
    assert not paths.backup.exists() and not paths.backup.is_symlink()
    assert storage._identity(paths.train) == original


def test_activation_checks_gold_markers_and_live_ram_file_sizes(prepared):
    paths, name, filename = prepared
    marker_path = paths.ram_root / ".verified_markers" / f"{name}.json"
    original = marker_path.read_bytes()
    bad = json.loads(original)
    bad["nested_tar_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="nested_tar_sha256"):
        storage.activate(paths)
    marker_path.write_bytes(original)
    (paths.ram_train / name / filename).write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="filename/size fingerprint"):
        storage.activate(paths)
    assert not paths.train.is_symlink()


@pytest.mark.parametrize("missing", ["tree", "manifest", "whole_ram_root"])
def test_watch_restores_original_when_ram_disappears(prepared, missing):
    paths, name, filename = prepared
    original = storage._identity(paths.train)
    storage.activate(paths)
    if missing == "tree":
        shutil.rmtree(paths.ram_train)
    elif missing == "manifest":
        paths.ram_manifest.unlink()
    else:
        shutil.rmtree(paths.ram_root)
    result = storage.watch_once(paths)
    assert result["mode"] == "disk" and result["needs_restore"] is False
    assert storage._identity(paths.train) == original
    assert (paths.train / name / filename).read_bytes() == b"identical original JPEG bytes"
    assert paths.backup.is_symlink()


def test_ensure_is_synchronous_and_never_activates(prepared):
    paths, _, _ = prepared
    assert storage.ensure(paths)["mode"] == "disk"
    assert not paths.train.is_symlink() and not paths.backup.exists()
    storage.activate(paths)
    paths.ram_manifest.unlink()
    assert storage.ensure(paths)["mode"] == "disk"
    assert not paths.train.is_symlink()


def test_watch_healthy_does_not_revalidate_or_rewrite_unchanged_status(prepared, monkeypatch):
    paths, _, _ = prepared
    storage.activate(paths)
    monkeypatch.setattr(storage, "_validate_complete_mirror", lambda p: pytest.fail("watch must not scan full dataset"))
    current = storage.watch_once(paths)
    report = paths.recovery_root / "raw_imagenet_storage_status.json"
    before = report.read_bytes(), report.stat().st_mtime_ns
    assert storage.watch_once(paths, current) == current
    assert (report.read_bytes(), report.stat().st_mtime_ns) == before
    assert paths.train.is_symlink()


def test_failed_exchange_keeps_original_and_allows_retry(prepared, monkeypatch):
    paths, _, _ = prepared
    original = storage._identity(paths.train)
    real_exchange = storage._rename_exchange
    monkeypatch.setattr(storage, "_rename_exchange", lambda *a: (_ for _ in ()).throw(OSError(18, "EXDEV")))
    with pytest.raises(OSError, match="EXDEV"):
        storage.activate(paths)
    assert storage._identity(paths.train) == original and paths.backup.is_symlink()
    monkeypatch.setattr(storage, "_rename_exchange", real_exchange)
    assert storage.activate(paths)["mode"] == "ram"


def test_wrong_link_missing_backup_and_changed_backup_identity_are_rejected(prepared):
    paths, _, _ = prepared
    paths.backup.symlink_to("/unexpected/target")
    with pytest.raises(RuntimeError, match="Unexpected storage layout"):
        storage.activate(paths)
    paths.backup.unlink()
    storage.activate(paths)
    saved = paths.source_root / ".owned_test_original"
    paths.backup.rename(saved)
    with pytest.raises(RuntimeError, match="Unexpected storage layout"):
        storage.restore(paths)
    paths.backup.mkdir()
    with pytest.raises(RuntimeError, match="inode differs"):
        storage.restore(paths)
    paths.backup.rmdir()
    saved.rename(paths.backup)
    assert storage.restore(paths)["mode"] == "disk"


def test_crash_after_exchange_before_active_record_can_restore(prepared, monkeypatch):
    paths, _, _ = prepared
    write_record = storage._write_record
    def fail_active_record(p, record):
        if record["state"] == "active":
            raise OSError("simulated interrupted post-exchange publication")
        write_record(p, record)
    monkeypatch.setattr(storage, "_write_record", fail_active_record)
    with pytest.raises(OSError, match="interrupted"):
        storage.activate(paths)
    assert paths.train.is_symlink()
    assert json.loads(paths.record.read_text())["state"] == "prepared"
    monkeypatch.setattr(storage, "_write_record", write_record)
    paths.ram_manifest.unlink()
    assert storage.ensure(paths)["mode"] == "disk"


def test_remount_restores_identical_backup_with_changed_inode(prepared, monkeypatch):
    paths, name, filename = prepared
    storage.activate(paths)
    record_before = json.loads(paths.record.read_text())
    old_backup = paths.source_root / ".held_original_for_test"
    paths.backup.rename(old_backup)
    shutil.copytree(old_backup, paths.backup)
    replacement_identity = storage._identity(paths.backup)
    assert replacement_identity != record_before["original_disk_identity"]
    paths.ram_manifest.unlink()
    with pytest.raises(RuntimeError, match="same mount"):
        storage.ensure(paths)
    assert paths.train.is_symlink()

    changed_mount = dict(record_before["original_mount_identity"])
    changed_mount["mount_namespace_inode"] += 1
    monkeypatch.setattr(storage, "_mount_identity", lambda _: changed_mount)
    result = storage.ensure(paths)
    assert result["mode"] == "disk"
    assert storage._identity(paths.train) == replacement_identity
    assert (paths.train / name / filename).read_bytes() == b"identical original JPEG bytes"
    record_after = json.loads(paths.record.read_text())
    assert record_after["original_mount_identity"] == changed_mount
    assert record_after["original_disk_identity"] == replacement_identity
    assert record_after["remount_validation"]["verified_images"] == 1
    assert record_after["remount_validation"]["verified_classes"] == 1
    monkeypatch.setattr(storage, "_validate_original_after_remount",
                        lambda *a: pytest.fail("same mount must not repeat the full scan"))
    assert storage.restore(paths)["mode"] == "disk"


def test_remount_rejects_changed_backup_file_size(prepared, monkeypatch):
    paths, name, filename = prepared
    storage.activate(paths)
    recorded_mount = json.loads(paths.record.read_text())["original_mount_identity"]
    changed_mount = dict(recorded_mount, source_root_device=recorded_mount["source_root_device"] + 1)
    monkeypatch.setattr(storage, "_mount_identity", lambda _: changed_mount)
    (paths.backup / name / filename).write_bytes(b"truncated")
    shutil.rmtree(paths.ram_root)
    with pytest.raises(RuntimeError, match="filename/size fingerprint"):
        storage.ensure(paths)
    assert paths.train.is_symlink()
    assert paths.backup.is_dir()
    assert json.loads(paths.record.read_text())["original_mount_identity"] == recorded_mount


def test_remount_rejects_unpinned_gold_markers_before_disk_walk(prepared, monkeypatch):
    paths, _, _ = prepared
    storage.activate(paths)
    recorded_mount = json.loads(paths.record.read_text())["original_mount_identity"]
    changed_mount = dict(recorded_mount, mount_id=recorded_mount["mount_id"] + 1)
    monkeypatch.setattr(storage, "_mount_identity", lambda _: changed_mount)
    original, manifest_sha, index_sha, entries, gold = storage.mirror._load_source(paths.source_root)
    changed_gold = {name: (marker, "e" * 64) for name, (marker, _) in gold.items()}
    monkeypatch.setattr(storage.mirror, "_load_source", lambda _: (
        original, manifest_sha, index_sha, entries, changed_gold))
    monkeypatch.setattr(storage.mirror.extract, "_validate_train_marker",
                        lambda *a: pytest.fail("untrusted provenance must fail before disk scan"))
    paths.ram_manifest.unlink()
    with pytest.raises(RuntimeError, match="source_gold_markers_sha256"):
        storage.ensure(paths)
    assert paths.train.is_symlink()
