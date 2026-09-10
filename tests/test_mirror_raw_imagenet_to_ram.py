from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from scripts import mirror_raw_imagenet_to_ram as mirror


class _ProcessRangeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = self.server.payload
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if match is None:
            self.send_error(400)
            return
        start, end = map(int, match.groups())
        if not 0 <= start <= end < len(data):
            self.send_error(416)
            return
        with self.server.request_count.get_lock():
            self.server.request_count.value += 1
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start:end + 1])

    def log_message(self, *args):
        pass


def _affinity_child_probe(plan, counter, queue):
    mirror._initialize_worker_affinity(plan, counter)
    queue.put((os.getpid(), sorted(os.sched_getaffinity(0))))


def _fixture(tmp_path):
    wnid = "n01440764"
    image_name = f"{wnid}_1.JPEG"
    image = b"\xff\xd8fake-jpeg-for-transport-check\xff\xd9"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo(image_name)
        info.size = len(image)
        tar.addfile(info, io.BytesIO(image))
    payload = stream.getvalue()
    entry = {"wnid": wnid, "header_offset": 0, "payload_offset": 512,
             "size": len(payload), "next_offset": 512 + len(payload)}
    gold = {"schema_version": 2, "complete": True, "wnid": wnid,
            "outer_header_offset": 0, "outer_member_size": len(payload),
            "payload_range": [512, 511 + len(payload)],
            "next_offset": 512 + len(payload), "image_count": 1,
            "nested_tar_sha256": hashlib.sha256(payload).hexdigest(),
            "file_sizes_sha256": mirror.extract._file_sizes_sha256({image_name: len(image)})}
    ram = tmp_path / "ram"
    for name in ("train", ".candidate_markers", ".verified_markers"):
        (ram / name).mkdir(parents=True)
    kwargs = dict(entry=entry, gold=gold, gold_sha="a" * 64, manifest_sha="b" * 64,
                  ram=ram, remote={"total_bytes": len(payload) + 512}, timeout=1,
                  retries=1, retry_delay=0)
    return kwargs, payload, image_name, image


def test_streamed_class_gold_verification_and_resume_without_network(tmp_path, monkeypatch):
    kwargs, payload, name, image = _fixture(tmp_path)
    requests = []
    def open_range(*args, **kwargs):
        requests.append((args, kwargs))
        return io.BytesIO(payload)
    monkeypatch.setattr(mirror.extract, "_open_range", open_range)
    marker = mirror._mirror_class(**kwargs)
    assert marker["mirror_verified"] is True
    assert marker["source_marker_sha256"] == "a" * 64
    assert (kwargs["ram"] / "train" / kwargs["entry"]["wnid"] / name).read_bytes() == image
    assert len(requests) == 1
    # Existing verified markers still require the current RAM tree and gold.
    assert mirror._mirror_class(**kwargs) == marker
    assert len(requests) == 1
    image_path = kwargs["ram"] / "train" / kwargs["entry"]["wnid"] / name
    image_path.write_bytes(image + b"damage")
    with pytest.raises(RuntimeError, match="filename/size fingerprint changed"):
        mirror._mirror_class(**kwargs)
    assert len(requests) == 1


@pytest.mark.parametrize("field", ["nested_tar_sha256", "file_sizes_sha256", "image_count", "outer_member_size"])
def test_mismatched_gold_never_publishes_verified_marker(tmp_path, monkeypatch, field):
    kwargs, payload, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(mirror.extract, "_open_range", lambda *a, **k: io.BytesIO(payload))
    kwargs["gold"][field] = "0" * 64 if field.endswith("sha256") else 999
    with pytest.raises(RuntimeError, match=f"Mirror/gold mismatch.*{field}"):
        mirror._mirror_class(**kwargs)
    assert not list((kwargs["ram"] / ".verified_markers").iterdir())


def test_resume_rejects_changed_original_provenance(tmp_path, monkeypatch):
    kwargs, payload, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(mirror.extract, "_open_range", lambda *a, **k: io.BytesIO(payload))
    mirror._mirror_class(**kwargs)
    kwargs["manifest_sha"] = "c" * 64
    with pytest.raises(RuntimeError, match="RAM marker provenance mismatch"):
        mirror._mirror_class(**kwargs)


def test_uncommitted_orphan_final_is_not_silently_trusted(tmp_path, monkeypatch):
    kwargs, _, name, image = _fixture(tmp_path)
    path = kwargs["ram"] / "train" / kwargs["entry"]["wnid"]
    path.mkdir()
    (path / name).write_bytes(image)
    monkeypatch.setattr(mirror.extract, "_extract_train_range", lambda **k: pytest.fail("must not trust orphan"))
    with pytest.raises(RuntimeError, match="Uncommitted RAM class"):
        mirror._mirror_class(**kwargs)


def test_full_remaining_space_reserved_even_for_pilot(tmp_path, monkeypatch):
    kwargs, _, _, _ = _fixture(tmp_path)
    entry, gold = kwargs["entry"], kwargs["gold"]
    entries = [entry]
    gold_by_name = {entry["wnid"]: (gold, "a" * 64)}
    monkeypatch.setattr(mirror.shutil, "disk_usage", lambda _: SimpleNamespace(free=entry["size"]))
    with pytest.raises(RuntimeError, match="entire remaining mirror"):
        mirror._preflight_space(kwargs["ram"], entries, gold_by_name, {}, 0)
    monkeypatch.setattr(mirror.shutil, "disk_usage", lambda _: SimpleNamespace(free=1 << 30))
    budget = mirror._preflight_space(kwargs["ram"], entries, gold_by_name, {}, 0)
    assert budget["required_free_bytes"] == entry["size"] + 4096 + 65536
    resumed = mirror._preflight_space(kwargs["ram"], entries, gold_by_name, {entry["wnid"]: gold}, 0)
    assert resumed["required_free_bytes"] == 0


def test_unsafe_output_roots_and_symlink_directories_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be /dev/shm"):
        mirror._validate_roots(tmp_path, tmp_path / "ram")
    with pytest.raises(ValueError, match="must be /dev/shm"):
        mirror._validate_roots(tmp_path, Path("/dev/shm"))
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "symlink"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink destination"):
        mirror._real_directory(link)


def test_candidate_marker_from_root_pilot_can_be_adopted(tmp_path, monkeypatch):
    kwargs, payload, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(mirror.extract, "_open_range", lambda *a, **k: io.BytesIO(payload))
    candidate = mirror.extract._extract_train_range(
        entry=kwargs["entry"], marker_root=kwargs["ram"] / ".candidate_markers",
        train_root=kwargs["ram"] / "train", remote=kwargs["remote"],
        timeout=1, retries=1, retry_delay=0)
    monkeypatch.setattr(mirror.extract, "_open_range", lambda *a, **k: pytest.fail("candidate is already downloaded"))
    verified = mirror._mirror_class(**kwargs)
    mirror._compare_gold(verified, kwargs["gold"])
    assert verified["nested_tar_sha256"] == candidate["nested_tar_sha256"]
    # Root's separately published verified marker schema is accepted unchanged.
    verified["verified_at_utc"] = "2026-09-08T00:00:00+00:00"
    (kwargs["ram"] / ".verified_markers" / f"{kwargs['entry']['wnid']}.json").write_text(json.dumps(verified))
    assert mirror._mirror_class(**kwargs) == verified


def test_complete_orchestration_resumes_pilot_and_never_writes_source(tmp_path, monkeypatch):
    source, ram = tmp_path / "source", tmp_path / "ram"
    source.mkdir()
    ram.mkdir()
    state = source / ".download_state"
    (state / "train_classes").mkdir(parents=True)
    entries, markers, payloads = [], {}, {}
    offset = 0
    for i, wnid in enumerate(("n01440764", "n01443537")):
        payload = io.BytesIO()
        name = f"{wnid}_1.JPEG"
        image = b"\xff\xd8some-jpeg-data\xff\xd9"
        with tarfile.open(fileobj=payload, mode="w") as tar:
            info = tarfile.TarInfo(name)
            info.size = len(image)
            tar.addfile(info, io.BytesIO(image))
        data = payload.getvalue()
        entry = {"wnid": wnid, "header_offset": offset, "payload_offset": offset + 512,
                 "size": len(data), "next_offset": offset + 512 + len(data)}
        marker = {"schema_version": 2, "complete": True, "wnid": wnid,
            "outer_header_offset": offset, "outer_member_size": len(data),
            "payload_range": [offset + 512, offset + 511 + len(data)],
            "next_offset": entry["next_offset"], "image_count": 1,
            "nested_tar_sha256": hashlib.sha256(data).hexdigest(),
            "file_sizes_sha256": mirror.extract._file_sizes_sha256({name: len(image)})}
        (state / "train_classes" / f"{wnid}.json").write_text(json.dumps(marker))
        entries.append(entry)
        markers[wnid] = marker
        payloads[entry["payload_offset"]] = data
        offset = entry["next_offset"]
    remote = {"url": mirror.extract.OFFICIAL_TRAIN_URL, "total_bytes": offset + 1024}
    index = {"complete": True, "remote": remote, "entries": entries, "archive_end_offset": offset + 1024}
    (state / "train_outer_index.json").write_text(json.dumps(index))
    class_map = {name: i for i, name in enumerate(sorted(markers))}
    manifest = {"schema_version": 1, "complete": True, "output_root": str(source), "state_root": str(state),
        "expected": {"classes": 2, "train_files": 2, "val_files": 1},
        "mapping": {"imagefolder_class_to_index": class_map, "train_val_devkit_wnids_identical": True,
                    "official_imagenet_id_to_wnid": {str(i + 1): name for i, name in enumerate(class_map)}},
        "train": {"complete": True, "class_count": 2, "file_count": 2,
                  "per_class_file_count": {name: 1 for name in class_map}},
        "source": {"train": remote}}
    (source / "raw_imagenet_manifest.json").write_text(json.dumps(manifest))
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    for key, value in (("OFFICIAL_CLASSES", 2), ("OFFICIAL_TRAIN_FILES", 2),
                       ("OFFICIAL_VAL_FILES", 1), ("OFFICIAL_TRAIN_BYTES", remote["total_bytes"])):
        monkeypatch.setattr(mirror.extract, key, value)
    monkeypatch.setattr(mirror.extract, "_probe_remote", lambda *a, **k: remote)
    requests = []
    def open_range(remote, start, **kwargs):
        requests.append(start)
        return io.BytesIO(payloads[start])
    monkeypatch.setattr(mirror.extract, "_open_range", open_range)
    args = SimpleNamespace(workers=2, max_classes=1, reserve_shm_gib=0,
                           timeout=1, retries=1, retry_delay=0)
    pilot = mirror._run_locked(args, source, ram, ram / "progress.json", mirror.time.monotonic())
    assert pilot["status"] == "pilot_complete" and not pilot["complete"]
    assert pilot["verified_classes"] == 1 and not (ram / "mirror_manifest.json").exists()
    args.max_classes = 0
    full = mirror._run_locked(args, source, ram, ram / "progress.json", mirror.time.monotonic())
    assert full["complete"] and full["verified_images"] == 2
    assert len(requests) == len(set(requests)) == 2  # No re-download of pilot class.
    final = json.loads((ram / "mirror_manifest.json").read_text())
    assert final["mapping"] == manifest["mapping"] and final["live_path_switched"] is False
    assert final["file_count"] == 2 and final["class_count"] == 2
    after = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.parametrize("mismatched_hash", [False, True])
@pytest.mark.parametrize("use_affinity", [False, True])
def test_process_executor_real_range_http_and_verified_resume(tmp_path, mismatched_hash, use_affinity):
    # Full-suite tests can leave tqdm's monitor thread alive. The real command
    # intentionally rejects a multithreaded parent before forking workers, so
    # exercise this integration case in the standalone CPU process it requires.
    # Keep the production guard intact; the separate rejection test covers it.
    program = """
from pathlib import Path
import sys
import pytest
from tests.test_mirror_raw_imagenet_to_ram import _exercise_process_executor
with pytest.MonkeyPatch.context() as monkeypatch:
    _exercise_process_executor(Path(sys.argv[1]), monkeypatch,
                               sys.argv[2] == 'True', sys.argv[3] == 'True')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), str(mismatched_hash), str(use_affinity)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _exercise_process_executor(tmp_path, monkeypatch, mismatched_hash, use_affinity):
    kwargs, payload, filename, image = _fixture(tmp_path)
    context = multiprocessing.get_context("fork")
    server = HTTPServer(("127.0.0.1", 0), _ProcessRangeHandler)
    server.payload = b"\0" * 512 + payload + b"\0" * 1024
    server.request_count = context.Value("i", 0)
    server_process = context.Process(target=server.serve_forever, daemon=True)
    server_process.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/train.tar"
        source = tmp_path / "source"
        source.mkdir()
        remote = {"url": url, "total_bytes": len(server.payload)}
        entry, gold, ram = kwargs["entry"], kwargs["gold"], kwargs["ram"]
        if mismatched_hash:
            gold["nested_tar_sha256"] = "0" * 64
        original = {"train": {"per_class_file_count": {entry["wnid"]: 1}},
                    "mapping": {"imagefolder_class_to_index": {entry["wnid"]: 0}},
                    "source": {"train": remote}}
        monkeypatch.setattr(mirror, "_load_source", lambda _: (
            original, kwargs["manifest_sha"], "e" * 64, [entry],
            {entry["wnid"]: (gold, kwargs["gold_sha"])}))
        monkeypatch.setattr(mirror.extract, "OFFICIAL_TRAIN_URL", url)
        monkeypatch.setattr(mirror.extract, "OFFICIAL_TRAIN_BYTES", len(server.payload))
        monkeypatch.setattr(mirror.extract, "OFFICIAL_TRAIN_FILES", 1)
        args = SimpleNamespace(workers=2, executor="process", max_classes=0,
                               reserve_shm_gib=0, timeout=2, retries=1, retry_delay=0)
        if use_affinity:
            allowed = sorted(os.sched_getaffinity(0))
            args.worker_affinity_json = tmp_path / "worker_affinity.json"
            args.worker_affinity_json.write_text(json.dumps([[allowed[0]], [allowed[-1]]]))
        if mismatched_hash:
            with pytest.raises(RuntimeError, match="nested_tar_sha256"):
                mirror._run_locked(args, source, ram, ram / "progress.json", mirror.time.monotonic())
            assert not list((ram / ".verified_markers").iterdir())
            assert not (ram / "mirror_manifest.json").exists()
            assert json.loads((ram / "progress.json").read_text())["status"] == "failed"
        else:
            result = mirror._run_locked(args, source, ram, ram / "progress.json", mirror.time.monotonic())
            assert result["complete"] and result["executor"] == "process"
            if use_affinity:
                assert result["worker_affinity_plan"] == [[allowed[0]], [allowed[-1]]]
            assert result["verified_classes"] == result["verified_images"] == 1
            assert (ram / "train" / entry["wnid"] / filename).read_bytes() == image
            assert server.request_count.value == 2  # One probe, one class Range.
            resumed = mirror._run_locked(args, source, ram, ram / "progress.json", mirror.time.monotonic())
            assert resumed["complete"] and server.request_count.value == 2
    finally:
        server_process.terminate()
        server_process.join(timeout=5)
        server.server_close()


def test_process_executor_rejects_multithreaded_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(mirror.threading, "active_count", lambda: 2)
    monkeypatch.setattr(mirror, "_load_source", lambda *a: pytest.fail("must reject before source reads"))
    with pytest.raises(RuntimeError, match="single-threaded standalone CPU parent"):
        mirror._run_locked(SimpleNamespace(executor="process"), tmp_path, tmp_path,
                           tmp_path / "progress.json", mirror.time.monotonic())


@pytest.mark.parametrize("plan,match", [
    ([[2]], "exactly one CPU array"),
    ([[2], []], "nonempty array"),
    ([[2], [True]], "nonempty array"),
    ([[2], ["3"]], "nonempty array"),
    ([[2], [-1]], "nonempty array"),
    ([[2], [3, 3]], "nonempty array"),
    ([[2], [4]], "outside the parent affinity"),
])
def test_worker_affinity_rejects_invalid_plans(tmp_path, monkeypatch, plan, match):
    path = tmp_path / "affinity.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(mirror.os, "sched_getaffinity", lambda _: {2, 3})
    with pytest.raises(ValueError, match=match):
        mirror._load_worker_affinity(path, workers=2, executor_kind="process")


def test_worker_affinity_is_opt_in_process_only_and_records_content_hash(tmp_path, monkeypatch):
    path = tmp_path / "affinity.json"
    data = b"[[2, 3], [3]]\n"
    path.write_bytes(data)
    monkeypatch.setattr(mirror.os, "sched_getaffinity", lambda _: {2, 3})
    assert mirror._load_worker_affinity(None, workers=2, executor_kind="thread") == (None, None)
    with pytest.raises(ValueError, match="requires --executor process"):
        mirror._load_worker_affinity(path, workers=2, executor_kind="thread")
    assert mirror._load_worker_affinity(path, workers=2, executor_kind="process") == (
        [[2, 3], [3]], hashlib.sha256(data).hexdigest())


def test_actual_fork_worker_affinities_are_assigned_once_and_parent_unchanged():
    parent_mask = set(os.sched_getaffinity(0))
    available = sorted(parent_mask)
    if len(available) < 2:
        pytest.skip("two allowed CPUs required to distinguish worker masks")
    plan = [[available[0]], [available[1]]]
    context = multiprocessing.get_context("fork")
    counter, queue = context.Value("i", 0), context.Queue()
    children = [context.Process(target=_affinity_child_probe, args=(plan, counter, queue)) for _ in plan]
    try:
        for child in children:
            child.start()
        results = [queue.get(timeout=5) for _ in children]
        for child in children:
            child.join(timeout=5)
            assert child.exitcode == 0
        assert len({pid for pid, _ in results}) == 2
        assert sorted(mask for _, mask in results) == plan
        assert counter.value == 2
        assert set(os.sched_getaffinity(0)) == parent_mask
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
        queue.close()
