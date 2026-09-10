from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from scipy.io import savemat


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER = REPO_ROOT / "scripts" / "download_extract_raw_imagenet.py"


def _tar_bytes(members: list[tuple[str, bytes]], *, gzip: bool = False) -> bytes:
    archive = io.BytesIO()
    mode = "w:gz" if gzip else "w"
    with tarfile.open(
        fileobj=archive,
        mode=mode,
        format=tarfile.USTAR_FORMAT,
    ) as handle:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            handle.addfile(info, io.BytesIO(payload))
    return archive.getvalue()


def _meta_mat_bytes(id_to_wnid: dict[int, str]) -> bytes:
    synsets = np.empty(
        len(id_to_wnid),
        dtype=[("ILSVRC2012_ID", object), ("WNID", object)],
    )
    for index, (class_id, wnid) in enumerate(sorted(id_to_wnid.items())):
        synsets[index] = (np.array([[class_id]]), np.array([wnid], dtype=object))
    output = io.BytesIO()
    savemat(output, {"synsets": synsets})
    return output.getvalue()


class _RangeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, payloads: dict[str, bytes]):
        super().__init__(("127.0.0.1", 0), _RangeRequestHandler)
        self.payloads = payloads
        self.requests: list[tuple[str, str | None]] = []


class _RangeRequestHandler(BaseHTTPRequestHandler):
    server: _RangeServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.partition("?")[0]
        payload = self.server.payloads.get(path)
        if payload is None:
            self.send_error(404)
            return

        range_header = self.headers.get("Range")
        self.server.requests.append((path, range_header))
        if range_header is None:
            self.send_response(200)
            response = payload
        else:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
            start = int(match.group(1)) if match is not None else -1
            end = (
                int(match.group(2))
                if match is not None and match.group(2)
                else len(payload) - 1
            )
            if (
                match is None
                or start < 0
                or end < start
                or end >= len(payload)
            ):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(payload)}")
                self.end_headers()
                return
            response = payload[start : end + 1]
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{end}/{len(payload)}",
            )
            self.send_header("Accept-Ranges", "bytes")

        self.send_header("Content-Length", str(len(response)))
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("ETag", f'"fixture-{path}"')
        self.send_header("Last-Modified", "Thu, 01 Jan 1970 00:00:00 GMT")
        self.end_headers()
        try:
            self.wfile.write(response)
        except (BrokenPipeError, ConnectionResetError):
            # Range probes intentionally consume one byte and close the response.
            pass

    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def _serve(payloads: dict[str, bytes]):
    server = _RangeServer(payloads)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield server, f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run_downloader(common_args: list[str], *extra_args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(DOWNLOADER), *common_args, *extra_args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"downloader exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def test_streams_official_layout_resumes_at_class_boundary_and_completes_manifest(
    tmp_path: Path,
) -> None:
    wnid_a = "n00000001"
    wnid_b = "n00000002"
    train_images = {
        # Some official ImageNet JPEG members contain a valid EOI followed by
        # an archive-preserved trailer. Preserve and accept those exact bytes.
        f"{wnid_a}_1.JPEG": b"\xff\xd8train-a-1\xff\xd9official-trailer",
        f"{wnid_a}_2.JPEG": b"\xff\xd8train-a-2\xff\xd9",
        # ILSVRC2012 contains one valid PNG whose official filename ends in
        # .JPEG.  Preserve that representation instead of transcoding it.
        f"{wnid_b}_1.JPEG": (
            b"\x89PNG\r\n\x1a\nfixture-payload"
            b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
        ),
    }
    nested_a = _tar_bytes(
        [(name, payload) for name, payload in train_images.items() if name.startswith(wnid_a)]
    )
    nested_b = _tar_bytes(
        [(name, payload) for name, payload in train_images.items() if name.startswith(wnid_b)]
    )
    train_archive = _tar_bytes(
        [(f"{wnid_a}.tar", nested_a), (f"{wnid_b}.tar", nested_b)]
    )

    val_images = {
        "ILSVRC2012_val_00000001.JPEG": b"\xff\xd8val-for-b\xff\xd9",
        "ILSVRC2012_val_00000002.JPEG": b"\xff\xd8val-for-a\xff\xd9",
    }
    val_archive = _tar_bytes(list(val_images.items()))
    val_archive_path = tmp_path / "val-local.tar"
    val_archive_path.write_bytes(val_archive)
    id_to_wnid = {1: wnid_b, 2: wnid_a}
    devkit_archive = _tar_bytes(
        [
            ("ILSVRC2012_devkit_t12/data/meta.mat", _meta_mat_bytes(id_to_wnid)),
            (
                "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
                b"1\n2\n",
            ),
        ],
        gzip=True,
    )

    payloads = {
        "/train.tar": train_archive,
        "/val.tar": val_archive,
        "/devkit.tar.gz": devkit_archive,
    }
    output_root = tmp_path / "raw"
    state_root = tmp_path / "state"

    with _serve(payloads) as (server, base_url):
        common_args = [
            "--output-root",
            os.fspath(output_root),
            "--state-root",
            os.fspath(state_root),
            "--train-url",
            f"{base_url}/train.tar",
            "--val-url",
            f"{base_url}/val.tar",
            "--devkit-url",
            f"{base_url}/devkit.tar.gz",
            "--expected-train-archive-bytes",
            str(len(train_archive)),
            "--expected-val-archive-bytes",
            str(len(val_archive)),
            "--expected-val-archive-md5",
            hashlib.md5(val_archive, usedforsecurity=False).hexdigest(),
            "--expected-train-files",
            str(len(train_images)),
            "--expected-val-files",
            str(len(val_images)),
            "--expected-classes",
            "2",
            "--allow-nonstandard-counts-for-test",
            "--reserve-bytes",
            "0",
            "--http-retries",
            "1",
            "--retry-delay",
            "0",
            "--timeout",
            "5",
            "--val-checkpoint-every",
            "1",
            "--train-workers",
            "2",
        ]

        first_run = _run_downloader(
            common_args,
            "--parts",
            "train",
            "--debug-max-train-classes",
            "1",
        )
        assert "[train] debug stop after 1 new classes" in first_run.stdout

        first_progress = json.loads(
            (state_root / "train_progress.json").read_text(encoding="utf-8")
        )
        train_index = json.loads(
            (state_root / "train_outer_index.json").read_text(encoding="utf-8")
        )
        indexed_by_wnid = {
            entry["wnid"]: entry for entry in train_index["entries"]
        }
        entry_a = indexed_by_wnid[wnid_a]
        entry_b = indexed_by_wnid[wnid_b]
        first_marker = json.loads(
            (state_root / "train_classes" / f"{wnid_a}.json").read_text(
                encoding="utf-8"
            )
        )
        payload_range_a = (
            f"bytes={entry_a['payload_offset']}-"
            f"{entry_a['payload_offset'] + entry_a['size'] - 1}"
        )
        payload_range_b = (
            f"bytes={entry_b['payload_offset']}-"
            f"{entry_b['payload_offset'] + entry_b['size'] - 1}"
        )
        assert first_marker["payload_range"] == [
            entry_a["payload_offset"],
            entry_a["payload_offset"] + entry_a["size"] - 1,
        ]
        assert train_index["complete"] is True
        assert [entry["wnid"] for entry in train_index["entries"]] == [wnid_a, wnid_b]
        assert first_progress["classes_completed"] == 1
        assert first_progress["images_completed"] == 2
        assert first_progress["payload_bytes_completed"] == len(nested_a)
        assert first_progress["archive_complete"] is False
        assert (output_root / "train" / wnid_a).is_dir()
        assert not (output_root / "train" / wnid_b).exists()
        assert json.loads(
            (output_root / "raw_imagenet_manifest.json").read_text(encoding="utf-8")
        )["complete"] is False

        first_run_requests = list(server.requests)
        expected_header_offsets = [
            entry_a["header_offset"],
            entry_b["header_offset"],
            train_index["archive_end_offset"] - 2 * 512,
            train_index["archive_end_offset"] - 512,
        ]
        for header_offset in expected_header_offsets:
            assert (
                "/train.tar",
                f"bytes={header_offset}-{header_offset + 511}",
            ) in first_run_requests
        assert first_run_requests.count(("/train.tar", payload_range_a)) == 1
        assert ("/train.tar", payload_range_b) not in first_run_requests

        second_run_request_index = len(server.requests)
        second_run = _run_downloader(
            common_args,
            "--parts",
            "train",
            "val",
            "--val-local-archive",
            os.fspath(val_archive_path),
        )
        assert "[train-index] reused 2 indexed class ranges" in second_run.stdout
        assert "pending_this_run=1 committed=1/2" in second_run.stdout

        second_run_requests = server.requests[second_run_request_index:]
        assert second_run_requests.count(("/train.tar", payload_range_b)) == 1
        assert ("/train.tar", payload_range_a) not in second_run_requests
        assert ("/train.tar", "bytes=0-0") in second_run_requests
        assert ("/val.tar", "bytes=0-0") in second_run_requests
        assert ("/val.tar", "bytes=0-") not in second_run_requests
        assert "[val] verified local archive" in second_run.stdout
        assert "[val] reading verified local archive" in second_run.stdout
        # The cached allowlisted devkit files make the resume independent of a
        # second non-Range download.
        assert ("/devkit.tar.gz", None) not in second_run_requests

    for name, expected in train_images.items():
        wnid = name.split("_", 1)[0]
        assert (output_root / "train" / wnid / name).read_bytes() == expected
    assert (
        output_root / "val" / wnid_b / "ILSVRC2012_val_00000001.JPEG"
    ).read_bytes() == val_images["ILSVRC2012_val_00000001.JPEG"]
    assert (
        output_root / "val" / wnid_a / "ILSVRC2012_val_00000002.JPEG"
    ).read_bytes() == val_images["ILSVRC2012_val_00000002.JPEG"]

    final_progress = json.loads(
        (state_root / "train_progress.json").read_text(encoding="utf-8")
    )
    assert final_progress["archive_complete"] is True
    assert final_progress["classes_completed"] == 2
    assert final_progress["images_completed"] == 3
    assert sorted(path.stem for path in (state_root / "train_classes").glob("*.json")) == [
        wnid_a,
        wnid_b,
    ]

    manifest = json.loads(
        (output_root / "raw_imagenet_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["complete"] is True
    assert manifest["expected"] == {"classes": 2, "train_files": 3, "val_files": 2}
    assert manifest["train"] == {
        "complete": True,
        "class_count": 2,
        "file_count": 3,
        "per_class_file_count": {wnid_a: 2, wnid_b: 1},
    }
    assert manifest["val"] == {
        "complete": True,
        "class_count": 2,
        "file_count": 2,
        "per_class_file_count": {wnid_a: 1, wnid_b: 1},
    }
    assert manifest["mapping"]["official_imagenet_id_to_wnid"] == {
        "1": wnid_b,
        "2": wnid_a,
    }
    assert manifest["mapping"]["imagefolder_class_to_index"] == {
        wnid_a: 0,
        wnid_b: 1,
    }
    assert manifest["mapping"]["train_val_devkit_wnids_identical"] is True
    assert not [
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and path.name.endswith(".partial")
    ]
    assert not (output_root / "train" / ".partial").exists()
