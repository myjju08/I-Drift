"""Certify an unchanged copied mmap and authentic source rows for node migration.

This does not build, convert, or modify latent payloads or the original packing
manifest. It hashes every payload byte before publishing local relocation
provenance. The source-evidence directory may contain only the 32 audit rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def certify(cache_path, evidence_root):
    import numpy as np
    from train.latent_data import (
        FEATURE_DIRECTORY, LABEL_DIRECTORY, LATENT_CACHE_RECIPE, LATENT_SHAPE,
        MmapLatentDataset, _directory_signature, _file_signature, _sha256,
        read_flat_latent_row,
    )
    cache = Path(cache_path).resolve(strict=True)
    evidence = Path(evidence_root).resolve(strict=True)
    metadata = json.loads((cache / "metadata.json").read_text())
    if (metadata.get("schema_version") != 1 or metadata.get("complete") is not True
            or metadata.get("recipe") != LATENT_CACHE_RECIPE):
        raise ValueError("A completed original exact-cache manifest is required")
    count = int(metadata["count"])
    if count <= 0:
        raise ValueError("The cache must contain training rows")
    shapes = {"latents.npy": ((count, *LATENT_SHAPE), np.dtype("float32")),
              "labels.npy": ((count,), np.dtype("int64")),
              "source_stats.npy": ((count, 4), np.dtype("int64"))}
    verified = {}
    arrays = {}
    for name, (shape, dtype) in shapes.items():
        path = cache / name
        before = _file_signature(path)
        digest = _sha256(path)
        if _file_signature(path) != before:
            raise ValueError(f"Payload changed during checksum verification: {name}")
        if digest != metadata["files"][name]["sha256"]:
            raise ValueError(f"Copied payload checksum differs from original: {name}")
        if path.stat().st_size != metadata["files"][name]["bytes"]:
            raise ValueError(f"Copied payload size differs from original: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous:
            raise ValueError(f"Invalid copied array geometry: {name}")
        arrays[name] = array
        verified[name] = {"sha256": digest, "signature": before}
    rows = sorted(set(np.linspace(0, count - 1, min(32, count), dtype=np.int64).tolist()))
    directories = {name: _directory_signature(evidence / name)
                   for name in (FEATURE_DIRECTORY, LABEL_DIRECTORY)}
    for row in rows:
        raw, label, stats = read_flat_latent_row(evidence, row, num_classes=int(metadata["num_classes"]))
        if (label != int(arrays["labels.npy"][row])
                or not np.array_equal(raw.view(np.uint32), arrays["latents.npy"][row].view(np.uint32))):
            raise ValueError(f"Authentic source evidence differs from copied mmap: row {row}")
        if list(stats) != arrays["source_stats.npy"][row].tolist():
            raise ValueError(f"Source evidence must preserve original sizes and nanosecond mtimes: row {row}")
    if any(_directory_signature(evidence / name) != signature for name, signature in directories.items()):
        raise ValueError("Source evidence inventory changed during certification")
    if any(_file_signature(cache / name) != record["signature"] for name, record in verified.items()):
        raise ValueError("Payload changed before relocation certification completed")
    labels = arrays["labels.npy"]
    if (labels.min() < 0 or labels.max() >= int(metadata["num_classes"])
            or np.bincount(labels, minlength=int(metadata["num_classes"])).tolist() != metadata["label_counts"]):
        raise ValueError("Copied label inventory differs from original")
    certificate = {
        "schema_version": 1, "kind": "verified_exact_latent_cache_relocation",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "verified_on_host": socket.gethostname(),
        "original_metadata_sha256": _sha256(cache / "metadata.json"),
        "original_source_root": metadata["source_root"],
        "source_evidence_root": str(evidence), "source_evidence_rows": rows,
        "source_evidence_is_full_dataset": False, "source_directories": directories,
        "files": verified,
    }
    path = cache / "relocation.json"
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(certificate, indent=2) + "\n")
    os.replace(temporary, path)
    # Exercise the production loader against the finished certificate.
    MmapLatentDataset(cache, source_root=evidence, num_classes=int(metadata["num_classes"]))
    return certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    args = parser.parse_args()
    certificate = certify(args.cache, args.source_evidence)
    print(json.dumps(certificate, indent=2))


if __name__ == "__main__":
    main()
