"""Copied full caches retain exact data while only 32 raw rows are available."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

from scripts.certify_latent_cache_relocation import certify
from train.latent_data import MmapLatentDataset


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def relocated(tmp_path):
    count, classes = 67, 5
    source, original = tmp_path / "original-source", tmp_path / "original-cache"
    evidence, copied = tmp_path / "only-audit-rows", tmp_path / "copied-cache"
    original.mkdir()
    for name in ("imagenet256_features", "imagenet256_labels"):
        (source / name).mkdir(parents=True)
        (evidence / name).mkdir(parents=True)
    values = np.random.default_rng(1321).standard_normal((count, 4, 32, 32), dtype=np.float32)
    values[:, 0, 0, 0] = np.float32(-0.0)
    values[:, 0, 0, 1] = np.nextafter(np.float32(0), np.float32(1))
    labels = np.arange(count, dtype=np.int64) % classes
    stats = []
    for index in range(count):
        np.save(source / "imagenet256_features" / f"{index}.npy", values[index:index + 1])
        np.save(source / "imagenet256_labels" / f"{index}.npy", labels[index:index + 1])
        row_stats = []
        for name in ("imagenet256_features", "imagenet256_labels"):
            stat = (source / name / f"{index}.npy").stat()
            row_stats.extend((stat.st_size, stat.st_mtime_ns))
        stats.append(row_stats)
    for name, array in (("latents.npy", values), ("labels.npy", labels),
                        ("source_stats.npy", np.asarray(stats, dtype=np.int64))):
        np.save(original / name, array)
    metadata = {
        "schema_version": 1, "complete": True,
        "recipe": "flat-npy-numeric-order-float32-bitwise-v1",
        "source_root": str(source), "source_directories": {},
        "count": count, "num_classes": classes,
        "label_counts": np.bincount(labels, minlength=classes).tolist(), "files": {},
    }
    for name in ("imagenet256_features", "imagenet256_labels"):
        stat = (source / name).stat()
        metadata["source_directories"][name] = [stat.st_dev, stat.st_ino, stat.st_mtime_ns]
    for path in original.glob("*.npy"):
        metadata["files"][path.name] = {"bytes": path.stat().st_size, "sha256": _digest(path)}
    (original / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copytree(original, copied)
    rows = np.linspace(0, count - 1, 32, dtype=np.int64).tolist()
    for row in rows:
        for name in ("imagenet256_features", "imagenet256_labels"):
            shutil.copy2(source / name / f"{row}.npy", evidence / name / f"{row}.npy")
    return dict(source=source, original=original, evidence=evidence, copied=copied,
                rows=rows, count=count, classes=classes, values=values, labels=labels)


def _dataset(fixture, *, cache=None, source=None):
    return MmapLatentDataset(cache or fixture["copied"], source_root=source or fixture["evidence"],
                             num_classes=fixture["classes"])


def test_full_payload_and_original_manifest_unchanged_with_only_32_raw_rows(relocated):
    f = relocated
    before = {path.name: path.read_bytes() for path in f["copied"].iterdir()}
    for name in ("imagenet256_features", "imagenet256_labels"):
        assert len(list((f["evidence"] / name).glob("*.npy"))) == 32
    certificate = certify(f["copied"], f["evidence"])
    assert certificate["source_evidence_rows"] == f["rows"]
    assert certificate["source_evidence_is_full_dataset"] is False
    assert certificate["original_source_root"] == str(f["source"])
    for name, original_bytes in before.items():
        assert (f["copied"] / name).read_bytes() == original_bytes
    dataset = _dataset(f)
    assert len(dataset) == 67 and dataset.relocation == certificate
    # Check every row, including the 35 rows absent from the raw evidence folder.
    for index in range(f["count"]):
        value, label = dataset[index]
        assert value.numpy().tobytes() == f["values"][index].tobytes()
        assert label == int(f["labels"][index])
        value.fill_(99)
        assert dataset[index][0].numpy().tobytes() == f["values"][index].tobytes()
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._features is None and restored._labels is None
    assert restored[1][0].numpy().tobytes() == f["values"][1].tobytes()


def test_relocation_preserves_rng_and_distributed_shuffled_sample_order(relocated):
    f = relocated
    torch.manual_seed(57)
    cpu_rng, numpy_rng = torch.get_rng_state(), np.random.get_state()
    certify(f["copied"], f["evidence"])
    current = _dataset(f)
    original = _dataset(f, cache=f["original"], source=f["source"])
    assert original.relocation is None
    assert torch.equal(cpu_rng, torch.get_rng_state())
    np.testing.assert_equal(numpy_rng, np.random.get_state())
    for rank in (0, 1):
        for epoch in (0, 3):
            samplers = [DistributedSampler(ds, num_replicas=2, rank=rank, seed=43,
                                           shuffle=True, drop_last=False)
                        for ds in (original, current)]
            for sampler in samplers:
                sampler.set_epoch(epoch)
            assert list(samplers[0]) == list(samplers[1])
            loaders = [DataLoader(ds, batch_size=7, sampler=sampler, num_workers=0)
                       for ds, sampler in zip((original, current), samplers)]
            for expected, actual in zip(*loaders):
                assert expected[0].numpy().tobytes() == actual[0].numpy().tobytes()
                assert torch.equal(expected[1], actual[1])


def test_uncertified_copy_cannot_claim_partial_evidence_as_original_source(relocated):
    f = relocated
    with pytest.raises(ValueError, match="original manifest"):
        _dataset(f)
    assert _dataset(f, cache=f["original"], source=f["source"]).relocation is None


@pytest.mark.parametrize("payload", ["latents.npy", "labels.npy", "source_stats.npy"])
def test_certification_checks_entire_payload_before_accepting_audit_rows(relocated, payload):
    f = relocated
    row = next(index for index in range(f["count"]) if index not in f["rows"])
    data = np.load(f["copied"] / payload, mmap_mode="r+")
    if payload == "latents.npy":
        data.view(np.uint32)[row, 0, 1, 0] ^= np.uint32(1)
    elif payload == "labels.npy":
        data[row] = (data[row] + 1) % f["classes"]
    else:
        data[row, 0] += 1
    data.flush()
    del data
    with pytest.raises(ValueError, match="payload checksum differs from original"):
        certify(f["copied"], f["evidence"])
    assert not (f["copied"] / "relocation.json").exists()


def test_payload_modification_after_certification_is_detected(relocated):
    f = relocated
    certify(f["copied"], f["evidence"])
    path = f["copied"] / "latents.npy"
    before = path.stat()
    data = np.load(path, mmap_mode="r+")
    data[1, 0, 1, 0] += 1
    data.flush()
    del data
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="changed after full checksum verification"):
        _dataset(f)


def test_original_metadata_change_invalidates_certificate(relocated):
    f = relocated
    certify(f["copied"], f["evidence"])
    path = f["copied"] / "metadata.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="provenance does not match"):
        _dataset(f)


def test_copied_certificate_cannot_authorize_different_payload_file_identities(relocated, tmp_path):
    f = relocated
    certify(f["copied"], f["evidence"])
    another = tmp_path / "another-cache"
    shutil.copytree(f["copied"], another)
    assert _digest(another / "latents.npy") == _digest(f["copied"] / "latents.npy")
    with pytest.raises(ValueError, match="changed after full checksum verification"):
        _dataset(f, cache=another)


def test_certificate_cannot_authorize_a_different_evidence_root(relocated, tmp_path):
    f = relocated
    certify(f["copied"], f["evidence"])
    another = tmp_path / "another-evidence"
    shutil.copytree(f["evidence"], another)
    with pytest.raises(ValueError, match="provenance does not match"):
        _dataset(f, source=another)


def test_authentic_evidence_values_and_original_mtimes_are_required(relocated):
    f = relocated
    row = f["rows"][0]
    path = f["evidence"] / "imagenet256_features" / f"{row}.npy"
    before = path.stat()
    value = np.load(path)
    value[0, 0, 1, 1] += 1
    np.save(path, value)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="Authentic source evidence differs"):
        certify(f["copied"], f["evidence"])
    shutil.copy2(f["source"] / "imagenet256_features" / f"{row}.npy", path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="preserve original sizes and nanosecond mtimes"):
        certify(f["copied"], f["evidence"])
