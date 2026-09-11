"""Exact cache transport and opt-in raw-validation semantics."""
import hashlib
import json
import pickle
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch

from train.latent_data import FlatNpyLatentDataset, MmapLatentDataset, read_flat_latent_row
from train.train_data import create_imagenet_split


@pytest.fixture
def packed(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    features, labels = source / "imagenet256_features", source / "imagenet256_labels"
    features.mkdir(parents=True)
    labels.mkdir()
    cache.mkdir()
    rng = np.random.default_rng(824)
    values = rng.standard_normal((9, 4, 32, 32), dtype=np.float32)
    values[:, 0, 0, 0] = np.float32(-0.0)
    values[:, 0, 0, 1] = np.nextafter(np.float32(0), np.float32(1))
    targets = np.arange(9, dtype=np.int64) % 3
    stats = []
    for index in range(9):
        np.save(features / f"{index}.npy", values[index:index + 1])
        np.save(labels / f"{index}.npy", targets[index:index + 1])
        row_stats = []
        for directory in (features, labels):
            stat = (directory / f"{index}.npy").stat()
            row_stats.extend((stat.st_size, stat.st_mtime_ns))
        stats.append(row_stats)
    for name, value in (("latents.npy", values), ("labels.npy", targets),
                        ("source_stats.npy", np.asarray(stats, dtype=np.int64))):
        np.save(cache / name, value)
    directories = {}
    for directory in (features, labels):
        stat = directory.stat()
        directories[directory.name] = [stat.st_dev, stat.st_ino, stat.st_mtime_ns]
    metadata = dict(
        schema_version=1, complete=True,
        recipe="flat-npy-numeric-order-float32-bitwise-v1",
        source_root=str(source), source_directories=directories,
        count=9, num_classes=3, label_counts=[3, 3, 3], files={},
    )
    for path in cache.glob("*.npy"):
        metadata["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (cache / "metadata.json").write_text(json.dumps(metadata))
    return source, cache


def _loader_options(source, cache):
    return dict(
        imagenet_path="", resolution=256, cache_format="npy_flat",
        use_latent=True, use_cache=True, cache_path=str(source),
        latent_mmap_cache_path=str(cache), latent_decoder_path="/existing/local-decoder",
        num_classes=3, batch_size=2, num_workers=0, pin_memory=False,
    )


def test_cache_preserves_bits_pairing_rng_and_readonly_storage(packed):
    source, cache = packed
    direct = FlatNpyLatentDataset(source, num_classes=3)
    packed_data = MmapLatentDataset(cache, source_root=source, num_classes=3)
    cpu_rng, numpy_rng = torch.get_rng_state(), np.random.get_state()
    for index in range(9):
        expected, label = direct[index]
        actual, actual_label = packed_data[index]
        assert actual.numpy().tobytes() == expected.numpy().tobytes()
        assert actual_label == label
        actual.fill_(99)
        assert packed_data[index][0].numpy().tobytes() == expected.numpy().tobytes()
    assert torch.equal(cpu_rng, torch.get_rng_state())
    np.testing.assert_equal(numpy_rng, np.random.get_state())
    restored = pickle.loads(pickle.dumps(packed_data))
    assert restored._features is None and restored._labels is None
    assert restored[4][0].numpy().tobytes() == direct[4][0].numpy().tobytes()


def test_direct_and_mmap_match_sampler_two_ranks_and_epochs(packed):
    source, cache = packed
    options = _loader_options(source, cache)
    for rank in (0, 1):
        for epoch in (0, 1):
            results = []
            for path in ("", str(cache)):
                torch.manual_seed(58)
                loader, preprocess, decoder = create_imagenet_split(
                    **{**options, "latent_mmap_cache_path": path, "num_workers": 2},
                    persistent_workers=False, distributed=True, rank=rank, world_size=2,
                )
                loader.sampler.set_epoch(epoch)
                results.append(([preprocess(batch) for batch in loader], torch.get_rng_state()))
                assert decoder._decoder is None
            expected, actual = results
            assert len(expected[0]) == len(actual[0])
            for left, right in zip(expected[0], actual[0]):
                assert left["images"].numpy().tobytes() == right["images"].numpy().tobytes()
                assert torch.equal(left["labels"], right["labels"])
            assert torch.equal(expected[1], actual[1])


def test_exact_validation_matches_original_raw_rgb_order_and_pixels(packed, tmp_path):
    source, cache = packed
    rgb = tmp_path / "rgb"
    for cls, color in (("z", (255, 128, 0)), ("a", (0, 64, 255))):
        directory = rgb / "val" / cls
        directory.mkdir(parents=True)
        for name in ("002.png", "001.png"):
            Image.new("RGB", (270, 266), color=color).save(directory / name)
    original, _, _ = create_imagenet_split(
        imagenet_path=str(rgb), split="val", batch_size=2, num_workers=0, pin_memory=False,
    )
    options = {**_loader_options(source, cache), "imagenet_path": str(rgb), "split": "val"}
    with mock.patch("vae_imagenet.get_vae_enc_dec", side_effect=AssertionError("VAE forbidden")):
        current, preprocess, decoder = create_imagenet_split(**options)
        assert original.dataset.samples == current.dataset.samples
        for expected, actual in zip(original, current):
            assert torch.equal(expected[0], actual[0])
            assert torch.equal(expected[1], actual[1])
            assert preprocess(actual)["images"].shape == (2, 3, 256, 256)
    assert decoder._decoder is None


@pytest.mark.parametrize("changes", [
    {"use_cache": False}, {"use_latent": False}, {"use_aug": True},
    {"return_uint8": True}, {"resolution": 128}, {"cache_format": "unknown"},
])
def test_exact_inputs_reject_encoding_rescaling_or_augmentation_paths(packed, changes):
    source, cache = packed
    with pytest.raises(ValueError):
        create_imagenet_split(**{**_loader_options(source, cache), **changes})


@pytest.mark.parametrize("bad", ["incomplete", "wrong_source", "labels_checksum", "source_changed"])
def test_incomplete_or_changed_cache_is_rejected(packed, bad):
    source, cache = packed
    if bad in ("incomplete", "wrong_source"):
        metadata = json.loads((cache / "metadata.json").read_text())
        metadata["complete" if bad == "incomplete" else "source_root"] = False if bad == "incomplete" else "/wrong/source"
        (cache / "metadata.json").write_text(json.dumps(metadata))
    elif bad == "labels_checksum":
        labels = np.load(cache / "labels.npy")
        labels[0] = 1
        np.save(cache / "labels.npy", labels)
    else:
        np.save(source / "imagenet256_labels" / "0.npy", np.asarray([2], dtype=np.int64))
    with pytest.raises(ValueError):
        MmapLatentDataset(cache, source_root=source, num_classes=3)


def test_source_dtype_is_validated_without_converting(packed):
    source, _ = packed
    path = source / "imagenet256_features" / "0.npy"
    np.save(path, np.load(path).astype(np.float16))
    with pytest.raises(ValueError, match="dtype"):
        read_flat_latent_row(source, 0, num_classes=3)
