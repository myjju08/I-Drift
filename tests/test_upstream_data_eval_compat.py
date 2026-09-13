"""Upstream data/evaluation APIs coexist with the local exact-cache runtime."""
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from scripts.eval_official_imagenet256 import _load_imagenet_val_labels, PixelDecodeModule, VaeDecodeModule
from train.train_data import _resolve_npy_flat_split, create_imagenet_split
import vae_imagenet


def _flat_split(root: Path, label: int):
    (root / "imagenet256_features").mkdir(parents=True)
    (root / "imagenet256_labels").mkdir()
    np.save(root / "imagenet256_features/0.npy", np.full((1, 4, 32, 32), label, dtype=np.float32))
    np.save(root / "imagenet256_labels/0.npy", np.asarray([label], dtype=np.int64))


def test_split_aware_flat_cache_prefers_requested_split_and_supports_direct_root(tmp_path):
    _flat_split(tmp_path, 9)
    _flat_split(tmp_path / "train", 1)
    _flat_split(tmp_path / "val", 2)
    assert _resolve_npy_flat_split(str(tmp_path), "train") == str(tmp_path / "train")
    assert _resolve_npy_flat_split(str(tmp_path / "val"), "val") == str(tmp_path / "val")
    with pytest.raises(FileNotFoundError, match="Flat latent cache split"):
        _resolve_npy_flat_split(str(tmp_path / "missing"), "train")
    train, _, _ = create_imagenet_split(imagenet_path="", use_latent=True, use_cache=True,
                                      cache_path=str(tmp_path), cache_format="npy_flat", split="train",
                                      batch_size=1, num_workers=0, pin_memory=False)
    val, preprocess, postprocess = create_imagenet_split(imagenet_path="", use_latent=True, use_cache=True,
                                      cache_path=str(tmp_path), cache_format="npy_flat", split="val",
                                      batch_size=1, num_workers=0, pin_memory=False, vae_variant="ema")
    assert train.dataset[0][1] == 1 and val.dataset[0][1] == 2
    assert preprocess(next(iter(val)))["images"].dtype == torch.float32
    with mock.patch("vae_imagenet.get_vae_enc_dec", return_value=(None, lambda value: value[:, :3])) as load:
        postprocess(torch.zeros(1, 4, 32, 32))
    assert load.call_args.kwargs["model_id"] == "stabilityai/sd-vae-ft-ema"


def test_vae_variant_keywords_and_legacy_positional_variant_preserve_explicit_models():
    fake = object()
    with mock.patch("vae_imagenet.load_vae", return_value=fake) as load:
        vae_imagenet.get_vae_enc_dec(variant="ema")
        assert load.call_args.kwargs["model_id"] == "stabilityai/sd-vae-ft-ema"
        vae_imagenet.get_vae_enc_dec("cpu", "ema")
        assert load.call_args.kwargs["model_id"] == "stabilityai/sd-vae-ft-ema"
        vae_imagenet._load_vae("cpu", variant="ema")
        assert load.call_args.kwargs["model_id"] == "stabilityai/sd-vae-ft-ema"
        vae_imagenet.get_vae_enc_dec(model_id="/existing/pinned-decoder", revision="pinned", variant="ema")
        assert load.call_args.kwargs == {"device": None, "model_id": "/existing/pinned-decoder", "revision": "pinned"}
        vae_imagenet.get_vae_enc_dec()
        assert load.call_args.kwargs["model_id"] == "stabilityai/sd-vae-ft-mse"
    with pytest.raises(ValueError, match="Unknown SD-VAE variant"):
        vae_imagenet.get_vae_enc_dec(variant="invalid")


def test_offline_eval_uses_pinned_decoder_and_scaling_with_legacy_fallback():
    from types import SimpleNamespace
    from scripts import eval_official_imagenet256 as evaluator

    class Decoder(torch.nn.Module):
        def decode(self, value):
            self.received = value
            return SimpleNamespace(sample=value[:, :3])

    decoder = Decoder()
    with mock.patch.object(evaluator, "load_vae", return_value=decoder) as load:
        module = VaeDecodeModule(torch.device("cpu"), "/pinned/sd-vae-ft-mse", 0.5)
        result = module(torch.full((1, 4, 2, 2), 0.25))
        load.assert_called_once_with(device=torch.device("cpu"), model_id="/pinned/sd-vae-ft-mse")
        torch.testing.assert_close(decoder.received, torch.full((1, 4, 2, 2), 0.5))
        torch.testing.assert_close(result, torch.full((1, 3, 2, 2), 0.75))
    with mock.patch.object(evaluator, "_load_vae", return_value=decoder) as load:
        assert VaeDecodeModule(torch.device("cpu")).scaling_factor == 0.18215
        load.assert_called_once_with(torch.device("cpu"))
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="scaling factor"):
            VaeDecodeModule(torch.device("cpu"), scaling_factor=invalid)


def test_val_label_loading_uses_numeric_flat_order_and_environment_override(tmp_path, monkeypatch):
    stale = tmp_path / "stale"
    actual = tmp_path / "actual"
    _flat_split(stale / "val", 99)
    labels = actual / "val/imagenet256_labels"
    labels.mkdir(parents=True)
    for index, value in ((10, 4), (2, 3), (0, 8)):
        np.save(labels / f"{index}.npy", np.asarray([value]))
    monkeypatch.setenv("IMAGENET_CACHE_PATH", str(actual))
    values = _load_imagenet_val_labels({"use_cache": True, "cache_path": str(stale)})
    np.testing.assert_array_equal(values, [8, 3, 4])


def test_val_label_loading_keeps_legacy_pt_support_and_pixel_decode(tmp_path, monkeypatch):
    monkeypatch.delenv("IMAGENET_CACHE_PATH", raising=False)
    for class_name in ("n2", "n1"):
        path = tmp_path / "val" / class_name
        path.mkdir(parents=True)
        (path / "0.pt").write_bytes(b"labels require filenames only")
    values = _load_imagenet_val_labels({"use_cache": True, "cache_path": str(tmp_path)})
    np.testing.assert_array_equal(values, [0, 1])
    pixels = torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1, 1)
    torch.testing.assert_close(PixelDecodeModule()(pixels), torch.tensor([0.0, 0.5, 1.0]).reshape_as(pixels))
