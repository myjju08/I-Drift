from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from train.latent_decoder import _DecoderOnly, LatentDecoderPostprocessor, load_local_vae_decoder


class _DecodeProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.inputs = []

    def forward(self, latents):
        self.inputs.append(latents.clone())
        return latents[:, :3] * self.weight


def test_lazy_decoder_only_scales_at_eval_and_has_explicit_cpu_release():
    decoder = _DecodeProbe()
    with mock.patch("train.latent_decoder.load_local_vae_decoder", return_value=decoder) as load:
        postprocess = LatentDecoderPostprocessor()
        load.assert_not_called()
        latents = torch.randn(2, 4, 32, 32, dtype=torch.bfloat16)
        original = latents.clone()
        actual = postprocess(latents)
        expected_input = latents.float() / 0.18215
        torch.testing.assert_close(decoder.inputs[-1], expected_input, rtol=0, atol=0)
        torch.testing.assert_close(actual, ((expected_input[:, :3] + 1) / 2).clamp(0, 1), rtol=0, atol=0)
        assert torch.equal(latents, original)
        assert not actual.requires_grad
        postprocess.release()
        assert postprocess._device == torch.device("cpu")
        postprocess(latents)
        load.assert_called_once_with("stabilityai/sd-vae-ft-mse")


def test_loader_requires_local_weights_and_never_retains_or_calls_encoder():
    fake_vae = SimpleNamespace(
        post_quant_conv=nn.Conv2d(4, 4, 1), decoder=nn.Conv2d(4, 3, 1),
        encoder=mock.Mock(side_effect=AssertionError("encoder forbidden")),
        quant_conv=mock.Mock(side_effect=AssertionError("encoder projection forbidden")),
    )
    with mock.patch("diffusers.AutoencoderKL.from_pretrained", return_value=fake_vae) as load:
        decoder = load_local_vae_decoder("/existing/vae")
    assert load.call_args.args == ("/existing/vae",)
    assert load.call_args.kwargs == dict(local_files_only=True, use_safetensors=True, torch_dtype=torch.float32)
    assert not hasattr(decoder, "encoder")
    assert not hasattr(decoder, "quant_conv")
    assert all(not p.requires_grad and p.device.type == "cpu" for p in decoder.parameters())
    decoder(torch.zeros(1, 4, 32, 32))
    fake_vae.encoder.assert_not_called()
    fake_vae.quant_conv.assert_not_called()


def test_decoder_only_matches_canonical_autoencoder_decode_exactly_on_cpu():
    from diffusers import AutoencoderKL

    torch.manual_seed(204)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16), layers_per_block=1, norm_num_groups=4,
        latent_channels=4, sample_size=64,
    ).eval()
    latents = torch.randn(1, 4, 32, 32)
    with mock.patch.object(vae.encoder, "forward", side_effect=AssertionError("encoder forbidden")), \
         mock.patch.object(vae.quant_conv, "forward", side_effect=AssertionError("encoder projection forbidden")), \
         torch.no_grad():
        expected = vae.decode(latents / 0.18215).sample
        decoder = _DecoderOnly(vae).eval()
        actual = decoder(latents / 0.18215)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_decoder_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="scaling_factor"):
        LatentDecoderPostprocessor(scaling_factor=scale)


def test_decoder_rejects_rgb_before_loading_weights():
    with mock.patch("train.latent_decoder.load_local_vae_decoder") as load:
        with pytest.raises(ValueError, match="Bx4x32x32"):
            LatentDecoderPostprocessor()(torch.zeros(1, 3, 256, 256))
    load.assert_not_called()
