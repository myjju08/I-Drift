import unittest
from collections import defaultdict
from unittest.mock import patch

import torch
import torch.nn as nn
from einops import rearrange

from models.feature_statistics import safe_mean_std, safe_rms
from models.ssl_resnet import SSLResNetFeatureExtractor




def _empty_backbone():
    backbone = nn.Sequential()
    backbone.fc = nn.Identity()
    return backbone


class _AffineBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return x * self.scale + 0.01


class _TinyResNet(nn.Module):
    """Cheap ResNet-shaped backbone with the R50 3/4/6/3 block geometry."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Identity()
        self.bn1 = nn.Identity()
        self.relu = nn.Identity()
        self.maxpool = nn.Identity()
        self.layer1 = nn.ModuleList([_AffineBlock(1.01) for _ in range(3)])
        self.layer2 = nn.ModuleList([_AffineBlock(1.02) for _ in range(4)])
        self.layer3 = nn.ModuleList([_AffineBlock(1.03) for _ in range(6)])
        self.layer4 = nn.ModuleList([_AffineBlock(1.04) for _ in range(3)])


class _TinySSLExtractor(SSLResNetFeatureExtractor):
    def __init__(self, *, real_microbatch_size=4, generated_microbatch_size=2):
        nn.Module.__init__(self)
        self.backbone = _TinyResNet()
        self.use_latent = False
        self.use_remat = False
        self.microbatch_size = 2
        self.real_microbatch_size = int(real_microbatch_size)
        self.generated_microbatch_size = int(generated_microbatch_size)
        self.spatial_pool = 1
        self.include_norm_x = True
        self.forward_batch_sizes = []

    def _decode_and_normalize(self, x):
        return x

    def _forward_selected_maps(self, x, names):
        self.forward_batch_sizes.append(int(x.shape[0]))
        return super()._forward_selected_maps(x, names)


def _chunked_reference(extractor, x):
    """Reference the former process-each-map-inside-each-chunk behavior."""
    names = extractor._selected_map_names(
        every_k_block=2,
        active={"stage3", "stage4"},
        exclude_terminal_block=False,
    )
    result = {
        "global": rearrange(x, "b c h w -> b 1 (c h w)"),
        "norm_x": safe_rms(x, dim=(2, 3)).unsqueeze(1),
    }
    parts = defaultdict(list)
    for start in range(0, int(x.shape[0]), extractor.generated_microbatch_size):
        maps = extractor._forward_selected_maps(
            x[start : start + extractor.generated_microbatch_size],
            names,
        )
        for name, feature in zip(names, maps):
            processed = extractor._process_feature(
                name,
                feature,
                patch_mean_size=[2, 4],
                patch_std_size=[2, 4],
                use_std=True,
                use_mean=True,
            )
            for key, value in processed.items():
                parts[key].append(value)
    result.update(
        {
            key: values[0] if len(values) == 1 else torch.cat(values, dim=0)
            for key, values in parts.items()
        }
    )
    return result


class SSLResNetConfigTest(unittest.TestCase):
    def _build(self, **kwargs):
        with (
            patch("models.ssl_resnet.resnet50", return_value=_empty_backbone()),
            patch("models.ssl_resnet._load_backbone_state", return_value={}),
        ):
            return SSLResNetFeatureExtractor("dino", "dummy.pth", **kwargs)

    def test_dino_defaults_to_raw_rgb_and_preserves_normalization(self):
        model = self._build(use_remat=False)
        values = torch.tensor([-1.3, -1.0, 0.0, 1.0, 1.3]).reshape(1, 1, 1, 5).repeat(2, 3, 4, 1)
        values.requires_grad_()
        expected = ((values + 1.0) * 0.5 - model.imagenet_mean) / model.imagenet_std
        actual = model._decode_and_normalize(values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        left, = torch.autograd.grad(actual.square().sum(), values, retain_graph=True)
        right, = torch.autograd.grad(expected.square().sum(), values)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertFalse(model.use_latent)
        self.assertFalse(model.training)

    def test_rejects_removed_backbones_and_latent_inputs(self):
        from models.ssl_resnet import canonical_ssl_backbone
        for name in ("moco", "moco_v2_resnet50", "mae", "dino_latent_bridge"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                canonical_ssl_backbone(name)
        with self.assertRaisesRegex(ValueError, "direct RGB"):
            self._build(use_latent=True)




class SSLResNetActivationOptimizationTest(unittest.TestCase):
    def test_terminal_stage_maps_are_preserved_for_adversarial_structure(self):
        extractor = _TinySSLExtractor(generated_microbatch_size=2)
        values = torch.randn(5, 3, 4, 4, requires_grad=True)
        features, maps = extractor.get_activations(
            values,
            active_stages=["stage3", "stage4"],
            return_stage_features=True,
        )
        self.assertEqual(set(maps), {"layer3", "layer4"})
        expected = extractor._forward_selected_maps(values, ("layer3", "layer4"))
        for name, reference in zip(("layer3", "layer4"), expected):
            torch.testing.assert_close(maps[name], reference, rtol=0, atol=0)
            torch.testing.assert_close(
                features[name], rearrange(maps[name], "b c h w -> b (h w) c"),
                rtol=0, atol=0,
            )

    def test_routes_real_and_generated_paths_to_separate_microbatch_sizes(self):
        extractor = _TinySSLExtractor(
            real_microbatch_size=4,
            generated_microbatch_size=2,
        )
        values = torch.randn(5, 3, 4, 4, requires_grad=True)
        kwargs = dict(
            patch_mean_size=[],
            patch_std_size=[],
            use_mean=False,
            use_std=False,
            with_global=False,
            with_norm_x=False,
            every_k_block=float("inf"),
            active_stages=["stage3"],
        )

        with torch.no_grad():
            extractor.get_activations(values, **kwargs)
        self.assertEqual(extractor.forward_batch_sizes, [4, 1])

        extractor.forward_batch_sizes.clear()
        extractor.get_activations(values, **kwargs)
        self.assertEqual(extractor.forward_batch_sizes, [2, 2, 1])

    def test_deferred_stats_and_duplicate_aliases_match_chunked_reference(self):
        extractor = _TinySSLExtractor(generated_microbatch_size=2)
        initial = torch.randn(5, 3, 4, 4)
        reference_input = initial.clone().requires_grad_(True)
        optimized_input = initial.clone().requires_grad_(True)

        reference = _chunked_reference(extractor, reference_input)
        optimized = extractor.get_activations(
            optimized_input,
            patch_mean_size=[2, 4],
            patch_std_size=[2, 4],
            use_std=True,
            use_mean=True,
            every_k_block=2,
            exclude_terminal_block=False,
            active_stages=["stage3", "stage4"],
        )

        self.assertEqual(list(reference), list(optimized))
        self.assertEqual(len(optimized), 44)
        for key in reference:
            self.assertTrue(torch.equal(reference[key], optimized[key]), key)

        suffixes = ("", "_mean", "_std", "_mean_2", "_mean_4", "_std_2", "_std_4")
        for suffix in suffixes:
            self.assertIs(
                optimized[f"layer3{suffix}"],
                optimized[f"layer3_blk6{suffix}"],
            )
        for name in ("layer4", "layer4_blk2"):
            self.assertIs(optimized[f"{name}_mean"], optimized[f"{name}_mean_4"])
            self.assertIs(optimized[f"{name}_std"], optimized[f"{name}_std_4"])

        weights = {
            key: (index + 1.0) / 37.0
            for index, key in enumerate(reference)
        }
        reference_loss = sum(
            weights[key] * value.float().square().mean()
            for key, value in reference.items()
        )
        optimized_loss = sum(
            weights[key] * value.float().square().mean()
            for key, value in optimized.items()
        )
        reference_grad, = torch.autograd.grad(reference_loss, reference_input)
        optimized_grad, = torch.autograd.grad(optimized_loss, optimized_input)
        torch.testing.assert_close(
            optimized_grad,
            reference_grad,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_full_map_patch_stats_reuse_preserves_independent_weights(self):
        extractor = _TinySSLExtractor()
        initial = torch.randn(3, 7, 4, 4)
        reference_input = initial.clone().requires_grad_(True)
        optimized_input = initial.clone().requires_grad_(True)

        spatial_mean, spatial_std = safe_mean_std(reference_input, dim=(2, 3))
        spatial_mean = spatial_mean.unsqueeze(1)
        spatial_std = spatial_std.unsqueeze(1)
        patches = rearrange(
            reference_input,
            "b c (h ph) (w pw) -> b (h w) (ph pw) c",
            ph=4,
            pw=4,
        )
        patch_mean, patch_std = safe_mean_std(patches, dim=2)
        optimized = extractor._process_feature(
            "layer4",
            optimized_input,
            patch_mean_size=[4],
            patch_std_size=[4],
            use_std=True,
            use_mean=True,
        )

        self.assertTrue(torch.equal(optimized["layer4_mean"], spatial_mean))
        self.assertTrue(torch.equal(optimized["layer4_std"], spatial_std))
        self.assertTrue(torch.equal(optimized["layer4_mean_4"], patch_mean))
        self.assertTrue(torch.equal(optimized["layer4_std_4"], patch_std))
        self.assertIs(optimized["layer4_mean"], optimized["layer4_mean_4"])
        self.assertIs(optimized["layer4_std"], optimized["layer4_std_4"])

        reference_loss = (
            0.7 * spatial_mean.float().square().mean()
            + 1.3 * patch_mean.float().square().mean()
            + 0.9 * spatial_std.float().square().mean()
            + 1.1 * patch_std.float().square().mean()
        )
        optimized_loss = (
            0.7 * optimized["layer4_mean"].float().square().mean()
            + 1.3 * optimized["layer4_mean_4"].float().square().mean()
            + 0.9 * optimized["layer4_std"].float().square().mean()
            + 1.1 * optimized["layer4_std_4"].float().square().mean()
        )
        reference_grad, = torch.autograd.grad(reference_loss, reference_input)
        optimized_grad, = torch.autograd.grad(optimized_loss, optimized_input)
        torch.testing.assert_close(
            optimized_grad,
            reference_grad,
            rtol=1e-5,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
