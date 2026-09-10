import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ssl_latent_bridge import (
    FrozenResNet50Stage34,
    LatentBridgeConfig,
    LatentToStage2Bridge,
    SSLLatentBridgeFeatureExtractor,
    compare_reverse_drift_latent_gradients,
    load_bridge_checkpoint,
    save_bridge_checkpoint,
    stage_map_distillation_metrics,
)
from scripts.train_ssl_latent_bridge import (
    build_local_finite_difference_teacher_bank,
    calibrate_bridge_output_rms,
    finite_difference_distillation_loss,
    local_finite_difference_distillation,
    local_finite_difference_distillation_from_bank,
)


class _ResizeProjection(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, 1)
        self.stride = stride

    def forward(self, x):
        if self.stride > 1:
            x = F.avg_pool2d(x, self.stride, self.stride)
        return F.relu(self.projection(x))


class _TinyTail(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.layer3 = nn.Sequential(
            _ResizeProjection(512, 1024, 2),
            *[_ResizeProjection(1024, 1024, 1) for _ in range(5)],
        )
        self.layer4 = nn.Sequential(
            _ResizeProjection(1024, 2048, 2),
            *[_ResizeProjection(2048, 2048, 1) for _ in range(2)],
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class _LinearActivationExtractor(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def get_activations(self, x, **_kwargs):
        base = x * self.scale
        tokens = base.flatten(2).transpose(1, 2)
        return {"layer3": tokens, "layer4": tokens * 0.5}


class _ScalarBridge(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(value)))


class SSLLatentBridgeTest(unittest.TestCase):
    def test_bridge_shape_input_affine_and_gradient(self):
        config = LatentBridgeConfig(
            out_channels=8,
            width=16,
            depth=2,
            expansion=2,
            kernel_size=3,
            norm_groups=4,
        )
        bridge = LatentToStage2Bridge(config)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        y = bridge(x)
        self.assertEqual(y.shape, (2, 8, 8, 8))
        self.assertTrue((y >= 0).all())
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(tuple(bridge.input_mean.shape), (1, 4, 1, 1))
        self.assertEqual(tuple(bridge.input_std.shape), (1, 4, 1, 1))

    def test_tiny_overfit(self):
        torch.manual_seed(3)
        bridge = LatentToStage2Bridge(
            LatentBridgeConfig(
                out_channels=6,
                width=12,
                depth=1,
                expansion=1,
                kernel_size=3,
                norm_groups=3,
            )
        )
        target_net = nn.Conv2d(4, 6, 1)
        for parameter in target_net.parameters():
            parameter.requires_grad_(False)
        x = torch.randn(2, 4, 6, 6)
        target = F.relu(target_net(x))
        optimizer = torch.optim.Adam(bridge.parameters(), lr=1e-2)
        losses = []
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(bridge(x), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0] * 0.35)

    def test_output_rms_calibration_is_exact_positive_row_rescale(self):
        torch.manual_seed(4)
        bridge = LatentToStage2Bridge(
            LatentBridgeConfig(
                out_channels=6,
                width=12,
                depth=1,
                expansion=1,
                kernel_size=3,
                norm_groups=3,
            )
        )
        latent = torch.randn(3, 4, 6, 6)
        with torch.no_grad():
            before = bridge(latent)
            channel_ratio = torch.tensor(
                [0.1, 0.2, 0.4, 0.8, 1.5, 2.0]
            ).view(1, 6, 1, 1)
            target = before * channel_ratio
        with patch(
            "scripts.train_ssl_latent_bridge.teacher_stage_maps",
            return_value={"layer2": target},
        ):
            calibrate_bridge_output_rms(
                bridge,
                nn.Identity(),
                latent,
                use_bf16=False,
            )
        with torch.no_grad():
            after = bridge(latent)
        torch.testing.assert_close(after, target, rtol=2e-5, atol=2e-6)

    def test_finite_difference_loss_matches_deltas_and_is_first_order(self):
        epsilon = 0.05
        base = torch.randn(2, 3, 4, 4)
        delta = torch.randn_like(base)
        scale = nn.Parameter(torch.tensor(0.4))
        target_base = {stage: base for stage in ("layer2", "layer3", "layer4")}
        target_perturbed = {
            stage: base + epsilon * delta
            for stage in ("layer2", "layer3", "layer4")
        }
        predicted_base = {
            stage: base * scale for stage in ("layer2", "layer3", "layer4")
        }
        predicted_perturbed = {
            stage: base * scale + epsilon * delta * scale
            for stage in ("layer2", "layer3", "layer4")
        }
        loss, metrics = finite_difference_distillation_loss(
            predicted_base,
            predicted_perturbed,
            target_base,
            target_perturbed,
            epsilon=epsilon,
            stage_weights={"layer2": 1.0, "layer3": 0.5, "layer4": 0.5},
            cosine_weight=0.5,
        )
        self.assertGreater(float(loss.detach()), 0.1)
        self.assertAlmostEqual(
            metrics["jacobian/layer3_delta_rms_ratio"], 0.4, places=5
        )
        loss.backward()
        self.assertIsNotNone(scale.grad)
        self.assertFalse(target_perturbed["layer2"].requires_grad)

    def test_local_finite_difference_uses_seeded_zero_pairs(self):
        bridge = _ScalarBridge(0.5)
        reference = torch.randn(8, 4, 5, 5)
        seen = []

        def maps(value):
            return {
                "layer2": value,
                "layer3": value * 2.0,
                "layer4": value * 3.0,
            }

        def teacher_maps(_teacher, value):
            seen.append(value.detach().clone())
            return maps(value)

        def student_maps(candidate, _teacher, value):
            return maps(value * candidate.scale)

        with patch(
            "scripts.train_ssl_latent_bridge.teacher_stage_maps",
            side_effect=teacher_maps,
        ), patch(
            "scripts.train_ssl_latent_bridge.student_stage_maps",
            side_effect=student_maps,
        ):
            loss, metrics = local_finite_difference_distillation(
                bridge,
                nn.Identity(),
                reference,
                count=3,
                epsilon=0.05,
                stage_weights={"layer2": 1.0, "layer3": 0.5, "layer4": 0.5},
                cosine_weight=0.5,
                use_bf16=False,
                seed=123,
            )
        self.assertEqual(seen[0].shape[0], 6)
        torch.testing.assert_close(seen[0][:3], torch.zeros_like(seen[0][:3]))
        per_sample_rms = seen[0][3:].square().mean(dim=(1, 2, 3)).sqrt()
        torch.testing.assert_close(per_sample_rms, torch.full_like(per_sample_rms, 0.05))
        self.assertAlmostEqual(metrics["jacobian/latent_delta_rms"], 0.05, places=6)
        self.assertEqual(metrics["jacobian/count_per_rank"], 3.0)
        loss.backward()
        self.assertIsNotNone(bridge.scale.grad)

    def test_cached_teacher_bank_chunks_once_and_cycles_without_teacher(self):
        reference = torch.randn(8, 4, 5, 5)
        teacher_batch_sizes = []

        def maps(value):
            return {
                "layer2": value,
                "layer3": value * 2.0,
                "layer4": value * 3.0,
            }

        def teacher_maps(_teacher, value):
            teacher_batch_sizes.append(int(value.shape[0]))
            return maps(value)

        with patch(
            "scripts.train_ssl_latent_bridge.teacher_stage_maps",
            side_effect=teacher_maps,
        ):
            bank = build_local_finite_difference_teacher_bank(
                nn.Identity(),
                reference,
                bank_size=5,
                epsilon=0.05,
                teacher_chunk_size=2,
                use_bf16=False,
                seed=321,
            )
        self.assertEqual(teacher_batch_sizes, [1, 2, 2, 1])
        self.assertEqual(bank.size, 5)
        self.assertEqual(bank.directions.device.type, "cpu")
        self.assertEqual(bank.target_delta["layer2"].dtype, torch.bfloat16)
        torch.testing.assert_close(
            bank.target_delta["layer2"].float(),
            bank.directions,
            rtol=5e-3,
            atol=5e-3,
        )

        bridge = _ScalarBridge(0.5)
        student_inputs = []

        def student_maps(candidate, _teacher, value):
            student_inputs.append(value.detach().clone())
            return maps(value * candidate.scale)

        with patch(
            "scripts.train_ssl_latent_bridge.teacher_stage_maps",
            side_effect=AssertionError("cached step must not call the teacher prefix"),
        ), patch(
            "scripts.train_ssl_latent_bridge.student_stage_maps",
            side_effect=student_maps,
        ):
            loss, metrics = local_finite_difference_distillation_from_bank(
                bridge,
                nn.Identity(),
                bank,
                count=2,
                application_index=0,
                stage_weights={"layer2": 1.0, "layer3": 0.5, "layer4": 0.5},
                cosine_weight=0.5,
                use_bf16=False,
            )
            _, second_metrics = local_finite_difference_distillation_from_bank(
                bridge,
                nn.Identity(),
                bank,
                count=2,
                application_index=1,
                stage_weights={"layer2": 1.0, "layer3": 0.5, "layer4": 0.5},
                cosine_weight=0.5,
                use_bf16=False,
            )
        self.assertEqual(student_inputs[0].shape[0], 3)
        torch.testing.assert_close(
            student_inputs[0][1:], bank.directions[:2] * 0.05
        )
        self.assertEqual(metrics["jacobian/bank_index_start"], 0.0)
        self.assertEqual(second_metrics["jacobian/bank_index_start"], 2.0)
        loss.backward()
        self.assertIsNotNone(bridge.scale.grad)

    def test_checkpoint_provenance_and_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher.pth"
            teacher.write_bytes(b"official-teacher-placeholder")
            checkpoint = root / "bridge.pt"
            bridge = LatentToStage2Bridge(
                LatentBridgeConfig(width=8, depth=1, expansion=1, kernel_size=3)
            )
            saved = save_bridge_checkpoint(
                checkpoint,
                bridge,
                backbone_name="dino",
                teacher_checkpoint_path=teacher,
                vae_model_id="test-vae",
                vae_revision="revision",
            )
            restored, metadata = load_bridge_checkpoint(
                checkpoint, expected_backbone_name="dino_resnet50"
            )
            self.assertEqual(restored.config, bridge.config)
            self.assertEqual(metadata["teacher"]["vae_revision"], "revision")
            self.assertEqual(len(saved["teacher"]["feature_checkpoint_sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "trained for"):
                load_bridge_checkpoint(
                    checkpoint, expected_backbone_name="moco_v2_resnet50"
                )

    def test_runtime_emits_exact_35_stage34_keys_and_owns_no_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher.pth"
            teacher.write_bytes(b"teacher")
            checkpoint = root / "bridge.pt"
            bridge = LatentToStage2Bridge(
                LatentBridgeConfig(width=8, depth=1, expansion=1, kernel_size=3)
            )
            save_bridge_checkpoint(
                checkpoint,
                bridge,
                backbone_name="dino",
                teacher_checkpoint_path=teacher,
                vae_model_id="test-vae",
                vae_revision="revision",
            )
            with patch("models.ssl_latent_bridge.FrozenResNet50Stage34", _TinyTail):
                runtime = SSLLatentBridgeFeatureExtractor(
                    "dino",
                    teacher,
                    checkpoint,
                    use_bf16=False,
                    use_remat=False,
                    microbatch_size=1,
                    spatial_pool=2,
                )

            called = []
            hooks = [
                module.register_forward_hook(
                    lambda _module, _inputs, _output, name=name: called.append(name)
                )
                for name, module in runtime.named_modules()
            ]
            output = runtime.get_activations(
                torch.randn(1, 4, 32, 32),
                active_stages=["stage3", "stage4"],
                patch_mean_size=[2, 4],
                patch_std_size=[2, 4],
                use_mean=True,
                use_std=True,
                with_global=False,
                with_norm_x=False,
                every_k_block=2,
                exclude_terminal_block=True,
            )
            for hook in hooks:
                hook.remove()
            self.assertEqual(len(output), 35)
            self.assertEqual(output["layer3"].shape, (1, 64, 1024))
            self.assertEqual(output["layer4"].shape, (1, 16, 2048))
            module_names = [name for name, _ in runtime.named_modules()]
            for forbidden in ("vae", "conv1", "bn1", "maxpool", "layer1", "layer2"):
                self.assertNotIn(forbidden, module_names)
                self.assertNotIn(forbidden, called)
            with self.assertRaisesRegex(ValueError, "only stage3/stage4"):
                runtime.get_activations(
                    torch.randn(1, 4, 32, 32), active_stages=["stage2"]
                )

    def test_stage_metrics_and_identical_drift_gradients(self):
        maps = {
            "layer3": torch.randn(4, 3, 4, 4),
            "layer4": torch.randn(4, 5, 2, 2),
        }
        metrics = stage_map_distillation_metrics(maps, maps)
        self.assertAlmostEqual(metrics["layer3/raw_cosine"], 1.0, places=6)
        self.assertAlmostEqual(metrics["layer4/normalized_rmse"], 0.0, places=6)
        self.assertAlmostEqual(
            metrics["layer3/pair_distance_correlation"], 1.0, places=5
        )

        latents = torch.randn(5, 4, 2, 2)
        extractor = _LinearActivationExtractor()
        gradient_metrics = compare_reverse_drift_latent_gradients(
            extractor,
            extractor,
            latents[:2],
            latents[2:4],
            latents[4:5],
            batch_size=1,
            generated_count=2,
            positive_count=2,
            negative_count=1,
        )
        self.assertAlmostEqual(
            gradient_metrics["drift_gradient/cosine"], 1.0, places=5
        )
        self.assertAlmostEqual(
            gradient_metrics["drift_gradient/normalized_rmse"], 0.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
