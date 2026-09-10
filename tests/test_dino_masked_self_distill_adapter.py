import copy
import unittest

import torch
import torch.nn.functional as F

from models.feature_adapter import (
    FeatureAdapterSystem,
    ResidualPatchPredictor,
    make_dino_patch_masked_images,
    project_patch_mask_to_feature_grid,
    update_adapter_ema,
)
from train_imagenet_gen import build_feature_adapter_system


class _DummyFeatureExtractor(torch.nn.Module):
    base_channels = 8


class DinoMaskedSelfDistillAdapterTest(unittest.TestCase):
    def test_builder_restricts_mdino_to_dino_and_splits_predictor_lr(self):
        base_cfg = {
            "feature_adapter": True,
            "feature_adapter_objective": "real_real_multipos_infonce_mdino",
            "feature_adapter_keys": ["layer3", "layer4"],
            "feature_adapter_bottleneck": 4,
            "feature_adapter_mdino_predictor_bottleneck": 3,
            "feature_adapter_lr": 1.0e-5,
            "feature_adapter_mdino_predictor_lr": 1.0e-4,
            "num_classes": 5,
        }
        with self.assertRaisesRegex(ValueError, "requires the direct DINO"):
            build_feature_adapter_system(
                _DummyFeatureExtractor(),
                {**base_cfg, "feature_extractor": "moco_v2_resnet50"},
                torch.device("cpu"),
                world_size=1,
            )

        online, target, optimizer = build_feature_adapter_system(
            _DummyFeatureExtractor(),
            {**base_cfg, "feature_extractor": "dino_resnet50"},
            torch.device("cpu"),
            world_size=1,
        )
        self.assertIsNotNone(online)
        self.assertIsNotNone(target)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [1.0e-5, 1.0e-4],
        )
        self.assertEqual(set(online.masked_predictors), {"stage3", "stage4"})

    def test_patch_mask_is_exact_deterministic_and_rng_isolated(self):
        torch.manual_seed(123)
        rng_before = torch.random.get_rng_state().clone()
        images = torch.arange(3 * 3 * 8 * 8, dtype=torch.float32).reshape(
            3, 3, 8, 8
        )
        fill = (0.1, 0.2, 0.3)
        masked, patch_mask = make_dino_patch_masked_images(
            images,
            patch_size=2,
            mask_ratio=0.25,
            seed=77,
            fill_values=fill,
        )
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        self.assertEqual(tuple(patch_mask.shape), (3, 4, 4))
        torch.testing.assert_close(
            patch_mask.sum(dim=(1, 2)),
            torch.full((3,), 4, dtype=torch.long),
        )

        pixel_mask = patch_mask.repeat_interleave(2, 1).repeat_interleave(2, 2)
        expected_fill = torch.tensor(fill).view(1, 3, 1, 1).expand_as(images)
        torch.testing.assert_close(masked[pixel_mask[:, None].expand_as(images)],
                                   expected_fill[pixel_mask[:, None].expand_as(images)])
        torch.testing.assert_close(masked[~pixel_mask[:, None].expand_as(images)],
                                   images[~pixel_mask[:, None].expand_as(images)])

        masked_again, mask_again = make_dino_patch_masked_images(
            images,
            patch_size=2,
            mask_ratio=0.25,
            seed=77,
            fill_values=fill,
        )
        torch.testing.assert_close(masked_again, masked)
        torch.testing.assert_close(mask_again, patch_mask)
        _, different_mask = make_dino_patch_masked_images(
            images,
            patch_size=2,
            mask_ratio=0.25,
            seed=78,
            fill_values=fill,
        )
        self.assertFalse(torch.equal(different_mask, patch_mask))

    def test_patch_predictor_starts_as_per_token_layer_norm(self):
        predictor = ResidualPatchPredictor(channels=8, bottleneck=3)
        feature = torch.randn(2, 8, 4, 4)
        expected = F.layer_norm(
            feature.permute(0, 2, 3, 1).float(), (8,), eps=1.0e-6
        ).permute(0, 3, 1, 2)
        torch.testing.assert_close(
            predictor(feature), expected, rtol=0.0, atol=0.0
        )

    def test_feature_grid_mask_keeps_exact_budget_without_stage4_expansion(self):
        _, patch_mask = make_dino_patch_masked_images(
            torch.zeros(5, 3, 256, 256),
            patch_size=32,
            mask_ratio=0.4,
            seed=19,
        )
        stage3 = project_patch_mask_to_feature_grid(
            patch_mask, output_size=(8, 8), mask_ratio=0.4
        )
        stage4 = project_patch_mask_to_feature_grid(
            patch_mask, output_size=(4, 4), mask_ratio=0.4
        )
        torch.testing.assert_close(
            stage3.sum(dim=(1, 2)), torch.full((5,), 25, dtype=torch.long)
        )
        torch.testing.assert_close(
            stage4.sum(dim=(1, 2)), torch.full((5,), 6, dtype=torch.long)
        )
        self.assertAlmostEqual(float(stage3.float().mean()), 25.0 / 64.0)
        self.assertAlmostEqual(float(stage4.float().mean()), 6.0 / 16.0)

    def test_mdino_updates_only_adapter_and_predictor(self):
        torch.manual_seed(9)
        system = FeatureAdapterSystem(
            {"stage3": 8},
            ["stage3"],
            bottleneck=4,
            projection_dim=0,
            masked_predictor_bottleneck=4,
        )
        clean = torch.randn(6, 8, 3, 3, requires_grad=True)
        masked = torch.randn(4, 8, 3, 3, requires_grad=True)
        generated = torch.randn(4, 8, 3, 3, requires_grad=True)
        patch_mask = torch.zeros(4, 3, 3, dtype=torch.bool)
        patch_mask[:, 0, 0] = True

        loss, metrics = system(
            {"layer3": clean},
            torch.tensor([1, 2]),
            batch_size=2,
            positive_count=3,
            samples_per_class=3,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.0,
            reg_weight=0.0,
            objective="real_real_multipos_infonce_mdino",
            generated_stage_features={"layer3": generated},
            generated_count=2,
            generated_samples_per_class=2,
            masked_stage_features={"layer3": masked},
            masked_patch_mask=patch_mask,
            mdino_options={
                "weight": 2.0,
                "mask_ratio": 1.0 / 9.0,
                "smooth_l1_weight": 1.0,
                "cosine_weight": 0.1,
                "smooth_l1_beta": 1.0,
                "samples_per_class": 2,
            },
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertIsNone(clean.grad)
        self.assertIsNone(masked.grad)
        self.assertIsNone(generated.grad)
        adapter_grad = system.adapters["stage3"].up.weight.grad
        predictor_grad = system.masked_predictors["stage3"].up.weight.grad
        self.assertIsNotNone(adapter_grad)
        self.assertIsNotNone(predictor_grad)
        self.assertGreater(float(adapter_grad.abs().sum()), 0.0)
        self.assertGreater(float(predictor_grad.abs().sum()), 0.0)
        self.assertIn("adapter/mdino_loss", metrics)
        self.assertIn("adapter/stage3_mdino_mask_fraction", metrics)
        self.assertAlmostEqual(
            float(metrics["adapter/stage3_mdino_mask_fraction"]), 1.0 / 9.0
        )

    def test_mdino_ignores_visible_teacher_positions(self):
        torch.manual_seed(17)
        template = FeatureAdapterSystem(
            {"stage3": 8},
            ["stage3"],
            bottleneck=4,
            projection_dim=0,
            masked_predictor_bottleneck=4,
        )
        clean = torch.randn(6, 8, 3, 3)
        changed = clean.clone()
        # MDINO selects clean indices [0,1,3,4].  Change only a visible cell.
        changed[[0, 1, 3, 4], :, 1, 1] += 100.0
        masked = torch.randn(4, 8, 3, 3)
        generated = torch.randn(4, 8, 3, 3)
        patch_mask = torch.zeros(4, 3, 3, dtype=torch.bool)
        patch_mask[:, 0, 0] = True
        kwargs = dict(
            labels=torch.tensor([1, 2]),
            batch_size=2,
            positive_count=3,
            samples_per_class=3,
            temperature=0.1,
            supcon_weight=0.0,
            ce_weight=0.0,
            reg_weight=0.0,
            objective="real_real_multipos_infonce_mdino",
            generated_stage_features={"layer3": generated},
            generated_count=2,
            generated_samples_per_class=2,
            masked_stage_features={"layer3": masked},
            masked_patch_mask=patch_mask,
            mdino_options={
                "weight": 1.0,
                "mask_ratio": 1.0 / 9.0,
                "smooth_l1_weight": 1.0,
                "cosine_weight": 0.0,
                "samples_per_class": 2,
            },
            collect_diagnostics=False,
        )
        first, _ = copy.deepcopy(template)({"layer3": clean}, **kwargs)
        second, _ = copy.deepcopy(template)({"layer3": changed}, **kwargs)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_ema_target_predictor_never_changes_generator_adapter_api(self):
        online = FeatureAdapterSystem(
            {"stage4": 8},
            ["stage4"],
            bottleneck=4,
            projection_dim=0,
            masked_predictor_bottleneck=4,
        )
        target = copy.deepcopy(online).eval()
        feature = torch.randn(2, 8, 3, 3)
        before = target.adapters["stage4"](feature)
        with torch.no_grad():
            online.masked_predictors["stage4"].up.bias.add_(5.0)
        update_adapter_ema(target, online, decay=0.9)
        after = target.adapters["stage4"](feature)
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
