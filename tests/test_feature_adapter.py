import copy
import unittest
from pathlib import Path

import torch
import yaml

from models.feature_adapter import (
    FeatureAdapterSystem,
    ResidualSpatialAdapter,
    canonical_adapter_stages,
    counterfactual_field_consistency_loss,
    generated_to_real_multi_positive_info_nce,
    pairwise_cosine_relation_mse,
    pairwise_distance_geometry_losses,
    pairwise_distance_relation_loss,
    real_to_real_multi_positive_info_nce,
    rotating_partition_indices,
    supervised_contrastive_loss,
    update_adapter_ema,
)
from models.mae_resnet import MAEResNet


class FeatureAdapterTest(unittest.TestCase):
    def test_dino_scale_config_is_geometry_only_ablation(self):
        config_root = Path(__file__).resolve().parents[1] / "configs" / "gen"
        baseline = yaml.safe_load(
            (
                config_root
                / (
                    "S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-"
                    "matched-genreal-mp-infonce-adapter-align.yaml"
                )
            ).read_text(encoding="utf-8")
        )
        scale = yaml.safe_load(
            (
                config_root
                / (
                    "S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-"
                    "matched-genreal-mp-infonce-adapter-align-scale.yaml"
                )
            ).read_text(encoding="utf-8")
        )

        missing = object()

        def changed_paths(left, right, prefix=()):
            if isinstance(left, dict) and isinstance(right, dict):
                changes = set()
                for key in set(left) | set(right):
                    changes.update(
                        changed_paths(
                            left.get(key, missing),
                            right.get(key, missing),
                            (*prefix, str(key)),
                        )
                    )
                return changes
            if left is missing or right is missing or left != right:
                return {".".join(prefix)}
            return set()

        self.assertEqual(
            changed_paths(baseline, scale),
            {
                "logging.name",
                "feature.feature_adapter_drift_align_lambda",
                "feature.feature_adapter_distance_scale_lambda",
                "feature.feature_adapter_distance_mean_lambda",
            },
        )

    def test_canonical_adapter_stages_accepts_layer_and_stage_names(self):
        self.assertEqual(
            canonical_adapter_stages(["layer4", "stage3", "layer4"]),
            ("stage3", "stage4"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown feature-adapter key"):
            canonical_adapter_stages(["layer5"])

    def test_residual_adapter_is_exact_identity_at_initialization(self):
        adapter = ResidualSpatialAdapter(channels=8, bottleneck=3)
        x = torch.randn(4, 8, 5, 5)
        torch.testing.assert_close(adapter(x), x, rtol=0.0, atol=0.0)

    def test_supcon_adapter_objective_is_finite_and_updates_online_only(self):
        system = FeatureAdapterSystem(
            {"stage3": 8, "stage4": 16},
            ["layer3", "layer4"],
            bottleneck=4,
            projection_dim=6,
            num_classes=5,
            use_ce=True,
        )
        target = copy.deepcopy(system).eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        features = {
            "layer3": torch.randn(6, 8, 4, 4),
            "layer4": torch.randn(6, 16, 2, 2),
        }
        labels = torch.tensor([1, 2])
        target_before = [parameter.clone() for parameter in target.parameters()]
        loss, metrics = system(
            features,
            labels,
            batch_size=2,
            positive_count=3,
            samples_per_class=2,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.1,
            reg_weight=0.01,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("adapter/stage3_supcon", metrics)
        loss.backward()
        self.assertIsNotNone(system.adapters["stage4"].up.weight.grad)
        self.assertTrue(torch.isfinite(system.adapters["stage4"].up.weight.grad).all())
        torch.optim.SGD(system.parameters(), lr=0.1).step()
        update_adapter_ema(target, system, decay=0.9)
        self.assertTrue(all(not parameter.requires_grad for parameter in target.parameters()))
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(target_before, target.parameters())
            )
        )

    def test_supervised_contrastive_rejects_singletons(self):
        with self.assertRaisesRegex(ValueError, "same-label positive"):
            supervised_contrastive_loss(
                torch.randn(3, 4), torch.tensor([0, 1, 2]), temperature=0.1
            )

    def test_generated_to_real_multi_positive_infonce_matches_formula(self):
        generated = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        real = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0]]
        )
        generated_labels = torch.tensor([3, 7])
        real_labels = torch.tensor([3, 3, 7, 7])
        temperature = 0.5

        actual = generated_to_real_multi_positive_info_nce(
            generated,
            real,
            generated_labels,
            real_labels,
            temperature,
        )
        logits = torch.nn.functional.normalize(generated, dim=-1) @ (
            torch.nn.functional.normalize(real, dim=-1).T
        ) / temperature
        positive = generated_labels[:, None].eq(real_labels[None, :])
        expected = -(
            (logits - torch.logsumexp(logits, dim=1, keepdim=True))
            .masked_fill(~positive, 0.0)
            .sum(dim=1)
            / positive.sum(dim=1)
        ).mean()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_real_to_real_multi_positive_infonce_masks_only_exact_self(self):
        real = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
        )
        generated = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([3, 3, 7, 7])
        candidate_embeddings = torch.cat([real, generated], dim=0)
        candidate_labels = torch.tensor([3, 3, 7, 7, 3, 7])
        positive_eligible = torch.tensor([True, True, True, True, False, False])
        temperature = 0.5
        self_indices = torch.arange(real.shape[0])

        actual = real_to_real_multi_positive_info_nce(
            real,
            candidate_embeddings,
            labels,
            candidate_labels,
            positive_eligible,
            self_indices,
            temperature,
        )
        normalized_queries = torch.nn.functional.normalize(real, dim=-1)
        normalized_candidates = torch.nn.functional.normalize(
            candidate_embeddings, dim=-1
        )
        logits = normalized_queries @ normalized_candidates.T / temperature
        self_mask = torch.zeros_like(logits, dtype=torch.bool)
        self_mask[torch.arange(real.shape[0]), self_indices] = True
        logits = logits.masked_fill(self_mask, float("-inf"))
        positive = (
            labels[:, None].eq(candidate_labels[None, :])
            & positive_eligible[None, :]
            & ~self_mask
        )
        expected = -(
            (logits - torch.logsumexp(logits, dim=1, keepdim=True))
            .masked_fill(~positive, 0.0)
            .sum(dim=1)
            / positive.sum(dim=1)
        ).mean()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_real_real_adapter_uses_no_generated_features_or_projection(self):
        system = FeatureAdapterSystem(
            {"stage3": 8}, ["stage3"], bottleneck=4, projection_dim=0
        )
        real = torch.randn(6, 8, 3, 3, requires_grad=True)
        generated = torch.randn(4, 8, 3, 3, requires_grad=True)
        loss, metrics = system(
            {"layer3": real},
            torch.tensor([1, 2]),
            batch_size=2,
            positive_count=3,
            samples_per_class=3,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.0,
            reg_weight=0.0,
            objective="real_real_multipos_infonce",
            generated_stage_features={"layer3": generated},
            generated_count=2,
            generated_samples_per_class=2,
        )
        loss.backward()

        self.assertIsNone(real.grad)
        self.assertIsNone(generated.grad)
        self.assertNotIn("stage3", system.projectors)
        self.assertIsNotNone(system.adapters["stage3"].up.weight.grad)
        self.assertGreater(
            float(system.adapters["stage3"].up.weight.grad.abs().sum()), 0.0
        )
        self.assertIn("adapter/stage3_real_real_infonce", metrics)
        self.assertEqual(metrics["adapter/stage3_generated_negatives"], 4.0)

    def test_pairwise_distance_relation_loss_preserves_backbone_topology(self):
        torch.manual_seed(7)
        base = torch.randn(12, 9, requires_grad=True)
        identical = base.detach().clone().requires_grad_(True)
        identity_loss = pairwise_distance_relation_loss(
            base,
            identical,
            real_count=7,
        )
        self.assertLess(float(identity_loss.detach()), 1.0e-6)
        identity_loss.backward()
        self.assertIsNone(base.grad)
        self.assertIsNotNone(identical.grad)
        self.assertTrue(torch.isfinite(identical.grad).all())

        distorted = base.detach().clone()
        distorted[:, 0] *= 8.0
        distorted[:, 1] *= 0.05
        distortion_loss = pairwise_distance_relation_loss(
            base,
            distorted,
            real_count=7,
        )
        self.assertGreater(float(distortion_loss.detach()), 1.0e-2)

    def test_pairwise_distance_geometry_detects_affine_scale_loophole(self):
        torch.manual_seed(17)
        unit = torch.nn.functional.normalize(torch.randn(10, 7), dim=-1)
        base = torch.cat([unit, torch.zeros(10, 1)], dim=-1)
        scale = 0.25
        adapted = torch.cat(
            [
                unit * scale**0.5,
                torch.full((10, 1), (1.0 - scale) ** 0.5),
            ],
            dim=-1,
        ).requires_grad_(True)
        relation, scale_loss, mean_loss = pairwise_distance_geometry_losses(
            base.requires_grad_(True),
            adapted,
            real_count=6,
            epsilon=1.0e-8,
        )
        self.assertLess(float(relation.detach()), 1.0e-6)
        torch.testing.assert_close(
            scale_loss,
            scale_loss.new_tensor(float(torch.log(torch.tensor(scale)).square())),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        self.assertGreater(float(mean_loss.detach()), 0.0)
        (relation + scale_loss + mean_loss).backward()
        self.assertIsNone(base.grad)
        self.assertIsNotNone(adapted.grad)
        self.assertTrue(torch.isfinite(adapted.grad).all())

    def test_pairwise_distance_geometry_is_zero_and_finite_at_identity(self):
        torch.manual_seed(23)
        base = torch.randn(8, 6, requires_grad=True)
        adapted = base.detach().clone().requires_grad_(True)
        losses = pairwise_distance_geometry_losses(
            base,
            adapted,
            real_count=5,
        )
        for loss in losses:
            self.assertTrue(torch.isfinite(loss))
            self.assertLess(abs(float(loss.detach())), 1.0e-6)
        sum(losses).backward()
        self.assertIsNone(base.grad)
        self.assertIsNotNone(adapted.grad)
        self.assertTrue(torch.isfinite(adapted.grad).all())

    def test_counterfactual_field_consistency_matches_formula_and_both_gradients(self):
        first = torch.tensor([[1.0, 2.0], [0.5, -1.0]], requires_grad=True)
        second = torch.tensor([[0.7, 1.5], [-0.2, -0.8]], requires_grad=True)
        epsilon = 1.0e-6
        actual = counterfactual_field_consistency_loss(
            first, second, epsilon=epsilon
        )
        expected = 1.0 - (first * second).mean() / (
            (first.square().mean() * second.square().mean()).sqrt() + epsilon
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        actual.backward()
        for field in (first, second):
            self.assertIsNotNone(field.grad)
            self.assertTrue(torch.isfinite(field.grad).all())
            self.assertGreater(float(field.grad.abs().sum()), 0.0)

    def test_pairwise_cosine_relation_is_exact_and_stops_backbone(self):
        torch.manual_seed(29)
        base = torch.randn(9, 7, requires_grad=True)
        identical = base.detach().clone().requires_grad_(True)
        identity_loss = pairwise_cosine_relation_mse(base, identical)
        torch.testing.assert_close(
            identity_loss, torch.zeros_like(identity_loss), rtol=0.0, atol=0.0
        )
        distorted = base.detach().clone()
        distorted[:, 0] *= 7.0
        distorted[:, 1] *= 0.03
        distortion_loss = pairwise_cosine_relation_mse(base, distorted)
        self.assertGreater(float(distortion_loss), 1.0e-3)
        (identity_loss + pairwise_cosine_relation_mse(base, distorted.requires_grad_())).backward()
        self.assertIsNone(base.grad)
        self.assertIsNotNone(identical.grad)

    def test_rotating_partitions_are_disjoint_complete_and_change_roles(self):
        first = rotating_partition_indices(
            10, (3, 3, 4), 0, device=torch.device("cpu")
        )
        second = rotating_partition_indices(
            10, (3, 3, 4), 1, device=torch.device("cpu")
        )
        for groups in (first, second):
            joined = torch.cat(groups)
            self.assertEqual(joined.unique().numel(), 10)
            self.assertEqual(set(joined.tolist()), set(range(10)))
        self.assertNotEqual(first[0].tolist(), second[0].tolist())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_pairwise_distance_relation_loss_is_fp32_under_cuda_bf16_autocast(self):
        device = torch.device("cuda:0")
        base = torch.randn(12, 9, device=device, dtype=torch.bfloat16)
        adapted = (base.float() + 0.1 * torch.randn_like(base.float())).to(
            torch.bfloat16
        )
        adapted.requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = pairwise_distance_relation_loss(
                base,
                adapted,
                real_count=7,
            )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(adapted.grad)
        self.assertTrue(torch.isfinite(adapted.grad).all())

    def test_generated_real_adapter_adds_weighted_drift_alignment(self):
        torch.manual_seed(11)
        template = FeatureAdapterSystem(
            {"stage3": 8}, ["stage3"], bottleneck=4, projection_dim=0
        )
        with torch.no_grad():
            template.adapters["stage3"].up.weight.normal_(std=0.2)
            template.adapters["stage3"].up.bias.normal_(std=0.1)
        baseline = copy.deepcopy(template)
        aligned = copy.deepcopy(template)
        real = torch.randn(6, 8, 3, 3, requires_grad=True)
        generated = torch.randn(4, 8, 3, 3, requires_grad=True)
        kwargs = dict(
            stage_features={"layer3": real},
            labels=torch.tensor([1, 2]),
            batch_size=2,
            positive_count=3,
            samples_per_class=3,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.0,
            reg_weight=0.0,
            objective="gen_real_multipos_infonce",
            generated_stage_features={"layer3": generated},
            generated_count=2,
            generated_samples_per_class=2,
        )
        baseline_loss, _ = baseline(**kwargs, drift_align_weight=0.0)
        aligned_loss, metrics = aligned(
            **kwargs,
            drift_align_weight=1.5,
            distance_scale_weight=0.1,
            distance_mean_weight=0.05,
        )
        expected = (
            baseline_loss
            + 1.5 * metrics["adapter/drift_align_loss"]
            + 0.1 * metrics["adapter/distance_scale_loss"]
            + 0.05 * metrics["adapter/distance_mean_loss"]
        )
        torch.testing.assert_close(aligned_loss, expected)
        self.assertIn("adapter/stage3_rr_relation_corr", metrics)
        self.assertIn("adapter/stage3_rg_relation_corr", metrics)
        self.assertIn("adapter/stage3_gg_relation_corr", metrics)
        self.assertIn("adapter/stage3_rr_distance_std_scale", metrics)
        self.assertIn("adapter/distance_scale_weighted_loss", metrics)
        self.assertIn("adapter/distance_mean_weighted_loss", metrics)
        aligned_loss.backward()
        self.assertIsNone(real.grad)
        self.assertIsNone(generated.grad)
        self.assertIsNotNone(aligned.adapters["stage3"].up.weight.grad)
        self.assertTrue(
            torch.isfinite(aligned.adapters["stage3"].up.weight.grad).all()
        )

    def test_generated_real_adapter_detaches_feature_inputs(self):
        system = FeatureAdapterSystem(
            {"stage3": 8},
            ["stage3"],
            bottleneck=4,
            projection_dim=0,
        )
        real = torch.randn(4, 8, 3, 3, requires_grad=True)
        generated = torch.randn(4, 8, 3, 3, requires_grad=True)
        loss, metrics = system(
            {"layer3": real},
            torch.tensor([1, 2]),
            batch_size=2,
            positive_count=2,
            samples_per_class=2,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.0,
            reg_weight=0.01,
            objective="gen_real_multipos_infonce",
            generated_stage_features={"layer3": generated},
            generated_count=2,
            generated_samples_per_class=2,
        )
        loss.backward()

        self.assertIsNone(real.grad)
        self.assertIsNone(generated.grad)
        self.assertIsNotNone(system.adapters["stage3"].up.weight.grad)
        self.assertTrue(
            torch.isfinite(system.adapters["stage3"].up.weight.grad).all()
        )
        self.assertGreater(
            float(system.adapters["stage3"].up.weight.grad.abs().sum()), 0.0
        )
        self.assertIn("adapter/stage3_gen_real_infonce", metrics)
        self.assertNotIn("stage3", system.projectors)

    def test_generated_real_nonlog_diagnostics_gate_preserves_loss_and_gradient(self):
        template = FeatureAdapterSystem(
            {"stage3": 8}, ["stage3"], bottleneck=4, projection_dim=0
        )
        with torch.no_grad():
            template.adapters["stage3"].up.weight.normal_(std=0.03)
            template.adapters["stage3"].up.bias.normal_(std=0.01)
        real = torch.randn(6, 8, 3, 3)
        generated = torch.randn(4, 8, 3, 3)
        for reg_weight in (0.0, 0.2):
            with self.subTest(reg_weight=reg_weight):
                reference = copy.deepcopy(template)
                optimized = copy.deepcopy(template)
                kwargs = dict(
                    stage_features={"layer3": real},
                    labels=torch.tensor([1, 2]),
                    batch_size=2,
                    positive_count=3,
                    samples_per_class=3,
                    temperature=0.1,
                    supcon_weight=1.0,
                    ce_weight=0.0,
                    reg_weight=reg_weight,
                    drift_align_weight=1.5,
                    objective="gen_real_multipos_infonce",
                    generated_stage_features={"layer3": generated},
                    generated_count=2,
                    generated_samples_per_class=2,
                )

                reference_loss, reference_metrics = reference(
                    **kwargs, collect_diagnostics=True
                )
                optimized_loss, optimized_metrics = optimized(
                    **kwargs, collect_diagnostics=False
                )
                torch.testing.assert_close(
                    optimized_loss, reference_loss, rtol=0.0, atol=0.0
                )
                self.assertIn("adapter/stage3_top1_accuracy", reference_metrics)
                self.assertEqual(optimized_metrics, {})

                reference_loss.backward()
                optimized_loss.backward()
                for reference_parameter, optimized_parameter in zip(
                    reference.parameters(), optimized.parameters()
                ):
                    if reference_parameter.grad is None:
                        self.assertIsNone(optimized_parameter.grad)
                    else:
                        torch.testing.assert_close(
                            optimized_parameter.grad,
                            reference_parameter.grad,
                            rtol=0.0,
                            atol=0.0,
                        )

    def test_frozen_hard_copy_adapter_passes_only_input_gradient(self):
        online = FeatureAdapterSystem(
            {"stage4": 8}, ["stage4"], bottleneck=4, projection_dim=0
        )
        with torch.no_grad():
            online.adapters["stage4"].up.weight.normal_()
        target = copy.deepcopy(online).eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)

        x = torch.randn(2, 8, 2, 2, requires_grad=True)
        target.adapters["stage4"](x).square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))

        with torch.no_grad():
            online.adapters["stage4"].up.weight.add_(0.25)
        update_adapter_ema(target, online, decay=0.0)
        for target_parameter, online_parameter in zip(
            target.parameters(), online.parameters()
        ):
            torch.testing.assert_close(
                target_parameter, online_parameter, rtol=0.0, atol=0.0
            )

    def test_mae_stage_adapter_hook_preserves_frozen_control_and_identity(self):
        mae = MAEResNet(
            num_classes=10,
            in_channels=4,
            base_channels=8,
            layers=(1, 1, 1, 1),
            use_bf16=False,
            input_patch_size=1,
        ).eval()
        x = torch.randn(2, 4, 8, 8)
        kwargs = dict(
            patch_mean_size=[],
            patch_std_size=[],
            use_mean=True,
            use_std=True,
            with_global=True,
            every_k_block=float("inf"),
        )
        baseline = mae.get_activations(x, **kwargs)
        system = FeatureAdapterSystem(
            {"stage4": 64}, ["stage4"], bottleneck=4, projection_dim=8
        )
        adapted, stage_features = mae.get_activations(
            x,
            **kwargs,
            stage_adapters=system.adapters,
            return_stage_features=True,
        )
        self.assertEqual(
            set(stage_features), {"layer1", "layer2", "layer3", "layer4"}
        )
        self.assertEqual(baseline.keys(), adapted.keys())
        for key in baseline:
            torch.testing.assert_close(
                baseline[key], adapted[key], rtol=0.0, atol=0.0
            )

    def test_mae_active_stages_omit_early_feature_computation(self):
        mae = MAEResNet(
            num_classes=10,
            in_channels=4,
            base_channels=8,
            layers=(2, 2, 2, 2),
            use_bf16=False,
            input_patch_size=1,
        ).eval()
        activations = mae.get_activations(
            torch.randn(2, 4, 8, 8),
            patch_mean_size=[2],
            patch_std_size=[2],
            use_mean=True,
            use_std=True,
            with_global=True,
            every_k_block=1,
            active_stages=["stage3", "stage4"],
        )

        self.assertIn("global", activations)
        self.assertIn("norm_x", activations)
        self.assertTrue(any(name.startswith("layer3") for name in activations))
        self.assertTrue(any(name.startswith("layer4") for name in activations))
        self.assertFalse(
            any(
                name == "conv1"
                or name.startswith("conv1_")
                or name == "layer1"
                or name.startswith("layer1_")
                or name == "layer2"
                or name.startswith("layer2_")
                for name in activations
            )
        )

        with self.assertRaisesRegex(ValueError, "Unknown active feature stages"):
            mae.get_activations(
                torch.randn(1, 4, 8, 8), active_stages=["stage5"]
            )


if __name__ == "__main__":
    unittest.main()
