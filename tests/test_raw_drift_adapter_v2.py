import unittest
from pathlib import Path

import torch
import yaml

from models.feature_adapter import (
    FeatureAdapterSystem,
    derive_ssl_map_features,
    deterministic_class_partition_indices,
    drift_field_snr_v2_statistic,
)


class RawDriftAdapterV2Test(unittest.TestCase):
    def test_v2_statistic_matches_exact_formula_and_full_gradient(self):
        values = (
            torch.linspace(-1.3, 2.1, 24).reshape(2, 3, 4),
            torch.linspace(0.2, 3.5, 24).reshape(2, 3, 4),
            torch.linspace(-2.0, 1.4, 16).reshape(2, 2, 4),
            torch.linspace(0.7, 4.6, 32).reshape(2, 4, 4),
        )
        actual_inputs = tuple(value.clone().requires_grad_() for value in values)
        expected_inputs = tuple(value.clone().requires_grad_() for value in values)
        epsilon = 1.0e-6

        actual = drift_field_snr_v2_statistic(
            *actual_inputs,
            epsilon=epsilon,
            global_statistics=False,
        )

        def vector_energy(field):
            return field.float().square().mean(dim=-1).reshape(-1)

        energy_a, energy_b, energy_rr, energy_qq = tuple(
            vector_energy(field) for field in expected_inputs
        )
        expected = {
            "D_a": energy_a.mean(),
            "D_b": energy_b.mean(),
            "Drr": energy_rr.mean(),
            "Dqq": energy_qq.mean(),
        }
        expected["Dpq"] = 0.5 * (expected["D_a"] + expected["D_b"])
        expected["D0"] = 0.5 * (expected["Drr"] + expected["Dqq"])
        null_count = energy_rr.numel() + energy_qq.numel()
        expected["Var0"] = (
            (energy_rr - expected["D0"]).square().sum()
            + (energy_qq - expected["D0"]).square().sum()
        ) / (null_count - 1)
        expected["J"] = (expected["Dpq"] - expected["D0"]) / (
            expected["Var0"] + epsilon
        ).sqrt()

        self.assertEqual(
            set(actual),
            {"D_a", "D_b", "Dpq", "Drr", "Dqq", "D0", "Var0", "J"},
        )
        for name, expected_value in expected.items():
            torch.testing.assert_close(
                actual[name], expected_value, rtol=0.0, atol=0.0
            )

        actual_gradients = torch.autograd.grad(actual["J"], actual_inputs)
        expected_gradients = torch.autograd.grad(expected["J"], expected_inputs)
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=0.0, atol=0.0
            )
            self.assertTrue(torch.isfinite(actual_gradient).all())
            self.assertGreater(float(actual_gradient.abs().sum()), 0.0)

    def test_deterministic_partitions_are_private_repeatable_and_disjoint(self):
        torch.manual_seed(20260905)
        state_before = torch.random.get_rng_state().clone()
        arguments = {
            "total": 32,
            "sizes": (12, 12, 8),
            "labels": torch.tensor([3, 101, 999]),
            "seed": 43,
            "salt": 11_830_291,
            "device": torch.device("cpu"),
        }

        first = deterministic_class_partition_indices(
            **arguments, split_index=17
        )
        state_after = torch.random.get_rng_state().clone()
        second = deterministic_class_partition_indices(
            **arguments, split_index=17
        )
        changed_step = deterministic_class_partition_indices(
            **arguments, split_index=18
        )

        self.assertTrue(torch.equal(state_before, state_after))
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(
            any(not torch.equal(left, right) for left, right in zip(first, changed_step))
        )
        self.assertEqual(tuple(part.shape for part in first), ((3, 12), (3, 12), (3, 8)))
        for row in range(arguments["labels"].numel()):
            combined = torch.cat([part[row] for part in first])
            self.assertEqual(combined.numel(), 32)
            self.assertEqual(torch.unique(combined).numel(), 32)
            self.assertEqual(int(combined.min()), 0)
            self.assertEqual(int(combined.max()), 31)

    def test_v2_end_to_end_ignores_negatives_and_stops_all_inputs(self):
        torch.manual_seed(314159)
        batch_size, positive_count, negative_count, generated_count = 1, 64, 32, 32
        channels = 3
        positive_values = torch.randn(positive_count, channels, 2, 2)
        negative_values_a = torch.randn(negative_count, channels, 2, 2)
        negative_values_b = 100.0 * torch.randn(negative_count, channels, 2, 2)
        generated_values = torch.randn(generated_count, channels, 2, 2)
        real_a = torch.cat([positive_values, negative_values_a]).requires_grad_()
        real_b = torch.cat([positive_values, negative_values_b]).requires_grad_()
        generated_a = generated_values.clone().requires_grad_()
        generated_b = generated_values.clone().requires_grad_()

        def target_family(raw):
            return {
                name: value.detach().clone().requires_grad_()
                for name, value in derive_ssl_map_features(
                    "layer3",
                    raw,
                    patch_mean_size=(),
                    patch_std_size=(),
                    use_std=False,
                    use_mean=False,
                ).items()
            }

        target_positive = target_family(positive_values)
        target_generated = target_family(generated_values)
        target_negative_a = target_family(negative_values_a)
        target_negative_b = target_family(negative_values_b)
        weight_negative_a = torch.ones(
            batch_size, negative_count, requires_grad=True
        )
        weight_negative_b = torch.linspace(
            -50.0, 50.0, negative_count
        ).reshape(batch_size, negative_count).requires_grad_()

        system = FeatureAdapterSystem(
            {"stage3": channels},
            ["stage3"],
            bottleneck=2,
            projection_dim=0,
            dropout=0.0,
        )
        common = {
            "labels": torch.tensor([71]),
            "batch_size": batch_size,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "generated_count": generated_count,
            "samples_per_class": positive_count,
            "generated_samples_per_class": generated_count,
            "temperature": 0.1,
            "supcon_weight": 1.0,
            "ce_weight": 0.0,
            "reg_weight": 0.0,
            "objective": "raw_drift_snr",
            "target_positive_features": target_positive,
            "target_generated_features": target_generated,
            "gather_distributed": False,
            "collect_diagnostics": True,
            "split_index": 23,
            "drift_options": {
                "R_list": (0.2,),
                "patch_mean_size": (),
                "patch_std_size": (),
                "use_mean": False,
                "use_std": False,
                "snr_epsilon": 1.0e-6,
                "stopgrad_variance": False,
                "snr_weight": 1.0,
                "consistency_weight": 0.25,
                "consistency_epsilon": 1.0e-8,
                "real_pool_count": 32,
                "query_count": 8,
                "bank_count": 12,
                "split_seed": 43,
                "global_scale_stats": False,
                "global_statistics": False,
                "expected_feature_count": 1,
                "expected_raw_map_count": 1,
            },
        }
        loss_a, metrics_a = system(
            {"layer3": real_a},
            generated_stage_features={"layer3": generated_a},
            target_negative_features=target_negative_a,
            weight_neg=weight_negative_a,
            **common,
        )
        loss_b, metrics_b = system(
            {"layer3": real_b},
            generated_stage_features={"layer3": generated_b},
            target_negative_features=target_negative_b,
            weight_neg=weight_negative_b,
            **common,
        )

        torch.testing.assert_close(loss_a, loss_b, rtol=0.0, atol=0.0)
        for name in metrics_a:
            torch.testing.assert_close(
                metrics_a[name], metrics_b[name], rtol=0.0, atol=0.0
            )
        expected_counts = {
            "adapter/drift_feature_count": 1.0,
            "adapter/drift_raw_map_count": 1.0,
            "adapter/drift_temperature_terms": 1.0,
            "adapter/v2_real_source_count": 64.0,
            "adapter/v2_real_pool_count": 32.0,
            "adapter/v2_real_unused_count": 8.0,
            "adapter/v2_query_count": 8.0,
            "adapter/v2_bank_count": 12.0,
            "adapter/v2_different_class_negative_count": 0.0,
            "adapter/v2_split_index": 23.0,
            "adapter/drift_global_statistics": 0.0,
        }
        for name, value in expected_counts.items():
            self.assertEqual(float(metrics_a[name]), value)
        torch.testing.assert_close(
            metrics_a["adapter/drift_Dpq"],
            0.5
            * (
                metrics_a["adapter/drift_Da"]
                + metrics_a["adapter/drift_Db"]
            ),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            metrics_a["adapter/drift_D0"],
            0.5
            * (
                metrics_a["adapter/drift_Drr"]
                + metrics_a["adapter/drift_Dqq"]
            ),
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(torch.isfinite(loss_a))
        self.assertTrue(all(torch.isfinite(value) for value in metrics_a.values()))

        loss_a.backward()
        adapter_gradients = [
            parameter.grad
            for parameter in system.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(adapter_gradients)
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in adapter_gradients)
        )
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in adapter_gradients),
            0.0,
        )
        for raw_input in (real_a, real_b, generated_a, generated_b):
            self.assertIsNone(raw_input.grad)
        for target in (
            target_positive,
            target_generated,
            target_negative_a,
            target_negative_b,
        ):
            self.assertTrue(all(value.grad is None for value in target.values()))
        self.assertIsNone(weight_negative_a.grad)
        self.assertIsNone(weight_negative_b.grad)

    def test_v2_config_is_generator_identical_to_running_dino_cf(self):
        config_root = Path(__file__).resolve().parents[1] / "configs" / "gen"
        cf_path = config_root / (
            "S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-matched-"
            "cf-drift-realreal-infonce-adapter.yaml"
        )
        v2_path = config_root / (
            "S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-matched-"
            "raw-drift-snr-v2-adapter.yaml"
        )
        cf = yaml.safe_load(cf_path.read_text(encoding="utf-8"))
        v2 = yaml.safe_load(v2_path.read_text(encoding="utf-8"))

        for section in ("env", "dataset", "model", "optimizer", "train"):
            self.assertEqual(
                cf[section],
                v2[section],
                f"Version 2 unexpectedly changes fair {section} settings",
            )
        self.assertEqual(v2["logging"]["name"], "S4 / DINO / adapter(raw-drift-snr v2)")

        train = v2["train"]
        feature = v2["feature"]
        self.assertEqual(
            (train["pos_per_sample"], train["neg_per_sample"], train["gen_per_label"]),
            (64, 32, 32),
        )
        self.assertEqual(
            train["layer_temperature_profile"],
            "raw_imagenet_dino_moco_symmetric_3seed_p32_v2",
        )
        self.assertEqual(train["temperature_calibration_role"], "dino")
        dino_tau = train["layer_temperature_profiles"][
            "raw_imagenet_dino_moco_symmetric_3seed_p32_v2"
        ]
        self.assertEqual(dino_tau["stage3"], 1.010352456195719)
        self.assertEqual(dino_tau["stage4"], 1.0110986372034372)
        self.assertFalse(train["historical_gen_replay"])
        self.assertEqual(train["historical_gen_replay_ratio"], 0.0)
        self.assertIsNone(train["historical_gen_replay_ratio_start"])

        expected_v2 = {
            "feature_adapter": True,
            "feature_adapter_objective": "raw_drift_snr",
            "feature_adapter_keys": ["layer3", "layer4"],
            "feature_adapter_bottleneck": 64,
            "feature_adapter_projection_dim": 0,
            "feature_adapter_dropout": 0.0,
            "feature_adapter_lr": 2.0e-5,
            "feature_adapter_weight_decay": 0.0,
            "feature_adapter_adam_b1": 0.9,
            "feature_adapter_adam_b2": 0.999,
            "feature_adapter_start_step": 0,
            "feature_adapter_update_freq": 4,
            "feature_adapter_freeze_generated_epochs": 30.0,
            "feature_adapter_samples_per_class": 64,
            "feature_adapter_generated_samples_per_class": 32,
            "feature_adapter_real_pool_per_class": 32,
            "feature_adapter_query_count": 8,
            "feature_adapter_bank_count": 12,
            "feature_adapter_split_seed": 43,
            "feature_adapter_require_unique_reals": True,
            "feature_adapter_drift_snr_weight": 1.0,
            "feature_adapter_snr_epsilon": 1.0e-6,
            "feature_adapter_snr_stopgrad_variance": False,
            "feature_adapter_consistency_weight": 0.25,
            "feature_adapter_consistency_epsilon": 1.0e-8,
            "feature_adapter_reg_lambda": 0.0,
            "feature_adapter_max_grad_norm": 1.0,
            "feature_adapter_ema_decay": 0.999,
            "feature_adapter_global_statistics": False,
            "feature_adapter_expected_raw_map_count": 6,
            "feature_adapter_expected_feature_count": 42,
        }
        for name, value in expected_v2.items():
            self.assertIn(name, feature)
            self.assertEqual(feature[name], value)
        self.assertEqual(feature["feature_extractor"], "dino_resnet50")
        self.assertEqual(
            feature["feature_checkpoint"], cf["feature"]["feature_checkpoint"]
        )

        missing = object()

        def changed_keys(left, right):
            return {
                key
                for key in set(left) | set(right)
                if left.get(key, missing) != right.get(key, missing)
            }

        expected_feature_changes = {
            "feature_adapter_bank_count",
            "feature_adapter_cf_epsilon",
            "feature_adapter_cf_rho",
            "feature_adapter_cf_weight",
            "feature_adapter_consistency_epsilon",
            "feature_adapter_consistency_weight",
            "feature_adapter_drift_snr_weight",
            "feature_adapter_global_contrastive",
            "feature_adapter_global_statistics",
            "feature_adapter_loss_weight",
            "feature_adapter_lr",
            "feature_adapter_null_weight",
            "feature_adapter_objective",
            "feature_adapter_query_count",
            "feature_adapter_real_pool_per_class",
            "feature_adapter_relation_weight",
            "feature_adapter_snr_epsilon",
            "feature_adapter_snr_stopgrad_variance",
            "feature_adapter_split_seed",
            "feature_adapter_temp",
        }
        self.assertEqual(changed_keys(cf["feature"], feature), expected_feature_changes)


if __name__ == "__main__":
    unittest.main()
