import unittest
from pathlib import Path

import torch
import yaml

from models.feature_adapter import (
    FeatureAdapterSystem,
    derive_ssl_map_features,
    drift_direction_consistency,
    drift_field_snr_statistic,
)


class RawDriftAdapterObjectiveTest(unittest.TestCase):
    def test_raw_drift_config_diff_is_adapter_only(self):
        config_root = Path(__file__).resolve().parents[1] / "configs" / "gen"
        infonce_path = config_root / (
            "S4_pixel-p32_direct_moco-v2-r50-stage34only-r64g32-mae-matched-"
            "genreal-mp-infonce-adapter.yaml"
        )
        raw_drift_path = config_root / (
            "S4_pixel-p32_direct_moco-v2-r50-stage34only-r64g32-mae-matched-"
            "raw-drift-snr-consistency-adapter.yaml"
        )
        infonce = yaml.safe_load(infonce_path.read_text(encoding="utf-8"))
        raw_drift = yaml.safe_load(raw_drift_path.read_text(encoding="utf-8"))

        # These sections define raw ImageNet, S4, generator optimization, and
        # the complete drift/tau/bank sampling protocol.  Requiring structural
        # equality guards B/G/P/N, MoCo provenance, global/norm_x, stage3/4,
        # and all calibrated-temperature inputs at once.
        for section in ("env", "dataset", "model", "optimizer", "train"):
            self.assertEqual(
                infonce[section],
                raw_drift[section],
                f"fair adapter configs unexpectedly differ in {section}",
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
            changed_paths(infonce["logging"], raw_drift["logging"]),
            {"name"},
        )
        allowed_feature_changes = {
            "feature_adapter_objective",
            "feature_adapter_lr",
            "feature_adapter_update_freq",
            "feature_adapter_freeze_generated_epochs",
            "feature_adapter_temp",
            "feature_adapter_loss_weight",
            "feature_adapter_reg_lambda",
            "feature_adapter_drift_align_lambda",
            "feature_adapter_ema_decay",
            "feature_adapter_global_contrastive",
            "feature_adapter_drift_snr_weight",
            "feature_adapter_snr_epsilon",
            "feature_adapter_snr_stopgrad_variance",
            "feature_adapter_consistency_weight",
            "feature_adapter_global_statistics",
            "feature_adapter_expected_raw_map_count",
            "feature_adapter_expected_feature_count",
        }
        self.assertEqual(
            changed_paths(infonce["feature"], raw_drift["feature"]),
            allowed_feature_changes,
        )
        self.assertEqual(set(infonce), set(raw_drift))

    def test_feature_adapter_raw_drift_objective_end_to_end(self):
        torch.manual_seed(314)
        batch_size, positive_count, negative_count, generated_count = 2, 4, 2, 4
        channels = 4
        real_raw = torch.randn(
            batch_size * (positive_count + negative_count),
            channels,
            4,
            4,
            requires_grad=True,
        )
        generated_raw = torch.randn(
            batch_size * generated_count,
            channels,
            4,
            4,
            requires_grad=True,
        )

        def target_family(raw):
            # Treat every derived EMA feature as an explicit stopped input so
            # the test can assert that no target branch receives gradients.
            return {
                name: value.detach().clone().requires_grad_()
                for name, value in derive_ssl_map_features(
                    "layer3",
                    raw,
                    patch_mean_size=(2,),
                    patch_std_size=(2,),
                    use_std=True,
                    use_mean=True,
                ).items()
            }

        target_positive = target_family(real_raw[: batch_size * positive_count])
        target_negative = target_family(real_raw[batch_size * positive_count :])
        target_generated = target_family(generated_raw)
        system = FeatureAdapterSystem(
            {"stage3": channels},
            ["stage3"],
            bottleneck=2,
            projection_dim=0,
        )

        loss, metrics = system(
            {"layer3": real_raw},
            torch.tensor([3, 9]),
            batch_size=batch_size,
            positive_count=positive_count,
            negative_count=negative_count,
            generated_count=generated_count,
            samples_per_class=positive_count,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.0,
            reg_weight=0.0,
            objective="raw_drift_snr_consistency",
            generated_stage_features={"layer3": generated_raw},
            weight_neg=torch.ones(batch_size, negative_count),
            target_positive_features=target_positive,
            target_negative_features=target_negative,
            target_generated_features=target_generated,
            gather_distributed=False,
            collect_diagnostics=True,
            drift_options={
                "R_list": (0.2,),
                "patch_mean_size": (2,),
                "patch_std_size": (2,),
                "use_mean": True,
                "use_std": True,
                "snr_epsilon": 1.0e-6,
                "stopgrad_variance": True,
                "snr_weight": 1.0,
                "consistency_weight": 0.1,
                "global_scale_stats": False,
                "global_statistics": False,
                "expected_feature_count": 5,
                "expected_raw_map_count": 1,
            },
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(metrics["adapter/drift_feature_count"]), 5.0)
        self.assertEqual(float(metrics["adapter/drift_raw_map_count"]), 1.0)
        self.assertEqual(float(metrics["adapter/drift_temperature_terms"]), 5.0)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))

        loss.backward()
        adapter_gradients = [
            parameter.grad
            for parameter in system.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(adapter_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in adapter_gradients))
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in adapter_gradients),
            0.0,
        )
        self.assertIsNone(real_raw.grad)
        self.assertIsNone(generated_raw.grad)
        for target_features in (
            target_positive,
            target_negative,
            target_generated,
        ):
            self.assertTrue(all(value.grad is None for value in target_features.values()))

    def test_snr_statistic_matches_formula_and_backpropagates_all_fields(self):
        batch_size = 3
        # [B*T, Q, D] with T=2, Q=2, D=2.  Deliberately use non-uniform
        # class energies so the unbiased null variances and their gradients are
        # both non-zero.
        signal = torch.linspace(0.2, 2.5, 24).reshape(6, 2, 2).requires_grad_()
        real_null = (
            torch.linspace(-1.7, 1.1, 24).reshape(6, 2, 2).requires_grad_()
        )
        generated_null = (
            torch.linspace(0.4, 3.2, 24).reshape(6, 2, 2).requires_grad_()
        )
        epsilon = 2.5e-5

        actual = drift_field_snr_statistic(
            signal,
            real_null,
            generated_null,
            batch_size=batch_size,
            epsilon=epsilon,
            global_statistics=False,
            stopgrad_variance=False,
        )

        def class_energy(field):
            return field.reshape(batch_size, 2, 2, 2).square().mean((1, 2, 3))

        energy_pq = class_energy(signal)
        energy_pp = class_energy(real_null)
        energy_qq = class_energy(generated_null)
        expected = {
            "Dpq": energy_pq.mean(),
            "Dpp": energy_pp.mean(),
            "Dqq": energy_qq.mean(),
        }
        expected["D0"] = 0.5 * (expected["Dpp"] + expected["Dqq"])
        expected["Var0"] = 0.5 * (
            energy_pp.var(correction=1) + energy_qq.var(correction=1)
        )
        expected["J"] = (expected["Dpq"] - expected["D0"]) / (
            expected["Var0"] + epsilon
        ).sqrt()

        self.assertEqual(set(actual), {"Dpq", "Dpp", "Dqq", "D0", "Var0", "J"})
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, rtol=0.0, atol=0.0)

        (-actual["J"]).backward()
        for field in (signal, real_null, generated_null):
            self.assertIsNotNone(field.grad)
            self.assertTrue(torch.isfinite(field.grad).all())
            self.assertGreater(float(field.grad.abs().sum()), 0.0)

    def test_snr_stopgrad_variance_preserves_value_but_changes_null_gradient(self):
        signal_values = torch.tensor(
            [
                [[[1.0, 0.0]]],
                [[[2.0, 0.0]]],
                [[[4.0, 0.0]]],
            ]
        ).reshape(3, 1, 2)
        real_values = torch.tensor(
            [
                [[[0.2, 0.1]]],
                [[[0.8, 0.3]]],
                [[[2.0, 0.5]]],
            ]
        ).reshape(3, 1, 2)
        generated_values = torch.tensor(
            [
                [[[0.4, 0.7]]],
                [[[1.1, 0.2]]],
                [[[2.5, 1.0]]],
            ]
        ).reshape(3, 1, 2)

        gradients = []
        statistics = []
        for stopgrad in (False, True):
            signal = signal_values.clone().requires_grad_()
            real_null = real_values.clone().requires_grad_()
            generated_null = generated_values.clone().requires_grad_()
            result = drift_field_snr_statistic(
                signal,
                real_null,
                generated_null,
                batch_size=3,
                epsilon=1.0e-6,
                global_statistics=False,
                stopgrad_variance=stopgrad,
            )
            result["J"].backward()
            statistics.append(result)
            gradients.append(
                (signal.grad.clone(), real_null.grad.clone(), generated_null.grad.clone())
            )

        for name in statistics[0]:
            torch.testing.assert_close(
                statistics[0][name], statistics[1][name], rtol=0.0, atol=0.0
            )
        # Dpq does not enter Var0, so stopping only the variance leaves the
        # signal-field gradient unchanged while changing both null gradients.
        torch.testing.assert_close(gradients[0][0], gradients[1][0], rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(gradients[0][1], gradients[1][1]))
        self.assertFalse(torch.equal(gradients[0][2], gradients[1][2]))

    def test_direction_consistency_aligned_opposite_and_stops_target(self):
        aligned = torch.tensor([[[1.0, 2.0], [-3.0, 4.0]]])
        torch.testing.assert_close(
            drift_direction_consistency(aligned, aligned.clone()),
            torch.zeros(()),
            rtol=0.0,
            atol=1.0e-7,
        )
        torch.testing.assert_close(
            drift_direction_consistency(aligned, -aligned),
            torch.tensor(2.0),
            rtol=0.0,
            atol=1.0e-7,
        )

        online = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
        target = torch.tensor([[[1.0, 1.0], [1.0, -1.0]]], requires_grad=True)
        consistency = drift_direction_consistency(online, target)
        consistency.backward()
        self.assertIsNotNone(online.grad)
        self.assertGreater(float(online.grad.abs().sum()), 0.0)
        self.assertIsNone(target.grad)

    def test_ssl_stage34_raw_map_families_expand_to_exactly_42_keys(self):
        torch.manual_seed(123)
        # Production ResNet-50 every_k_block=2 exports four stage-3 maps and
        # two stage-4 maps.  Each map contributes raw, spatial mean/std, and
        # patch-2/patch-4 mean/std: 6 maps * 7 keys = 42.
        raw_maps = {
            **{
                name: torch.randn(2, 5, 8, 8, requires_grad=True)
                for name in ("layer3", "layer3_blk2", "layer3_blk4", "layer3_blk6")
            },
            **{
                name: torch.randn(2, 7, 4, 4, requires_grad=True)
                for name in ("layer4", "layer4_blk2")
            },
        }
        derived = {}
        for name, feature in raw_maps.items():
            family = derive_ssl_map_features(
                name,
                feature,
                patch_mean_size=(2, 4),
                patch_std_size=(2, 4),
                use_std=True,
                use_mean=True,
            )
            expected_suffixes = {
                "",
                "_mean",
                "_std",
                "_mean_2",
                "_std_2",
                "_mean_4",
                "_std_4",
            }
            self.assertEqual(set(family), {name + suffix for suffix in expected_suffixes})
            self.assertTrue(set(derived).isdisjoint(family))
            derived.update(family)

        self.assertEqual(len(derived), 42)
        self.assertEqual(
            sum(key == "layer3" or key.startswith("layer3_") for key in derived),
            28,
        )
        self.assertEqual(
            sum(key == "layer4" or key.startswith("layer4_") for key in derived),
            14,
        )
        self.assertNotIn("global", derived)
        self.assertNotIn("norm_x", derived)

        self.assertEqual(derived["layer3"].shape, (2, 64, 5))
        self.assertEqual(derived["layer3_mean_2"].shape, (2, 16, 5))
        self.assertEqual(derived["layer3_std_4"].shape, (2, 4, 5))
        self.assertEqual(derived["layer4"].shape, (2, 16, 7))
        self.assertEqual(derived["layer4_mean_2"].shape, (2, 4, 7))
        self.assertEqual(derived["layer4_std_4"].shape, (2, 1, 7))

        sum(value.float().sum() for value in derived.values()).backward()
        for feature in raw_maps.values():
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(float(feature.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
