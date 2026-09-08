import unittest

from train_imagenet_gen import (
    _crosses_generated_epoch_interval,
    _feature_loss_group,
    _feature_loss_weights_for_groups,
    _resolve_drift_top_k_groups,
    _resolve_feature_loss_group_weights,
    _steps_for_generated_epochs,
    _top_k_for_feature,
)


def metric_config():
    profiles = {"all": {"default": 1.0}, "global_x8": {"default": 1.0, "global": 8.0}}
    for group in ("global", "norm_x", "stage1", "stage2", "stage3", "stage4"):
        profiles["no_" + group] = {"default": 1.0, group: 0.0}
    profiles.update({
        "no_stage1_norm_x2": {"norm_x": 2.0, "stage1": 0.0},
        "no_stage2_norm_x2": {"norm_x": 2.0, "stage2": 0.0},
        "no_stage12": {"stage1": 0.0, "stage2": 0.0},
        "no_stage12_norm_x2": {"norm_x": 2.0, "stage1": 0.0, "stage2": 0.0},
    })
    return {"feature_loss_profiles": profiles}


class FeatureLossGroupsTest(unittest.TestCase):
    def test_raw_and_derived_features_map_to_expected_groups(self):
        self.assertEqual(_feature_loss_group("global"), "global")
        self.assertEqual(_feature_loss_group("norm_x"), "norm_x")
        self.assertEqual(_feature_loss_group("conv1_std_4"), "stage1")
        self.assertEqual(_feature_loss_group("layer1_blk2_mean"), "stage1")
        self.assertEqual(_feature_loss_group("layer2"), "stage2")
        self.assertEqual(_feature_loss_group("layer3_std"), "stage3")
        self.assertEqual(_feature_loss_group("layer4_mean_2"), "stage4")

    def test_leave_one_out_normalization_preserves_coefficient_mass(self):
        names = (
            "global",
            "norm_x",
            "layer1",
            "layer1_mean",
            "layer2",
        )
        weights = _feature_loss_weights_for_groups(
            names,
            {"default": 1.0, "stage1": 0.0},
            normalize=True,
        )
        self.assertEqual(weights["layer1"], 0.0)
        self.assertEqual(weights["layer1_mean"], 0.0)
        self.assertAlmostEqual(sum(weights.values()), len(names))
        self.assertAlmostEqual(weights["global"], 5.0 / 3.0)

    def test_all_leave_one_out_profiles_resolve(self):
        cfg = metric_config()
        for profile in (
            "all",
            "no_global",
            "no_norm_x",
            "no_stage1",
            "no_stage2",
            "no_stage3",
            "no_stage4",
        ):
            with self.subTest(profile=profile):
                cfg["feature_loss_profile"] = profile
                weights = _resolve_feature_loss_group_weights(cfg)
                if profile == "all":
                    self.assertEqual(weights["default"], 1.0)
                else:
                    self.assertEqual(weights[profile.removeprefix("no_")], 0.0)

    def test_negative_group_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite and >= 0"):
            _resolve_feature_loss_group_weights(
                {"feature_loss_group_weights": {"stage2": -0.1}}
            )

    def test_top_k_can_be_restricted_to_selected_encoder_stages(self):
        groups = _resolve_drift_top_k_groups(
            {"drift_top_k_groups": "stage2,stage3,stage4"}
        )
        self.assertEqual(groups, ("stage2", "stage3", "stage4"))
        self.assertEqual(_top_k_for_feature("norm_x", 8, 20, groups), (0, 0))
        self.assertEqual(_top_k_for_feature("global", 8, 20, groups), (0, 0))
        self.assertEqual(_top_k_for_feature("layer1_mean", 8, 20, groups), (0, 0))
        self.assertEqual(_top_k_for_feature("layer2_mean", 8, 20, groups), (8, 20))
        self.assertEqual(_top_k_for_feature("layer4_std", 8, 20, groups), (8, 20))

    def test_top_k_group_default_preserves_all_feature_behavior(self):
        groups = _resolve_drift_top_k_groups({})
        self.assertIsNone(groups)
        self.assertEqual(_top_k_for_feature("norm_x", 4, 10, groups), (4, 10))

    def test_unknown_top_k_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown drift top-k groups"):
            _resolve_drift_top_k_groups({"drift_top_k_groups": "stage5"})

    def test_global_x8_profile_preserves_eight_to_one_relative_weight(self):
        cfg = metric_config()
        cfg["feature_loss_profile"] = "global_x8"
        group_weights = _resolve_feature_loss_group_weights(cfg)
        names = ("global", "norm_x", "layer1", "layer2")
        weights = _feature_loss_weights_for_groups(
            names, group_weights, normalize=True
        )
        self.assertAlmostEqual(weights["global"] / weights["norm_x"], 8.0)
        self.assertAlmostEqual(sum(weights.values()), len(names))

    def test_efficiency_profiles_resolve_requested_groups(self):
        cfg = metric_config()
        expected = {
            "no_stage1_norm_x2": {"norm_x": 2.0, "stage1": 0.0},
            "no_stage2_norm_x2": {"norm_x": 2.0, "stage2": 0.0},
            "no_stage12": {"stage1": 0.0, "stage2": 0.0},
            "no_stage12_norm_x2": {
                "norm_x": 2.0,
                "stage1": 0.0,
                "stage2": 0.0,
            },
        }
        for profile, requested in expected.items():
            with self.subTest(profile=profile):
                cfg["feature_loss_profile"] = profile
                group_weights = _resolve_feature_loss_group_weights(cfg)
                for group, value in requested.items():
                    self.assertEqual(group_weights[group], value)

                # A synthetic 86-objective layout: raw/global=2, stages=21/21/28/14.
                names = ["global", "norm_x"]
                for stage, count in ((1, 21), (2, 21), (3, 28), (4, 14)):
                    names.extend(f"layer{stage}_objective_{i}" for i in range(count))
                weights = _feature_loss_weights_for_groups(
                    tuple(names), group_weights, normalize=True
                )
                self.assertAlmostEqual(sum(weights.values()), 86.0)
                if "norm_x" in requested:
                    self.assertAlmostEqual(
                        weights["norm_x"] / weights["global"], 2.0
                    )

    def test_generated_epoch_step_and_checkpoint_boundaries(self):
        dataset_size = 1_281_167
        generated_per_step = 2 * 10 * 64
        self.assertEqual(
            _steps_for_generated_epochs(
                dataset_size=dataset_size,
                generated_per_step=generated_per_step,
                epochs=100,
            ),
            100_092,
        )
        boundaries = [
            step
            for step in range(1, 100_093)
            if _crosses_generated_epoch_interval(
                completed_steps=step,
                generated_per_step=generated_per_step,
                dataset_size=dataset_size,
                interval_epochs=10,
            )
        ]
        self.assertEqual(
            boundaries,
            [
                10_010,
                20_019,
                30_028,
                40_037,
                50_046,
                60_055,
                70_064,
                80_073,
                90_083,
                100_092,
            ],
        )


if __name__ == "__main__":
    unittest.main()
