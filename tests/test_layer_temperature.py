import unittest
from unittest.mock import patch

import torch

from train_imagenet_gen import (
    _feature_temperature_multiplier,
    _resolve_layer_temperature_multipliers,
    compute_drift_loss_from_features,
)


def metric_config():
    return {"layer_temperature_profiles": {
        "uniform": {"stage1": 1.0, "stage2": 1.0, "stage3": 1.0, "stage4": 1.0},
        "shallow_hot": {"stage1": 1.5, "stage2": 1.0, "stage3": 1.0, "stage4": 1.0},
        "deep_sharp": {"stage1": 1.0, "stage2": 1.0, "stage3": 1.0, "stage4": 0.75},
        "depth_profile": {"stage1": 1.5, "stage2": 1.25, "stage3": 1.0, "stage4": 0.75},
    }}


class LayerTemperatureTest(unittest.TestCase):
    def test_encoder_stage_mapping_applies_to_all_derived_statistics(self):
        multipliers = {
            "default": 1.0,
            "stage1": 1.5,
            "stage2": 1.25,
            "stage3": 1.0,
            "stage4": 0.75,
        }
        self.assertEqual(
            _feature_temperature_multiplier("layer1_blk2_std_4", multipliers),
            1.5,
        )
        self.assertEqual(
            _feature_temperature_multiplier("layer2_mean", multipliers), 1.25
        )
        self.assertEqual(
            _feature_temperature_multiplier("layer3", multipliers), 1.0
        )
        self.assertEqual(
            _feature_temperature_multiplier("layer4_std", multipliers), 0.75
        )
        self.assertEqual(
            _feature_temperature_multiplier("global", multipliers), 1.0
        )
        self.assertEqual(
            _feature_temperature_multiplier("norm_x", multipliers), 1.0
        )

    def test_all_four_metric_profiles_resolve(self):
        cfg = metric_config()
        expected = {
            "uniform": (1.0, 1.0, 1.0, 1.0),
            "shallow_hot": (1.5, 1.0, 1.0, 1.0),
            "deep_sharp": (1.0, 1.0, 1.0, 0.75),
            "depth_profile": (1.5, 1.25, 1.0, 0.75),
        }
        for profile_name, values in expected.items():
            with self.subTest(profile=profile_name):
                cfg["layer_temperature_profile"] = profile_name
                actual = _resolve_layer_temperature_multipliers(cfg)
                self.assertEqual(
                    tuple(actual[f"stage{i}"] for i in range(1, 5)), values
                )

    def test_reverse_loss_receives_feature_specific_temperature_lists(self):
        gen_feats = {
            "layer1_mean": torch.randn(2, 1, 4),
            "layer4_std": torch.randn(2, 1, 4),
            "global": torch.randn(2, 1, 4),
        }
        pos_feats = {
            name: torch.randn(3, 1, 4) for name in gen_feats
        }
        observed = []

        def fake_reverse_loss(**kwargs):
            observed.append(tuple(kwargs["R_list"]))
            return torch.zeros(kwargs["gen"].shape[0]), {}

        with patch(
            "train_imagenet_gen.drift_loss_imagenet",
            side_effect=fake_reverse_loss,
        ):
            compute_drift_loss_from_features(
                gen_feats=gen_feats,
                pos_feats=pos_feats,
                neg_feats=None,
                B=1,
                G=2,
                P=3,
                N=0,
                weight_neg=None,
                R_list=(0.2, 0.05, 0.02),
                drift_matching="rev-drift",
                compute_raw_winner_stats_flag=False,
                feature_temperature_multipliers={
                    "stage1": 1.5,
                    "stage4": 0.75,
                    "default": 1.0,
                },
            )

        self.assertEqual(observed[0], (0.3, 0.075, 0.03))
        self.assertEqual(observed[1], (0.15, 0.0375, 0.015))
        self.assertEqual(observed[2], (0.2, 0.05, 0.02))

    def test_nonpositive_multiplier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite and > 0"):
            _resolve_layer_temperature_multipliers(
                {"layer_temperature_multipliers": {"stage1": 0.0}}
            )


if __name__ == "__main__":
    unittest.main()
