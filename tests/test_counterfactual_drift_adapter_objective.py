import unittest
from pathlib import Path

import torch
import yaml

from models.feature_adapter import FeatureAdapterSystem


class CounterfactualDriftAdapterObjectiveTest(unittest.TestCase):
    def test_objective_is_finite_detaches_encoder_inputs_and_updates_adapter(self):
        torch.manual_seed(20260903)
        batch_size, positive_count, negative_count, generated_count = 2, 9, 2, 6
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
            generated_samples_per_class=generated_count,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.0,
            reg_weight=0.0,
            objective="dino_cf_drift_realreal_infonce",
            generated_stage_features={"layer3": generated_raw},
            gather_distributed=False,
            collect_diagnostics=True,
            split_index=4,
            drift_options={
                "R_list": (0.2,),
                "patch_mean_size": (2,),
                "patch_std_size": (2,),
                "use_mean": True,
                "use_std": True,
                "cf_epsilon": 1.0e-6,
                "cf_rho": 0.7,
                "cf_weight": 1.0,
                "null_weight": 0.5,
                "relation_weight": 0.25,
                "global_scale_stats": False,
                "global_statistics": False,
                "expected_feature_count": 5,
                "expected_raw_map_count": 1,
            },
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        self.assertEqual(float(metrics["adapter/drift_feature_count"]), 5.0)
        self.assertEqual(float(metrics["adapter/drift_raw_map_count"]), 1.0)
        self.assertEqual(float(metrics["adapter/drift_temperature_terms"]), 5.0)
        self.assertEqual(float(metrics["adapter/cf_query_count"]), 2.0)
        self.assertEqual(float(metrics["adapter/cf_generated_support_count"]), 2.0)
        self.assertEqual(float(metrics["adapter/cf_real_support_count"]), 4.0)
        self.assertEqual(float(metrics["adapter/cf_null_query_count"]), 3.0)
        self.assertEqual(float(metrics["adapter/cf_null_support_count"]), 3.0)
        expected = metrics["adapter/infonce_loss"] + metrics["adapter/cf_objective"]
        torch.testing.assert_close(loss.detach(), expected)

        loss.backward()
        self.assertIsNone(real_raw.grad)
        self.assertIsNone(generated_raw.grad)
        gradients = [
            parameter.grad
            for parameter in system.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in gradients), 0.0
        )

    def test_configs_preserve_generator_spec_and_history_is_exact_ablation(self):
        config_root = Path(__file__).resolve().parents[1] / "configs" / "gen"
        dino_adapter = yaml.safe_load(
            (
                config_root
                / (
                    "S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-"
                    "matched-realreal-mp-infonce-adapter.yaml"
                )
            ).read_text(encoding="utf-8")
        )
        cf_adapter = yaml.safe_load(
            (
                config_root
                / (
                    "S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-"
                    "matched-cf-drift-realreal-infonce-adapter.yaml"
                )
            ).read_text(encoding="utf-8")
        )
        dino_baseline = yaml.safe_load(
            (
                config_root
                / "S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-matched.yaml"
            ).read_text(encoding="utf-8")
        )
        history = yaml.safe_load(
            (
                config_root
                / (
                    "S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-"
                    "matched-historical-replay-e10-rho050.yaml"
                )
            ).read_text(encoding="utf-8")
        )

        for section in ("env", "dataset", "model", "optimizer", "train"):
            self.assertEqual(dino_adapter[section], cf_adapter[section])
        self.assertEqual(
            dino_adapter["feature"]["feature_extractor"],
            cf_adapter["feature"]["feature_extractor"],
        )
        for key in (
            "pos_per_sample",
            "neg_per_sample",
            "gen_per_label",
            "activation_kwargs",
            "R_list",
            "layer_temperature_profile",
        ):
            self.assertEqual(dino_adapter["train"][key], cf_adapter["train"][key])

        def changed_paths(left, right, prefix=()):
            missing = object()
            if isinstance(left, dict) and isinstance(right, dict):
                result = set()
                for key in set(left) | set(right):
                    result.update(
                        changed_paths(
                            left.get(key, missing),
                            right.get(key, missing),
                            (*prefix, str(key)),
                        )
                    )
                return result
            return {".".join(prefix)} if left != right else set()

        self.assertEqual(
            changed_paths(dino_baseline, history),
            {
                "logging.name",
                # The archived resumed run starts W&B logging at its recovery
                # step; this changes logging only, not the replay objective.
                "logging.wandb_log_min_step",
                "train.historical_gen_replay",
                "train.historical_gen_replay_ratio",
            },
        )
        self.assertNotIn("wandb_log_min_step", dino_baseline["logging"])
        self.assertEqual(history["logging"]["wandb_log_min_step"], 73761)


if __name__ == "__main__":
    unittest.main()
