import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from train_imagenet_gen import _validate_raw_temperature_calibration
from scripts.calibrate_feature_encoder_temperatures import _run_finalize_symmetric


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RawTemperatureGuardTest(unittest.TestCase):
    def test_capture_mode_requires_one_step_offline_invariants(self):
        cfg = {
            "temperature_calibration_status": "calibration_capture",
            "use_wandb": False,
            "eval_at_start": False,
            "total_generated_epochs": 0.0,
            "train_max_step_exclusive": 1,
        }
        _validate_raw_temperature_calibration(cfg)
        cfg["use_wandb"] = True
        with self.assertRaisesRegex(RuntimeError, "Unsafe temperature"):
            _validate_raw_temperature_calibration(cfg)

    def test_ready_mode_pins_data_checkpoint_inputs_and_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "raw_imagenet_manifest.json"
            manifest.write_text('{"complete":true}\n', encoding="utf-8")
            manifest_sha = _sha256(manifest)
            checkpoint = root / "dino.pth"
            checkpoint.write_bytes(b"frozen-dino-test-weights")

            q3, q4 = 0.81, 1.21
            profiles = {
                "dino": {
                    "default": 1.0,
                    "stage3": q3 ** -0.5,
                    "stage4": q4 ** -0.5,
                },
                "moco": {
                    "default": 1.0,
                    "stage3": q3 ** 0.5,
                    "stage4": q4 ** 0.5,
                },
            }
            tensor_hashes = {
                key: hashlib.sha256(key.encode()).hexdigest()
                for key in (
                    "labels",
                    "positive_samples",
                    "negative_samples",
                    "generated_samples",
                    "cfg_scales",
                )
            }
            feature_layout = {
                **{
                    f"stage3_feature_{index}": {
                        "stage": "stage3",
                        "tokens": 1,
                        "dimension": 1024,
                    }
                    for index in range(28)
                },
                **{
                    f"stage4_feature_{index}": {
                        "stage": "stage4",
                        "tokens": 1,
                        "dimension": 2048,
                    }
                    for index in range(14)
                },
            }
            artifact = {
                "schema_version": 3,
                "calibration_kind": "symmetric_raw_imagenet_dino_moco",
                "centering": "geometric_mean_one",
                "required_seeds": [43, 44, 45],
                "min_distance": 0.02,
                "huber_tuning": 1.5,
                "trim_fraction": 0.1,
                "capture_protocol": {
                    "generator_geometry": {
                        "input_size": 256,
                        "in_channels": 3,
                        "out_channels": 3,
                        "patch_size": 32,
                        "image_tokens": 64,
                        "class_tokens": 16,
                    },
                    "batch": {"B": 4, "G": 32, "P": 64, "N": 32},
                    "R_list": [0.2, 0.05, 0.02],
                    "max_token_rows_per_feature": 64,
                    "feature_layout": feature_layout,
                },
                "fixed_input_feature_multipliers": {
                    "global": 1.0,
                    "norm_x": 1.0,
                },
                "dataset_provenance": {"manifest_sha256": manifest_sha},
                "encoder_provenance": {
                    "dino": {
                        "feature_extractor": "dino_resnet50",
                        "checkpoint_bytes": checkpoint.stat().st_size,
                        "checkpoint_sha256": _sha256(checkpoint),
                    },
                    "moco": {},
                },
                "symmetric_profiles": profiles,
                "candidates": {
                    "moco": {
                        "stages": {
                            "stage3": {"selected_multiplier": q3},
                            "stage4": {"selected_multiplier": q4},
                        }
                    }
                },
                "capture_validation": {
                    role: {
                        str(seed): {"tensor_sha256": tensor_hashes}
                        for seed in (43, 44, 45)
                    }
                    for role in ("dino", "moco")
                },
                "reciprocity_validation": {
                    "verified": True,
                    "max_log_reciprocity_error": 1e-10,
                    "stages": {
                        "stage3": {"product": 1.0, "abs_log_product": 0.0},
                        "stage4": {"product": 1.0, "abs_log_product": 0.0},
                    },
                },
            }
            artifact_path = root / "tau.json"
            artifact_path.write_text(
                json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8"
            )
            cfg = {
                "temperature_calibration_status": "ready",
                "temperature_calibration_role": "dino",
                "temperature_calibration_artifact": str(artifact_path),
                "temperature_calibration_artifact_sha256": _sha256(artifact_path),
                "temperature_calibration_raw_manifest_sha256": manifest_sha,
                "layer_temperature_profile": (
                    "raw_imagenet_dino_moco_symmetric_3seed_p32_v2"
                ),
                "layer_temperature_profiles": {
                    "raw_imagenet_dino_moco_symmetric_3seed_p32_v2": profiles["dino"]
                },
                "imagenet_path": str(root),
                "use_wandb": True,
                "project": "Feature encoder - S4 model",
                "require_complete_raw_imagenet": True,
                "use_latent": False,
                "use_cache": False,
                "resolution": 256,
                "memory_bank_storage_mode": "pixel_uint8",
                "hidden_size": 384,
                "depth": 12,
                "num_heads": 6,
                "input_size": 256,
                "in_channels": 3,
                "out_channels": 3,
                "patch_size": 32,
                "R_list": [0.2, 0.05, 0.02],
                "batch_size": 4,
                "gen_per_label": 32,
                "pos_per_sample": 64,
                "neg_per_sample": 32,
                "feature_extractor": "dino_resnet50",
                "feature_checkpoint": str(checkpoint),
                "feature_loss_profile": "no_stage12_norm_x2",
                "feature_loss_group_normalize": True,
                "prune_skipped_feature_tensors": True,
                "feature_include_norm_x": True,
                "feature_loss_profiles": {
                    "no_stage12_norm_x2": {
                        "default": 1.0,
                        "norm_x": 2.0,
                        "stage1": 0.0,
                        "stage2": 0.0,
                    }
                },
                "activation_kwargs": {
                    "active_stages": ["stage3", "stage4"],
                    "with_global": True,
                    "with_norm_x": True,
                    "every_k_block": 2,
                    "exclude_terminal_block": False,
                },
            }
            _validate_raw_temperature_calibration(cfg)

            artifact["reciprocity_validation"]["verified"] = False
            artifact_path.write_text(
                json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8"
            )
            cfg["temperature_calibration_artifact_sha256"] = _sha256(artifact_path)
            with self.assertRaisesRegex(RuntimeError, "reciprocity"):
                _validate_raw_temperature_calibration(cfg)

            artifact["reciprocity_validation"]["verified"] = True
            artifact_path.write_text(
                json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8"
            )
            cfg["temperature_calibration_artifact_sha256"] = _sha256(artifact_path)

            cfg["layer_temperature_profiles"][
                "raw_imagenet_dino_moco_symmetric_3seed_p32_v2"
            ]["stage3"] += 0.01
            with self.assertRaisesRegex(RuntimeError, "tau multiplier"):
                _validate_raw_temperature_calibration(cfg)

    def test_symmetric_finalize_requires_reciprocal_forward_reverse_fits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provenance = {"manifest_sha256": "a" * 64}
            dino_captures = {str(seed): f"/capture/dino_s{seed}.json" for seed in (43, 44, 45)}
            moco_captures = {str(seed): f"/capture/moco_s{seed}.json" for seed in (43, 44, 45)}
            forward = {
                "schema_version": 2,
                "calibration_kind": "symmetric_raw_imagenet_dino_moco",
                "dataset_provenance": provenance,
                "min_distance": 0.02,
                "huber_tuning": 1.5,
                "trim_fraction": 0.1,
                "groups_considered": ["pos", "neg", "gen"],
                "quantiles": ["q10", "q50", "q90"],
                "references": dino_captures,
                "symmetric_profiles": {"dino": {}, "moco": {}},
                "candidates": {
                    "moco": {
                        "captures": moco_captures,
                        "stages": {
                            "stage3": {"selected_multiplier": 0.8},
                            "stage4": {"selected_multiplier": 1.25},
                        }
                    }
                },
            }
            reverse = {
                "schema_version": 2,
                "dataset_provenance": provenance,
                "min_distance": 0.02,
                "huber_tuning": 1.5,
                "trim_fraction": 0.1,
                "groups_considered": ["pos", "neg", "gen"],
                "quantiles": ["q10", "q50", "q90"],
                "references": moco_captures,
                "candidates": {
                    "dino": {
                        "captures": dino_captures,
                        "stages": {
                            "stage3": {"selected_multiplier": 1.25},
                            "stage4": {"selected_multiplier": 0.8},
                        }
                    }
                },
            }
            forward_path = root / "forward.json"
            reverse_path = root / "reverse.json"
            output_path = root / "final.json"
            forward_path.write_text(json.dumps(forward), encoding="utf-8")
            reverse_path.write_text(json.dumps(reverse), encoding="utf-8")
            args = argparse.Namespace(
                forward=str(forward_path),
                reverse=str(reverse_path),
                output=str(output_path),
                max_log_reciprocity_error=1e-10,
            )
            _run_finalize_symmetric(args)
            final = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(final["schema_version"], 3)
            self.assertTrue(final["reciprocity_validation"]["verified"])

            reverse["candidates"]["dino"]["stages"]["stage3"][
                "selected_multiplier"
            ] = 1.2
            reverse_path.write_text(json.dumps(reverse), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not reciprocal"):
                _run_finalize_symmetric(args)


if __name__ == "__main__":
    unittest.main()
