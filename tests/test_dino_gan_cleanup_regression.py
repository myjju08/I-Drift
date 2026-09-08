"""DINO-only boundaries and disabled legacy-checkpoint compatibility."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import train_imagenet_gen as trainer
from utils import EMA


class DinoOnlyCompatibilityTests(unittest.TestCase):
    def test_historical_disabled_fields_are_preserved(self):
        config = {
            "feature_extractor": "dino_resnet50", "feature_adapter": False,
            "feature_gan": False, "use_convnext": False, "use_latent": False,
            "use_cache": False, "cache_path": "", "in_channels": 3,
            "out_channels": 3,
        }
        before = copy.deepcopy(config)
        trainer._validate_dino_only_config(config)
        self.assertEqual(config, before)

    def test_retired_feature_modes_fail_before_loading_weights(self):
        for name in ("mae", "moco_v2_resnet50", "dino_latent_bridge", "convnext"):
            with self.subTest(name=name), mock.patch.object(trainer, "build_ssl_resnet_from_config") as build:
                with self.assertRaises(ValueError):
                    trainer.load_feature_extractor(
                        {"feature_extractor": name, "feature_checkpoint": "unused.pth"},
                        torch.device("cpu"),
                    )
                build.assert_not_called()

    def test_enabled_legacy_flags_fail_before_training_initialization(self):
        for flag in ("feature_adapter", "feature_gan", "use_convnext", "use_latent", "use_cache"):
            with self.subTest(flag=flag), mock.patch.object(trainer, "_validate_raw_temperature_calibration") as calibration, mock.patch.object(trainer, "build_ditgen_from_config") as build:
                config = {"feature_extractor": "dino_resnet50", flag: True,
                          "require_raw_temperature_calibration": True}
                with self.assertRaises(ValueError):
                    trainer.train_gen(config, "unused", 0, 1, torch.device("cpu"))
                calibration.assert_not_called()
                build.assert_not_called()

    def test_trainable_feature_checkpoint_cannot_silently_resume_as_frozen_dino(self):
        for key in ("feature_adapter", "feature_adapter_target", "feature_discriminator"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "checkpoints" / "ckpt_latest.pt"
                checkpoint.parent.mkdir()
                torch.save({"step": 9, key: {}}, checkpoint)
                with self.assertRaisesRegex(ValueError, "removed trainable feature system"):
                    trainer.load_checkpoint(directory, None, None, None, torch.device("cpu"))

    def test_legacy_disabled_checkpoint_restores_generator_ema_and_optimizer(self):
        torch.manual_seed(19)
        generator = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(generator.parameters(), lr=0.001)
        ema = EMA(generator, decay=0.99)
        generator(torch.ones(3, 2)).square().mean().backward()
        optimizer.step()
        ema.update(generator)
        old_payload = {
            "step": 9, "model": generator.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {"feature_adapter": False, "feature_gan": False, "use_convnext": False},
        }
        restored = torch.nn.Linear(2, 1)
        restored_ema = EMA(restored, decay=0.99)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.3)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoints" / "ckpt_latest.pt"
            checkpoint.parent.mkdir()
            torch.save(old_payload, checkpoint)
            step = trainer.load_checkpoint(directory, restored, restored_ema, restored_optimizer, torch.device("cpu"))
        self.assertEqual(step, 9)
        for key, value in generator.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
        for key, value in ema.state_dict().items():
            torch.testing.assert_close(restored_ema.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 0.001)
        for old, new in zip(optimizer.state.values(), restored_optimizer.state.values()):
            for key, value in old.items():
                torch.testing.assert_close(new[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
