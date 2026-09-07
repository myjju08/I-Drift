"""Check GAN integration with the trainer's existing replay and adapter APIs."""

import copy
import tempfile
import types
import unittest
from unittest import mock

import torch

import train_imagenet_gen as trainer
from tests.test_adversarial_drift import _TinyFrozenDino, _TinyGenerator, _system, _trainer_inputs


class _StrictFrozenDino(_TinyFrozenDino):
    # The real SSL encoder has no MAE-only return_all_stage_features keyword.
    def get_activations(self, images, *, return_stage_features=False, stage_adapters=None):
        return super().get_activations(images, return_stage_features=return_stage_features)


class GanReplayIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mixed_dino_does_not_receive_mae_adapter_only_keywords(self):
        generator = _TinyGenerator()
        optimizer = torch.optim.SGD(generator.parameters(), lr=0.001)
        loss, metrics, _ = trainer.train_step(
            generator, _StrictFrozenDino(), optimizer,
            adversarial_system=_system("mixed"), **_trainer_inputs(),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["adversarial/d_updates"], 1.0)

    def test_full_mixed_loop_wires_uint8_banks_checkpoint_and_resume(self):
        generator = _TinyGenerator()
        generator.model = types.SimpleNamespace(
            blocks=[types.SimpleNamespace(attn=types.SimpleNamespace(use_sdpa=False))],
            use_remat=False,
        )
        teacher = _StrictFrozenDino()
        images = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
        labels = torch.tensor([0, 0, 1, 1])
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        cfg = {
            **_trainer_inputs()["cfg"],
            "feature_extractor": "dino_resnet50", "use_latent": False,
            "use_cache": False, "in_channels": 3, "num_classes": 2,
            "batch_size": 2, "loader_batch_size": 4, "num_workers": 0,
            "positive_bank_size": 4, "negative_bank_size": 4,
            "memory_bank_storage_mode": "pixel_uint8", "raw_train_uint8": True,
            "positive_memory_bank_backend": "zstd_delta",
            "positive_memory_bank_codec_workers": 2,
            "pos_per_sample": 2, "neg_per_sample": 1, "push_per_step": 4,
            "push_at_resume": 1, "total_steps": 2, "train_max_step_exclusive": 1,
            "eval_enabled": False, "eval_at_start": False, "save_per_step": 1,
            "keep_every": 100, "checkpoint_latest_hardlink": True,
            "use_wandb": False, "log_every_k": 1, "seed": 43,
            "lr": 0.001, "warmup_steps": 1, "adversarial_mode": "mixed",
            "adversarial_base_channels": 2, "adversarial_r1_gamma": 0.0,
            "adversarial_samples_per_class": 2,
        }
        loader_kwargs = []

        def create_loader(**kwargs):
            loader_kwargs.append(kwargs)
            return loader, lambda batch: {"images": batch[0], "labels": batch[1]}, lambda value: value

        def run(config, workdir):
            with (
                mock.patch.object(trainer, "build_ditgen_from_config", side_effect=lambda *_: copy.deepcopy(generator)),
                mock.patch.object(trainer, "load_feature_extractor", side_effect=lambda *_: copy.deepcopy(teacher)),
                mock.patch.object(trainer, "create_imagenet_split", side_effect=create_loader),
                mock.patch.object(trainer, "tqdm", side_effect=lambda values, **_: values),
            ):
                trainer.train_gen(config, workdir, rank=0, world_size=1, device=torch.device("cpu"))

        with tempfile.TemporaryDirectory(prefix="gan-replay-integration-") as workdir:
            run(cfg, workdir)
            checkpoint = torch.load(workdir + "/checkpoints/ckpt_latest.pt", weights_only=False)
            self.assertEqual(checkpoint["step"], 1)
            self.assertEqual(checkpoint["adversarial_system"]["updates"], 1)
            run({**cfg, "train_max_step_exclusive": 2}, workdir)
            resumed = torch.load(workdir + "/checkpoints/ckpt_latest.pt", weights_only=False)
            self.assertEqual(resumed["step"], 2)
            self.assertEqual(resumed["adversarial_system"]["updates"], 2)
            self.assertTrue(all(kwargs["return_uint8"] for kwargs in loader_kwargs))


if __name__ == "__main__":
    unittest.main()
