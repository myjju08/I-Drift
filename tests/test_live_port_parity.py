"""CPU checks for the live DINO replay and MAE structure-teacher contracts.

Behavioral tests run in any checkout. Optional source comparisons additionally
run only when IDRIFT_LIVE_WORKSPACE explicitly selects the original
archived experiment directories; no checkpoints, data, or GPUs are required.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import copy
import hashlib
import inspect
import io
import os
from pathlib import Path
import unittest
from unittest import mock

import torch

import train_imagenet_gen as trainer
from tests.test_adversarial_drift import (
    _TinyFrozenDino, _TinyGenerator, _system, _trainer_inputs,
)


REPO = Path(__file__).resolve().parents[1]
LIVE_WORKSPACE = Path(os.environ.get("IDRIFT_LIVE_WORKSPACE", str(REPO.parent)))
LIVE_DINO = (
    LIVE_WORKSPACE / "idrift2_transfer_20260908"
    / "dino_only_replay_feature_drift_20260909/trainer_dino_only_replay.py"
)


class _FeatureOnlyMAE(_TinyFrozenDino):
    """A frozen latent encoder that has no DINO terminal-map interface."""

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(4, 4, 1).requires_grad_(False)
        self.terminal_requests = []

    def get_activations(self, images, return_stage_features=False, **kwargs):
        self.terminal_requests.append(return_stage_features)
        if return_stage_features:
            raise AssertionError("MAE must not be asked for a disabled DINO structure teacher")
        return super().get_activations(images, return_stage_features=False, **kwargs)


def _assert_nested_equal(test, first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            _assert_nested_equal(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            _assert_nested_equal(test, left, right)
    else:
        test.assertEqual(first, second)


def _live_train_step():
    """Compile only the reference function, without importing a live runtime."""
    parsed = ast.parse(LIVE_DINO.read_text())
    node = next(node for node in parsed.body
                if isinstance(node, ast.FunctionDef) and node.name == "train_step")
    namespace = dict(trainer.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(LIVE_DINO), "exec"), namespace)
    return namespace["train_step"]


class LivePortBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_dino_only_replay_preserves_teacher_rho_but_excludes_cnn_history(self):
        for apply_replay in (False, True):
            with self.subTest(adversarial_apply_replay=apply_replay):
                generator, teacher = _TinyGenerator(), _TinyFrozenDino()
                optimizer = torch.optim.SGD(generator.parameters(), lr=0.001)
                system = _system("feature_drift", ema_decay=0, structure_weight=0)
                inputs = _trainer_inputs()
                inputs["cfg"].update(historical_gen_replay=True,
                                     historical_gen_replay_ratio=0.35)
                # Omitting the key deliberately checks backward-compatible default True.
                if not apply_replay:
                    inputs["cfg"]["adversarial_apply_replay"] = False
                reference_drift = trainer.compute_drift_loss_from_features
                target_snapshot = copy.deepcopy(system.target.state_dict())
                target_features = system.target_features
                calls, reads = [], []

                def drift(*args, **kwargs):
                    cnn = set(kwargs["gen_feats"]) == {"stage2", "stage3", "stage4"}
                    calls.append("cnn" if cnn else "dino")
                    if cnn and not apply_replay:
                        self.assertIsNone(kwargs["historical_feats"])
                        self.assertEqual(kwargs["historical_count"], 0)
                        self.assertIsNone(kwargs["weight_gen"])
                        self.assertIsNone(kwargs["weight_history"])
                    else:
                        self.assertEqual(kwargs["historical_count"], 1)
                        self.assertIsNotNone(kwargs["historical_feats"])
                        torch.testing.assert_close(kwargs["weight_gen"],
                                                   torch.full_like(kwargs["weight_gen"], 0.65))
                        # H=1, G=2: rho * G/H = .7 keeps total repulsion mass fixed.
                        torch.testing.assert_close(kwargs["weight_history"],
                                                   torch.full_like(kwargs["weight_history"], 0.7))
                    return reference_drift(*args, **kwargs)

                def features(images, labels=None):
                    _assert_nested_equal(self, target_snapshot, system.target.state_dict())
                    reads.append((len(images), torch.is_grad_enabled(), images.requires_grad))
                    return target_features(images, labels)

                with (
                    mock.patch.object(trainer, "compute_drift_loss_from_features", side_effect=drift),
                    mock.patch.object(system, "target_features", side_effect=features),
                    mock.patch.object(system, "target_logits", side_effect=AssertionError("Direct GAN in feature-only G")),
                ):
                    loss, metrics, _ = trainer.train_step(
                        generator, teacher, optimizer, adversarial_system=system, **inputs)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(generator.images.grad).all())
                self.assertGreater(float(generator.images.grad.norm()), 0)
                self.assertEqual(calls, ["dino", "cnn"])
                expected_reads = [(4, False, False), (2, False, False)]
                if apply_replay:
                    expected_reads.append((2, False, False))
                self.assertEqual(reads, expected_reads + [(4, True, True)])
                self.assertEqual(metrics["adversarial/history_count"], int(apply_replay))
                self.assertEqual(metrics["adversarial/replay_enabled"], float(apply_replay))
                self.assertNotIn("adversarial/raw_gan_loss", metrics)
                _assert_nested_equal(self, system.online.state_dict(), system.target.state_dict())
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_zero_structure_weight_never_requests_dino_maps_for_latent_mae(self):
        for mode in ("feature_drift", "mixed"):
            with self.subTest(mode=mode):
                generator, teacher = _TinyGenerator(), _FeatureOnlyMAE()
                generator.images = torch.nn.Parameter(torch.randn(4, 4, 32, 32) * 0.2)
                optimizer = torch.optim.SGD(generator.parameters(), lr=0.001)
                system = _system(mode, in_channels=4, ema_decay=0, structure_weight=0)
                inputs = _trainer_inputs()
                inputs.update(pos_samples=torch.randn(2, 2, 4, 32, 32),
                              neg_samples=torch.randn(2, 1, 4, 32, 32),
                              historical_samples=None)
                update = system.discriminator_step

                def check_update(*args, **kwargs):
                    self.assertIsNone(kwargs.get("teacher_real_features"))
                    self.assertIsNotNone(generator.images.grad)
                    self.assertGreater(float(generator.images.grad.norm()), 0)
                    return update(*args, **kwargs)

                with mock.patch.object(system, "discriminator_step", side_effect=check_update):
                    loss, metrics, _ = trainer.train_step(
                        generator, teacher, optimizer, adversarial_system=system, **inputs)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(teacher.terminal_requests)
                self.assertFalse(any(teacher.terminal_requests))
                self.assertEqual(metrics["adversarial/structure_loss"], 0)
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    @unittest.skipUnless(os.environ.get("IDRIFT_LIVE_WORKSPACE") and LIVE_DINO.is_file(), "Live DINO source is unavailable")
    def test_live_dino_and_port_match_loss_gradients_and_next_states_exactly(self):
        live_step = _live_train_step()
        for mode, structure, replay in (
            ("raw_gan", 0, True), ("feature_drift", 0, False),
            ("feature_drift", 1, True), ("mixed", 0, False),
        ):
            with self.subTest(mode=mode, structure=structure, replay=replay):
                results = []
                for function in (live_step, trainer.train_step):
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(12345)
                        generator, teacher = _TinyGenerator(), _TinyFrozenDino()
                        optimizer = torch.optim.AdamW(generator.parameters(), lr=0.001)
                        system = _system(mode, ema_decay=0, structure_weight=structure,
                                         r1_gamma=1, r1_interval=16)
                        inputs = _trainer_inputs()
                        inputs["step"] = 0  # Exercise fp32 lazy R1 and auxiliary gradients.
                        inputs["cfg"].update(historical_gen_replay=True,
                                             historical_gen_replay_ratio=0.35,
                                             adversarial_apply_replay=replay)
                        loss, metrics, _ = function(
                            generator, teacher, optimizer, adversarial_system=system, **inputs)
                        results.append((loss, metrics, generator.images.grad,
                                        generator.state_dict(), optimizer.state_dict(), system.state_dict()))
                _assert_nested_equal(self, results[0], results[1])


class LiveSourceParityTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("IDRIFT_LIVE_WORKSPACE") and LIVE_DINO.is_file(), "Live DINO source is unavailable")
    def test_train_step_is_live_dino_with_only_the_mae_teacher_guards(self):
        source = ast.parse(LIVE_DINO.read_text())
        expected = next(node for node in source.body
                        if isinstance(node, ast.FunctionDef) and node.name == "train_step")
        actual = ast.parse(inspect.getsource(trainer.train_step)).body[0]

        class NormalizeDisabledTeacherGuard(ast.NodeTransformer):
            count = 0

            def visit_BoolOp(self, node):
                node = self.generic_visit(node)
                if (isinstance(node.op, ast.And) and len(node.values) == 2
                        and ast.dump(node.values[1]) == ast.dump(ast.parse(
                            "adversarial_system.structure_weight > 0", mode="eval").body)
                        and ast.dump(node.values[0]) == ast.dump(ast.parse(
                            'adversarial_mode in {"feature_drift", "mixed"}', mode="eval").body)):
                    self.count += 1
                    return node.values[0]
                return node

        normalizer = NormalizeDisabledTeacherGuard()
        normalized = normalizer.visit(actual)
        self.assertEqual(normalizer.count, 2)
        self.assertEqual(ast.dump(normalized), ast.dump(expected))

    @unittest.skipUnless(os.environ.get("IDRIFT_LIVE_WORKSPACE") and (LIVE_WORKSPACE / "I-Drift/models/adversarial_drift.py").is_file(),
                         "Original shared source is unavailable")
    def test_shared_training_math_matches_original_files(self):
        # Direct comparison supplements numerical tests: these helpers are also
        # used by the compiled live function above, so their identity is explicit.
        for relative in (
            "models/adversarial_drift.py", "models/feature_gan.py", "models/feature_adapter.py",
            "models/imagenet_generator.py", "models/mae_resnet.py", "models/ssl_resnet.py",
            "drifting_core/imagenet_loss.py", "memory_bank.py", "train/train_data.py", "utils.py",
        ):
            with self.subTest(path=relative):
                original = LIVE_WORKSPACE / "I-Drift" / relative
                self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                                 hashlib.sha256((REPO / relative).read_bytes()).hexdigest())


class RuntimeRoutingTest(unittest.TestCase):
    def test_dino_cli_accepts_older_raw_and_mixed_configs_without_preset_guards(self):
        from experiments.dino import train as entry

        suffixes = {
            "raw_gan": "raw-conditional-gan.yaml",
            "mixed": "mixed-gan-feature-drift-d32-opt4.yaml",
        }
        for mode, suffix in suffixes.items():
            path = REPO / "configs/gen" / (
                "S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-matched-" + suffix)
            with self.subTest(mode=mode), mock.patch.object(entry, "train") as run:
                entry.main(["--config", str(path), "--train", "--workdir", "runs/test-custom",
                            "--imagenet-path", "/tmp/test-raw", "--feature-checkpoint", "/tmp/test-dino.pth"])
                run.assert_called_once()
                cfg = run.call_args.args[0]
                self.assertEqual(cfg["adversarial_mode"], mode)
                self.assertIsNone(run.call_args.kwargs["preset"])
                self.assertEqual(run.call_args.kwargs["io_backend"], "standard")
                self.assertEqual(cfg["imagenet_path"], "/tmp/test-raw")
                self.assertEqual(cfg["_raw"]["env"]["imagenet_path"], "/tmp/test-raw")
                self.assertEqual(cfg["feature_checkpoint"], "/tmp/test-dino.pth")
                self.assertTrue(Path(cfg["temperature_calibration_artifact"]).is_file())

    def test_dino_default_cli_only_dispatches_cpu_preflight(self):
        from experiments.dino import train as entry

        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(torch, "set_num_threads"),
            mock.patch.object(entry, "train") as run,
            mock.patch.object(entry, "preflight", return_value={"passed": True}) as check,
            redirect_stdout(io.StringIO()),
        ):
            entry.main([])
            run.assert_not_called()
            check.assert_called_once()
            self.assertEqual(check.call_args.kwargs["preset"], "dino_only_replay_feature")
            self.assertEqual(check.call_args.kwargs["io_backend"], "packed")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")


if __name__ == "__main__":
    unittest.main()
