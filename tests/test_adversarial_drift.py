"""Behavioral checks for the isolated adversarial ImageNet experiments."""

import copy
import datetime
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models.adversarial_drift import (
    AdversarialDriftSystem,
    ConditionalMultiScaleDiscriminator,
    logistic_discriminator_loss,
    logistic_generator_loss,
    within_class_structure_loss,
)


class _TinyGenerator(torch.nn.Module):
    use_bf16 = False

    def __init__(self):
        super().__init__()
        self.images = torch.nn.Parameter(torch.randn(4, 3, 32, 32) * 0.2)

    def forward(self, labels, cfg_scale=None, train=True):
        return {"samples": self.images[: labels.numel()] + 0.01 * labels[:, None, None, None]}


class _TinyFrozenDino(torch.nn.Module):
    use_bf16 = False

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 4, 1).requires_grad_(False)

    def get_activations(self, images, return_stage_features=False, **_kwargs):
        projected = self.projection(images)
        maps = {
            "layer3": torch.nn.functional.adaptive_avg_pool2d(projected, 4),
            "layer4": torch.nn.functional.adaptive_avg_pool2d(projected, 2),
        }
        features = {name: value.flatten(2).transpose(1, 2) for name, value in maps.items()}
        features["norm_x"] = torch.nn.functional.adaptive_avg_pool2d(images, 2).flatten(2).transpose(1, 2)
        return (features, maps) if return_stage_features else features


def _trainer_inputs():
    rng = torch.Generator().manual_seed(76)
    config = {
        "gen_per_label": 2,
        "cfg_min": 1.0,
        "cfg_max": 2.0,
        "R_list": [0.2],
        "drift_matching": "rev-drift",
        "global_scale_stats": False,
        "global_fnorm_stats": False,
        "compute_wpos_stats": False,
        "activation_kwargs": {},
        "adversarial_samples_per_class": 2,
        "adversarial_diagnostics_every": 1000,
    }
    return dict(
        labels=torch.tensor([0, 1]),
        pos_samples=torch.randn(2, 2, 3, 32, 32, generator=rng),
        neg_samples=torch.randn(2, 1, 3, 32, 32, generator=rng),
        historical_samples=torch.randn(2, 1, 3, 32, 32, generator=rng),
        device=torch.device("cpu"),
        step=1,
        cfg=config,
    )


def _system(mode="raw_gan", **overrides):
    options = dict(
        device=torch.device("cpu"),
        mode=mode,
        in_channels=3,
        num_classes=4,
        base_channels=4,
        lr=1.0e-3,
        ema_decay=0.75,
        r1_gamma=0.0,
        r1_interval=2,
        structure_weight=1.0,
        d_chunk_size=2,
        g_chunk_size=2,
        seed=43,
    )
    options.update(overrides)
    return AdversarialDriftSystem(**options)


def _batch(seed=19, requires_grad=False):
    rng = torch.Generator().manual_seed(seed)
    real = torch.randn(4, 3, 32, 32, generator=rng).requires_grad_(requires_grad)
    fake = torch.randn(4, 3, 32, 32, generator=rng).requires_grad_(requires_grad)
    labels = torch.tensor([0, 0, 1, 1])
    return real, fake, labels, labels.clone()


def _structure_features():
    rng = torch.Generator().manual_seed(81)
    teacher = {
        name: torch.randn(4, channels, generator=rng).requires_grad_()
        for name, channels in (("layer3", 4), ("layer4", 6))
    }
    student = {
        name.replace("layer", "stage"): value.detach()[:, None].repeat(1, 16, 1)
        for name, value in teacher.items()
    }
    student["stage2"] = student["stage3"].clone()
    return student, teacher, torch.tensor([0, 0, 1, 1])


def _ddp_lazy_r1_worker(rank, world_size, port):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        real, fake, real_labels, fake_labels = _batch(seed=100 + rank)
        teacher = {name: real.mean(dim=(2, 3)) for name in ("layer3", "layer4")}
        mixed_reference = None
        for mode, fused in (("raw_gan", False), ("feature_drift", False), ("mixed", False), ("mixed", True)):
            system = _system(mode, base_channels=2, r1_gamma=0.1, fuse_grad_reduce=fused)
            for step in (0, 1, 2):
                metrics = system.discriminator_step(
                    real, fake, real_labels, fake_labels, step=step,
                    teacher_real_features=teacher if mode in {"feature_drift", "mixed"} else None,
                )
                assert metrics["adversarial/r1_applied"] == float(step % 2 == 0)
                assert metrics["adversarial/d_updated"] == 1.0
                for module in (system.online, system.target):
                    actual = torch.cat([p.detach().flatten() for p in module.parameters()])
                    reference = actual.clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
                    assert torch.isfinite(actual).all()
            if mode == "mixed":
                actual = torch.cat([p.detach().flatten() for p in system.online.parameters()])
                if fused:
                    torch.testing.assert_close(actual, mixed_reference, rtol=1.0e-6, atol=1.0e-7)
                else:
                    mixed_reference = actual.clone()
            previous = torch.cat([p.detach().flatten() for p in system.online.parameters()])
            bad_fake = fake.clone()
            if rank == 1:
                bad_fake[0, 0, 0, 0] = float("nan")
            try:
                system.discriminator_step(
                    real, bad_fake, real_labels, fake_labels, step=3,
                    teacher_real_features=teacher if mode in {"feature_drift", "mixed"} else None,
                )
            except FloatingPointError:
                pass
            else:
                raise AssertionError("Every rank must reject the non-finite update")
            current = torch.cat([p.detach().flatten() for p in system.online.parameters()])
            torch.testing.assert_close(previous, current, rtol=0.0, atol=0.0)
    finally:
        dist.destroy_process_group()


class AdversarialDriftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def assert_state_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_logistic_gradients_raise_real_and_lower_fake_scores(self):
        real = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        fake = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        logistic_discriminator_loss(real, fake).backward()
        self.assertTrue((real.grad < 0).all())
        self.assertTrue((fake.grad > 0).all())
        fake.grad = None
        logistic_generator_loss(fake).backward()
        self.assertTrue((fake.grad < 0).all())

    def test_logistic_extreme_logits_remain_finite(self):
        real = torch.tensor([-1000.0, 1000.0], requires_grad=True)
        fake = torch.tensor([1000.0, -1000.0], requires_grad=True)
        loss = logistic_discriminator_loss(real, fake) + logistic_generator_loss(fake)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(real.grad).all())
        self.assertTrue(torch.isfinite(fake.grad).all())

    def test_discriminator_features_have_dimension_independent_token_scale(self):
        model = ConditionalMultiScaleDiscriminator(
            in_channels=3, num_classes=4, base_channels=4
        )
        images, _, labels, _ = _batch()
        logits, features = model(images, labels, return_features=True)
        self.assertEqual(logits.shape, (4,))
        self.assertEqual(set(features), {"stage2", "stage3", "stage4"})
        for feature in features.values():
            self.assertEqual(feature.shape[:2], (4, 16))
            torch.testing.assert_close(
                feature.square().mean(dim=-1),
                torch.ones_like(feature[..., 0]),
                rtol=1.0e-5,
                atol=1.0e-6,
            )

    def test_frozen_target_still_backpropagates_to_generated_images(self):
        for mode in ("raw_gan", "feature_drift"):
            with self.subTest(mode=mode):
                system = _system(mode)
                images, _, labels, _ = _batch(requires_grad=True)
                if mode == "raw_gan":
                    loss = logistic_generator_loss(system.target_logits(images, labels))
                else:
                    features = system.target_features(images, labels)
                    loss = sum(
                        (value * torch.linspace(-1.0, 1.0, value.numel()).reshape_as(value)).sum()
                        for value in features.values()
                    )
                loss.backward()
                self.assertIsNotNone(images.grad)
                self.assertTrue(torch.isfinite(images.grad).all())
                self.assertGreater(float(images.grad.abs().sum()), 0.0)
                self.assertTrue(all(not p.requires_grad for p in system.target.parameters()))
                self.assertTrue(all(p.grad is None for p in system.target.parameters()))
                self.assertTrue(all(p.grad is None for p in system.online.parameters()))

    def test_target_reads_leave_snapshot_unchanged(self):
        system = _system("feature_drift")
        real, fake, labels, _ = _batch()
        before = copy.deepcopy(system.target.state_dict())
        expected = system.target_features(real, labels)
        system.target_features(fake, labels)
        actual = system.target_features(real, labels)
        self.assert_state_equal(expected, actual)
        self.assert_state_equal(before, system.target.state_dict())

    def test_mixed_shared_forward_matches_separate_outputs_and_image_gradients(self):
        system = _system("mixed")
        images, _, labels, _ = _batch(requires_grad=True)
        expected_logits = system.target_logits(images, labels)
        expected_features = system.target_features(images)
        feature_loss = lambda features: sum(
            (value * torch.linspace(-1.0, 1.0, value.numel()).reshape_as(value)).mean()
            for value in features.values()
        )
        expected_loss = 0.1 * logistic_generator_loss(expected_logits) + feature_loss(expected_features)
        expected_grad = torch.autograd.grad(expected_loss, images)[0]
        with mock.patch.object(system.target, "_maps", wraps=system.target._maps) as encode:
            logits, features = system.target_logits_and_features(images, labels)
        self.assertEqual(encode.call_count, 2)  # Four images in two chunks, once each.
        torch.testing.assert_close(logits, expected_logits, rtol=0.0, atol=0.0)
        self.assert_state_equal(features, expected_features)
        gan_loss = 0.1 * logistic_generator_loss(logits)
        drift_loss = feature_loss(features)
        for term in (gan_loss, drift_loss):
            gradient = torch.autograd.grad(term, images, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        actual_grad = torch.autograd.grad(gan_loss + drift_loss, images)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1.0e-4, atol=1.0e-10)
        self.assertTrue(all(parameter.grad is None for parameter in system.target.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in system.online.parameters()))

    def test_structure_preserves_real_within_class_angles_and_detaches_teacher(self):
        student, teacher, labels = _structure_features()
        identity = within_class_structure_loss(student, teacher, labels)
        self.assertLess(float(identity), 1.0e-12)
        changed = {}
        for name, value in student.items():
            value = value.clone()
            value[1] = 0.8 * value[0] + 0.2 * value[1]
            changed[name] = value.requires_grad_()
        loss = within_class_structure_loss(changed, teacher, labels)
        self.assertGreater(float(loss.detach()), 1.0e-5)
        loss.backward()
        self.assertTrue(all(value.grad is None for value in teacher.values()))
        self.assertTrue(all(value.grad is not None for value in changed.values()))
        self.assertGreater(sum(float(value.grad.abs().sum()) for value in changed.values()), 0.0)

    def test_structure_ignores_cross_class_angles_and_absolute_scale(self):
        student, teacher, labels = _structure_features()
        rotated = {}
        for name, value in student.items():
            value = value.clone()
            value[:2] = value[:2].roll(shifts=1, dims=-1)
            value[:2, :, 0] *= -1.0
            rotated[name] = value * 7.0
        loss = within_class_structure_loss(rotated, teacher, labels)
        self.assertLess(float(loss), 1.0e-12)

    def test_structure_ignores_singleton_classes(self):
        student, teacher, _ = _structure_features()
        unrelated = {name: torch.flip(value, dims=(0,)) for name, value in student.items()}
        loss = within_class_structure_loss(unrelated, teacher, torch.arange(4))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)

    def test_structure_detects_spatial_distortion_even_when_gap_is_unchanged(self):
        rng = torch.Generator().manual_seed(39)
        teacher = {
            name: torch.randn(4, 16, channels, generator=rng)
            for name, channels in (("layer3", 4), ("layer4", 6))
        }
        student = {
            stage: teacher["layer4" if stage == "stage4" else "layer3"].clone()
            for stage in ("stage2", "stage3", "stage4")
        }
        labels = torch.tensor([0, 0, 1, 1])
        self.assertLess(float(within_class_structure_loss(student, teacher, labels)), 1.0e-12)
        # Shuffle token locations for one image only; its GAP remains exact.
        changed = {stage: value.clone() for stage, value in student.items()}
        for stage in changed:
            changed[stage][0] = changed[stage][0].roll(shifts=1, dims=0)
            torch.testing.assert_close(changed[stage].mean(dim=1), student[stage].mean(dim=1))
        self.assertGreater(float(within_class_structure_loss(changed, teacher, labels)), 1.0e-3)

    def test_structure_penalizes_distortion_in_every_student_stage(self):
        student, teacher, labels = _structure_features()
        for stage in ("stage2", "stage3", "stage4"):
            changed = {name: value.clone() for name, value in student.items()}
            changed[stage][1] = changed[stage][0]
            with self.subTest(stage=stage):
                self.assertGreater(float(within_class_structure_loss(changed, teacher, labels)), 1.0e-4)

    def test_discriminator_update_detaches_inputs_and_ema_interpolates(self):
        system = _system(r1_gamma=0.1)
        real, fake, real_labels, fake_labels = _batch(requires_grad=True)
        before_online = copy.deepcopy(system.online.state_dict())
        before_target = copy.deepcopy(system.target.state_dict())
        system.discriminator_step(real, fake, real_labels, fake_labels, step=0)
        self.assertIsNone(real.grad)
        self.assertIsNone(fake.grad)
        self.assertTrue(
            any(not torch.equal(before_online[k], value) for k, value in system.online.state_dict().items())
        )
        for name, value in system.target.named_parameters():
            expected = 0.75 * before_target[name] + 0.25 * system.online.state_dict()[name]
            torch.testing.assert_close(value, expected, rtol=1.0e-6, atol=1.0e-7)

    def test_checkpoint_restores_optimizer_and_exact_next_update(self):
        original = _system(r1_gamma=0.1)
        batch = _batch()
        original.discriminator_step(*batch, step=0)
        checkpoint = copy.deepcopy(original.state_dict())
        restored = _system(r1_gamma=0.1)
        restored.load_state_dict(checkpoint)
        self.assert_state_equal(original.state_dict(), restored.state_dict())
        self.assert_state_equal(original.optimizer.state_dict(), restored.optimizer.state_dict())
        self.assertGreater(len(restored.optimizer.state), 0)
        original.discriminator_step(*batch, step=1)
        restored.discriminator_step(*batch, step=1)
        self.assert_state_equal(original.state_dict(), restored.state_dict())
        self.assert_state_equal(original.optimizer.state_dict(), restored.optimizer.state_dict())

    def test_checkpoint_rejects_different_mode(self):
        checkpoint = _system("raw_gan").state_dict()
        with self.assertRaises((ValueError, RuntimeError)):
            _system("feature_drift").load_state_dict(checkpoint)

    def test_mixed_checkpoint_restores_both_targets_optimizer_and_next_update(self):
        original = _system("mixed", r1_gamma=0.1, fuse_grad_reduce=True)
        _, teacher, _ = _structure_features()
        batch = _batch()
        original.discriminator_step(*batch, step=0, teacher_real_features=teacher)
        restored = _system("mixed", r1_gamma=0.1, fuse_grad_reduce=True)
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        self.assert_state_equal(original.state_dict(), restored.state_dict())
        original.discriminator_step(*batch, step=1, teacher_real_features=teacher)
        restored.discriminator_step(*batch, step=1, teacher_real_features=teacher)
        self.assert_state_equal(original.state_dict(), restored.state_dict())
        self.assertEqual(restored.updates, 2)
        self.assertTrue(all(not parameter.requires_grad for parameter in restored.target.parameters()))
        with self.assertRaisesRegex(ValueError, "structure preservation"):
            restored.discriminator_step(*batch, step=2)

    def test_discriminator_chunking_preserves_update_scale_including_r1(self):
        full = _system(r1_gamma=0.1)
        chunked = _system(r1_gamma=0.1)
        chunked.load_state_dict(copy.deepcopy(full.state_dict()))
        batch = _batch()
        full.discriminator_step(*batch, step=0, chunk_size=4)
        chunked.discriminator_step(*batch, step=0, chunk_size=2)
        for left, right in zip(full.online.parameters(), chunked.online.parameters()):
            torch.testing.assert_close(left, right, rtol=1.0e-4, atol=1.0e-6)

    def test_builder_preserves_rng_and_disabled_branch_is_absent(self):
        from train_imagenet_gen import build_adversarial_system

        before = torch.random.get_rng_state().clone()
        self.assertIsNone(build_adversarial_system({}, torch.device("cpu")))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        config = {
            "adversarial_mode": "raw_gan",
            "use_latent": False,
            "in_channels": 3,
            "feature_extractor": "dino_resnet50",
            "adversarial_base_channels": 4,
            "num_classes": 4,
        }
        build_adversarial_system(config, torch.device("cpu"))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_feature_drift_trainer_uses_one_snapshot_for_all_particle_types(self):
        from train_imagenet_gen import train_step

        generator = _TinyGenerator()
        teacher = _TinyFrozenDino()
        optimizer = torch.optim.SGD(generator.parameters(), lr=0.001)
        system = _system("feature_drift")
        snapshot = copy.deepcopy(system.target.state_dict())
        feature_reads = []
        original_features = system.target_features
        original_update = system.discriminator_step

        def read_features(images, labels=None):
            self.assert_state_equal(snapshot, system.target.state_dict())
            feature_reads.append((images.shape[0], torch.is_grad_enabled(), images.requires_grad))
            return original_features(images, labels)

        def update_after_generator(*args, **kwargs):
            self.assertIsNotNone(generator.images.grad)
            self.assertGreater(float(generator.images.grad.abs().sum()), 0.0)
            self.assertFalse(args[0].requires_grad)
            self.assertFalse(args[1].requires_grad)
            self.assertTrue(all(not value.requires_grad for value in kwargs["teacher_real_features"].values()))
            self.assertEqual(len(feature_reads), 4)
            return original_update(*args, **kwargs)

        with (
            mock.patch.object(system, "target_features", side_effect=read_features),
            mock.patch.object(system, "target_logits", side_effect=AssertionError("feature drift must not use direct GAN logits")),
            mock.patch.object(system, "discriminator_step", side_effect=update_after_generator),
        ):
            loss, metrics, _ = train_step(
                generator, teacher, optimizer, adversarial_system=system, **_trainer_inputs()
            )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(feature_reads, [(4, False, False), (2, False, False), (2, False, False), (4, True, True)])
        self.assertIn("adversarial/feature_drift_loss", metrics)
        self.assertNotIn("adversarial/raw_gan_loss", metrics)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_mixed_builder_validates_both_weights_and_resume_configuration(self):
        from train_imagenet_gen import build_adversarial_system

        config = {
            "adversarial_mode": "mixed", "use_latent": False,
            "in_channels": 3, "feature_extractor": "dino_resnet50",
            "adversarial_base_channels": 4, "num_classes": 4,
            "adversarial_loss_weight": 0.1, "adversarial_drift_weight": 1.0,
        }
        system = build_adversarial_system(config, torch.device("cpu"))
        self.assertEqual(system.ema_decay, 0.99)
        self.assertEqual(system.structure_weight, 1.0)
        for name in ("adversarial_loss_weight", "adversarial_drift_weight"):
            for invalid in (0.0, -1.0, float("nan")):
                with self.subTest(name=name, invalid=invalid), self.assertRaisesRegex(ValueError, name):
                    build_adversarial_system({**config, name: invalid}, torch.device("cpu"))
            changed = build_adversarial_system({**config, name: 0.3}, torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                changed.load_state_dict(system.state_dict())

    def test_mixed_trainer_adds_both_gradients_and_updates_discriminator_once(self):
        import train_imagenet_gen as trainer

        generator = _TinyGenerator()
        teacher = _TinyFrozenDino()
        optimizer = torch.optim.SGD(generator.parameters(), lr=0.001)
        system = _system("mixed")
        inputs = _trainer_inputs()
        inputs["cfg"].update(
            adversarial_loss_weight=0.37, adversarial_drift_weight=1.7,
            adversarial_diagnostics_every=1, max_grad_norm=1.0e20,
        )
        snapshot = copy.deepcopy(system.target.state_dict())
        component_gradients = []
        particle_reads = []
        original_drift = trainer.compute_drift_loss_from_features
        original_shared = system.target_logits_and_features
        original_features = system.target_features
        original_update = system.discriminator_step

        def drift_with_gradient(**kwargs):
            result = original_drift(**kwargs)
            component_gradients.append(torch.autograd.grad(result[0], generator.images, retain_graph=True)[0])
            return result

        def shared_with_gradient(images, labels):
            self.assert_state_equal(snapshot, system.target.state_dict())
            result = original_shared(images, labels)
            raw_gan = logistic_generator_loss(result[0])
            component_gradients.append(torch.autograd.grad(raw_gan, generator.images, retain_graph=True)[0])
            return result

        def read_particles(images, labels=None):
            self.assert_state_equal(snapshot, system.target.state_dict())
            particle_reads.append((images.shape[0], torch.is_grad_enabled(), images.requires_grad))
            return original_features(images, labels)

        def update_after_generator(*args, **kwargs):
            self.assertIsNotNone(generator.images.grad)
            self.assertFalse(args[0].requires_grad)
            self.assertFalse(args[1].requires_grad)
            self.assertTrue(all(not value.requires_grad for value in kwargs["teacher_real_features"].values()))
            self.assertEqual(len(component_gradients), 3)
            expected = component_gradients[0] + 0.37 * component_gradients[1] + 1.7 * component_gradients[2]
            torch.testing.assert_close(generator.images.grad, expected, rtol=2.0e-4, atol=1.0e-7)
            return original_update(*args, **kwargs)

        with (
            mock.patch.object(trainer, "compute_drift_loss_from_features", side_effect=drift_with_gradient),
            mock.patch.object(system, "target_logits_and_features", side_effect=shared_with_gradient) as shared,
            mock.patch.object(system, "target_logits", side_effect=AssertionError("mixed must share target encoding")),
            mock.patch.object(system, "target_features", side_effect=read_particles),
            mock.patch.object(system, "discriminator_step", side_effect=update_after_generator) as update,
        ):
            loss, metrics, _ = trainer.train_step(
                generator, teacher, optimizer, adversarial_system=system, **inputs
            )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(shared.call_count, 1)
        self.assertEqual(update.call_count, 1)
        self.assertEqual(system.updates, 1)
        self.assertEqual(particle_reads, [(4, False, False), (2, False, False), (2, False, False)])
        self.assertAlmostEqual(
            metrics["adversarial/g_extra_loss"],
            0.37 * metrics["adversarial/raw_gan_loss"] + 1.7 * metrics["adversarial/feature_drift_loss"],
            places=5,
        )
        for name in ("raw_gan", "feature_drift"):
            self.assertGreater(metrics[f"adversarial/{name}_pixel_grad_norm"], 0.0)
        self.assertGreater(metrics["adversarial/structure_loss"], 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_disabled_trainer_branch_preserves_loss_and_generator_update(self):
        from train_imagenet_gen import train_step

        generator = _TinyGenerator()
        restored = copy.deepcopy(generator)
        teacher = _TinyFrozenDino()
        original_optimizer = torch.optim.SGD(generator.parameters(), lr=0.001)
        restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.001)
        before = torch.random.get_rng_state().clone()
        first = train_step(generator, teacher, original_optimizer, **_trainer_inputs())
        torch.random.set_rng_state(before)
        second = train_step(restored, teacher, restored_optimizer, adversarial_system=None, **_trainer_inputs())
        torch.testing.assert_close(first[0], second[0], rtol=0.0, atol=0.0)
        self.assert_state_equal(generator.state_dict(), restored.state_dict())

    def test_trainer_checkpoint_persists_adversarial_branch(self):
        from train_imagenet_gen import EMA, load_checkpoint, save_checkpoint

        generator = _TinyGenerator()
        optimizer = torch.optim.SGD(generator.parameters(), lr=0.001)
        ema = EMA(generator)
        system = _system()
        system.discriminator_step(*_batch(), step=0)
        restored = _system()
        with tempfile.TemporaryDirectory(prefix="idrift-adversarial-test-") as workdir:
            save_checkpoint(workdir, 7, generator, ema, optimizer, {}, adversarial_system=system)
            step = load_checkpoint(workdir, generator, ema, optimizer, torch.device("cpu"), adversarial_system=restored)
            self.assertEqual(step, 7)
            self.assert_state_equal(system.state_dict(), restored.state_dict())
            with self.assertRaisesRegex(ValueError, "adversarial state"):
                load_checkpoint(workdir, generator, ema, optimizer, torch.device("cpu"))

    def test_hardlink_checkpoint_republication_preserves_retained_snapshots(self):
        from train_imagenet_gen import EMA, save_checkpoint

        generator = _TinyGenerator()
        optimizer = torch.optim.SGD(generator.parameters(), lr=0.001)
        ema = EMA(generator)
        system = _system()
        config = {"checkpoint_latest_hardlink": True}
        with tempfile.TemporaryDirectory(prefix="idrift-adversarial-hardlink-") as workdir:
            checkpoint_dir = Path(workdir) / "checkpoints"
            save_checkpoint(workdir, 1, generator, ema, optimizer, config, keep_last=4, adversarial_system=system)
            first = checkpoint_dir / "ckpt_step_0000001.pt"
            latest = checkpoint_dir / "ckpt_latest.pt"
            self.assertEqual(first.stat().st_ino, latest.stat().st_ino)
            old_first_contents = first.read_bytes()
            with torch.no_grad():
                generator.images.add_(0.1)
            save_checkpoint(workdir, 2, generator, ema, optimizer, config, keep_last=4, adversarial_system=system)
            second = checkpoint_dir / "ckpt_step_0000002.pt"
            self.assertEqual(first.read_bytes(), old_first_contents)
            self.assertEqual(second.stat().st_ino, latest.stat().st_ino)
            second_alias = checkpoint_dir / "retained_step2_alias.pt"
            second_alias.hardlink_to(second)
            old_second_contents = second_alias.read_bytes()
            with torch.no_grad():
                generator.images.add_(0.1)
            save_checkpoint(workdir, 2, generator, ema, optimizer, config, keep_last=4, adversarial_system=system)
            self.assertEqual(second_alias.read_bytes(), old_second_contents)
            self.assertEqual(first.read_bytes(), old_first_contents)
            self.assertNotEqual(second.stat().st_ino, second_alias.stat().st_ino)
            self.assertEqual(second.stat().st_ino, latest.stat().st_ino)

    def test_lazy_r1_is_safe_and_synchronized_across_two_cpu_ranks(self):
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("Gloo distributed backend is unavailable")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_ddp_lazy_r1_worker, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
