"""End-to-end optimizer checks for the three latent corrective-field arms."""
import copy
from unittest import mock

import pytest
import torch

import train_imagenet_gen as trainer
from tests.test_adversarial_drift import _TinyFrozenDino, _TinyGenerator, _trainer_inputs


def fixtures():
    torch.manual_seed(43)
    generator = _TinyGenerator()
    generator.images = torch.nn.Parameter(torch.randn(4, 4, 32, 32) * .2)
    teacher = _TinyFrozenDino()
    teacher.projection = torch.nn.Conv2d(4, 4, 1).requires_grad_(False)
    inputs = _trainer_inputs()
    inputs.update(
        pos_samples=torch.randn(2, 2, 4, 32, 32),
        neg_samples=torch.randn(2, 1, 4, 32, 32),
        historical_samples=torch.randn(2, 1, 4, 32, 32).requires_grad_(),
    )
    inputs["cfg"].update(
        feature_extractor="mae", use_latent=True, in_channels=4, out_channels=4,
        num_classes=4, historical_gen_replay=True, historical_gen_replay_ratio=.35,
        double_drift=True, adversarial_mode="raw_gan", adversarial_loss_weight=.1,
        adversarial_base_channels=4, adversarial_r1_gamma=.1,
        adversarial_d_chunk_size=2, adversarial_g_chunk_size=2,
    )
    return generator, teacher, inputs


@pytest.mark.parametrize("gan", [False, True])
def test_double_replay_updates_generator_without_changing_frozen_inputs(gan):
    generator, teacher, inputs = fixtures()
    cfg = inputs["cfg"]
    cfg["adversarial_mode"] = "raw_gan" if gan else "none"
    system = trainer.build_adversarial_system(cfg, inputs["device"])
    optimizer = torch.optim.SGD(generator.parameters(), lr=.001)
    old_gen = generator.images.detach().clone()
    teacher_before = copy.deepcopy(teacher.state_dict())
    history_before = inputs["historical_samples"].detach().clone()
    calls = []
    implementation = trainer.drift_loss_imagenet

    def checked_loss(*args, **kwargs):
        calls.append(kwargs)
        return implementation(*args, **kwargs)

    with mock.patch.object(trainer, "drift_loss_imagenet", side_effect=checked_loss):
        loss, metrics, _ = trainer.train_step(
            generator, teacher, optimizer, adversarial_system=system, **inputs,
        )
    assert torch.isfinite(loss) and torch.isfinite(generator.images.grad).all()
    assert not torch.equal(old_gen, generator.images.detach())
    assert calls and all(call["double_drift"] for call in calls)
    for call in calls:
        assert call["historical_gen"] is not None
        torch.testing.assert_close(call["weight_gen"], torch.full_like(call["weight_gen"], .65))
        torch.testing.assert_close(call["weight_history"], torch.full_like(call["weight_history"], .7))
    assert inputs["historical_samples"].grad is None
    assert torch.equal(history_before, inputs["historical_samples"].detach())
    for name, value in teacher.state_dict().items():
        assert torch.equal(value, teacher_before[name])
    assert all(parameter.grad is None for parameter in teacher.parameters())
    if gan:
        assert metrics["adversarial/d_updates"] == 1
        assert metrics["loss"] == pytest.approx(
            metrics["drift_loss"] + .1 * metrics["adversarial/raw_gan_loss"], rel=1e-6,
        )
        assert all(parameter.grad is None for parameter in system.target.parameters())
    else:
        assert metrics["loss"] == metrics["drift_loss"]


def test_double_drift_rejects_forward_or_mixed_modes():
    features = {"layer3": torch.randn(4, 1, 3, requires_grad=True)}
    for mode in ("fwd-drift", "dual-drift"):
        with pytest.raises(ValueError, match="double_drift requires rev-drift"):
            trainer.compute_drift_loss_from_features(
                features, {"layer3": torch.randn(4, 1, 3)}, None,
                B=2, G=2, P=2, N=0, weight_neg=None,
                drift_matching=mode, double_drift=True,
            )


def test_frozen_replay_boundary_and_checkpoint_resume(tmp_path):
    generator, teacher, inputs = fixtures()
    labels = torch.tensor([0, 0, 1, 1])
    dataset = torch.utils.data.TensorDataset(torch.randn(4, 4, 32, 32), labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    cfg = dict(inputs["cfg"], num_classes=2, batch_size=2, loader_batch_size=4,
               num_workers=0, use_cache=True, positive_bank_size=4,
               negative_bank_size=4, pos_per_sample=2, neg_per_sample=1,
               push_per_step=4, push_at_resume=1, total_steps=10,
               train_max_step_exclusive=4, eval_at_start=False,
               eval_per_step=1000, save_per_step=1, keep_every=100,
               keep_last=4, use_wandb=False, log_every_k=1, seed=43,
               lr=.001, warmup_steps=1, historical_gen_replay_count=1,
               historical_gen_replay_bank_count=1,
               historical_gen_replay_start_generated_epochs=2.,
               historical_gen_replay_source="frozen_snapshot")
    observed = []
    real_train_step = trainer.train_step

    def tracked_step(*args, **kwargs):
        observed.append((args[7], kwargs["historical_samples"] is not None))
        return real_train_step(*args, **kwargs)

    def run(config):
        with (
            mock.patch.object(trainer, "build_ditgen_from_config", side_effect=lambda *_: copy.deepcopy(generator)),
            mock.patch.object(trainer, "load_feature_extractor", side_effect=lambda *_: copy.deepcopy(teacher)),
            mock.patch.object(trainer, "create_imagenet_split", return_value=(
                loader, lambda batch: {"images": batch[0], "labels": batch[1]}, lambda value: value,
            )),
            mock.patch.object(trainer, "_step_choice_indices", return_value=torch.tensor([0, 2]).numpy()),
            mock.patch.object(trainer, "tqdm", side_effect=lambda values, **_: values),
            mock.patch.object(trainer, "train_step", side_effect=tracked_step),
        ):
            trainer.train_gen(config, str(tmp_path), 0, 1, torch.device("cpu"))

    run(dict(cfg, train_max_step_exclusive=1, checkpoint_latest_hardlink=True))
    capture = tmp_path / "historical_gen_replay_capture_step0000001_rank00.npz"
    assert capture.is_file()
    run(dict(cfg, checkpoint_latest_hardlink=True))
    assert observed == [(0, False), (1, False), (2, True), (3, True)]
    snapshot = tmp_path / "historical_gen_replay_rank00.npz"
    frozen_bytes = snapshot.read_bytes()
    state = torch.load(tmp_path / "checkpoints/ckpt_latest.pt", weights_only=False)
    assert state["step"] == 4 and state["adversarial_system"]["updates"] == 4
    run(dict(cfg, train_max_step_exclusive=5))
    assert observed[-1] == (4, True)
    assert snapshot.read_bytes() == frozen_bytes
    state = torch.load(tmp_path / "checkpoints/ckpt_latest.pt", weights_only=False)
    assert state["step"] == 5 and state["adversarial_system"]["updates"] == 5
    snapshot.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        run(dict(cfg, train_max_step_exclusive=6))
