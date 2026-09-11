"""Policy boundaries and exact resume continuity for historical replay banks."""
import numpy as np
import pytest
import torch

from memory_bank import ArrayMemoryBank, HistoricalReplayMemoryBank


ARRAY_STATE = ("bank", "ptr", "count", "use_count", "insert_step", "seen_count")
COUNTER_STATE = ("sample_count", "replacement_count", "discard_count")


def _assert_same_state(left, right):
    for name in ARRAY_STATE:
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)
    for name in COUNTER_STATE:
        assert getattr(left, name) == getattr(right, name), name
    assert left.metrics(step=100) == right.metrics(step=100)


def _stream_step(bank, step):
    labels = np.array([0, 1, 0], dtype=np.int64)
    # Repeated class IDs and replacement sampling exercise every actual use.
    replay = bank.sample(labels, 5, rng=np.random.default_rng([814, step]))
    candidates = np.random.default_rng([927, step]).normal(size=(6, 2, 3)).astype(np.float32)
    bank.update(candidates, np.array([0, 0, 1, 1, 0, 1]), step=step,
                rng=np.random.default_rng([318, step]))
    return replay


@pytest.mark.parametrize("policy", HistoricalReplayMemoryBank.POLICIES)
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_policy_resume_matches_uninterrupted_future_samples_and_replacements(tmp_path, policy, dtype):
    options = dict(num_classes=2, max_size=4, dtype=dtype, policy=policy, usage_budget=3)
    uninterrupted = HistoricalReplayMemoryBank(**options)
    samples = torch.arange(48, dtype=torch.float32).reshape(8, 2, 3) / 10
    uninterrupted.add(samples, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]), step=10)
    for step in range(11, 20):
        _stream_step(uninterrupted, step)
    path = tmp_path / "historical_step0000019.npz"
    uninterrupted.save_npz(path)
    resumed = HistoricalReplayMemoryBank(**options)
    resumed.load_npz(path, default_step=10)
    _assert_same_state(uninterrupted, resumed)
    for step in range(20, 36):
        expected = _stream_step(uninterrupted, step)
        observed = _stream_step(resumed, step)
        assert torch.equal(observed, expected)
        _assert_same_state(uninterrupted, resumed)


def test_usage_budget_counts_repeated_draws_then_retires_oldest_tied_anchor():
    bank = HistoricalReplayMemoryBank(num_classes=1, max_size=2,
                                      policy="usage_budget", usage_budget=2)
    bank.add(torch.tensor([[10.0]]), torch.tensor([0]), step=8)
    bank.add(torch.tensor([[20.0]]), torch.tensor([0]), step=10)
    bank.sample(np.array([0, 0]), 2, rng=np.random.default_rng(41))
    np.testing.assert_array_equal(bank.use_count, [[2, 2]])
    assert bank.sample_count == 4
    bank.update(torch.tensor([[30.0]]), torch.tensor([0]), step=11)
    np.testing.assert_array_equal(bank.bank[:, :, 0], [[30, 20]])
    np.testing.assert_array_equal(bank.use_count, [[0, 2]])
    np.testing.assert_array_equal(bank.insert_step, [[11, 10]])
    bank.update(torch.tensor([[40.0], [50.0]]), torch.tensor([0, 0]), step=12)
    np.testing.assert_array_equal(bank.bank[:, :, 0], [[30, 40]])
    assert bank.replacement_count == 2 and bank.discard_count == 1


@pytest.mark.parametrize("policy", ["fifo", "reservoir", "usage_budget"])
def test_partial_bank_fill_precedes_replacement_policy(policy):
    bank = HistoricalReplayMemoryBank(num_classes=2, max_size=3, policy=policy)
    bank.add(torch.tensor([[1.0]]), torch.tensor([0]), step=4)
    with pytest.raises(RuntimeError, match="class 1 has no anchors"):
        bank.sample(np.array([1]), 1)
    bank.update(torch.tensor([[2.0], [3.0], [9.0]]), torch.tensor([0, 0, 1]),
                step=5, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(bank.count, [3, 1])
    np.testing.assert_array_equal(bank.ptr, [0, 1])
    np.testing.assert_array_equal(bank.bank[0, :, 0], [1, 2, 3])
    assert bank.bank[1, 0, 0] == 9
    assert bank.replacement_count == bank.discard_count == 0


def test_target_legacy_raw_snapshot_migrates_at_explicit_boundary(tmp_path):
    legacy = ArrayMemoryBank(num_classes=2, max_size=2, dtype=np.float32)
    legacy.add(torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([0, 0, 1]))
    path = tmp_path / "legacy.npz"
    legacy.save_npz(path)
    restored = HistoricalReplayMemoryBank(num_classes=2, max_size=2,
                                          dtype=np.float32, policy="reservoir")
    restored.load_npz(path, default_step=25023)
    np.testing.assert_array_equal(restored.bank, legacy.bank)
    np.testing.assert_array_equal(restored.use_count, 0)
    np.testing.assert_array_equal(restored.insert_step, [[25023, 25023], [25023, -1]])
    np.testing.assert_array_equal(restored.seen_count, [2, 1])
    assert restored.sample_count == restored.replacement_count == restored.discard_count == 0


@pytest.mark.parametrize("options", [{"policy": "random"}, {"usage_budget": 0}, {"usage_budget": -1}])
def test_invalid_policy_configuration_fails_before_use(options):
    with pytest.raises(ValueError):
        HistoricalReplayMemoryBank(**options)


@pytest.mark.parametrize("policy", HistoricalReplayMemoryBank.POLICIES)
def test_trainer_resume_uses_bank_paired_with_older_checkpoint(tmp_path, monkeypatch, policy):
    """Exercise actual trainer persistence while replacing costly model/data work."""
    import shutil
    from unittest import mock
    from torch.utils.data import DataLoader, TensorDataset
    import train_imagenet_gen as trainer

    dataset = TensorDataset(torch.arange(16, dtype=torch.float32).reshape(4, 4, 1, 1),
                            torch.zeros(4, dtype=torch.long))

    def make_loader(**unused):
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        return loader, lambda batch: {"images": batch[0], "labels": batch[1]}, lambda value: value

    observations = {}

    def inexpensive_train_step(generator, feature_extractor, optimizer, labels,
                               positive, negative, device, step, cfg, **kwargs):
        history = kwargs["historical_samples"]
        observations[step] = None if history is None else history.detach().clone()
        optimizer.zero_grad()
        loss = sum(parameter.square().sum() for parameter in generator.parameters())
        loss.backward()
        optimizer.step()
        samples = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1, 1) + 10 * step
        return loss.detach(), {"loss": float(loss.detach())}, {"gen_samples_detached": samples}

    monkeypatch.setattr(trainer, "build_ditgen_from_config", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(trainer, "build_optimizer", lambda model, cfg: torch.optim.SGD(model.parameters(), lr=0.01))
    monkeypatch.setattr(trainer, "load_feature_extractor", lambda *args: torch.nn.Identity())
    monkeypatch.setattr(trainer, "build_feature_adapter_system", lambda *args: (None, None, None))
    monkeypatch.setattr(trainer, "build_feature_discriminator_system", lambda *args: (None, None))
    monkeypatch.setattr(trainer, "build_adversarial_system", lambda *args: None)
    monkeypatch.setattr(trainer, "create_imagenet_split", make_loader)
    monkeypatch.setattr(trainer, "train_step", inexpensive_train_step)
    monkeypatch.setattr(trainer, "_run_generator_evaluation", lambda **kwargs: None)
    monkeypatch.setattr(trainer, "Logger", lambda *args: mock.Mock())
    monkeypatch.setattr(trainer, "tqdm", lambda values, **kwargs: values)
    cfg = dict(seed=42, seed_host_rng=True, batch_size=1, loader_batch_size=2, gen_per_label=2,
               num_classes=1, positive_bank_size=2, negative_bank_size=2,
               pos_per_sample=2, neg_per_sample=2, push_per_step=2, push_at_resume=1,
               historical_gen_replay=True, historical_gen_replay_ratio=0.35,
               historical_gen_replay_count=2, historical_gen_replay_bank_count=2,
               historical_gen_replay_policy=policy,
               historical_gen_replay_start_generated_epochs=1.0,
               historical_gen_replay_update_count=1,
               historical_gen_replay_update_interval_steps=2,
               historical_gen_replay_usage_budget=2,
               historical_gen_replay_storage_dtype="float32",
               total_steps=8, save_per_step=2, eval_per_step=1000,
               keep_last=8, keep_every=2, log_every_k=1, warmup_steps=0,
               checkpoint_latest_hardlink=True)
    uninterrupted_dir, resumed_dir = tmp_path / "uninterrupted", tmp_path / "resumed"
    trainer.train_gen(dict(cfg), str(uninterrupted_dir), 0, 1, torch.device("cpu"))
    expected = {step: value for step, value in observations.items() if step >= 4}
    assert observations[0] is None and observations[1] is None
    assert all(observations[step] is not None for step in range(2, 8))

    # Preserve newer bank files deliberately: a latest-only replay snapshot
    # would make this older retained checkpoint resume with future anchors.
    shutil.copytree(uninterrupted_dir, resumed_dir)
    checkpoint_dir = resumed_dir / "checkpoints"
    (checkpoint_dir / "ckpt_latest.pt").unlink()
    shutil.copy2(checkpoint_dir / "ckpt_step_0000004.pt", checkpoint_dir / "ckpt_latest.pt")
    observations.clear()
    trainer.train_gen(dict(cfg), str(resumed_dir), 0, 1, torch.device("cpu"))
    assert set(observations) == set(expected)
    for step in expected:
        assert torch.equal(observations[step], expected[step]), f"replay mismatch at step {step}"
    snapshot = "historical_gen_replay_state_step0000008_rank00.npz"
    with np.load(uninterrupted_dir / snapshot, allow_pickle=False) as first, \
         np.load(resumed_dir / snapshot, allow_pickle=False) as second:
        assert set(first.files) == set(second.files)
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
