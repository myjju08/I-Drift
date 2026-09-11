import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from drifting_core.imagenet_loss import drift_loss_imagenet
from memory_bank import ArrayMemoryBank, HistoricalReplayMemoryBank
from train_imagenet_gen import (
    _historical_replay_update_due,
    _historical_replay_ratio_for_step,
    compute_drift_loss_from_features,
)


class HistoricalGeneratedReplayTest(unittest.TestCase):
    def test_rolling_update_cadence_is_relative_to_snapshot_step(self):
        due = [
            step
            for step in range(99, 111)
            if _historical_replay_update_due(
                step=step,
                start_step=100,
                interval_steps=4,
            )
        ]
        self.assertEqual(due, [100, 104, 108])
        with self.assertRaises(ValueError):
            _historical_replay_update_due(
                step=100,
                start_step=100,
                interval_steps=0,
            )

    @staticmethod
    def _policy_bank(policy: str, *, usage_budget: int = 2):
        bank = HistoricalReplayMemoryBank(
            num_classes=1,
            max_size=4,
            dtype=np.float32,
            policy=policy,
            usage_budget=usage_budget,
        )
        bank.add(
            torch.arange(4, dtype=torch.float32).reshape(4, 1),
            torch.zeros(4, dtype=torch.long),
            step=10,
        )
        return bank

    def test_frozen_policy_ignores_stream_updates(self):
        bank = self._policy_bank("frozen")
        before = bank.bank.copy()
        bank.update(
            torch.tensor([[99.0]]),
            torch.tensor([0]),
            step=20,
            rng=np.random.default_rng(1),
        )
        np.testing.assert_array_equal(bank.bank, before)
        self.assertEqual(bank.replacement_count, 0)

    def test_fifo_policy_replaces_ring_order(self):
        bank = self._policy_bank("fifo")
        bank.update(
            torch.tensor([[10.0], [11.0]]),
            torch.tensor([0, 0]),
            step=20,
        )
        np.testing.assert_array_equal(
            bank.bank[0, :, 0], np.array([10.0, 11.0, 2.0, 3.0])
        )
        np.testing.assert_array_equal(bank.insert_step[0], [20, 20, 10, 10])
        self.assertEqual(bank.replacement_count, 2)

    def test_reservoir_policy_is_deterministic_and_counts_stream(self):
        first = self._policy_bank("reservoir")
        second = self._policy_bank("reservoir")
        samples = torch.arange(20, 30, dtype=torch.float32).reshape(10, 1)
        labels = torch.zeros(10, dtype=torch.long)
        first.update(
            samples,
            labels,
            step=20,
            rng=np.random.default_rng(7),
        )
        second.update(
            samples,
            labels,
            step=20,
            rng=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(first.bank, second.bank)
        self.assertEqual(first.seen_count[0], 14)
        self.assertEqual(
            first.replacement_count + first.discard_count,
            10,
        )

    def test_usage_budget_retires_only_expired_anchors(self):
        bank = self._policy_bank("usage_budget", usage_budget=2)
        bank.use_count[0] = np.array([2, 1, 0, 0], dtype=np.int32)
        bank.update(
            torch.tensor([[99.0], [98.0]]),
            torch.tensor([0, 0]),
            step=20,
        )
        self.assertEqual(bank.bank[0, 0, 0], 99.0)
        self.assertNotIn(98.0, bank.bank[0, :, 0])
        self.assertEqual(bank.use_count[0, 0], 0)
        self.assertEqual(bank.replacement_count, 1)
        self.assertEqual(bank.discard_count, 1)

    def test_policy_metadata_round_trip_and_legacy_migration(self):
        bank = self._policy_bank("fifo")
        bank.sample(
            np.array([0]),
            n_samples=2,
            rng=np.random.default_rng(3),
        )
        bank.update(torch.tensor([[9.0]]), torch.tensor([0]), step=21)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.npz"
            bank.save_npz(path)
            restored = HistoricalReplayMemoryBank(
                num_classes=1,
                max_size=4,
                policy="fifo",
            )
            restored.load_npz(path, default_step=10)
            np.testing.assert_array_equal(restored.bank, bank.bank)
            np.testing.assert_array_equal(restored.use_count, bank.use_count)
            np.testing.assert_array_equal(restored.insert_step, bank.insert_step)
            self.assertEqual(restored.sample_count, bank.sample_count)
            self.assertEqual(restored.replacement_count, bank.replacement_count)

            legacy_path = Path(tmpdir) / "legacy.npz"
            legacy = ArrayMemoryBank(num_classes=1, max_size=4)
            legacy.add(
                torch.arange(4, dtype=torch.float32).reshape(4, 1),
                torch.zeros(4, dtype=torch.long),
            )
            legacy.save_npz(legacy_path)
            migrated = HistoricalReplayMemoryBank(
                num_classes=1,
                max_size=4,
                policy="usage_budget",
            )
            migrated.load_npz(legacy_path, default_step=123)
        np.testing.assert_array_equal(migrated.use_count, 0)
        np.testing.assert_array_equal(migrated.insert_step, 123)
        self.assertEqual(migrated.seen_count[0], 4)

    def test_replay_ratio_linear_ramp(self):
        cfg = {
            "historical_gen_replay_ratio": 0.5,
            "historical_gen_replay_ratio_start": 0.0,
            "historical_gen_replay_ratio_ramp_start_step": 10,
            "historical_gen_replay_ratio_ramp_end_step": 20,
        }
        self.assertEqual(_historical_replay_ratio_for_step(cfg, 9, True), 0.0)
        self.assertEqual(_historical_replay_ratio_for_step(cfg, 10, True), 0.0)
        self.assertEqual(_historical_replay_ratio_for_step(cfg, 15, True), 0.25)
        self.assertEqual(_historical_replay_ratio_for_step(cfg, 20, True), 0.5)
        self.assertEqual(_historical_replay_ratio_for_step(cfg, 30, True), 0.5)
        self.assertEqual(_historical_replay_ratio_for_step(cfg, 15, False), 0.0)

    def test_snapshot_round_trip(self):
        bank = ArrayMemoryBank(num_classes=3, max_size=2, dtype=np.float16)
        samples = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        bank.add(samples, labels)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot.npz"
            bank.save_npz(path)
            restored = ArrayMemoryBank(
                num_classes=3, max_size=2, dtype=np.float16
            )
            restored.load_npz(path)
        self.assertTrue(restored.is_ready(2))
        np.testing.assert_array_equal(restored.bank, bank.bank)
        np.testing.assert_array_equal(restored.ptr, bank.ptr)
        np.testing.assert_array_equal(restored.count, bank.count)

    def test_sampling_can_use_an_isolated_rng(self):
        bank = ArrayMemoryBank(num_classes=1, max_size=4)
        bank.add(torch.arange(16).reshape(4, 4), torch.zeros(4, dtype=torch.long))
        np.random.seed(123)
        expected_global = np.random.random()
        np.random.seed(123)
        first = bank.sample(
            np.zeros(1, dtype=np.int64),
            n_samples=4,
            rng=np.random.default_rng(9),
        )
        actual_global = np.random.random()
        second = bank.sample(
            np.zeros(1, dtype=np.int64),
            n_samples=4,
            rng=np.random.default_rng(9),
        )
        self.assertEqual(actual_global, expected_global)
        self.assertTrue(torch.equal(first, second))

    def test_reduced_current_weight_preserves_temperature_scale(self):
        torch.manual_seed(5)
        gen = torch.randn(2, 4, 5)
        pos = torch.randn(2, 5, 5)
        neg = torch.randn(2, 3, 5)
        common = dict(
            R_list=(0.75,),
            affinity_kernel="generalized_exponential",
            kernel_shape=2.0,
            kernel_adaptive_k_pos=2,
            kernel_adaptive_k_neg=2,
            global_scale_stats=False,
            global_fnorm_stats=False,
        )
        baseline, baseline_info = drift_loss_imagenet(gen, pos, neg, **common)
        weakened, weakened_info = drift_loss_imagenet(
            gen,
            pos,
            neg,
            weight_gen=torch.full((2, 4), 0.5),
            **common,
        )
        self.assertAlmostEqual(
            baseline_info["scale"], weakened_info["scale"], places=6
        )
        self.assertAlmostEqual(
            baseline_info["kernel_bandwidth_mean"],
            weakened_info["kernel_bandwidth_mean"],
            places=6,
        )
        self.assertFalse(torch.allclose(baseline, weakened))

    def test_replay_preserves_baseline_scale_and_bandwidth(self):
        torch.manual_seed(7)
        gen = torch.randn(2, 4, 5, requires_grad=True)
        pos = torch.randn(2, 5, 5)
        neg = torch.randn(2, 3, 5)
        history = torch.randn(2, 2, 5)
        weight_neg = torch.rand(2, 3) + 0.5
        common = dict(
            R_list=(0.75,),
            affinity_kernel="generalized_exponential",
            kernel_shape=2.0,
            kernel_adaptive_k_pos=2,
            kernel_adaptive_k_neg=2,
            kernel_adaptive_margin=1.05,
            global_scale_stats=False,
            global_fnorm_stats=False,
        )
        baseline, baseline_info = drift_loss_imagenet(
            gen, pos, neg, weight_neg=weight_neg, **common
        )
        replay, replay_info = drift_loss_imagenet(
            gen,
            pos,
            neg,
            weight_gen=torch.full((2, 4), 0.75),
            weight_neg=weight_neg,
            historical_gen=history,
            weight_history=torch.full((2, 2), 0.5),
            **common,
        )
        self.assertAlmostEqual(baseline_info["scale"], replay_info["scale"], places=6)
        self.assertAlmostEqual(
            baseline_info["kernel_bandwidth_mean"],
            replay_info["kernel_bandwidth_mean"],
            places=6,
        )
        self.assertEqual(replay_info["history/current_mass"], 3.0)
        self.assertEqual(replay_info["history/replay_mass"], 1.0)
        self.assertTrue(torch.isfinite(replay).all())
        replay.mean().backward()
        self.assertIsNotNone(gen.grad)
        self.assertGreater(float(gen.grad.abs().sum()), 0.0)
        self.assertFalse(torch.allclose(baseline, replay))

    def test_current_repulsion_ablation_weights_are_finite_and_match_masses(self):
        torch.manual_seed(17)
        pos, neg, history = torch.randn(1, 32, 5), torch.randn(1, 32, 5), torch.randn(1, 16, 5)
        for current_weight, history_weight in ((0.25, 2.0), (0.0, 2.0), (0.25, 3.0), (0.0, 4.0)):
            with self.subTest(current=current_weight, history=history_weight):
                gen = torch.randn(1, 64, 5, requires_grad=True)
                loss, info = drift_loss_imagenet(
                    gen, pos, neg,
                    historical_gen=history,
                    weight_gen=torch.full((1, 64), current_weight),
                    weight_history=torch.full((1, 16), history_weight),
                    R_list=(0.2,),
                    global_scale_stats=False,
                    global_fnorm_stats=False,
                )
                self.assertEqual(info["history/current_mass"], 64 * current_weight)
                self.assertEqual(info["history/replay_mass"], 16 * history_weight)
                self.assertTrue(torch.isfinite(loss).all())
                loss.mean().backward()
                self.assertTrue(torch.isfinite(gen.grad).all())
                self.assertGreater(float(gen.grad.abs().sum()), 0.0)

    def test_feature_wrapper_routes_replay_as_detached_targets(self):
        torch.manual_seed(11)
        B, G, P, N, H, T, D = 2, 3, 4, 2, 1, 2, 5
        gen = torch.randn(B * G, T, D, requires_grad=True)
        pos = torch.randn(B * P, T, D)
        neg = torch.randn(B * N, T, D)
        history = torch.randn(B * H, T, D, requires_grad=True)
        loss, info = compute_drift_loss_from_features(
            gen_feats={"norm_x": gen},
            pos_feats={"norm_x": pos},
            neg_feats={"norm_x": neg},
            B=B,
            G=G,
            P=P,
            N=N,
            weight_neg=torch.ones(B, N),
            R_list=(0.2,),
            drift_matching="rev-drift",
            compute_raw_winner_stats_flag=False,
            global_scale_stats=False,
            global_fnorm_stats=False,
            historical_feats={"norm_x": history},
            historical_count=H,
            weight_gen=torch.full((B, G), 0.75),
            weight_history=torch.full((B, H), 0.75),
        )
        loss.backward()
        self.assertIsNotNone(gen.grad)
        self.assertIsNone(history.grad)
        self.assertEqual(info["history/count/norm_x"], 1.0)


if __name__ == "__main__":
    unittest.main()
