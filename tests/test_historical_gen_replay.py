import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from drifting_core.imagenet_loss import drift_loss_imagenet
from memory_bank import ArrayMemoryBank
from train_imagenet_gen import (
    _historical_replay_ratio_for_step,
    compute_drift_loss_from_features,
)


class HistoricalGeneratedReplayTest(unittest.TestCase):
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
