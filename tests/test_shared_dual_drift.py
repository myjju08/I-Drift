import unittest
from unittest import mock

import torch

import drifting_core.imagenet_loss as imagenet_loss


class SharedDualDriftDistanceTest(unittest.TestCase):
    @staticmethod
    def _inputs():
        torch.manual_seed(123)
        batch, n_gen, n_pos, n_neg, dim = 3, 4, 5, 2, 7
        gen = torch.randn(batch, n_gen, dim)
        pos = torch.randn(batch, n_pos, dim)
        neg = torch.randn(batch, n_neg, dim)
        weight_neg = torch.rand(batch, n_neg) + 0.5
        active_pos = torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.float32,
        )
        active_neg = torch.tensor(
            [[1, 1], [1, 0], [1, 1]],
            dtype=torch.float32,
        )
        return gen, pos, neg, weight_neg, active_pos, active_neg

    @staticmethod
    def _run(
        *,
        share_distances: bool,
        return_raw_winner_stats: bool = True,
        baseline_top_p: float = 1.0,
        versionb_top_p: float = 1.0,
        top_p_min_keep: int = 1,
    ):
        gen, pos, neg, weight_neg, active_pos, active_neg = (
            SharedDualDriftDistanceTest._inputs()
        )
        gen = gen.requires_grad_()
        loss, info = imagenet_loss.drift_loss_imagenet_mixed(
            gen=gen,
            fixed_pos=pos,
            fixed_neg=neg,
            weight_neg=weight_neg,
            alpha=0.37,
            R_list_baseline=(0.2, 0.05, 0.02),
            R_list_versionb=(0.4, 0.10, 0.04),
            active_mask_pos=active_pos,
            active_mask_neg=active_neg,
            return_raw_winner_stats=return_raw_winner_stats,
            global_scale_stats=False,
            global_fnorm_stats=False,
            share_distances=share_distances,
            baseline_top_p=baseline_top_p,
            versionb_top_p=versionb_top_p,
            top_p_min_keep=top_p_min_keep,
        )
        loss.sum().backward()
        return loss.detach(), gen.grad.detach(), info

    def test_shared_path_matches_sequential_loss_gradient_and_diagnostics(self):
        reference_loss, reference_grad, reference_info = self._run(
            share_distances=False
        )
        shared_loss, shared_grad, shared_info = self._run(
            share_distances=True
        )

        torch.testing.assert_close(
            shared_loss,
            reference_loss,
            rtol=2e-5,
            atol=2e-6,
        )
        torch.testing.assert_close(
            shared_grad,
            reference_grad,
            rtol=2e-5,
            atol=2e-6,
        )
        self.assertEqual(set(shared_info), set(reference_info))
        for key in reference_info:
            with self.subTest(metric=key):
                # Both paths reorder FP32 arithmetic. Apply the same
                # scale-aware tolerance used for their loss and gradients.
                torch.testing.assert_close(
                    float(shared_info[key]),
                    float(reference_info[key]),
                    rtol=2e-5,
                    atol=2e-6,
                )

    def test_shared_path_halves_pairwise_distance_calls(self):
        original_cdist = imagenet_loss._cdist_batched
        with mock.patch.object(
            imagenet_loss,
            "_cdist_batched",
            wraps=original_cdist,
        ) as sequential_cdist:
            self._run(
                share_distances=False,
                return_raw_winner_stats=False,
            )
        with mock.patch.object(
            imagenet_loss,
            "_cdist_batched",
            wraps=original_cdist,
        ) as shared_cdist:
            self._run(
                share_distances=True,
                return_raw_winner_stats=False,
            )

        # Each expert previously computed gen-gen, gen-neg, and gen-pos.
        self.assertEqual(sequential_cdist.call_count, 6)
        self.assertEqual(shared_cdist.call_count, 3)

    def test_shared_top_p_path_matches_sequential_loss_and_gradient(self):
        reference_loss, reference_grad, _ = self._run(
            share_distances=False,
            baseline_top_p=0.90,
            versionb_top_p=0.95,
            top_p_min_keep=2,
        )
        shared_loss, shared_grad, _ = self._run(
            share_distances=True,
            baseline_top_p=0.90,
            versionb_top_p=0.95,
            top_p_min_keep=2,
        )

        torch.testing.assert_close(
            shared_loss, reference_loss, rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            shared_grad, reference_grad, rtol=2e-5, atol=2e-6
        )

    def test_raw_winner_stats_reuse_shared_positive_distance_view(self):
        original_cdist = imagenet_loss._cdist_batched
        with mock.patch.object(
            imagenet_loss,
            "_cdist_batched",
            wraps=original_cdist,
        ) as shared_cdist:
            _, _, info = self._run(
                share_distances=True,
                return_raw_winner_stats=True,
            )

        # Winner statistics use the positive slice rather than a fourth cdist.
        self.assertEqual(shared_cdist.call_count, 3)
        self.assertIn("raw/pos_winner_count", info)
        self.assertIn("raw/gen_winner_count", info)


if __name__ == "__main__":
    unittest.main()
