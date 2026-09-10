import unittest
from unittest import mock

import torch

import drifting_core.imagenet_loss as imagenet_loss
from train_imagenet_gen import (
    _collect_mix_alpha_stats_this_step,
    _collect_training_metrics_this_step,
    _configure_level4_cuda_runtime,
    _zero_generator_grad,
    compute_drift_loss_from_features,
)


class ReverseLossLevel4Test(unittest.TestCase):
    @staticmethod
    def _inputs():
        torch.manual_seed(1234)
        gen = torch.randn(2, 3, 7, requires_grad=True)
        pos = torch.randn(2, 4, 7)
        neg = torch.randn(2, 2, 7)
        return gen, pos, neg

    def test_collect_diagnostics_false_preserves_loss_and_gradient(self) -> None:
        gen, pos, neg = self._inputs()
        common = dict(
            fixed_pos=pos,
            fixed_neg=neg,
            R_list=(0.2, 0.05, 0.02),
            compute_wpos_stats=True,
            global_scale_stats=False,
            global_fnorm_stats=False,
            force_multiplier=1.25,
        )

        reference_loss, reference_info = imagenet_loss.drift_loss_imagenet(
            gen=gen,
            collect_diagnostics=True,
            **common,
        )
        reference_loss.sum().backward()
        reference_grad = gen.grad.detach().clone()

        fast_gen = gen.detach().clone().requires_grad_(True)
        with mock.patch.object(
            imagenet_loss,
            "_wpos_stats_from_matrix",
            side_effect=AssertionError("W_pos diagnostics must be skipped"),
        ):
            fast_loss, fast_info = imagenet_loss.drift_loss_imagenet(
                gen=fast_gen,
                collect_diagnostics=False,
                **common,
            )
        fast_loss.sum().backward()

        self.assertIn("scale", reference_info)
        self.assertIn("loss_0.2", reference_info)
        self.assertIn("force_multiplier", reference_info)
        self.assertIn("wpos/peak", reference_info)
        self.assertEqual(fast_info, {})
        torch.testing.assert_close(fast_loss, reference_loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(fast_gen.grad, reference_grad, rtol=0.0, atol=0.0)

    def test_fused_fnorm_uses_one_fp64_packed_all_reduce(self) -> None:
        gen, pos, neg = self._inputs()
        common = dict(
            fixed_pos=pos,
            fixed_neg=neg,
            R_list=(0.2, 0.05, 0.02),
            global_scale_stats=False,
            global_fnorm_stats=True,
            collect_diagnostics=False,
        )

        streaming_collectives = []

        def record_streaming(tensor, **_kwargs):
            streaming_collectives.append(tensor.detach().clone())

        with (
            mock.patch.object(imagenet_loss, "_dist_ready", return_value=True),
            mock.patch.object(
                imagenet_loss.dist,
                "all_reduce",
                side_effect=record_streaming,
            ),
        ):
            streaming_loss, _ = imagenet_loss.drift_loss_imagenet(
                gen=gen,
                fuse_fnorm_across_R=False,
                **common,
            )
        streaming_loss.sum().backward()
        streaming_grad = gen.grad.detach().clone()

        fused_gen = gen.detach().clone().requires_grad_(True)
        fused_collectives = []

        def record_fused(tensor, **_kwargs):
            fused_collectives.append(tensor.detach().clone())

        with (
            mock.patch.object(imagenet_loss, "_dist_ready", return_value=True),
            mock.patch.object(
                imagenet_loss.dist,
                "all_reduce",
                side_effect=record_fused,
            ),
        ):
            fused_loss, _ = imagenet_loss.drift_loss_imagenet(
                gen=fused_gen,
                fuse_fnorm_across_R=True,
                **common,
            )
        fused_loss.sum().backward()

        self.assertEqual([value.numel() for value in streaming_collectives], [2, 2, 2])
        self.assertEqual(len(fused_collectives), 1)
        packed = fused_collectives[0]
        self.assertEqual(packed.dtype, torch.float64)
        self.assertEqual(packed.numel(), 6)
        expected_count = float(2 * 3 * 7)
        torch.testing.assert_close(
            packed[1::2],
            torch.full((3,), expected_count, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(fused_loss, streaming_loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(fused_gen.grad, streaming_grad, rtol=0.0, atol=0.0)

    def test_feature_wrapper_forwards_reverse_level4_flags(self) -> None:
        captured = {}

        def fake_reverse_loss(*, gen, **kwargs):
            captured.update(kwargs)
            return gen.square().mean(dim=(-1, -2)), {}

        gen_feats = {"global": torch.randn(2, 1, 5, requires_grad=True)}
        pos_feats = {"global": torch.randn(2, 1, 5)}
        with mock.patch(
            "train_imagenet_gen.drift_loss_imagenet",
            side_effect=fake_reverse_loss,
        ):
            compute_drift_loss_from_features(
                gen_feats=gen_feats,
                pos_feats=pos_feats,
                neg_feats=None,
                B=1,
                G=2,
                P=2,
                N=0,
                weight_neg=None,
                drift_matching="rev-drift",
                collect_diagnostics=False,
                fuse_fnorm_across_R=True,
            )

        self.assertIs(captured["collect_diagnostics"], False)
        self.assertIs(captured["fuse_fnorm_across_R"], True)


class Level4TrainingPolicyTest(unittest.TestCase):
    def test_fixed_mix_diagnostics_follow_diagnostic_cadence(self) -> None:
        for mode in ("rev-drift", "fwd-drift"):
            with self.subTest(mode=mode):
                self.assertFalse(
                    _collect_mix_alpha_stats_this_step(
                        throughput_opt_level=4,
                        collect_diagnostics=False,
                        drift_matching=mode,
                    )
                )
                self.assertTrue(
                    _collect_mix_alpha_stats_this_step(
                        throughput_opt_level=4,
                        collect_diagnostics=True,
                        drift_matching=mode,
                    )
                )

    def test_dual_drift_state_remains_per_step(self) -> None:
        self.assertTrue(
            _collect_mix_alpha_stats_this_step(
                throughput_opt_level=4,
                collect_diagnostics=False,
                drift_matching="dual-drift",
            )
        )

    def test_lower_levels_keep_existing_per_step_behavior(self) -> None:
        self.assertTrue(
            _collect_mix_alpha_stats_this_step(
                throughput_opt_level=3,
                collect_diagnostics=False,
                drift_matching="rev-drift",
            )
        )

    def test_pure_metrics_are_gated_only_on_level4_non_log_steps(self) -> None:
        self.assertFalse(
            _collect_training_metrics_this_step(
                throughput_opt_level=4,
                step=9,
                log_every_k=10,
            )
        )
        self.assertTrue(
            _collect_training_metrics_this_step(
                throughput_opt_level=4,
                step=10,
                log_every_k=10,
            )
        )
        self.assertTrue(
            _collect_training_metrics_this_step(
                throughput_opt_level=3,
                step=9,
                log_every_k=10,
            )
        )

    def test_level4_zero_grad_deallocates_gradients(self) -> None:
        optimizer = mock.Mock()
        _zero_generator_grad(optimizer, throughput_opt_level=4)
        optimizer.zero_grad.assert_called_once_with(set_to_none=True)

        optimizer.reset_mock()
        _zero_generator_grad(optimizer, throughput_opt_level=3)
        optimizer.zero_grad.assert_called_once_with()

    def test_level4_cuda_runtime_enables_configured_fast_paths(self) -> None:
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_benchmark = torch.backends.cudnn.benchmark
        has_benchmark_limit = hasattr(torch.backends.cudnn, "benchmark_limit")
        old_benchmark_limit = (
            torch.backends.cudnn.benchmark_limit
            if has_benchmark_limit
            else None
        )
        try:
            with mock.patch.object(
                torch,
                "set_float32_matmul_precision",
            ) as set_precision:
                _configure_level4_cuda_runtime(
                    {
                        "allow_tf32": True,
                        "cudnn_benchmark": True,
                        "cudnn_benchmark_limit": 7,
                    },
                    throughput_opt_level=4,
                    device=torch.device("cuda"),
                )
                set_precision.assert_called_once_with("high")
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cudnn.benchmark)
            if has_benchmark_limit:
                self.assertEqual(torch.backends.cudnn.benchmark_limit, 7)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.backends.cudnn.benchmark = old_benchmark
            if has_benchmark_limit:
                torch.backends.cudnn.benchmark_limit = old_benchmark_limit


if __name__ == "__main__":
    unittest.main()
