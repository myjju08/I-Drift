import copy
import unittest
from unittest import mock

import torch

from models.imagenet_generator import (
    Attention,
    apply_rope,
    build_ditgen_from_config,
)


def _relative_l2_error(reference: torch.Tensor, actual: torch.Tensor) -> float:
    reference_fp32 = reference.detach().float()
    actual_fp32 = actual.detach().float()
    denominator = reference_fp32.norm().clamp_min(torch.finfo(torch.float32).tiny)
    return float((reference_fp32 - actual_fp32).norm() / denominator)


class ImageNetGeneratorAttentionTest(unittest.TestCase):
    def _run_manual_sdpa_pair(
        self,
        *,
        attn_fp32: bool,
        use_bf16: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Attention,
        Attention,
    ]:
        torch.manual_seed(123)
        manual = Attention(
            dim=64,
            num_heads=4,
            use_qk_norm=True,
            use_rope=True,
            use_rmsnorm=True,
            attn_fp32=attn_fp32,
            use_sdpa=False,
        )
        sdpa = copy.deepcopy(manual)
        sdpa.use_sdpa = True

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        inputs = torch.randn(2, 17, 64, dtype=dtype)
        output_grad = torch.randn_like(inputs)
        manual_inputs = inputs.clone().requires_grad_()
        sdpa_inputs = inputs.clone().requires_grad_()

        with torch.autocast(
            device_type="cpu", dtype=torch.bfloat16, enabled=use_bf16
        ):
            manual_output = manual(manual_inputs)
        with torch.autocast(
            device_type="cpu", dtype=torch.bfloat16, enabled=use_bf16
        ):
            sdpa_output = sdpa(sdpa_inputs)

        manual_output.backward(output_grad)
        sdpa_output.backward(output_grad)
        return (
            manual_output,
            sdpa_output,
            manual_inputs.grad,
            sdpa_inputs.grad,
            manual,
            sdpa,
        )

    def test_sdpa_matches_manual_attention_forward_and_backward_fp32(self):
        for attn_fp32 in (False, True):
            with self.subTest(attn_fp32=attn_fp32):
                (
                    manual_output,
                    sdpa_output,
                    manual_input_grad,
                    sdpa_input_grad,
                    manual,
                    sdpa,
                ) = self._run_manual_sdpa_pair(
                    attn_fp32=attn_fp32,
                    use_bf16=False,
                )

                torch.testing.assert_close(
                    sdpa_output, manual_output, rtol=1e-5, atol=5e-6
                )
                torch.testing.assert_close(
                    sdpa_input_grad, manual_input_grad, rtol=1e-5, atol=5e-6
                )
                for (manual_name, manual_param), (sdpa_name, sdpa_param) in zip(
                    manual.named_parameters(), sdpa.named_parameters()
                ):
                    self.assertEqual(sdpa_name, manual_name)
                    torch.testing.assert_close(
                        sdpa_param.grad,
                        manual_param.grad,
                        rtol=1e-5,
                        atol=5e-6,
                        msg=lambda msg, name=manual_name: f"{name}: {msg}",
                    )

    def test_sdpa_matches_manual_attention_forward_and_backward_bf16(self):
        for attn_fp32 in (False, True):
            with self.subTest(attn_fp32=attn_fp32):
                (
                    manual_output,
                    sdpa_output,
                    manual_input_grad,
                    sdpa_input_grad,
                    manual,
                    sdpa,
                ) = self._run_manual_sdpa_pair(
                    attn_fp32=attn_fp32,
                    use_bf16=True,
                )

                self.assertEqual(sdpa_output.dtype, torch.bfloat16)
                torch.testing.assert_close(
                    sdpa_output, manual_output, rtol=5e-2, atol=1e-2
                )
                self.assertLess(
                    _relative_l2_error(manual_input_grad, sdpa_input_grad), 1e-2
                )
                for (manual_name, manual_param), (sdpa_name, sdpa_param) in zip(
                    manual.named_parameters(), sdpa.named_parameters()
                ):
                    self.assertEqual(sdpa_name, manual_name)
                    self.assertLess(
                        _relative_l2_error(manual_param.grad, sdpa_param.grad),
                        1e-2,
                        manual_name,
                    )

    def test_sdpa_preserves_rope_dtype_policy(self):
        for attn_fp32, expected_dtype in (
            (False, torch.bfloat16),
            (True, torch.float32),
        ):
            with self.subTest(attn_fp32=attn_fp32):
                attention = Attention(
                    dim=64,
                    num_heads=4,
                    use_rope=True,
                    attn_fp32=attn_fp32,
                    use_sdpa=True,
                )
                inputs = torch.randn(2, 17, 64, dtype=torch.bfloat16)
                with mock.patch(
                    "models.imagenet_generator.apply_rope", wraps=apply_rope
                ) as rope:
                    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                        attention(inputs)
                self.assertEqual(rope.call_args.kwargs["rope_dtype"], expected_dtype)

    def test_use_sdpa_config_propagates_to_every_block(self):
        model = build_ditgen_from_config(
            {
                "cond_dim": 64,
                "noise_classes": 2,
                "noise_coords": 1,
                "input_size": 8,
                "in_channels": 4,
                "patch_size": 4,
                "hidden_size": 64,
                "depth": 2,
                "num_heads": 4,
                "out_channels": 4,
                "n_cls_tokens": 2,
                "use_bf16": False,
                "use_sdpa": True,
            },
            {"num_classes": 10},
        )

        self.assertTrue(all(block.attn.use_sdpa for block in model.model.blocks))
        self.assertFalse(Attention(dim=64, num_heads=4).use_sdpa)


if __name__ == "__main__":
    unittest.main()
