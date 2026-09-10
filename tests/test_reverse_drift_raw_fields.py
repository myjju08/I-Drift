import unittest

import torch

from drifting_core.imagenet_loss import (
    drift_loss_imagenet,
    reverse_drift_raw_fields,
)


class ReverseDriftRawFieldsTest(unittest.TestCase):
    @staticmethod
    def _inputs(*, requires_grad: bool = False):
        torch.manual_seed(20260902)
        gen = torch.randn(2, 5, 7, requires_grad=requires_grad)
        pos = torch.randn(2, 8, 7, requires_grad=requires_grad)
        neg = torch.randn(2, 4, 7, requires_grad=requires_grad)
        return gen, pos, neg

    def test_each_raw_field_energy_matches_baseline_loss_R(self):
        gen, pos, neg = self._inputs()
        weight_gen = torch.rand(2, 5).add(0.5)
        weight_pos = torch.rand(2, 8).add(0.5)
        weight_neg = torch.rand(2, 4).add(0.5)
        active_pos = torch.ones(2, 8)
        active_pos[:, -1] = 0
        active_neg = torch.ones(2, 4)
        active_neg[:, -1] = 0
        R_list = (0.2, 0.05, 0.02)
        common = dict(
            weight_gen=weight_gen,
            weight_pos=weight_pos,
            weight_neg=weight_neg,
            R_list=R_list,
            active_mask_pos=active_pos,
            active_mask_neg=active_neg,
            global_scale_stats=False,
            top_p=1.0,
            affinity_kernel="exponential",
        )

        _, info = drift_loss_imagenet(
            gen,
            pos,
            neg,
            global_fnorm_stats=False,
            **common,
        )
        fields = reverse_drift_raw_fields(gen, pos, neg, **common)

        self.assertEqual(len(fields), len(R_list))
        for R, field in zip(R_list, fields):
            self.assertEqual(tuple(field.shape), tuple(gen.shape))
            self.assertEqual(
                field.detach().square().mean().item(),
                info[f"loss_{R}"],
            )

    def test_field_energy_backpropagates_to_query_pos_and_neg(self):
        gen, pos, neg = self._inputs(requires_grad=True)
        fields = reverse_drift_raw_fields(
            gen,
            pos,
            neg,
            R_list=(0.2, 0.05),
            global_scale_stats=False,
            top_p=0.9,
            top_p_min_keep=2,
            affinity_kernel="exponential",
        )
        energy = torch.stack([field.square().mean() for field in fields]).mean()
        energy.backward()

        self.assertTrue(torch.isfinite(energy))
        for name, value in (("gen", gen), ("pos", pos), ("neg", neg)):
            with self.subTest(name=name):
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(float(value.grad.abs().sum()), 0.0)

    def test_top_k_field_energy_is_differentiable(self):
        gen, pos, neg = self._inputs(requires_grad=True)
        field, = reverse_drift_raw_fields(
            gen,
            pos,
            neg,
            R_list=(0.2,),
            global_scale_stats=False,
            top_k_pos=3,
            top_k_neg=4,
        )
        field.square().mean().backward()

        for value in (gen, pos, neg):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.0)

    def test_support_only_field_has_all_gradients_and_requires_repulsion(self):
        gen, pos, neg = self._inputs(requires_grad=True)
        field, = reverse_drift_raw_fields(
            gen,
            pos,
            neg,
            R_list=(0.2,),
            global_scale_stats=False,
            include_query_targets=False,
            repulsion_coefficient=0.7,
        )
        field.square().mean().backward()

        for value in (gen, pos, neg):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "non-empty repulsive bank"):
            reverse_drift_raw_fields(
                gen.detach(),
                pos.detach(),
                R_list=(0.2,),
                global_scale_stats=False,
                include_query_targets=False,
            )

    def test_repulsion_coefficient_is_affine_and_default_is_exact(self):
        gen, pos, neg = self._inputs()
        common = dict(
            R_list=(0.2,),
            global_scale_stats=False,
            include_query_targets=False,
        )
        default, = reverse_drift_raw_fields(gen, pos, neg, **common)
        explicit_one, = reverse_drift_raw_fields(
            gen, pos, neg, repulsion_coefficient=1.0, **common
        )
        attraction, = reverse_drift_raw_fields(
            gen, pos, neg, repulsion_coefficient=0.0, **common
        )
        rho = 0.37
        mixed, = reverse_drift_raw_fields(
            gen, pos, neg, repulsion_coefficient=rho, **common
        )

        torch.testing.assert_close(default, explicit_one, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            mixed,
            attraction + rho * (explicit_one - attraction),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        self.assertGreater(float(attraction.abs().sum()), 0.0)
        for invalid in (-0.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "repulsion_coefficient"):
                    reverse_drift_raw_fields(
                        gen,
                        pos,
                        neg,
                        repulsion_coefficient=invalid,
                        **common,
                    )


if __name__ == "__main__":
    unittest.main()
