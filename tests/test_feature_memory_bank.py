import unittest

import numpy as np
import torch

from models.feature_memory_bank import FeatureMemoryBank


def _feature_batch(values):
    base = torch.tensor(values, dtype=torch.bfloat16).reshape(-1, 1, 1, 1)
    return {
        "layer3": base.expand(-1, 2, 2, 2).clone(),
        "layer4": (base + 100).expand(-1, 3, 1, 1).clone(),
    }


class FeatureMemoryBankTest(unittest.TestCase):
    def test_bfloat16_round_trip_preserves_exact_bits(self):
        bank = FeatureMemoryBank(num_classes=2, max_size=2)
        features = _feature_batch([1.25, -3.5])
        labels = torch.tensor([0, 1])
        bank.add(features, labels)

        sampled = bank.sample(labels, 1, rng=np.random.default_rng(7))
        self.assertEqual(sampled["layer3"].dtype, torch.bfloat16)
        self.assertEqual(sampled["layer4"].device.type, "cpu")
        self.assertTrue(torch.equal(sampled["layer3"][:, 0], features["layer3"]))
        self.assertTrue(torch.equal(sampled["layer4"][:, 0], features["layer4"]))

    def test_wraparound_keeps_latest_entries_per_class(self):
        bank = FeatureMemoryBank(num_classes=2, max_size=2)
        bank.add(_feature_batch([10, 20, 30, 40, 50]), [0, 1, 0, 1, 0])

        self.assertEqual(len(bank), 4)
        np.testing.assert_array_equal(bank.count, np.array([2, 2], np.int32))
        np.testing.assert_array_equal(bank.ptr, np.array([1, 0], np.int32))
        self.assertEqual(
            set(bank.bank["layer3"][0, :, 0, 0, 0].float().tolist()),
            {30.0, 50.0},
        )
        self.assertEqual(
            set(bank.bank["layer3"][1, :, 0, 0, 0].float().tolist()),
            {20.0, 40.0},
        )

    def test_sampling_respects_labels_and_is_rng_deterministic(self):
        bank = FeatureMemoryBank(num_classes=3, max_size=3)
        values = [10, 11, 12, 20, 21, 22, 30, 31, 32]
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        bank.add(_feature_batch(values), labels)

        query = np.array([2, 0, 1, 2], dtype=np.int64)
        np.random.seed(123)
        expected_global = np.random.random()
        np.random.seed(123)
        first, first_indices = bank.sample(
            query,
            5,
            rng=np.random.default_rng(99),
            return_indices=True,
        )
        actual_global = np.random.random()
        second, second_indices = bank.sample(
            query,
            5,
            rng=np.random.default_rng(99),
            return_indices=True,
        )

        self.assertEqual(actual_global, expected_global)
        np.testing.assert_array_equal(first_indices, second_indices)
        for name in bank.map_names:
            self.assertTrue(torch.equal(first[name], second[name]))
        sampled_values = first["layer3"][:, :, 0, 0, 0].float()
        expected_ranges = [(30, 32), (10, 12), (20, 22), (30, 32)]
        for row, (low, high) in zip(sampled_values, expected_ranges):
            self.assertTrue(bool(((row >= low) & (row <= high)).all()))

    def test_map_names_and_shapes_are_validated(self):
        expected = {"layer3": (2, 2, 2), "layer4": (3, 1, 1)}
        bank = FeatureMemoryBank(
            num_classes=1,
            max_size=2,
            map_shapes=expected,
        )
        bank.add(_feature_batch([1]), [0])
        self.assertEqual(bank.map_names, ("layer3", "layer4"))
        self.assertEqual(bank.map_shapes, expected)

        with self.assertRaisesRegex(ValueError, "names/order"):
            bank.add(
                {
                    "layer4": torch.zeros(1, 3, 1, 1),
                    "layer3": torch.zeros(1, 2, 2, 2),
                },
                [0],
            )
        with self.assertRaisesRegex(ValueError, "shapes"):
            bank.add(
                {
                    "layer3": torch.zeros(1, 2, 2, 3),
                    "layer4": torch.zeros(1, 3, 1, 1),
                },
                [0],
            )
        with self.assertRaisesRegex(ValueError, "same batch size"):
            bank.add(
                {
                    "layer3": torch.zeros(2, 2, 2, 2),
                    "layer4": torch.zeros(1, 3, 1, 1),
                },
                [0, 0],
            )

    def test_rejects_invalid_or_empty_labels(self):
        bank = FeatureMemoryBank(num_classes=2, max_size=2)
        bank.add(_feature_batch([1]), [0])
        with self.assertRaisesRegex(RuntimeError, "empty class"):
            bank.sample([1], 1)
        with self.assertRaises(IndexError):
            bank.sample([2], 1)
        with self.assertRaises(TypeError):
            bank.add(_feature_batch([2]), [0.5])


if __name__ == "__main__":
    unittest.main()
