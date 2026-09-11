import unittest

import numpy as np

from scripts.eval_official_imagenet256 import _within_class_feature_diversity


class ClassDiversityTest(unittest.TestCase):
    def test_matches_explicit_pairwise_distance_and_equal_class_weighting(self):
        features = np.array([[0, 0], [3, 0], [10, 0], [10, 4], [10, 8]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1, 1])
        result = _within_class_feature_diversity(features, labels)
        # Class 0: 9; class 1: (16 + 64 + 16) / 3 = 32.
        self.assertAlmostEqual(result["mean_pairwise_squared_distance"], 20.5)
        self.assertEqual(result["num_classes"], 2)
        self.assertEqual(result["min_samples_per_class"], 2)

    def test_class_translation_invariance_and_singleton_exclusion(self):
        labels = np.array([0, 0, 1, 1, 2])
        features = np.array([[0, 0], [3, 0], [0, 0], [0, 4], [99, 99]], dtype=np.float64)
        before = _within_class_feature_diversity(features, labels)
        features[labels == 0] += 100
        features[labels == 1] -= 200
        self.assertEqual(before, _within_class_feature_diversity(features, labels))
        self.assertAlmostEqual(before["mean_pairwise_squared_distance"], 12.5)
        self.assertEqual(before["num_classes"], 2)

    def test_identical_samples_have_zero_diversity(self):
        result = _within_class_feature_diversity(np.ones((3, 2)), np.zeros(3))
        self.assertEqual(result["mean_pairwise_squared_distance"], 0.0)

    def test_rejects_misaligned_inputs_and_no_pairs(self):
        with self.assertRaises(ValueError):
            _within_class_feature_diversity(np.ones((3, 2)), np.zeros(2))
        with self.assertRaises(ValueError):
            _within_class_feature_diversity(np.ones((3, 2)), np.arange(3))


if __name__ == "__main__":
    unittest.main()
