"""Synthetic independent checks of exact multiplicity-weighted bootstrap AUC."""
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from analysis_auc_bootstrap_v1 import WeightedAUC, class_stratified_counts


class WeightedAUCTest(unittest.TestCase):
    def test_counted_auc_matches_expanded_rows_and_sklearn_weights(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        score = np.array([.3, .3, .4, .4, .4, .8, .8, .9])
        structure = WeightedAUC.prepare(y, score)
        fixtures = [np.ones(8, dtype=int), np.array([0, 2, 1, 0, 1, 2, 2, 0]),
                    np.array([4, 4, 0, 0, 0, 0, 0, 0])]
        for counts in fixtures:
            expected = roc_auc_score(np.repeat(y, counts), np.repeat(score, counts))
            self.assertAlmostEqual(structure.calculate(counts), expected, places=15)
            self.assertAlmostEqual(structure.calculate(counts),
                                   roc_auc_score(y, score, sample_weight=counts), places=15)

    def test_random_replicates_match_expanded_oracle(self):
        rng = np.random.default_rng(15)
        y = np.r_[np.ones(13), np.zeros(17)]
        scores = [rng.integers(0, 5, size=len(y)), rng.normal(size=len(y)), np.ones(len(y))]
        for score in scores:
            structure = WeightedAUC.prepare(y, score)
            for _ in range(10):
                counts = class_stratified_counts(y, rng)
                self.assertEqual(counts[y == 1].sum(), 13)
                self.assertEqual(counts[y == 0].sum(), 17)
                self.assertAlmostEqual(structure.calculate(counts),
                                       roc_auc_score(np.repeat(y, counts), np.repeat(score, counts)),
                                       places=15)

    def test_seed_reproducibility(self):
        y = np.r_[np.ones(23), np.zeros(29)]
        np.testing.assert_array_equal(
            class_stratified_counts(y, np.random.default_rng(20261002)),
            class_stratified_counts(y, np.random.default_rng(20261002)))

    def test_invalid_count_vectors_rejected(self):
        structure = WeightedAUC.prepare(np.array([0, 1, 0, 1]), np.arange(4))
        for counts in (np.ones(3), np.array([1, -1, 1, 1]), np.full(4, np.nan),
                       np.array([1, 0, 1, 0])):
            with self.assertRaises(ValueError):
                structure.calculate(counts)


if __name__ == "__main__":
    unittest.main()
