"""Synthetic tests only. No competition files or model fitting."""
from __future__ import annotations

import unittest
import numpy as np
from sklearn.metrics import roc_auc_score

from analysis_auc_uncertainty_v1 import paired_auc, placements


def brute(y, score):
    differences = score[y == 1, None] - score[None, y == 0]
    win = (differences > 0).astype(float) + 0.5 * (differences == 0)
    return float(win.mean()), win.mean(axis=1), win.mean(axis=0)


class ExactPairedAUCTest(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        self.a = np.array([0.1, 0.4, 0.4, 0.4, 0.8, 0.6, 0.9, 0.9])
        self.b = np.array([0.3, 0.3, 0.2, 0.7, 0.6, 0.6, 0.9, 0.8])

    def test_tied_placements_match_brute_force_and_sklearn(self):
        for score in (self.a, self.b, np.ones(8), self.y, 1 - self.y):
            measured = placements(self.y, score)
            expected = brute(self.y, np.asarray(score))
            self.assertAlmostEqual(measured[0], roc_auc_score(self.y, score), places=15)
            for actual, reference in zip(measured, expected):
                np.testing.assert_array_equal(actual, reference)

    def test_random_discrete_and_continuous_against_pair_oracle(self):
        rng = np.random.default_rng(20261002)
        y = np.r_[np.ones(39), np.zeros(57)]
        for score in (rng.integers(0, 7, len(y)), rng.normal(size=len(y))):
            measured, expected = placements(y, score), brute(y, score)
            self.assertAlmostEqual(measured[0], roc_auc_score(y, score), places=15)
            for actual, reference in zip(measured, expected):
                np.testing.assert_allclose(actual, reference, atol=2e-16, rtol=0)

    def test_covariance_matches_independent_brute_formula(self):
        result = paired_auc(self.y, self.a, self.b)
        left, right = brute(self.y, self.a), brute(self.y, self.b)
        covariance = (np.cov(np.stack([left[1], right[1]]), ddof=1) / 4
                      + np.cov(np.stack([left[2], right[2]]), ddof=1) / 4)
        np.testing.assert_allclose(result["paired_covariance"], covariance, atol=1e-16)
        contrast = np.array([1.0, -1.0])
        self.assertAlmostEqual(result["paired_standard_error"] ** 2,
                               float(contrast @ covariance @ contrast), places=15)

    def test_identical_scores_have_exact_zero_difference_and_variance(self):
        result = paired_auc(self.y, self.a, self.a.copy())
        self.assertEqual(result["difference_reference_minus_comparator"], 0)
        self.assertEqual(result["paired_standard_error"], 0)
        self.assertEqual(result["descriptive_conditional_interval_95"], [0, 0])

    def test_all_tied_scores_have_auc_half_and_zero_variance(self):
        result = paired_auc(self.y, np.ones(8), np.zeros(8))
        self.assertEqual(result["reference_auc"], 0.5)
        self.assertEqual(result["comparator_auc"], 0.5)
        self.assertEqual(result["paired_standard_error"], 0)

    def test_reversal_reverses_difference_and_interval(self):
        ab, ba = paired_auc(self.y, self.a, self.b), paired_auc(self.y, self.b, self.a)
        self.assertEqual(ab["difference_reference_minus_comparator"],
                         -ba["difference_reference_minus_comparator"])
        self.assertEqual(ab["paired_standard_error"], ba["paired_standard_error"])
        np.testing.assert_allclose(ab["descriptive_conditional_interval_95"],
                                   -np.asarray(ba["descriptive_conditional_interval_95"])[::-1])

    def test_row_permutation_preserves_results(self):
        order = np.array([6, 3, 1, 7, 0, 5, 2, 4])
        a = paired_auc(self.y, self.a, self.b)
        b = paired_auc(self.y[order], self.a[order], self.b[order])
        self.assertAlmostEqual(a["difference_reference_minus_comparator"],
                               b["difference_reference_minus_comparator"], places=15)
        self.assertAlmostEqual(a["paired_standard_error"], b["paired_standard_error"], places=15)

    def test_invalid_inputs_are_rejected(self):
        fixtures = [(self.y[:-1], self.a), (self.y[:, None], self.a),
                    (np.array([0, 0, 1, 2]), np.ones(4)),
                    (np.array([0, 0, 0, 1]), np.ones(4)),
                    (self.y, np.full(8, np.nan)), (self.y, np.full(8, np.inf))]
        for y, score in fixtures:
            with self.subTest(y=y, score=score), self.assertRaises(ValueError):
                placements(y, score)


if __name__ == "__main__":
    unittest.main()
