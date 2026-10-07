"""Synthetic numerical contracts for fixed-grid joint AUC uncertainty."""
import unittest

import numpy as np

from analysis_auc_uncertainty_v1 import paired_auc, placements
from analysis_grid_uncertainty_v1 import gaussian_summary, placement_covariance, psd_factor


def differences(y, reference, scores):
    _, positive, negative = placements(y, reference)
    parts = [placements(y, score) for score in scores]
    return (np.column_stack([part[1] - positive for part in parts]),
            np.column_stack([part[2] - negative for part in parts]))


class JointCovarianceTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 1, 0, 0, 1, 0, 1])
        self.reference = np.array([.2, .7, .5, .3, .8, .9, .1, .4])
        self.scores = [np.array([.4, .7, .5, .2, .8, .6, .4, .9]),
                       np.array([.7, .3, .5, .6, .1, .9, .2, .8]),
                       np.ones(8) * .5]

    def test_diagonal_matches_independent_paired_standard_errors(self):
        for scores in (self.scores, [1 - score for score in self.scores]):
            cov = placement_covariance(*differences(self.y, self.reference, scores))
            expected = [paired_auc(self.y, score, self.reference)["paired_standard_error"] ** 2
                        for score in scores]
            np.testing.assert_allclose(np.diag(cov), expected, atol=1e-16, rtol=1e-13)

    def test_brute_pair_covariance(self):
        wins = lambda score: (np.sign(score[self.y == 1, None] - score[self.y == 0]) + 1) / 2
        reference = wins(self.reference)
        positive = np.column_stack([(wins(s) - reference).mean(axis=1) for s in self.scores])
        negative = np.column_stack([(wins(s) - reference).mean(axis=0) for s in self.scores])
        expected = np.cov(positive, rowvar=False, ddof=1) / 4 + np.cov(negative, rowvar=False, ddof=1) / 4
        actual = placement_covariance(*differences(self.y, self.reference, self.scores))
        np.testing.assert_allclose(actual, expected, atol=1e-16, rtol=1e-13)

    def test_row_and_column_permutation(self):
        row_order = np.array([7, 2, 1, 4, 5, 6, 3, 0])
        column_order = [2, 0, 1]
        original = placement_covariance(*differences(self.y, self.reference, self.scores))
        permuted = placement_covariance(*differences(
            self.y[row_order], self.reference[row_order], [self.scores[i][row_order] for i in column_order]))
        np.testing.assert_allclose(permuted, original[np.ix_(column_order, column_order)], atol=1e-16)

    def test_swapped_comparison_signs_preserve_covariance(self):
        positive, negative = differences(self.y, self.reference, self.scores)
        actual = placement_covariance(positive, negative)
        np.testing.assert_array_equal(placement_covariance(-positive, -negative), actual)
        signs = np.array([-1, 1, -1])
        np.testing.assert_array_equal(placement_covariance(positive * signs, negative * signs),
                                      actual * signs[:, None] * signs[None, :])

    def test_identical_and_zero_contrasts(self):
        positive, negative = differences(self.y, self.reference,
                                         [self.scores[0], self.scores[0], self.reference])
        covariance = placement_covariance(positive, negative)
        np.testing.assert_array_equal(covariance[:, 0], covariance[:, 1])
        np.testing.assert_array_equal(covariance[:, 2], np.zeros(3))
        result = gaussian_summary(covariance, np.array([10., 10., -10.]), np.array([1, 0, 2]),
                                  seed=10, draws=200)
        self.assertEqual(result["winner_counts"], [0, 200, 0])
        self.assertEqual(result["zero_standard_error_columns"], [2])
        self.assertEqual(result["exact_identical_covariance_columns_forced_to_share_noise"], [[1, 0]])

    def test_invalid_placement_matrices(self):
        for positive, negative in ((np.zeros((1, 2)), np.zeros((3, 2))),
                                   (np.zeros((3, 2)), np.zeros((3, 3))),
                                   (np.zeros((3, 0)), np.zeros((3, 0))),
                                   (np.array([[np.nan], [0]]), np.zeros((3, 1)))):
            with self.assertRaises(ValueError):
                placement_covariance(positive, negative)


class GaussianTests(unittest.TestCase):
    def test_fixed_seed_and_psd_reconstruction(self):
        covariance = np.array([[2., .5], [.5, 1.]])
        factor, metadata = psd_factor(covariance)
        np.testing.assert_allclose(factor @ factor.T, covariance, atol=1e-14)
        self.assertEqual(metadata["negative_eigenvalues_clipped"], 0)
        kwargs = dict(seed=20261003, draws=20000)
        first = gaussian_summary(covariance, np.zeros(2), np.arange(2), **kwargs)
        second = gaussian_summary(covariance, np.zeros(2), np.arange(2), **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(sum(first["winner_counts"]), 20000)
        self.assertGreater(first["simultaneous_95_critical_value"], 2.0)
        self.assertLess(first["simultaneous_95_critical_value"], 2.5)

    def test_material_non_psd_and_asymmetry_rejected(self):
        for covariance in (np.array([[1., 1.01], [1.01, 1.]]),
                           np.array([[1., .2], [.3, 1.]]),
                           np.array([[np.inf]]), np.array([])):
            with self.assertRaises(ValueError):
                psd_factor(covariance)

    def test_only_roundoff_negative_eigenvalue_clipped(self):
        covariance = np.array([[1., 1. + 1e-15], [1. + 1e-15, 1.]])
        factor, metadata = psd_factor(covariance)
        self.assertEqual(metadata["negative_eigenvalues_clipped"], 1)
        self.assertLessEqual(metadata["maximum_eigenvalue_correction"], metadata["negative_eigenvalue_tolerance"])
        self.assertEqual(metadata["diagonal_jitter_added"], 0)
        np.testing.assert_allclose(factor @ factor.T, covariance, atol=3e-15)
        scaled_factor, scaled = psd_factor(covariance * 1e-20)
        self.assertLess(scaled["negative_eigenvalue_tolerance"], 1e-32)
        np.testing.assert_allclose(scaled_factor @ scaled_factor.T, covariance * 1e-20, atol=4e-35)

    def test_all_zero_covariance_and_fixed_tie_policy(self):
        result = gaussian_summary(np.zeros((3, 3)), np.ones(3), np.array([2, 0, 1]), seed=2, draws=30)
        self.assertEqual(result["simultaneous_95_critical_value"], 0)
        self.assertEqual(result["winner_counts"], [0, 0, 30])
        self.assertEqual(result["draws_with_multiple_eligible_winners"], 30)
        self.assertEqual(result["zero_standard_error_columns"], [0, 1, 2])

    def test_invalid_gaussian_contracts(self):
        for delta, order, draws in ((np.zeros(2), np.array([0, 0]), 20),
                                   (np.zeros(2), np.array([0., 1.]), 20),
                                   (np.zeros(2), np.arange(2), 1),
                                   (np.array([np.nan, 0]), np.arange(2), 20)):
            with self.assertRaises(ValueError):
                gaussian_summary(np.eye(2), delta, order, seed=1, draws=draws)


if __name__ == "__main__":
    unittest.main()
