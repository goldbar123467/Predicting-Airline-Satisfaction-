"""Registered duration contrasts reject recipe drift and unstable gains."""
import copy
import unittest

from third_pass_release import duration_comparison


class DurationGateTests(unittest.TestCase):
    def setUp(self):
        self.control = dict(id='control', family='realmlp_cat', max_rounds=4,
                            seed=123, params={'lr': .05}, timeout_seconds=600)
        self.candidate = {**copy.deepcopy(self.control), 'id': 'candidate',
                          'max_rounds': 60, 'timeout_seconds': 3000}
        self.before = dict(pooled=.8, mean_fold=.8, folds=[.8, .8, .8])

    def test_same_recipe_positive_or_tied(self):
        self.assertTrue(duration_comparison(self.candidate, self.control,
                                            self.before, self.before))

    def test_negative_pooled_or_macro(self):
        for key in ['pooled', 'mean_fold']:
            after = {**self.before, key: .79999}
            self.assertFalse(duration_comparison(self.candidate, self.control, after, self.before))

    def test_fold_regression(self):
        after = dict(pooled=.81, mean_fold=.81, folds=[.79997, .82, .82])
        self.assertFalse(duration_comparison(self.candidate, self.control, after, self.before))

    def test_recipe_seed_and_horizon_drift(self):
        for key, value in [('seed', 456), ('max_rounds', 12), ('params', {'lr': .01})]:
            after = {**self.candidate, key: value}
            with self.assertRaises(ValueError):
                duration_comparison(after, self.control, self.before, self.before)


if __name__ == '__main__':
    unittest.main()
