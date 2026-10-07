"""Generated-data regression contracts for full ensemble selection nesting."""
import unittest
import numpy as np

from research_nested_blend_contract_v1 import (
    ToyData, cross_fitted_features, fit_base, generated_data, perturb_features,
    perturb_labels, regenerate_nested, run_proof, split_existing_oof,
)


class NestedDataflowTests(unittest.TestCase):
    def setUp(self):
        self.data = generated_data()
        self.protected = frozenset(int(value) for value in self.data.ids[self.data.outer_fold == 0])

    def test_old_oof_is_row_held_out_but_meta_features_depend_on_outer_labels(self):
        table = cross_fitted_features(self.data, self.data.ids, self.data.outer_fold)
        for row, used in zip(table.ids, table.source_training_ids):
            self.assertNotIn(int(row), used)
        old = split_existing_oof(self.data, 0)
        changed = split_existing_oof(perturb_labels(self.data, 0), 0)
        self.assertTrue(all(self.protected.intersection(ids) for ids in old.meta_feature_sources))
        self.assertGreater(np.max(np.abs(old.meta_features - changed.meta_features)), 0)
        self.assertGreater(np.max(np.abs(old.heldout_predictions - changed.heldout_predictions)), 0)
        np.testing.assert_array_equal(old.heldout_base_scores, changed.heldout_base_scores)

    def test_one_sided_outer_label_perturbation_leaves_nested_fitted_objects_invariant(self):
        before = regenerate_nested(self.data, 0)
        after = regenerate_nested(perturb_labels(self.data, 0), 0)
        np.testing.assert_array_equal(before.meta_features, after.meta_features)
        np.testing.assert_array_equal(before.outer_model.feature_mean, after.outer_model.feature_mean)
        self.assertEqual(before.outer_model.label_mean, after.outer_model.label_mean)
        self.assertEqual(before.meta_model, after.meta_model)
        np.testing.assert_array_equal(before.heldout_predictions, after.heldout_predictions)

    def test_heldout_features_do_not_fit_nested_preprocessing_but_can_change_query_predictions(self):
        before = regenerate_nested(self.data, 0)
        after = regenerate_nested(perturb_features(self.data, 0), 0)
        np.testing.assert_array_equal(before.meta_features, after.meta_features)
        np.testing.assert_array_equal(before.meta_feature_means, after.meta_feature_means)
        np.testing.assert_array_equal(before.outer_model.feature_mean, after.outer_model.feature_mean)
        self.assertGreater(np.max(np.abs(before.heldout_predictions - after.heldout_predictions)), 0)

    def test_original_oof_preprocessing_contains_future_outer_features(self):
        before = split_existing_oof(self.data, 0)
        after = split_existing_oof(perturb_features(self.data, 0), 0)
        self.assertGreater(np.max(np.abs(before.meta_feature_means - after.meta_feature_means)), 0)

    def test_all_nested_fit_lineage_excludes_protected_population(self):
        result = regenerate_nested(self.data, 0)
        for fit_ids in (*result.meta_feature_sources, result.meta_model.training_ids,
                        result.outer_model.training_ids, result.outer_model.preprocessing_ids):
            self.assertFalse(self.protected.intersection(fit_ids))
        for row, source_ids in zip(result.training_ids, result.meta_feature_sources):
            self.assertNotIn(int(row), source_ids)

    def test_protected_fit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Protected"):
            fit_base(self.data, self.data.ids, self.protected)

    def test_nested_meta_features_differ_from_full_outer_model_in_sample_scores(self):
        result = regenerate_nested(self.data, 0)
        fitted_scores = result.outer_model.score(self.data.x[self.data.positions(result.training_ids)])
        self.assertGreater(np.max(np.abs(fitted_scores - result.meta_features)), 0)

    def test_keyed_inner_partition_and_predictions_survive_input_row_permutation(self):
        order = np.random.default_rng(17).permutation(len(self.data.ids))
        permuted = ToyData(self.data.ids[order], self.data.x[order], self.data.y[order], self.data.outer_fold[order])
        before, after = regenerate_nested(self.data, 0), regenerate_nested(permuted, 0)
        before_index, after_index = np.argsort(before.heldout_ids), np.argsort(after.heldout_ids)
        np.testing.assert_allclose(before.heldout_predictions[before_index], after.heldout_predictions[after_index], atol=2e-16, rtol=0)

    def test_unknown_duplicate_and_invalid_data_are_rejected(self):
        for ids in (np.array([111111]), np.array([self.data.ids[0], self.data.ids[0]])):
            with self.assertRaises(ValueError):
                fit_base(self.data, ids)
        with self.assertRaises(ValueError):
            ToyData(self.data.ids, self.data.x, np.full(36, 2), self.data.outer_fold)

    def test_executable_proof_checks_all_invariants_without_files(self):
        proof = run_proof()
        self.assertTrue(all(proof["invariants"].values()))
        self.assertEqual(proof["old_meta_rows_whose_feature_model_used_outer_ids"], 24)
        self.assertEqual(proof["nested_meta_rows_whose_feature_model_used_outer_ids"], 0)
        self.assertTrue(proof["no_performance_metric_computed"])


if __name__ == "__main__":
    unittest.main()
