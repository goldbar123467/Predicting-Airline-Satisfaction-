"""Synthetic regression checks for diagnostic alignment and bounded selection."""
import unittest
import numpy as np
import pandas as pd
from analysis_blend_diagnostics_v1 import aligned_oof, choose_grid, make_units, mix, remove_unit


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.asarray(["01", "02", "03", "04"])
        self.fold = np.asarray([0, 1, 2, 0])
        self.y = np.asarray([0, 1, 0, 1])
        self.frame = pd.DataFrame(dict(id=self.ids, fold=self.fold, satisfaction=self.y, prediction=[.1, .8, .2, .7]))

    def test_key_alignment_preserves_identifier_strings(self):
        p, y = aligned_oof(self.frame.iloc[[2, 0, 3, 1]], self.ids, self.fold, self.y)
        np.testing.assert_array_equal(y, self.y)
        np.testing.assert_array_equal(p, [.1, .8, .2, .7])

    def test_misaligned_duplicate_target_fold_and_probability_rejected(self):
        for field, value in (("id", "02"), ("fold", -1), ("satisfaction", 1), ("prediction", np.nan)):
            frame = self.frame.copy()
            frame.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                aligned_oof(frame, self.ids, self.fold, self.y)

    def test_missing_id_rejected(self):
        frame = self.frame.copy()
        frame.loc[0, "id"] = "other"
        with self.assertRaises(ValueError):
            aligned_oof(frame, self.ids, self.fold, self.y)

    def test_seed_group_ablation_keeps_pair_together(self):
        predictions = {"a": np.array([.1, .9]), "b": np.array([.2, .8]), "c": np.array([.4, .6])}
        groups = make_units(["a", "b", "c"], [["a", "b"]])
        self.assertEqual(list(groups.values()), [["a", "b"], ["c"]])
        p, mass = remove_unit(predictions, {"a": .2, "b": .2, "c": .6}, ["a", "b"])
        np.testing.assert_array_equal(p, predictions["c"])
        self.assertAlmostEqual(mass, .4)
        with self.assertRaises(ValueError):
            make_units(["a", "b", "c"], [["a", "b"], ["b", "c"]])
        with self.assertRaises(ValueError):
            make_units(["a", "b", "c"], [["a", "a"]])

    def test_invalid_convex_weights_rejected(self):
        p = {"a": np.array([.1]), "b": np.array([.9])}
        for w in ({"a": -.1, "b": 1.1}, {"a": .4, "b": .4}, {}):
            with self.assertRaises(ValueError):
                mix(p, w)

    def test_excluded_fold_cannot_choose_grid_point(self):
        rows = [{"weights": [.7, .1, .2], "metrics": {"fold_auc": [.9, .9, .5]}},
                {"weights": [.6, .2, .2], "metrics": {"fold_auc": [.8, .8, 1.]}}]
        selected = choose_grid(rows, [0, 1], np.array([.64, .16, .2]))
        self.assertIs(selected, rows[0])
        rows[1]["metrics"]["fold_auc"][2] = .1
        self.assertIs(choose_grid(rows, [0, 1], np.array([.64, .16, .2])), rows[0])

    def test_tie_prefers_nearest_incumbent(self):
        rows = [{"weights": [.8, .1, .1], "metrics": {"fold_auc": [.9, .9, .9]}},
                {"weights": [.65, .15, .2], "metrics": {"fold_auc": [.9, .9, .9]}}]
        self.assertIs(choose_grid(rows, [0, 1], np.array([.64, .16, .2])), rows[1])


if __name__ == "__main__":
    unittest.main()
