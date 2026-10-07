"""Synthetic cache-join regressions for optional second-pass feature flags."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import common


class FeatureCacheContracts(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({"id": [103, 101, 102], "Flight Distance": np.array([300, 100, 200], np.float32),
                                 "Class": ["Business", "Eco", "Eco"], "Type of Travel": ["Business travel"] * 3,
                                 "Gender": ["Male", "Female", "Male"], common.TARGET: [0, 1, 0]},
                                index=[91, 13, 44])
        self.aux_columns = [f"orig_aux_{i:02d}" for i in range(13)]
        self.aux = pd.DataFrame({"id": [101, 102, 103], **{
            name: np.array([1., 2., 3.], np.float32) + np.float32(j / 100)
            for j, name in enumerate(self.aux_columns)}})
        self.lgb = pd.DataFrame({"id": [102, 103, 101],
                                 "original_lgb_teacher_probability": np.array([.2, .3, .1], np.float32)})

    def reader(self, path, *args, **kwargs):
        if Path(path).name == "original_aux_predictions.parquet":
            return self.aux.copy()
        if Path(path).name == "original_lgb_teacher_predictions.parquet":
            return self.lgb.copy()
        raise AssertionError(f"Unexpected file read: {path}")

    def test_disabled_flags_preserve_legacy_features_without_cache_reads(self):
        original = self.raw.copy(deep=True)
        with patch.object(pd, "read_parquet", side_effect=AssertionError("Disabled flags must not load caches")):
            absent = common.features(self.raw, {"route": True, "categorical_twins": True})
            explicit = common.features(self.raw, {"route": True, "categorical_twins": True,
                                                  "original_aux": False, "original_lgb_teacher": False})
        pd.testing.assert_frame_equal(absent, explicit)
        pd.testing.assert_frame_equal(self.raw, original)
        self.assertNotIn("id", absent)
        self.assertNotIn(common.TARGET, absent)

    def test_join_is_by_id_with_arbitrary_index_and_fixed_float32_append_order(self):
        original = self.raw.copy(deep=True)
        with patch.object(pd, "read_parquet", side_effect=self.reader):
            result = common.features(self.raw, {"categorical_twins": True, "route": True,
                                                "original_aux": True, "original_lgb_teacher": True})
        appended = self.aux_columns + ["original_lgb_teacher_probability"]
        self.assertEqual(list(result)[-14:], appended)
        np.testing.assert_array_equal(result.index, self.raw.index)
        np.testing.assert_array_equal(result.orig_aux_00, np.array([3., 1., 2.], np.float32))
        np.testing.assert_array_equal(result.original_lgb_teacher_probability, np.array([.3, .1, .2], np.float32))
        self.assertTrue(all(result[c].dtype == np.float32 for c in appended))
        self.assertFalse(any(c + "_category" in result for c in appended), "New features append after raw category twins")
        pd.testing.assert_frame_equal(self.raw, original)

    def test_feature_flags_are_independent(self):
        for flag, expected in [("original_aux", self.aux_columns),
                               ("original_lgb_teacher", ["original_lgb_teacher_probability"])]:
            with self.subTest(flag=flag), patch.object(pd, "read_parquet", side_effect=self.reader) as reader:
                result = common.features(self.raw, {flag: True})
                self.assertEqual(reader.call_count, 1)
                self.assertEqual(list(result)[-len(expected):], expected)

    def test_missing_duplicate_nonfinite_and_wrong_cache_schema_are_rejected(self):
        valid = self.aux.copy(deep=True)
        bad_frames = {"missing_id": valid.iloc[:2],
                      "duplicate_id": pd.concat([valid, valid.iloc[:1]], ignore_index=True),
                      "column_order": valid[["id"] + self.aux_columns[::-1]],
                      "extra_column": valid.assign(unexpected=0)}
        nonfinite = valid.copy()
        nonfinite.loc[0, "orig_aux_00"] = np.inf
        bad_frames["nonfinite"] = nonfinite
        for name, frame in bad_frames.items():
            with self.subTest(name=name), patch.object(pd, "read_parquet", return_value=frame):
                with self.assertRaises(ValueError):
                    common.features(self.raw, {"original_aux": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
