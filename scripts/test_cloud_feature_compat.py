"""Synthetic pandas transport semantics; no models, data access, or fitting."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import cloud_feature_compat as compat


class CloudFeatureCompatTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({c: np.array([1, np.nan, 9876], dtype=np.float32)
                                 for c in compat.NUMERIC_COLUMNS}, index=[7, 2, 9])
        self.features = self.raw.copy()
        for c in compat.NUMERIC_COLUMNS:
            self.features[c + "_category"] = self.raw[c].astype(str)
        self.features["literal_category"] = ["nan", "__MISSING__", "new"]
        self.run = {"execution_backend": "kaggle", "categorical_twins": True}

    def apply(self):
        return compat.cloud_numeric_category_compat(self.features, self.raw, run=self.run,
                                                    source_versions=compat.SOURCE_VERSIONS)

    def test_missing_tokens_finite_values_literals_dtypes_and_order(self):
        original = self.features.copy(deep=True)
        result = self.apply()
        self.assertEqual(list(result.columns), list(original.columns))
        self.assertTrue(result.index.equals(original.index))
        pd.testing.assert_frame_equal(self.features, original)
        pd.testing.assert_frame_equal(result[list(compat.NUMERIC_COLUMNS)], self.raw)
        pd.testing.assert_series_equal(result.literal_category, original.literal_category)
        for c in compat.NUMERIC_COLUMNS:
            twin = c + "_category"
            self.assertEqual(result[twin].tolist(), ["1.0", "nan", "9876.0"])
            self.assertEqual(result[twin].dtype, original[twin].dtype)
            # No vocabulary expansion: a later trained mapper still rejects 9876.
            self.assertEqual(pd.Index(["1.0", "nan"]).get_indexer(result[twin]).tolist(), [0, 1, -1])

    def test_idempotent_on_already_legacy_tokens(self):
        self.features = self.apply()
        pd.testing.assert_frame_equal(self.apply(), self.features)

    def test_batch_and_row_order_invariance(self):
        expected = self.apply()
        parts = [compat.cloud_numeric_category_compat(self.features.iloc[indices], self.raw.iloc[indices],
                    run=self.run, source_versions=compat.SOURCE_VERSIONS)
                 for indices in [[2], [0, 1]]]
        pd.testing.assert_frame_equal(pd.concat(parts).loc[expected.index], expected)

    def test_rejects_wrong_source_or_inference_platform_and_noncloud_run(self):
        with self.assertRaisesRegex(ValueError, "source"):
            compat.cloud_numeric_category_compat(self.features, self.raw, run=self.run,
                                                 source_versions={"pandas": "3.0.6", "numpy": "2.5.3"})
        with patch.object(compat.pd, "__version__", "4.0.0"):
            with self.assertRaisesRegex(ValueError, "inference"):
                self.apply()
        self.run["execution_backend"] = "local"
        with self.assertRaisesRegex(ValueError, "cloud"):
            self.apply()

    def test_rejects_wrong_finite_spelling_unknown_sentinel_and_raw_string(self):
        self.features.loc[7, "Age_category"] = "1"
        with self.assertRaisesRegex(ValueError, "spelling"):
            self.apply()
        self.features.loc[7, "Age_category"] = "1.0"
        self.features.loc[2, "Age_category"] = "__MISSING__"
        with self.assertRaisesRegex(ValueError, "token"):
            self.apply()
        self.features.loc[2, "Age_category"] = "nan"
        self.raw["Age"] = self.raw.Age.astype(str)
        with self.assertRaisesRegex(ValueError, "float32"):
            self.apply()

    def test_rejects_changed_index_schema_or_infinity(self):
        self.raw.index = [2, 7, 9]
        with self.assertRaisesRegex(ValueError, "index"):
            self.apply()
        self.raw.index = [7, 2, 9]
        self.raw.loc[7, "Age"] = np.inf
        with self.assertRaisesRegex(ValueError, "Infinite"):
            self.apply()
        self.raw.loc[7, "Age"] = 1
        self.features = self.features.drop(columns="Age_category")
        with self.assertRaisesRegex(ValueError, "Missing canonical"):
            self.apply()


if __name__ == "__main__":
    unittest.main()
