"""Synthetic, feature-only regression checks; no competition data is loaded."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from route_profiles import DEFAULT_COLUMNS, RouteProfiles


class RouteProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {"key": "Flight Distance", "columns": ["Age", "delay", "all_missing"]}
        self.x = pd.DataFrame({
            "Flight Distance": [1.0, 1.0, 2.0, 2.0, np.nan],
            "Age": [20.0, 40.0, 80.0, np.nan, 100.0],
            "delay": [1.0, np.nan, np.nan, np.nan, 5.0],
            "all_missing": [np.nan] * 5,
            "untouched": ["a", "b", "c", "d", "e"],
        }, index=[9, 3, 9, 1, 5])

    def test_means_missing_fallback_and_identity(self) -> None:
        original = self.x.copy(deep=True)
        model = RouteProfiles(self.config)
        actual = model.fit_transform(self.x)
        np.testing.assert_array_equal(model.global_means_, [60.0, 3.0, 0.0])
        np.testing.assert_array_equal(actual[model.output_columns].to_numpy(), [
            [30, 1, 0], [30, 1, 0], [80, 3, 0], [80, 3, 0], [60, 3, 0],
        ])
        assert_frame_equal(self.x, original)
        assert_frame_equal(actual.iloc[:, :len(self.x.columns)], self.x)
        self.assertTrue(all(dtype == np.dtype("float32") for dtype in actual[model.output_columns].dtypes))

    def test_validation_extremes_and_unseen_keys_do_not_fit(self) -> None:
        model = RouteProfiles(self.config).fit(self.x)
        stats = model.map_.copy(deep=True)
        prior = model.global_means_.copy()
        validation = self.x.iloc[[1, 2, 0]].copy()
        validation["Flight Distance"] = [99.0, np.nan, 1.0]
        validation["Age"] = [1e20, -1e20, 0.0]
        values = model.transform(validation)[model.output_columns].to_numpy()
        np.testing.assert_array_equal(values, [[60, 3, 0], [60, 3, 0], [30, 1, 0]])
        assert_frame_equal(model.map_, stats)
        np.testing.assert_array_equal(model.global_means_, prior)
        self.assertNotIn(99, model.map_.index)

    def test_numeric_csv_and_float32_keys_match(self) -> None:
        # 16777217 intentionally rounds to 16777216 at the agreed key precision.
        raw = pd.read_csv(io.StringIO("Flight Distance,Age,delay,all_missing,untouched\n16777217,20,1,,a\n1000,30,2,,b\n"))
        model = RouteProfiles(self.config).fit(raw)
        converted = raw.copy()
        converted["Flight Distance"] = converted["Flight Distance"].astype(np.float32)
        np.testing.assert_array_equal(
            model.transform(raw)[model.output_columns], model.transform(converted)[model.output_columns],
        )
        nullable = converted.copy()
        nullable["Flight Distance"] = pd.array([16777216.0, None], dtype="Float32")
        actual = model.transform(nullable)[model.output_columns].to_numpy()
        np.testing.assert_array_equal(actual, [[20, 1, 0], [25, 1.5, 0]])

    def test_save_load_empty_and_row_permutation(self) -> None:
        model = RouteProfiles(self.config).fit(self.x)
        with tempfile.TemporaryDirectory(prefix="route-profiles-synthetic-") as directory:
            model.save(directory)
            loaded = RouteProfiles.load(directory)
            for x in (self.x, self.x.iloc[[4, 2, 0, 2]], self.x.iloc[:0]):
                assert_frame_equal(loaded.transform(x), model.transform(x))
            metadata = json.loads((Path(directory) / "metadata.json").read_text())
            map_path = Path(directory) / metadata["map_file"]
            with map_path.open("ab") as stream:
                stream.write(b"invalid")
            with self.assertRaisesRegex(ValueError, "checksum"):
                RouteProfiles.load(directory)

    def test_all_missing_routes_and_default_schema(self) -> None:
        x = pd.DataFrame({"Flight Distance": [np.nan, np.nan]})
        for i, name in enumerate(DEFAULT_COLUMNS):
            x[name] = [float(i), float(i + 2)]
        model = RouteProfiles().fit(x)
        self.assertEqual(len(model.output_columns), 16)
        self.assertEqual(len(model.map_), 0)
        np.testing.assert_array_equal(model.transform(x)[model.output_columns], [np.arange(1, 17)] * 2)
        with tempfile.TemporaryDirectory(prefix="route-profiles-empty-map-") as directory:
            model.save(directory)
            assert_frame_equal(RouteProfiles.load(directory).transform(x), model.transform(x))

    def test_contract_rejections(self) -> None:
        with self.assertRaisesRegex(ValueError, "target"):
            RouteProfiles({"columns": ["satisfaction"]})
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            RouteProfiles(self.config).transform(self.x)
        with self.assertRaisesRegex(ValueError, "zero rows"):
            RouteProfiles(self.config).fit(self.x.iloc[:0])
        model = RouteProfiles(self.config).fit(self.x)
        with self.assertRaisesRegex(ValueError, "order"):
            model.transform(self.x[self.x.columns[::-1]])
        invalid = self.x.copy()
        invalid["Flight Distance"] = np.inf
        with self.assertRaisesRegex(ValueError, "infinity"):
            model.transform(invalid)
        collision = self.x.copy()
        collision[model.output_columns[0]] = 0.0
        with self.assertRaisesRegex(ValueError, "already exist"):
            model.transform(collision)


if __name__ == "__main__":
    unittest.main()
