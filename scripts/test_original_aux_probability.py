"""CPU-native tests using only generated feature rows and generated rating labels."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost as xgb

from original_aux import CATEGORICAL, RAW_FEATURES, RATINGS, fit_bank, load_bank
from original_aux_probability import EPSILON, OUTPUT_COLUMNS, _log_score, _lookup, _publish, build, predict_aux_probability


def fixture() -> pd.DataFrame:
    rng = np.random.default_rng(20261030)
    raw = pd.DataFrame(index=np.arange(129)[::-1])
    for name in RAW_FEATURES:
        if name in CATEGORICAL:
            raw[name] = rng.choice(["a", "b", "c"], len(raw))
        elif name in RATINGS:
            support = [0., 2., 5.] if RATINGS.index(name) % 2 else [1., 3., 5.]
            raw[name] = rng.choice(support, len(raw)).astype(np.float32)
        else:
            raw[name] = rng.normal(20, 4, len(raw)).astype(np.float32)
    raw["id"] = np.arange(len(raw)) + 500
    raw["satisfaction"] = "unused poison target"
    return raw


class ProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="aux-probability-synthetic-")
        cls.directory = Path(cls.temp.name) / "bank"
        cls.raw = fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            fit_bank(cls.raw, cls.directory, rounds=3, n_jobs=2, device="cpu",
                     provenance={"fixture": "generated feature rows only"})
        cls.snapshot = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in cls.directory.iterdir()}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_explicit_nonconsecutive_class_lookup_and_log_clip(self) -> None:
        probability = np.asarray([[.2, .3, .5], [0, .9, .1], [.1, .2, .7], [.4, .3, .3]], dtype=np.float32)
        q, counts = _lookup(probability, np.array([5., 1., np.nan, 999.]), np.array([1., 3., 5.]))
        np.testing.assert_array_equal(q, np.asarray([.5, 0., EPSILON, EPSILON], dtype=np.float32))
        self.assertEqual(counts, {"missing_rows": 1, "unseen_rows": 1, "supported_log_clipped_rows": 1})
        matrix = np.tile(q[:, None], (1, 13))
        expected = np.sum(np.log(np.maximum(matrix.astype(np.float64), EPSILON)), axis=1, dtype=np.float64).astype(np.float32)
        np.testing.assert_array_equal(_log_score(matrix), expected)

    def test_native_probabilities_exact_saved_class_mapping(self) -> None:
        query = self.raw.iloc[:17]
        actual = predict_aux_probability(query, self.directory, batch_size=7, n_jobs=2)
        processor, manifest = load_bank(self.directory)
        encoded = processor.transform(query)
        for i, entry in enumerate(manifest["models"]):
            self.assertNotIn(entry["target"], entry["predictors"])
            self.assertNotIn("satisfaction", entry["predictors"])
            self.assertNotIn("id", entry["predictors"])
            model = xgb.Booster(model_file=self.directory / entry["model_file"])
            model.set_param({"device": "cpu", "nthread": 2})
            matrix = xgb.DMatrix(encoded[:, entry["predictor_positions"]], feature_names=[f"f{j}" for j in range(20)], nthread=2)
            probability = model.predict(matrix, strict_shape=True)
            lookup = {value: j for j, value in enumerate(entry["classes"])}
            expected = np.asarray([probability[r, lookup[v]] for r, v in enumerate(query[entry["target"]])], dtype=np.float32)
            np.testing.assert_array_equal(actual.iloc[:, i], expected)
        self.assertTrue(actual.index.equals(query.index))
        self.assertEqual(list(actual.columns), list(OUTPUT_COLUMNS))
        self.assertTrue(all(dtype == np.dtype("float32") for dtype in actual.dtypes))

    def test_missing_unseen_fractional_and_near_integer_support(self) -> None:
        query = self.raw.iloc[:7].copy()
        query[RATINGS[0]] = [np.nan, 999., -1., 2.5, 1.0000000001, 1., 3.]
        query.loc[query.index[0], "Class"] = "new-category"
        actual = predict_aux_probability(query, self.directory, batch_size=1, n_jobs=2)
        np.testing.assert_array_equal(actual.iloc[:5, 0], np.full(5, EPSILON, dtype=np.float32))
        self.assertTrue(np.isfinite(actual.to_numpy()).all())

    def test_id_and_satisfaction_are_not_inputs(self) -> None:
        query = self.raw.iloc[:17].copy()
        expected = predict_aux_probability(query, self.directory, n_jobs=2)
        query["id"] = -999
        query["satisfaction"] = np.arange(len(query))
        pd.testing.assert_frame_equal(predict_aux_probability(query, self.directory, n_jobs=2), expected, check_exact=True)

    def test_batch_dtypes_order_empty_and_bank_immutability(self) -> None:
        query = self.raw.iloc[[8, 1, 3, 9, 2, 5, 0, 7, 12, 13, 14, 23, 45, 43, 44, 66, 67, 68, 69]].copy()
        query.index = [8] * len(query)
        a = predict_aux_probability(query, self.directory, batch_size=1, n_jobs=2)
        for c in RAW_FEATURES:
            if c not in CATEGORICAL:
                query[c] = query[c].astype(np.float64)
        for batch in [17, 32768]:
            pd.testing.assert_frame_equal(predict_aux_probability(query, self.directory, batch_size=batch, n_jobs=2), a, check_exact=True)
        empty = predict_aux_probability(query.iloc[:0], self.directory, n_jobs=2)
        self.assertEqual(empty.shape, (0, 14))
        self.assertTrue(empty.index.equals(query.iloc[:0].index))
        self.assertEqual(self.snapshot, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.directory.iterdir()})

    def test_invalid_inputs_and_probability_shape(self) -> None:
        for kwargs in [{"batch_size": 0}, {"n_jobs": 0}, {"batch_size": True}]:
            with self.assertRaises(ValueError):
                predict_aux_probability(self.raw.iloc[:1], self.directory, **kwargs)
        with self.assertRaisesRegex(ValueError, "Missing raw"):
            predict_aux_probability(self.raw.drop(columns=RATINGS[0]), self.directory)
        with self.assertRaisesRegex(ValueError, "probability matrix"):
            _lookup(np.array([[np.nan, .5]]), np.array([1.]), np.array([1., 3.]))

    def test_keyed_publication_duplicate_rejection_and_replay(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aux-probability-cache-") as temporary:
            root = Path(temporary)
            raw = self.raw.iloc[[8, 1, 3, 9]].copy()
            raw.index = [2, 2, 5, 1]
            features = predict_aux_probability(raw, self.directory, n_jobs=2)
            path = root / "synthetic.parquet"
            _publish(raw, features, path)
            cached = pd.read_parquet(path)
            np.testing.assert_array_equal(cached.id, raw.id)
            np.testing.assert_array_equal(cached[list(OUTPUT_COLUMNS)], features)
            with self.assertRaisesRegex(ValueError, "unique"):
                _publish(pd.concat([raw, raw.iloc[:1]]), pd.concat([features, features.iloc[:1]]), root / "bad.parquet")
            fixtures = (self.raw.iloc[:23].copy(), self.raw.iloc[23:38].copy(), {"fixture": "generated-only"})
            with patch("original_aux_probability._load_inputs", return_value=fixtures), contextlib.redirect_stdout(io.StringIO()):
                manifest = build(root, directory=self.directory, batch_size=17, n_jobs=2)
                snapshot = {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
                self.assertEqual(build(root, directory=self.directory, batch_size=1, n_jobs=2), manifest)
                self.assertEqual(snapshot, {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()})
                with (root / manifest["output_file"]).open("ab") as stream:
                    stream.write(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    build(root, directory=self.directory, n_jobs=2)


if __name__ == "__main__":
    unittest.main()
