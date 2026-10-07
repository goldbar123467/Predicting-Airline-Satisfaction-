"""Feature-only synthetic tests for the native original auxiliary bank."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.testing import assert_frame_equal

from original_aux import (AuxiliaryPreprocessor, CATEGORICAL, OUTPUT_COLUMNS, RATINGS, RAW_FEATURES,
                          _publish_keyed, benchmark_first_model, build, fit_bank, load_bank, predict_aux)


def synthetic_rows(n: int = 384) -> pd.DataFrame:
    rng = np.random.default_rng(20261002)
    raw = pd.DataFrame(index=np.arange(n)[::-1])
    for name in RAW_FEATURES:
        if name in CATEGORICAL:
            raw[name] = rng.choice(["first", "second", "third"], n)
        elif name in RATINGS:
            # Nonconsecutive observed classes exercise encoded-class inversion.
            raw[name] = rng.choice([1.0, 3.0, 5.0], n).astype(np.float32)
        else:
            raw[name] = rng.normal(20, 5, n).astype(np.float32)
    raw.loc[raw.index[::11], "Arrival Delay in Minutes"] = np.nan
    raw["satisfaction"] = "must never be read or fitted"
    raw["id"] = np.arange(n) + 100000
    return raw


class OriginalAuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="original-aux-synthetic-")
        cls.directory = Path(cls.temporary.name)
        cls.raw = synthetic_rows()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.manifest = fit_bank(cls.raw, cls.directory, rounds=8, n_jobs=2, device="cpu",
                                    provenance={"fixture": "generated feature data only"})

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_class_mapping_predictor_exclusion_and_manual_ev(self) -> None:
        processor, manifest = load_bank(self.directory)
        query = self.raw.iloc[:17]
        actual = predict_aux(query, self.directory, batch_size=7, n_jobs=2)
        self.assertEqual(list(actual.columns), list(OUTPUT_COLUMNS))
        self.assertTrue(actual.index.equals(query.index))
        self.assertTrue(all(dtype == np.dtype("float32") for dtype in actual.dtypes))
        encoded = processor.transform(query)
        for i, entry in enumerate(manifest["models"]):
            self.assertEqual(entry["classes"], [1, 3, 5])
            self.assertNotIn("satisfaction", entry["predictors"])
            self.assertNotIn("id", entry["predictors"])
            self.assertNotIn(RATINGS[i], entry["predictors"])
            model = xgb.Booster(model_file=self.directory / entry["model_file"])
            model.set_param({"device": "cpu", "nthread": 2})
            data = xgb.DMatrix(encoded[:, entry["predictor_positions"]],
                               feature_names=[f"f{j}" for j in range(20)], nthread=2)
            expected = (model.predict(data).astype(np.float64)
                        @ np.asarray([1, 3, 5], dtype=np.float64)).astype(np.float32)
            np.testing.assert_array_equal(actual.iloc[:, i].to_numpy(), expected)

    def test_own_observed_rating_and_satisfaction_are_unused_for_own_ev(self) -> None:
        query = self.raw.iloc[:17].copy()
        before = predict_aux(query, self.directory, n_jobs=2)
        query[RATINGS[0]] = 999.0  # unseen auxiliary class, only other models may use this feature
        query["satisfaction"] = np.arange(len(query))
        after = predict_aux(query, self.directory, n_jobs=2)
        np.testing.assert_array_equal(after[OUTPUT_COLUMNS[0]], before[OUTPUT_COLUMNS[0]])
        query = self.raw.iloc[:17].copy()
        query["satisfaction"] = 1
        query["id"] = -999
        assert_frame_equal(predict_aux(query, self.directory, n_jobs=2), before)

    def test_unknown_strings_numeric_dtypes_batch_sizes_and_empty_rows(self) -> None:
        processor, _ = load_bank(self.directory)
        query = self.raw.iloc[:257].copy()
        query.loc[query.index[0], "Class"] = "unseen-at-fit"
        query.loc[query.index[1], "Gender"] = None
        encoded = processor.transform(query)
        self.assertEqual(encoded[0, RAW_FEATURES.index("Class")], -1)
        self.assertEqual(encoded[1, RAW_FEATURES.index("Gender")], -1)
        for c in RAW_FEATURES:
            if c not in CATEGORICAL:
                query[c] = query[c].astype(np.float64)
        a = predict_aux(query, self.directory, batch_size=17, n_jobs=2)
        b = predict_aux(query, self.directory, batch_size=32768, n_jobs=2)
        assert_frame_equal(a, b, check_exact=True)
        assert_frame_equal(predict_aux(query.iloc[:1], self.directory, batch_size=1, n_jobs=2),
                           a.iloc[:1], check_exact=True)
        empty = predict_aux(query.iloc[:0], self.directory, n_jobs=2)
        self.assertEqual(empty.shape, (0, 13))
        self.assertTrue(empty.index.equals(query.iloc[:0].index))

    def test_order_and_keyed_cache_readback(self) -> None:
        query = self.raw.iloc[[8, 1, 3, 9]].copy()
        query.index = [6, 6, 2, 1]
        expected = predict_aux(query, self.directory, n_jobs=2)
        destination = self.directory / "synthetic_predictions.parquet"
        _publish_keyed(query, expected, destination)
        actual = pd.read_parquet(destination)
        np.testing.assert_array_equal(actual.id, query.id)
        np.testing.assert_array_equal(actual[list(OUTPUT_COLUMNS)], expected)
        with self.assertRaisesRegex(ValueError, "unique"):
            _publish_keyed(pd.concat([query, query.iloc[:1]]), pd.concat([expected, expected.iloc[:1]]), destination)

    def test_cache_reuse_and_changed_source_contract_rejection(self) -> None:
        hashes = [entry["model_sha256"] for entry in self.manifest["models"]]
        changed_unused = self.raw.copy()
        changed_unused["satisfaction"] = "different ignored values"
        with contextlib.redirect_stdout(io.StringIO()):
            reused = fit_bank(changed_unused, self.directory, rounds=8, n_jobs=2, device="cpu",
                              provenance={"fixture": "generated feature data only"})
        self.assertEqual(hashes, [entry["model_sha256"] for entry in reused["models"]])
        changed = self.raw.copy()
        changed.loc[changed.index[0], "Age"] += 100
        with self.assertRaisesRegex(ValueError, "contract differs"):
            fit_bank(changed, self.directory, rounds=8, n_jobs=2, device="cpu",
                     provenance={"fixture": "generated feature data only"})

    def test_manifest_schema_and_model_corruption_fail_closed(self) -> None:
        manifest_path = self.directory / "manifest.json"
        original_bytes = manifest_path.read_bytes()
        corrupt = json.loads(original_bytes)
        corrupt["models"][0]["predictors"][0] = "satisfaction"
        try:
            manifest_path.write_text(json.dumps(corrupt))
            with self.assertRaisesRegex(ValueError, "schema"):
                load_bank(self.directory)
        finally:
            manifest_path.write_bytes(original_bytes)
        model_path = self.directory / self.manifest["models"][0]["model_file"]
        model_bytes = model_path.read_bytes()
        try:
            with model_path.open("ab") as stream:
                stream.write(b"broken")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_bank(self.directory)
        finally:
            model_path.write_bytes(model_bytes)

    def test_cpu_pilot_and_schema_errors(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            result = benchmark_first_model(self.raw.iloc[:37], self.directory, n_jobs=2, batch_size=17)
        self.assertEqual(result["rows"], 37)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["cpu_probe_max_abs_error"], 0)
        with self.assertRaisesRegex(ValueError, "Missing raw"):
            predict_aux(self.raw.drop(columns="Age"), self.directory, n_jobs=2)

    def test_completed_build_replay_is_immutable_and_corruption_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="original-aux-replay-") as temporary:
            root = Path(temporary)
            train, test = self.raw.iloc[:37].copy(), self.raw.iloc[37:68].copy()
            # Only the IO validator is substituted, using generated feature rows.
            # The real fit, native IO, keyed publication and replay paths execute.
            fixtures = (self.raw, train, test, {"fixture_sha256": "generated-only"})
            with patch("original_aux.validate_inputs", return_value=fixtures), \
                    patch("original_aux.fit_bank", new=partial(fit_bank, rounds=8)), \
                    contextlib.redirect_stdout(io.StringIO()):
                first = build(root, n_jobs=2, batch_size=17)
                snapshot = {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
                            for p in root.rglob("*") if p.is_file()}
                replay = build(root, n_jobs=2, batch_size=31)
                self.assertEqual(first, replay)
                self.assertEqual(snapshot, {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
                                            for p in root.rglob("*") if p.is_file()})
                with self.assertRaisesRegex(ValueError, "contract differs"):
                    build(root, device="cuda", n_jobs=2)
                cache = root / first["output_file"]
                with cache.open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "cache checksum mismatch"):
                    build(root, n_jobs=2)
                self.assertEqual((root / "artifacts/original_aux/manifest.json").read_bytes(),
                                 snapshot[Path("artifacts/original_aux/manifest.json")][0])


if __name__ == "__main__":
    unittest.main()
