"""Small synthetic CPU fits verify the original-only teacher and native logit IO."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import common
import original_realmlp_teacher as teacher


class OriginalRealMLPTeacherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="original-realmlp-cpu-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "data").mkdir()
        self.directory = self.root / "artifacts/original_realmlp_teacher"
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(teacher, "FIXED_EPOCHS", 1))
        self.stack.enter_context(patch.object(teacher, "ORIGINAL_ROWS", 160))
        self.stack.enter_context(patch.object(teacher, "PARAMETERS", {
            **teacher.PARAMETERS, "device": "cpu", "n_ens": 2, "hidden_sizes": [16, 8],
            "batch_size": 32, "eval_batch_size": 17, "threads": 1,
        }))
        self.feature_columns = [*common.CAT, *common.RATINGS, "Age", "Flight Distance",
                                "Departure Delay in Minutes", "Arrival Delay in Minutes"]
        rng = np.random.default_rng(19203)
        self.original = pd.DataFrame({
            c: rng.choice(["source_a", "source_b"], 160) if c in common.CAT
            else rng.integers(0, 6, 160).astype(np.float32)
            for c in self.feature_columns})
        self.original["Flight Distance"] = np.arange(160, dtype=np.float32) + 100
        self.original.loc[::13, "Arrival Delay in Minutes"] = np.nan
        self.original[common.TARGET] = ((self.original["Age"] + self.original["Online boarding"]) >= 5).astype(np.int8)
        self.original.to_parquet(self.root / "data/original.parquet", index=False)
        self.train = self.original[self.feature_columns].iloc[:17].copy().reset_index(drop=True)
        self.test = self.original[self.feature_columns].iloc[17:28].copy().reset_index(drop=True)
        self.train["Flight Distance"] += 20000
        self.test["Flight Distance"] += 30000
        self.train.loc[0, "Gender"] = "synthetic_only_category"
        self.test.loc[0, "Class"] = "unseen_test_category"
        self.train.insert(0, "id", np.arange(901, 918))
        self.test.insert(0, "id", np.arange(5001, 5012))
        self.train[common.TARGET] = "DO_NOT_READ_SYNTHETIC_LABEL"
        self.train.to_csv(self.root / "data/train.csv", index=False)
        self.test.to_csv(self.root / "data/test.csv", index=False)
        common.atomic_json(self.root / "research/original_audit.json", {
            "output_sha256": common.sha256(self.root / "data/original.parquet"),
            "synthetic_train_sha256": common.sha256(self.root / "data/train.csv"),
            "synthetic_test_sha256": common.sha256(self.root / "data/test.csv"),
            "features": self.feature_columns,
        })

    def test_source_only_fixed_fit_cpu_native_reload_and_unknown_categories(self):
        read_csv = pd.read_csv
        fitter = teacher.fit_model
        captured = []

        def guarded_read(path, *args, **kwargs):
            if Path(path) == self.root / "data/train.csv":
                self.assertIsNotNone(kwargs.get("usecols"))
                self.assertNotIn(common.TARGET, kwargs["usecols"])
            return read_csv(path, *args, **kwargs)

        def capture_fit(x, y, valid, y_valid, run, out, rounds):
            self.assertEqual(len(x), 160)
            np.testing.assert_array_equal(y, self.original[common.TARGET])
            self.assertIsNone(valid)
            self.assertIsNone(y_valid)
            self.assertEqual(rounds, 1)
            self.assertEqual(run["seed"], 0)
            self.assertEqual(run["params"]["device"], "cpu")
            captured.append(run)
            return fitter(x, y, valid, y_valid, run, out, rounds=rounds)

        with patch.object(pd, "read_csv", side_effect=guarded_read), \
                patch.object(teacher, "fit_model", side_effect=capture_fit):
            # build_contract hashes the actual reused fit implementation, not a mock.
            with patch.object(teacher, "source_hash", side_effect=lambda value: "synthetic_capture"):
                manifest = teacher.build(self.root)
        self.assertEqual(len(captured), 1)
        self.assertFalse(manifest["synthetic_labels_read"])
        self.assertEqual(manifest["contract"]["rows"]["original"], 160)
        model, transform, metadata = teacher._load_native(self.directory)
        self.assertEqual(str(model.device), "cpu")
        self.assertEqual(transform.vocab["Gender"], ["source_a", "source_b"])
        self.assertNotIn("synthetic_only_category", transform.vocab["Gender"])
        encoded = transform.transform(teacher.feature_frame(self.train, self.feature_columns))
        gender_index = len(transform.nums) + transform.cats.index("Gender")
        self.assertEqual(encoded[0, gender_index], -1)
        expected = teacher.clipped_logit(teacher._probabilities(model, transform, self.train, self.feature_columns))
        actual = teacher.predict_teacher(self.train, self.directory)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(actual.dtype, np.float32)
        raw = self.train.copy()
        raw["id"] = np.arange(900001, 900001 + len(raw))
        raw[common.TARGET] = np.arange(len(raw)) % 2
        np.testing.assert_allclose(teacher.predict_teacher(raw, self.directory, batch_size=1), actual, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(teacher.predict_teacher(raw.iloc[::-1], self.directory, batch_size=3), actual[::-1], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(teacher.predict_teacher(raw.drop(columns=["id", common.TARGET]), self.directory), actual, rtol=0, atol=0)
        self.assertEqual(teacher.predict_teacher(raw.iloc[:0], self.directory).shape, (0,))
        with self.assertRaisesRegex(ValueError, "every raw feature"):
            teacher.predict_teacher(raw.drop(columns="Age"), self.directory)

    def test_cache_checksum_resume_native_tamper_and_changed_recipe(self):
        manifest = teacher.build(self.root)
        output = self.root / "data/original_realmlp_teacher_predictions.parquet"
        cached = pd.read_parquet(output)
        self.assertEqual(list(cached), ["id", teacher.PREDICTION_COLUMN])
        np.testing.assert_array_equal(cached.id, np.r_[self.train.id, self.test.id])
        self.assertEqual(cached[teacher.PREDICTION_COLUMN].dtype, np.float32)
        self.assertEqual(common.sha256(output), manifest["output_sha256"])
        raw = pd.concat([self.train, self.test], ignore_index=True)
        np.testing.assert_allclose(cached[teacher.PREDICTION_COLUMN], teacher.predict_teacher(raw, self.directory, batch_size=17), rtol=2e-5, atol=2e-6)
        manifest_bytes = (self.directory / "manifest.json").read_bytes()
        with patch.object(teacher, "_fit_native", side_effect=AssertionError("Completed teacher cannot refit")):
            self.assertEqual(teacher.build(self.root), manifest)
        self.assertEqual((self.directory / "manifest.json").read_bytes(), manifest_bytes)
        with patch.object(teacher, "FIXED_EPOCHS", 2):
            with self.assertRaisesRegex(ValueError, "Immutable teacher contract changed"):
                teacher.build(self.root)
        native_meta = common.load_config(self.directory / "model/metadata.json")
        graph = self.directory / "model" / native_meta["graph_file"]
        graph.write_bytes(graph.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
            teacher.predict_teacher(raw, self.directory)
        self.assertEqual((self.directory / "manifest.json").read_bytes(), manifest_bytes)

    def test_export_failure_resumes_native_checkpoint_without_retraining(self):
        with patch.object(teacher, "_check_export", side_effect=OSError("synthetic cache failure")):
            with self.assertRaisesRegex(OSError, "cache failure"):
                teacher.build(self.root)
        self.assertTrue((self.directory / "model_complete.json").exists())
        self.assertFalse((self.directory / "manifest.json").exists())
        checkpoint = (self.directory / "model_complete.json").read_bytes()
        with patch.object(teacher, "_fit_native", side_effect=AssertionError("Committed native model cannot refit")):
            manifest = teacher.build(self.root)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual((self.directory / "model_complete.json").read_bytes(), checkpoint)

    def test_prepare_and_validation_never_fit_and_twins_match_existing_raw_recipe(self):
        with patch.object(teacher, "_fit_native", side_effect=AssertionError("Prepare cannot fit")):
            result = teacher.build(self.root, validate_only=True)
            self.assertEqual(result["status"], "validated")
            self.assertFalse(self.directory.exists())
            result = teacher.build(self.root, prepare_only=True)
            self.assertEqual(result["status"], "prepared")
            self.assertTrue((self.directory / "contract.json").exists())
            self.assertFalse((self.directory / "model_complete.json").exists())
        actual = teacher.feature_frame(self.original, self.feature_columns)
        expected = common.features(self.original, {"categorical_twins": True})
        pd.testing.assert_frame_equal(actual, expected)
        with patch.object(teacher, "ORIGINAL_ROWS", 161):
            with self.assertRaisesRegex(ValueError, "cleaned original-only"):
                teacher.build(self.root, validate_only=True)

    def test_logit_rule_endpoints_dtype_and_invalid_values(self):
        p = np.array([0., 1e-12, .25, .5, .75, 1 - 1e-12, 1.])
        clipped = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
        expected = (np.log(clipped) - np.log1p(-clipped)).astype(np.float32)
        np.testing.assert_array_equal(teacher.clipped_logit(p), expected)
        self.assertEqual(teacher.clipped_logit(np.array([])).shape, (0,))
        for invalid in [np.array([np.nan]), np.array([np.inf]), np.array([-.1]), np.array([1.1]), np.ones((2, 1))]:
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                teacher.clipped_logit(invalid)
        for size in [0, -1, True, 1.5]:
            with self.assertRaisesRegex(ValueError, "batch_size"):
                teacher.predict_teacher(self.test, self.directory, batch_size=size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
