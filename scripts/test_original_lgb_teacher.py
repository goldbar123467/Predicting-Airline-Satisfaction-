"""Synthetic CPU-only tests for the immutable original LightGBM teacher."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import common
import original_lgb_teacher as teacher


class OriginalLightGBMTeacherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="airline-original-lgb-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "data").mkdir()
        (self.root / "research").mkdir()
        self.directory = self.root / "artifacts/original_lgb_teacher"
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(redirect_stdout(io.StringIO()))
        # Production CLI has no round-count override. Only this synthetic fixture
        # substitutes eight rounds for the fixed 1,500-round production contract.
        stack.enter_context(patch.object(teacher, "FIXED_ROUNDS", 8))
        self.features = [*common.CAT, *common.RATINGS, "Age", "Flight Distance",
                         "Departure Delay in Minutes", "Arrival Delay in Minutes"]
        self.assertEqual(len(self.features), 21)
        rng = np.random.default_rng(8117)
        rows = 384
        self.original = pd.DataFrame({
            column: rng.choice(["source_a", "source_b"], rows) if column in common.CAT
            else rng.integers(0, 6, rows).astype(float)
            for column in self.features})
        self.original["Flight Distance"] = np.arange(rows, dtype=float) + 100
        self.original.loc[::23, "Arrival Delay in Minutes"] = np.nan
        probability = np.where(self.original["Online boarding"] >= 3, .75, .25)
        self.original[common.TARGET] = (rng.random(rows) < probability).astype(np.int8)
        self.original.to_parquet(self.root / "data/original.parquet", index=False)
        self.train = self.original[self.features].iloc[:17].copy().reset_index(drop=True)
        self.test = self.original[self.features].iloc[17:28].copy().reset_index(drop=True)
        self.train["Flight Distance"] += 20000
        self.test["Flight Distance"] += 30000
        self.train.loc[0, "Gender"] = "synthetic_only_category"
        self.test.loc[0, "Class"] = "new_test_category"
        self.train.insert(0, "id", np.arange(901, 918))
        self.test.insert(0, "id", np.arange(5001, 5012))
        # Deliberately not usable satisfaction values. Reading/fitting these
        # instead of original source labels cannot produce a valid binary fit.
        self.train[common.TARGET] = "DO_NOT_READ_SYNTHETIC_LABEL"
        self.train.to_csv(self.root / "data/train.csv", index=False)
        self.test.to_csv(self.root / "data/test.csv", index=False)
        common.atomic_json(self.root / "research/original_audit.json", {
            "output_sha256": common.sha256(self.root / "data/original.parquet"),
            "synthetic_train_sha256": common.sha256(self.root / "data/train.csv"),
            "synthetic_test_sha256": common.sha256(self.root / "data/test.csv"),
            "features": self.features,
        })

    def test_fit_reads_only_original_labels_and_fits_source_vocabulary(self):
        actual_read_csv = pd.read_csv
        actual_fit = teacher.lgb.LGBMClassifier.fit
        fit_calls = []

        def guarded_read(path, *args, **kwargs):
            if Path(path) == self.root / "data/train.csv":
                self.assertIsNotNone(kwargs.get("usecols"))
                self.assertNotIn(common.TARGET, kwargs["usecols"])
            return actual_read_csv(path, *args, **kwargs)

        def capture_fit(model, x, y, *args, **kwargs):
            fit_calls.append((x.copy(), np.asarray(y).copy(), kwargs))
            return actual_fit(model, x, y, *args, **kwargs)

        with patch.object(pd, "read_csv", side_effect=guarded_read), \
                patch.object(teacher.lgb.LGBMClassifier, "fit", autospec=True, side_effect=capture_fit):
            manifest = teacher.build(self.root)
        self.assertEqual(len(fit_calls), 1)
        fitted_x, fitted_y, kwargs = fit_calls[0]
        self.assertEqual(len(fitted_x), len(self.original))
        np.testing.assert_array_equal(fitted_y, self.original[common.TARGET])
        self.assertNotIn("eval_set", kwargs)
        self.assertNotIn("callbacks", kwargs)
        transform = common.Transform.load(self.directory / "transform.json")
        self.assertEqual(transform.vocab["Gender"], ["source_a", "source_b"])
        self.assertNotIn("synthetic_only_category", transform.vocab["Gender"])
        encoded = transform.transform(self.train[self.features])
        self.assertTrue(pd.isna(encoded.iloc[0, transform.columns.index("Gender")]))
        self.assertFalse(manifest["synthetic_labels_read"])
        self.assertEqual(manifest["contract"]["positive_class"], 1)
        self.assertEqual(manifest["input_hashes"]["original_audit_sha256"],
                         common.sha256(self.root / "research/original_audit.json"))

    def test_native_reload_ignores_ids_and_satisfaction_and_preserves_class_one(self):
        teacher.build(self.root)
        raw = self.train.copy()
        first = teacher.predict_teacher(raw, self.directory)
        raw["id"] = np.arange(900001, 900001 + len(raw))
        raw[common.TARGET] = np.arange(len(raw)) % 2
        second = teacher.predict_teacher(raw, self.directory)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, teacher.predict_teacher(raw.drop(columns=["id", common.TARGET]), self.directory))
        np.testing.assert_array_equal(first[::-1], teacher.predict_teacher(raw.iloc[::-1], self.directory))
        model, transform = common.load_model(self.directory)
        reference = common.predict(model, transform, raw[self.features])
        np.testing.assert_allclose(first, reference, rtol=1e-6, atol=6e-8)
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all() and ((first >= 0) & (first <= 1)).all())
        self.assertEqual(teacher.predict_teacher(raw.iloc[:0], self.directory).shape, (0,))
        with self.assertRaisesRegex(ValueError, "every raw feature"):
            teacher.predict_teacher(raw.drop(columns="Age"), self.directory)

    def test_completed_export_is_keyed_float32_and_resume_does_not_refit_or_rewrite(self):
        first = teacher.build(self.root)
        manifest_bytes = (self.directory / "manifest.json").read_bytes()
        output = self.root / "data/original_lgb_teacher_predictions.parquet"
        frame = pd.read_parquet(output)
        self.assertEqual(list(frame.columns), ["id", teacher.PREDICTION_COLUMN])
        np.testing.assert_array_equal(frame.id, np.r_[self.train.id, self.test.id])
        self.assertEqual(frame[teacher.PREDICTION_COLUMN].dtype, np.float32)
        with patch.object(teacher, "_fit_native", side_effect=AssertionError("Completed teacher cannot refit")):
            second = teacher.build(self.root)
        self.assertEqual(first, second)
        self.assertEqual((self.directory / "manifest.json").read_bytes(), manifest_bytes)
        self.assertEqual(common.sha256(output), first["output_sha256"])

    def test_prediction_crash_resumes_native_checkpoint_without_retraining(self):
        with patch.object(teacher, "_check_export", side_effect=OSError("synthetic export failure")):
            with self.assertRaisesRegex(OSError, "synthetic export failure"):
                teacher.build(self.root)
        self.assertTrue((self.directory / "model_complete.json").exists())
        self.assertFalse((self.directory / "manifest.json").exists())
        native_hash = common.sha256(self.directory / "model.txt")
        with patch.object(teacher, "_fit_native", side_effect=AssertionError("Committed native model cannot refit")):
            manifest = teacher.build(self.root)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(common.sha256(self.directory / "model.txt"), native_hash)

    def test_changed_native_artifact_is_rejected_without_overwriting_completion(self):
        teacher.build(self.root)
        manifest_bytes = (self.directory / "manifest.json").read_bytes()
        model_path = self.directory / "model.txt"
        model_path.write_bytes(model_path.read_bytes() + b"\ncorrupted\n")
        with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
            teacher.predict_teacher(self.test, self.directory)
        with patch.object(teacher, "_fit_native", side_effect=AssertionError("Must not overwrite corruption")):
            with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
                teacher.build(self.root)
        self.assertEqual((self.directory / "manifest.json").read_bytes(), manifest_bytes)

    def test_changed_recipe_is_rejected_and_validation_only_creates_no_artifact(self):
        result = teacher.build(self.root, validate_only=True)
        self.assertTrue(result["validated"])
        self.assertFalse(self.directory.exists())
        teacher.build(self.root)
        with patch.object(teacher, "FIXED_ROUNDS", 9):
            with self.assertRaisesRegex(ValueError, "Immutable teacher contract changed"):
                teacher.build(self.root)


if __name__ == "__main__":
    unittest.main()
