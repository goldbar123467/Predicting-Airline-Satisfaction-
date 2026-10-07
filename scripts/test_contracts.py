"""Small contract checks; no training, audit scoring, or artifact mutation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from blend import checked_predictions, optimize
from common import ROOT, TARGET, Transform, load_config, sha256


class PipelineContracts(unittest.TestCase):
    def test_preprocessing_fits_training_only_and_round_trips(self):
        train = pd.DataFrame({"number": [0.0, 2.0, np.nan], "category": ["a", "b", "a"]})
        heldout = pd.DataFrame({"number": [1_000_000.0, np.nan], "category": ["unseen", "a"]})
        for family in ["lightgbm", "catboost", "xgboost", "tabm"]:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                transform = Transform(family).fit(train)
                self.assertEqual(transform.vocab["category"], ["a", "b"])
                self.assertEqual(transform.median["number"], 1.0)
                self.assertEqual(transform.mean["number"], 1.0)
                encoded = transform.transform(heldout)
                transform.save(Path(directory) / "transform.json")
                restored = Transform.load(Path(directory) / "transform.json").transform(heldout)
                if isinstance(encoded, np.ndarray):
                    np.testing.assert_array_equal(encoded, restored)
                    self.assertTrue(np.isfinite(encoded).all())
                    np.testing.assert_array_equal(encoded[0, 1:], [0, 0])
                else:
                    pd.testing.assert_frame_equal(encoded, restored)
                self.assertNotIn("unseen", transform.vocab["category"])

    def test_keyed_prediction_loading_reorders_and_rejects_missing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.parquet"
            pd.DataFrame({"id": [30, 10, 20], "prediction": [.3, .1, .2]}).to_parquet(path)
            np.testing.assert_array_equal(checked_predictions(path, [10, 20, 30]), [.1, .2, .3])
            with self.assertRaises(AssertionError):
                checked_predictions(path, [10, 20, 40])

    def test_blend_remains_convex_and_does_not_degrade_best_input(self):
        rng = np.random.default_rng(7102)
        labels = rng.integers(0, 2, size=300)
        matrix = np.column_stack([
            np.clip(.2 + .5 * labels + rng.normal(0, .25, len(labels)), 0, 1),
            np.clip(.2 + .5 * labels + rng.normal(0, .25, len(labels)), 0, 1),
            rng.random(len(labels)),
        ])
        weights, score, single = optimize(labels, matrix)
        self.assertTrue((weights >= 0).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreaterEqual(score + 1e-12, float(single.max()))
        self.assertAlmostEqual(score, roc_auc_score(labels, matrix @ weights))
        singleton, _, _ = optimize(labels, matrix[:, :1])
        np.testing.assert_array_equal(singleton, [1.0])

    def test_actual_nested_partitions_exclude_outer_and_audit_rows(self):
        split = pd.read_parquet(ROOT / "data" / "splits.parquet")
        targets = pd.read_parquet(ROOT / "data" / "train.parquet", columns=["id", TARGET])
        np.testing.assert_array_equal(split.id, targets.id)
        self.assertTrue(split.id.is_unique)
        fold = split.fold.to_numpy()
        self.assertEqual(set(fold), {-1, 0, 1, 2})
        audit = np.flatnonzero(fold < 0)
        y = targets[TARGET].to_numpy()
        for k in range(3):
            outer_train = np.flatnonzero((fold >= 0) & (fold != k))
            outer_valid = np.flatnonzero(fold == k)
            inner, stop = train_test_split(outer_train, test_size=.1, random_state=20261001 + k,
                                          stratify=y[outer_train])
            self.assertEqual(np.intersect1d(inner, stop).size, 0)
            np.testing.assert_array_equal(np.sort(np.concatenate([inner, stop])), outer_train)
            for indexes in [inner, stop, outer_train]:
                self.assertEqual(np.intersect1d(indexes, outer_valid).size, 0)
                self.assertEqual(np.intersect1d(indexes, audit).size, 0)

    def test_completed_artifact_hashes_and_fold_shapes(self):
        config = load_config(ROOT / "configs" / "overnight.json")
        split = pd.read_parquet(ROOT / "data" / "splits.parquet")
        test_ids = pd.read_parquet(ROOT / "data" / "test.parquet", columns=["id"]).id.to_numpy()
        split_hash = sha256(ROOT / "data" / "splits.parquet")
        fold = split.fold.to_numpy()
        for run in config["runs"]:
            base = ROOT / "artifacts" / "runs" / run["id"]
            if not (base / "result.json").exists():
                continue
            with self.subTest(run=run["id"]):
                result = load_config(base / "result.json")
                self.assertEqual(result["run"], run)
                self.assertEqual(result["split_hash"], split_hash)
                for filename, expected_hash in result["artifacts"].items():
                    self.assertEqual(sha256(base / filename), expected_hash)
                for k in range(3):
                    folder = base / f"fold_{k}"
                    done = load_config(folder / "done.json")
                    self.assertEqual(sha256(folder / "predictions.npz"), done["artifact_hash"])
                    with np.load(folder / "predictions.npz", allow_pickle=False) as predictions:
                        np.testing.assert_array_equal(predictions["validation_ids"], split.id.to_numpy()[fold == k])
                        expected_lengths = {"valid": int((fold == k).sum()),
                                            "audit": int((fold < 0).sum()), "test": len(test_ids)}
                        for key, expected in expected_lengths.items():
                            self.assertEqual(predictions[key].shape, (expected,))
                            self.assertTrue(np.isfinite(predictions[key]).all())
                            self.assertTrue(((predictions[key] >= 0) & (predictions[key] <= 1)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
