"""Runner boundary tests use generated data only; no campaign execution."""
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_fixed_epoch_v1 import parity, partition_indices, read_rows, require_frame_ids
import run_fixed_epoch_v1 as runner


class RunnerBoundaryTests(unittest.TestCase):
    def test_filtered_read_excludes_audit_and_preserves_requested_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            frame = pd.DataFrame({"id": [10, 20, 30, 40], "satisfaction": [0, 1, 999, 999], "value": [1., 2., 3., 4.]})
            frame.to_parquet(path, index=False)
            selected = read_rows(path, np.array([20, 10]))
            self.assertEqual(selected.id.tolist(), [20, 10])
            self.assertEqual(selected.satisfaction.tolist(), [1, 0])
            raw_features = read_rows(path, np.array([20]), columns=["id", "value"])
            self.assertNotIn("satisfaction", raw_features)

    def test_missing_extra_and_duplicate_ids_rejected(self):
        for values in ([1, 1], [1, 3], [1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                require_frame_ids(pd.DataFrame({"id": values}), np.array([1, 2]), "toy")

    def test_partition_matches_original_policy_and_excludes_outer(self):
        from sklearn.model_selection import train_test_split
        folds = np.repeat(np.arange(3), 100)
        y = np.tile([0, 1], 150)
        train, monitor, held = partition_indices(y, folds, 1, "inner", 20261005)
        expected_train, expected_monitor = train_test_split(np.flatnonzero(folds != 1), test_size=.1,
                     random_state=20261006, stratify=y[folds != 1])
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(monitor, expected_monitor)
        self.assertFalse(set(train) & set(held) or set(monitor) & set(held))
        outer, absent, held2 = partition_indices(y, folds, 1, "outer", 20261005)
        np.testing.assert_array_equal(outer, np.flatnonzero(folds != 1))
        np.testing.assert_array_equal(held, held2)
        self.assertIsNone(absent)

    def test_scaled_parity_is_reference_based_and_rejects_nonfinite(self):
        reference = np.array([0., .5, 1.])
        self.assertTrue(parity(reference, reference + np.array([1e-6, 2e-6, 3e-6]))["parity_passed"])
        for candidate in (reference + .001, np.array([0., np.nan, 1.]), np.array([0., 1.])):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                parity(reference, candidate)

    def test_failed_owned_child_is_reported_and_identity_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            state = {}
            with patch.multiple(runner, ROOT=root, STATE=root / "state"):
                with self.assertRaisesRegex(RuntimeError, "Worker exited 3"):
                    runner.run_child([sys.executable, "-c", "raise SystemExit(3)"],
                        datetime.now(timezone.utc) + timedelta(seconds=20), root / "failure.log", state)
            self.assertIsNone(state["active_child"])
            self.assertIn("create_time", state["last_child"])
            self.assertIn("command", state["last_child"])

    def test_deadline_stops_only_owned_child_tree(self):
        from supervisor import owned_process, surviving_descendants
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            state = {}
            with patch.multiple(runner, ROOT=root, STATE=root / "state"):
                with self.assertRaises(TimeoutError):
                    runner.run_child([sys.executable, "-c", "import time; time.sleep(60)"],
                        datetime.now(timezone.utc) + timedelta(seconds=.1), root / "timeout.log", state)
            self.assertIsNone(owned_process(state["last_child"]))
            self.assertEqual(surviving_descendants(state["last_child"]), [])


if __name__ == "__main__":
    unittest.main()
