"""Runner boundary tests use generated data only; no campaign execution."""
import tempfile
import json
import copy
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_fixed_epoch_dropout_v1 import parity, partition_indices, read_rows, require_frame_ids
import run_fixed_epoch_dropout_v1 as runner


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


class DropoutGuardTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        control, treatment = root / "A", root / "C"
        for path in (control, treatment):
            path.mkdir()
            (path / "transform.json").write_text('{"categories":["fit-only"]}')
            np.savez_compressed(path / "partitions.npz", training_ids=np.array([1, 2]),
                monitor_ids=np.array([3]), outer_validation_ids=np.array([4, 5]))
        context = {"campaign": runner.ID, "fold": 0, "phase": "inner", "training_rows": 2,
            "monitor_rows": 1, "training_ids_sha256": "ids", "input_feature_columns": ["x", "cat"],
            "transform_sha256": runner.digest(control / "transform.json"), "feature_policy": "fixed"}
        runner.write(control / "context.json", context, exclusive=True)
        return control, treatment, context

    def test_identical_prefix_inputs_pass_and_order_matters(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment, context = self.fixture(directory)
            self.assertTrue(runner.pre_fit_match(control, treatment, context)["before_treatment_fit"])
            np.savez_compressed(treatment / "partitions.npz", training_ids=np.array([2, 1]),
                monitor_ids=np.array([3]), outer_validation_ids=np.array([4, 5]))
            with self.assertRaisesRegex(ValueError, "pre-fit"):
                runner.pre_fit_match(control, treatment, context)

    def test_transform_and_column_and_monitor_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            control, treatment, context = self.fixture(directory)
            for key, value in [("input_feature_columns", ["cat", "x"]), ("monitor_rows", 0), ("fold", 1)]:
                wrong = {**context, key: value}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.pre_fit_match(control, treatment, wrong)
            (treatment / "transform.json").write_text('{"categories":["held-only"]}')
            with self.assertRaises(ValueError):
                runner.pre_fit_match(control, treatment, context)

    def test_protocol_forbids_short_control_or_extra_epochs(self):
        from prepare_fixed_epoch_dropout_v1 import SCIENCE
        valid = {"id": runner.ID, **copy.deepcopy(SCIENCE)}
        runner.validate_protocol(valid)
        for key, value in [("common_horizon_epochs", 4), ("total_executed_epochs", 120),
                           ("treatment_dropout_base", .1), ("intervention_after_completed_epoch", 5)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_protocol({**valid, key: value})
        valid["logical_endpoints"]["A"]["epoch"] = 4
        with self.assertRaises(ValueError):
            runner.validate_protocol(valid)

    def test_gate_must_explicitly_precede_epoch_five(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            valid = {"status": "passed", "epoch": 4, "gate_completed_before_epoch5": True}
            runner.write(path / "prefix_match.json", valid)
            self.assertEqual(runner.require_prefix_match(path), valid)
            for key, value in [("status", "failed"), ("epoch", 5), ("gate_completed_before_epoch5", False)]:
                runner.write(path / "prefix_match.json", {**valid, key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.require_prefix_match(path)


if __name__ == "__main__":
    unittest.main()
