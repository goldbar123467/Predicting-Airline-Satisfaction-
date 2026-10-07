"""Release crash/replay regressions using isolated, synthetic data only."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import blend
import common
import verify


class SimulatedPublicationFailure(RuntimeError):
    pass


class ReleaseContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="airline-release-regression-")
        self.root = Path(self.temporary.name)
        self.stack = ExitStack()
        for module in [common, blend, verify]:
            self.stack.enter_context(patch.object(module, "ROOT", self.root))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        (self.root / "data").mkdir()
        (self.root / "configs").mkdir()
        self.dev_count = 60
        self.audit_count = 60
        self.test_count = 12
        self.labels = np.tile([0, 1], (self.dev_count + self.audit_count) // 2)
        self.train = pd.DataFrame({"id": np.arange(len(self.labels)), common.TARGET: self.labels,
                                   "synthetic_feature": np.arange(len(self.labels), dtype=float)})
        self.test = pd.DataFrame({"id": np.arange(1000, 1000 + self.test_count),
                                  "synthetic_feature": np.arange(self.test_count, dtype=float)})
        self.folds = np.concatenate([np.arange(self.dev_count) % 3, np.full(self.audit_count, -1)])
        split = pd.DataFrame({"id": self.train.id, "fold": self.folds, "is_audit": self.folds < 0})
        self.train.to_parquet(self.root / "data" / "train.parquet", index=False)
        self.test.to_parquet(self.root / "data" / "test.parquet", index=False)
        split.to_parquet(self.root / "data" / "splits.parquet", index=False)
        pd.DataFrame({"id": self.test.id, common.TARGET: np.full(self.test_count, .5)}).to_csv(
            self.root / "data" / "sample_submission.csv", index=False)
        self.config_path = self.root / "configs" / "overnight.json"
        self.config = {"runs": []}
        self.add_candidate("synthetic_a", seed=101)
        self.add_candidate("synthetic_b", seed=207)

    def tearDown(self) -> None:
        self.stack.close()
        self.temporary.cleanup()

    def add_candidate(self, name: str, seed: int, perfect: bool = False) -> None:
        rng = np.random.default_rng(seed)
        run = {"id": name, "family": "synthetic", "seed": seed}
        self.config["runs"].append(run)
        path = self.root / "artifacts" / "runs" / name
        path.mkdir(parents=True)
        probability = .1 + .8 * self.labels if perfect else np.clip(
            .25 + .3 * self.labels + rng.uniform(-.35, .35, len(self.labels)), .01, .99)
        pd.DataFrame({"id": self.train.id[:self.dev_count], "fold": self.folds[:self.dev_count],
                      common.TARGET: self.labels[:self.dev_count],
                      "prediction": probability[:self.dev_count]}).to_parquet(path / "oof.parquet", index=False)
        pd.DataFrame({"id": self.train.id[self.dev_count:],
                      "prediction": probability[self.dev_count:]}).to_parquet(path / "audit.parquet", index=False)
        pd.DataFrame({"id": self.test.id, "prediction": rng.uniform(.1, .9, self.test_count)}).to_parquet(
            path / "test.parquet", index=False)
        common.atomic_json(path / "result.json", {
            "id": name, "run": run, "split_hash": common.sha256(self.root / "data" / "splits.parquet"),
            "artifacts": {filename: common.sha256(path / filename)
                          for filename in ["oof.parquet", "audit.parquet", "test.parquet"]},
        })
        common.atomic_json(self.config_path, self.config)

    def finalize(self) -> None:
        with patch.object(sys, "argv", ["blend.py", "--config", str(self.config_path), "--finalize"]):
            blend.main()

    def test_crash_then_new_candidate_cannot_change_frozen_selection(self) -> None:
        original_atomic_json = blend.atomic_json

        def fail_final_publication(path: Path, value: dict) -> None:
            if path.name == "frozen.json":
                self.assertTrue((self.root / "artifacts" / "blend" / "selection.json").exists())
                raise SimulatedPublicationFailure("Crash after selection and audit arithmetic, before final publication")
            original_atomic_json(path, value)

        with patch.object(blend, "atomic_json", side_effect=fail_final_publication), \
                patch.object(blend, "optimize", wraps=blend.optimize) as initial_optimize:
            with self.assertRaises(SimulatedPublicationFailure):
                self.finalize()
            self.assertEqual(initial_optimize.call_count, 1)
        folder = self.root / "artifacts" / "blend"
        selected = common.load_config(folder / "selection.json")
        selection_hash = common.sha256(folder / "selection.json")
        self.assertEqual(selected["candidate_ids"], ["synthetic_a", "synthetic_b"])
        self.assertLess(selected["oof_auc"], 1.0, "The extra perfect candidate must be genuinely more attractive")
        self.assertFalse((folder / "frozen.json").exists())
        self.add_candidate("synthetic_perfect_late", seed=999, perfect=True)
        with patch.object(blend, "optimize", side_effect=AssertionError("Replay must not optimize weights")) as replay_optimize:
            self.finalize()
            replay_optimize.assert_not_called()
        frozen = common.load_config(folder / "frozen.json")
        self.assertEqual(frozen["candidate_ids"], selected["candidate_ids"])
        self.assertEqual(frozen["weights"], selected["weights"])
        self.assertEqual(frozen["source_predictions"], selected["source_predictions"])
        self.assertEqual(common.sha256(folder / "selection.json"), selection_hash)
        expected = sum(
            weight * pd.read_parquet(self.root / "artifacts" / "runs" / name / "test.parquet").prediction.to_numpy()
            for name, weight in selected["weights"].items())
        fallback = pd.read_csv(folder / "submission_fallback.csv")
        np.testing.assert_array_equal(fallback.id, self.test.id)
        np.testing.assert_allclose(fallback[common.TARGET], expected, rtol=1e-10, atol=1e-11)
        self.assertTrue(np.isfinite(frozen["audit_auc"]))
        final_hashes = {name: common.sha256(folder / name)
                        for name in ["selection.json", "frozen.json", "submission_fallback.csv"]}
        with patch.object(blend, "load_data", side_effect=AssertionError("Finalized replay must not load labels")) as data_loader, \
                patch.object(blend, "optimize", side_effect=AssertionError("Finalized replay must not optimize")) as optimizer:
            self.finalize()
            data_loader.assert_not_called()
            optimizer.assert_not_called()
        for name, expected_hash in final_hashes.items():
            self.assertEqual(common.sha256(folder / name), expected_hash)

    def make_final_package(self, wrong_selection: bool = False, wrong_submission: bool = False) -> None:
        self.finalize()
        final = self.root / "artifacts" / "final"
        final.mkdir(parents=True)
        folder = self.root / "artifacts" / "blend"
        shutil.copyfile(folder / "submission_fallback.csv", final / "submission.csv")
        common.atomic_json(final / "manifest.json", {
            "frozen_selection_hash": "0" * 64 if wrong_selection else common.sha256(folder / "frozen.json"),
            "submission_hash": "0" * 64 if wrong_submission else common.sha256(final / "submission.csv"),
        })

    def test_verifier_rejects_final_package_from_another_selection(self) -> None:
        self.make_final_package(wrong_selection=True)
        with patch.object(sys, "argv", ["verify.py", "--config", str(self.config_path)]), \
                patch.object(verify, "load_model", side_effect=AssertionError("Models must not load before identity checks")) as loader:
            with self.assertRaisesRegex(AssertionError, "different selection"):
                verify.main()
            loader.assert_not_called()

    def test_verifier_rejects_changed_submission_before_model_loading(self) -> None:
        self.make_final_package(wrong_submission=True)
        with patch.object(sys, "argv", ["verify.py", "--config", str(self.config_path)]), \
                patch.object(verify, "load_model", side_effect=RuntimeError("Models must not load before checks")) as loader:
            with self.assertRaises(AssertionError):
                verify.main()
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
