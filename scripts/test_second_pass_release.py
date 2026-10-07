"""Second-pass release regressions using temporary synthetic fixtures only.

No competition data, audit labels, native learner, or GPU is read by this suite.
Run: .venv\\Scripts\\python.exe scripts/test_second_pass_release.py
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import common
import second_pass_release as release


def policy(**changes):
    result = {"new_candidate_prefix": "v2_", "alpha_grid": [.05, .1, .2, .3],
              "max_additions": 3, "minimum_pooled_gain": .00001,
              "minimum_mean_fold_gain": .00001, "maximum_fold_regression": .00002}
    result.update(changes)
    return result


class SearchContracts(unittest.TestCase):
    def setUp(self):
        self.y = np.tile([0, 1], 30)
        self.folds = np.arange(60) % 3
        self.anchor = np.full(60, .5)

    def test_candidate_must_pass_pooled_mean_and_worst_fold_gates(self):
        before = {"pooled": .8, "mean_fold": .8, "folds": [.8, .8, .8]}
        valid = {"pooled": .801, "mean_fold": .801, "folds": [.801, .801, .801]}
        self.assertTrue(release.eligible_change(before, valid, policy()))
        for field in ["pooled", "mean_fold", "folds"]:
            invalid = copy.deepcopy(valid)
            invalid[field] = [.79, .81, .81] if field == "folds" else .8
            self.assertFalse(release.eligible_change(before, invalid, policy()), field)

    def test_deterministic_ties_prefer_smallest_alpha_then_declared_order(self):
        # Every positive mixture perfectly separates the labels, so all grid
        # values and both identical candidates have the same AUC.
        candidates = {"first": self.y.astype(float), "second": self.y.astype(float)}
        args = (self.y, self.folds, self.anchor, candidates, policy(max_additions=1))
        first = release.anchored_search(*args)
        second = release.anchored_search(*args)
        self.assertEqual(first[0], {"anchor": .95, "first": .05})
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[3], second[3])
        np.testing.assert_array_equal(first[1], second[1])

    def test_cumulative_fold_regression_is_checked_against_immutable_anchor(self):
        # Controlled score oracle isolates the search policy: each step would
        # individually lose .015 on fold 0, but their cumulative .030 loss is
        # inadmissible under a .020 immutable-anchor limit.
        table = {0.: (.8, [.8, .8, .8]),
                 .5: (.825, [.785, .845, .845]),
                 1.: (.81, [.79, .82, .82]),
                 1.25: (.835, [.770, .8675, .8675])}

        def controlled_scores(y, probability, folds):
            pooled, values = table[float(probability[0])]
            return {"pooled": pooled, "mean_fold": float(np.mean(values)), "folds": values}

        with patch.object(release, "scores", side_effect=controlled_scores):
            weights, _, value, history = release.anchored_search(
                np.array([0, 1]), np.array([0, 1]), np.array([0.]),
                {"first": np.array([1.]), "second": np.array([2.])},
                policy(alpha_grid=[.5], max_additions=2, maximum_fold_regression=.02))
        self.assertEqual(weights, {"anchor": .5, "first": .5})
        self.assertEqual(len(history), 1)
        self.assertGreaterEqual(min(np.asarray(value["folds"]) - .8), -.02)


class SyntheticRelease(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="second-pass-regression-")
        self.root = Path(self.temporary.name)
        self.stack = ExitStack()
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(common, "ROOT", self.root))
        for key, value in {
            "ROOT": self.root, "OUT": self.root / "artifacts/second_pass",
            "BLEND": self.root / "artifacts/second_pass/blend",
            "FINAL": self.root / "artifacts/second_pass/final",
            "BASELINE": self.root / "artifacts/blend/frozen.json",
        }.items():
            self.stack.enter_context(patch.object(release, key, value))
        (self.root / "data").mkdir()
        self.y = np.tile([0, 1], 30)
        self.ids = np.arange(60)
        self.folds = np.arange(60) % 3
        # Audit IDs exist in split metadata, but audit labels/predictions do not.
        pd.DataFrame({"id": np.arange(66), "fold": np.r_[self.folds, [-1] * 6]}).to_parquet(
            self.root / "data/splits.parquet", index=False)
        self.test = pd.DataFrame({"id": [f"{i:04d}" for i in range(12)],
                                  "measurement": np.arange(12, dtype=np.float32),
                                  "Class": ["a", "b"] * 6})
        self.test.to_parquet(self.root / "data/test.parquet", index=False)
        self.test.to_csv(self.root / "data/test.csv", index=False)
        pd.DataFrame({"id": self.test.id, common.TARGET: .5}).to_csv(
            self.root / "data/sample_submission.csv", index=False)
        common.atomic_json(self.root / "data/manifest.json", {
            "rows": {"train": 66, "test": 12}, "feature_columns": ["measurement", "Class"],
            "dtypes": {"id": "int64", "measurement": "float32", "Class": "str"}})
        self.config_path = self.root / "configs/second_pass.json"
        self.config = {"campaign": "second_pass", "runs": [], "blend_groups": {},
                       "selection_policy": policy()}
        self.add_candidate("old_anchor", np.full(60, .5))
        common.atomic_json(release.BASELINE, {
            "weights": {"old_anchor": 1.}, "oof_auc": .5,
            "split_hash": common.sha256(self.root / "data/splits.parquet"),
            "audit_auc": "UNREADABLE_SENTINEL", "audit_unexposed_auc": "UNREADABLE_SENTINEL"})
        self.baseline_hash = common.sha256(release.BASELINE)
        self.stack.enter_context(patch.object(release, "BASELINE_SHA", self.baseline_hash))

    def tearDown(self):
        self.stack.close()
        self.temporary.cleanup()

    def write_config(self):
        common.atomic_json(self.config_path, self.config)

    def add_candidate(self, name, probabilities=None, complete=True):
        run = {"id": name, "family": "synthetic", "seed": 123}
        self.config["runs"].append(run)
        self.write_config()
        if not complete:
            return run
        path = self.root / "artifacts/runs" / name
        path.mkdir(parents=True)
        values = self.y.astype(float) if probabilities is None else probabilities
        pd.DataFrame({"id": self.ids, "fold": self.folds, common.TARGET: self.y,
                      "prediction": values}).to_parquet(path / "oof.parquet", index=False)
        pd.DataFrame({"id": self.test.id, "prediction": np.linspace(.1, .9, len(self.test))}).to_parquet(
            path / "test.parquet", index=False)
        common.atomic_json(path / "result.json", {
            "id": name, "run": run, "rounds": [2, 3, 4],
            "split_hash": common.sha256(self.root / "data/splits.parquet"),
            "artifacts": {key: common.sha256(path / key) for key in ["oof.parquet", "test.parquet"]}})
        return run

    def freeze(self):
        release.blend(self.config_path, freeze=True)
        return common.load_config(release.BLEND / "frozen.json")

    def make_package(self):
        selection = self.freeze()
        _, specs = release.context(self.config_path)
        release.FINAL.mkdir(parents=True, exist_ok=True)
        submission = pd.read_csv(release.BLEND / "submission_fallback.csv", dtype={"id": str})
        submission.to_csv(release.FINAL / "submission.csv", index=False)
        members = [{**s, "path": str(release.member_path(s) / "model"),
                    "predictions": str(release.member_path(s) / "test.parquet")} for s in specs]
        v1_hashes = {}
        for spec in specs:
            directory = self.create_native_member(spec)
            if directory.parent == self.root / "artifacts/final":
                v1_hashes.update({p.relative_to(self.root).as_posix(): common.sha256(p)
                                  for p in directory.rglob("*") if p.is_file()})
        common.atomic_json(self.root / "artifacts/final/release_provenance.json", {
            "native_model_and_prediction_hashes": v1_hashes})
        manifest = {"frozen_selection_hash": common.sha256(release.BLEND / "frozen.json"),
                    "submission_hash": common.sha256(release.FINAL / "submission.csv"),
                    "members": members}
        common.atomic_json(release.FINAL / "manifest.json", manifest)
        return selection, specs, manifest

    def test_selection_never_reads_audit_or_full_training_labels(self):
        original_reader = pd.read_parquet
        original_loader = release.load_config

        class GuardedBaseline(dict):
            def __getitem__(self, key):
                if "audit" in key:
                    raise AssertionError("Selection accessed a historical audit field")
                return super().__getitem__(key)

            def get(self, key, default=None):
                if "audit" in key:
                    raise AssertionError("Selection accessed a historical audit field")
                return super().get(key, default)

        def guarded_read(path, *args, **kwargs):
            self.assertNotIn(Path(path).name, {"train.parquet", "audit.parquet"})
            return original_reader(path, *args, **kwargs)

        def guarded_load(path):
            result = original_loader(path)
            return GuardedBaseline(result) if Path(path) == release.BASELINE else result

        self.add_candidate("v2_better")
        with patch.object(pd, "read_parquet", side_effect=guarded_read), \
                patch.object(release, "load_config", side_effect=guarded_load), \
                patch.object(release, "load_data", side_effect=AssertionError("No full-data reads in selection")):
            frozen = self.freeze()
        self.assertFalse(frozen["audit_evaluated"])
        self.assertEqual(common.sha256(release.BASELINE), self.baseline_hash)
        self.assertFalse((self.root / "artifacts/final").exists())

    def test_frozen_replay_is_idempotent_and_does_not_read_data_or_config(self):
        self.freeze()
        names = ["frozen.json", "submission_fallback.csv", "oof.parquet"]
        hashes = {name: common.sha256(release.BLEND / name) for name in names}
        self.add_candidate("v2_late_perfect")
        with patch.object(release, "load_config", side_effect=AssertionError("No config reload")), \
                patch.object(pd, "read_parquet", side_effect=AssertionError("No data reread")), \
                patch.object(release, "anchored_search", side_effect=AssertionError("No reselection")):
            release.blend(self.config_path, freeze=True)
        self.assertEqual(hashes, {name: common.sha256(release.BLEND / name) for name in names})
        self.assertEqual(common.sha256(release.BASELINE), self.baseline_hash)

    def test_incomplete_seed_group_is_withheld(self):
        self.add_candidate("v2_seed1")
        self.add_candidate("v2_seed2", complete=False)
        self.config["blend_groups"] = {"v2_equal_pair": ["v2_seed1", "v2_seed2"]}
        self.write_config()
        selected = self.freeze()
        self.assertEqual(selected["weights"], {"old_anchor": 1.})
        self.assertEqual(selected["addition_history"], [])

    def test_complete_seed_group_expands_with_equal_weights(self):
        self.add_candidate("v2_seed1")
        self.add_candidate("v2_seed2")
        self.config["blend_groups"] = {"v2_equal_pair": ["v2_seed1", "v2_seed2"]}
        self.write_config()
        selected = self.freeze()
        self.assertEqual(selected["weights"], {"old_anchor": .95, "v2_seed1": .025, "v2_seed2": .025})
        self.assertEqual(selected["addition_history"][0]["candidate"], "v2_equal_pair")
        self.assertNotIn("v2_seed1", selected["anchor_and_addition_weights"])

    def test_group_must_pass_gate_as_average_not_its_lucky_seed(self):
        self.add_candidate("v2_seed1")
        self.add_candidate("v2_seed2", 1. - self.y)
        self.config["blend_groups"] = {"v2_equal_pair": ["v2_seed1", "v2_seed2"]}
        self.write_config()
        self.assertEqual(self.freeze()["weights"], {"old_anchor": 1.})

    def test_overlapping_groups_are_rejected(self):
        for name in ["v2_a", "v2_b", "v2_c"]:
            self.add_candidate(name)
        self.config["blend_groups"] = {"pair1": ["v2_a", "v2_b"], "pair2": ["v2_b", "v2_c"]}
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            self.freeze()

    def test_reserved_group_name_is_rejected(self):
        self.add_candidate("v2_a")
        self.add_candidate("v2_b")
        self.config["blend_groups"] = {"anchor": ["v2_a", "v2_b"]}
        self.write_config()
        with self.assertRaises(ValueError):
            self.freeze()

    def test_mixed_old_new_seed_group_and_run_id_collision_are_rejected(self):
        self.add_candidate("v2_a")
        self.add_candidate("v2_b")
        for groups in [{"mixed": ["old_anchor", "v2_a"]},
                       {"v2_a": ["v2_a", "v2_b"]}]:
            with self.subTest(groups=groups):
                self.config["blend_groups"] = groups
                self.write_config()
                with self.assertRaises(ValueError):
                    self.freeze()

    def test_changed_source_predictions_rejected_before_selection(self):
        self.add_candidate("v2_changed")
        path = self.root / "artifacts/runs/v2_changed/oof.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "prediction"] = .333
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "Changed completed predictions"):
            self.freeze()

    def test_changed_selected_config_rejected_after_freeze(self):
        self.add_candidate("v2_better")
        self.freeze()
        self.config["runs"][1]["seed"] += 1
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Changed selected config"):
            release.context(self.config_path)

    def test_missing_v1_member_cannot_trigger_a_refit_or_overwrite(self):
        self.freeze()
        with patch.object(release, "load_data", side_effect=AssertionError("Must fail before fitting")) as loader:
            with self.assertRaisesRegex(ValueError, "refuse to overwrite"):
                release.refit(self.config_path, "old_anchor")
            loader.assert_not_called()
        self.assertFalse((self.root / "artifacts/final").exists())
        self.assertEqual(common.sha256(release.BASELINE), self.baseline_hash)

    def create_native_member(self, spec):
        directory = release.member_path(spec)
        (directory / "model").mkdir(parents=True)
        common.atomic_json(directory / "model/model_metadata.json", {"run": spec["run"], "rounds": spec["rounds"]})
        (directory / "model/native.bin").write_bytes(b"SYNTHETIC_NATIVE_BYTES")
        pd.DataFrame({"id": self.test.id, "prediction": .5}).to_parquet(directory / "test.parquet", index=False)
        common.atomic_json(directory / "done.json", {
            "id": spec["id"], "run": spec["run"], "rounds": spec["rounds"], "rows": 66,
            "frozen_selection_hash": common.sha256(release.BLEND / "frozen.json"),
            "test_hash": common.sha256(directory / "test.parquet"),
            "native_hashes": {p.relative_to(directory).as_posix(): common.sha256(p)
                              for p in (directory / "model").rglob("*") if p.is_file()}})
        return directory

    def test_new_native_member_rejects_changed_missing_or_unrecorded_files(self):
        self.add_candidate("v2_better")
        self.freeze()
        _, specs = release.context(self.config_path)
        spec = next(s for s in specs if s["id"] == "v2_better")
        directory = self.create_native_member(spec)
        np.testing.assert_array_equal(release.validate_member(spec, self.test.id.to_numpy()), np.full(12, .5))
        native = directory / "model/native.bin"
        original = native.read_bytes()
        native.write_bytes(b"CHANGED_NATIVE_BYTES")
        with self.assertRaisesRegex(ValueError, "Changed native"):
            release.validate_member(spec, self.test.id.to_numpy())
        native.write_bytes(original)
        native.unlink()
        with self.assertRaisesRegex(ValueError, "provenance"):
            release.validate_member(spec, self.test.id.to_numpy())
        native.write_bytes(original)
        (directory / "model/unrecorded.bin").write_bytes(b"UNRECORDED")
        with self.assertRaisesRegex(ValueError, "provenance"):
            release.validate_member(spec, self.test.id.to_numpy())

    def test_postfreeze_full_data_refit_uses_fixed_rounds_without_validation(self):
        self.add_candidate("v2_better")
        self.freeze()
        train = pd.DataFrame({"id": np.arange(66), "measurement": np.arange(66, dtype=np.float32),
                              "Class": ["a", "b"] * 33, common.TARGET: np.tile([0, 1], 33)})

        def fake_fit(x, y, valid, y_valid, run, output, rounds):
            self.assertEqual(len(x), 66, "After freeze, production refit includes all labels")
            self.assertIsNone(valid)
            self.assertIsNone(y_valid)
            self.assertEqual(rounds, 3, "Use the frozen median development length")
            output.mkdir(parents=True)
            common.atomic_json(output / "model_metadata.json", {"run": run, "rounds": rounds})
            (output / "native.bin").write_bytes(b"SYNTHETIC_FIT")
            return object(), object(), rounds

        with patch.object(release, "load_data", return_value=(train, self.test, pd.DataFrame())), \
                patch.object(release, "fit_model", side_effect=fake_fit) as fitter, \
                patch.object(release, "predict", return_value=np.full(12, .5)):
            release.refit(self.config_path, "v2_better")
            self.assertEqual(fitter.call_count, 1)
        self.assertEqual(common.sha256(release.BASELINE), self.baseline_hash)
        self.assertFalse((self.root / "artifacts/final").exists())
        with patch.object(release, "load_data", side_effect=AssertionError("Completed native cache should be reused")), \
                patch.object(release, "fit_model", side_effect=AssertionError("No repeat fitting")):
            release.refit(self.config_path, "v2_better")

    def test_raw_schema_and_duplicate_ids_fail_before_loading_models(self):
        self.make_package()
        for name, frame in [
            ("missing", self.test.drop(columns=["measurement"])),
            ("duplicate", pd.concat([self.test, self.test.iloc[:1]], ignore_index=True)),
            ("empty", self.test.iloc[:0]),
        ]:
            with self.subTest(name=name):
                path = self.root / f"{name}.csv"
                frame.to_csv(path, index=False)
                with patch.object(release, "load_model", side_effect=AssertionError("Schema must fail first")) as loader:
                    with self.assertRaisesRegex(ValueError, "schema|IDs"):
                        release.raw_inference(self.config_path, path, release.FINAL / "invalid.csv")
                    loader.assert_not_called()
                self.assertFalse((release.FINAL / "invalid.csv").exists())

    def test_raw_inference_preserves_string_ids_and_checks_probability_bounds(self):
        self.make_package()
        with patch.object(release, "load_model", return_value=(object(), object())), \
                patch.object(release, "predict", return_value=np.linspace(.1, .9, 12)):
            release.raw_inference(self.config_path, self.root / "data/test.csv", release.FINAL / "raw.csv")
        actual = pd.read_csv(release.FINAL / "raw.csv", dtype={"id": str})
        np.testing.assert_array_equal(actual.id, self.test.id)
        with patch.object(release, "load_model", return_value=(object(), object())), \
                patch.object(release, "predict", return_value=np.full(12, np.nan)):
            with self.assertRaisesRegex(ValueError, "probabilities"):
                release.raw_inference(self.config_path, self.root / "data/test.csv", release.FINAL / "invalid.csv")
        self.assertFalse((release.FINAL / "invalid.csv").exists())

    def test_raw_duplicate_headers_and_invalid_numeric_values_fail_before_models(self):
        self.make_package()
        texts = {"duplicate_header": "id,measurement,measurement,Class\n0000,1,2,a\n",
                 "invalid_numeric": "id,measurement,Class\n0000,garbage,a\n",
                 "infinite_numeric": "id,measurement,Class\n0000,inf,a\n",
                 "overflowing_numeric": "id,measurement,Class\n0000,1e99,a\n"}
        for name, text in texts.items():
            path = self.root / f"{name}.csv"
            path.write_text(text, encoding="utf-8")
            with self.subTest(name=name), \
                    patch.object(release, "load_model", side_effect=AssertionError("Reject invalid input before native load")) as loader:
                with self.assertRaises(ValueError), np.errstate(over="ignore"):
                    release.raw_inference(self.config_path, path, release.FINAL / "invalid.csv")
                loader.assert_not_called()
            self.assertFalse((release.FINAL / "invalid.csv").exists())

    def test_manifest_weights_must_match_frozen_selection_before_model_load(self):
        _, _, manifest = self.make_package()
        manifest["members"][0]["weight"] = .5
        common.atomic_json(release.FINAL / "manifest.json", manifest)
        with patch.object(release, "load_model", side_effect=AssertionError("Reject manifest before loading")) as loader:
            with self.assertRaises(ValueError):
                release.raw_inference(self.config_path, self.root / "data/test.csv", release.FINAL / "bad.csv")
            loader.assert_not_called()

    def test_manifest_rejects_changed_run_path_rounds_and_membership(self):
        _, specs, manifest = self.make_package()
        for key, value in [("path", str(self.root / "unrelated/model")), ("rounds", 999),
                           ("run", {"id": "old_anchor", "family": "different"})]:
            changed = copy.deepcopy(manifest)
            changed["members"][0][key] = value
            common.atomic_json(release.FINAL / "manifest.json", changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.checked_manifest(specs)
        manifest["members"] = []
        common.atomic_json(release.FINAL / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "identity"):
            release.checked_manifest(specs)

    def test_raw_output_cannot_overwrite_v1_selection(self):
        self.make_package()
        with patch.object(release, "load_model", return_value=(object(), object())), \
                patch.object(release, "predict", return_value=np.full(12, .5)):
            with self.assertRaises(ValueError):
                release.raw_inference(self.config_path, self.root / "data/test.csv", release.BASELINE)
        self.assertEqual(common.sha256(release.BASELINE), self.baseline_hash)

    def test_verifier_rejects_hash_failure_before_inference(self):
        _, _, manifest = self.make_package()
        manifest["submission_hash"] = "0" * 64
        common.atomic_json(release.FINAL / "manifest.json", manifest)
        with patch.object(release.subprocess, "run", side_effect=AssertionError("No inference after hash failure")) as runner:
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                release.verify(self.config_path)
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
