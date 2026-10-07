"""Third-pass release contracts using synthetic labels and native-file stubs only."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import common
import third_pass_release as release


class ThirdPassReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="third-pass-release-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(common, "ROOT", self.root))
        for key, value in {
            "ROOT": self.root, "OUT": self.root / "artifacts/third_pass",
            "BLEND": self.root / "artifacts/third_pass/blend",
            "FINAL": self.root / "artifacts/third_pass/final",
            "REPORT": self.root / "THIRD_PASS_REPORT.md",
            "BASELINE": self.root / "artifacts/second_pass/blend/frozen.json",
            "ANCHOR_MANIFEST": self.root / "artifacts/second_pass/final/manifest.json",
        }.items():
            self.stack.enter_context(patch.object(release, key, value))
        (self.root / "data").mkdir()
        self.y = np.tile([0, 1], 30)
        self.ids = np.arange(60)
        self.folds = np.arange(60) % 3
        self.stack.enter_context(patch.object(release, "CLOUD_DEVELOPMENT_ROWS", len(self.ids)))
        pd.DataFrame({"id": np.arange(66), "fold": np.r_[self.folds, [-1] * 6]}).to_parquet(
            self.root / "data/splits.parquet", index=False)
        self.test = pd.DataFrame({"id": [f"{i:04d}" for i in range(12)],
                                  "measurement": np.arange(12, dtype=np.float32), "Class": ["a", "b"] * 6})
        self.test.to_parquet(self.root / "data/test.parquet", index=False)
        self.test.to_csv(self.root / "data/test.csv", index=False)
        common.atomic_json(self.root / "data/manifest.json", {
            "rows": {"train": 66, "test": 12}, "feature_columns": ["measurement", "Class"]})
        self.config_path = self.root / "configs/third_pass.json"
        self.config = {"campaign": "third_pass", "runs": [], "blend_groups": {},
                       "selection_policy": copy.deepcopy(release.REGISTERED_POLICY)}
        self.anchor_specs = []
        for index in range(13):
            name = f"old_{index}" if index < 10 else f"v2_old_{index}"
            run = self.add_candidate(name, np.full(60, .5))
            self.anchor_specs.append({"id": name, "run": run, "rounds": 3, "weight": 1 / 13})
        common.atomic_json(release.BASELINE, {
            "weights": {s["id"]: s["weight"] for s in self.anchor_specs}, "oof_auc": .5,
            "split_hash": common.sha256(self.root / "data/splits.parquet"),
            "audit_auc": "UNREADABLE_SENTINEL", "audit_unexposed_auc": "UNREADABLE_SENTINEL"})
        self.baseline_hash = common.sha256(release.BASELINE)
        self.stack.enter_context(patch.object(release, "BASELINE_SHA", self.baseline_hash))
        anchor_members = []
        provenance = {"v1": {}, "v2": {}}
        for spec in self.anchor_specs:
            old = not spec["id"].startswith("v2_")
            directory = self.root / ("artifacts/final" if old else "artifacts/second_pass/final") / spec["id"]
            self.create_native_member(spec, directory)
            anchor_members.append({**spec, "path": str(directory / "model"), "predictions": str(directory / "test.parquet")})
            provenance["v1" if old else "v2"].update({
                p.relative_to(self.root).as_posix(): common.sha256(p)
                for p in directory.rglob("*") if p.is_file()})
        common.atomic_json(release.ANCHOR_MANIFEST, {"members": anchor_members})
        self.stack.enter_context(patch.object(release, "ANCHOR_MANIFEST_SHA", common.sha256(release.ANCHOR_MANIFEST)))
        for kind, path, key, constant in [
            ("v1", "artifacts/final", "native_model_and_prediction_hashes", "V1_PROVENANCE_SHA"),
            ("v2", "artifacts/second_pass/final", "native_hashes", "V2_PROVENANCE_SHA"),
        ]:
            file = self.root / path / "release_provenance.json"
            common.atomic_json(file, {key: provenance[kind]})
            self.stack.enter_context(patch.object(release, constant, common.sha256(file)))

    def write_config(self):
        common.atomic_json(self.config_path, self.config)

    def add_candidate(self, name, probability=None, **changes):
        run = {"id": name, "family": "synthetic", "seed": 123, **changes}
        self.config["runs"].append(run)
        self.write_config()
        path = self.root / "artifacts/runs" / name
        path.mkdir(parents=True)
        pd.DataFrame({"id": self.ids, "fold": self.folds, common.TARGET: self.y,
                      "prediction": self.y.astype(float) if probability is None else probability}).to_parquet(
            path / "oof.parquet", index=False)
        pd.DataFrame({"id": self.test.id, "prediction": .5}).to_parquet(path / "test.parquet", index=False)
        common.atomic_json(path / "result.json", {
            "id": name, "run": run, "rounds": [2, 3, 4],
            "split_hash": common.sha256(self.root / "data/splits.parquet"),
            "artifacts": {key: common.sha256(path / key) for key in ["oof.parquet", "test.parquet"]}})
        return run

    def create_native_member(self, spec, directory):
        (directory / "model").mkdir(parents=True)
        common.atomic_json(directory / "model/model_metadata.json", {"run": spec["run"], "rounds": spec["rounds"]})
        (directory / "model/native.bin").write_bytes(b"SYNTHETIC_NOT_A_MODEL")
        pd.DataFrame({"id": self.test.id, "prediction": .5}).to_parquet(directory / "test.parquet", index=False)
        frozen = release.BLEND / "frozen.json"
        common.atomic_json(directory / "done.json", {
            "id": spec["id"], "run": spec["run"], "rounds": spec["rounds"], "rows": 66,
            "frozen_selection_hash": common.sha256(frozen) if frozen.exists() else "prior_selection",
            "test_hash": common.sha256(directory / "test.parquet"),
            "native_hashes": {p.relative_to(directory).as_posix(): common.sha256(p)
                              for p in (directory / "model").rglob("*") if p.is_file()}})

    def verify_cloud_fixture(self, run):
        directory = self.root / "artifacts/runs" / run["id"]
        result_path = directory / "result.json"
        result = common.load_config(result_path)
        frame = pd.read_parquet(directory / "oof.parquet")
        measured = release.scores(self.y, frame.prediction.to_numpy(), self.folds)
        result.update(run=copy.deepcopy(run), oof_auc=measured["pooled"], fold_auc=measured["folds"])
        common.atomic_json(result_path, result)
        receipt = {"result_sha256": common.sha256(result_path),
                   "oof_sha256": common.sha256(directory / "oof.parquet"),
                   "test_sha256": common.sha256(directory / "test.parquet"),
                   "split_sha256": common.sha256(self.root / "data/splits.parquet"),
                   "all_heldout_native_inference": True,
                   "heldout_rows": len(self.ids), "run": copy.deepcopy(run),
                   "execution_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
                   "bundle_manifest_sha256": "a5eee31904c7700357a32f174130195f044b47d545dd77cd97579269d741bd96",
                   "runtime_sha256": "b534194f99c10545a50ddd52a2a7e91bd60b2e39aefbb6df1bc4de38ba169235"}
        for field, name in [("source_hashes", "source.txt"), ("native_hashes", "native.stub"),
                            ("heldout_replay_receipt_hashes", "heldout_replay.json")]:
            path = directory / name
            path.write_text(f"synthetic {field}\n", encoding="utf-8")
            receipt[field] = {path.relative_to(self.root).as_posix(): common.sha256(path)}
        common.atomic_json(directory / "cloud_import_verification.json", receipt)
        return receipt

    def freeze(self):
        release.blend(self.config_path, freeze=True)
        return release.context(self.config_path)

    def compatibility_receipt(self, run):
        archive = self.root / 'artifacts/runs' / run['id'] / 'source/cloud_feature_compat.py'
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(release.CLOUD_FEATURE_COMPAT.read_bytes())
        return {'source_versions': dict(release.SOURCE_VERSIONS),
                'feature_compat_sha256': common.sha256(archive),
                'source_hashes': {archive.relative_to(self.root).as_posix(): common.sha256(archive)}}

    def test_cloud_compatibility_requires_versions_live_hash_and_archived_copy(self):
        run = {'id': 'v3_compat', 'execution_backend': 'kaggle', 'categorical_twins': True}
        receipt = self.compatibility_receipt(run)
        release.checked_cloud_feature_compat(run, receipt)
        for key, value in [('source_versions', {}), ('feature_compat_sha256', 'wrong'),
                           ('source_hashes', {}), ('source_hashes', None)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'compatibility'):
                release.checked_cloud_feature_compat(run, {**receipt, key: value})
        archive = self.root / next(iter(receipt['source_hashes']))
        archive.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'compatibility'):
            release.checked_cloud_feature_compat(run, receipt)
        release.checked_cloud_feature_compat({**run, 'execution_backend': 'local'}, {})
        release.checked_cloud_feature_compat({**run, 'categorical_twins': False}, {})

    def test_raw_cloud_adapter_runs_before_aux_only_for_cloud_twins(self):
        runs = [{'id': 'local_twins', 'categorical_twins': True},
                {'id': 'cloud_plain', 'execution_backend': 'kaggle'},
                {'id': 'cloud_twins', 'execution_backend': 'kaggle', 'categorical_twins': True,
                 'original_aux': True}]
        receipt = self.compatibility_receipt(runs[-1])
        common.atomic_json(self.root / 'artifacts/runs/cloud_twins/cloud_import_verification.json', receipt)
        release.FINAL.mkdir(parents=True, exist_ok=True)
        members = [{'id': r['id'], 'run': r, 'weight': 1 / 3, 'path': r['id']} for r in runs]
        events = []
        def adapt(frame, raw, *, run, source_versions):
            self.assertEqual(run['id'], 'cloud_twins')
            self.assertEqual(source_versions, release.SOURCE_VERSIONS)
            self.assertNotIn('orig_aux_00', frame)
            events.append('compat')
            return frame.assign(adapted=True)
        def auxiliary(raw):
            events.append('aux')
            return pd.DataFrame({'orig_aux_00': np.full(len(raw), .2)})
        def prediction(model, transform, frame):
            if str(model) == 'cloud_twins':
                self.assertIn('adapted', frame)
                self.assertIn('orig_aux_00', frame)
            else:
                self.assertNotIn('adapted', frame)
            return np.full(len(frame), .5)
        with patch.object(release, 'context', return_value=({}, [])), \
                patch.object(release, 'checked_manifest', return_value={'members': members}), \
                patch.object(release, 'features', side_effect=lambda raw, run: raw[['measurement']].copy()), \
                patch.object(release, 'cloud_numeric_category_compat', side_effect=adapt) as adapter, \
                patch.object(release, 'load_model', side_effect=lambda path: (path, None)), \
                patch.object(release, 'predict', side_effect=prediction), \
                patch.dict(sys.modules, {'original_aux': SimpleNamespace(predict_aux=auxiliary)}):
            release.raw_inference(self.config_path, self.root / 'data/test.csv', release.FINAL / 'raw.csv')
        adapter.assert_called_once()
        self.assertEqual(events, ['compat', 'aux'])

    def package(self):
        selection, specs = self.freeze()
        for spec in specs:
            directory = release.member_path(spec)
            if not (directory / "done.json").exists():
                self.create_native_member(spec, directory)
        with patch.object(release, "load_data", side_effect=AssertionError("No real fitting")), \
                patch.object(release.subprocess, "run", side_effect=AssertionError("No child fitting")):
            release.refit(self.config_path)
        return selection, specs

    def test_v2_anchor_maps_all_thirteen_prior_members_and_does_not_overwrite(self):
        previous = {p: common.sha256(p) for parent in [self.root / "artifacts/final", self.root / "artifacts/second_pass"]
                    for p in parent.rglob("*") if p.is_file()}
        _, specs = self.package()
        self.assertEqual(len(specs), 13)
        self.assertEqual(sum(release.member_path(s).parent == self.root / "artifacts/final" for s in specs), 10)
        self.assertEqual(sum(release.member_path(s).parent == self.root / "artifacts/second_pass/final" for s in specs), 3)
        manifest = release.checked_manifest(specs)
        self.assertEqual({x["id"] for x in manifest["members"]}, {s["id"] for s in specs})
        for path, digest in previous.items():
            self.assertEqual(common.sha256(path), digest)

    def test_selection_excludes_audit_and_full_training_reads(self):
        self.add_candidate("v3_better")
        reader = pd.read_parquet
        loader = release.load_config

        class GuardedBaseline(dict):
            def __getitem__(self, key):
                if "audit" in key:
                    raise AssertionError("Historical audit used for selection")
                return super().__getitem__(key)

        def guarded_read(path, *args, **kwargs):
            self.assertNotIn(Path(path).name, {"train.parquet", "audit.parquet"})
            return reader(path, *args, **kwargs)

        def guarded_load(path):
            result = loader(path)
            return GuardedBaseline(result) if Path(path) == release.BASELINE else result

        with patch.object(pd, "read_parquet", side_effect=guarded_read), \
                patch.object(release, "load_config", side_effect=guarded_load), \
                patch.object(release, "load_data", side_effect=AssertionError("Full labels unavailable")):
            selection, _ = self.freeze()
        self.assertFalse(selection["audit_evaluated"])
        self.assertEqual(common.sha256(release.BASELINE), self.baseline_hash)

    def test_anchor_selection_and_manifest_hashes_are_required(self):
        with patch.object(release, "BASELINE_SHA", "0" * 64):
            with self.assertRaisesRegex(ValueError, "selection changed"):
                self.freeze()
        with patch.object(release, "ANCHOR_MANIFEST_SHA", "0" * 64):
            with self.assertRaisesRegex(ValueError, "anchor manifest"):
                release.member_path(self.anchor_specs[0])

    def test_postfreeze_anchor_change_is_rejected_before_release_or_member_resolution(self):
        self.freeze()
        original = release.BASELINE.read_bytes()
        # Even semantically unchanged bytes must not silently replace the
        # immutable anchor to which the frozen selection was bound.
        release.BASELINE.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "anchor|baseline|selection"):
            release.context(self.config_path)
        with self.assertRaisesRegex(ValueError, "anchor|baseline|selection"):
            release.member_path(self.anchor_specs[0])

    def test_anchor_bad_config_rounds_and_path_are_rejected(self):
        for key, value in [("run", {"id": "old_0", "family": "wrong"}), ("rounds", 99)]:
            spec = {**self.anchor_specs[0], key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity/configuration"):
                release.member_path(spec)
        changed = common.load_config(release.ANCHOR_MANIFEST)
        changed["members"][0]["path"] = str(self.root / "unrelated/old_0/model")
        common.atomic_json(release.ANCHOR_MANIFEST, changed)
        with patch.object(release, "ANCHOR_MANIFEST_SHA", common.sha256(release.ANCHOR_MANIFEST)):
            with self.assertRaisesRegex(ValueError, "anchor member directory"):
                release.member_path(self.anchor_specs[0])

    def test_changed_provenance_and_native_bytes_fail_for_both_prior_roots(self):
        for spec in [self.anchor_specs[0], self.anchor_specs[-1]]:
            with self.subTest(member=spec["id"]):
                directory = release.member_path(spec)
                provenance = directory.parent / "release_provenance.json"
                original = provenance.read_bytes()
                provenance.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "anchor member provenance"):
                    release.validate_member(spec, self.test.id.to_numpy())
                provenance.write_bytes(original)
                native = directory / "model/native.bin"
                original = native.read_bytes()
                native.write_bytes(b"CHANGED")
                with self.assertRaisesRegex(ValueError, "member artifact"):
                    release.validate_member(spec, self.test.id.to_numpy())
                native.write_bytes(original)

    def test_missing_anchor_member_cannot_fit_or_overwrite(self):
        self.freeze()
        spec = self.anchor_specs[-1]
        done = release.member_path(spec) / "done.json"
        done.unlink()
        with patch.object(release, "load_data", side_effect=AssertionError("No anchor fitting")) as loader:
            with self.assertRaisesRegex(ValueError, "refuse to overwrite"):
                release.refit(self.config_path, spec["id"])
            loader.assert_not_called()
        self.assertFalse(done.exists())

    def test_new_native_changed_extra_missing_and_wrong_selection_fail(self):
        self.add_candidate("v3_better")
        _, specs = self.freeze()
        spec = next(s for s in specs if s["id"] == "v3_better")
        directory = release.member_path(spec)
        self.create_native_member(spec, directory)
        release.validate_member(spec, self.test.id.to_numpy())
        native = directory / "model/native.bin"
        original = native.read_bytes()
        native.write_bytes(b"CHANGED")
        with self.assertRaisesRegex(ValueError, "Changed native"):
            release.validate_member(spec, self.test.id.to_numpy())
        native.unlink()
        with self.assertRaisesRegex(ValueError, "provenance"):
            release.validate_member(spec, self.test.id.to_numpy())
        native.write_bytes(original)
        extra = directory / "model/unrecorded.bin"
        extra.write_bytes(b"EXTRA")
        with self.assertRaisesRegex(ValueError, "provenance"):
            release.validate_member(spec, self.test.id.to_numpy())
        extra.unlink()
        done = common.load_config(directory / "done.json")
        done["frozen_selection_hash"] = "wrong"
        common.atomic_json(directory / "done.json", done)
        with self.assertRaisesRegex(ValueError, "another selection"):
            release.validate_member(spec, self.test.id.to_numpy())

    def test_batch_outputs_and_frozen_replay_are_isolated(self):
        self.freeze()
        prior_path = release.BLEND / "frozen.json"
        prior_hash = common.sha256(prior_path)
        self.config["campaign_id"] = "third_pass_batch02"
        self.write_config()
        release.configure_campaign(self.config)
        self.assertEqual(release.OUT, self.root / "artifacts/third_pass_batch02")
        self.assertEqual(release.REPORT, self.root / "THIRD_PASS_BATCH02_REPORT.md")
        selection, _ = self.freeze()
        self.assertEqual(selection["campaign"], "third_pass_batch02")
        self.assertEqual(common.sha256(prior_path), prior_hash)
        selected_hash = common.sha256(release.BLEND / "frozen.json")
        self.add_candidate("v3_too_late")
        with patch.object(release, "anchored_search", side_effect=AssertionError("Frozen replay reselected")), \
                patch.object(pd, "read_parquet", side_effect=AssertionError("Frozen replay read data")):
            release.blend(self.config_path, freeze=True)
        self.assertEqual(common.sha256(release.BLEND / "frozen.json"), selected_hash)
        with self.assertRaises(ValueError):
            release.configure_campaign({"campaign_id": "../escape"})

    def test_raw_inference_disables_every_id_cache_and_reconstructs_native_sources(self):
        self.add_candidate("v3_all_sources", teacher=True, original_aux=True,
                           original_lgb_teacher=True, original_aux_probability=True)
        self.package()
        source_calls = {
            "original_aux": Mock(return_value=pd.DataFrame({"orig_aux_00": np.full(12, .25)})),
            "original_aux_probability": Mock(return_value=pd.DataFrame({"orig_aux_prob_00": np.full(12, .3)})),
            "original_lgb_teacher": Mock(return_value=np.full(12, .4)),
        }
        modules = {
            "original_aux": SimpleNamespace(predict_aux=source_calls["original_aux"]),
            "original_aux_probability": SimpleNamespace(predict_aux_probability=source_calls["original_aux_probability"]),
            "original_lgb_teacher": SimpleNamespace(predict_teacher=source_calls["original_lgb_teacher"]),
        }
        seen = []

        def no_cached_features(raw, run):
            for key in ["teacher", "original_aux", "original_lgb_teacher", "original_aux_probability"]:
                self.assertIs(run[key], False)
            seen.append(run["id"])
            return raw[["measurement", "Class"]].copy()

        def native_predict(model, transform, frame):
            if "v3_all_sources" in str(model):
                self.assertTrue({"teacher_probability", "orig_aux_00", "orig_aux_prob_00",
                                 "original_lgb_teacher_probability"}.issubset(frame.columns))
            return np.full(len(frame), .5)

        with patch.dict(sys.modules, modules), \
                patch.object(release, "features", side_effect=no_cached_features), \
                patch.object(release, "load_model", side_effect=lambda path: (path, None)) as loader, \
                patch.object(release, "predict", side_effect=native_predict):
            release.raw_inference(self.config_path, self.root / "data/test.csv", release.FINAL / "raw.csv")
        self.assertEqual(len(seen), 14)
        self.assertEqual(loader.call_count, 15)
        for source in source_calls.values():
            source.assert_called_once()
            self.assertEqual(list(source.call_args.args[0]), ["measurement", "Class"])
        actual = pd.read_csv(release.FINAL / "raw.csv", dtype={"id": str})
        np.testing.assert_array_equal(actual.id, self.test.id)
        np.testing.assert_allclose(actual[common.TARGET], .5, rtol=0, atol=1e-12)

    def test_raw_output_cannot_overwrite_prior_or_current_submission(self):
        for path in [release.BASELINE, self.root / "artifacts/second_pass/final/submission.csv",
                     release.FINAL / "submission.csv"]:
            with self.subTest(path=path), patch.object(release, "context", side_effect=AssertionError("Fail before loading")):
                with self.assertRaisesRegex(ValueError, "Inference output"):
                    release.raw_inference(self.config_path, self.root / "data/test.csv", path)

    def test_perfect_cloud_control_is_logged_but_never_selected(self):
        control = self.add_candidate("v3_cloud_control", execution_backend="kaggle")
        self.verify_cloud_fixture(control)
        self.config["selection_exclusions"] = [control["id"]]
        self.write_config()
        selection, _ = self.freeze()
        self.assertNotIn(control["id"], selection["weights"])
        self.assertEqual(selection["addition_history"], [])
        ledger = pd.read_csv(release.OUT / "experiment_ledger.csv")
        self.assertTrue(bool(ledger.loc[ledger.id == control["id"], "selection_excluded"].iloc[0]))

    def test_cloud_candidate_failing_paired_control_cannot_enter_blend(self):
        control = self.add_candidate("v3_cloud_control", execution_backend="kaggle")
        self.verify_cloud_fixture(control)
        imperfect = self.y.astype(float).copy()
        imperfect[1] = 0.0
        candidate = self.add_candidate("v3_cloud_candidate", imperfect, execution_backend="kaggle")
        self.verify_cloud_fixture(candidate)
        self.config.update(selection_exclusions=[control["id"]],
                           comparison_controls={candidate["id"]: control["id"]})
        self.write_config()
        reader = pd.read_parquet

        def guard(path, *args, **kwargs):
            self.assertNotIn(Path(path).name, {"train.parquet", "audit.parquet"})
            return reader(path, *args, **kwargs)

        with patch.object(pd, "read_parquet", side_effect=guard), \
                patch.object(release, "load_data", side_effect=AssertionError("No cloud selection labels")):
            selection, _ = self.freeze()
        self.assertGreater(selection["candidate_scores"][candidate["id"]]["pooled"], .5)
        self.assertNotIn(candidate["id"], selection["weights"])
        self.assertFalse(selection["audit_evaluated"])
        ledger = pd.read_csv(release.OUT / "experiment_ledger.csv")
        row = ledger.loc[ledger.id == candidate["id"]].iloc[0]
        self.assertFalse(bool(row.matched_cloud_gate))
        self.assertTrue(bool(row.selection_excluded))

    def test_cloud_receipt_binds_all_artifacts_and_selected_cloud_cannot_refit_locally(self):
        control = self.add_candidate("v3_cloud_control", np.full(60, .5), execution_backend="kaggle")
        self.verify_cloud_fixture(control)
        candidate = self.add_candidate("v3_cloud_candidate", execution_backend="kaggle")
        receipt = self.verify_cloud_fixture(candidate)
        receipt_path = self.root / "artifacts/runs" / candidate["id"] / "cloud_import_verification.json"
        self.config.update(selection_exclusions=[control["id"]],
                           comparison_controls={candidate["id"]: control["id"]})
        self.write_config()
        scalar_keys = set(receipt) - {"source_hashes", "native_hashes", "heldout_replay_receipt_hashes"}
        for key in scalar_keys:
            invalid = {**receipt, key: False if key == "all_heldout_native_inference" else "0" * 64}
            common.atomic_json(receipt_path, invalid)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Cloud candidate lacks"):
                self.freeze()
            self.assertFalse((release.BLEND / "frozen.json").exists())
        common.atomic_json(receipt_path, receipt)
        selected, _ = self.freeze()
        self.assertGreater(selected["weights"][candidate["id"]], 0)
        with patch.object(release, "load_data", side_effect=AssertionError("No local cloud retraining")) as loader, \
                patch.object(release, "fit_model", side_effect=AssertionError("No local cloud fit")):
            with self.assertRaisesRegex(RuntimeError, "local retraining is forbidden"):
                release.refit(self.config_path, candidate["id"])
            loader.assert_not_called()
        # A seemingly complete local cache must not bypass the required
        # same-image cloud production-refit admission check.
        _, specs = release.context(self.config_path)
        cloud_spec = next(s for s in specs if s["id"] == candidate["id"])
        self.create_native_member(cloud_spec, release.member_path(cloud_spec))
        with patch.object(release, "load_data", side_effect=AssertionError("No cached cloud retraining")):
            with self.assertRaises((RuntimeError, ValueError)):
                release.refit(self.config_path, candidate["id"])

    def test_cloud_source_native_replay_maps_reject_empty_tampered_and_unowned_paths(self):
        run = self.add_candidate("v3_cloud_candidate", execution_backend="kaggle")
        receipt = self.verify_cloud_fixture(run)
        receipt_path = self.root / "artifacts/runs" / run["id"] / "cloud_import_verification.json"
        split_hash = common.sha256(self.root / "data/splits.parquet")
        release.checked_cloud_import(run, split_hash)
        for field in ["source_hashes", "native_hashes", "heldout_replay_receipt_hashes"]:
            for empty in [{}, None, []]:
                common.atomic_json(receipt_path, {**receipt, field: empty})
                with self.subTest(field=field, value=empty), self.assertRaisesRegex(ValueError, "Missing cloud"):
                    release.checked_cloud_import(run, split_hash)
            common.atomic_json(receipt_path, receipt)
            name = next(iter(receipt[field]))
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"tamper")
            with self.subTest(field=field, case="changed bytes"), self.assertRaisesRegex(ValueError, "Changed cloud"):
                release.checked_cloud_import(run, split_hash)
            path.write_bytes(original)
            # Correct bytes/checksum outside this exact owned run still fail.
            outside = self.root / "artifacts/runs/other_run/unowned.bin"
            outside.parent.mkdir(parents=True, exist_ok=True)
            outside.write_bytes(original)
            common.atomic_json(receipt_path, {**receipt, field: {
                outside.relative_to(self.root).as_posix(): common.sha256(outside)}})
            with self.subTest(field=field, case="other run"), self.assertRaisesRegex(ValueError, "Changed cloud"):
                release.checked_cloud_import(run, split_hash)
            common.atomic_json(receipt_path, receipt)

    def test_cloud_candidate_cannot_use_a_local_or_selectable_cloud_control(self):
        control = self.add_candidate("v3_platform_control", np.full(60, .5))
        self.verify_cloud_fixture(control)
        candidate = self.add_candidate("v3_cloud_candidate", execution_backend="kaggle")
        self.verify_cloud_fixture(candidate)
        self.config.update(selection_exclusions=[control["id"]],
                           comparison_controls={candidate["id"]: control["id"]})
        self.write_config()
        with self.assertRaisesRegex(ValueError, "excluded cloud platform control"):
            self.freeze()
        control["execution_backend"] = "kaggle"
        self.verify_cloud_fixture(control)
        self.config["selection_exclusions"] = []
        # Check the candidate first so the specific selectable-control condition,
        # rather than that control's missing comparator, is what fails.
        self.config["runs"].remove(candidate)
        self.config["runs"].insert(13, candidate)
        self.write_config()
        with self.assertRaisesRegex(ValueError, "excluded cloud platform control"):
            self.freeze()

    def test_registered_cloud_horizon_pair_rejects_other_changes_and_wrong_lengths(self):
        control = self.add_candidate("v3_cloud_realmlp_raw_aux_e4", np.full(60, .5),
                                     execution_backend="kaggle", max_rounds=4, params={"width": 8})
        candidate = self.add_candidate("v3_cloud_realmlp_raw_aux_e12", execution_backend="kaggle",
                                       max_rounds=12, params={"width": 8})
        self.verify_cloud_fixture(control)
        self.verify_cloud_fixture(candidate)
        self.config.update(selection_exclusions=[control["id"]],
                           comparison_controls={candidate["id"]: control["id"]})
        for changed, key, value in [(candidate, "seed", 999), (candidate, "params", {"width": 16}),
                                    (candidate, "max_rounds", 8), (control, "max_rounds", 5)]:
            original = copy.deepcopy(changed[key])
            changed[key] = value
            self.verify_cloud_fixture(changed)
            self.write_config()
            with self.subTest(member=changed["id"], key=key), self.assertRaisesRegex(ValueError, "registered horizon"):
                self.freeze()
            changed[key] = original
            self.verify_cloud_fixture(changed)
        self.write_config()
        selection, _ = self.freeze()
        self.assertGreater(selection["weights"][candidate["id"]], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
