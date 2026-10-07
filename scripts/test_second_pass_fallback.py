"""Synthetic-only fallback checks; no native model or competition data is used."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import common
import second_pass_fallback as fallback


class FallbackContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="second-pass-fallback-")
        self.root = Path(self.temporary.name)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(fallback, "ROOT", self.root))
        self.final = self.root / "artifacts/final"
        self.final.mkdir(parents=True)
        (self.root / "data").mkdir()
        self.ids = np.array(["0001", "0002", "0003", "0004", "0005", "0006"])
        self.test = pd.DataFrame({"id": self.ids, "numeric": np.arange(6, dtype=np.float32)})
        self.test.to_csv(self.root / "data/test.csv", index=False)
        self.test.to_parquet(self.root / "data/test.parquet", index=False)
        pd.DataFrame({"id": self.ids, common.TARGET: .5}).to_csv(self.root / "data/sample_submission.csv", index=False)
        self.weights = {"one": .4, "two": .6}
        self.predictions = {"one": np.linspace(.1, .8, 6), "two": np.linspace(.2, .7, 6)}
        self.values = sum(self.weights[name] * self.predictions[name] for name in self.weights)
        common.atomic_csv(pd.DataFrame({"id": self.ids, common.TARGET: self.values}), self.final / "submission.csv")
        self.submission_hash = common.sha256(self.final / "submission.csv")
        self.stack.enter_context(patch.object(fallback, "V1_SUBMISSION_SHA", self.submission_hash))
        self.selection = {"weights": self.weights, "oof_auc": .71, "audit_auc": .70,
                          "audit_unexposed_auc": .69, "audit_sensitivity": {"excluded_rows": 2, "remaining_rows": 8}}
        self.selection_path = self.root / "artifacts/blend/frozen.json"
        common.atomic_json(self.selection_path, self.selection)
        self.selection_hash = common.sha256(self.selection_path)
        self.stack.enter_context(patch.object(fallback, "V1_SELECTION_SHA", self.selection_hash))
        native_hashes, members = {}, []
        for name, weight in self.weights.items():
            directory = self.final / name
            (directory / "model").mkdir(parents=True)
            (directory / "model/native.bin").write_bytes(f"SYNTHETIC_{name}".encode())
            common.atomic_json(directory / "done.json", {"id": name, "rounds": 2})
            # Reverse stored row order so success requires an actual ID join.
            pd.DataFrame({"id": self.ids[::-1], "prediction": self.predictions[name][::-1]}).to_parquet(
                directory / "test.parquet", index=False)
            native_hashes.update({p.relative_to(self.root).as_posix(): common.sha256(p)
                                  for p in directory.rglob("*") if p.is_file()})
            members.append({"id": name, "weight": weight, "rounds": 2, "path": str(directory / "model")})
        common.atomic_json(self.final / "manifest.json", {
            "members": members, "submission_hash": self.submission_hash,
            "frozen_selection_hash": self.selection_hash})
        snapshot = self.final / "reproduction_source/scripts/placeholder.py"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text("# Synthetic archived source only.\n", encoding="utf-8")
        common.atomic_csv(pd.DataFrame({"id": self.ids, common.TARGET: self.values + 1e-8}),
                          self.final / "reproduced_from_raw.csv")
        proof = {"verified_utc": "2026-10-02T04:00:00+00:00", "rows": 6,
                 "expected_submission_sha256": self.submission_hash,
                 "reproduced_submission_sha256": common.sha256(self.final / "reproduced_from_raw.csv"),
                 "frozen_selection_sha256": self.selection_hash,
                 "raw_csv_sha256": common.sha256(self.root / "data/test.csv"),
                 "rtol": 2e-5, "atol": 2e-6, "max_absolute_probability_difference": 1e-8}
        common.atomic_json(self.final / "raw_inference_verification.json", proof)
        archive = self.root / "artifacts/verified" / self.submission_hash
        common.atomic_json(archive / "verification.json", {
            "sha256": self.submission_hash, "rows": 6, "independent_blend_recomputation": True})
        self.provenance = {
            "submission_sha256": self.submission_hash,
            "final_manifest_sha256": common.sha256(self.final / "manifest.json"),
            "native_model_and_prediction_hashes": native_hashes,
            "source_snapshot_hashes": {snapshot.relative_to(self.root).as_posix(): common.sha256(snapshot)},
            "raw_inference_verification_sha256": common.sha256(self.final / "raw_inference_verification.json"),
            "independent_verification_sha256": common.sha256(archive / "verification.json")}
        self.provenance_path = self.final / "release_provenance.json"
        common.atomic_json(self.provenance_path, self.provenance)
        self.stack.enter_context(patch.object(fallback, "V1_PROVENANCE_SHA", common.sha256(self.provenance_path)))
        self.output = self.root / "artifacts/second_pass/fallback_verification.json"

    def tearDown(self):
        self.stack.close()
        self.temporary.cleanup()

    def test_valid_fallback_recomputes_keyed_blend_and_never_mutates_old_files(self):
        before = {p: common.sha256(p) for p in self.root.rglob("*") if p.is_file()}
        original_reader = pd.read_parquet

        def read_only_allowed_files(path, *args, **kwargs):
            self.assertNotIn(Path(path).name, {"train.parquet", "audit.parquet"})
            if Path(path) == self.root / "data/test.parquet":
                self.assertEqual(kwargs.get("columns"), ["id"])
            return original_reader(path, *args, **kwargs)

        with patch.object(pd, "read_parquet", side_effect=read_only_allowed_files), \
                patch.object(common, "load_model", side_effect=AssertionError("Fallback cannot run native inference")), \
                patch.object(common, "load_data", side_effect=AssertionError("Fallback cannot read training labels")):
            result = fallback.verify_v1_fallback("Synthetic v2 refit failure")
        self.assertEqual(result["release_status"], "fallback_v1")
        self.assertEqual(result["submission_path"], str(self.final / "submission.csv"))
        self.assertFalse(result["native_inference_executed"])
        self.assertFalse(result["audit_labels_read"])
        self.assertTrue(result["stored_native_checksums_revalidated"])
        self.assertTrue(result["prior_all_row_raw_proof_revalidated"])
        self.assertEqual(result["historical_v1"]["kaggle_submission_id"], 56771783)
        self.assertEqual(common.load_config(self.output), result)
        report = (self.root / "SECOND_PASS_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("FAILED V2 RELEASE", report)
        self.assertIn("No new native inference", report)
        self.assertIn("0.96093", report)
        for path, digest in before.items():
            self.assertEqual(common.sha256(path), digest, str(path))
        added = {p for p in self.root.rglob("*") if p.is_file()} - set(before)
        self.assertEqual(added, {self.output, self.root / "SECOND_PASS_REPORT.md"})

    def test_tampering_any_trust_or_native_proof_file_fails_without_publication(self):
        paths = [self.final / "submission.csv", self.selection_path, self.provenance_path,
                 self.final / "manifest.json", self.final / "one/model/native.bin",
                 self.final / "one/test.parquet", self.final / "raw_inference_verification.json",
                 self.final / "reproduced_from_raw.csv", self.root / "data/test.csv",
                 self.final / "reproduction_source/scripts/placeholder.py",
                 self.root / "artifacts/verified" / self.submission_hash / "verification.json"]
        for path in paths:
            with self.subTest(path=path.relative_to(self.root)):
                original = path.read_bytes()
                path.write_bytes(original + b"TAMPER")
                try:
                    with self.assertRaises(ValueError):
                        fallback.verify_v1_fallback("Synthetic tampering")
                    self.assertFalse(self.output.exists())
                    self.assertFalse((self.root / "SECOND_PASS_REPORT.md").exists())
                finally:
                    path.write_bytes(original)

    def test_unrecorded_native_file_is_rejected(self):
        (self.final / "one/model/unrecorded.bin").write_bytes(b"EXTRA")
        with self.assertRaisesRegex(ValueError, "inventory"):
            fallback.verify_v1_fallback("Synthetic extra native file")
        self.assertFalse(self.output.exists())

    def test_sample_identifiers_and_schema_are_checked(self):
        path = self.root / "data/sample_submission.csv"
        frames = [pd.DataFrame({"id": self.ids[::-1], common.TARGET: .5}),
                  pd.DataFrame({"id": ["0001"] * 6, common.TARGET: .5}),
                  pd.DataFrame({"id": self.ids, "wrong_target": .5})]
        for number, frame in enumerate(frames):
            with self.subTest(case=number):
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    fallback.verify_v1_fallback("Synthetic invalid sample")
                self.assertFalse(self.output.exists())

    def test_independent_arithmetic_detects_wrong_even_checksum_consistent_predictions(self):
        path = self.final / "one/test.parquet"
        frame = pd.read_parquet(path)
        frame["prediction"] = .99
        frame.to_parquet(path, index=False)
        self.provenance["native_model_and_prediction_hashes"][path.relative_to(self.root).as_posix()] = common.sha256(path)
        common.atomic_json(self.provenance_path, self.provenance)
        # A synthetic re-pinned fixture isolates arithmetic from checksum checks.
        with patch.object(fallback, "V1_PROVENANCE_SHA", common.sha256(self.provenance_path)):
            with self.assertRaisesRegex(AssertionError, "blend recomputation"):
                fallback.verify_v1_fallback("Synthetic incorrect prediction arithmetic")
        self.assertFalse(self.output.exists())

    def test_reason_is_required_before_any_reads_or_writes(self):
        with patch.object(fallback, "sha256", side_effect=AssertionError("No work without a failure reason")):
            for value in [None, "", "  "]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    fallback.verify_v1_fallback(value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
