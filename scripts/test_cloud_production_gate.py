"""Synthetic receipt/native-byte tests; no real labels, models, fitting or GPU."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import third_pass_release as release


class CloudProductionGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cloud-production-proof-")
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name)
        self.root = self.temporary / "project"
        self.root.mkdir()
        self.blend = self.root / "artifacts/third_pass_batch03/blend"
        self.blend.mkdir(parents=True)
        self.run = {"id": "v3_cloud_fixture", "family": "realmlp_cat",
                    "execution_backend": "kaggle", "seed": 20261005, "max_rounds": 12}
        self.spec = {"id": self.run["id"], "run": self.run, "weight": .1, "rounds": 7}
        self.directory = self.blend.parent / "final" / self.run["id"]
        (self.directory / "model").mkdir(parents=True)
        self.cv_dir = self.root / "artifacts/runs" / self.run["id"]
        self.cv_dir.mkdir(parents=True)
        (self.root / "data").mkdir()
        (self.root / "data/splits.parquet").write_bytes(b"SYNTHETIC_SPLIT_IDS")
        (self.blend / "frozen.json").write_text('{"weights":{"v3_cloud_fixture":0.1}}')
        (self.cv_dir / "result.json").write_text('{"synthetic":"cv result"}')
        (self.cv_dir / "cloud_import_verification.json").write_text('{"synthetic":"CV proof"}')
        (self.directory / "test.parquet").write_bytes(b"SYNTHETIC_TEST_PREDICTIONS")
        self.native = self.directory / "model/graph-fixture.pt"
        self.native.write_bytes(b"NOT_A_REAL_MODEL")
        self.provenance = self.root / "cloud/fixture/remote_metadata.json"
        self.provenance.parent.mkdir(parents=True)
        self.provenance.write_text('{"isPrivate":true,"fixture":true}')
        self.cv = {"execution_image": "pinned-test-image", "runtime_sha256": "test-runtime-digest"}
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(release, "ROOT", self.root))
        stack.enter_context(patch.object(release, "BLEND", self.blend))
        self.check_cv = stack.enter_context(patch.object(release, "checked_cloud_import", return_value=self.cv))
        stack.enter_context(patch.object(release, "load_data", side_effect=AssertionError("No labels or fitting")))
        self.proof_path = self.directory / "cloud_production_verification.json"
        self.proof = {
            "status": "verified", "id": self.run["id"], "run": self.run,
            "rows": 699635, "rounds": 7, "test_rows": 299844,
            "frozen_selection_hash": release.sha256(self.blend / "frozen.json"),
            "cv_result_sha256": release.sha256(self.cv_dir / "result.json"),
            "cv_import_receipt_sha256": release.sha256(self.cv_dir / "cloud_import_verification.json"),
            **self.cv, "all_test_native_inference": True, "local_inference_device": "cpu",
            "test_hash": release.sha256(self.directory / "test.parquet"),
            "cloud_test_sha256": release.sha256(self.directory / "test.parquet"),
            "rtol": 1e-5, "atol": 2e-6, "full_test_max_abs_error": 1e-7,
            "native_hashes": {self.native.relative_to(self.root).as_posix(): release.sha256(self.native)},
            "production_provenance_hashes": {
                self.provenance.relative_to(self.root).as_posix(): release.sha256(self.provenance)},
        }
        self.done = {}
        self.write_proof(self.proof)

    def write_proof(self, proof):
        # Permit NaN only in the deliberately malformed synthetic receipt test.
        self.proof_path.write_text(json.dumps(proof), encoding="utf-8")
        self.done = {"cloud_production_verification_sha256": release.sha256(self.proof_path)}

    def check(self):
        return release.validate_cloud_production(self.spec, self.directory, self.done)

    def test_valid_proof_and_cv_revalidation(self):
        self.assertEqual(self.check(), self.proof)
        self.check_cv.assert_called_once_with(self.run, release.sha256(self.root / "data/splits.parquet"))

    def test_cloud_twins_production_binds_adapter_and_source_versions(self):
        self.run['categorical_twins'] = True
        helper = self.root / 'scripts/cloud_feature_compat.py'
        helper.parent.mkdir()
        helper.write_bytes(release.CLOUD_FEATURE_COMPAT.read_bytes())
        self.cv.update(source_versions=dict(release.SOURCE_VERSIONS), feature_compat_sha256=release.sha256(helper))
        proof = copy.deepcopy(self.proof)
        proof.update(source_versions=self.cv['source_versions'], feature_compat_sha256=self.cv['feature_compat_sha256'])
        proof['production_provenance_hashes']['scripts/cloud_feature_compat.py'] = release.sha256(helper)
        self.write_proof(proof)
        self.check()
        for field, value in [('source_versions', {}), ('feature_compat_sha256', 'wrong')]:
            self.write_proof({**proof, field: value})
            with self.assertRaisesRegex(ValueError, 'proof mismatch'):
                self.check()
        changed = copy.deepcopy(proof)
        del changed['production_provenance_hashes']['scripts/cloud_feature_compat.py']
        self.write_proof(changed)
        with self.assertRaisesRegex(ValueError, 'compatibility provenance'):
            self.check()
        self.write_proof(proof)
        helper.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            self.check()

    def test_each_identity_platform_scope_and_tolerance_field_is_bound(self):
        invalid = {
            "status": "pending", "id": "other", "run": {**self.run, "seed": 1},
            "rows": 699634, "rounds": 8, "test_rows": 299843,
            "frozen_selection_hash": "wrong", "cv_result_sha256": "wrong",
            "cv_import_receipt_sha256": "wrong", "execution_image": "different-image",
            "runtime_sha256": "wrong", "all_test_native_inference": False,
            "local_inference_device": "cuda", "test_hash": "wrong", "cloud_test_sha256": "wrong",
            "rtol": 1e-4, "atol": 2e-5,
        }
        for key, value in invalid.items():
            with self.subTest(field=key):
                self.write_proof({**self.proof, key: value})
                with self.assertRaisesRegex(ValueError, "proof mismatch"):
                    self.check()

    def test_nonfinite_negative_and_excessive_error_are_rejected(self):
        for value in [float("nan"), float("inf"), -1e-9, 1.20001e-5]:
            with self.subTest(error=value):
                self.write_proof({**self.proof, "full_test_max_abs_error": value})
                with self.assertRaisesRegex(ValueError, "tolerance"):
                    self.check()
        missing = {k: v for k, v in self.proof.items() if k != "full_test_max_abs_error"}
        self.write_proof(missing)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            self.check()

    def test_missing_or_changed_receipt_cannot_use_cached_done(self):
        original = self.proof_path.read_bytes()
        self.proof_path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "Unverified"):
            self.check()
        self.proof_path.unlink()
        with self.assertRaisesRegex(ValueError, "Unverified"):
            self.check()

    def test_frozen_cv_and_test_file_mutations_are_rejected(self):
        for file in [self.blend / "frozen.json", self.cv_dir / "result.json",
                     self.cv_dir / "cloud_import_verification.json", self.directory / "test.parquet"]:
            with self.subTest(file=file.name):
                original = file.read_bytes()
                file.write_bytes(original + b" ")
                try:
                    with self.assertRaisesRegex(ValueError, "proof mismatch"):
                        self.check()
                finally:
                    file.write_bytes(original)

    def test_exact_native_file_set_and_native_bytes_are_required(self):
        for value in [None, {}, []]:
            self.write_proof({**self.proof, "native_hashes": value})
            with self.assertRaisesRegex(ValueError, "file set"):
                self.check()
        self.write_proof(self.proof)
        extra = self.native.with_name("unexpected.bin")
        extra.write_bytes(b"EXTRA")
        with self.assertRaisesRegex(ValueError, "file set"):
            self.check()
        extra.unlink()
        self.native.write_bytes(b"CHANGED_MODEL")
        with self.assertRaisesRegex(ValueError, "artifact"):
            self.check()
        self.native.unlink()
        with self.assertRaisesRegex(ValueError, "file set"):
            self.check()

    def test_provenance_cannot_be_empty_changed_deleted_or_outside_project(self):
        for value in [None, {}, []]:
            self.write_proof({**self.proof, "production_provenance_hashes": value})
            with self.assertRaisesRegex(ValueError, "provenance"):
                self.check()
        self.write_proof(self.proof)
        self.provenance.write_text("CHANGED")
        with self.assertRaisesRegex(ValueError, "artifact"):
            self.check()
        self.provenance.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.check()
        outside = self.temporary / "outside.json"
        outside.write_text("outside fixture")
        self.write_proof({**self.proof, "production_provenance_hashes": {
            "../outside.json": release.sha256(outside)}})
        with self.assertRaisesRegex(ValueError, "artifact|provenance"):
            self.check()

    def test_provenance_digest_cannot_mask_a_changed_native_digest(self):
        self.native.write_bytes(b"TAMPERED_NATIVE")
        proof = copy.deepcopy(self.proof)
        proof["production_provenance_hashes"][self.native.relative_to(self.root).as_posix()] = release.sha256(self.native)
        self.write_proof(proof)
        with self.assertRaisesRegex(ValueError, "overlap|native|provenance|artifact"):
            self.check()


if __name__ == "__main__":
    unittest.main()
