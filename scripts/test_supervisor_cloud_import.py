"""Cloud admission and local-dispatch regressions using only opaque fake files."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import supervisor


class CloudImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="supervisor-cloud-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = patch.object(supervisor, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.run = {"id": "v3_cloud_fixture", "family": "realmlp_cat", "execution_backend": "kaggle"}
        self.config = {"campaign": "third_pass", "runs": [self.run],
                       "stop_new_runs_utc": "2026-10-02T15:45:00Z",
                       "finalize_utc": "2026-10-02T16:00:00Z",
                       "deadline_utc": "2026-10-02T17:00:00Z"}
        self.config_path = self.root / "config.json"
        supervisor.atomic_json(self.config_path, self.config)
        self.instance = supervisor.Supervisor(self.config_path, self.config)
        self.directory = self.root / "artifacts/runs" / self.run["id"]
        self.directory.mkdir(parents=True)
        self.result_path = self.directory / "result.json"
        self.receipt_path = self.directory / "cloud_import_verification.json"
        split = self.root / "data/splits.parquet"
        split.parent.mkdir()
        split.write_bytes(b"synthetic fixed split")
        self.result = {"run": self.run, "split_hash": self.sha(split), "artifacts": {}}
        for name in ["oof.parquet", "test.parquet"]:
            path = self.directory / name
            path.write_bytes(name.encode())
            self.result["artifacts"][name] = self.sha(path)
        supervisor.atomic_json(self.result_path, self.result)
        self.receipt = {"status": "verified", "run_id": self.run["id"], "run": self.run,
                        "all_heldout_native_inference": True,
                        "result_sha256": self.sha(self.result_path),
                        "oof_sha256": self.result["artifacts"]["oof.parquet"],
                        "test_sha256": self.result["artifacts"]["test.parquet"],
                        "split_sha256": self.result["split_hash"]}
        for field, name in [("source_hashes", "source/frozen.py"),
                            ("native_hashes", "fold_0/model/native.pt"),
                            ("heldout_replay_receipt_hashes", "heldout_replay.json")]:
            path = self.directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic proof " + name.encode())
            self.receipt[field] = {path.relative_to(self.root).as_posix(): self.sha(path)}
        self.write_receipt()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_receipt(self):
        supervisor.atomic_json(self.receipt_path, self.receipt)

    def test_verified_cloud_accepts_exact_two_artifacts_without_audit(self):
        self.assertEqual(self.instance.validate_completed(self.result_path, self.run), self.result)
        self.assertEqual(self.instance.read_config(), self.config)
        self.assertFalse((self.directory / "audit.parquet").exists())

    def test_each_receipt_identity_hash_and_inference_claim_is_required(self):
        original = copy.deepcopy(self.receipt)
        for key in ["status", "run_id", "run", "result_sha256", "oof_sha256",
                    "test_sha256", "split_sha256", "all_heldout_native_inference"]:
            with self.subTest(key=key):
                self.receipt = copy.deepcopy(original)
                self.receipt.pop(key)
                self.write_receipt()
                with self.assertRaises(ValueError):
                    self.instance.validate_completed(self.result_path, self.run)

    def test_every_proof_map_rejects_empty_tampered_and_escaped_paths(self):
        original = copy.deepcopy(self.receipt)
        outside = self.root / "outside.txt"
        outside.write_bytes(b"not owned")
        for key in ["source_hashes", "native_hashes", "heldout_replay_receipt_hashes"]:
            for replacement in [{}, {"outside.txt": self.sha(outside)},
                                {next(iter(original[key])): "0" * 64}]:
                with self.subTest(key=key, replacement=replacement):
                    self.receipt = copy.deepcopy(original)
                    self.receipt[key] = replacement
                    self.write_receipt()
                    with self.assertRaises(ValueError):
                        self.instance.validate_completed(self.result_path, self.run)

    def test_cloud_rejects_audit_or_missing_test_artifact(self):
        original = copy.deepcopy(self.result)
        for artifacts in [{"oof.parquet": original["artifacts"]["oof.parquet"]},
                          {**original["artifacts"], "audit.parquet": "0" * 64}]:
            self.result["artifacts"] = artifacts
            supervisor.atomic_json(self.result_path, self.result)
            with self.assertRaisesRegex(ValueError, "artifact manifest"):
                self.instance.validate_completed(self.result_path, self.run)

    def test_local_still_requires_audit(self):
        local = {"id": self.run["id"], "family": "realmlp_cat"}
        self.result["run"] = local
        supervisor.atomic_json(self.result_path, self.result)
        with self.assertRaisesRegex(ValueError, "artifact manifest"):
            self.instance.validate_completed(self.result_path, local)
        audit = self.directory / "audit.parquet"
        audit.write_bytes(b"synthetic audit")
        self.result["artifacts"]["audit.parquet"] = self.sha(audit)
        supervisor.atomic_json(self.result_path, self.result)
        self.assertEqual(self.instance.validate_completed(self.result_path, local), self.result)

    def test_earlier_campaigns_reject_remote_backend(self):
        for campaign in ["overnight", "second_pass"]:
            with self.subTest(campaign=campaign):
                self.config["campaign"] = campaign
                supervisor.atomic_json(self.config_path, self.config)
                with self.assertRaisesRegex(ValueError, "only in third_pass"):
                    supervisor.load_config(self.config_path)

    def test_missing_cloud_result_refuses_before_recovery_or_local_training(self):
        self.result_path.unlink()
        with patch.object(self.instance, "recover") as recovery, patch.object(self.instance, "child") as child:
            with self.assertRaisesRegex(ValueError, "local training is forbidden"):
                self.instance.run()
            recovery.assert_not_called()
            child.assert_not_called()

    def test_missing_receipt_refuses_before_recovery_or_local_training(self):
        self.receipt_path.unlink()
        with patch.object(self.instance, "recover") as recovery, patch.object(self.instance, "child") as child:
            with self.assertRaises(FileNotFoundError):
                self.instance.run()
            recovery.assert_not_called()
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
