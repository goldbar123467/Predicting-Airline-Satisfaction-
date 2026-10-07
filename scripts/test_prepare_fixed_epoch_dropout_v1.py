"""Generated packaging-contract tests; never prepare real data or call Kaggle."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import prepare_fixed_epoch_dropout_v1 as prep


class DropoutPreparationTests(unittest.TestCase):
    def policy(self):
        return {"id": prep.ID, **prep.BUDGET, **copy.deepcopy(prep.SCIENCE), "parent_registry_path": prep.PARENT,
            "parent_registry_sha256": prep.PARENT_SHA, "prepared_parent_payload": prep.PARENT_PAYLOAD,
            "parent_bundle_manifest_sha256": prep.PARENT_MANIFEST_SHA, "maximum_gpu_jobs": 1,
            "paid_compute": False, "private_required": True, "training_location": "Kaggle only",
            "cloud_namespace": prep.CLOUD, "dataset_id": prep.DATASET, "kernel_id": prep.KERNEL}

    def test_fixed_policy_rejects_extra_budget_or_altered_arm(self):
        policy = self.policy()
        prep.validate_policy(policy)
        for key, value in [("paid_compute", True), ("hard_timeout_seconds", 7201),
                           ("maximum_gpu_jobs", 2), ("private_required", False), ("treatment_dropout_base", .1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                prep.validate_policy({**policy, key: value})
        policy["logical_endpoints"]["A"]["epoch"] = 4
        with self.assertRaises(ValueError):
            prep.validate_policy(policy)

    def test_parent_manifest_transitively_binds_each_reused_file(self):
        parent = {"source_split_sha256": prep.SPLIT_SHA, "incumbent_selection_sha256": prep.INCUMBENT_SHA,
            "evaluation": prep.EVALUATION, "input_hashes": {name: "a" * 64 for name in prep.OLD_INPUTS},
            "source_hashes": {f"scripts/{name}": "b" * 64 for name in prep.OLD_SOURCES}}
        manifest = {"id": "fixed_epoch_cloud_v1", "registry_sha256": prep.PARENT_SHA,
            "audit_rows_exported": 0, "test_rows_exported": 0, "development_rows": 629671,
            "source_split_sha256": prep.SPLIT_SHA,
            "files": {name: {"sha256": value} for name, value in {**parent["input_hashes"], **parent["source_hashes"]}.items()}}
        prep.validate_parent_manifest(parent, manifest)
        for name in manifest["files"]:
            wrong = copy.deepcopy(manifest)
            wrong["files"][name]["sha256"] = "c" * 64
            with self.subTest(name=name), self.assertRaises(ValueError):
                prep.validate_parent_manifest(parent, wrong)
        with self.assertRaises(ValueError):
            prep.validate_parent_manifest(parent, {**manifest, "audit_rows_exported": 1})

    def test_receipt_requires_current_sources_and_passed_execution(self):
        helper = prep.helpers(prep.ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "adapter.py"
            source.write_text("# synthetic fixture")
            receipt = root / "verification.json"
            value = {"status": "passed", "tests": 2, "returncode": 0,
                     "source_sha256": {"adapter.py": prep.sha(source)}}
            receipt.write_text(json.dumps(value))
            self.assertEqual(prep.receipt_sources(root, receipt, {"adapter.py"}, 2, helper), value["source_sha256"])
            with self.assertRaises(ValueError):
                prep.receipt_sources(root, receipt, {"missing_test.py"}, 2, helper)
            source.write_text("# changed after test")
            with self.assertRaises(ValueError):
                prep.receipt_sources(root, receipt, {"adapter.py"}, 2, helper)
            value.update(status="failed", source_sha256={"adapter.py": prep.sha(source)})
            receipt.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                prep.receipt_sources(root, receipt, {"adapter.py"}, 2, helper)

    def test_bound_copy_preserves_exact_bytes_and_is_exclusive(self):
        helper = prep.helpers(prep.ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source.bin", root / "payload/data.bin"
            source.write_bytes(b"generated\x00fixture\xff")
            helper.copy_bound(source, destination, prep.sha(source))
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            with self.assertRaises(FileExistsError):
                helper.copy_bound(source, destination, prep.sha(source))
            with self.assertRaises(ValueError):
                helper.copy_bound(source, root / "bad.bin", "0" * 64)

    def test_archive_exact_allowlist_roundtrip_and_bad_path_rejection(self):
        helper = prep.helpers(prep.ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload"
            payload.mkdir()
            (payload / "sample.json").write_text('{"generated":true}')
            archive = root / "payload.zip"
            helper.archive_payload(payload, archive, ["sample.json"])
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(zipped.namelist(), ["payload/sample.json"])
                self.assertEqual(zipped.read("payload/sample.json"), (payload / "sample.json").read_bytes())
            for names in [["sample.json", "sample.json"], ["../sample.json"]]:
                with self.subTest(names=names), self.assertRaises(ValueError):
                    helper.archive_payload(payload, root / "bad.zip", names)


if __name__ == "__main__":
    unittest.main()
