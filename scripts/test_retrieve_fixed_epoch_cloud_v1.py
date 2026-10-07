"""Generated files and fake response objects only; no network, ML or real data."""
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import retrieve_fixed_epoch_cloud_v1 as retrieval


def data(value):
    return json.dumps(value, sort_keys=True).encode()


def details(value):
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def fixture(root, *, failed=False, altered_registry=False):
    payload, returned = root / "payload", root / "download/fixed_epoch_return"
    payload.mkdir()
    returned.mkdir(parents=True)
    registry = data({"id": retrieval.ID, "fixture": "generated"})
    wrapper = data({"id": retrieval.ID, "registry_sha256": details(registry)["sha256"]})
    payload_files = {"registry.json": registry, f"configs/{retrieval.ID}.json": wrapper,
                     "data/generated.json": b"{}", "scripts/generated.py": b"value=1\n"}
    for name, value in payload_files.items():
        path = payload / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    bundle = {"id": retrieval.ID, "registry_sha256": details(registry)["sha256"],
              "files": {name: details(value) for name, value in payload_files.items()}}
    (payload / "bundle-manifest.json").write_bytes(data(bundle))
    bindings = {"id": retrieval.ID, "registry_sha256": details(registry)["sha256"],
                "campaign_sha256": details(wrapper)["sha256"]}
    completed = data({**bindings, "trajectories": [], "endpoints": []})
    complete = data({**bindings, "status": "completed", "completed_fit_count": 12,
        "completed_endpoint_count": 9, "evaluation_ready": True,
        "completed_manifest_sha256": details(completed)["sha256"]})
    state = data({**bindings, "status": "training_complete", "completed_fits": 12,
                  "active_child": None, "outer_metrics_computed": False})
    output_files = {"registry.json": registry if not altered_registry else b"wrong",
        f"configs/{retrieval.ID}.json": wrapper, f"artifacts/{retrieval.ID}/completion_receipt.json": complete,
        f"artifacts/{retrieval.ID}/completed_manifest.json": completed, f"state/{retrieval.ID}/run_state.json": state,
        f"artifacts/{retrieval.ID}/model.bin": b"opaque generated model bytes, never deserialized"}
    manifest = {"id": retrieval.ID, "files": {name: details(value) for name, value in output_files.items()}}
    manifest_bytes = data(manifest)
    (returned / "output-manifest.json").write_bytes(manifest_bytes)
    with zipfile.ZipFile(returned / "results.zip", "w") as archive:
        for name, value in output_files.items():
            archive.writestr(name, value)
        archive.writestr("output-manifest.json", manifest_bytes)
    status = {"id": retrieval.ID, "status": "failed" if failed else "training_complete",
        "completed_fits": 12, "outer_metrics_computed": False, "local_training": False,
        "returned_artifacts": {"archive_sha256": retrieval.sha(returned / "results.zip"),
            "manifest_sha256": details(manifest_bytes)["sha256"], "file_count": len(output_files)}}
    (returned / "cloud_status.json").write_bytes(data(status))
    return payload, returned, status, bundle


class RetrievalTests(unittest.TestCase):
    def test_success_assembly_retains_exact_evaluator_inventory_and_originals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, returned, status, bundle = fixture(root)
            destination = root / "assessment"
            result = retrieval.assemble_workspace(payload, returned, destination, "COMPLETE",
                expected_bundle=retrieval.sha(payload / "bundle-manifest.json"), expected_registry=bundle["registry_sha256"])
            self.assertEqual(result["status"], "assembled_unscored")
            manifest = retrieval.read(destination / "output-manifest.json")
            actual = {"registry.json", f"configs/{retrieval.ID}.json"}
            for part in ("artifacts", "state", "logs"):
                actual.update(p.relative_to(destination).as_posix() for p in (destination / part).rglob("*") if p.is_file())
            self.assertEqual(actual, set(manifest["files"]))
            self.assertEqual(retrieval.sha(returned / "results.zip"), status["returned_artifacts"]["archive_sha256"])
            self.assertFalse((destination / "cloud_status.json").exists())
            with self.assertRaises(FileExistsError):
                retrieval.assemble_workspace(payload, returned, destination, "COMPLETE",
                    expected_bundle=retrieval.sha(payload / "bundle-manifest.json"), expected_registry=bundle["registry_sha256"])

    def test_failed_terminal_never_creates_assessment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, returned, status, bundle = fixture(root, failed=True)
            retrieval.validate_archive(returned, status)
            with self.assertRaises(ValueError):
                retrieval.assemble_workspace(payload, returned, root / "assessment", "ERROR")
            self.assertFalse((root / "assessment").exists())

    def test_registry_conflict_rejected_before_any_assembly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, returned, _, bundle = fixture(root, altered_registry=True)
            with self.assertRaises(ValueError):
                retrieval.assemble_workspace(payload, returned, root / "assessment", "COMPLETE",
                    expected_bundle=retrieval.sha(payload / "bundle-manifest.json"), expected_registry=bundle["registry_sha256"])
            self.assertFalse((root / "assessment").exists())

    def test_completion_metadata_cap_precedes_archive_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, returned, _, bundle = fixture(root)
            manifest = retrieval.read(returned / "output-manifest.json")
            manifest["files"][f"artifacts/{retrieval.ID}/completion_receipt.json"]["bytes"] = retrieval.MAX_METADATA_BYTES + 1
            with patch.object(retrieval, "validate_archive", return_value=manifest), \
                    patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("must reject before allocation")):
                with self.assertRaisesRegex(ValueError, "metadata exceeds bounded size"):
                    retrieval.assemble_workspace(payload, returned, root / "assessment", "COMPLETE",
                        expected_bundle=retrieval.sha(payload / "bundle-manifest.json"), expected_registry=bundle["registry_sha256"])
            self.assertFalse((root / "assessment").exists())

    def test_archive_outer_digest_and_member_digest_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, returned, status, _ = fixture(root)
            changed = {**status, "returned_artifacts": {**status["returned_artifacts"], "archive_sha256": "0" * 64}}
            with self.assertRaises(ValueError):
                retrieval.validate_archive(returned, changed)
            with zipfile.ZipFile(returned / "results.zip") as archive:
                contents = {item.filename: archive.read(item) for item in archive.infolist()}
            contents[f"artifacts/{retrieval.ID}/model.bin"] = b"x" * len(contents[f"artifacts/{retrieval.ID}/model.bin"])
            with zipfile.ZipFile(returned / "results.zip", "w") as archive:
                for name, value in contents.items():
                    archive.writestr(name, value)
            status["returned_artifacts"]["archive_sha256"] = retrieval.sha(returned / "results.zip")
            with self.assertRaises(ValueError):
                retrieval.validate_archive(returned, status)

    def test_zip_symlink_rejected_even_with_matching_names_and_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, returned, status, _ = fixture(root)
            with zipfile.ZipFile(returned / "results.zip") as archive:
                contents = {item.filename: archive.read(item) for item in archive.infolist()}
            with zipfile.ZipFile(returned / "results.zip", "w") as archive:
                for name, value in contents.items():
                    info = zipfile.ZipInfo(name)
                    if name.endswith("model.bin"):
                        info.create_system = 3
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    archive.writestr(info, value)
            status["returned_artifacts"]["archive_sha256"] = retrieval.sha(returned / "results.zip")
            with self.assertRaises(ValueError):
                retrieval.validate_archive(returned, status)

    def test_portable_paths_duplicate_and_prefix_collision_rejected(self):
        for names in (["../x"], ["/absolute"], ["a\\x"], ["C:x"], ["a/./x"], ["CON.txt"],
                      ["a", "a"], ["A", "a"], ["a", "a/x"], ["a /b"], ["a//b"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                retrieval.inventory_names(names)

    def test_official_pagination_is_exhausted_and_filters_exact_names(self):
        calls = []
        def fetch(token):
            calls.append(token)
            entries = [SimpleNamespace(name="fixed_epoch_workspace/payload/private.csv"),
                       SimpleNamespace(name="fixed_epoch_return/cloud_status.json")] if not token else [SimpleNamespace(name="fixed_epoch_return/results.zip")]
            return SimpleNamespace(files=entries, next_page_token="next" if not token else "")
        selected, pages = retrieval.selected_pages(fetch, field="name")
        self.assertEqual(calls, ["", "next"])
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(pages), 2)
        self.assertNotIn("next", json.dumps(pages).replace("has_next_page", ""))

    def test_repeated_pagination_token_rejected(self):
        def fetch(token):
            return SimpleNamespace(files=[], next_page_token="loop")
        with self.assertRaises(ValueError):
            retrieval.selected_pages(fetch, field="file_name")

    def test_stream_size_cap_hash_and_secret_free_error(self):
        class Response:
            url = "https://official.example/output?signature=DO_NOT_RECORD"
            headers = {"Content-Length": "3"}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def raise_for_status(self): pass
            def iter_content(self, chunk_size):
                self.chunk_size = chunk_size
                yield b"a"
                yield b"bc"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = retrieval.stream_download(Response.url, root / "results.zip", 3, get=lambda *a, **k: Response())
            self.assertEqual(result["sha256"], hashlib.sha256(b"abc").hexdigest())
            self.assertNotIn("DO_NOT_RECORD", json.dumps(result))
            with self.assertRaises(RuntimeError) as raised:
                retrieval.stream_download(Response.url, root / "wrong.zip", 2, get=lambda *a, **k: Response())
            self.assertNotIn("DO_NOT_RECORD", str(raised.exception))
            with self.assertRaises(ValueError):
                retrieval.stream_download(Response.url, root / "too_big.zip", retrieval.MAX_METADATA_BYTES + 1,
                                           get=lambda *a, **k: self.fail("must reject before network"))


if __name__ == "__main__":
    unittest.main()
