"""Generated files and fake response objects only; no network, ML or real data."""
import hashlib
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import retrieve_fixed_epoch_dropout_v1 as retrieval


def data(value):
    return json.dumps(value, sort_keys=True).encode()


def details(value):
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def fixture(root, *, failed=False, altered_registry=False, incomplete_prefix=False):
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
        "completed_endpoint_count": 9, "completed_prefix_count": 5 if incomplete_prefix else 6,
        "completed_prefix_native_count": 3, "evaluation_ready": True,
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
    def test_missing_dropout_prefix_completion_rejects_before_assembly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, returned, _, bundle = fixture(root, incomplete_prefix=True)
            with self.assertRaisesRegex(ValueError, "completion counts"):
                retrieval.assemble_workspace(payload, returned, root / "assessment", "COMPLETE",
                    expected_bundle=retrieval.sha(payload / "bundle-manifest.json"), expected_registry=bundle["registry_sha256"])
            self.assertFalse((root / "assessment").exists())

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
                retrieval.assemble_workspace(payload, returned, root / "assessment", "ERROR",
                    expected_bundle=retrieval.sha(payload / "bundle-manifest.json"), expected_registry=bundle["registry_sha256"])
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

class Response:
    url = "https://provider.invalid/get?signature=PRIVATE"
    def __init__(self, chunks=(b"abc",), length="3", encoding=None):
        self.chunks = chunks
        self.headers = {}
        if length is not None:
            self.headers["Content-Length"] = length
        if encoding is not None:
            self.headers["Content-Encoding"] = encoding
        self.iterated = False
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def raise_for_status(self): pass
    def iter_content(self, chunk_size):
        self.iterated = True
        assert chunk_size == 1024 * 1024
        yield from self.chunks


class TransportTests(unittest.TestCase):
    def test_fullpath_listing_requires_no_ambiguous_basename_size_join(self):
        calls = []
        def fetch(token):
            calls.append(token)
            names = ["fixed_epoch_return/cloud_status.json", "other/cloud_status.json"] if not token else [
                "fixed_epoch_return/output-manifest.json", "fixed_epoch_return/results.zip"]
            return SimpleNamespace(files=[SimpleNamespace(file_name=name, url=Response.url) for name in names],
                                   next_page_token="next" if not token else "")
        selected, pages = retrieval.selected_pages(fetch, field="file_name")
        self.assertEqual(set(selected), retrieval.SELECTED)
        self.assertEqual(calls, ["", "next"])
        self.assertNotIn("PRIVATE", json.dumps(pages))

    def test_same_get_length_and_hash_are_authoritative(self):
        with tempfile.TemporaryDirectory() as temporary:
            response = Response((b"a", b"bc"))
            def get(url, **kwargs):
                self.assertEqual(kwargs["headers"], {"Accept-Encoding": "identity"})
                self.assertTrue(kwargs["stream"])
                return response
            result = retrieval.stream_http_length(Response.url, Path(temporary) / "results.zip", remaining_bytes=10, get=get)
            self.assertEqual(result["bytes"], 3)
            self.assertEqual(result["sha256"], hashlib.sha256(b"abc").hexdigest())
            self.assertNotIn("PRIVATE", json.dumps(result))

    def test_missing_invalid_or_excess_length_rejects_before_file_allocation(self):
        for length in (None, "-1", "3.0", "", "99999999999999"):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as temporary:
                response = Response(length=length)
                path = Path(temporary) / "results.zip"
                with self.assertRaises(RuntimeError) as caught:
                    retrieval.stream_http_length(Response.url, path, remaining_bytes=10, get=lambda *a, **k: response)
                self.assertFalse(response.iterated)
                self.assertFalse(path.exists())
                self.assertNotIn("PRIVATE", str(caught.exception))

    def test_short_long_and_encoded_body_fail_without_overwrite(self):
        for response in (Response((b"a",)), Response((b"abcd",)), Response(encoding="gzip")):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "results.zip"
                with self.assertRaises(RuntimeError):
                    retrieval.stream_http_length(Response.url, path, remaining_bytes=10, get=lambda *a, **k: response)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.zip"
            path.write_bytes(b"original")
            with self.assertRaises(RuntimeError):
                retrieval.stream_http_length(Response.url, path, remaining_bytes=10, get=lambda *a, **k: Response())
            self.assertEqual(path.read_bytes(), b"original")


def dispatch_fixture(root):
    import shutil
    payload, returned, status, bundle = fixture(root)
    folder = root / retrieval.FOLDER
    folder.mkdir(parents=True)
    shutil.copytree(payload, root / retrieval.PAYLOAD)
    retrieval.write(folder / "kernel-metadata.json", {"id": retrieval.KERNEL, "is_private": True})
    (folder / "run.py").write_text("# generated dispatch runtime\n", encoding="utf-8")
    prepared = {"id": retrieval.ID, "status": "prepared_not_uploaded", "kernel_id": retrieval.KERNEL,
                "private_required": True, "runtime_sha256": retrieval.sha(folder / "run.py"),
                "registry_sha256": bundle["registry_sha256"],
                "manifest_sha256": retrieval.sha(payload / "bundle-manifest.json")}
    retrieval.write(root / retrieval.PREPARATION, prepared)
    bound = [retrieval.PREPARATION, f"{retrieval.FOLDER}/run.py", f"{retrieval.FOLDER}/kernel-metadata.json"]
    retrieval.write(folder / "dispatch_review.json", {"status": "passed", "hard_timeout_seconds": 7200,
                    "files": {name: retrieval.sha(root / name) for name in bound}})
    retrieval.write(folder / "push_intent.json", {"id": retrieval.KERNEL, "timeout_seconds": 7200,
                    "source_sha256": retrieval.sha(folder / "run.py"),
                    "metadata_sha256": retrieval.sha(folder / "kernel-metadata.json")})
    retrieval.write(folder / "push_receipt.json", {"versionNumber": 1, "kernelId": 12345,
                    "ref": "/code/" + retrieval.KERNEL, "error": None})
    return folder


class DispatchTests(unittest.TestCase):
    def test_receipt_supplies_numeric_identity_and_review_binds_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = dispatch_fixture(root)
            result = retrieval.local_dispatch(root)
            self.assertEqual(result["kernel_id"], 12345)
            self.assertEqual(result["kernel"], retrieval.KERNEL)
            self.assertEqual(result["preparation_manifest_sha256"], retrieval.sha(root / retrieval.PREPARATION))
            self.assertEqual(result["registry_sha256"], retrieval.sha(root / retrieval.PAYLOAD / "registry.json"))
            with (root / retrieval.PREPARATION).open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Reviewed dispatch bytes changed"):
                retrieval.local_dispatch(root)

    def test_invalid_dispatch_identity_fails_closed(self):
        for change in ({"kernelId": True}, {"kernelId": 0}, {"versionNumber": 2}, {"error": "provider error"},
                       {"ref": "/code/unrelated/wrong"}, {"invalidDatasetSources": ["wrong"]}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                folder = dispatch_fixture(root)
                path = folder / "push_receipt.json"
                value = retrieval.read(path)
                path.write_bytes(data({**value, **change}))
                with self.assertRaisesRegex(ValueError, "Unique dispatched kernel identity"):
                    retrieval.local_dispatch(root)

    def test_original_sources_and_pure_archive_logic_unchanged(self):
        import ast
        root = Path(__file__).resolve().parents[1]
        frozen = root / "scripts/retrieve_fixed_epoch_cloud_v1.py"
        self.assertEqual(retrieval.sha(frozen), "17fe4bed4a622ffbe14860b65bddf711bb2fcbab0520a316ebfde0f3cfeeecaf")
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef)}
        old, new = functions(frozen), functions(Path(retrieval.__file__))
        for name in ("sha", "read", "write", "safe_name", "checked", "inventory_names", "validate_item",
                     "output_allowed", "validate_archive", "assess_success", "selected_pages"):
            self.assertEqual(old[name], new[name], name)
        transport = root / "scripts/retrieve_fixed_epoch_cloud_transport_v2.py"
        self.assertEqual(retrieval.sha(transport), "83f7b09b6369fadf8408d065441df482fcd79ca0822d96992985cc5f4e6cd80d")
        class UnqualifyBase(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == "base":
                    return ast.Name(id=node.attr, ctx=node.ctx)
                return self.generic_visit(node)
        original = next(node for node in ast.parse(transport.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "stream_http_length")
        self.assertEqual(ast.dump(UnqualifyBase().visit(original), include_attributes=False), new["stream_http_length"])


if __name__ == "__main__":
    unittest.main()
