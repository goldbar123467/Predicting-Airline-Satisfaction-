"""Generated runtime-amendment tests, without CUDA, training or remote operations."""
import ast
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_fixed_epoch_dropout_retry1 as retry


class RuntimeRetryTests(unittest.TestCase):
    def original(self):
        return '''from pathlib import Path
from datetime import timedelta
import os, hashlib

def bootstrap() -> None:
        install_minimal(out, result)
        runtime = {"delivery_deadline_utc": (started + timedelta(seconds=7140)).isoformat(),
                   "hard_timeout_seconds": 7200, "bundle_manifest_sha256": EXPECTED_BUNDLE_MANIFEST}
        if registry["hard_timeout_seconds"] != 7200:
            raise ValueError("Registered limit")
        smoke()
'''

    def test_only_startup_and_operational_deadlines_change(self):
        original = self.original()
        generated = retry.render_entry(original, '# generated hook\n', '{"generated":true}\n')
        tree = ast.parse(generated)
        self.assertEqual([node.name for node in tree.body if isinstance(node, ast.FunctionDef)],
                         ["install_deterministic_runtime", "bootstrap"])
        bootstrap = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "bootstrap"][0]
        calls = [node.value.func.id for node in bootstrap.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
        self.assertEqual(calls, ["install_minimal", "install_deterministic_runtime", "smoke"])
        self.assertIn('registry["hard_timeout_seconds"] != 7200', generated)
        self.assertIn('"registered_hard_timeout_seconds": 7200', generated)
        self.assertIn('timedelta(seconds=6840)', generated)
        self.assertIn('"hard_timeout_seconds": 6900', generated)

    def test_hook_installation_preserves_bytes_outside_payload_and_sets_child_environment(self):
        hook, amendment = '# generated CRLF hook\r\n', '{"generated":true}\n'
        source = retry.render_entry(self.original(), hook, amendment)
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "install_deterministic_runtime")
        namespace = {"Path": Path, "hashlib": hashlib, "os": os, "digest": retry.sha}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<generated installer>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            workspace = Path(directory)
            payload, out = workspace / "payload", workspace / "payload/artifacts/fixed_epoch_dropout_v1"
            out.mkdir(parents=True)
            (payload / "scripts").mkdir()
            sentinel = payload / "registry.json"
            sentinel.write_bytes(b'untouched scientific fixture')
            namespace["install_deterministic_runtime"](payload, out)
            installed = workspace / "deterministic_startup/sitecustomize.py"
            self.assertFalse(installed.is_relative_to(payload))
            self.assertEqual(installed.read_bytes(), hook.encode())
            self.assertEqual((out / "runtime_amendment/sitecustomize.py").read_bytes(), hook.encode())
            self.assertEqual((out / "runtime_amendment/registration.json").read_bytes(), amendment.encode())
            self.assertEqual(sentinel.read_bytes(), b'untouched scientific fixture')
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ':4096:8')
            self.assertEqual(os.environ["FIXED_EPOCH_DETERMINISTIC_HOOK_SHA256"], hashlib.sha256(hook.encode()).hexdigest())
            self.assertEqual(os.environ["FIXED_EPOCH_DETERMINISTIC_AMENDMENT_SHA256"], hashlib.sha256(amendment.encode()).hexdigest())
            self.assertEqual(Path(os.environ["FIXED_EPOCH_DETERMINISTIC_ROOT"]), payload.resolve())
            self.assertEqual(Path(os.environ["FIXED_EPOCH_DETERMINISTIC_RECEIPTS"]), (out / "runtime_amendment/processes").resolve())
            self.assertEqual(os.environ["PYTHONPATH"].split(os.pathsep), [str(installed.parent.resolve()), str((payload / "scripts").resolve())])
            with self.assertRaises(FileExistsError):
                namespace["install_deterministic_runtime"](payload, out)

    def test_missing_or_duplicate_insertion_points_fail_closed(self):
        for source in [self.original().replace('        install_minimal(out, result)\n', ''),
                       self.original() + '\ndef bootstrap() -> None:\n    pass\n',
                       self.original().replace('seconds=7140', 'seconds=9999')]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                retry.render_entry(source, '# hook\n', '{}\n')

    def test_authorization_rejects_changed_budget_or_science(self):
        value = {"id": retry.ID, "cloud_status": "failed", "real_fits_completed": 0, "retry_kernel": retry.KERNEL,
            "retry_namespace": retry.FOLDER, "dataset_id": retry.DATASET, "dataset_version": 1,
            "provider_timeout_seconds": 6900, "original_provider_cap_seconds": 7200,
            "failed_attempt_reserve_seconds": 300, "fit_budget_seconds": 6300,
            "maximum_concurrent_gpu_jobs": 1, "private_required": True, "paid_compute": False,
            "new_dataset_upload": False, "same_scientific_payload_required": True,
            "registry_sha256": retry.REGISTRY_SHA, "bundle_manifest_sha256": retry.BUNDLE_SHA,
            "failure_archive_sha256": retry.ARCHIVE_SHA, "required_execution_policy": copy.deepcopy(retry.POLICY)}
        retry.validate_authorization(value)
        for key, wrong in [('provider_timeout_seconds', 7200), ('new_dataset_upload', True),
                           ('same_scientific_payload_required', False), ('real_fits_completed', 1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                retry.validate_authorization({**value, key: wrong})
        value['required_execution_policy']['torch_deterministic_warn_only'] = True
        with self.assertRaises(ValueError):
            retry.validate_authorization(value)


if __name__ == '__main__':
    unittest.main()
