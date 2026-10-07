"""Generated startup and inherited data-boundary tests. Never fit real data."""
from __future__ import annotations

import copy
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import continuation_six_deterministic_v1 as hook
import continuation_six_policy_v1 as policies
import prepare_continuation_six_v1 as prep
import run_continuation_six_v1 as runner

ROOT = Path(__file__).resolve().parents[1]


def inherited_tests():
    source = (ROOT / "scripts/test_run_fixed_epoch_dropout_v1.py").read_text()
    source = source.replace("run_fixed_epoch_dropout_v1", "run_continuation_six_v1")
    source = source.replace("from prepare_fixed_epoch_dropout_v1 import SCIENCE",
        "SCIENCE = {**json.loads((Path(__file__).resolve().parents[1] / 'configs/continuation_six_v1.json').read_text())['science'], 'treatment_dropout_base': .05, 'trajectory_policies': {'A': policies.policy_for('control'), 'C': policies.policy_for_stage(runner.ID)}}")
    namespace = {"__file__": str(ROOT / "scripts/test_continuation_six_infrastructure_v1.py"),
                 "__name__": "generated_boundary_suite", "policies": policies}
    exec(compile(source, "inherited_generated_boundary_tests", "exec"), namespace)
    return unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(namespace[name])
        for name in ("RunnerBoundaryTests", "DropoutGuardTests"))


def startup_tests():
    source = (ROOT / "scripts/test_fixed_epoch_deterministic_execution_v1.py").read_text()
    source = source[:source.index("    def test_actual_torch_lightning_child")]
    source = source.replace("fixed_epoch_deterministic_execution_v1", "continuation_six_deterministic_v1")
    source = source.replace("run_fixed_epoch_dropout_v1.py", "run_continuation_six_v1.py")
    source = source.replace("smoke_fixed_epoch_dropout_v1.py", "smoke_continuation_six_v1.py")
    source = source.replace("fixed_epoch_dropout_v1", "fixed_epoch_continuation_01")
    source = source.replace('env = {"FIXED_EPOCH_DETERMINISTIC_ROOT"',
                            'env = {"CONTINUATION_STAGE_ID": "fixed_epoch_continuation_01", "FIXED_EPOCH_DETERMINISTIC_ROOT"')
    source = source.replace("policy._launch(args, root, target.name)",
                            'policy._launch(args, root, target.name, "fixed_epoch_continuation_01")')
    source = source.replace("policy._launch(smoke, root, smoke[0])",
                            'policy._launch(smoke, root, smoke[0], "fixed_epoch_continuation_01")')
    source = source.replace("policy._launch(bad, root, Path(bad[0]).name)",
                            'policy._launch(bad, root, Path(bad[0]).name, "fixed_epoch_continuation_01")')
    namespace = {"__file__": str(Path(__file__)), "__name__": "generated_startup_suite"}
    exec(compile(source, "inherited_generated_startup_tests", "exec"), namespace)
    return unittest.defaultTestLoader.loadTestsFromTestCase(namespace["DeterministicExecutionTests"])


class RegistrationTests(unittest.TestCase):
    def test_six_exact_policy_mappings_and_no_extra_knobs(self):
        for index, name in enumerate(policies.POLICY_IDS, 1):
            value = policies.policy_for_stage(f"fixed_epoch_continuation_{index:02d}")
            self.assertEqual(value, policies.policy_for(name))
            with self.assertRaises(ValueError):
                policies.validate_policy({**value, "extra": 1})
        for bad in ("fixed_epoch_continuation_00", "fixed_epoch_continuation_07", "../01"):
            with self.assertRaises(ValueError):
                policies.policy_for_stage(bad)

    def test_wrong_canonical_policy_for_stage_rejected(self):
        plan = json.loads((ROOT / "configs/continuation_six_v1.json").read_text())
        for index in range(1, 7):
            stage = f"fixed_epoch_continuation_{index:02d}"
            science = {**plan["science"], "id": stage, "treatment_dropout_base": .05,
                       "trajectory_policies": {"A": policies.policy_for("control"), "C": policies.policy_for_stage(stage)}}
            with patch.object(runner, "ID", stage):
                runner.validate_protocol(science)
                wrong = copy.deepcopy(science)
                wrong["trajectory_policies"]["C"] = policies.policy_for(policies.POLICY_IDS[index % 6])
                with self.assertRaises(ValueError):
                    runner.validate_protocol(wrong)

    def test_test_receipt_requires_current_source_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, receipt = root / "source.py", root / "proof.json"
            source.write_text("# generated\n")
            prep.write(receipt, {"status": "passed", "source_hashes": {"source.py": prep.sha(source)}})
            with patch.object(prep, "ROOT", root):
                prep.validate_tests(receipt)
                source.write_text("# changed\n")
                with self.assertRaises(ValueError):
                    prep.validate_tests(receipt)

    def test_copy_checks_bytes_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target = root / "source", root / "target"
            source.write_bytes(b"generated")
            prep.copy_bound(source, target, prep.sha(source))
            self.assertEqual(source.read_bytes(), target.read_bytes())
            with self.assertRaises(FileExistsError):
                prep.copy_bound(source, target, prep.sha(source))
            with self.assertRaises(ValueError):
                prep.copy_bound(source, root / "bad", "a" * 64)

    def test_runtime_and_smoke_compile_without_execution(self):
        for name in prep.SOURCES:
            source = (ROOT / "scripts" / name).read_text()
            compile(source, name, "exec")
        source = (ROOT / "scripts/run_continuation_six_v1.py").read_text()
        self.assertEqual(source.count('"MANIFEST_HASH_PLACEHOLDER"'), 1)
        self.assertIn("continuation_policy=registry", source)
        preparer = (ROOT / "scripts/prepare_continuation_six_v1.py").read_text()
        self.assertIn('newline="\\n"', preparer)

    def test_cloud_bootstrap_import_path_precedes_policy_validation(self):
        source = (ROOT / "scripts/run_continuation_six_v1.py").read_text()
        tree = ast.parse(source)
        bootstrap = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "bootstrap")
        calls = [node for node in ast.walk(bootstrap) if isinstance(node, ast.Call)]
        path_line = next(node.lineno for node in calls if ast.unparse(node.func) == "sys.path.insert")
        verify_line = next(node.lineno for node in calls if ast.unparse(node.func) == "validate_protocol")
        manifest_lines = [node.lineno for node in calls if ast.unparse(node.func) == "verify_bundle"]
        self.assertLess(max(manifest_lines), path_line)
        self.assertLess(path_line, verify_line)


def load_tests(loader, tests, pattern):
    tests.addTests(inherited_tests())
    tests.addTests(startup_tests())
    return tests


if __name__ == "__main__":
    unittest.main()
