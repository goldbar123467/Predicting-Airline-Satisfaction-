"""CPU-only generated startup, inheritance, failure and prefix-isolation checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import fixed_epoch_deterministic_execution_v1 as policy

ROOT = Path(__file__).resolve().parents[1]


class FakeTorch:
    __version__ = "generated-test"
    version = SimpleNamespace(cuda=None)

    def __init__(self, initialized=False):
        self.cuda = SimpleNamespace(is_initialized=lambda: initialized)
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, deterministic=False),
                                        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)))
        self.enabled, self.warn = False, True

    def use_deterministic_algorithms(self, enabled, *, warn_only):
        self.enabled, self.warn = enabled, warn_only

    def are_deterministic_algorithms_enabled(self):
        return self.enabled

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self.warn


def fixture(root):
    target = root / "scripts/run_fixed_epoch_dropout_v1.py"
    target.parent.mkdir(parents=True)
    target.write_text("# generated target\n", encoding="utf-8")
    receipts = root / "artifacts/fixed_epoch_dropout_v1/runtime_amendment/processes"
    env = {"FIXED_EPOCH_DETERMINISTIC_ROOT": str(root),
           "FIXED_EPOCH_DETERMINISTIC_RECEIPTS": str(receipts),
           "FIXED_EPOCH_DETERMINISTIC_AMENDMENT_SHA256": "a" * 64,
           "FIXED_EPOCH_DETERMINISTIC_HOOK_SHA256": policy._digest(Path(policy.__file__))}
    return target, receipts, env


def args_for(target):
    return [str(target), "--execute", "--campaign", str(target.parents[1] / "configs/fixed_epoch_dropout_v1.json")]


class DeterministicExecutionTests(unittest.TestCase):
    def test_exact_worker_matrix_and_restricted_smoke_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, _, _ = fixture(root)
            identities = set()
            for fold in range(3):
                for phase in ("inner", "outer"):
                    for trajectory in ("A", "C"):
                        args = [str(target), "--campaign", str(root / "configs/fixed_epoch_dropout_v1.json"),
                                "--worker", "--fold", str(fold), "--phase", phase, "--trajectory", trajectory]
                        value = policy._launch(args, root, target.name)
                        identities.add((value["role"], value["fold"], value["phase"], value["trajectory"]))
            self.assertEqual(len(identities), 12)
            smoke = ["smoke_fixed_epoch_dropout_v1.py", "--device", "cuda", "--output",
                     str(root / "artifacts/fixed_epoch_dropout_v1/smoke_cuda")]
            self.assertEqual(policy._launch(smoke, root, smoke[0])["role"], "smoke")
            for bad in (args_for(target) + ["--unknown", "PRIVATE"], args_for(target) + ["--worker"], smoke[:-1] + [str(root / "wrong")]):
                with self.assertRaises(ValueError) as caught:
                    policy._launch(bad, root, Path(bad[0]).name)
                self.assertNotIn("PRIVATE", str(caught.exception))

    def test_environment_precedes_import_and_flags_receipt_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            target, receipts, env = fixture(Path(temporary))
            torch = FakeTorch()
            def loader():
                self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                return torch
            result = policy.configure(argv=args_for(target), environ=env, torch_loader=loader)
            self.assertEqual(result["flags"], {"deterministic_algorithms": True, "deterministic_warn_only": False,
                "cudnn_benchmark": False, "cudnn_deterministic": True, "cuda_matmul_allow_tf32": False,
                "cublas_workspace_config": ":4096:8", "startup_cuda_initialized": False})
            self.assertEqual(result, json.loads(next(receipts.glob("process_*.json")).read_text()))
            self.assertEqual(result["target_sha256"], policy._digest(target))
            with self.assertRaises(FileExistsError):
                policy.configure(argv=args_for(target), environ=env, torch_loader=loader)

    def test_late_cuda_setup_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target, receipts, env = fixture(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "after CUDA"):
                policy.configure(argv=args_for(target), environ=env, torch_loader=lambda: FakeTorch(True))
            self.assertFalse(receipts.exists())

    def test_conflicting_workspace_or_hook_hash_fails_before_torch_import(self):
        for change in ({"CUBLAS_WORKSPACE_CONFIG": ":16:8"}, {"FIXED_EPOCH_DETERMINISTIC_HOOK_SHA256": "b" * 64}):
            with tempfile.TemporaryDirectory() as temporary:
                target, _, env = fixture(Path(temporary))
                with self.assertRaises(ValueError):
                    policy.configure(argv=args_for(target), environ={**env, **change},
                                     torch_loader=lambda: self.fail("must reject before torch import"))

    def test_unrelated_program_has_no_import_or_environment_side_effect(self):
        env = {}
        result = policy.configure(argv=["unrelated.py"], environ=env,
                                  torch_loader=lambda: self.fail("unrelated process must stay untouched"))
        self.assertIsNone(result)
        self.assertEqual(env, {})

    def test_wrong_target_path_or_receipt_directory_rejected(self):
        for wrong in ("target", "receipts"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                target, _, env = fixture(root)
                if wrong == "target":
                    target = root / target.name
                    target.write_text("# unrelated same basename\n")
                else:
                    env["FIXED_EPOCH_DETERMINISTIC_RECEIPTS"] = str(root / "elsewhere")
                with self.assertRaises(ValueError):
                    policy.configure(argv=args_for(target), environ=env, torch_loader=lambda: FakeTorch())

    def test_sitecustomize_failure_terminates_before_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, _, env = fixture(root)
            target.write_text("raise AssertionError('target must never execute')\n")
            hookdir = root / "hooks"
            hookdir.mkdir()
            shutil.copyfile(policy.__file__, hookdir / "sitecustomize.py")
            env.update(PYTHONPATH=str(hookdir), FIXED_EPOCH_DETERMINISTIC_HOOK_SHA256="b" * 64,
                       CUDA_VISIBLE_DEVICES="")
            result = subprocess.run([sys.executable, *args_for(target)], env={**os.environ, **env},
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 86, result.stderr)
            self.assertIn("deterministic startup failed", result.stderr)
            self.assertNotIn("target must never execute", result.stderr)

    def test_actual_torch_lightning_child_inheritance_and_all_six_prefix_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, receipts, env = fixture(root)
            hookdir = root / "hooks"
            hookdir.mkdir()
            shutil.copyfile(policy.__file__, hookdir / "sitecustomize.py")
            code = '''import json,os,subprocess,sys,unittest
from pathlib import Path
import torch
assert torch.are_deterministic_algorithms_enabled()
assert not torch.is_deterministic_algorithms_warn_only_enabled()
assert not torch.cuda.is_initialized()
if "--worker" in sys.argv:
    print("CHILD_POLICY_PASSED")
else:
    sys.path.insert(0, REAL_SCRIPTS)
    import pytorch_lightning as pl
    trainer=pl.Trainer(accelerator="cpu",devices=1,max_epochs=0,logger=False,enable_checkpointing=False,enable_model_summary=False)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(REAL_SCRIPTS,pattern="test_fixed_epoch_dropout_adapter_v1.py"))
    if not result.wasSuccessful():raise SystemExit(1)
    child=subprocess.run([sys.executable,__file__,"--worker","--campaign",str(Path(__file__).parents[1]/"configs/fixed_epoch_dropout_v1.json"),"--fold","0","--phase","inner","--trajectory","A"],check=True,capture_output=True,text=True,timeout=40)
    assert "CHILD_POLICY_PASSED" in child.stdout
    assert not torch.cuda.is_initialized()
    print("PARENT_POLICY_AND_SIX_PREFIX_TESTS_PASSED")
'''.replace("REAL_SCRIPTS", repr(str(ROOT / "scripts")))
            target.write_text(code, encoding="utf-8")
            env.update(PYTHONPATH=str(hookdir), CUDA_VISIBLE_DEVICES="")
            result = subprocess.run([sys.executable, *args_for(target)], env={**os.environ, **env},
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PARENT_POLICY_AND_SIX_PREFIX_TESTS_PASSED", result.stdout)
            actual = [json.loads(path.read_text()) for path in receipts.glob("process_*.json")]
            self.assertEqual(len(actual), 2)
            self.assertEqual({row["launch"]["role"] for row in actual}, {"controller", "worker"})
            self.assertTrue(all(row["flags"]["deterministic_algorithms"] for row in actual))
            self.assertTrue(all(not row["flags"]["startup_cuda_initialized"] for row in actual))
            self.assertTrue(all(row["hook_sha256"] == env["FIXED_EPOCH_DETERMINISTIC_HOOK_SHA256"] for row in actual))


if __name__ == "__main__":
    unittest.main()
