"""Synthetic release boundary checks; no real fits, native models or network."""
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import continuation_six_release_v1 as release
from continuation_six_policy_v1 import policy_for_stage, policy_for


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def fake_torch(initialized=False):
    state = {"enabled": False, "warn": True}
    def configure(enabled, *, warn_only): state.update(enabled=enabled, warn=warn_only)
    return NS(__version__="generated", version=NS(cuda="fixture"), cuda=NS(is_initialized=lambda: initialized),
              use_deterministic_algorithms=configure, are_deterministic_algorithms_enabled=lambda: state["enabled"],
              is_deterministic_algorithms_warn_only_enabled=lambda: state["warn"],
              backends=NS(cudnn=NS(benchmark=True, deterministic=False), cuda=NS(matmul=NS(allow_tf32=True))),
              jit=NS(optimized_execution=lambda _: nullcontext()))


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.stage = "fixed_epoch_continuation_01"; self.cid = self.stage + "_release"
        plan = {"id": "continuation_six_v1", "stages": [{"id": f"fixed_epoch_continuation_{i:02d}", "index": i,
                   "policy": policy_for_stage(f"fixed_epoch_continuation_{i:02d}")} for i in range(1, 7)],
                "incumbent_selection_sha256": "", "incumbent_selection_path": "artifacts/third_pass/blend/frozen.json"}
        self.original = self.root / plan["incumbent_selection_path"]
        dump(self.original, {"oof_auc": .7, "fold_auc": [.7, .7, .7]})
        plan["incumbent_selection_sha256"] = release.sha(self.original)
        self.plan = self.root / "configs/continuation_six_v1.json"; dump(self.plan, plan)
        self.authority = self.root / "state/continuation_six_v1/authorization.json"
        now = datetime.now(timezone.utc)
        dump(self.authority, {"status": "authorized", "plan_sha256": release.sha(self.plan),
              "requested_utc": (now-timedelta(seconds=10)).isoformat(), "expires_utc": (now+timedelta(hours=23)).isoformat()})
        self.experiment = self.root / f"cloud/continuation_six_v1/{self.stage}/experiment"
        self.workspace = self.experiment / "assessment_workspace"
        self.evaluation_path = self.workspace / f"artifacts/{self.stage}/evaluation.json"
        metric = {"pooled_auc": .701, "fold_auc": [.701, .701, .701]}
        self.evaluation = {"id": self.stage, "row_count": 629671, "mixture_alpha": .1, "weight_search_performed": False,
            "advancement": {"advance_to_confirmation": True}, "metrics": {"mixture_C": metric,
              **{name: {"pooled_auc": .7, "fold_auc": [.7, .7, .7]} for name in ["incumbent", "mixture_A", "mixture_B"]}}}
        dump(self.evaluation_path, self.evaluation)
        self.protocol = self.root / f"artifacts/{self.stage}/local_evaluation_protocol.json"; dump(self.protocol, {"fixture": True})
        self.selection_path = self.root / f"artifacts/{self.stage}/release_selection.json"
        self.selection = {"status": "frozen_for_release", "stage": self.stage, "gain_gate_passed": True,
            "fixed_alpha": .1, "audit_evaluated": False, "weight_search_performed": False,
            "plan_sha256": release.sha(self.plan), "authority_sha256": release.sha(self.authority),
            "original_incumbent_submission_sha256": release.INCUMBENT_SUBMISSION, "policy": policy_for_stage(self.stage),
            "oof_auc": .701, "fold_auc": [.701, .701, .701]}
        for stem, path in [("evaluation", self.evaluation_path), ("protocol", self.protocol),
                           ("original_incumbent_selection", self.original), ("best_prior_selection", self.original)]:
            self.selection[stem+"_path"] = path.relative_to(self.root).as_posix()
            self.selection[stem+"_sha256"] = release.sha(path)
        dump(self.selection_path, self.selection)

    def test_valid_frozen_selection_gate(self):
        _, _, path, selection = release.validate_selection(self.root, self.stage)
        self.assertEqual(path, self.selection_path); self.assertTrue(selection["gain_gate_passed"])

    def test_changed_selection_evidence_rejected(self):
        dump(self.protocol, {"changed": True})
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            release.validate_selection(self.root, self.stage)

    def test_unfrozen_or_alpha_search_rejected(self):
        for key, value in [("status", "draft"), ("fixed_alpha", .2), ("weight_search_performed", True), ("audit_evaluated", True)]:
            value_dict = dict(self.selection); value_dict[key] = value; dump(self.selection_path, value_dict)
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.validate_selection(self.root, self.stage)

    def test_claimed_gate_does_not_override_actual_comparison(self):
        self.evaluation["metrics"]["mixture_B"]["pooled_auc"] = .702
        dump(self.evaluation_path, self.evaluation)
        self.selection["evaluation_sha256"] = release.sha(self.evaluation_path); dump(self.selection_path, self.selection)
        with self.assertRaisesRegex(ValueError, "comparison failed"):
            release.validate_selection(self.root, self.stage)

    def test_fold_regression_and_nonfinite_rejected(self):
        self.assertFalse(release.gain({"oof_auc": .701, "fold_auc": [.702, .702, .699]}, {"oof_auc": .7, "fold_auc": [.7]*3}))
        with self.assertRaises(ValueError): release.gain({"oof_auc": float('nan'), "fold_auc": [.7]*3}, {"oof_auc": .6, "fold_auc": [.6]*3})

    def execution_fixture(self):
        registry = {"id": self.cid, "selection_sha256": release.sha(self.selection_path)}
        dump(self.root / "registry.json", registry)
        policy = {"campaign": self.cid, "source_sha256": release.sha(Path(release.__file__)),
            "selection_sha256": registry["selection_sha256"], "required_roles": ["release_controller", "release_worker_A", "release_worker_C"],
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True, "warn_only": False,
            "cudnn_benchmark": False, "cudnn_deterministic": True, "cuda_matmul_allow_tf32": False}
        dump(self.root / "provenance/execution_policy.json", policy)
        return registry, policy

    def test_strict_policy_before_torch_loader_and_receipt(self):
        registry, _ = self.execution_fixture()
        def load():
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            return fake_torch()
        with patch.dict(os.environ, {}, clear=True):
            receipt = release.strict_start(self.root, registry, "release_controller", torch_loader=load)
        self.assertFalse(receipt["flags"]["startup_cuda_initialized"])
        self.assertTrue(receipt["flags"]["deterministic_algorithms"])
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(FileExistsError):
            release.strict_start(self.root, registry, "release_controller", torch_loader=load)

    def test_late_cuda_or_unknown_role_rejected(self):
        registry, _ = self.execution_fixture()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            release.strict_start(self.root, registry, "release_worker_A", torch_loader=lambda: fake_torch(True))
        with self.assertRaises(ValueError):
            release.strict_start(self.root, registry, "experiment_worker", torch_loader=fake_torch)

    def test_conflicting_workspace_and_policy_rejected(self):
        registry, policy = self.execution_fixture()
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}), self.assertRaises(ValueError):
            release.strict_start(self.root, registry, "release_controller", torch_loader=fake_torch)
        policy["warn_only"] = True; dump(self.root / "provenance/execution_policy.json", policy)
        with self.assertRaises(ValueError): release.strict_start(self.root, registry, "release_controller", torch_loader=fake_torch)

    def test_no_ml_import_or_local_worker_execution(self):
        script = Path(release.__file__)
        result = subprocess.run([sys.executable, "-c", "import sys;import continuation_six_release_v1;assert 'torch' not in sys.modules and 'pandas' not in sys.modules"],
                                cwd=script.parent, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, str(script), "worker", "--workspace", str(self.root), "--arm", "A"],
                                capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0); self.assertIn("No local full-data fitting", result.stderr)

    def test_native_chunking_probability_contract(self):
        import numpy as np
        import pandas as pd
        frame = pd.DataFrame({"x": [.1, .2, .3, .4, .5]})
        model = NS(classes_=np.array([0, 1]), predict_proba=lambda x: np.column_stack([1-x.x.to_numpy(), x.x.to_numpy()]))
        transform = NS(transform=lambda x: x)
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            one = release.probabilities(model, transform, frame, 1)
            partial = release.probabilities(model, transform, frame, 3)
            np.testing.assert_array_equal(one, partial)
            model.classes_ = np.array([1, 0])
            with self.assertRaises(ValueError): release.probabilities(model, transform, frame, 3)

    def test_probability_invalid_values_rejected(self):
        import numpy as np
        import pandas as pd
        frame = pd.DataFrame({"x": [1.]}); transform = NS(transform=lambda x: x)
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            for value in [[[-.1, 1.1]], [[.2, .2]], [[float('nan'), .5]], [[.5]]]:
                model = NS(classes_=np.array([0, 1]), predict_proba=lambda x, v=value: np.asarray(v))
                with self.subTest(value=value), self.assertRaises(ValueError): release.probabilities(model, transform, frame, 2)

    def test_runtime_source_or_input_drift_rejected(self):
        receipt = self.preparation_fixture()
        release.prepare(self.root, self.stage, receipt)
        payload = self.experiment.parent / "release/bundle/payload"
        release.verify_workspace(payload)
        source = payload / "scripts/common.py"
        source.write_text("changed\n")
        with self.assertRaises(ValueError): release.verify_workspace(payload)

    def test_source_copy_refuses_changed_or_existing_destination(self):
        source = self.root / "source"; source.write_bytes(b"fixture")
        destination = self.root / "destination"; release.copy_bound(source, destination, release.sha(source))
        with self.assertRaises(FileExistsError): release.copy_bound(source, destination, release.sha(source))
        with self.assertRaises(ValueError): release.copy_bound(source, self.root / "new", "0"*64)

    def test_rendered_cloud_entry_compiles_with_exact_LF_identity(self):
        source = Path(release.__file__).read_text()
        rendered = release.render_entry(source, self.stage, "a"*64)
        compile(rendered, "run.py", "exec")
        self.assertIn("EXPECTED_MANIFEST = '" + "a"*64 + "'", rendered)
        self.assertIn("STAGE = '" + self.stage + "'", rendered)
        self.assertNotIn("\r", rendered)
        with self.assertRaises(ValueError): release.render_entry(source + '\nSTAGE = "duplicate"\n', self.stage, "a"*64)

    def preparation_fixture(self):
        source_path = Path(release.__file__)
        for name, source in [("continuation_six_release_v1.py", source_path),
                             ("test_continuation_six_release_v1.py", Path(__file__))]:
            destination = self.root / "scripts" / name; destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(source.read_bytes())
        receipt = self.root / "generated_tests.json"
        dump(receipt, {"status": "passed", "source_sha256": {f"scripts/{name}": release.sha(self.root / "scripts" / name)
                for name in ["continuation_six_release_v1.py", "test_continuation_six_release_v1.py"]}})
        for name in ("train.parquet", "test.parquet", "original_aux_predictions.parquet", "test.csv", "sample_submission.csv"):
            path = self.root / "data" / name; path.parent.mkdir(exist_ok=True); path.write_bytes(b"generated input bytes")
        dump(self.root / "data/manifest.json", {"rows": {"train": release.TRAIN_ROWS, "test": release.TEST_ROWS},
             "raw_hashes": {name: release.sha(self.root / "data" / name) for name in ["test.csv", "sample_submission.csv"]}})
        binding = self.root / "state/continuation_six_v1/full_refit_input_binding.json"
        dump(binding, {"status": "frozen_before_new_assessment", "files": {name: release.sha(self.root / name)
                                                                                      for name in release.FULL_INPUT_FILES}})
        patcher = patch.object(release, "FULL_INPUT_BINDING_SHA256", release.sha(binding))
        patcher.start(); self.addCleanup(patcher.stop)
        runtime = self.workspace / "scripts/common.py"; runtime.parent.mkdir(); runtime.write_bytes(b"# synthetic source\n")
        registry = {"id": self.stage, "recipe": {"params": {}, "seed": 1}, "execution_image": "fixed-fixture",
            "source_hashes": {"scripts/common.py": release.sha(runtime)}, "parent_hashes": {},
            "trajectory_policies": {"A": policy_for("control"), "C": policy_for_stage(self.stage)}}
        dump(self.workspace / "registry.json", registry)
        smoke = self.workspace / f"artifacts/{self.stage}/smoke_cuda/verification.json"
        dump(smoke, {"status": "passed", "device": "cuda", "n_ens": 8, "architecture": [512, 256, 128], "prefix_guard_before_epoch5": True})
        runtime_sources = {}; installed = []; source_map = {}
        for module in release.RUNTIME_MODULES:
            name = f"artifacts/{self.stage}/runtime_source/{module.replace('.', '/')}.py"
            source = self.workspace / name; source.parent.mkdir(parents=True, exist_ok=True); source.write_bytes(b"# generated runtime source\n")
            digest = release.sha(source); runtime_sources[module] = digest; source_map[name] = digest
            installed.append({"module": module, "recorded_source_path": name, "sha256": digest,
                              "distribution": "pytabkit", "version": "generated"})
        environment = self.workspace / f"artifacts/{self.stage}/environment.json"
        dump(environment, {"versions": {name: "generated" for name in release.RUNTIME_PACKAGES}})
        trajectory = self.workspace / f"artifacts/{self.stage}/fold_0/inner/A/trajectory.json"
        dump(trajectory, {"status": "complete", "installed_source_provenance": installed, "source_sha256": source_map,
                          "versions": {name: "generated" for name in ["pytabkit", "torch", "pytorch-lightning", "numpy"]}})
        dump(self.protocol, {"runtime_source_sha256": runtime_sources})
        self.selection["protocol_sha256"] = release.sha(self.protocol); dump(self.selection_path, self.selection)
        evidence = [smoke, environment, trajectory, *[self.workspace / name for name in source_map]]
        dump(self.workspace / "output-manifest.json", {"files": {p.relative_to(self.workspace).as_posix(): {"sha256": release.sha(p)} for p in evidence}})
        dump(self.experiment / "preparation_manifest.json", {"registry_sha256": release.sha(self.workspace / "registry.json")})
        for name in ["licenses/pytabkit_LICENSE.txt", "ATTRIBUTION.md", "provenance/original_aux_manifest.json"]:
            target = self.workspace / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b"fixture provenance")
        return receipt

    def test_generated_preparation_is_pinned_and_never_contains_OOF(self):
        receipt = self.preparation_fixture()
        prepared = release.prepare(self.root, self.stage, receipt)
        folder = self.experiment.parent / "release"; payload = folder / "bundle/payload"
        compile((folder / "run.py").read_bytes(), "run.py", "exec")
        self.assertEqual(prepared["phase"], "release")
        self.assertEqual(prepared["hard_timeout_seconds"], 2400)
        for name, digest in prepared["frozen_files"].items(): self.assertEqual(release.sha(self.root / name), digest)
        names = release.read(payload / "bundle-manifest.json")["files"]
        self.assertNotIn("data/splits.parquet", names)
        self.assertFalse(any("oof" in name.lower() for name in names))
        self.assertEqual(release.read(payload / "registry.json")["expected_fit_count"], 2)
        release.verify_workspace(payload)
        self.assertEqual(release.sha(payload / "provenance/full_refit_input_binding.json"), release.FULL_INPUT_BINDING_SHA256)
        with self.assertRaises(FileExistsError): release.prepare(self.root, self.stage, receipt)

    def test_preassessment_binding_and_test_parquet_drift_rejected_before_packaging(self):
        receipt = self.preparation_fixture()
        (self.root / "data/test.parquet").write_bytes(b"changed after freeze")
        with self.assertRaisesRegex(ValueError, "bytes changed after"):
            release.prepare(self.root, self.stage, receipt)
        self.assertFalse((self.experiment.parent / "release").exists())
        binding = self.root / "state/continuation_six_v1/full_refit_input_binding.json"
        value = release.read(binding); value["files"]["data/test.parquet"] = release.sha(self.root / "data/test.parquet"); dump(binding, value)
        with self.assertRaisesRegex(ValueError, "binding changed"):
            release.prepare(self.root, self.stage, receipt)

    def test_registry_wrapper_and_ordered_policy_drift_rejected(self):
        receipt = self.preparation_fixture(); release.prepare(self.root, self.stage, receipt)
        payload = self.experiment.parent / "release/bundle/payload"; path = payload / "registry.json"
        registry = release.read(path); registry["trajectory_policies"]["C"] = policy_for("ema"); dump(path, registry)
        with self.assertRaisesRegex(ValueError, "wrapper registry"):
            release.verify_workspace(payload)
        dump(payload / f"configs/{self.cid}.json", {"id": self.cid, "registry_path": "registry.json", "registry_sha256": release.sha(path)})
        with self.assertRaisesRegex(ValueError, "recipe/input/runtime"):
            release.verify_workspace(payload)

    def test_installed_runtime_sources_and_versions_must_match_assessed_fit(self):
        receipt = self.preparation_fixture(); release.prepare(self.root, self.stage, receipt)
        payload = self.experiment.parent / "release/bundle/payload"; registry = release.verify_workspace(payload)
        def source(name): return self.workspace / f"artifacts/{self.stage}/runtime_source/{name.replace('.', '/')}.py"
        result = release.verify_installed_runtime(payload, registry, "release_controller", source_resolver=source,
                                                  version_resolver=lambda _: "generated")
        self.assertEqual(set(result["installed_sources"]), set(release.RUNTIME_MODULES)); self.assertTrue(result["before_full_fit"])
        with self.assertRaisesRegex(ValueError, "versions differ"):
            release.verify_installed_runtime(payload, registry, "release_worker_A", source_resolver=source,
                                            version_resolver=lambda _: "drifted")
        source(release.RUNTIME_MODULES[0]).write_bytes(b"changed installed optimizer source")
        with self.assertRaisesRegex(ValueError, "Actual installed release source"):
            release.verify_installed_runtime(payload, registry, "release_worker_A", source_resolver=source,
                                            version_resolver=lambda _: "generated")

    def test_assessed_runtime_evidence_drift_rejected_before_packaging(self):
        receipt = self.preparation_fixture()
        source = self.workspace / f"artifacts/{self.stage}/runtime_source/pytabkit/models/optim/optimizers.py"
        source.write_bytes(b"source changed")
        with self.assertRaisesRegex(ValueError, "Assessed installed runtime source"):
            release.prepare(self.root, self.stage, receipt)
        self.assertFalse((self.experiment.parent / "release").exists())

    def test_return_archive_is_explicit_and_hash_bound(self):
        import zipfile
        out = self.root / "artifacts" / self.cid
        dump(out / "completion_receipt.json", {"status": "completed"})
        dump(self.root / "registry.json", {"id": self.cid})
        dump(self.root / f"configs/{self.cid}.json", {"id": self.cid})
        unrelated = self.root / "data/private_input"; unrelated.parent.mkdir(); unrelated.write_bytes(b"never returned")
        destination = self.root / "return"; destination.mkdir()
        result = release.save_return(self.root, destination, self.cid)
        self.assertEqual(result["archive_sha256"], release.sha(destination / "results.zip"))
        with zipfile.ZipFile(destination / "results.zip") as archive:
            self.assertNotIn("data/private_input", archive.namelist())
            self.assertIn("output-manifest.json", archive.namelist())


if __name__ == "__main__":
    unittest.main()
