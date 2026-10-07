"""Conditional cloud-only paired full refit; never scores held-out quality or submits.

Prepare is local and requires a frozen qualifying selection. The cloud entry
starts one strict controller and two fresh strict workers. Only the selected
candidate's keyed test probabilities are returned; final blending is separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INCUMBENT_SUBMISSION = "177eca4563e41b91c30c12d494284ea24d68951b6ada332edf49dc10646ccbc9"
TRAIN_ROWS, TEST_ROWS = 699635, 299844
FULL_INPUT_BINDING_SHA256 = "c1e58af45d2874677dde1e0f0c0861a929c98c741512150e6b09f7e367af0b2c"
FULL_INPUT_FILES = ("data/train.parquet", "data/test.parquet", "data/original_aux_predictions.parquet",
                    "data/manifest.json", "data/test.csv", "data/sample_submission.csv")
RUNTIME_MODULES = ("pytabkit.models.optim.optimizers", "pytabkit.models.training.lightning_modules",
    "pytabkit.models.training.nn_creator", "pytabkit.models.training.scheduling", "pytabkit.models.training.coord",
    "pytabkit.models.training.lightning_callbacks", "pytabkit.models.data.data", "pytabkit.models.torch_utils",
    "pytabkit.models.nn_models.nn", "pytabkit.models.nn_models.base", "pytabkit.models.nn_models.categorical",
    "pytabkit.models.nn_models.activations")
RUNTIME_PACKAGES = ("torch", "numpy", "pandas", "scikit-learn", "pytabkit", "pytorch-lightning")
EXPECTED_MANIFEST = "RELEASE_MANIFEST_PLACEHOLDER"
STAGE = "RELEASE_STAGE_PLACEHOLDER"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def now():
    return datetime.now(timezone.utc)


def replace_status(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def rel(root, path):
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def copy_bound(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if sha(source) != expected:
        raise ValueError("Frozen input/source bytes changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1024**2)
    if sha(destination) != expected:
        raise ValueError("Copied input/source bytes changed")


def checked(root, name):
    if not isinstance(name, str) or "\\" in name or ":" in name or name.startswith("/") or any(p in {"", ".", ".."} for p in name.split("/")):
        raise ValueError("Nonportable artifact path")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Artifact path escapes root")
    return path


def gain(candidate, previous):
    values = [candidate["oof_auc"], previous["oof_auc"], *candidate["fold_auc"], *previous["fold_auc"]]
    if len(candidate["fold_auc"]) != 3 or len(previous["fold_auc"]) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError("Invalid fixed three-fold metrics")
    delta = [a-b for a, b in zip(candidate["fold_auc"], previous["fold_auc"], strict=True)]
    return candidate["oof_auc"] - previous["oof_auc"] >= 1e-5 and sum(delta)/3 >= 1e-5 and min(delta) >= -2e-5


def render_entry(source, stage, manifest_hash):
    lines = source.splitlines(keepends=True)
    for prefix, value in [("EXPECTED_MANIFEST = ", manifest_hash), ("STAGE = ", stage)]:
        positions = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        if len(positions) != 1: raise ValueError("Release entry identity assignment is not unique")
        lines[positions[0]] = prefix + repr(value) + "\n"
    rendered = "".join(lines)
    compile(rendered, "run.py", "exec")
    return rendered


def validate_selection(root, stage):
    from continuation_six_policy_v1 import policy_for_stage
    from continuation_six_selection_v1 import best_prior, sequence_context
    plan, authority = sequence_context(root, stage)
    selection_path = root / f"artifacts/{stage}/release_selection.json"
    selection = read(selection_path)
    required = {"status": "frozen_for_release", "stage": stage, "gain_gate_passed": True, "fixed_alpha": .1,
                "audit_evaluated": False, "weight_search_performed": False,
                "plan_sha256": sha(root / "configs/continuation_six_v1.json"),
                "authority_sha256": sha(root / "state/continuation_six_v1/authorization.json"),
                "original_incumbent_submission_sha256": INCUMBENT_SUBMISSION, "policy": policy_for_stage(stage)}
    if any(selection.get(k) != v for k, v in required.items()):
        raise ValueError("Full-data release lacks a qualifying frozen selection")
    for stem in ("evaluation", "protocol", "original_incumbent_selection", "best_prior_selection"):
        if sha(checked(root, selection[stem + "_path"])) != selection[stem + "_sha256"]:
            raise ValueError("Release selection evidence changed")
    prior_path, prior = best_prior(root, stage)
    if sha(prior_path) != selection["best_prior_selection_sha256"]:
        raise ValueError("Best prior release changed before packaging")
    original = read(checked(root, selection["original_incumbent_selection_path"]))
    evaluation = read(checked(root, selection["evaluation_path"]))
    metric = evaluation["metrics"]["mixture_C"]
    if (evaluation.get("id") != stage or evaluation.get("row_count") != 629671
            or evaluation.get("mixture_alpha") != .1 or evaluation.get("weight_search_performed") is not False
            or evaluation["advancement"]["advance_to_confirmation"] is not True
            or metric["pooled_auc"] != selection["oof_auc"] or metric["fold_auc"] != selection["fold_auc"]
            or not gain(selection, original) or not gain(selection, prior)):
        raise ValueError("Release must preserve evaluated fixed mixture and incremental gain")
    # Recheck the actual registered three comparisons from already-computed
    # metrics. This is gate arithmetic, not a new prediction/quality assessment.
    for key in ("incumbent", "mixture_A", "mixture_B"):
        item = evaluation["metrics"][key]
        if not gain(selection, {"oof_auc": item["pooled_auc"], "fold_auc": item["fold_auc"]}):
            raise ValueError("Evaluated advancement comparison failed")
    return plan, authority, selection_path, selection


def full_input_binding(root):
    path = root / "state/continuation_six_v1/full_refit_input_binding.json"
    if sha(path) != FULL_INPUT_BINDING_SHA256:
        raise ValueError("Canonical preassessment full-input binding changed")
    binding = read(path)
    if binding.get("status") != "frozen_before_new_assessment" or set(binding["files"]) != set(FULL_INPUT_FILES):
        raise ValueError("Canonical full-input inventory differs")
    for name, digest in binding["files"].items():
        if sha(checked(root, name)) != digest:
            raise ValueError("Canonical full-data bytes changed after preassessment freeze")
    return path, binding


def experiment_runtime_binding(assessment, stage, returned_manifest, protocol):
    names = [f"artifacts/{stage}/environment.json", f"artifacts/{stage}/fold_0/inner/A/trajectory.json"]
    for name in names:
        if returned_manifest["files"][name]["sha256"] != sha(checked(assessment, name)):
            raise ValueError("Experiment runtime evidence return binding differs")
    environment, trajectory = (read(assessment / name) for name in names)
    versions = environment["versions"]
    expected_sources = protocol["runtime_source_sha256"]
    if set(versions) != set(RUNTIME_PACKAGES) or set(expected_sources) != set(RUNTIME_MODULES):
        raise ValueError("Experiment runtime source/version inventory differs")
    installed = trajectory["installed_source_provenance"]
    if (trajectory["status"] != "complete" or len(installed) != len(RUNTIME_MODULES)
            or {item["module"] for item in installed} != set(RUNTIME_MODULES)):
        raise ValueError("Incomplete experiment runtime evidence")
    for item in installed:
        recorded = f"artifacts/{stage}/runtime_source/{item['module'].replace('.', '/')}.py"
        digest = expected_sources[item["module"]]
        if (item["sha256"] != digest or item["recorded_source_path"] != recorded
                or item["distribution"] != "pytabkit" or item["version"] != versions["pytabkit"]
                or trajectory["source_sha256"][recorded] != digest
                or returned_manifest["files"][recorded]["sha256"] != digest or sha(assessment / recorded) != digest):
            raise ValueError("Assessed installed runtime source differs from frozen protocol")
    if any(versions[name] != value for name, value in trajectory["versions"].items()):
        raise ValueError("Experiment package versions disagree")
    return {"status": "frozen_from_completed_experiment", "versions": versions,
            "module_sha256": expected_sources, "experiment_evidence": {name: sha(assessment / name) for name in names}}


def prepare(root, stage, test_receipt):
    plan, authority, selection_path, selection = validate_selection(root, stage)
    input_binding_path, input_binding = full_input_binding(root)
    if (datetime.fromisoformat(authority["expires_utc"]) - now()).total_seconds() < 2400:
        raise ValueError("Insufficient authorized wall time for full refit")
    proof = read(test_receipt)
    source_key = "scripts/continuation_six_release_v1.py"
    tested = proof.get("source_sha256", {})
    if (proof.get("status") != "passed" or tested.get(source_key) != sha(root / source_key)
            or "scripts/test_continuation_six_release_v1.py" not in tested
            or any(sha(checked(root, p)) != value for p, value in tested.items())):
        raise ValueError("Release helper needs source-bound generated verification")
    experiment = root / f"cloud/continuation_six_v1/{stage}/experiment"
    assessment = experiment / "assessment_workspace"
    exp = read(experiment / "preparation_manifest.json")
    registry_path = assessment / "registry.json"
    if sha(registry_path) != exp["registry_sha256"]:
        raise ValueError("Verified experiment registry differs")
    registry = read(registry_path)
    smoke_path = assessment / f"artifacts/{stage}/smoke_cuda/verification.json"
    smoke = read(smoke_path)
    if (smoke.get("status") != "passed" or smoke.get("device") != "cuda" or smoke.get("n_ens") != 8
            or smoke.get("architecture") != [512, 256, 128] or smoke.get("prefix_guard_before_epoch5") is not True):
        raise ValueError("Missing matching full-architecture generated cloud smoke")
    # A successful once-only evaluator is the authority for the untouched
    # experiment return manifest, including this smoke's source bindings.
    returned_manifest = read(assessment / "output-manifest.json")
    if returned_manifest["files"][rel(assessment, smoke_path)]["sha256"] != sha(smoke_path):
        raise ValueError("Smoke receipt return binding differs")
    runtime_binding = experiment_runtime_binding(assessment, stage, returned_manifest,
                                                read(checked(root, selection["protocol_path"])))
    folder = root / f"cloud/continuation_six_v1/{stage}/release"
    if folder.exists():
        raise FileExistsError("Preserve prior release preparation")
    cid = stage + "_release"; payload = folder / "bundle/payload"
    payload.mkdir(parents=True)
    sources = {}
    for name, digest in registry["source_hashes"].items():
        # Exact adapter/runtime dependencies from the assessed stage; unrelated
        # experiment runner/smoke/hook are not executable release entry points.
        if Path(name).name in {"run_continuation_six_v1.py", "smoke_continuation_six_v1.py", "continuation_six_deterministic_v1.py"}:
            continue
        copy_bound(checked(assessment, name), checked(payload, name), digest); sources[name] = digest
    copy_bound(root / source_key, payload / source_key, sha(root / source_key)); sources[source_key] = sha(root / source_key)
    data_manifest = read(root / "data/manifest.json")
    if data_manifest["rows"]["train"] != TRAIN_ROWS or data_manifest["rows"]["test"] != TEST_ROWS:
        raise ValueError("Canonical full-data row contract differs")
    for name, digest in input_binding["files"].items():
        pinned = registry["parent_hashes"].get(name)
        if name in {"data/test.csv", "data/sample_submission.csv"}:
            pinned = data_manifest["raw_hashes"][Path(name).name]
        if pinned is not None and digest != pinned:
            raise ValueError("Full-data source differs from historical canonical provenance")
        copy_bound(root / name, payload / name, digest)
    evidence = {"release_selection.json": selection_path, "sequence_plan.json": root / "configs/continuation_six_v1.json",
                "sequence_authorization.json": root / "state/continuation_six_v1/authorization.json",
                "experiment_smoke.json": smoke_path, "release_tests.json": test_receipt,
                "full_refit_input_binding.json": input_binding_path,
                "experiment_registry.json": registry_path, "local_evaluation_protocol.json": checked(root, selection["protocol_path"])}
    for name, source in evidence.items():
        copy_bound(source, payload / "provenance" / name, sha(source))
    write(payload / "provenance/experiment_runtime_binding.json", runtime_binding)
    for name in ("licenses/pytabkit_LICENSE.txt", "ATTRIBUTION.md", "provenance/original_aux_manifest.json"):
        copy_bound(assessment / name, payload / name, sha(assessment / name))
    execution = {"id": cid + "_strict_execution", "campaign": cid, "stage": stage,
                 "method": "direct checked startup before torch import; no sitecustomize",
                 "source_sha256": sources[source_key], "required_roles": ["release_controller", "release_worker_A", "release_worker_C"],
                 "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True,
                 "warn_only": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
                 "cuda_matmul_allow_tf32": False, "selection_sha256": sha(selection_path)}
    write(payload / "provenance/execution_policy.json", execution)
    inputs = {rel(payload, p): sha(p) for p in payload.rglob("*") if p.is_file() and rel(payload, p) not in sources}
    release_registry = {"id": cid, "stage": stage, "phase": "release", "recipe": registry["recipe"],
                        "trajectory_policies": registry["trajectory_policies"], "selection_sha256": sha(selection_path),
                        "experiment_registry_sha256": sha(registry_path), "experiment_smoke_sha256": sha(smoke_path),
                        "full_refit_input_binding_sha256": sha(input_binding_path),
                        "experiment_runtime_binding_sha256": sha(payload / "provenance/experiment_runtime_binding.json"),
                        "installed_runtime_source_sha256": runtime_binding["module_sha256"],
                        "expected_versions": runtime_binding["versions"],
                        "execution_image": registry["execution_image"], "source_hashes": sources, "input_hashes": inputs,
                        "train_rows": TRAIN_ROWS, "test_rows": TEST_ROWS, "hard_timeout_seconds": 2400,
                        "fit_budget_seconds": 2100, "delivery_budget_seconds": 2340,
                        "expected_fit_count": 2, "expected_prefix_count": 1, "expected_startup_roles": 3,
                        "audit_quality_evaluated": False, "full_train_after_selection_freeze": True}
    write(payload / "registry.json", release_registry)
    write(payload / f"configs/{cid}.json", {"id": cid, "registry_path": "registry.json", "registry_sha256": sha(payload / "registry.json")})
    files = {rel(payload, p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in payload.rglob("*") if p.is_file()}
    write(payload / "bundle-manifest.json", {"id": cid, "registry_sha256": sha(payload / "registry.json"), "files": files})
    upload = folder / "bundle/upload"; upload.mkdir()
    with zipfile.ZipFile(upload / "payload.zip", "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in sorted(set(files) | {"bundle-manifest.json"}): archive.write(payload / name, name)
    index = int(stage[-2:]); dataset = f"clarkkitchen/s6e10-continuation-{index:02d}-release-bundle-20261004"
    kernel = f"clarkkitchen/s6e10-continuation-{index:02d}-release-20261004"
    write(upload / "dataset-metadata.json", {"id": dataset, "title": f"S6E10 Continuation {index:02d} Release Bundle 20261004",
          "licenses": [{"name": "other"}], "description": "Private frozen full-data refit only. Preserve attribution. Never publish."})
    runtime = render_entry((root / source_key).read_text(encoding="utf-8"), stage, sha(payload / "bundle-manifest.json"))
    (folder / "run.py").write_bytes(runtime.encode("utf-8"))
    write(folder / "kernel-metadata.json", {"id": kernel, "title": f"S6E10 Continuation {index:02d} Release 20261004",
          "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
          "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
          "docker_image": registry["execution_image"], "docker_image_pinning_type": "original",
          "dataset_sources": [dataset], "competition_sources": [], "kernel_sources": [], "model_sources": []})
    prepared = {"id": cid, "stage": stage, "phase": "release", "private_required": True,
                "dataset_id": dataset, "kernel_id": kernel, "hard_timeout_seconds": 2400,
                "fit_budget_seconds": 2100, "manifest_sha256": sha(payload / "bundle-manifest.json"),
                "registry_sha256": sha(payload / "registry.json"), "archive_sha256": sha(upload / "payload.zip"),
                "runtime_sha256": sha(folder / "run.py"), "selection_sha256": sha(selection_path),
                "frozen_files": {rel(root, p): sha(p) for p in folder.rglob("*") if p.is_file()}}
    write(folder / "preparation_manifest.json", prepared)
    write(root / f"state/continuation_six_v1/registrations/{stage}.release.json", {
        "status": "registered", "stage": stage, "phase": "release", "plan_sha256": sha(root / "configs/continuation_six_v1.json"),
        "preparation_path": rel(root, folder / "preparation_manifest.json"), "preparation_sha256": sha(folder / "preparation_manifest.json"),
        "release_selection_path": rel(root, selection_path), "release_selection_sha256": sha(selection_path)})
    return prepared


def verify_workspace(root):
    from continuation_six_policy_v1 import policy_for, policy_for_stage
    registry = read(root / "registry.json")
    expected_policies = {"A": policy_for("control"), "C": policy_for_stage(registry["stage"])}
    if registry["id"] != registry["stage"] + "_release": raise ValueError("Release campaign identity differs")
    wrapper = read(root / f"configs/{registry['id']}.json")
    if wrapper != {"id": registry["id"], "registry_path": "registry.json", "registry_sha256": sha(root / "registry.json")}:
        raise ValueError("Release wrapper registry binding differs")
    for name, digest in {**registry["source_hashes"], **registry["input_hashes"]}.items():
        if sha(checked(root, name)) != digest:
            raise ValueError("Release runtime input/source changed")
    if (registry["phase"] != "release" or registry["train_rows"] != TRAIN_ROWS or registry["test_rows"] != TEST_ROWS
            or registry["expected_fit_count"] != 2 or registry["hard_timeout_seconds"] != 2400):
        raise ValueError("Release runtime contract differs")
    experiment_path = root / "provenance/experiment_registry.json"
    experiment = read(experiment_path)
    binding_path = root / "provenance/full_refit_input_binding.json"
    runtime_path = root / "provenance/experiment_runtime_binding.json"
    binding, runtime = read(binding_path), read(runtime_path)
    if (sha(experiment_path) != registry["experiment_registry_sha256"]
            or experiment["id"] != registry["stage"] or registry["recipe"] != experiment["recipe"]
            or registry["trajectory_policies"] != expected_policies or experiment["trajectory_policies"] != expected_policies
            or sha(binding_path) != FULL_INPUT_BINDING_SHA256 or registry["full_refit_input_binding_sha256"] != FULL_INPUT_BINDING_SHA256
            or binding["status"] != "frozen_before_new_assessment" or set(binding["files"]) != set(FULL_INPUT_FILES)
            or any(registry["input_hashes"].get(name) != digest for name, digest in binding["files"].items())
            or sha(runtime_path) != registry["experiment_runtime_binding_sha256"]
            or runtime["module_sha256"] != registry["installed_runtime_source_sha256"]
            or runtime["versions"] != registry["expected_versions"]
            or set(runtime["module_sha256"]) != set(RUNTIME_MODULES) or set(runtime["versions"]) != set(RUNTIME_PACKAGES)):
        raise ValueError("Release canonical recipe/input/runtime binding differs")
    return registry


def verify_installed_runtime(root, registry, role, *, source_resolver=None, version_resolver=None):
    """Run only after strict startup; inspect the actual imported package files."""
    if source_resolver is None:
        def source_resolver(name):
            source = inspect.getsourcefile(importlib.import_module(name))
            if source is None: raise RuntimeError("Installed runtime source is not inspectable")
            return Path(source).resolve()
    if version_resolver is None: version_resolver = importlib.metadata.version
    versions = {name: version_resolver(name) for name in RUNTIME_PACKAGES}
    if versions != registry["expected_versions"]:
        raise ValueError("Installed release versions differ from completed experiment")
    sources = {}
    for name in RUNTIME_MODULES:
        path = Path(source_resolver(name))
        digest = sha(path)
        if path.suffix != ".py" or digest != registry["installed_runtime_source_sha256"][name]:
            raise ValueError("Actual installed release source differs from completed experiment")
        sources[name] = {"installed_source_path": str(path), "sha256": digest}
    receipt = {"status": "passed", "role": role, "versions": versions, "installed_sources": sources,
               "registry_sha256": sha(root / "registry.json"), "before_full_fit": True}
    write(root / f"artifacts/{registry['id']}/runtime_compatibility/{role}.json", receipt)
    return receipt


def strict_start(root, registry, role, *, torch_loader=None):
    policy_path = root / "provenance/execution_policy.json"; policy = read(policy_path)
    expected_policy = {"campaign": registry["id"], "selection_sha256": registry["selection_sha256"],
        "required_roles": ["release_controller", "release_worker_A", "release_worker_C"],
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True, "warn_only": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True, "cuda_matmul_allow_tf32": False}
    if (any(policy.get(k) != v for k, v in expected_policy.items())
            or role not in policy["required_roles"] or sha(Path(__file__)) != policy["source_sha256"]):
        raise ValueError("Release startup role/source is unregistered")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise ValueError("Conflicting deterministic workspace")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch_loader is None:
        import torch
    else:
        torch = torch_loader()
    if torch.cuda.is_initialized(): raise RuntimeError("Release startup happened after CUDA initialization")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    flags = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
             "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
             "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
             "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "startup_cuda_initialized": torch.cuda.is_initialized()}
    expected = {"deterministic_algorithms": True, "deterministic_warn_only": False, "cudnn_benchmark": False,
                "cudnn_deterministic": True, "cuda_matmul_allow_tf32": False, "cublas_workspace_config": ":4096:8", "startup_cuda_initialized": False}
    if flags != expected: raise RuntimeError("Strict release execution flags did not apply")
    receipt = {"status": "passed", "role": role, "pid": os.getpid(), "created_utc": now().isoformat(),
               "source_sha256": sha(Path(__file__)), "execution_policy_sha256": sha(policy_path), "flags": flags,
               "torch_version": str(torch.__version__), "cuda_build_version": torch.version.cuda,
               "registry_sha256": sha(root / "registry.json")}
    write(root / f"artifacts/{registry['id']}/runtime_amendment/processes/{role}.json", receipt)
    return receipt


def cloud_features(frame, recipe):
    from common import features
    from cloud_feature_compat import cloud_numeric_category_compat, SOURCE_VERSIONS
    return cloud_numeric_category_compat(features(frame, recipe), frame,
        run={**recipe, "execution_backend": "kaggle"}, source_versions=SOURCE_VERSIONS)


def probabilities(model, transform, frame, chunk):
    import numpy as np
    import torch
    if not np.array_equal(model.classes_, [0, 1]): raise ValueError("Native class order differs")
    output = []
    with torch.jit.optimized_execution(False):
        for start in range(0, len(frame), chunk):
            value = model.predict_proba(transform.transform(frame.iloc[start:start+chunk]))
            if (value.shape != (min(chunk, len(frame)-start), 2) or not np.isfinite(value).all()
                    or ((value < 0) | (value > 1)).any() or not np.allclose(value.sum(axis=1), 1, atol=2e-6, rtol=1e-5)):
                raise ValueError("Native probability shape/range/simplex failure")
            output.append(value[:, 1].astype(np.float64))
    return np.concatenate(output)


def worker(root, registry, arm):
    strict_start(root, registry, "release_worker_" + arm)
    verify_installed_runtime(root, registry, "release_worker_" + arm)
    import gc
    import numpy as np
    import pandas as pd
    import torch
    from categorical_transform import CategoricalTransform
    from continuation_six_adapter_v1 import fit_fixed_trajectory
    from realmlp_categorical import load_realmlp_categorical
    from common import CAT
    out = root / "artifacts" / registry["id"]; directory = out / ("trajectory_" + arm)
    directory.mkdir(exist_ok=False)
    remaining = (datetime.fromisoformat(read(out / "job_runtime.json")["fit_deadline_utc"]) - now()).total_seconds()
    if remaining <= 0: raise TimeoutError("Full refit deadline")
    deadline = time.monotonic() + remaining
    torch.set_num_threads(4)
    train = pd.read_parquet(root / "data/train.parquet")
    if len(train) != TRAIN_ROWS or not train.id.is_unique or not np.isin(train.satisfaction, [0, 1]).all():
        raise ValueError("Full train row/label contract differs")
    ids = train.id.to_numpy(); recipe = registry["recipe"]; target = train.satisfaction.to_numpy(dtype=np.int64)
    features = cloud_features(train, recipe)
    transform = CategoricalTransform("realmlp_cat").fit(features); transform.save(directory / "transform.json")
    encoded = transform.transform(features)
    context = {"campaign": registry["id"], "phase": "outer", "fit_scope": "full_refit_after_selection_freeze",
               "trajectory": arm, "training_rows": TRAIN_ROWS, "monitor_rows": 0,
               "training_ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
               "transform_sha256": sha(directory / "transform.json"), "selection_sha256": registry["selection_sha256"],
               "input_feature_columns": list(features.columns)}
    write(directory / "context.json", context)
    if arm == "C":
        prior = read(out / "trajectory_A/context.json")
        for key in context:
            if key != "trajectory" and prior[key] != context[key]: raise ValueError("Full refit A/C identity differs")
        write(directory / "pre_fit_match.json", {"status": "passed", "before_treatment_fit": True,
              "training_ids_sha256": context["training_ids_sha256"], "transform_sha256": context["transform_sha256"]})
    del train, features; gc.collect()
    params = {**recipe["params"], "seed": recipe["seed"], "device": "cuda", "categorical_indices": transform.categorical_indices}
    result = fit_fixed_trajectory(encoded, target, None, None, params, directory, horizon=16, endpoint_epochs=[4, 16],
        context=context, deadline_monotonic=deadline, continuation_policy=registry["trajectory_policies"][arm],
        prefix_reference_dir=None if arm == "A" else out / "trajectory_A")
    if result["status"] != "complete" or result["executed_epochs"] != 16: raise ValueError("Incomplete full refit")
    del encoded, target, ids; gc.collect(); torch.cuda.empty_cache()
    if arm == "C":
        prefix = read(directory / "prefix_match.json")
        if (prefix["status"] != "passed" or not prefix["gate_completed_before_epoch5"]
                or not all(prefix["components_equal"].values()) or not prefix["native_probabilities_exact"]):
            raise ValueError("Full refit exact prefix failed")
        manifest = read(root / "data/manifest.json")
        test = pd.read_parquet(root / "data/test.parquet")
        sample = pd.read_csv(root / "data/sample_submission.csv", dtype={"id": str})
        if (len(test) != TEST_ROWS or not test.id.is_unique or len(sample) != TEST_ROWS or not sample.id.is_unique
                or not np.array_equal(test.id.astype(str), sample.id)):
            raise ValueError("Test/sample keyed identity differs")
        model = load_realmlp_categorical(directory / "epoch_016", device="cuda")
        first = probabilities(model, transform, cloud_features(test, recipe), 32768)
        del model, test, transform; gc.collect(); torch.cuda.empty_cache()
        raw = pd.read_csv(root / "data/test.csv", dtype={"id": np.int64, **{c: str for c in CAT}})
        if list(raw.columns) != ["id", *manifest["feature_columns"]] or not np.array_equal(raw.id.astype(str), sample.id):
            raise ValueError("Raw test CSV schema/order differs")
        for col in manifest["feature_columns"]:
            if col not in CAT: raw[col] = pd.to_numeric(raw[col], errors="raise").astype("float32")
        reloaded_transform = CategoricalTransform.load(directory / "transform.json")
        reloaded = load_realmlp_categorical(directory / "epoch_016", device="cuda")
        second = probabilities(reloaded, reloaded_transform, cloud_features(raw, recipe), 8191)
        np.testing.assert_allclose(second, first, rtol=1e-5, atol=2e-6)
        if time.monotonic() >= deadline: raise TimeoutError("Full-test inference deadline")
        path = out / "test_predictions.parquet"
        test_ids = sample.id.to_numpy(dtype=np.int64)
        pd.DataFrame({"id": test_ids, "prediction": first}).to_parquet(path, index=False)
        checked_prediction = pd.read_parquet(path)
        if not np.array_equal(checked_prediction.id, test_ids) or not np.array_equal(checked_prediction.prediction, first):
            raise ValueError("Prediction serialization changed")
        native = {rel(root, p): sha(p) for p in directory.rglob("*") if p.is_file() and (p.name in {"graph.pt", "metadata.json", "transform.json"})}
        sources = {**registry["source_hashes"], **{rel(root, p): sha(p) for p in (out / "runtime_source").rglob("*.py")}}
        if {rel(out / "runtime_source", p) for p in (out / "runtime_source").rglob("*.py")} != {
                name.replace(".", "/") + ".py" for name in RUNTIME_MODULES}:
            raise ValueError("Full refit installed runtime snapshot inventory differs")
        verification = {"id": registry["id"], "stage": registry["stage"], "status": "passed", "rows": TEST_ROWS,
            "train_rows": TRAIN_ROWS, "class_order": [0, 1], "full_test_native_verified": True,
            "full_test_raw_inference": True, "audit_evaluated": False, "selection_sha256": registry["selection_sha256"],
            "registry_sha256": sha(root / "registry.json"), "test_predictions_path": rel(root, path),
            "test_predictions_sha256": sha(path), "native_hashes": native, "source_hashes": sources,
            "expected_test_ids_sha256": hashlib.sha256(test_ids.astype("<i8").tobytes()).hexdigest(),
            "keyed_schema_verified": True, "native_reload_verified": True, "atol": 2e-6, "rtol": 1e-5,
            "full_test_max_abs_error": float(np.abs(second-first).max()), "chunks": [32768, 8191],
            "raw_test_sha256": sha(root / "data/test.csv"), "sample_submission_sha256": sha(root / "data/sample_submission.csv"),
            "training_ids_sha256": context["training_ids_sha256"], "prefix_match_sha256": sha(directory / "prefix_match.json"),
            "native_execution_location": "private Kaggle cloud; local composer verifies receipts",
            "heldout_quality_metrics_computed": False, "training_probe_metrics_recorded": True}
        write(out / "verification.json", verification)
    write(directory / "done.json", {"status": "completed", "arm": arm, "train_rows": TRAIN_ROWS,
          "trajectory_sha256": sha(directory / "trajectory.json"), "selection_sha256": registry["selection_sha256"]})


def bounded(command, deadline, log, cwd):
    from supervisor import capture_descendants, terminate_owned, surviving_descendants
    import psutil
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        parent = psutil.Process(process.pid)
        record = {"pid": parent.pid, "create_time": parent.create_time(), "command": parent.cmdline()}
        try:
            while process.poll() is None:
                capture_descendants(record)
                if now() >= deadline: raise TimeoutError("Registered full-refit process deadline")
                time.sleep(1)
            if process.returncode != 0: raise RuntimeError("Full-refit child failed; inspect preserved log")
        finally:
            if process.poll() is None or surviving_descendants(record): terminate_owned(record)
            write(log.with_suffix(".ownership.json"), record)
            if surviving_descendants(record): raise RuntimeError("Owned full-refit descendant survived cleanup")


def execute(root, registry):
    strict_start(root, registry, "release_controller")
    verify_installed_runtime(root, registry, "release_controller")
    out = root / "artifacts" / registry["id"]
    deadline = datetime.fromisoformat(read(out / "job_runtime.json")["fit_deadline_utc"])
    for arm in ("A", "C"):
        verify_workspace(root)
        bounded([sys.executable, str(root / "scripts/continuation_six_release_v1.py"), "worker", "--workspace", str(root), "--arm", arm],
                deadline, out / ("worker_" + arm + ".log"), root)
        if read(out / ("trajectory_" + arm) / "done.json")["status"] != "completed": raise ValueError("Missing full refit receipt")
    roles = out / "runtime_amendment/processes"
    if {p.stem for p in roles.glob("*.json")} != {"release_controller", "release_worker_A", "release_worker_C"}:
        raise ValueError("Incomplete strict release startup roles")
    write(out / "completion_receipt.json", {"id": registry["id"], "status": "completed", "registry_sha256": sha(root / "registry.json"),
          "completed_fit_count": 2, "completed_prefix_count": 1, "full_test_native_verified": True, "release_ready": True,
          "verification_sha256": sha(out / "verification.json"), "selection_sha256": registry["selection_sha256"],
          "completed_startup_count": 3, "completed_utc": now().isoformat()})


def save_return(root, destination, cid):
    names = [rel(root, p) for part in ("artifacts", "state", "logs") for p in (root / part / cid).rglob("*") if p.is_file()]
    names.extend(["registry.json", f"configs/{cid}.json"])
    files = {name: {"sha256": sha(root / name), "bytes": (root / name).stat().st_size} for name in sorted(set(names))}
    write(destination / "output-manifest.json", {"id": cid, "files": files})
    with zipfile.ZipFile(destination / "results.zip", "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in files: archive.write(root / name, name)
        archive.write(destination / "output-manifest.json", "output-manifest.json")
    return {"archive_sha256": sha(destination / "results.zip"), "manifest_sha256": sha(destination / "output-manifest.json"), "file_count": len(files)}


def bootstrap():
    if sys.platform != "linux" or not Path("/kaggle/input").is_dir(): raise RuntimeError("Full-data fitting is Kaggle-only")
    os.environ.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    started = now(); cid = STAGE + "_release"
    destination = Path("/kaggle/working/fixed_epoch_return"); destination.mkdir(exist_ok=False)
    root = Path("/kaggle/working/continuation_release_workspace/payload")
    result = {"id": cid, "status": "starting", "started_utc": started.isoformat(), "local_training": False,
              "heldout_quality_metrics_computed": False, "training_probe_metrics_recorded": True}
    write(destination / "cloud_status.json", result)
    try:
        matches = [p.parent for p in Path("/kaggle/input").rglob("bundle-manifest.json") if sha(p) == EXPECTED_MANIFEST]
        if len(matches) != 1: raise ValueError("Require unique hash-matched mounted release payload")
        manifest = read(matches[0] / "bundle-manifest.json")
        actual = {rel(matches[0], p) for p in matches[0].rglob("*") if p.is_file()}
        if actual != set(manifest["files"]) | {"bundle-manifest.json"}: raise ValueError("Full release input inventory differs")
        for name, item in manifest["files"].items():
            if sha(checked(matches[0], name)) != item["sha256"]: raise ValueError("Mounted release bytes differ")
        shutil.copytree(matches[0], root)
        sys.path.insert(0, str(root / "scripts")); os.environ["PYTHONPATH"] = str(root / "scripts")
        registry = verify_workspace(root)
        if registry["id"] != cid: raise ValueError("Release runtime identity differs")
        out = root / "artifacts" / cid; out.mkdir(parents=True)
        write(out / "job_runtime.json", {"started_utc": started.isoformat(), "fit_deadline_utc": (started+timedelta(seconds=2100)).isoformat(),
              "delivery_deadline_utc": (started+timedelta(seconds=2340)).isoformat(), "hard_timeout_seconds": 2400})
        from reviewed_bootstrap import install_minimal
        install_minimal(out, result)
        versions = {k: importlib.metadata.version(k) for k in RUNTIME_PACKAGES}
        if versions != registry["expected_versions"]: raise ValueError("Changed cloud release stack versus completed experiment")
        write(out / "environment.json", {"versions": versions, "dependency_install": result.get("dependency_install")})
        policy_folder = out / "runtime_amendment"; policy_folder.mkdir()
        copy_bound(root / "provenance/execution_policy.json", policy_folder / "registration.json", sha(root / "provenance/execution_policy.json"))
        bounded([sys.executable, str(root / "scripts/continuation_six_release_v1.py"), "execute", "--workspace", str(root)],
                started + timedelta(seconds=2120), out / "controller.log", root)
        if read(out / "completion_receipt.json")["status"] != "completed": raise ValueError("Full release incomplete")
        result.update(status="release_complete", completed_fits=2, completed_utc=now().isoformat())
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        result["finished_utc"] = now().isoformat()
        replace_status(destination / "cloud_status.json", result)
        if (root / "registry.json").exists(): result["returned_artifacts"] = save_return(root, destination, cid)
        replace_status(destination / "cloud_status.json", result)
        print(json.dumps(result, indent=2))


def main():
    if EXPECTED_MANIFEST != "RELEASE_MANIFEST_" + "PLACEHOLDER":
        bootstrap(); return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "execute", "worker"])
    parser.add_argument("--stage"); parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--workspace", type=Path); parser.add_argument("--arm", choices=["A", "C"])
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.stage or args.test_receipt is None: parser.error("Prepare needs stage and generated verification receipt")
        print(json.dumps(prepare(ROOT, args.stage, args.test_receipt.resolve()), indent=2)); return
    if sys.platform != "linux" or not Path("/kaggle/input").is_dir(): raise RuntimeError("No local full-data fitting")
    if args.workspace is None: parser.error("Cloud role requires workspace")
    root = args.workspace.resolve()
    if not root.is_relative_to(Path("/kaggle/working")): raise ValueError("Unregistered full-refit workspace")
    registry = verify_workspace(root)
    if args.action == "execute": execute(root, registry)
    else:
        if args.arm is None: parser.error("Worker requires A or C")
        worker(root, registry, args.arm)


if __name__ == "__main__":
    main()
