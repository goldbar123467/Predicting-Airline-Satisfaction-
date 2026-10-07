"""Prepare a new private dropout package from byte-identical development-only parent data.

No training, metric computation, raw competition-table read or Kaggle API call occurs.
The parent allowlist and original split/anchor pins remain immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
ID = "fixed_epoch_dropout_v1"
CLOUD = "cloud/fixed_epoch_dropout_v1"
POLICY = "configs/fixed_epoch_dropout_policy_v1.json"
POLICY_SHA = "e7d8b0959a74b21f7943d26ab3cd88cade4bd41690334c8977f9c74eb0153cc4"
PARENT = "artifacts/fixed_epoch_cloud_v1/registry.json"
PARENT_SHA = "2aa065324331a5eed7ad464c82bcd6d967d91d38c5a453f9a7a71ee9fd03266d"
PARENT_PAYLOAD = "cloud/fixed_epoch_v1/bundle/payload"
PARENT_MANIFEST_SHA = "d7a5cce4652f33f3a81512d1b18681942ffc7065b931b3d014fc475e19ab494b"
HELPER = "scripts/prepare_fixed_epoch_cloud_v1.py"
HELPER_SHA = "028eee571e1f0e12970f7e42832157c9854a20159a9f769ff86a751583b7e236"
SPLIT_SHA = "4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c"
INCUMBENT_SHA = "bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311"
DATASET = "clarkkitchen/s6e10-dropout-bundle-20261004"
KERNEL = "clarkkitchen/s6e10-dropout-20261004"
BUDGET = {"hard_timeout_seconds": 7200, "fit_budget_seconds": 6300, "remote_smoke_timeout_seconds": 330}
EVALUATION = {"mixture_alpha": .1, "min_pooled_gain": 1e-5, "min_macro_gain": 1e-5,
              "max_fold_regression": 2e-5, "class_order": [0, 1], "fold_ids": [0, 1, 2]}
ENDPOINTS = {"A": {"trajectory": "A", "epoch": 16}, "B": {"trajectory": "C", "epoch": 4},
             "C": {"trajectory": "C", "epoch": 16}}
SCIENCE = {"expected_fit_count": 12, "expected_outer_endpoint_count": 9, "total_executed_epochs": 192,
           "common_horizon_epochs": 16, "intervention_after_completed_epoch": 4, "treatment_dropout_base": .05,
           "logical_endpoints": ENDPOINTS, "evaluation": EVALUATION}
OLD_SOURCES = ["common.py", "categorical_transform.py", "realmlp.py", "realmlp_categorical.py",
               "supervisor.py", "long_local_500_policy.py", "reviewed_bootstrap.py", "cloud_feature_compat.py"]
NEW_SOURCES = ["run_fixed_epoch_dropout_v1.py", "fixed_epoch_dropout_adapter_v1.py", "smoke_fixed_epoch_dropout_v1.py"]
OLD_INPUTS = ["data/train.parquet", "data/splits.parquet", "data/original_aux_predictions.parquet", "data/manifest.json",
              "provenance/original_aux_manifest.json", "licenses/pytabkit_LICENSE.txt", "ATTRIBUTION.md"]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def helpers(root: Path):
    path = root / HELPER
    if sha(path) != HELPER_SHA:
        raise ValueError("Frozen packaging helper changed")
    spec = importlib.util.spec_from_file_location("frozen_packaging_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_policy(policy: dict) -> None:
    expected = {"id": ID, **BUDGET, **SCIENCE, "parent_registry_path": PARENT,
        "parent_registry_sha256": PARENT_SHA, "prepared_parent_payload": PARENT_PAYLOAD,
        "parent_bundle_manifest_sha256": PARENT_MANIFEST_SHA, "maximum_gpu_jobs": 1,
        "paid_compute": False, "private_required": True, "training_location": "Kaggle only",
        "cloud_namespace": CLOUD, "dataset_id": DATASET, "kernel_id": KERNEL}
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("Dropout authorization/protocol differs")


def receipt_sources(root: Path, path: Path, required: set[str], minimum: int, helper) -> dict:
    receipt = helper.read(path)
    count = receipt.get("tests_passed", receipt.get("tests", 0))
    if (receipt.get("status") != "passed" or not isinstance(count, int) or count < minimum
            or receipt.get("returncode", 0) != 0):
        raise ValueError("Synthetic test receipt did not pass")
    sources = receipt.get("source_hashes", receipt.get("source_sha256", {}))
    result = {}
    for name, expected in sources.items():
        source = helper.checked(root, name)
        helper.require_hash(source, expected)
        result[helper.relative(root, source)] = expected
    if not required.issubset(result):
        raise ValueError("Synthetic receipt omits required source binding")
    return result


def validate_parent_manifest(parent: dict, manifest: dict) -> None:
    if (manifest.get("id") != "fixed_epoch_cloud_v1" or manifest.get("registry_sha256") != PARENT_SHA
            or manifest.get("audit_rows_exported") != 0 or manifest.get("test_rows_exported") != 0
            or manifest.get("development_rows") != 629671 or manifest.get("source_split_sha256") != SPLIT_SHA
            or parent.get("source_split_sha256") != SPLIT_SHA
            or parent.get("incumbent_selection_sha256") != INCUMBENT_SHA or parent.get("evaluation") != EVALUATION):
        raise ValueError("Parent development/provenance contract differs")
    for name in OLD_INPUTS:
        if manifest["files"][name]["sha256"] != parent["input_hashes"][name]:
            raise ValueError("Parent input binding differs")
    for name in OLD_SOURCES:
        relative = f"scripts/{name}"
        if manifest["files"][relative]["sha256"] != parent["source_hashes"][relative]:
            raise ValueError("Parent source binding differs")


def prepare(root: Path, adapter_tests: Path, runner_tests: Path) -> dict:
    start = time.monotonic()
    root = root.resolve()
    helper = helpers(root)
    cloud = root / CLOUD
    payload = cloud / "bundle/payload"
    final = cloud / "preparation_manifest.json"
    if final.exists():
        saved = helper.read(final)
        for name, expected in saved["frozen_files"].items():
            helper.require_hash(helper.checked(root, name), expected)
        return saved
    if ((cloud / "preparation_claim.json").exists() or payload.parent.exists()
            or (root / f"configs/{ID}.json").exists() or (root / f"artifacts/{ID}/registry.json").exists()):
        raise FileExistsError("Existing preparation is preserved; implicit retry forbidden")
    helper.require_hash(root / POLICY, POLICY_SHA)
    policy = helper.read(root / POLICY)
    validate_policy(policy)
    helper.require_hash(root / policy["proposal_path"], policy["proposal_sha256"])
    helper.require_hash(root / PARENT, PARENT_SHA)
    parent = helper.read(root / PARENT)
    old = root / PARENT_PAYLOAD
    helper.require_hash(old / "bundle-manifest.json", PARENT_MANIFEST_SHA)
    old_manifest = helper.read(old / "bundle-manifest.json")
    validate_parent_manifest(parent, old_manifest)
    helper.require_hash(root / "data/splits.parquet", SPLIT_SHA)
    helper.require_hash(root / "artifacts/third_pass/blend/frozen.json", INCUMBENT_SHA)
    adapter_receipt = helper.checked(root, adapter_tests)
    runner_receipt = helper.checked(root, runner_tests)
    adapter_sources = receipt_sources(root, adapter_receipt,
        {"scripts/fixed_epoch_dropout_adapter_v1.py", "scripts/test_fixed_epoch_dropout_adapter_v1.py"}, 1, helper)
    runner_sources = receipt_sources(root, runner_receipt,
        {"scripts/run_fixed_epoch_dropout_v1.py", "scripts/test_run_fixed_epoch_dropout_v1.py",
         "scripts/smoke_fixed_epoch_dropout_v1.py", "scripts/prepare_fixed_epoch_dropout_v1.py",
         "scripts/test_prepare_fixed_epoch_dropout_v1.py"}, 1, helper)
    source_map = {f"scripts/{name}": (old / f"scripts/{name}", parent["source_hashes"][f"scripts/{name}"])
                  for name in OLD_SOURCES}
    source_map.update({f"scripts/{name}": (root / f"scripts/{name}", sha(root / f"scripts/{name}")) for name in NEW_SOURCES})
    for source, expected in source_map.values():
        helper.require_hash(source, expected)
    if source_map["scripts/fixed_epoch_dropout_adapter_v1.py"][1] != adapter_sources["scripts/fixed_epoch_dropout_adapter_v1.py"]:
        raise ValueError("Adapter changed after synthetic proof")
    runtime = (root / "scripts/run_fixed_epoch_dropout_v1.py").read_text(encoding="utf-8")
    if runtime.count("MANIFEST_HASH_PLACEHOLDER") != 1:
        raise ValueError("Runtime manifest placeholder count differs")
    compile(runtime, "run.py", "exec")
    data = helper.read(old / "data/manifest.json")
    if (data["rows"] != {"development": 629671, "train": 629671, "audit": 0, "test": 0}
            or data["source_split_sha256"] != SPLIT_SHA or data["audit_rows_exported"] != 0
            or data["test_rows_exported"] != 0 or data["split_sha256"] != parent["input_hashes"]["data/splits.parquet"]):
        raise ValueError("Frozen filtered-data manifest differs")
    source_hashes = {name: value[1] for name, value in source_map.items()}
    prep_sources = {HELPER: HELPER_SHA, **adapter_sources, **runner_sources}
    claim = {"id": ID, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": source_hashes, "preparation_source_hashes": prep_sources,
        "policy_sha256": POLICY_SHA, "parent_registry_sha256": PARENT_SHA,
        "parent_bundle_manifest_sha256": PARENT_MANIFEST_SHA,
        "adapter_test_receipt_sha256": sha(adapter_receipt), "runner_test_receipt_sha256": sha(runner_receipt),
        "training_performed": False, "cloud_operations_performed": False, "data_reuse": "Exact parent development-only bytes"}
    helper.write(cloud / "preparation_claim.json", claim)
    payload.mkdir(parents=True)
    for name, (source, expected) in source_map.items():
        helper.copy_bound(source, payload / name, expected)
    for name in OLD_INPUTS:
        helper.copy_bound(old / name, payload / name, parent["input_hashes"][name])
    evidence = {"provenance/cloud_policy.json": root / POLICY,
        "provenance/portable_adapter_tests.json": adapter_receipt,
        "provenance/runner_preparation_tests.json": runner_receipt,
        "provenance/dropout_proposal.md": root / policy["proposal_path"]}
    for name, source in evidence.items():
        helper.copy_bound(source, payload / name, sha(source))
    inputs = {helper.relative(payload, path): sha(path) for path in payload.rglob("*")
              if path.is_file() and helper.relative(payload, path) not in source_hashes}
    registry = {"id": ID, "schema_version": 1, "registered_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": parent["recipe"], **SCIENCE, "expected_total_epochs": 192, "development_rows": 629671,
        "source_hashes": source_hashes, "input_hashes": inputs, "source_split_sha256": SPLIT_SHA,
        "incumbent_selection_sha256": INCUMBENT_SHA, "parent_cloud_registry_sha256": PARENT_SHA,
        "parent_bundle_manifest_sha256": PARENT_MANIFEST_SHA, "parent_hashes": parent["parent_hashes"],
        "preparation_source_hashes": prep_sources, "cloud_policy_sha256": POLICY_SHA,
        "trajectory_policies": policy["trajectory_policies"], "prefix_contract": policy["prefix_contract"],
        "arms": {name: {"horizon": 16, "endpoint": endpoint["epoch"], "trajectory": endpoint["trajectory"]}
                 for name, endpoint in ENDPOINTS.items()},
        "execution_order": parent["execution_order"], "inner_monitor_fraction": .1,
        "inner_partition_seeds": [20261005, 20261006, 20261007], "model_seed_policy": parent["model_seed_policy"],
        "acceptance": parent["acceptance"], "native_jit_optimized_execution": False, "native_cpu_threads": 4, **BUDGET,
        "deadline_policy": parent["deadline_policy"], "execution_image": parent["execution_image"],
        "dataset_id": DATASET, "kernel_id": KERNEL, "private_required": True, "training_device": "cuda:0",
        "remote_smoke_before_real_fits": True, "audit_rows_exported": 0, "test_rows_exported": 0,
        "incumbent_oof_exported": False, "cloud_quality_scoring": False, "post_download_local_evaluation_only": True,
        "prohibitions": parent["prohibitions"], "scope_amendment": "One explicitly authorized private dropout-only cloud experiment; preserve current blend; no promotion, submission or full-data refit",
        "evidence_limit": policy["new_environment_limit"], "data_reuse": "Byte-identical filtered development files from pinned parent payload"}
    helper.write(payload / "registry.json", registry)
    wrapper = {"id": ID, "registry_path": "registry.json", "registry_sha256": sha(payload / "registry.json"),
        "output_dir": f"artifacts/{ID}", "split_path": "data/splits.parquet", "split_sha256": inputs["data/splits.parquet"],
        "source_split_sha256": SPLIT_SHA, "incumbent_selection_sha256": INCUMBENT_SHA,
        "completed_manifest_path": f"artifacts/{ID}/completed_manifest.json", "completion_receipt_path": f"artifacts/{ID}/completion_receipt.json"}
    helper.write(payload / f"configs/{ID}.json", wrapper)
    allowed = set(source_hashes) | set(inputs) | {"registry.json", f"configs/{ID}.json"}
    actual = {helper.relative(payload, path) for path in payload.rglob("*") if path.is_file()}
    if actual != allowed:
        raise ValueError("Payload file allowlist differs")
    manifest = {"schema_version": 1, "id": ID, "dataset_id": DATASET, "private_required": True,
        "development_rows": 629671, "audit_rows_exported": 0, "test_rows_exported": 0,
        "development_id_sha256": data["development_id_sha256"], "source_split_sha256": SPLIT_SHA,
        "parent_hashes": parent["parent_hashes"], "parent_bundle_manifest_sha256": PARENT_MANIFEST_SHA,
        "registry_sha256": wrapper["registry_sha256"],
        "files": {name: {"sha256": sha(payload / name), "bytes": (payload / name).stat().st_size} for name in sorted(allowed)}}
    helper.write(payload / "bundle-manifest.json", manifest)
    manifest_hash = sha(payload / "bundle-manifest.json")
    upload = cloud / "bundle/upload"
    archive = upload / "payload.zip"
    helper.archive_payload(payload, archive, [*sorted(allowed), "bundle-manifest.json"])
    helper.write(upload / "dataset-metadata.json", {"id": DATASET, "title": "S6E10 Dropout Bundle 20261004",
        "licenses": [{"name": "other"}], "description": "PRIVATE development-only dropout experiment. No audit/test rows or incumbent predictions. See payload ATTRIBUTION.md. Do not publish."})
    deployed = runtime.replace("MANIFEST_HASH_PLACEHOLDER", manifest_hash)
    compile(deployed, "run.py", "exec")
    with (cloud / "run.py").open("x", encoding="utf-8") as stream:
        stream.write(deployed)
    helper.write(cloud / "kernel-metadata.json", {"id": KERNEL, "title": "S6E10 Dropout 20261004",
        "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "docker_image": parent["execution_image"], "docker_image_pinning_type": "original", "dataset_sources": [DATASET],
        "competition_sources": [], "kernel_sources": [], "model_sources": []})
    helper.copy_bound(payload / "registry.json", root / f"artifacts/{ID}/registry.json", wrapper["registry_sha256"])
    helper.copy_bound(payload / f"configs/{ID}.json", root / f"configs/{ID}.json", sha(payload / f"configs/{ID}.json"))
    for source, expected in source_map.values():
        helper.require_hash(source, expected)
    for name, expected in prep_sources.items():
        helper.require_hash(root / name, expected)
    for name in OLD_INPUTS:
        helper.require_hash(old / name, parent["input_hashes"][name])
    helper.require_hash(adapter_receipt, claim["adapter_test_receipt_sha256"])
    helper.require_hash(runner_receipt, claim["runner_test_receipt_sha256"])
    helper.require_hash(root / POLICY, POLICY_SHA)
    helper.require_hash(root / PARENT, PARENT_SHA)
    helper.require_hash(old / "bundle-manifest.json", PARENT_MANIFEST_SHA)
    helper.require_hash(root / policy["proposal_path"], policy["proposal_sha256"])
    frozen = [path for path in cloud.rglob("*") if path.is_file()] + [root / f"artifacts/{ID}/registry.json",
        root / f"configs/{ID}.json", root / POLICY, root / policy["proposal_path"], adapter_receipt, runner_receipt]
    prepared = {"id": ID, "status": "prepared_not_uploaded", "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET, "kernel_id": KERNEL, "private_required": True,
        "dataset_create_requires_public_false": True, "remote_privacy_readback_required_before_execution": True,
        "manifest_sha256": manifest_hash, "registry_sha256": wrapper["registry_sha256"],
        "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size, "runtime_sha256": sha(cloud / "run.py"),
        "development_rows": 629671, "audit_rows_exported": 0, "test_rows_exported": 0,
        "training_performed": False, "cloud_operations_performed": False, "seconds": time.monotonic() - start,
        **BUDGET, "frozen_files": {helper.relative(root, path): sha(path) for path in frozen}}
    helper.write(final, prepared)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-tests", type=Path, required=True)
    parser.add_argument("--runner-tests", type=Path, required=True)
    args = parser.parse_args()
    value = prepare(ROOT, args.adapter_tests, args.runner_tests)
    print(json.dumps({key: value[key] for key in value if key != "frozen_files"}, indent=2))


if __name__ == "__main__":
    main()
