"""Freeze one allowlisted private continuation experiment. Never fit or upload."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from continuation_six_policy_v1 import policy_for, policy_for_stage

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "cloud/fixed_epoch_dropout_v1/bundle/payload"
PARENT_SHA = "16d88ba105ce5e4f8c61cc22d45d3f7242ae83f8796a47fed2f22189cd4c9e5b"
SOURCES = ("run_continuation_six_v1.py", "continuation_six_adapter_v1.py",
           "continuation_six_policy_v1.py", "smoke_continuation_six_v1.py",
           "continuation_six_deterministic_v1.py")
REUSED = ("common.py", "categorical_transform.py", "realmlp.py", "realmlp_categorical.py",
          "supervisor.py", "long_local_500_policy.py", "reviewed_bootstrap.py", "cloud_feature_compat.py")
INPUTS = ("data/train.parquet", "data/splits.parquet", "data/original_aux_predictions.parquet",
          "data/manifest.json", "provenance/original_aux_manifest.json", "licenses/pytabkit_LICENSE.txt", "ATTRIBUTION.md")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def copy_bound(source: Path, destination: Path, expected: str) -> None:
    if sha(source) != expected:
        raise ValueError(f"Source/input identity changed: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    if sha(destination) != expected:
        raise ValueError("Copied bytes changed")


def validate_tests(receipt: Path) -> dict:
    result = read(receipt)
    if result.get("status") != "passed" or result.get("returncode", 0) != 0:
        raise ValueError("Generated verification did not pass")
    sources = result.get("source_hashes", result.get("source_sha256", {}))
    if not sources or any(sha(ROOT / path) != digest for path, digest in sources.items()):
        raise ValueError("Source changed after generated verification")
    return sources


def prepare(stage: str, adapter_receipt: Path, infrastructure_receipt: Path) -> dict:
    plan_path = ROOT / "configs/continuation_six_v1.json"
    authority_path = ROOT / "state/continuation_six_v1/authorization.json"
    plan, authority = read(plan_path), read(authority_path)
    if (authority["status"] != "authorized" or sha(plan_path) != authority["plan_sha256"]
            or datetime.now(timezone.utc) >= datetime.fromisoformat(authority["expires_utc"])):
        raise ValueError("Sequence authorization or deadline differs")
    matches = [item for item in plan["stages"] if item["id"] == stage]
    if len(matches) != 1:
        raise ValueError("Unknown stage")
    selected = matches[0]
    if selected["policy"] != policy_for_stage(stage) or selected["control"] != policy_for("control"):
        raise ValueError("Stage policy differs")
    folder = ROOT / "cloud/continuation_six_v1" / stage / "experiment"
    if folder.exists():
        raise FileExistsError("Preparation namespace already exists; inspect, never overwrite")
    if sha(PARENT / "bundle-manifest.json") != PARENT_SHA:
        raise ValueError("Frozen development payload changed")
    parent_manifest, parent = read(PARENT / "bundle-manifest.json"), read(PARENT / "registry.json")
    if parent_manifest["development_rows"] != 629671 or parent_manifest["audit_rows_exported"] != 0 or parent_manifest["test_rows_exported"] != 0:
        raise ValueError("Parent development-only scope differs")
    proofs = {**validate_tests(adapter_receipt), **validate_tests(infrastructure_receipt)}
    if not {f"scripts/{name}" for name in SOURCES}.issubset(proofs):
        raise ValueError("Generated proofs omit deployed continuation sources")
    for name, digest in {"data/splits.parquet": plan["source_split_sha256"],
                         plan["incumbent_selection_path"]: plan["incumbent_selection_sha256"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Canonical split/incumbent changed")
    payload = folder / "bundle/payload"
    index = selected["index"]
    dataset = f"clarkkitchen/s6e10-continuation-0{index}-bundle-20261004"
    kernel = f"clarkkitchen/s6e10-continuation-0{index}-experiment-20261004"
    folder.mkdir(parents=True)
    write(folder / "preparation_claim.json", {"stage": stage, "plan_sha256": sha(plan_path),
          "created_utc": datetime.now(timezone.utc).isoformat(), "sources_after_tests": proofs,
          "adapter_receipt_sha256": sha(adapter_receipt), "infrastructure_receipt_sha256": sha(infrastructure_receipt)})
    for name in SOURCES:
        copy_bound(ROOT / "scripts" / name, payload / "scripts" / name, proofs[f"scripts/{name}"])
    for name in REUSED:
        key = f"scripts/{name}"
        copy_bound(PARENT / key, payload / key, parent_manifest["files"][key]["sha256"])
    for name in INPUTS:
        copy_bound(PARENT / name, payload / name, parent_manifest["files"][name]["sha256"])
    for name, source in {"sequence_plan.json": plan_path, "sequence_authorization.json": authority_path,
                         "adapter_tests.json": adapter_receipt, "infrastructure_tests.json": infrastructure_receipt}.items():
        copy_bound(source, payload / "provenance" / name, sha(source))
    hook_sha = sha(payload / "scripts/continuation_six_deterministic_v1.py")
    execution = {"id": stage + "_strict_execution", "campaign": stage,
                 "created_utc": datetime.now(timezone.utc).isoformat(), "hook_source_sha256": hook_sha,
                 "sequence_plan_sha256": sha(plan_path), "sequence_authorization_sha256": sha(authority_path),
                 "execution_policy": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "torch_deterministic_algorithms": True,
                                      "torch_deterministic_warn_only": False, "cudnn_benchmark": False,
                                      "cudnn_deterministic": True}, "matmul_allow_tf32": False,
                 "startup_failure_exit_code": 86, "required_startup_roles": 14}
    write(payload / "provenance/execution_policy.json", execution)
    source_hashes = {p.relative_to(payload).as_posix(): sha(p) for p in (payload / "scripts").glob("*.py")}
    inputs = {p.relative_to(payload).as_posix(): sha(p) for p in payload.rglob("*")
              if p.is_file() and p.relative_to(payload).as_posix() not in source_hashes}
    science = plan["science"]
    registry = {"id": stage, "schema_version": 1, "registered_utc": datetime.now(timezone.utc).isoformat(),
                "recipe": parent["recipe"], **science, "expected_total_epochs": 192, "development_rows": 629671,
                "source_hashes": source_hashes, "input_hashes": inputs,
                "source_split_sha256": plan["source_split_sha256"], "incumbent_selection_sha256": plan["incumbent_selection_sha256"],
                "sequence_plan_sha256": sha(plan_path), "authorization_sha256": sha(authority_path),
                "parent_bundle_manifest_sha256": PARENT_SHA, "parent_hashes": parent["parent_hashes"],
                "trajectory_policies": {"A": selected["control"], "C": selected["policy"]},
                "continuation_policy": selected["policy"], "evaluation": plan["evaluation"],
                "treatment_dropout_base": 0.05, "inner_monitor_fraction": .1,
                "inner_partition_seeds": [20261005, 20261006, 20261007], "model_seed_policy": parent["model_seed_policy"],
                "acceptance": parent["acceptance"], "native_jit_optimized_execution": False, "native_cpu_threads": 4,
                "hard_timeout_seconds": 7200, "fit_budget_seconds": 6300, "remote_smoke_timeout_seconds": 330,
                "execution_image": parent["execution_image"], "dataset_id": dataset, "kernel_id": kernel,
                "private_required": True, "training_device": "cuda:0", "remote_smoke_before_real_fits": True,
                "audit_rows_exported": 0, "test_rows_exported": 0, "incumbent_oof_exported": False,
                "cloud_quality_scoring": False, "post_download_local_evaluation_only": True,
                "execution_policy_sha256": sha(payload / "provenance/execution_policy.json")}
    write(payload / "registry.json", registry)
    wrapper = {"id": stage, "registry_path": "registry.json", "registry_sha256": sha(payload / "registry.json"),
               "output_dir": f"artifacts/{stage}", "split_path": "data/splits.parquet", "split_sha256": inputs["data/splits.parquet"],
               "source_split_sha256": plan["source_split_sha256"], "incumbent_selection_sha256": plan["incumbent_selection_sha256"],
               "completed_manifest_path": f"artifacts/{stage}/completed_manifest.json", "completion_receipt_path": f"artifacts/{stage}/completion_receipt.json"}
    write(payload / f"configs/{stage}.json", wrapper)
    allowed = set(source_hashes) | set(inputs) | {"registry.json", f"configs/{stage}.json"}
    manifest = {"id": stage, "schema_version": 1, "dataset_id": dataset, "private_required": True,
                "development_rows": 629671, "audit_rows_exported": 0, "test_rows_exported": 0,
                "source_split_sha256": plan["source_split_sha256"], "registry_sha256": wrapper["registry_sha256"],
                "parent_bundle_manifest_sha256": PARENT_SHA,
                "files": {name: {"bytes": (payload / name).stat().st_size, "sha256": sha(payload / name)} for name in sorted(allowed)}}
    write(payload / "bundle-manifest.json", manifest)
    manifest_sha = sha(payload / "bundle-manifest.json")
    upload = folder / "bundle/upload"
    upload.mkdir()
    archive = upload / "payload.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as packed:
        for name in sorted(allowed | {"bundle-manifest.json"}):
            packed.write(payload / name, name)
    write(upload / "dataset-metadata.json", {"id": dataset, "title": f"S6E10 Continuation 0{index} Bundle 20261004",
          "licenses": [{"name": "other"}], "description": "Private development-only registered experiment. No audit/test rows or incumbent predictions. Preserve attribution; never publish."})
    runtime = (ROOT / "scripts/run_continuation_six_v1.py").read_text()
    if runtime.count('"MANIFEST_HASH_PLACEHOLDER"') != 1:
        raise ValueError("Entry manifest placeholder count differs")
    runtime = runtime.replace('"MANIFEST_HASH_PLACEHOLDER"', json.dumps(manifest_sha))
    runtime = runtime.replace('os.environ.get("CONTINUATION_STAGE_ID", "fixed_epoch_continuation_01")', json.dumps(stage))
    compile(runtime, "run.py", "exec")
    with (folder / "run.py").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(runtime)
    if (folder / "run.py").read_bytes() != runtime.encode("utf-8"):
        raise ValueError("Provider entry newline normalization differs")
    write(folder / "kernel-metadata.json", {"id": kernel, "title": f"S6E10 Continuation 0{index} Experiment 20261004",
          "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
          "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
          "docker_image": parent["execution_image"], "docker_image_pinning_type": "original",
          "dataset_sources": [dataset], "competition_sources": [], "kernel_sources": [], "model_sources": []})
    copy_bound(payload / "registry.json", ROOT / f"artifacts/{stage}/registry.json", wrapper["registry_sha256"])
    copy_bound(payload / f"configs/{stage}.json", ROOT / f"configs/{stage}.json", sha(payload / f"configs/{stage}.json"))
    frozen = [p for p in folder.rglob("*") if p.is_file()] + [adapter_receipt, infrastructure_receipt]
    prepared = {"id": stage, "stage": stage, "phase": "experiment", "private_required": True, "status": "prepared_not_uploaded",
                "created_utc": datetime.now(timezone.utc).isoformat(), "dataset_id": dataset, "kernel_id": kernel,
                "manifest_sha256": manifest_sha, "registry_sha256": wrapper["registry_sha256"], "runtime_sha256": sha(folder / "run.py"),
                "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
                "hard_timeout_seconds": 7200, "fit_budget_seconds": 6300,
                "execution_policy_sha256": registry["execution_policy_sha256"], "hook_source_sha256": hook_sha,
                "source_tests": {"adapter": sha(adapter_receipt), "infrastructure": sha(infrastructure_receipt)},
                "development_rows": 629671, "audit_rows_exported": 0, "test_rows_exported": 0,
                "frozen_files": {p.relative_to(ROOT).as_posix(): sha(p) for p in frozen}}
    preparation = folder / "preparation_manifest.json"
    write(preparation, prepared)
    write(ROOT / f"state/continuation_six_v1/registrations/{stage}.json",
          {"status": "registered", "stage": stage, "phase": "experiment", "plan_sha256": sha(plan_path),
           "preparation_path": preparation.relative_to(ROOT).as_posix(), "preparation_sha256": sha(preparation)})
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--adapter-receipt", type=Path, required=True)
    parser.add_argument("--infrastructure-receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.stage, args.adapter_receipt.resolve(), args.infrastructure_receipt.resolve()), indent=2))


if __name__ == "__main__":
    main()
