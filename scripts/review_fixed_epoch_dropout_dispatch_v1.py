"""Verify prepared upload bytes and freeze a reviewed one-shot dispatch contract.

Read-only checks until the final exclusive receipt; no network, training or scores.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from prepare_fixed_epoch_dropout_v1 import (
    ROOT, CLOUD, ID, PARENT, PARENT_SHA, POLICY, POLICY_SHA, OLD_INPUTS,
    PARENT_PAYLOAD, INCUMBENT_SHA, SPLIT_SHA, validate_policy,
)
from run_fixed_epoch_dropout_v1 import validate_protocol, verify_bundle


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    cloud = ROOT / CLOUD
    destination = cloud / "dispatch_review.json"
    if destination.exists():
        raise FileExistsError("Preserve the existing dispatch review")
    prepared = read(cloud / "preparation_manifest.json")
    if prepared["id"] != ID or prepared["status"] != "prepared_not_uploaded":
        raise ValueError("Wrong preparation")
    files = dict(prepared["frozen_files"])
    payload = cloud / "bundle/payload"
    manifest = verify_bundle(payload, prepared["manifest_sha256"])
    parent = read(ROOT / PARENT)
    if sha(ROOT / PARENT) != PARENT_SHA or sha(ROOT / POLICY) != POLICY_SHA:
        raise ValueError("Parent or policy differs")
    validate_policy(read(ROOT / POLICY))
    registry = read(payload / "registry.json")
    validate_protocol(registry)
    if registry["recipe"] != parent["recipe"] or registry["evaluation"] != parent["evaluation"]:
        raise ValueError("Recipe or fixed assessment gate changed")
    for name in OLD_INPUTS:
        if sha(payload / name) != sha(ROOT / PARENT_PAYLOAD / name):
            raise ValueError("Previously filtered data/provenance changed")
    anchors = {
        "artifacts/third_pass/blend/frozen.json": INCUMBENT_SHA,
        "data/splits.parquet": SPLIT_SHA,
        PARENT: PARENT_SHA,
        "scripts/fixed_epoch_cloud_adapter_v1.py": "8ed0b2bf48a6975d0020f06a159ff0c4a289acb2bc8a816fccaabbf180423854",
        "scripts/evaluate_fixed_epoch_cloud_v1.py": "93825d0e64fdf0fc7b8629965e89e712d5a4300cf1b6c65370dcb9b1b21482f7",
    }
    files.update(anchors)
    independent_path = ROOT / "state/fixed_epoch_dropout_v1/independent_launch_review.json"
    independent = read(independent_path)
    if independent["status"] != "passed" or independent.get("blockers"):
        raise ValueError("Independent review has not cleared launch")
    files.update(independent["source_hashes"])
    required_review = {f"scripts/{name}" for name in (
        "fixed_epoch_dropout_adapter_v1.py", "run_fixed_epoch_dropout_v1.py",
        "smoke_fixed_epoch_dropout_v1.py", "prepare_fixed_epoch_dropout_v1.py",
        "kaggle_fixed_epoch_dropout_v1.py")}
    if not required_review.issubset(independent["source_hashes"]):
        raise ValueError("Independent review omits a dispatch source")
    source = ROOT / "scripts/run_fixed_epoch_dropout_v1.py"
    expected_entry = source.read_text(encoding="utf-8").replace(
        "MANIFEST_HASH_PLACEHOLDER", prepared["manifest_sha256"])
    if (cloud / "run.py").read_text(encoding="utf-8") != expected_entry:
        raise ValueError("Deployed source differs from reviewed source")
    compile(expected_entry, "run.py", "exec")
    archive_path = cloud / "bundle/upload/payload.zip"
    expected_names = {"payload/" + name for name in manifest["files"]} | {"payload/bundle-manifest.json"}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise ValueError("Upload archive allowlist differs")
        for name in names:
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != sha(payload / name.removeprefix("payload/")):
                raise ValueError("Upload archive member hash mismatch")
    for path in (Path(__file__), independent_path, cloud / "preparation_manifest.json",
                 ROOT / "scripts/kaggle_fixed_epoch_dropout_v1.py", ROOT / "scripts/kaggle_cloud_control.py"):
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError(f"Reviewed file changed: {name}")
    result = {
        "status": "passed", "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "hard_timeout_seconds": 7200, "files": files,
        "scope": "One private free paired dropout intervention, twelve fits, no incumbent change",
        "verification": "All manifest/archive members, exact reviewed runtime substitution, reused development input bytes, unchanged recipe/evaluation and preservation anchors verified",
        "independent_review": independent_path.relative_to(ROOT).as_posix(),
        "archive_members_verified": len(expected_names), "cloud_gpu_jobs_max": 1,
        "dataset_private_required": True, "kernel_private_required": True,
        "local_gpu_used": False, "real_data_fitting_performed": False,
        "cloud_full_smoke_required_before_real_fits": True,
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
