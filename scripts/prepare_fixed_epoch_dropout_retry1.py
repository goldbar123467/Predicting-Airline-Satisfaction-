"""Prepare a hash-bound deterministic runtime retry; never upload, fit or score.

The original private dataset, payload, registry and fitted-source bytes are reused.
Only the outer entry point sets stricter deadlines and installs an external startup hook.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "fixed_epoch_dropout_v1"
ID = "fixed_epoch_dropout_v1_deterministic_retry1"
FOLDER = "cloud/fixed_epoch_dropout_v1_retry1"
ORIGINAL = "cloud/fixed_epoch_dropout_v1"
KERNEL = "clarkkitchen/s6e10-dropout-20261004-r2"
DATASET = "clarkkitchen/s6e10-dropout-bundle-20261004"
AUTHORIZATION = "state/fixed_epoch_dropout_v1/deterministic_retry_authorization.json"
AUTHORIZATION_SHA = "c1d23f1adfb1aed334d1c5d42ba88873641fed12609ec0484df1d1c819135d19"
REGISTRY_SHA = "6ec58209d08edc0b29bb330a33404bbd0c1717a1eb451e0aa533aeb77e937282"
BUNDLE_SHA = "16d88ba105ce5e4f8c61cc22d45d3f7242ae83f8796a47fed2f22189cd4c9e5b"
ENTRY_SHA = "e4e1618a85fde7b8db1cb114490ddd2d5dde657b5cf44135045da67c71e46beb"
ARCHIVE_SHA = "756585c1a6db44619c5f9028f51572a8d70c87daafd87706470e288cdd9c6225"
HOOK = "scripts/fixed_epoch_deterministic_execution_v1.py"
CONTROLLER = "scripts/kaggle_fixed_epoch_dropout_retry1.py"
TEST = "scripts/test_prepare_fixed_epoch_dropout_retry1.py"
POLICY = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "torch_deterministic_algorithms": True,
          "torch_deterministic_warn_only": False, "cudnn_benchmark": False, "cudnn_deterministic": True}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"Frozen bytes differ: {path}")


def checked(root: Path, name: str) -> Path:
    path = (root / name.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes project")
    return path


def encode(value: dict) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def validate_authorization(value: dict) -> None:
    expected = {"id": ID, "cloud_status": "failed", "real_fits_completed": 0, "retry_kernel": KERNEL,
        "retry_namespace": FOLDER, "dataset_id": DATASET, "dataset_version": 1,
        "provider_timeout_seconds": 6900, "original_provider_cap_seconds": 7200,
        "failed_attempt_reserve_seconds": 300, "fit_budget_seconds": 6300,
        "maximum_concurrent_gpu_jobs": 1, "private_required": True, "paid_compute": False,
        "new_dataset_upload": False, "same_scientific_payload_required": True,
        "registry_sha256": REGISTRY_SHA, "bundle_manifest_sha256": BUNDLE_SHA,
        "failure_archive_sha256": ARCHIVE_SHA, "required_execution_policy": POLICY}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("Runtime retry authorization differs")


def test_evidence(root: Path, path: Path, required_sources: set[str]) -> dict:
    value = read(path)
    if (value.get("status") != "passed" or value.get("tests_passed", value.get("tests", 0)) < 1
            or value.get("returncode", 0) != 0):
        raise ValueError("Generated runtime checks did not pass")
    sources = value.get("source_hashes", value.get("source_sha256", {}))
    normalized = {}
    for name, expected in sources.items():
        target = checked(root, name)
        require(target, expected)
        normalized[target.relative_to(root).as_posix()] = expected
    if not required_sources.issubset(normalized):
        raise ValueError("Generated check receipt omits required sources")
    return normalized


def render_entry(original: str, hook: str, amendment_text: str) -> str:
    """Install a byte-bound startup hook outside the verified scientific payload."""
    hook_hash = hashlib.sha256(hook.encode()).hexdigest()
    amendment_hash = hashlib.sha256(amendment_text.encode()).hexdigest()
    function = f'''

def install_deterministic_runtime(payload: Path, out: Path) -> None:
    hook_text = {hook!r}
    amendment_text = {amendment_text!r}
    if hashlib.sha256(hook_text.encode()).hexdigest() != {hook_hash!r}:
        raise ValueError("Deterministic startup source hash differs")
    if hashlib.sha256(amendment_text.encode()).hexdigest() != {amendment_hash!r}:
        raise ValueError("Runtime amendment hash differs")
    hook_dir = payload.parent / "deterministic_startup"
    evidence = out / "runtime_amendment"
    hook_dir.mkdir(exist_ok=False)
    evidence.mkdir(exist_ok=False)
    (evidence / "processes").mkdir()
    for path, content in ((hook_dir / "sitecustomize.py", hook_text),
                          (evidence / "sitecustomize.py", hook_text),
                          (evidence / "registration.json", amendment_text)):
        with path.open("x", encoding="utf-8", newline="\\n") as stream:
            stream.write(content)
    if digest(hook_dir / "sitecustomize.py") != {hook_hash!r} or digest(evidence / "registration.json") != {amendment_hash!r}:
        raise ValueError("Runtime startup write verification failed")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["FIXED_EPOCH_DETERMINISTIC_ROOT"] = str(payload.resolve())
    os.environ["FIXED_EPOCH_DETERMINISTIC_RECEIPTS"] = str((evidence / "processes").resolve())
    os.environ["FIXED_EPOCH_DETERMINISTIC_AMENDMENT_SHA256"] = {amendment_hash!r}
    os.environ["FIXED_EPOCH_DETERMINISTIC_HOOK_SHA256"] = {hook_hash!r}
    os.environ["PYTHONPATH"] = str(hook_dir.resolve()) + os.pathsep + str((payload / "scripts").resolve())

'''
    if original.count("\ndef bootstrap() -> None:") != 1 or original.count("        install_minimal(out, result)\n") != 1:
        raise ValueError("Reviewed entry insertion points differ")
    result = original.replace("\ndef bootstrap() -> None:", function + "\ndef bootstrap() -> None:")
    result = result.replace("        install_minimal(out, result)\n",
        "        install_minimal(out, result)\n        install_deterministic_runtime(payload, out)\n", 1)
    old_deadline = '(started + timedelta(seconds=7140)).isoformat()'
    old_cap = '"hard_timeout_seconds": 7200, "bundle_manifest_sha256": EXPECTED_BUNDLE_MANIFEST'
    if result.count(old_deadline) != 1 or result.count(old_cap) != 1:
        raise ValueError("Reviewed operational deadline fields differ")
    result = result.replace(old_deadline, '(started + timedelta(seconds=6840)).isoformat()', 1)
    result = result.replace(old_cap, '"hard_timeout_seconds": 6900, "registered_hard_timeout_seconds": 7200, "bundle_manifest_sha256": EXPECTED_BUNDLE_MANIFEST', 1)
    compile(result, "run.py", "exec")
    return result


def prepare(root: Path, hook_tests: Path, preparation_tests: Path) -> dict:
    root = root.resolve()
    folder = root / FOLDER
    if folder.exists():
        raise FileExistsError("Immutable runtime retry namespace already exists")
    require(root / AUTHORIZATION, AUTHORIZATION_SHA)
    authorization = read(root / AUTHORIZATION)
    validate_authorization(authorization)
    original = root / ORIGINAL
    preparation = read(original / "preparation_manifest.json")
    for name, expected in preparation["frozen_files"].items():
        require(checked(root, name), expected)
    require(original / "run.py", ENTRY_SHA)
    payload = original / "bundle/payload"
    require(payload / "registry.json", REGISTRY_SHA)
    require(payload / "bundle-manifest.json", BUNDLE_SHA)
    manifest = read(payload / "bundle-manifest.json")
    for name, item in manifest["files"].items():
        require(checked(payload, name), item["sha256"])
    failed_receipt = checked(root, authorization["failed_original_receipt"])
    require(failed_receipt, authorization["failed_original_receipt_sha256"])
    if (read(failed_receipt).get("status") != "retrieved_terminal_failure"
            or read(failed_receipt).get("archive_verified") is not True):
        raise ValueError("Prior cloud failure was not preserved and verified")
    failed_folder = original / "download_01/fixed_epoch_return"
    require(failed_folder / "results.zip", ARCHIVE_SHA)
    if read(failed_folder / "cloud_status.json").get("status") != "failed":
        raise ValueError("Prior cloud attempt is not failed")
    output = read(failed_folder / "output-manifest.json")
    if any(name.startswith(f"artifacts/{CAMPAIGN}/fold_") for name in output["files"]):
        raise ValueError("Prior attempt contains real-fit artifacts")
    hook_tests, preparation_tests = checked(root, str(hook_tests)), checked(root, str(preparation_tests))
    sources = test_evidence(root, hook_tests, {HOOK, "scripts/test_fixed_epoch_deterministic_execution_v1.py"})
    sources.update(test_evidence(root, preparation_tests, {"scripts/prepare_fixed_epoch_dropout_retry1.py", TEST, CONTROLLER}))
    sources[CONTROLLER] = sha(root / CONTROLLER)
    hook = (root / HOOK).read_bytes().decode("utf-8")
    compile(hook, "sitecustomize.py", "exec")
    if hashlib.sha256(hook.encode()).hexdigest() != sources[HOOK]:
        raise ValueError("Startup hook must use UTF-8 with unchanged newlines")
    amendment = {"id": ID, "campaign": CAMPAIGN, "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_path": AUTHORIZATION, "authorization_sha256": AUTHORIZATION_SHA,
        "original_entry_sha256": ENTRY_SHA, "scientific_registry_sha256": REGISTRY_SHA,
        "scientific_bundle_manifest_sha256": BUNDLE_SHA, "scientific_payload_unchanged": True,
        "dataset_id": DATASET, "kernel_id": KERNEL, "provider_timeout_seconds": 6900,
        "fit_budget_seconds": 6300, "delivery_deadline_seconds": 6840,
        "source_hashes": sources, "hook_source_sha256": sources[HOOK],
        "hook_tests_path": hook_tests.relative_to(root).as_posix(), "hook_tests_sha256": sha(hook_tests),
        "preparation_tests_path": preparation_tests.relative_to(root).as_posix(), "preparation_tests_sha256": sha(preparation_tests),
        "execution_policy": POLICY, "matmul_allow_tf32": False,
        "startup_failure_exit_code": 86, "startup_receipts": f"artifacts/{CAMPAIGN}/runtime_amendment/processes/process_<PID>.json",
        "scope": "Apply identical strict deterministic execution to every frozen smoke/controller/worker before CUDA initialization; preserve exact prefix gate and all scientific inputs",
        "evidence_limit": "Generated CPU startup checks; CUDA determinism must still pass the unchanged full remote smoke before real fitting"}
    amendment_text = encode(amendment)
    entry = render_entry((original / "run.py").read_text(encoding="utf-8"), hook, amendment_text)
    metadata = read(original / "kernel-metadata.json")
    metadata.update(id=KERNEL, title="S6E10 Dropout Deterministic Retry 20261004")
    if metadata["dataset_sources"] != [DATASET] or metadata["is_private"] is not True:
        raise ValueError("Parent private metadata differs")
    folder.mkdir(parents=True, exist_ok=False)
    exclusive(folder / "runtime_amendment.json", amendment_text)
    exclusive(folder / "run.py", entry)
    exclusive(folder / "kernel-metadata.json", encode(metadata))
    for name, expected in sources.items():
        source = checked(root, name)
        require(source, expected)
        destination = folder / "registered_source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
        require(destination, expected)
    files = {path.relative_to(root).as_posix(): sha(path) for path in folder.rglob("*") if path.is_file()}
    files.update(sources)
    for path in (root / AUTHORIZATION, hook_tests, preparation_tests, failed_receipt,
                 original / "preparation_manifest.json", original / "run.py", payload / "registry.json", payload / "bundle-manifest.json"):
        files[path.relative_to(root).as_posix()] = sha(path)
    result = {"id": ID, "status": "prepared_not_dispatched", "created_utc": datetime.now(timezone.utc).isoformat(),
        "kernel_id": KERNEL, "dataset_id": DATASET, "new_dataset_upload": False,
        "registry_sha256": REGISTRY_SHA, "manifest_sha256": BUNDLE_SHA,
        "runtime_amendment_sha256": sha(folder / "runtime_amendment.json"), "runtime_sha256": sha(folder / "run.py"),
        "hard_timeout_seconds": 6900, "fit_budget_seconds": 6300, "files": files,
        "real_fits": 0, "remote_operations": 0}
    exclusive(folder / "preparation_manifest.json", encode(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hook-tests", type=Path, required=True)
    parser.add_argument("--preparation-tests", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(ROOT, args.hook_tests, args.preparation_tests)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
