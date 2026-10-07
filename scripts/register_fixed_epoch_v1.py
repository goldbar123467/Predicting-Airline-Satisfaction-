"""Validate evidence and optionally freeze the isolated fixed A/B/C campaign.

Default is read-only. --freeze creates the registry, source snapshots and wrapper
exclusively. This module never imports training libraries, fits or scores data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
ID = "fixed_epoch_v1"
OUTPUT = "artifacts/fixed_epoch_v1"
WRAPPER = "configs/fixed_epoch_v1.json"
SPLIT = "4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c"
INCUMBENT = "bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311"
AUX = "0d3c98df25579e0ceb17cbc65ab43d20a1d0a4ab00a479f09019d76732df9c03"
PINS = {
    "data/splits.parquet": SPLIT,
    "data/train.parquet": "f95ca8b83420e2648fbae2bd2593948697747c873fcaaa38b42b4db7b2bef9c5",
    "data/manifest.json": "7dfbb4b0c1bc62df4d38fee45299c6b3184ae2dff47954ce6b56979574ba2fd4",
    "data/original_aux_predictions.parquet": AUX,
    "artifacts/original_aux/manifest.json": "4c0adfc0b94c5811d801cf058459159db3835d6bb8a0aa3a2029facc9c4d971c",
    "artifacts/runs/v2_realmlp_cat_raw_aux/contract.json": "0c8d9385e11bd4fb6643ece7bad5f3e8e8876662ff6897e98853f5aaf25c5e07",
    "artifacts/third_pass/blend/frozen.json": INCUMBENT,
    "requirements.lock.txt": "2f2b8b07b8ed1d4a444c97097032dc64c41b3c10a0f8812e3d92058b6e926f20",
}
BASELINE_PINS = {
    "artifacts/second_pass/blend/frozen.json": "d75c7eba29f8ed9cb3e9713e4f43b102a5d48dacdcfbba616f9c331a28bb0dc1",
    "artifacts/third_pass_batch07/final/release_provenance.json": "b3788575af52f6dc7e807d332c3226dd3b739c56dd7b3129da82db52e3be0690",
    "artifacts/third_pass_batch07/final/submission.csv": "3bcc2627880c1bfdf02f6e9f4304795c7cd41648db08d12d51e5cae7e9afe344",
    "artifacts/third_pass_batch07/final/manifest.json": "cff6966bfb09b9a0c71708521075ea96c71b8e3ff25efcfe07df5a58849e4619",
    "artifacts/third_pass_batch07/blend/frozen.json": "2d93bc0328833a5b1c2493e07adc8fd853b8551782f09ffcbff65661cd0afa86",
}
RECIPE = {
    "id": "v2_realmlp_cat_raw_aux", "family": "realmlp_cat", "teacher": False,
    "seed": 20261005, "max_rounds": 4, "timeout_seconds": 1200,
    "params": {"n_ens": 8, "hidden_sizes": [512, 256, 128], "batch_size": 256,
               "eval_batch_size": 2048, "learning_rate": .053, "weight_decay": .015,
               "threads": 4, "ls_eps": 0.0},
    "categorical_twins": True, "original_aux": True,
}
EVALUATION = {"mixture_alpha": .1, "min_pooled_gain": 1e-5, "min_macro_gain": 1e-5,
              "max_fold_regression": 2e-5, "class_order": [0, 1], "fold_ids": [0, 1, 2]}
SOURCE_NAMES = [
    "register_fixed_epoch_v1.py", "test_register_fixed_epoch_v1.py",
    "run_fixed_epoch_v1.py", "test_run_fixed_epoch_v1.py",
    "fixed_epoch_adapter_v1.py", "test_fixed_epoch_adapter_v1.py",
    "evaluate_fixed_epoch_v1.py", "test_evaluate_fixed_epoch_v1.py",
    "smoke_fixed_epoch_v1.py", "common.py", "categorical_transform.py",
    "realmlp.py", "realmlp_categorical.py", "supervisor.py", "long_local_500_policy.py",
]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def local(root: Path, name: str | Path) -> Path:
    path = (root / Path(str(name).replace("\\", "/"))).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes workspace")
    return path


def relative(root: Path, path: Path) -> str:
    return local(root, path).relative_to(root.resolve()).as_posix()


def check_hashes(root: Path, hashes: dict) -> dict:
    result = {}
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Empty hash manifest")
    for name, expected in hashes.items():
        path = local(root, name)
        key = relative(root, path)
        if key in result or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Duplicate or invalid hash identity")
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Pinned bytes changed: {key}")
        result[key] = actual
    return result


def validate_test_receipt(root: Path, path: Path, expected_sources: set[str], count: int) -> dict:
    receipt = read(path)
    if (receipt.get("status") != "passed" or receipt.get("tests_passed") != count
            or receipt.get("real_data_rows") != 0 or receipt.get("returncode", 0) != 0
            or receipt.get("source_unchanged", True) is not True):
        raise ValueError(f"Test receipt is not passing: {path}")
    sources = check_hashes(root, receipt.get("source_hashes", receipt.get("source_sha256")))
    if not expected_sources.issubset(sources):
        raise ValueError("Test receipt omits required implementation/test sources")
    evidence = {relative(root, path): digest(path)}
    for name, expected in receipt.get("logs_sha256", {}).items():
        log_path = local(root, path.parent / name)
        if log_path.parent != path.parent.resolve():
            raise ValueError("Test log must be in receipt directory")
        evidence.update(check_hashes(root, {relative(root, log_path): expected}))
    if "registry_sha256" in receipt:
        evidence.update(check_hashes(root, {relative(root, path.parent / "registry.json"): receipt["registry_sha256"]}))
    return {"receipt_path": relative(root, path), "receipt_sha256": digest(path),
            "tests_passed": count, "source_hashes": sources, "evidence_hashes": evidence}


def validate_parity(value: dict) -> None:
    if value.get("parity_passed") is not True or value.get("atol") != 2e-6 or value.get("rtol") != 1e-5:
        raise ValueError("Native parity scope/tolerance differs")
    for field, limit in (("max_absolute_error", 1.2e-5), ("max_scaled_error", 1.0)):
        v = value.get(field)
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= limit:
            raise ValueError("Native parity failed")


def validate_cuda_smoke(root: Path, smoke: Path) -> dict:
    smoke = local(root, smoke)
    if smoke.is_file():
        if smoke.name != "verification.json":
            raise ValueError("CUDA smoke must name its directory or verification.json")
        smoke = smoke.parent
    if not smoke.is_relative_to(local(root, OUTPUT)):
        raise ValueError("CUDA smoke is outside isolated campaign")
    path, registry_path = smoke / "verification.json", smoke / "registry.json"
    receipt, registry = read(path), read(registry_path)
    evidence = check_hashes(root, {relative(root, registry_path): receipt["registry_sha256"]})
    evidence[relative(root, path)] = digest(path)
    sources = check_hashes(root, registry["source_hashes"])
    required = {f"scripts/{name}" for name in ["smoke_fixed_epoch_v1.py", "fixed_epoch_adapter_v1.py",
                                              "run_fixed_epoch_v1.py", "common.py", "categorical_transform.py"]}
    if not required.issubset(sources):
        raise ValueError("CUDA registry omits required full-path sources")
    if (receipt.get("status") != "passed" or receipt.get("device") != "cuda"
            or receipt.get("architecture") != [512, 256, 128] or receipt.get("n_ens") != 8
            or receipt.get("native_jit_optimized_execution") is not False
            or receipt.get("actual_external_transform_exclusion") is not True
            or registry.get("device") != "cuda" or registry.get("real_data_rows") != 0
            or registry.get("recipe") != RECIPE or registry.get("generated_rows") != 512
            or registry.get("fit_rows") != 384 or registry.get("monitor_rows") != 128
            or receipt.get("feature_count") != 52 or receipt.get("categorical_count") != 22
            or not 0 < receipt.get("seconds", math.inf) <= registry.get("maximum_seconds", 0)):
        raise ValueError("CUDA smoke contract differs or failed")
    expected = {"inner_C": (16, [4, 16], 128), "outer_A": (4, [4], 0), "outer_C": (16, [4, 16], 0)}
    if set(receipt["records"]) != set(expected):
        raise ValueError("CUDA smoke trajectory inventory differs")
    for name, (horizon, epochs, monitor_rows) in expected.items():
        record = receipt["records"][name]
        trajectory_path = local(root, record["trajectory_path"])
        if trajectory_path != smoke / name / "trajectory.json":
            raise ValueError("CUDA trajectory path differs")
        evidence.update(check_hashes(root, {relative(root, trajectory_path): record["trajectory_sha256"]}))
        trajectory = read(trajectory_path)
        sources.update(check_hashes(root, trajectory["source_sha256"]))
        if (trajectory.get("status") != "complete" or trajectory.get("horizon") != horizon
                or trajectory.get("executed_epochs") != horizon or trajectory.get("endpoint_epochs") != epochs
                or trajectory.get("train_rows") != 384 or trajectory.get("monitor_rows") != monitor_rows
                or trajectory.get("clock") != "epoch_fraction" or trajectory.get("legacy_decay_semantics") is not True
                or trajectory.get("split_seed") != RECIPE["seed"]
                or trajectory.get("resolved_config", {}).get("use_best_epoch") is not False
                or trajectory.get("resolved_config", {}).get("use_early_stopping") is not False):
            raise ValueError("CUDA literal endpoint/monitor/selection contract differs")
        if [item["epoch"] for item in record["native_reload"]] != epochs:
            raise ValueError("CUDA endpoint verification incomplete")
        for item in record["native_reload"]:
            validate_parity(item)
            validate_parity(item["cpu_gpu_parity"])
        evidence.update(check_hashes(root, {relative(root, local(root, trajectory["curves_path"])): trajectory["curves_sha256"]}))
        transform = smoke / name / "transform.json"
        evidence[relative(root, transform)] = digest(transform)
        if set(trajectory["endpoints"]) != {str(epoch) for epoch in epochs}:
            raise ValueError("CUDA endpoint inventory differs")
        for epoch in epochs:
            endpoint = trajectory["endpoints"][str(epoch)]
            directory = smoke / name / f"epoch_{epoch:03d}"
            if local(root, endpoint["path"]) != directory or endpoint["epoch"] != epoch or endpoint["horizon"] != horizon:
                raise ValueError("CUDA endpoint context differs")
            evidence.update(check_hashes(root, {relative(root, directory / "metadata.json"): endpoint["metadata_sha256"],
                                                relative(root, directory / "graph.pt"): endpoint["graph_sha256"]}))
    for field in ("initial_network_sha256", "training_index_sha256", "split_seed", "sub_split_seed"):
        if receipt["records"]["outer_A"][field] != receipt["records"]["outer_C"][field]:
            raise ValueError("CUDA A/C initial state differs")
    return {"receipt_path": relative(root, path), "receipt_sha256": digest(path),
            "source_hashes": sources, "evidence_hashes": evidence}


def incumbent_bindings(root: Path) -> dict:
    """Verify metadata and file bytes only. Never decode any OOF values."""
    selection = read(root / "artifacts/third_pass/blend/frozen.json")
    weights = selection["weights"]
    if (selection["split_hash"] != SPLIT or len(weights) != 15
            or any(not math.isfinite(w) or w <= 0 for w in weights.values())
            or abs(sum(weights.values()) - 1) > 1e-12):
        raise ValueError("Frozen incumbent contract differs")
    evidence = {}
    for name in weights:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Invalid incumbent run name")
        directory = root / "artifacts/runs" / name
        evidence.update(check_hashes(root, {relative(root, directory / "result.json"): selection["source_result_hashes"][name]}))
        result = read(directory / "result.json")
        evidence.update(check_hashes(root, {relative(root, directory / "contract.json"): result["contract_hash"]}))
        contract = read(directory / "contract.json")
        if (result["id"] != name or result["split_hash"] != SPLIT or contract["split_hash"] != SPLIT
                or contract["run"] != result["run"]):
            raise ValueError("Incumbent result/contract identity differs")
        hashes = {relative(root, directory / "oof.parquet"): result["artifacts"]["oof.parquet"],
                  relative(root, directory / "source/requirements.lock.txt"): contract["environment_hash"]}
        for source, expected in contract["code_hashes"].items():
            path = local(root, directory / "source" / source)
            if not path.is_relative_to((directory / "source").resolve()):
                raise ValueError("Archived incumbent source escapes run directory")
            hashes[relative(root, path)] = expected
        evidence.update(check_hashes(root, hashes))
    return evidence


def gather(root: Path, cuda_smoke: Path, cpu_tests: Path, runner_tests: Path, evaluator_tests: Path) -> dict:
    inputs = check_hashes(root, {**PINS, **BASELINE_PINS})
    inputs.update(incumbent_bindings(root))
    contract = read(root / "artifacts/runs/v2_realmlp_cat_raw_aux/contract.json")
    recipes = read(root / "configs/second_pass.json")["runs"]
    if contract["run"] != RECIPE or [r for r in recipes if r["id"] == RECIPE["id"]] != [RECIPE]:
        raise ValueError("Existing recipe differs from registered controls")
    aux = read(root / "artifacts/original_aux/manifest.json")
    if (aux.get("status") != "complete" or aux.get("format") != "original_aux_bank"
            or aux.get("output_file") != "data/original_aux_predictions.parquet" or aux.get("output_sha256") != AUX
            or aux.get("synthetic_labels_read") is not False or aux.get("satisfaction_labels_read") is not False
            or aux.get("synthetic_smoke_configuration") is not False or aux.get("output_rows") != 999479):
        raise ValueError("Auxiliary manifest is incomplete or not the frozen label-free bank")
    evidence = {"cuda_smoke": validate_cuda_smoke(root, cuda_smoke)}
    for name, path, implementation, count in (
        ("cpu_adapter_tests", cpu_tests, "fixed_epoch_adapter_v1", 4),
        ("runner_tests", runner_tests, "run_fixed_epoch_v1", 6),
        ("evaluator_tests", evaluator_tests, "evaluate_fixed_epoch_v1", 19),
    ):
        evidence[name] = validate_test_receipt(root, local(root, path),
            {f"scripts/{implementation}.py", f"scripts/test_{implementation}.py"}, count)
    sources = {f"scripts/{name}": digest(root / "scripts" / name) for name in SOURCE_NAMES}
    package = root / ".venv/Lib/site-packages/pytabkit"
    if not (package / "__init__.py").is_file():
        raise ValueError("Expected installed PyTabKit package missing")
    for path in sorted(package.rglob("*.py")):
        sources[relative(root, path)] = digest(path)
    for entry in evidence.values():
        for name, expected in entry["source_hashes"].items():
            if sources.get(name) != expected:
                raise ValueError(f"Current source collection differs from passing evidence: {name}")
        inputs.update(entry["evidence_hashes"])
    for name in ("configs/second_pass.json", "FIXED_EPOCH_V1_PLAN.md", "research/fixed_epoch_v1_evaluation_contract.md"):
        inputs[name] = digest(root / name)
    versions = {name: importlib.metadata.version(name) for name in
                ("pytabkit", "torch", "pytorch-lightning", "numpy", "pandas", "scikit-learn", "pyarrow", "psutil", "threadpoolctl")}
    return {"source_hashes": dict(sorted(sources.items())), "input_hashes": dict(sorted(inputs.items())),
            "passing_evidence": evidence, "baseline_frozen_release_hashes": BASELINE_PINS,
            "versions": {"python": platform.python_version(), "platform": platform.platform(), **versions}}


def registry_payload(collected: dict, registered: datetime) -> dict:
    if registered.tzinfo is None or registered.utcoffset() != timedelta(0):
        raise ValueError("Registration timestamp must be UTC-aware")
    return {"id": ID, "schema_version": 1, "registered_utc": registered.isoformat(),
            "fit_deadline_utc": (registered + timedelta(minutes=90)).isoformat(),
            "delivery_deadline_utc": (registered + timedelta(minutes=120)).isoformat(),
            "expected_fit_count": 12, "expected_outer_endpoint_count": 9,
            "expected_total_epochs": 120, "development_rows": 629671,
            "recipe": RECIPE, "evaluation": EVALUATION,
            "native_jit_optimized_execution": False, "native_cpu_threads": 4,
            "class_order": [0, 1], "outer_folds": [0, 1, 2],
            "inner_monitor_fraction": .1, "inner_partition_seeds": [20261005, 20261006, 20261007],
            "model_seed_policy": "Original public-estimator seed mapping; same model seed 20261005 for every fit",
            "arms": {"A": {"horizon": 4, "endpoint": 4}, "B": {"horizon": 16, "endpoint": 4, "prefix_of": "C"},
                     "C": {"horizon": 16, "endpoint": 16}},
            "execution_order": [{"fold": f, "phase": p, "trajectory": t} for f in (0, 1, 2)
                                for p in ("inner", "outer") for t in ("A", "C")],
            "acceptance": "C mixture must exceed incumbent, A mixture and B mixture by >=1e-5 pooled AND macro AUC, with every fold delta >=-2e-5 against each; advance only to later confirmation",
            "prohibitions": ["audit/test prediction or scoring", "partial-fold quality scoring", "adaptive stopping or endpoint selection",
                             "weight search", "release promotion", "submission", "full-data refit", "cloud jobs"],
            "evidence_limit": "Historically reused development data; not independent or genuinely nested confirmation",
            **collected}


def exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def freeze(root: Path, collected: dict) -> dict:
    """No overwrites or automatic repair after partial registration failure."""
    output = local(root, OUTPUT)
    registry_path, wrapper_path = output / "registry.json", local(root, WRAPPER)
    snapshot_root = output / "registered_source"
    if registry_path.exists() or wrapper_path.exists() or snapshot_root.exists():
        raise FileExistsError("Campaign already registered or partial registration exists")
    if any((output / f"fold_{fold}").exists() for fold in (0, 1, 2)):
        raise FileExistsError("Cannot register after real trajectory artifacts exist")
    # Permanent exclusive claim makes a failure reviewable and forbids silent retry.
    claim = {"id": ID, "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
    exclusive_json(output / "registration_claim.json", claim)
    check_hashes(root, collected["source_hashes"])
    check_hashes(root, collected["input_hashes"])
    registered = datetime.now(timezone.utc)
    registry = registry_payload(collected, registered)
    snapshots = {}
    for name, expected in collected["source_hashes"].items():
        target = local(root, snapshot_root / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = local(root, name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Source changed during snapshot: {name}")
        with target.open("xb") as stream:
            stream.write(data)
        if digest(target) != expected:
            raise ValueError("Source snapshot verification failed")
        snapshots[relative(root, target)] = expected
    registry["source_snapshot_hashes"] = snapshots
    check_hashes(root, collected["source_hashes"])
    check_hashes(root, collected["input_hashes"])
    exclusive_json(registry_path, registry)
    wrapper = {"id": ID, "registry_path": relative(root, registry_path), "registry_sha256": digest(registry_path),
               "output_dir": OUTPUT, "split_path": "data/splits.parquet", "split_sha256": SPLIT,
               "incumbent_selection_path": "artifacts/third_pass/blend/frozen.json", "incumbent_selection_sha256": INCUMBENT,
               "completed_manifest_path": f"{OUTPUT}/completed_manifest.json", "completion_receipt_path": f"{OUTPUT}/completion_receipt.json"}
    exclusive_json(wrapper_path, wrapper)
    return {"status": "registered", "registry_sha256": digest(registry_path), "wrapper_sha256": digest(wrapper_path),
            "registered_utc": registry["registered_utc"], "fit_deadline_utc": registry["fit_deadline_utc"],
            "delivery_deadline_utc": registry["delivery_deadline_utc"], "source_count": len(snapshots)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-smoke", type=Path, required=True)
    parser.add_argument("--cpu-tests", type=Path, default=Path(f"{OUTPUT}/smoke_adapter/cpu_verification.json"))
    parser.add_argument("--runner-tests", type=Path, default=Path(f"{OUTPUT}/runner_tests/verification.json"))
    parser.add_argument("--evaluator-tests", type=Path, default=Path(f"{OUTPUT}/evaluator_tests/verification.json"))
    parser.add_argument("--freeze", action="store_true", help="Create immutable campaign files after all checks; default only validates")
    args = parser.parse_args()
    collected = gather(ROOT, args.cuda_smoke, args.cpu_tests, args.runner_tests, args.evaluator_tests)
    result = freeze(ROOT, collected) if args.freeze else {"status": "ready_not_registered",
        "source_count": len(collected["source_hashes"]), "input_count": len(collected["input_hashes"]),
        "passing_evidence": {k: {f: v[f] for f in ("receipt_path", "receipt_sha256")} for k, v in collected["passing_evidence"].items()}}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
