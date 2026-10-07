"""One-time local evaluation of a downloaded fixed A/B/C cloud campaign.

No training, audit/test scoring, alpha search, release mutation or submission.
Cloud-produced full raw/native reload receipts are checked locally. This
evaluator does not execute native inference or decode raw train/auxiliary data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "fixed_epoch_cloud_v1"
DEVELOPMENT_ROWS = 629671
LOCAL_PROTOCOL = ROOT / f"artifacts/{CAMPAIGN_ID}/local_evaluation_protocol.json"
PREPARATION = ROOT / "cloud/fixed_epoch_v1/preparation_manifest.json"
RUNTIME_MODULES = ("pytabkit.models.optim.optimizers", "pytabkit.models.training.lightning_modules",
                   "pytabkit.models.training.nn_creator")
ADAPTER_WORKSPACE_SOURCES = ("scripts/fixed_epoch_cloud_adapter_v1.py", "scripts/realmlp.py",
    "scripts/realmlp_categorical.py", "scripts/common.py", "scripts/categorical_transform.py")
SPLIT_HASH = "4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c"
INCUMBENT_HASH = "bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311"
EVALUATION = {"mixture_alpha": .1, "min_pooled_gain": 1e-5, "min_macro_gain": 1e-5,
              "max_fold_regression": 2e-5, "class_order": [0, 1], "fold_ids": [0, 1, 2]}
LIMITATION = (
    "Fixed prospective endpoint comparisons on historically reused development data. "
    "Not a genuinely nested or independent confirmation. Advancement is an engineering "
    "gate for a later confirmation study, not significance, release promotion or submission."
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def checked_path(root: Path, relative: str, *, inside: Path | None = None) -> Path:
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Expected a portable POSIX-relative receipt path")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Expected a workspace-relative path without parent traversal")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or (inside is not None and not path.is_relative_to(inside.resolve())):
        raise ValueError("Artifact path escapes its allowed directory")
    return path


def require_hash(path: Path, expected: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Invalid SHA256 identity for {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return actual


def integral_ids(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise ValueError("IDs must be a one-dimensional integer array")
    if values.dtype.kind == "u" and len(values) and values.max() > np.iinfo(np.int64).max:
        raise ValueError("IDs exceed int64 range")
    return values.astype(np.int64, copy=False)


def ids_sha256(ids: np.ndarray) -> str:
    return hashlib.sha256(np.sort(integral_ids(ids)).astype("<i8").tobytes()).hexdigest()


def development_split(split: pd.DataFrame, expected_rows: int | None = None) -> pd.DataFrame:
    if list(split.columns) != ["id", "fold"] or split.isna().any().any() or not split.id.is_unique:
        raise ValueError("Invalid authoritative split identity/schema")
    integral_ids(split.id.to_numpy())
    if split.fold.dtype.kind not in "iu":
        raise ValueError("Fold identities must be integral")
    # Exclude the audit before any labels or predictions are joined.
    selected = split.loc[split.fold >= 0, ["id", "fold"]].copy()
    if (set(selected.fold) != {0, 1, 2}
            or (expected_rows is not None and len(selected) != expected_rows)):
        raise ValueError("Unexpected development fold population")
    return selected


def aligned_oof(frame: pd.DataFrame, expected_ids: np.ndarray, expected_folds: np.ndarray,
                expected_labels: np.ndarray | None, *, class_order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Strict keyed join; includes no ranking or other metric operation."""
    if class_order != [0, 1]:
        raise ValueError("Required class order is [0,1], prediction is class 1")
    expected_ids = integral_ids(expected_ids)
    expected_folds = np.asarray(expected_folds)
    if (set(frame.columns) != {"id", "fold", "satisfaction", "prediction"}
            or len(frame.columns) != 4 or len(frame) != len(expected_ids)
            or not len(frame) or frame.isna().any().any() or not frame.id.is_unique
            or len(np.unique(expected_ids)) != len(expected_ids)
            or expected_folds.shape != expected_ids.shape or (expected_folds < 0).any()):
        raise ValueError("OOF schema, row count, uniqueness or development boundary violation")
    integral_ids(frame.id.to_numpy())
    if frame.fold.dtype.kind not in "iu" or (frame.fold < 0).any():
        raise ValueError("OOF must contain integral development folds only")
    order = pd.Index(frame.id).get_indexer(expected_ids)
    if (order < 0).any():
        raise ValueError("OOF has missing or unexpected IDs")
    joined = frame.iloc[order]
    if not np.array_equal(joined.fold.to_numpy(), expected_folds):
        raise ValueError("OOF fold mismatch after keyed join")
    labels = joined.satisfaction.to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("OOF target is not binary")
    labels = labels.astype(np.int8)
    if expected_labels is not None and not np.array_equal(labels, expected_labels):
        raise ValueError("OOF labels differ from verified development labels")
    for fold in np.unique(expected_folds):
        if len(np.unique(labels[expected_folds == fold])) != 2:
            raise ValueError("Every scored fold must contain both classes")
    if joined.prediction.dtype.kind not in "fiu":
        raise ValueError("Prediction must be numeric probability")
    probability = joined.prediction.to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid finite [0,1] positive-class probability")
    return probability, labels


def validate_manifest_inventory(manifest: dict) -> tuple[list[dict], list[dict]]:
    trajectories, endpoints = manifest["trajectories"], manifest["endpoints"]
    expected_trajectories = {(phase, fold, trajectory) for phase in ("inner", "outer")
                             for fold in (0, 1, 2) for trajectory in ("A", "C")}
    expected_endpoints = {(fold, arm) for fold in (0, 1, 2) for arm in ("A", "B", "C")}
    actual_trajectories = [(row["phase"], row["fold"], row["trajectory"]) for row in trajectories]
    actual_endpoints = [(row["fold"], row["arm"]) for row in endpoints]
    if any(type(row["fold"]) is not int for row in [*trajectories, *endpoints]):
        raise ValueError("Manifest folds must be JSON integers")
    if len(actual_trajectories) != 12 or set(actual_trajectories) != expected_trajectories:
        raise ValueError("Exactly all twelve completed trajectories are required")
    if len(actual_endpoints) != 9 or set(actual_endpoints) != expected_endpoints:
        raise ValueError("Exactly all nine verified outer endpoints are required")
    return trajectories, endpoints


def validate_partition(partition: dict[str, np.ndarray], ids: np.ndarray,
                       folds: np.ndarray, fold: int, phase: str) -> dict:
    if set(partition) != {"training_ids", "monitor_ids", "outer_validation_ids"}:
        raise ValueError("Unexpected saved partition identity fields")
    arrays = {name: integral_ids(value) for name, value in partition.items()}
    if any(len(np.unique(value)) != len(value) for value in arrays.values()):
        raise ValueError("Duplicate partition IDs")
    training, monitor, held = (arrays[name] for name in ("training_ids", "monitor_ids", "outer_validation_ids"))
    expected_train, expected_held = ids[folds != fold], ids[folds == fold]
    if not np.array_equal(np.sort(held), np.sort(expected_held)):
        raise ValueError("Outer validation partition IDs differ")
    if (np.intersect1d(training, monitor).size or np.intersect1d(training, held).size
            or np.intersect1d(monitor, held).size
            or not np.array_equal(np.sort(np.concatenate([training, monitor])), np.sort(expected_train))):
        raise ValueError("Training/monitor partition leaks, omits or adds rows")
    expected_monitor_count = 0 if phase == "outer" else int(np.ceil(.1 * len(expected_train)))
    if phase not in {"inner", "outer"} or len(monitor) != expected_monitor_count:
        raise ValueError("Unexpected monitor partition size")
    return {"fold": fold, "phase": phase, "training_rows": len(training), "monitor_rows": len(monitor),
            "training_ids_sha256": ids_sha256(training), "monitor_ids_sha256": ids_sha256(monitor),
            "outer_validation_ids_sha256": ids_sha256(held),
            "scope": "Exact outer membership, train/monitor disjointness and registered sizes; seed/order enforced by pinned runner and adapter"}


def validate_native_receipt(receipt: dict, endpoint: dict, expected_ids: np.ndarray,
                            registry_hash: str, root: Path, output: Path) -> dict:
    fold, arm = endpoint["fold"], endpoint["arm"]
    trajectory, epoch = ("A", 4) if arm == "A" else ("C", 4 if arm == "B" else 16)
    model_dir = output / f"fold_{fold}/outer/{trajectory}/epoch_{epoch:03d}"
    required = {
        "id": CAMPAIGN_ID, "status": "passed", "fold": fold, "arm": arm,
        "registry_sha256": registry_hash, "verification_scope": "full_outer_fold",
        "class_order": [0, 1], "row_count": len(expected_ids), "ids_sha256": ids_sha256(expected_ids),
        "prediction_path": endpoint["prediction_path"], "prediction_sha256": endpoint["prediction_sha256"],
        "reference_kind": "first_native_endpoint_reload", "adapter_parity_scope": "fixed_probes_at_capture",
        "atol": 2e-6, "rtol": 1e-5, "parity_passed": True, "raw_reload_verified": True,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Native receipt identity, scope, class or verification contract differs")
    if checked_path(root, receipt["model_directory"], inside=output) != model_dir.resolve():
        raise ValueError("Native receipt points to wrong endpoint directory")
    for field in ("max_absolute_error", "max_scaled_error"):
        value = receipt[field]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("Invalid native parity error")
    if receipt["max_scaled_error"] > 1 or receipt["max_absolute_error"] > 2e-6 + 1e-5:
        raise ValueError("Native parity exceeds fixed tolerance")
    hashes = receipt["artifact_hashes"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing native artifact identities")
    paths = []
    for relative, expected in hashes.items():
        path = checked_path(root, relative, inside=output)
        require_hash(path, expected)
        paths.append(path)
    metadata_path = model_dir / "metadata.json"
    if ((output / f"fold_{fold}/outer/{trajectory}/transform.json").resolve() not in paths
            or metadata_path.resolve() not in paths):
        raise ValueError("Native verification must bind endpoint metadata and fitted transformer")
    metadata = read_json(metadata_path)
    graph_name = metadata["graph_file"]
    if not isinstance(graph_name, str) or Path(graph_name).name != graph_name:
        raise ValueError("Native metadata graph filename must be local to endpoint")
    graph_path = model_dir / graph_name
    if graph_path.resolve() not in paths or graph_path.suffix != ".pt":
        raise ValueError("Receipt must hash the exact graph consumed by native loader")
    require_hash(graph_path, metadata["graph_sha256"])
    expected_horizon = 4 if arm == "A" else 16
    if (metadata.get("schema_version") != 1 or metadata.get("model_type") != "realmlp_categorical"
            or metadata.get("classes") != [0, 1] or metadata.get("executed_epochs") != epoch
            or metadata.get("schedule_horizon_epochs") != expected_horizon
            or not isinstance(metadata.get("input_schema"), dict)
            or metadata["input_schema"].get("n_features") != metadata.get("n_features")
            or not isinstance(metadata.get("n_features"), int) or metadata["n_features"] < 1):
        raise ValueError("Native metadata class/schema or fixed endpoint differs")
    return {"fold": fold, "arm": arm, "row_count": len(expected_ids),
            "receipt_verification": "Cloud worker performed full raw/native reload; downloaded receipt and artifact identities checked locally; evaluator did not re-execute inference",
            "reference_kind": receipt["reference_kind"], "adapter_parity_scope": receipt["adapter_parity_scope"],
            "max_absolute_error": receipt["max_absolute_error"], "max_scaled_error": receipt["max_scaled_error"],
            "artifact_hashes": hashes}


def load_incumbent(selection: dict, ids: np.ndarray, folds: np.ndarray, root: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Reconstruct only accepted development OOF; no metrics are called here."""
    weights = selection["weights"]
    if (len(weights) != 15 or any(not np.isfinite(w) or w <= 0 for w in weights.values())
            or abs(sum(weights.values()) - 1) > 1e-12 or selection["split_hash"] != SPLIT_HASH):
        raise ValueError("Unexpected frozen incumbent selection")
    y, score, evidence = None, np.zeros(len(ids), dtype=np.float64), []
    for name, weight in weights.items():
        directory = checked_path(root, f"artifacts/runs/{name}", inside=root / "artifacts/runs")
        result_path = directory / "result.json"
        result_hash = require_hash(result_path, selection["source_result_hashes"][name])
        result = read_json(result_path)
        contract_hash = require_hash(directory / "contract.json", result["contract_hash"])
        contract = read_json(directory / "contract.json")
        if (result["id"] != name or result["split_hash"] != SPLIT_HASH
                or contract["split_hash"] != SPLIT_HASH or contract["run"] != result["run"]):
            raise ValueError("Incumbent run identity changed")
        for relative, expected in contract["code_hashes"].items():
            source = checked_path(directory / "source", relative)
            require_hash(source, expected)
        require_hash(directory / "source/requirements.lock.txt", contract["environment_hash"])
        oof_path = directory / "oof.parquet"
        oof_hash = require_hash(oof_path, result["artifacts"]["oof.parquet"])
        frame = pd.read_parquet(oof_path, columns=["id", "fold", "satisfaction", "prediction"])
        probability, y = aligned_oof(frame, ids, folds, y, class_order=[0, 1])
        score += float(weight) * probability
        evidence.append({"id": name, "weight": weight, "result_sha256": result_hash,
                         "contract_sha256": contract_hash, "oof_sha256": oof_hash})
    if y is None:
        raise ValueError("Missing incumbent labels")
    return y, score, evidence


def cloud_development_split(cloud: pd.DataFrame, canonical: pd.DataFrame,
                            expected_rows: int) -> pd.DataFrame:
    """Prove exact ID/fold membership; canonical audit keys are filtered first."""
    canonical_dev = development_split(canonical, expected_rows)
    cloud_dev = development_split(cloud, expected_rows)
    if len(cloud_dev) != len(cloud) or (cloud.fold < 0).any():
        raise ValueError("Cloud split must contain development rows only")
    order = pd.Index(cloud_dev.id).get_indexer(canonical_dev.id)
    if ((order < 0).any() or not np.array_equal(cloud_dev.iloc[order].id.to_numpy(), canonical_dev.id.to_numpy())
            or not np.array_equal(cloud_dev.iloc[order].fold.to_numpy(), canonical_dev.fold.to_numpy())):
        raise ValueError("Cloud split IDs/folds differ from canonical development split")
    return canonical_dev


def validate_runtime_sources(adapter: dict, root: Path, registry: dict, protocol: dict) -> dict:
    sources = adapter["source_sha256"]
    runtime = adapter["installed_source_provenance"]
    expected_paths = {module: f"artifacts/{CAMPAIGN_ID}/runtime_source/{module.replace('.', '/')}.py"
                      for module in RUNTIME_MODULES}
    if (len(runtime) != 3 or {item["module"] for item in runtime} != set(RUNTIME_MODULES)
            or set(sources) != set(ADAPTER_WORKSPACE_SOURCES) | set(expected_paths.values())):
        raise ValueError("Runtime source inventory differs from the portable adapter contract")
    for name in ADAPTER_WORKSPACE_SOURCES:
        if sources[name] != registry["source_hashes"][name]:
            raise ValueError("Adapter did not bind the registered payload source")
    for item in runtime:
        module = item["module"]
        path = expected_paths[module]
        expected = protocol["runtime_source_sha256"][module]
        if (item["recorded_source_path"] != path or item["sha256"] != expected
                or sources[path] != expected or item["distribution"] != "pytabkit" or item["version"] != "1.7.3"
                or not isinstance(item.get("installed_source_path"), str) or not item["installed_source_path"]):
            raise ValueError("Actual imported runtime source differs from the pinned PyTabKit contract")
    for source, expected in sources.items():
        require_hash(checked_path(root, source), expected)
    if adapter["versions"].get("pytabkit") != "1.7.3":
        raise ValueError("Unexpected installed PyTabKit version")
    return {"source_sha256": sources, "versions": adapter["versions"]}


def validate_return_manifest(root: Path) -> dict:
    path = root / "output-manifest.json"
    manifest = read_json(path)
    if manifest.get("id") != CAMPAIGN_ID or not isinstance(manifest.get("files"), dict):
        raise ValueError("Missing downloaded output inventory")
    required = {"registry.json", f"configs/{CAMPAIGN_ID}.json",
                f"artifacts/{CAMPAIGN_ID}/completion_receipt.json",
                f"artifacts/{CAMPAIGN_ID}/completed_manifest.json", f"state/{CAMPAIGN_ID}/run_state.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Downloaded output inventory omits required completion files")
    actual = {"registry.json", f"configs/{CAMPAIGN_ID}.json"}
    for part in ("artifacts", "state", "logs"):
        folder = root / part / CAMPAIGN_ID
        actual.update(path.relative_to(root).as_posix() for path in folder.rglob("*") if path.is_file())
    if actual != set(manifest["files"]):
        raise ValueError("Downloaded output tree has missing or unlisted files")
    for name, item in manifest["files"].items():
        if name not in {"registry.json", f"configs/{CAMPAIGN_ID}.json"} and not any(
                name.startswith(f"{part}/{CAMPAIGN_ID}/") for part in ("artifacts", "state", "logs")):
            raise ValueError("Downloaded output member is outside the campaign allowlist")
        source = checked_path(root, name)
        require_hash(source, item["sha256"])
        if type(item["bytes"]) is not int or item["bytes"] != source.stat().st_size:
            raise ValueError("Downloaded output byte count differs")
    return {"manifest_sha256": sha256(path), "file_count": len(manifest["files"])}


def validate_local_protocol(path: Path, canonical_root: Path) -> dict:
    protocol = read_json(path)
    required = {"id": CAMPAIGN_ID, "status": "frozen_before_assessment", "evaluation": EVALUATION,
                "source_split_sha256": SPLIT_HASH, "incumbent_selection_sha256": INCUMBENT_HASH,
                "quality_metrics_read_when_frozen": False, "real_predictions_scored_when_frozen": False}
    if any(protocol.get(key) != expected for key, expected in required.items()):
        raise ValueError("Local assessment protocol is not frozen with the original scientific gate")
    expected_sources = {"scripts/evaluate_fixed_epoch_cloud_v1.py", "scripts/test_evaluate_fixed_epoch_cloud_v1.py",
                        "scripts/evaluate_fixed_epoch_v1.py", "scripts/test_evaluate_fixed_epoch_v1.py"}
    if set(protocol["source_sha256"]) != expected_sources or set(protocol["runtime_source_sha256"]) != set(RUNTIME_MODULES):
        raise ValueError("Local protocol source inventory differs")
    for name, expected in protocol["source_sha256"].items():
        require_hash(checked_path(canonical_root, name), expected)
    return protocol


def prepare_campaign(campaign_path: Path, *, root: Path, canonical_root: Path = ROOT,
                     protocol_path: Path | None = None, preparation_path: Path | None = None) -> dict:
    """Perform all completion, provenance and row checks before any metric call."""
    root, canonical_root = root.resolve(), canonical_root.resolve()
    campaign_path = campaign_path.resolve()
    if not campaign_path.is_relative_to(root.resolve()):
        raise ValueError("Campaign wrapper must be in workspace")
    campaign = read_json(campaign_path)
    if (campaign["id"] != CAMPAIGN_ID or campaign.get("source_split_sha256") != SPLIT_HASH
            or campaign["incumbent_selection_sha256"] != INCUMBENT_HASH):
        raise ValueError("Campaign must preserve fixed split and incumbent")
    output = checked_path(root, campaign["output_dir"])
    if output != (root / "artifacts" / CAMPAIGN_ID).resolve():
        raise ValueError("Only isolated cloud campaign output is allowed")
    if (output / "evaluation.json").exists() or (output / "evaluation_claim.json").exists():
        raise FileExistsError("Evaluation has already been claimed or completed")
    protocol_path = protocol_path or canonical_root / f"artifacts/{CAMPAIGN_ID}/local_evaluation_protocol.json"
    protocol = validate_local_protocol(protocol_path, canonical_root)
    preparation_path = preparation_path or canonical_root / "cloud/fixed_epoch_v1/preparation_manifest.json"
    require_hash(preparation_path, protocol["preparation_sha256"])
    preparation = read_json(preparation_path)
    registry_path = checked_path(root, campaign["registry_path"])
    if registry_path != root / "registry.json":
        raise ValueError("Cloud wrapper must preserve payload-root registry path")
    registry_hash = require_hash(registry_path, campaign["registry_sha256"])
    registry = read_json(registry_path)
    if (registry["id"] != CAMPAIGN_ID or registry["expected_fit_count"] != 12
            or registry["evaluation"] != EVALUATION or registry["development_rows"] != DEVELOPMENT_ROWS
            or registry["source_split_sha256"] != SPLIT_HASH or registry["incumbent_selection_sha256"] != INCUMBENT_HASH
            or registry.get("audit_rows_exported") != 0 or registry.get("test_rows_exported") != 0
            or registry.get("cloud_quality_scoring") is not False):
        raise ValueError("Registered fixed evaluation rule differs")
    sources = registry["source_hashes"]
    if not set(ADAPTER_WORKSPACE_SOURCES).issubset(sources) or "scripts/run_fixed_epoch_cloud_v1.py" not in sources:
        raise ValueError("Registry must bind cloud adapter, feature and worker sources")
    if (preparation["id"] != CAMPAIGN_ID or preparation["registry_sha256"] != registry_hash
            or protocol["registry_sha256"] != registry_hash
            or protocol["bundle_manifest_sha256"] != preparation["manifest_sha256"]):
        raise ValueError("Downloaded registry differs from locally prepared cloud package")
    manifest_bundle_path = root / "bundle-manifest.json"
    require_hash(manifest_bundle_path, preparation["manifest_sha256"])
    bundle = read_json(manifest_bundle_path)
    if bundle["id"] != CAMPAIGN_ID or bundle["registry_sha256"] != registry_hash:
        raise ValueError("Bundle manifest identity differs")
    payload_inventory = set(sources) | set(registry["input_hashes"]) | {"registry.json", f"configs/{CAMPAIGN_ID}.json"}
    if set(bundle["files"]) != payload_inventory:
        raise ValueError("Payload inventory differs from registered source/input allowlist")
    for relative, item in bundle["files"].items():
        path = checked_path(root, relative)
        require_hash(path, item["sha256"])
        if item["bytes"] != path.stat().st_size:
            raise ValueError("Payload file byte count differs")
    for hashes in (sources, registry["input_hashes"]):
        for relative, expected in hashes.items():
            require_hash(checked_path(root, relative), expected)
    for required in ("data/train.parquet", "data/original_aux_predictions.parquet", "data/splits.parquet", "data/manifest.json"):
        if required not in registry["input_hashes"]:
            raise ValueError("Missing cloud input hash")
    return_inventory = validate_return_manifest(root)
    wrapper_hash = sha256(campaign_path)
    completion_path = checked_path(root, campaign["completion_receipt_path"], inside=output)
    completion = read_json(completion_path)
    bindings = {"id": CAMPAIGN_ID, "campaign_sha256": wrapper_hash, "registry_sha256": registry_hash}
    complete_fields = {**bindings, "status": "completed", "completed_fit_count": 12,
                       "completed_endpoint_count": 9, "evaluation_ready": True}
    if any(completion.get(key) != value for key, value in complete_fields.items()):
        raise ValueError("Campaign is not fully completed and ready for one-time evaluation")
    state_path = root / f"state/{CAMPAIGN_ID}/run_state.json"
    state = read_json(state_path)
    if any(state.get(key) != value for key, value in {**bindings, "status": "training_complete",
            "completed_fits": 12, "active_child": None, "outer_metrics_computed": False}.items()):
        raise ValueError("Cloud controller is not completely finished without quality scoring")
    manifest_path = checked_path(root, campaign["completed_manifest_path"], inside=output)
    manifest_hash = require_hash(manifest_path, completion["completed_manifest_sha256"])
    manifest = read_json(manifest_path)
    if any(manifest.get(key) != value for key, value in bindings.items()):
        raise ValueError("Completed manifest is not bound to this campaign and registry")
    trajectories, endpoints = validate_manifest_inventory(manifest)
    trajectory_evidence = []
    runtime_evidence = None
    for row in trajectories:
        receipt_path = checked_path(root, row["receipt_path"], inside=output)
        require_hash(receipt_path, row["receipt_sha256"])
        receipt = read_json(receipt_path)
        horizon = 4 if row["trajectory"] == "A" else 16
        required = {"id": CAMPAIGN_ID, "status": "completed", "phase": row["phase"],
                    "fold": row["fold"], "trajectory": row["trajectory"], "registry_sha256": registry_hash,
                    "schedule_horizon_epochs": horizon, "executed_epochs": horizon,
                    "checkpoint_epochs": [4] if horizon == 4 else [4, 16]}
        if any(receipt.get(key) != value for key, value in required.items()):
            raise ValueError("Incomplete or mismatched trajectory receipt")
        telemetry = checked_path(root, receipt["telemetry_path"], inside=output)
        require_hash(telemetry, receipt["telemetry_sha256"])
        partition_path = checked_path(root, receipt["partition_path"], inside=output)
        require_hash(partition_path, receipt["partition_sha256"])
        adapter_path = checked_path(root, receipt["adapter_receipt_path"], inside=output)
        require_hash(adapter_path, receipt["adapter_receipt_sha256"])
        trajectory_dir = output / f"fold_{row['fold']}/{row['phase']}/{row['trajectory']}"
        if adapter_path != (trajectory_dir / "trajectory.json").resolve() or partition_path != (trajectory_dir / "partitions.npz").resolve():
            raise ValueError("Adapter or partition receipt points to wrong trajectory")
        adapter = read_json(adapter_path)
        if (adapter.get("status") != "complete" or adapter.get("horizon") != horizon
                or adapter.get("executed_epochs") != horizon
                or adapter.get("endpoint_epochs") != required["checkpoint_epochs"]
                or adapter.get("clock") != "epoch_fraction" or adapter.get("legacy_decay_semantics") is not True
                or adapter.get("curves_sha256") != receipt["telemetry_sha256"]
                or adapter.get("artifact_path_base") != "runtime_workspace"
                or set(adapter["endpoints"]) != {str(epoch) for epoch in required["checkpoint_epochs"]}):
            raise ValueError("Adapter receipt does not prove the registered completed trajectory")
        for key, expected in {"campaign": CAMPAIGN_ID, "fold": row["fold"], "phase": row["phase"],
                              "trajectory": row["trajectory"], "partition_sha256": receipt["partition_sha256"]}.items():
            if adapter["context"].get(key) != expected:
                raise ValueError("Adapter context differs from trajectory identity")
        for epoch, endpoint in adapter["endpoints"].items():
            endpoint_dir = trajectory_dir / f"epoch_{int(epoch):03d}"
            if checked_path(root, endpoint["path"], inside=output) != endpoint_dir.resolve():
                raise ValueError("Adapter endpoint path differs")
            require_hash(endpoint_dir / "metadata.json", endpoint["metadata_sha256"])
            require_hash(endpoint_dir / "graph.pt", endpoint["graph_sha256"])
        if (checked_path(root, adapter["curves_path"], inside=output) != telemetry
                or checked_path(root, adapter["trajectory_path"], inside=output) != adapter_path):
            raise ValueError("Portable adapter receipt paths differ")
        environment = validate_runtime_sources(adapter, root, registry, protocol)
        if runtime_evidence is None:
            runtime_evidence = environment
        elif runtime_evidence != environment:
            raise ValueError("Cloud trajectories used different source or package versions")
        trajectory_evidence.append({**row, "telemetry_sha256": receipt["telemetry_sha256"],
                                    "partition_path": receipt["partition_path"], "partition_sha256": receipt["partition_sha256"],
                                    "adapter_receipt_sha256": receipt["adapter_receipt_sha256"]})
    # Verify all nine file receipts before opening any prediction table.
    for endpoint in endpoints:
        path = checked_path(root, endpoint["prediction_path"], inside=output)
        expected_path = output / f"fold_{endpoint['fold']}/outer/predictions_{endpoint['arm']}.parquet"
        if path != expected_path.resolve():
            raise ValueError("Unexpected outer endpoint prediction path")
        require_hash(path, endpoint["prediction_sha256"])
        native_path = checked_path(root, endpoint["native_receipt_path"], inside=output)
        require_hash(native_path, endpoint["native_receipt_sha256"])
    split_path = checked_path(root, campaign["split_path"])
    if split_path != root / "data/splits.parquet":
        raise ValueError("Cloud split path differs from payload contract")
    require_hash(split_path, campaign["split_sha256"])
    canonical_split = canonical_root / "data/splits.parquet"
    require_hash(canonical_split, SPLIT_HASH)
    split = cloud_development_split(pd.read_parquet(split_path, columns=["id", "fold"]),
        pd.read_parquet(canonical_split, columns=["id", "fold"]), DEVELOPMENT_ROWS)
    ids, folds = split.id.to_numpy(), split.fold.to_numpy()
    partition_evidence = []
    for row in trajectory_evidence:
        with np.load(checked_path(root, row["partition_path"], inside=output), allow_pickle=False) as partition:
            checked = validate_partition(dict(partition), ids, folds, row["fold"], row["phase"])
        partition_evidence.append({"trajectory": row["trajectory"], **checked})
    selection_path = canonical_root / "artifacts/third_pass/blend/frozen.json"
    require_hash(selection_path, INCUMBENT_HASH)
    selection = read_json(selection_path)
    y, incumbent, incumbent_evidence = load_incumbent(selection, ids, folds, canonical_root)
    arms = {arm: np.full(len(ids), np.nan) for arm in ("A", "B", "C")}
    native_evidence = []
    for endpoint in endpoints:
        mask = folds == endpoint["fold"]
        native = read_json(checked_path(root, endpoint["native_receipt_path"], inside=output))
        native_evidence.append(validate_native_receipt(native, endpoint, ids[mask], registry_hash, root, output))
        frame = pd.read_parquet(checked_path(root, endpoint["prediction_path"], inside=output))
        arms[endpoint["arm"]][mask], _ = aligned_oof(frame, ids[mask], folds[mask], y[mask], class_order=native["class_order"])
    if any(not np.isfinite(score).all() for score in arms.values()):
        raise ValueError("Incomplete assembled outer OOF vector")
    return {"output": output, "y": y, "folds": folds, "incumbent": incumbent, "arms": arms,
            "selection": selection, "campaign_sha256": wrapper_hash, "registry_sha256": registry_hash,
            "manifest_sha256": manifest_hash, "completion_sha256": sha256(completion_path),
            "source_hashes": sources, "ids_sha256": ids_sha256(ids),
            "trajectory_evidence": trajectory_evidence, "native_evidence": native_evidence,
            "partition_evidence": partition_evidence,
            "incumbent_evidence": incumbent_evidence, "endpoint_manifest": endpoints,
            "local_protocol_sha256": sha256(protocol_path), "preparation_sha256": sha256(preparation_path),
            "return_inventory": return_inventory, "runtime_evidence": runtime_evidence,
            "canonical_split_sha256": SPLIT_HASH, "cloud_split_sha256": campaign["split_sha256"],
            "canonical_development_rows_exactly_matched": True, "raw_cloud_train_aux_decoded": False,
            "assessment_workspace": str(root), "canonical_project_root": str(canonical_root)}


def metrics(y: np.ndarray, score: np.ndarray, folds: np.ndarray) -> dict:
    values = [float(roc_auc_score(y[folds == fold], score[folds == fold])) for fold in (0, 1, 2)]
    return {"pooled_auc": float(roc_auc_score(y, score)), "fold_auc": values,
            "mean_fold_auc": float(np.mean(values)), "log_loss": float(log_loss(y, score, labels=[0, 1])),
            "brier": float(brier_score_loss(y, score))}


def contrast(left: dict, right: dict) -> dict:
    return {"pooled_auc_delta": left["pooled_auc"] - right["pooled_auc"],
            "mean_fold_auc_delta": left["mean_fold_auc"] - right["mean_fold_auc"],
            "fold_auc_delta": (np.asarray(left["fold_auc"]) - right["fold_auc"]).tolist()}


def advancement_gate(summaries: dict[str, dict]) -> dict:
    """Require every registered C-mixture comparison, not a chosen subset."""
    comparisons = []
    for name in ("incumbent", "mixture_A", "mixture_B"):
        difference = contrast(summaries["mixture_C"], summaries[name])
        numbers = [difference["pooled_auc_delta"], difference["mean_fold_auc_delta"], *difference["fold_auc_delta"]]
        if len(difference["fold_auc_delta"]) != 3 or not np.isfinite(numbers).all():
            raise ValueError("Gate needs finite metrics for all three original folds")
        checks = {"pooled_gain_pass": difference["pooled_auc_delta"] >= EVALUATION["min_pooled_gain"],
                  "macro_gain_pass": difference["mean_fold_auc_delta"] >= EVALUATION["min_macro_gain"],
                  "fold_regression_pass": min(difference["fold_auc_delta"]) >= -EVALUATION["max_fold_regression"]}
        comparisons.append({"reference": name, "candidate": "mixture_C", **difference,
                            **checks, "passed": all(checks.values())})
    return {"advance_to_confirmation": all(row["passed"] for row in comparisons),
            "comparisons": comparisons, "rule": EVALUATION, "interpretation": LIMITATION,
            "release_promoted": False, "submission_authorized": False}


def score_fixed(y: np.ndarray, folds: np.ndarray, incumbent: np.ndarray,
                arms: dict[str, np.ndarray]) -> dict:
    if set(arms) != {"A", "B", "C"}:
        raise ValueError("Exactly fixed arms A/B/C are required")
    y, folds, incumbent = np.asarray(y), np.asarray(folds), np.asarray(incumbent)
    if y.ndim != 1 or folds.shape != y.shape or incumbent.shape != y.shape or set(folds) != {0, 1, 2}:
        raise ValueError("Score arrays must align to the three development folds")
    vectors = {"incumbent": incumbent, **arms}
    if any(np.asarray(value).shape != y.shape or not np.isfinite(value).all()
           or ((np.asarray(value) < 0) | (np.asarray(value) > 1)).any() for value in vectors.values()):
        raise ValueError("Aligned finite probability vectors required")
    if not np.isin(y, [0, 1]).all() or any(len(np.unique(y[folds == fold])) != 2 for fold in (0, 1, 2)):
        raise ValueError("Each development fold must contain both binary classes")
    for arm in ("A", "B", "C"):
        vectors[f"mixture_{arm}"] = .9 * incumbent + .1 * arms[arm]
    summaries = {name: metrics(y, np.asarray(value), folds) for name, value in vectors.items()}
    comparisons = {}
    for prefix in ("", "mixture_"):
        for left, right in (("B", "A"), ("C", "B"), ("C", "A")):
            comparisons[f"{prefix}{left}_minus_{prefix}{right}"] = contrast(summaries[prefix + left], summaries[prefix + right])
    for arm in ("A", "B", "C"):
        comparisons[f"mixture_{arm}_minus_incumbent"] = contrast(summaries[f"mixture_{arm}"], summaries["incumbent"])
    return {"metrics": summaries, "contrasts": comparisons, "advancement": advancement_gate(summaries),
            "mixture_alpha": .1, "weight_search_performed": False, "interpretation": LIMITATION}


def claim_evaluation(prepared: dict, invocation: list[str]) -> dict:
    """Create the exclusive durable scoring claim only after preparation passes."""
    claim = {"claimed_utc": datetime.now(timezone.utc).isoformat(),
             "campaign_sha256": prepared["campaign_sha256"], "registry_sha256": prepared["registry_sha256"],
             "manifest_sha256": prepared["manifest_sha256"], "local_protocol_sha256": prepared["local_protocol_sha256"],
             "invocation": invocation}
    with (prepared["output"] / "evaluation_claim.json").open("x", encoding="utf-8") as stream:
        json.dump(claim, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return claim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--campaign", default=f"configs/{CAMPAIGN_ID}.json")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        workspace = args.workspace.resolve()
        allowed = ROOT / "cloud/fixed_epoch_v1/assessment_workspace"
        if workspace != allowed.resolve():
            raise ValueError("Use the isolated registered assessment workspace")
        campaign_path = checked_path(workspace, args.campaign)
        prepared = prepare_campaign(campaign_path, root=workspace)
        claim = claim_evaluation(prepared, [sys.executable, *sys.argv])
        result = score_fixed(prepared["y"], prepared["folds"], prepared["incumbent"], prepared["arms"])
        reference = result["metrics"]["incumbent"]
        selection = prepared["selection"]
        if (abs(reference["pooled_auc"] - selection["oof_auc"]) > 2e-12
                or not np.allclose(reference["fold_auc"], selection["fold_auc"], atol=2e-12, rtol=0)):
            raise ValueError("Frozen incumbent metrics cannot be reproduced; claim retained for investigation")
    excluded = {"output", "y", "folds", "incumbent", "arms", "selection"}
    result.update({"id": CAMPAIGN_ID, "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "provenance": {key: value for key, value in prepared.items() if key not in excluded},
                   "claim": claim, "script_sha256": sha256(Path(__file__)),
                   "test_script_sha256": sha256(Path(__file__).with_name("test_evaluate_fixed_epoch_cloud_v1.py")),
                   "row_count": len(prepared["y"]), "class_order": [0, 1],
                   "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")},
                   "raw_train_audit_test_read": False, "native_inference_executed_by_evaluator": False,
                   "native_verification_execution_location": "cloud worker; receipt checked locally",
                   "release_or_incumbent_modified": False})
    destination = prepared["output"] / "evaluation.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(destination), "advance_to_confirmation": result["advancement"]["advance_to_confirmation"],
                      "metrics": result["metrics"], "contrasts": result["contrasts"]}), flush=True)


if __name__ == "__main__":
    main()
