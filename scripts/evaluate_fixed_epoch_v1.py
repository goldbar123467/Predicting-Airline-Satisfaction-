"""One-time development evaluation of the complete fixed A/B/C campaign.

No training, audit/test scoring, alpha search, release mutation or submission.
Runner-produced full raw/native reload receipts are independently checked;
this evaluator does not itself execute native model inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "fixed_epoch_v1"
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
            "receipt_verification": "Runner's full raw/native reload receipt independently checked; evaluator did not re-execute inference",
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


def prepare_campaign(campaign_path: Path, *, root: Path = ROOT) -> dict:
    """Perform all completion, provenance and row checks before any metric call."""
    campaign_path = campaign_path.resolve()
    if not campaign_path.is_relative_to(root.resolve()):
        raise ValueError("Campaign wrapper must be in workspace")
    campaign = read_json(campaign_path)
    if (campaign["id"] != CAMPAIGN_ID or campaign["split_sha256"] != SPLIT_HASH
            or campaign["incumbent_selection_sha256"] != INCUMBENT_HASH):
        raise ValueError("Campaign must preserve fixed split and incumbent")
    output = checked_path(root, campaign["output_dir"])
    if output != (root / "artifacts/fixed_epoch_v1").resolve():
        raise ValueError("Only isolated fixed_epoch_v1 output is allowed")
    if (output / "evaluation.json").exists() or (output / "evaluation_claim.json").exists():
        raise FileExistsError("Evaluation has already been claimed or completed")
    registry_path = checked_path(root, campaign["registry_path"], inside=output)
    registry_hash = require_hash(registry_path, campaign["registry_sha256"])
    registry = read_json(registry_path)
    if (registry["id"] != CAMPAIGN_ID or registry["expected_fit_count"] != 12
            or registry["evaluation"] != EVALUATION):
        raise ValueError("Registered fixed evaluation rule differs")
    sources = registry["source_hashes"]
    if "scripts/evaluate_fixed_epoch_v1.py" not in sources:
        raise ValueError("Registry must bind evaluator source")
    for relative, expected in sources.items():
        require_hash(checked_path(root, relative), expected)
    wrapper_hash = sha256(campaign_path)
    completion_path = checked_path(root, campaign["completion_receipt_path"], inside=output)
    completion = read_json(completion_path)
    bindings = {"id": CAMPAIGN_ID, "campaign_sha256": wrapper_hash, "registry_sha256": registry_hash}
    complete_fields = {**bindings, "status": "completed", "completed_fit_count": 12,
                       "completed_endpoint_count": 9, "evaluation_ready": True}
    if any(completion.get(key) != value for key, value in complete_fields.items()):
        raise ValueError("Campaign is not fully completed and ready for one-time evaluation")
    manifest_path = checked_path(root, campaign["completed_manifest_path"], inside=output)
    manifest_hash = require_hash(manifest_path, completion["completed_manifest_sha256"])
    manifest = read_json(manifest_path)
    if any(manifest.get(key) != value for key, value in bindings.items()):
        raise ValueError("Completed manifest is not bound to this campaign and registry")
    trajectories, endpoints = validate_manifest_inventory(manifest)
    trajectory_evidence = []
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
                or set(adapter["endpoints"]) != {str(epoch) for epoch in required["checkpoint_epochs"]}):
            raise ValueError("Adapter receipt does not prove the registered completed trajectory")
        for key, expected in {"campaign": CAMPAIGN_ID, "fold": row["fold"], "phase": row["phase"],
                              "trajectory": row["trajectory"], "partition_sha256": receipt["partition_sha256"]}.items():
            if adapter["context"].get(key) != expected:
                raise ValueError("Adapter context differs from trajectory identity")
        for epoch, endpoint in adapter["endpoints"].items():
            endpoint_dir = trajectory_dir / f"epoch_{int(epoch):03d}"
            if Path(endpoint["path"]).resolve() != endpoint_dir.resolve():
                raise ValueError("Adapter endpoint path differs")
            require_hash(endpoint_dir / "metadata.json", endpoint["metadata_sha256"])
            require_hash(endpoint_dir / "graph.pt", endpoint["graph_sha256"])
        for source, expected in adapter["source_sha256"].items():
            require_hash(checked_path(root, source), expected)
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
    require_hash(split_path, SPLIT_HASH)
    split = development_split(pd.read_parquet(split_path, columns=["id", "fold"]), 629671)
    ids, folds = split.id.to_numpy(), split.fold.to_numpy()
    partition_evidence = []
    for row in trajectory_evidence:
        with np.load(checked_path(root, row["partition_path"], inside=output), allow_pickle=False) as partition:
            checked = validate_partition(dict(partition), ids, folds, row["fold"], row["phase"])
        partition_evidence.append({"trajectory": row["trajectory"], **checked})
    selection_path = checked_path(root, campaign["incumbent_selection_path"])
    require_hash(selection_path, INCUMBENT_HASH)
    selection = read_json(selection_path)
    y, incumbent, incumbent_evidence = load_incumbent(selection, ids, folds, root)
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
            "incumbent_evidence": incumbent_evidence, "endpoint_manifest": endpoints}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        prepared = prepare_campaign(args.campaign)
        claim = {"claimed_utc": datetime.now(timezone.utc).isoformat(),
                 "campaign_sha256": prepared["campaign_sha256"], "registry_sha256": prepared["registry_sha256"],
                 "manifest_sha256": prepared["manifest_sha256"], "invocation": [sys.executable, *sys.argv]}
        with (prepared["output"] / "evaluation_claim.json").open("x", encoding="utf-8") as stream:
            json.dump(claim, stream, indent=2)
            stream.write("\n")
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
                   "test_script_sha256": sha256(Path(__file__).with_name("test_evaluate_fixed_epoch_v1.py")),
                   "row_count": len(prepared["y"]), "class_order": [0, 1],
                   "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")},
                   "raw_train_audit_test_read": False, "native_inference_executed_by_evaluator": False,
                   "release_or_incumbent_modified": False})
    destination = prepared["output"] / "evaluation.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(destination), "advance_to_confirmation": result["advancement"]["advance_to_confirmation"],
                      "metrics": result["metrics"], "contrasts": result["contrasts"]}), flush=True)


if __name__ == "__main__":
    main()
