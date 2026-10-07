"""Development-only OOF diagnostics and a read-only supervisor snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow.dataset as ds
from sklearn.metrics import roc_auc_score

from common import ROOT, TARGET, atomic_json, load_config, sha256
from supervisor import owned_process, owned_tree_metrics


def auc_or_none(labels: np.ndarray, probability: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, probability)) if np.unique(labels).size == 2 else None


def correlation_mapping(matrix: np.ndarray, names: list[str]) -> dict[str, dict[str, float | None]]:
    correlation = np.corrcoef(matrix, rowvar=False) if matrix.shape[1] > 1 else np.ones((1, 1))
    return {a: {b: float(correlation[i, j]) if np.isfinite(correlation[i, j]) else None
                for j, b in enumerate(names)} for i, a in enumerate(names)}


def operations_snapshot() -> dict[str, Any]:
    path = ROOT / "state" / "run_state.json"
    if not path.exists():
        return {"state": "not_started"}
    state = load_config(path)
    result: dict[str, Any] = {
        "status": state.get("status"), "state_updated_utc": state.get("updated_utc"),
        "supervisor_pid": state.get("supervisor_pid"),
        "available_ram_bytes": psutil.virtual_memory().available,
    }
    try:
        process = psutil.Process(state["supervisor_pid"])
        result["supervisor_identity_verified"] = abs(process.create_time() - state["supervisor_create_time"]) <= .01
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        result["supervisor_identity_verified"] = False
    active = state.get("active_child")
    if active:
        live = owned_tree_metrics(active)  # Mutates only this in-memory copy.
        result["active_child"] = {
            "label": active["label"], "launcher_pid": active["pid"],
            "launcher_identity_verified": owned_process(active) is not None,
            "recorded_metrics": {key: active.get(key) for key in live},
            "live_metrics": live,
            "verified_descendant_pids": [entry["pid"] for entry in active.get("descendants", [])
                                         if owned_process(entry) is not None],
            "cutoff_utc": active.get("cutoff_utc"),
            "latest_log_utc": active.get("latest_log_utc"),
        }
    return result


def generate(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    split = pd.read_parquet(ROOT / "data" / "splits.parquet")
    if not split.id.is_unique:
        raise ValueError("Split IDs are not unique")
    development = split.loc[split.fold >= 0, ["id", "fold"]].reset_index(drop=True)
    ids = development.id.to_numpy()
    folds = development.fold.to_numpy()
    # Dataset filtering happens before conversion to a pandas frame. No audit
    # target rows or test data are returned to this diagnostic process.
    dataset = ds.dataset(ROOT / "data" / "train.parquet", format="parquet")
    columns = ["id", TARGET, "Type of Travel", "Class", "Gender", "Age", "Arrival Delay in Minutes"]
    rows = dataset.to_table(columns=columns, filter=ds.field("id").isin(ids)).to_pandas()
    if not rows.id.is_unique or len(rows) != len(ids):
        raise ValueError("Development feature IDs do not match the frozen split")
    rows = rows.set_index("id").loc[ids].reset_index()
    np.testing.assert_array_equal(rows.id.to_numpy(), ids)
    labels = rows[TARGET].to_numpy(dtype=np.int8)
    split_hash = sha256(ROOT / "data" / "splits.parquet")
    blend_path = ROOT / "artifacts" / "blend" / "current.json"
    blend_snapshot = load_config(blend_path) if blend_path.exists() else None
    predictions: dict[str, np.ndarray] = {}
    model_scores: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    for run in config["runs"]:
        base = ROOT / "artifacts" / "runs" / run["id"]
        if not (base / "result.json").exists():
            continue
        result = load_config(base / "result.json")
        if result["run"] != run or result["split_hash"] != split_hash:
            raise ValueError(f"Run identity mismatch: {run['id']}")
        digest = sha256(base / "oof.parquet")
        if digest != result["artifacts"]["oof.parquet"]:
            raise ValueError(f"OOF checksum mismatch: {run['id']}")
        frame = pd.read_parquet(base / "oof.parquet")
        if not frame.id.is_unique or len(frame) != len(ids):
            raise ValueError(f"OOF identity mismatch: {run['id']}")
        frame = frame.set_index("id").loc[ids]
        np.testing.assert_array_equal(frame.fold.to_numpy(), folds)
        np.testing.assert_array_equal(frame[TARGET].to_numpy(), labels)
        probability = frame.prediction.to_numpy(dtype=np.float64)
        if probability.shape != labels.shape or not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError(f"Invalid probabilities: {run['id']}")
        predictions[run["id"]] = probability
        input_hashes[run["id"]] = digest
        model_scores[run["id"]] = {
            "pooled_auc": float(roc_auc_score(labels, probability)),
            "fold_auc": [float(roc_auc_score(labels[folds == k], probability[folds == k])) for k in range(3)],
            "brier_score": float(np.mean((probability - labels) ** 2)),
            "training_seconds": result.get("seconds"), "rounds": result.get("rounds"),
        }
    if not predictions:
        raise RuntimeError("No completed candidate available")
    best = max(model_scores, key=lambda name: model_scores[name]["pooled_auc"])
    for name, value in model_scores.items():
        value["auc_minus_best"] = value["pooled_auc"] - model_scores[best]["pooled_auc"]
        value["fold_auc_minus_best"] = (np.array(value["fold_auc"]) - model_scores[best]["fold_auc"]).tolist()
    names = list(predictions)
    matrix = np.column_stack([predictions[name] for name in names])
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "development OOF only; no audit labels, audit scores, or test predictions read",
        "rows": len(rows), "positive_rate": float(labels.mean()), "split_hash": split_hash,
        "oof_hashes": input_hashes, "best_single": best, "models": model_scores,
        "prediction_correlation": correlation_mapping(matrix, names),
        "residual_correlation": correlation_mapping(matrix - labels[:, None], names),
        "paired_difference_note": "AUC deltas use identical rows and folds. Weight selection uses these OOF rows, so these are development diagnostics, not an independent generalization estimate.",
    }
    if blend_snapshot:
        if blend_snapshot["split_hash"] != split_hash:
            raise ValueError("Current blend uses a different split")
        weights = blend_snapshot["weights"]
        if not set(weights).issubset(predictions):
            raise ValueError("Current blend has candidates absent from this completed-run snapshot")
        if any(weight < 0 for weight in weights.values()) or abs(sum(weights.values()) - 1) > 1e-9:
            raise ValueError("Invalid current blend weights")
        blended = sum(weight * predictions[name] for name, weight in weights.items())
        blend_auc = float(roc_auc_score(labels, blended))
        if abs(blend_auc - blend_snapshot["oof_auc"]) > 1e-10:
            raise ValueError("Current blend metadata does not reproduce from OOF predictions")
        blend_folds = [float(roc_auc_score(labels[folds == k], blended[folds == k])) for k in range(3)]
        result["current_blend"] = {
            "snapshot_created_utc": blend_snapshot["created_utc"], "weights": weights,
            "pooled_auc": blend_auc, "auc_minus_best": blend_auc - model_scores[best]["pooled_auc"],
            "fold_auc": blend_folds,
            "fold_auc_minus_best": (np.array(blend_folds) - model_scores[best]["fold_auc"]).tolist(),
            "contains_all_completed_candidates": set(weights) == set(predictions),
        }
        predictions["current_blend"] = blended
    age = pd.cut(rows.Age, bins=[-np.inf, 20, 35, 50, 65, np.inf],
                 labels=["0-20", "21-35", "36-50", "51-65", "66+"]).astype("string").fillna("missing")
    dimensions = {
        "travel": rows["Type of Travel"].astype("string").fillna("missing"),
        "class": rows["Class"].astype("string").fillna("missing"),
        "gender": rows["Gender"].astype("string").fillna("missing"),
        "age_bucket": age,
        "arrival_delay_missing": pd.Series(np.where(rows["Arrival Delay in Minutes"].isna(), "missing", "observed")),
    }
    slices = []
    for dimension, groups in dimensions.items():
        for value in sorted(groups.unique()):
            mask = groups.eq(value).to_numpy(dtype=bool)
            aucs = {name: auc_or_none(labels[mask], probability[mask]) for name, probability in predictions.items()}
            current = aucs.get("current_blend")
            baseline = aucs[best]
            slices.append({
                "dimension": dimension, "value": str(value), "rows": int(mask.sum()),
                "positive_rate": float(labels[mask].mean()), "auc": aucs,
                "blend_minus_best": current - baseline if current is not None and baseline is not None else None,
            })
    result["slices"] = slices
    result["operations"] = operations_snapshot()
    return result


def markdown_report(result: dict[str, Any]) -> str:
    text = ["# Development OOF diagnostics", "", f"Generated: {result['created_utc']}", "",
            result["evaluation_scope"] + ".", "",
            f"{result['rows']:,} development rows. Best completed single: `{result['best_single']}`.", "",
            "| Model | Pooled AUC | Fold 0 | Fold 1 | Fold 2 | Delta vs best |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, value in result["models"].items():
        text.append(f"| {name} | {value['pooled_auc']:.8f} | " + " | ".join(f"{score:.8f}" for score in value["fold_auc"]) + f" | {value['auc_minus_best']:+.8f} |")
    if "current_blend" in result:
        blend = result["current_blend"]
        text.extend(["", f"Current blend AUC: **{blend['pooled_auc']:.8f}**, delta vs best: **{blend['auc_minus_best']:+.8f}**.",
                     "Fold deltas: " + ", ".join(f"{delta:+.8f}" for delta in blend["fold_auc_minus_best"]) + ".",
                     f"Blend contains every completed candidate in this snapshot: {blend['contains_all_completed_candidates']}."])
    text.extend(["", "| Slice | Rows | Positive rate | Best single AUC | Blend delta |",
                 "|---|---:|---:|---:|---:|"])
    for value in result["slices"]:
        auc = value["auc"][result["best_single"]]
        delta = value["blend_minus_best"]
        auc_text = "undefined" if auc is None else f"{auc:.6f}"
        delta_text = "undefined" if delta is None else f"{delta:+.6f}"
        text.append(f"| {value['dimension']}: {value['value']} | {value['rows']:,} | {value['positive_rate']:.3f} | {auc_text} | {delta_text} |")
    text.extend(["", result["paired_difference_note"], "",
                 "The JSON artifact includes complete model and residual correlation matrices, per-model slice scores, input hashes, and the supervisor process-tree snapshot.", ""])
    return "\n".join(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "overnight.json")
    args = parser.parse_args()
    result = generate(args.config)
    atomic_json(ROOT / "artifacts" / "diagnostics.json", result)
    report = ROOT / "research" / "diagnostics.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({"models": list(result["models"]), "best_single": result["best_single"],
                      "current_blend": result.get("current_blend"), "operations": result["operations"]}, indent=2))


if __name__ == "__main__":
    main()
