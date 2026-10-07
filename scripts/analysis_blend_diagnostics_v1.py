"""Bounded diagnostics of the frozen accepted blend. Never train or release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "2"

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/research_pass_v1.json"
OUT = ROOT / "artifacts/research_pass_v1/blend"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def require_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Input changed: {path}")
    return actual


def mix(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    values = np.asarray(list(weights.values()), dtype=np.float64)
    if (not len(values) or not np.isfinite(values).all() or (values < 0).any()
            or abs(float(values.sum()) - 1.0) > 1e-12):
        raise ValueError("Convex weights required")
    shape = next(iter(predictions.values())).shape
    result = np.zeros(shape, dtype=np.float64)
    for key, value in weights.items():
        if key not in predictions or predictions[key].shape != shape:
            raise ValueError("Prediction shape/name mismatch")
        result += value * predictions[key]
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1 + 1e-12).any():
        raise ValueError("Invalid mixture probabilities")
    return np.clip(result, 0.0, 1.0)


def aligned_oof(frame: pd.DataFrame, ids: np.ndarray, folds: np.ndarray,
                target: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    required = {"id", "fold", "satisfaction", "prediction"}
    if not required.issubset(frame.columns) or len(frame) != len(ids) or not frame.id.is_unique:
        raise ValueError("Invalid OOF schema or key cardinality")
    order = pd.Index(frame.id).get_indexer(ids)
    if (order < 0).any():
        raise ValueError("Missing development IDs")
    selected = frame.iloc[order]
    if not np.array_equal(selected.fold.to_numpy(), folds):
        raise ValueError("Fold identities differ")
    labels = selected.satisfaction.to_numpy()
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Binary positive-class labels required")
    labels = labels.astype(np.int8)
    if target is not None and not np.array_equal(labels, target):
        raise ValueError("OOF target disagreement")
    probability = selected.prediction.to_numpy(dtype=np.float64)
    if (not np.isfinite(probability).all() or (probability < 0).any()
            or (probability > 1).any()):
        raise ValueError("Invalid OOF probabilities")
    return probability, labels


def load_verified(protocol: dict[str, Any]) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for path_key, hash_key in (("anchor_path", "anchor_sha256"),
                               ("v2_path", "v2_sha256"),
                               ("split_path", "split_sha256")):
        path = ROOT / protocol[path_key]
        hashes[protocol[path_key]] = require_hash(path, protocol[hash_key])
    anchor = read_json(ROOT / protocol["anchor_path"])
    v2 = read_json(ROOT / protocol["v2_path"])
    split = pd.read_parquet(ROOT / protocol["split_path"], columns=["id", "fold"])
    split = split.loc[split.fold >= 0].copy()
    if (len(split) != protocol["development_rows"] or not split.id.is_unique
            or sorted(split.fold.unique().tolist()) != protocol["fold_ids"]):
        raise ValueError("Frozen development split contract failed")
    ids, folds = split.id.to_numpy(), split.fold.to_numpy()
    if set(v2["weights"]) - set(anchor["weights"]):
        raise ValueError("V2 includes non-incumbent candidate")
    predictions, results = {}, {}
    y = None
    for name in anchor["weights"]:
        directory = ROOT / "artifacts/runs" / name
        result_path = directory / "result.json"
        hashes[result_path.relative_to(ROOT).as_posix()] = require_hash(
            result_path, anchor["source_result_hashes"][name])
        result = read_json(result_path)
        if result["id"] != name or result["split_hash"] != protocol["split_sha256"]:
            raise ValueError("Run identity or split hash mismatch")
        oof_path = directory / "oof.parquet"
        hashes[oof_path.relative_to(ROOT).as_posix()] = require_hash(
            oof_path, result["artifacts"]["oof.parquet"])
        probability, y = aligned_oof(pd.read_parquet(oof_path), ids, folds, y)
        predictions[name], results[name] = probability, result
    assert y is not None
    current, baseline = mix(predictions, anchor["weights"]), mix(predictions, v2["weights"])
    for value, frozen in ((current, anchor), (baseline, v2)):
        if abs(roc_auc_score(y, value) - frozen["oof_auc"]) > 1e-12:
            raise ValueError("Frozen pooled AUC cannot be reproduced")
        check_folds = [roc_auc_score(y[folds == k], value[folds == k]) for k in protocol["fold_ids"]]
        if not np.allclose(check_folds, frozen["fold_auc"], atol=1e-12, rtol=0):
            raise ValueError("Frozen fold AUC cannot be reproduced")
    block = (protocol["current_block_weights"][0] * baseline
             + protocol["current_block_weights"][1] * predictions[protocol["blocks"][1]]
             + protocol["current_block_weights"][2] * predictions[protocol["blocks"][2]])
    if not np.allclose(current, block, atol=1e-14, rtol=0):
        raise ValueError("Three-block incumbent arithmetic differs")
    return dict(ids=ids, folds=folds, y=y, predictions=predictions, results=results,
                anchor=anchor, v2=v2, current=current, baseline=baseline, input_hashes=hashes)


def metrics(y: np.ndarray, p: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    aucs = [float(roc_auc_score(y[folds == k], p[folds == k])) for k in (0, 1, 2)]
    return {"pooled_auc": float(roc_auc_score(y, p)), "fold_auc": aucs,
            "mean_fold_auc": float(np.mean(aucs)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p))}


def score_change(score: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    return {"pooled_delta": score["pooled_auc"] - anchor["pooled_auc"],
            "mean_fold_delta": score["mean_fold_auc"] - anchor["mean_fold_auc"],
            "fold_delta": (np.asarray(score["fold_auc"]) - anchor["fold_auc"]).tolist()}


def make_units(names: list[str], groups: list[list[str]]) -> dict[str, list[str]]:
    seen: set[str] = set()
    units: dict[str, list[str]] = {}
    for group in groups:
        if (len(group) < 2 or len(set(group)) != len(group)
                or set(group) - set(names) or seen.intersection(group)):
            raise ValueError("Invalid or overlapping registered seed group")
        units["equal_seed_group:" + group[0]] = group
        seen.update(group)
    for name in names:
        if name not in seen:
            units[name] = [name]
    return units


def remove_unit(predictions: dict[str, np.ndarray], weights: dict[str, float],
                names: list[str]) -> tuple[np.ndarray, float]:
    mass = sum(weights[name] for name in names)
    if not 0 < mass < 1:
        raise ValueError("Removal must leave positive mixture mass")
    new_weights = {key: value / (1 - mass) for key, value in weights.items() if key not in names}
    return mix(predictions, new_weights), mass


def choose_grid(rows: list[dict[str, Any]], include_folds: list[int],
                current: np.ndarray, tolerance: float = 1e-12) -> dict[str, Any]:
    if not rows or not include_folds or any(k not in (0, 1, 2) for k in include_folds):
        raise ValueError("Missing candidate grid or invalid fold selection")
    values = np.asarray([np.mean(np.asarray(row["metrics"]["fold_auc"])[include_folds]) for row in rows])
    eligible = [row for row, score in zip(rows, values) if values.max() - score <= tolerance]
    return min(eligible, key=lambda row: (float(np.sum((np.asarray(row["weights"]) - current) ** 2)),
                                          row["weights"][1], row["weights"][2]))


def analyze(data: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    y, folds, predictions = data["y"], data["folds"], data["predictions"]
    weights = data["anchor"]["weights"]
    current_score = metrics(y, data["current"], folds)
    inventory, families = [], {}
    for name, weight in weights.items():
        result = data["results"][name]
        family = "realmlp" if result["family"].startswith("realmlp") else result["family"]
        families.setdefault(family, []).append(name)
        inventory.append({"id": name, "family": family, "weight": weight,
                          "selected_rounds": result["rounds"], "seconds": result["seconds"],
                          "standalone_pooled_auc": result["oof_auc"]})
    ablations = []
    units = make_units(list(weights), protocol["seed_groups"])
    for unit_type, groups in (("seed_preserving_unit", units), ("family", families)):
        for name, members in groups.items():
            if name.startswith("equal_seed_group:"):
                values = [weights[member] for member in members]
                if max(values) - min(values) > 1e-14:
                    raise ValueError("Registered equal seed weights differ")
            p, mass = remove_unit(predictions, weights, members)
            value = metrics(y, p, folds)
            ablations.append({"type": unit_type, "name": name, "members": members,
                              "removed_weight": mass, "metrics": value, **score_change(value, current_score)})
    nn_names = families["realmlp"]
    nn_mass = sum(weights[name] for name in nn_names)
    nn = mix(predictions, {name: weights[name] / nn_mass for name in nn_names})
    tree = mix(predictions, {name: value / (1 - nn_mass) for name, value in weights.items() if name not in nn_names})
    neural_curve = []
    for alpha in sorted(set(protocol["neural_weight_grid"] + [nn_mass])):
        value = metrics(y, alpha * nn + (1 - alpha) * tree, folds)
        neural_curve.append({"neural_weight": alpha, "metrics": value, **score_change(value, current_score)})
    block_grid = []
    nn_add = predictions[protocol["blocks"][1]]
    xgb_add = predictions[protocol["blocks"][2]]
    for nn_weight in protocol["block_weight_grid"]:
        for xgb_weight in protocol["block_weight_grid"]:
            v2_weight = 1 - nn_weight - xgb_weight
            value = metrics(y, v2_weight * data["baseline"] + nn_weight * nn_add + xgb_weight * xgb_add, folds)
            block_grid.append({"weights": [v2_weight, nn_weight, xgb_weight], "metrics": value,
                               **score_change(value, current_score)})
    assert len(block_grid) == 49
    stability = []
    current_weights = np.asarray(protocol["current_block_weights"])
    for excluded_fold in range(3):
        used = [fold for fold in range(3) if fold != excluded_fold]
        choice = choose_grid(block_grid, used, current_weights, protocol["tie_tolerance"])
        stability.append({"excluded_fold": excluded_fold, "selected_using_folds": used,
                          "selected_weights": choice["weights"],
                          "included_mean_auc": float(np.mean(np.asarray(choice["metrics"]["fold_auc"])[used])),
                          "excluded_auc": choice["metrics"]["fold_auc"][excluded_fold],
                          "excluded_delta_vs_incumbent": choice["fold_delta"][excluded_fold],
                          "warning": "Conditional meta-fold stability only; base fits/recipe selection are not independent of excluded fold."})
    family_predictions = {name: mix(predictions, {member: weights[member] / sum(weights[m] for m in members)
                                                  for member in members}) for name, members in families.items()}
    family_matrix = np.column_stack(list(family_predictions.values()))
    probability_correlation = np.corrcoef(family_matrix, rowvar=False)
    residual_correlation = np.corrcoef(family_matrix - y[:, None], rowvar=False)
    eigvals = np.linalg.eigvalsh(residual_correlation)
    equal_blocks = (data["baseline"] + nn_add + xgb_add) / 3
    return {"incumbent": current_score, "v2": metrics(y, data["baseline"], folds),
            "inventory": inventory, "families": families,
            "family_weights": {name: sum(weights[m] for m in members) for name, members in families.items()},
            "ablations": ablations, "neural_weight_curve": neural_curve,
            "block_weight_grid": block_grid, "meta_fold_stability": stability,
            "equal_three_block_metrics": metrics(y, equal_blocks, folds),
            "family_correlation": {"order": list(family_predictions),
                                   "probability": probability_correlation.tolist(),
                                   "residual": residual_correlation.tolist(),
                                   "residual_eigenvalues": eigvals.tolist(),
                                   "spectral_participation_ratio": float(eigvals.sum() ** 2 / np.square(eigvals).sum()),
                                   "warning": "Spectral concentration of residual correlation is descriptive, not an effective count of independent trained models."}}


def save_plots(result: dict[str, Any], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    curve = result["neural_weight_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), layout="constrained")
    x = [r["neural_weight"] for r in curve]
    for fold in range(3):
        axes[0].plot(x, [1e4 * r["fold_delta"][fold] for r in curve], marker=".", label=f"Fold {fold + 1}", alpha=.8)
    axes[0].plot(x, [1e4 * r["pooled_delta"] for r in curve], color="#222222", linewidth=2.2, label="Pooled")
    axes[0].axhline(0, color="#777777", linewidth=.8)
    axes[0].axvline(result["family_weights"]["realmlp"], color="#777777", linestyle=":", label="Incumbent")
    axes[0].set(xlabel="Total neural weight (internal proportions fixed)", ylabel="AUC change vs incumbent × 10,000", title="How sensitive is the accepted blend?")
    axes[0].legend(fontsize=8)
    grid = result["block_weight_grid"]
    z = np.asarray([r["pooled_delta"] * 1e5 for r in grid]).reshape(7, 7)
    im = axes[1].imshow(z, origin="lower", extent=[-.025, .325, -.025, .325], cmap="RdBu_r", aspect="equal")
    axes[1].scatter([.20], [.16], marker="*", s=160, color="#ffe36e", edgecolor="black", linewidth=.8, label="Incumbent")
    axes[1].set(xticks=np.arange(0, .31, .05), yticks=np.arange(0, .31, .05),
                xlabel="V3 XGBoost weight", ylabel="V3 RealMLP weight", title="Two accepted additions; remaining weight on frozen v2")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=axes[1], label="Pooled AUC change × 100,000", shrink=.85)
    fig.suptitle("Development diagnostics only. Reused OOF; no independent estimate or release change.", fontsize=11)
    fig.savefig(out / "blend_weight_sensitivity.png", dpi=160)
    fig.savefig(out / "blend_weight_sensitivity.svg")
    plt.close(fig)
    rows = sorted([r for r in result["ablations"] if r["type"] == "seed_preserving_unit"], key=lambda r: r["pooled_delta"])
    labels = [r["name"].replace("equal_seed_group:", "seed pair: ").replace("v3_", "v3 ").replace("v2_", "v2 ") for r in rows]
    fig, ax = plt.subplots(figsize=(12, 6.2), layout="constrained")
    ax.barh(labels, [-1e5 * r["pooled_delta"] for r in rows], color="#315a7d")
    ax.axvline(0, color="#222222", linewidth=.8)
    ax.set(xlabel="AUC lost when removed × 100,000 (remaining weights renormalized)", title="Conditional contribution inside the frozen blend; seed pairs kept together")
    fig.savefig(out / "conditional_component_contribution.png", dpi=160)
    fig.savefig(out / "conditional_component_contribution.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    if (OUT / "diagnostics.json").exists():
        raise FileExistsError("Diagnostic pass already completed; preserve its results")
    started = datetime.now(timezone.utc)
    protocol = read_json(args.protocol)
    data = load_verified(protocol)
    print(json.dumps({"stage": "inputs_verified", "rows": len(data["y"]), "models": len(data["predictions"])}), flush=True)
    result = analyze(data, protocol)
    result.update({"started_utc": started.isoformat(), "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "protocol_sha256": sha256(args.protocol), "source_sha256": sha256(Path(__file__)),
                   "input_hashes": data["input_hashes"], "rows": len(data["y"]),
                   "fold_sizes": [int(np.sum(data["folds"] == k)) for k in range(3)],
                   "warning": protocol["evaluation_warning"], "training_performed": False,
                   "audit_evaluated": False, "release_modified": False,
                   "versions": {"numpy": np.__version__, "pandas": pd.__version__}})
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "diagnostics.json", result)
    pd.DataFrame(result["inventory"]).to_csv(OUT / "inventory.csv", index=False)
    pd.DataFrame([{**{f"weight_{n}": r["weights"][i] for i, n in enumerate(("v2", "v3_nn", "v3_xgb"))},
                   "pooled_auc": r["metrics"]["pooled_auc"], "mean_fold_auc": r["metrics"]["mean_fold_auc"],
                   **{f"fold_{k}_auc": r["metrics"]["fold_auc"][k] for k in range(3)}} for r in result["block_weight_grid"]]).to_csv(OUT / "block_weight_grid.csv", index=False)
    save_plots(result, OUT)
    print(json.dumps({"stage": "complete", "incumbent": result["incumbent"],
                      "family_weights": result["family_weights"], "meta_fold_stability": result["meta_fold_stability"]}), flush=True)


if __name__ == "__main__":
    main()
