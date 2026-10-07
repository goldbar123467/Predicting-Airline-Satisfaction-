"""Fixed-prediction paired AUC diagnostics. Never a release-selection command.

The descriptive intervals assume independent sampled rows conditional on the
already fitted and selected scores. They omit training, adaptive selection,
cross-fold training dependence, and distribution-shift uncertainty.
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

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "2"

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TARGET = "satisfaction"
EXPECTED_HASHES = {
    "DEEP_RESEARCH_PASS_PLAN.md": "a52374b99626aa31e539757990fa4e940c6a38916ecfee3f5965406b124f1ef1",
    "data/splits.parquet": "4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c",
    "artifacts/third_pass/blend/frozen.json": "bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311",
    "artifacts/second_pass/blend/frozen.json": "d75c7eba29f8ed9cb3e9713e4f43b102a5d48dacdcfbba616f9c331a28bb0dc1",
}
LIMITATION = (
    "Descriptive conditional interval for fixed scores under independent-row "
    "sampling; excludes model-training variance, overlapping-fold training "
    "dependence, all adaptive recipe/weight selection, and distribution shift. "
    "It is not a selection-adjusted confidence interval or a release gate."
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def placements(y: np.ndarray, score: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return AUC and class-conditional pair win rates, with half-credit ties.

    Positive row i: mean_j h(score_i - score_j), j negative.
    Negative row j: mean_i h(score_i - score_j), i positive.
    Time O(n log n), memory O(n), without materializing n_pos*n_neg pairs.
    """
    y, score = np.asarray(y), np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or score.ndim != 1 or y.shape != score.shape:
        raise ValueError("Labels and scores must be matching one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("Labels must be binary and scores must be finite")
    positive, negative = score[y == 1], score[y == 0]
    if min(len(positive), len(negative)) < 2:
        raise ValueError("Each class needs at least two observations for sample variance")
    negative_sorted, positive_sorted = np.sort(negative), np.sort(positive)
    pos_left = np.searchsorted(negative_sorted, positive, side="left")
    pos_right = np.searchsorted(negative_sorted, positive, side="right")
    negative_below = (pos_left + pos_right) / (2.0 * len(negative))
    neg_left = np.searchsorted(positive_sorted, negative, side="left")
    neg_right = np.searchsorted(positive_sorted, negative, side="right")
    positive_above = 1.0 - (neg_left + neg_right) / (2.0 * len(positive))
    auc = float(negative_below.mean())
    if abs(auc - float(positive_above.mean())) > 2e-14:
        raise ValueError("Positive and negative pair accounting disagree")
    return auc, negative_below, positive_above


def paired_auc(y: np.ndarray, reference: np.ndarray, comparator: np.ndarray) -> dict:
    """Paired DeLong structural-component covariance and difference interval.

    Direct variance of placement differences avoids subtracting nearly equal
    variances when scores are strongly correlated. Sample covariance uses ddof=1.
    """
    ref_auc, ref_pos, ref_neg = placements(y, reference)
    cmp_auc, cmp_pos, cmp_neg = placements(y, comparator)
    m, n = len(ref_pos), len(ref_neg)
    covariance = (np.cov(np.stack([ref_pos, cmp_pos]), ddof=1) / m
                  + np.cov(np.stack([ref_neg, cmp_neg]), ddof=1) / n)
    difference = ref_auc - cmp_auc
    variance = float(np.var(ref_pos - cmp_pos, ddof=1) / m
                     + np.var(ref_neg - cmp_neg, ddof=1) / n)
    se = float(np.sqrt(variance))
    margin = 1.959963984540054 * se
    return {
        "reference_auc": ref_auc, "comparator_auc": cmp_auc,
        "difference_reference_minus_comparator": difference,
        "paired_standard_error": se,
        "descriptive_conditional_interval_95": [difference - margin, difference + margin],
        "paired_covariance": covariance.tolist(),
        "positive_rows": m, "negative_rows": n, "positive_negative_pairs": m * n,
        "interval_interpretation": LIMITATION,
    }


def metric_summary(y: np.ndarray, score: np.ndarray, folds: np.ndarray) -> dict:
    fold_values = [float(roc_auc_score(y[folds == fold], score[folds == fold]))
                   for fold in range(3)]
    return {"pooled_auc": float(roc_auc_score(y, score)), "fold_auc": fold_values,
            "mean_fold_auc": float(np.mean(fold_values)),
            "log_loss": float(log_loss(y, score, labels=[0, 1])),
            "brier": float(brier_score_loss(y, score)),
            "mean_probability": float(score.mean())}


def verify_inputs() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    """Read only splits and accepted development OOF; no raw/audit/test files."""
    checked = {}
    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(f"Registered input hash changed: {relative}")
        checked[relative] = actual
    split = pd.read_parquet(ROOT / "data/splits.parquet", columns=["id", "fold"])
    split = split.loc[split.fold >= 0].copy()
    if len(split) != 629671 or not split.id.is_unique or set(split.fold) != {0, 1, 2}:
        raise ValueError("Frozen development identity is invalid")
    ids, folds = split.id.to_numpy(), split.fold.to_numpy()
    incumbent = json_read(ROOT / "artifacts/third_pass/blend/frozen.json")
    v2 = json_read(ROOT / "artifacts/second_pass/blend/frozen.json")
    for selection in (incumbent, v2):
        if selection["split_hash"] != checked["data/splits.parquet"]:
            raise ValueError("Selection split mismatch")
        weights = selection["weights"]
        if (any(not np.isfinite(w) or w <= 0 for w in weights.values())
                or abs(sum(weights.values()) - 1) > 1e-12):
            raise ValueError("Invalid frozen weights")
    if len(incumbent["weights"]) != 15 or not set(v2["weights"]).issubset(incumbent["weights"]):
        raise ValueError("Unexpected registered incumbent members")
    y = None
    current, previous = np.zeros(len(ids)), np.zeros(len(ids))
    family_scores = {family: np.zeros(len(ids))
                     for family in ("realmlp", "catboost", "lightgbm", "xgboost")}
    family_weights = dict.fromkeys(family_scores, 0.0)
    blocks, member_manifest = {}, []
    for name, weight in incumbent["weights"].items():
        directory = ROOT / "artifacts/runs" / name
        result_path = directory / "result.json"
        result_hash = sha256(result_path)
        if result_hash != incumbent["source_result_hashes"][name]:
            raise ValueError(f"Selected result changed: {name}")
        if name in v2["weights"] and result_hash != v2["source_result_hashes"][name]:
            raise ValueError(f"V2 result changed: {name}")
        result = json_read(result_path)
        contract_path = directory / "contract.json"
        if sha256(contract_path) != result["contract_hash"]:
            raise ValueError(f"Run contract changed: {name}")
        contract = json_read(contract_path)
        if (contract["run"] != result["run"] or result["id"] != name
                or contract["split_hash"] != checked["data/splits.parquet"]
                or result["split_hash"] != contract["split_hash"]):
            raise ValueError(f"Run identity mismatch: {name}")
        for source, expected in contract["code_hashes"].items():
            source_path = directory / "source" / source
            if sha256(source_path) != expected:
                raise ValueError(f"Recorded source changed: {name}/{source}")
        if sha256(directory / "source/requirements.lock.txt") != contract["environment_hash"]:
            raise ValueError(f"Recorded environment changed: {name}")
        oof_path = directory / "oof.parquet"
        oof_hash = sha256(oof_path)
        if oof_hash != result["artifacts"]["oof.parquet"]:
            raise ValueError(f"OOF bytes changed: {name}")
        frame = pd.read_parquet(oof_path, columns=["id", "fold", TARGET, "prediction"])
        if len(frame) != len(ids) or not frame.id.is_unique:
            raise ValueError(f"OOF duplicate/missing IDs: {name}")
        order = pd.Index(frame.id).get_indexer(ids)
        if (order < 0).any():
            raise ValueError(f"OOF missing development IDs: {name}")
        frame = frame.iloc[order]
        if not np.array_equal(frame.fold.to_numpy(), folds):
            raise ValueError(f"OOF fold mismatch: {name}")
        labels = frame[TARGET].to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"OOF labels are not binary: {name}")
        if y is None:
            y = labels.astype(np.int8)
        elif not np.array_equal(labels, y):
            raise ValueError(f"OOF label mismatch: {name}")
        score = frame.prediction.to_numpy(dtype=np.float64)
        if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
            raise ValueError(f"OOF probability violation: {name}")
        summary = metric_summary(y, score, folds)
        if (abs(summary["pooled_auc"] - result["oof_auc"]) > 2e-12
                or np.max(np.abs(np.asarray(summary["fold_auc"]) - result["fold_auc"])) > 2e-12):
            raise ValueError(f"Recorded AUC cannot be reproduced: {name}")
        family = result["family"]
        if family in {"realmlp", "realmlp_cat"}:
            family = "realmlp"
        if family not in family_scores:
            raise ValueError(f"Unexpected accepted model family: {family}")
        current += weight * score
        previous += v2["weights"].get(name, 0.0) * score
        family_scores[family] += weight * score
        family_weights[family] += weight
        if name in {"v3_realmlp_cat_raw_aux_probability", "v3_xgb_route_te_teacher_profiles_aux_probability"}:
            blocks[name] = score.copy()
        member_manifest.append({"id": name, "family": family, "weight": weight,
                                "run": result["run"], "metrics": summary,
                                "result_sha256": result_hash, "oof_sha256": oof_hash,
                                "contract_sha256": result["contract_hash"],
                                "source_hashes_verified": contract["code_hashes"]})
    if y is None:
        raise ValueError("No development labels loaded")
    for score, selection in ((current, incumbent), (previous, v2)):
        if abs(float(roc_auc_score(y, score)) - selection["oof_auc"]) > 2e-12:
            raise ValueError("Frozen ensemble AUC cannot be reproduced")
    score_vectors = {"incumbent": current, "frozen_v2": previous}
    for family, numerator in family_scores.items():
        denominator = 1.0 - family_weights[family]
        if denominator <= 0:
            raise ValueError("Cannot remove all ensemble mass")
        score_vectors[f"without_{family}"] = (current - numerator) / denominator
    if len(blocks) != 2:
        raise ValueError("Missing registered v3 blocks")
    score_vectors["equal_three_blocks"] = (previous + sum(blocks.values())) / 3.0
    evidence = {"registered_input_hashes": checked, "members": member_manifest,
                "family_weights": family_weights,
                "development_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                "development_labels_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                "development_folds_sha256": hashlib.sha256(folds.tobytes()).hexdigest(),
                "labels_obtained_only_from_verified_development_oof": True,
                "read_raw_train_audit_or_test": False}
    return y, folds, score_vectors, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/research_pass_v1/validation/paired_auc.json")
    args = parser.parse_args()
    allowed = (ROOT / "artifacts/research_pass_v1/validation").resolve()
    destination = args.output.resolve()
    if not destination.is_relative_to(allowed):
        raise ValueError("Output must remain inside this isolated diagnostic directory")
    if destination.exists():
        raise FileExistsError("Preserve completed diagnostics; use a new output filename")
    with threadpool_limits(limits=2):
        y, folds, vectors, evidence = verify_inputs()
        summaries = {name: metric_summary(y, score, folds) for name, score in vectors.items()}
        comparisons = []
        for name, score in vectors.items():
            if name == "incumbent":
                continue
            item = paired_auc(y, vectors["incumbent"], score)
            item.update(reference="incumbent", comparator=name,
                        mean_fold_difference=(summaries["incumbent"]["mean_fold_auc"]
                                              - summaries[name]["mean_fold_auc"]))
            item["fold_comparisons"] = [
                {"fold": fold, **paired_auc(y[folds == fold], vectors["incumbent"][folds == fold],
                                          score[folds == fold])}
                for fold in range(3)]
            comparisons.append(item)
            print(json.dumps({key: item[key] for key in (
                "comparator", "difference_reference_minus_comparator", "paired_standard_error",
                "descriptive_conditional_interval_95")}), flush=True)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "analysis": "six preregistered fixed-score paired AUC comparisons",
              "interval_interpretation": LIMITATION, "input_verification": evidence,
              "metrics": summaries, "comparisons": comparisons,
              "bootstrap_executed": False, "no_weight_optimization": True,
              "script_sha256": sha256(Path(__file__)),
              "test_script_sha256": sha256(Path(__file__).with_name("test_analysis_auc_uncertainty_v1.py")),
              "invocation": [sys.executable, *sys.argv], "python": sys.version,
              "versions": {name: importlib.metadata.version(name)
                           for name in ("numpy", "pandas", "scikit-learn", "threadpoolctl")}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
