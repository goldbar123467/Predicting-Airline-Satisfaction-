"""Exact ranking-change geometry for four fixed accepted-model comparisons."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from analysis_blend_diagnostics_v1 import (
    DEFAULT_PROTOCOL, ROOT, load_verified, mix, read_json, sha256, write_json,
)
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score


def rounded_integer(value: float, tolerance: float = .001) -> int:
    nearest = round(value)
    if abs(value - nearest) > tolerance:
        raise ValueError("Floating statistic cannot establish an integer pair count")
    return nearest


def discordances_no_ties(first: np.ndarray, second: np.ndarray) -> int:
    first, second = np.asarray(first), np.asarray(second)
    if (first.ndim != 1 or first.shape != second.shape or len(first) < 2
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("Matching finite score vectors with at least two rows required")
    if len(np.unique(first)) != len(first) or len(np.unique(second)) != len(second):
        raise ValueError("Strict-rank accounting requires no tied scores")
    if len(first) == 2:
        return int((first[0] < first[1]) != (second[0] < second[1]))
    pairs = len(first) * (len(first) - 1) // 2
    tau = kendalltau(first, second, method="asymptotic").statistic
    return rounded_integer((1.0 - tau) * pairs / 2.0)


def ranking_changes(y: np.ndarray, reference: np.ndarray, comparator: np.ndarray) -> dict:
    y = np.asarray(y)
    reference, comparator = np.asarray(reference), np.asarray(comparator)
    if y.ndim != 1 or reference.shape != y.shape or comparator.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Binary aligned labels and scores required")
    positive, negative = y == 1, y == 0
    if min(positive.sum(), negative.sum()) < 2:
        raise ValueError("Both classes need at least two rows")
    all_changes = discordances_no_ties(reference, comparator)
    positive_changes = discordances_no_ties(reference[positive], comparator[positive])
    negative_changes = discordances_no_ties(reference[negative], comparator[negative])
    relevant_changes = all_changes - positive_changes - negative_changes
    pair_count = int(positive.sum()) * int(negative.sum())
    reference_auc = roc_auc_score(y, reference)
    comparator_auc = roc_auc_score(y, comparator)
    reference_correct = rounded_integer(reference_auc * pair_count)
    comparator_correct = rounded_integer(comparator_auc * pair_count)
    net = reference_correct - comparator_correct
    if abs(net) > relevant_changes or (relevant_changes + net) % 2:
        raise ValueError("Pair accounting inconsistent with direct AUC")
    repaired = (relevant_changes + net) // 2
    broken = (relevant_changes - net) // 2
    return {"positive_negative_pairs": pair_count,
            "reference_auc": float(reference_auc), "comparator_auc": float(comparator_auc),
            "auc_difference": float(reference_auc - comparator_auc),
            "opposite_class_pairs_reordered": relevant_changes,
            "pairs_repaired_by_reference": repaired, "pairs_broken_by_reference": broken,
            "net_pairs_repaired": net,
            "fraction_of_cross_class_pairs_reordered": relevant_changes / pair_count,
            "repair_fraction_among_reordered_pairs": repaired / relevant_changes if relevant_changes else None,
            "score_correlation": float(np.corrcoef(reference, comparator)[0, 1]),
            "ties_in_either_score": 0}


def main() -> None:
    protocol_path = ROOT / "configs/research_pair_geometry_v1.json"
    protocol = read_json(protocol_path)
    output = ROOT / "artifacts/research_pass_v1/blend/pair_geometry.json"
    if output.exists():
        raise FileExistsError("Preserve completed diagnostic")
    data = load_verified(read_json(DEFAULT_PROTOCOL))
    weights = data["anchor"]["weights"]
    nn_names = [name for name, result in data["results"].items() if result["family"].startswith("realmlp")]
    mass = sum(weights[name] for name in nn_names)
    nn = mix(data["predictions"], {name: weights[name] / mass for name in nn_names})
    tree = mix(data["predictions"], {name: value / (1 - mass) for name, value in weights.items() if name not in nn_names})
    scores = {"incumbent": data["current"], "v2": data["baseline"], "neural_only": nn,
              "trees_only": tree, "neural_half_trees_half": .5 * (nn + tree)}
    comparisons = []
    for row in protocol["comparisons"]:
        value = {**row, **ranking_changes(data["y"], scores[row["reference"]], scores[row["comparator"]])}
        comparisons.append(value)
        print(json.dumps(value), flush=True)
    write_json(output, {"completed_utc": datetime.now(timezone.utc).isoformat(),
                        "protocol_sha256": sha256(protocol_path),
                        "source_sha256": sha256(Path(__file__)),
                        "base_source_sha256": sha256(Path(__file__).with_name("analysis_blend_diagnostics_v1.py")),
                        "input_hashes": data["input_hashes"], "comparisons": comparisons,
                        "interpretation_limit": protocol["limit"]})


if __name__ == "__main__":
    main()
