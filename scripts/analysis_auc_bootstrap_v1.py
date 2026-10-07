"""Stratified paired row-bootstrap extension to frozen-score AUC diagnostics.

This does not train, tune, select, refit, or assess audit/test predictions. All
comparators share exactly the same resampled rows in each replicate. Sorting is
cached; integer bootstrap counts give exact tied-pair AUC without re-sorting.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from analysis_auc_uncertainty_v1 import ROOT, LIMITATION, sha256, verify_inputs
import numpy as np
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class WeightedAUC:
    order: np.ndarray
    starts: np.ndarray
    positive_sorted: np.ndarray

    @classmethod
    def prepare(cls, y: np.ndarray, score: np.ndarray) -> "WeightedAUC":
        y, score = np.asarray(y), np.asarray(score, dtype=np.float64)
        if (y.ndim != 1 or y.shape != score.shape or not np.isin(y, [0, 1]).all()
                or len(np.unique(y)) != 2 or not np.isfinite(score).all()):
            raise ValueError("Finite aligned scores and both binary classes required")
        order = np.argsort(score, kind="stable")
        sorted_scores = score[order]
        starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        return cls(order, starts, (y[order] == 1))

    def calculate(self, counts: np.ndarray) -> float:
        counts = np.asarray(counts, dtype=np.float64)
        if (counts.shape != self.order.shape or not np.isfinite(counts).all()
                or (counts < 0).any()):
            raise ValueError("Finite nonnegative row counts required")
        ordered_counts = counts[self.order]
        positive = np.add.reduceat(ordered_counts * self.positive_sorted, self.starts)
        negative = np.add.reduceat(ordered_counts * ~self.positive_sorted, self.starts)
        denominator = positive.sum() * negative.sum()
        if denominator <= 0:
            raise ValueError("Positive bootstrap weight required in each class")
        negative_below = np.cumsum(negative) - 0.5 * negative
        return float(np.dot(positive, negative_below) / denominator)


def class_stratified_counts(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw n_positive and n_negative rows separately with replacement."""
    result = np.zeros(len(y), dtype=np.int64)
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        if len(indices) == 0:
            raise ValueError("Both classes required")
        result[indices] = np.bincount(rng.integers(len(indices), size=len(indices)),
                                     minlength=len(indices))
    return result


def main() -> None:
    destination = ROOT / "artifacts/research_pass_v1/validation/paired_bootstrap.json"
    if destination.exists():
        raise FileExistsError("Preserve completed bootstrap diagnostic")
    before = ROOT / "artifacts/research_pass_v1/validation/paired_auc.json"
    original_hash = sha256(before)
    original = json.loads(before.read_text(encoding="utf-8"))
    started = datetime.now(timezone.utc)
    with threadpool_limits(limits=2):
        y, folds, vectors, evidence = verify_inputs()
        prepared = {name: WeightedAUC.prepare(y, score) for name, score in vectors.items()}
        for name, structure in prepared.items():
            expected = original["metrics"][name]["pooled_auc"]
            if abs(structure.calculate(np.ones(len(y))) - expected) > 2e-14:
                raise ValueError("Weighted pair AUC cannot reproduce independent recorded metric")
        rng = np.random.default_rng(20261002)
        draws = {name: [] for name in vectors if name != "incumbent"}
        for iteration in range(200):
            counts = class_stratified_counts(y, rng)
            reference = prepared["incumbent"].calculate(counts)
            for name in draws:
                draws[name].append(reference - prepared[name].calculate(counts))
            if (iteration + 1) % 50 == 0:
                print(json.dumps({"completed_replicates": iteration + 1}), flush=True)
        comparisons = []
        for paired in original["comparisons"]:
            name = paired["comparator"]
            delta = np.asarray(draws[name])
            standard_error = float(delta.std(ddof=1))
            comparisons.append({
                "reference": "incumbent", "comparator": name,
                "difference_reference_minus_comparator": paired["difference_reference_minus_comparator"],
                "bootstrap_mean_difference": float(delta.mean()),
                "bootstrap_standard_error": standard_error,
                "placement_standard_error": paired["paired_standard_error"],
                "bootstrap_to_placement_se_ratio": standard_error / paired["paired_standard_error"],
                "descriptive_conditional_percentile_interval_95": np.quantile(delta, [.025, .975]).tolist(),
                "replicate_differences": delta.tolist(), "interval_interpretation": LIMITATION,
            })
    if sha256(before) != original_hash:
        raise ValueError("The original paired diagnostic changed during bootstrap")
    record = {"started_utc": started.isoformat(),
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "seed": 20261002, "replicates": 200,
              "stratification": "Binary class, preserving the two observed class sample sizes",
              "pairing": "Exactly the same resampled row multiplicities for every score vector per replicate",
              "fold_policy": "Original fold labels retained; no resplitting/retraining; not fold-stratified",
              "approximation": "Only 200 replicates; percentile tails and bootstrap SE have Monte Carlo error",
              "interval_interpretation": LIMITATION, "comparisons": comparisons,
              "original_paired_diagnostic_sha256": original_hash,
              "input_verification": evidence, "no_weight_optimization": True,
              "script_sha256": sha256(Path(__file__)),
              "shared_diagnostic_script_sha256": sha256(Path(__file__).with_name("analysis_auc_uncertainty_v1.py")),
              "test_script_sha256": sha256(Path(__file__).with_name("test_analysis_auc_bootstrap_v1.py")),
              "invocation": [sys.executable, *sys.argv]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for result in comparisons:
        print(json.dumps({key: result[key] for key in (
            "comparator", "bootstrap_standard_error", "bootstrap_to_placement_se_ratio",
            "descriptive_conditional_percentile_interval_95")}), flush=True)


if __name__ == "__main__":
    main()
