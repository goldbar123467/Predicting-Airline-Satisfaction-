"""Conditional joint AUC uncertainty over an existing, frozen 49-point grid.

No model fitting, weight optimization, audit/test reading, or release mutation.
All contrasts are grid minus incumbent; the incumbent is not one of the 49
perturbation-winner candidates. This is a fixed-score iid-row approximation.
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
from threadpoolctl import threadpool_limits

from analysis_auc_uncertainty_v1 import ROOT, json_read, placements, sha256, verify_inputs

PINNED = {
    "configs/research_grid_uncertainty_v1.json": "89929030085377d015ac3d7ecfa15001e23c33359efcb9a82775f8132abe571e",
    "configs/research_pass_v1.json": "24e644e49dd77923290da513f7d642d5b43c1b8f5722904c02694e0d6fbac21d",
    "artifacts/research_pass_v1/blend/diagnostics.json": "a6fe12626b8a533feb7ecf58c58988968e439618182fd4c0a5b253b50f89a9c7",
}
LIMITATION = (
    "Descriptive Gaussian approximation conditional on these 49 fixed score vectors "
    "and the frozen incumbent, assuming independent sampled rows. The simultaneous "
    "band covers only this fixed family of contrasts under that approximation. "
    "It omits training variance, overlapping-fold training dependence, all historical "
    "recipe/anchor/weight selection, and distribution shift. It is not an independent "
    "generalization estimate, exact finite-sample band, posterior, or release gate."
)


def placement_covariance(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Covariance of paired placement differences; rows must have shared identity.

    Each column holds grid placements minus reference placements in the same
    class-row order. Center within each class, use ddof=1, then divide by its n.
    Only one centered class matrix is held at a time.
    """
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if (positive.ndim != 2 or negative.ndim != 2 or positive.shape[1] == 0
            or positive.shape[1] != negative.shape[1]
            or min(positive.shape[0], negative.shape[0]) < 2
            or not np.isfinite(positive).all() or not np.isfinite(negative).all()):
        raise ValueError("Finite matching placement matrices with at least two rows per class required")
    result = np.zeros((positive.shape[1], positive.shape[1]), dtype=np.float64)
    for values in (positive, negative):
        centered = values - values.mean(axis=0)
        result += (centered.T @ centered) / (len(values) * (len(values) - 1))
        del centered
    return result


def psd_factor(covariance: np.ndarray) -> tuple[np.ndarray, dict]:
    """Reject substantive indefiniteness; clip only a reported roundoff allowance.

    The tolerance 64*eps*k*||Sigma||_2 is a conservative backward-error allowance
    for a k-dimensional symmetric float64 eigensolve, not statistical regularization.
    There is no absolute floor or added diagonal jitter.
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    if (covariance.ndim != 2 or covariance.shape[0] == 0
            or covariance.shape[0] != covariance.shape[1]
            or not np.isfinite(covariance).all()):
        raise ValueError("Finite nonempty square covariance required")
    k = len(covariance)
    entry_scale = float(np.max(np.abs(covariance)))
    symmetry_tolerance = 64 * np.finfo(np.float64).eps * k * entry_scale
    asymmetry = float(np.max(np.abs(covariance - covariance.T)))
    if asymmetry > symmetry_tolerance:
        raise ValueError("Covariance is not symmetric within float64 roundoff allowance")
    symmetric = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    spectral_scale = float(np.max(np.abs(eigenvalues)))
    tolerance = 64 * np.finfo(np.float64).eps * k * spectral_scale
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("Covariance has a materially negative eigenvalue")
    clipped = np.maximum(eigenvalues, 0)
    factor = eigenvectors * np.sqrt(clipped)
    return factor, {
        "eigenvalues_before_clipping": eigenvalues.tolist(),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "spectral_scale": spectral_scale, "negative_eigenvalue_tolerance": tolerance,
        "roundoff_rule": "64 * float64_epsilon * number_of_columns * spectral_scale; no absolute floor",
        "negative_eigenvalues_clipped": int(np.count_nonzero(eigenvalues < 0)),
        "maximum_eigenvalue_correction": float(np.max(clipped - eigenvalues)),
        "maximum_input_asymmetry": asymmetry, "symmetry_tolerance": symmetry_tolerance,
        "diagonal_jitter_added": 0.0,
    }


def gaussian_summary(covariance: np.ndarray, observed_delta: np.ndarray,
                     tie_order: np.ndarray, *, seed: int, draws: int,
                     tie_tolerance: float = 1e-12) -> dict:
    """Fixed-grid max-|Z/SE| band and Gaussian-perturbed grid-winner frequencies."""
    covariance = np.asarray(covariance, dtype=np.float64)
    observed_delta = np.asarray(observed_delta, dtype=np.float64)
    tie_order = np.asarray(tie_order)
    k = len(observed_delta)
    if (observed_delta.ndim != 1 or not np.isfinite(observed_delta).all()
            or covariance.shape != (k, k) or tie_order.shape != (k,)
            or not np.issubdtype(tie_order.dtype, np.integer)
            or not np.array_equal(np.sort(tie_order), np.arange(k))
            or draws < 2 or tie_tolerance < 0 or not np.isfinite(tie_tolerance)):
        raise ValueError("Invalid Gaussian contrast dimensions, draws, or tie rule")
    factor, psd = psd_factor(covariance)
    diagonal = np.diag(covariance)
    if (diagonal < 0).any():
        raise ValueError("Covariance diagonal must be nonnegative")
    standard_error = np.sqrt(diagonal)
    noise = np.random.default_rng(seed).standard_normal((draws, k)) @ factor.T
    active = standard_error > 0
    # An exactly constant contrast has no sampling perturbation. Avoid 0/0.
    noise[:, ~active] = 0.0
    identical_columns = []
    for column in range(k):
        for previous in range(column):
            if np.array_equal(covariance[:, column], covariance[:, previous]):
                # Exact covariance identity implies zero difference variance;
                # remove any eigensolver roundoff splitting of these columns.
                noise[:, column] = noise[:, previous]
                identical_columns.append([column, previous])
                break
    if active.any():
        maximum = np.max(np.abs(noise[:, active] / standard_error[active]), axis=1)
        critical = float(np.quantile(maximum, .95, method="linear"))
    else:
        maximum = np.zeros(draws)
        critical = 0.0
    perturbed = observed_delta + noise
    eligible = perturbed[:, tie_order] >= perturbed.max(axis=1, keepdims=True) - tie_tolerance
    winners = tie_order[np.argmax(eligible, axis=1)]
    counts = np.bincount(winners, minlength=k)
    return {
        "standard_error": standard_error.tolist(),
        "simultaneous_95_critical_value": critical,
        "critical_value_estimator": "linear interpolated empirical 0.95 quantile of max(abs(Gaussian noise / SE))",
        "pointwise_95_critical_value": 1.959963984540054,
        "zero_standard_error_columns": np.flatnonzero(~active).tolist(),
        "winner_counts": counts.tolist(), "winner_frequencies": (counts / draws).tolist(),
        "winner_interpretation": f"Frequency of winner among exactly these {k} fixed grid points after adding estimated Gaussian contrast noise; not true-optimum probability. The incumbent is not a candidate in this frequency calculation.",
        "tie_priority_indices": tie_order.tolist(), "tie_tolerance": tie_tolerance,
        "draws_with_multiple_eligible_winners": int(np.count_nonzero(eligible.sum(axis=1) > 1)),
        "exact_identical_covariance_columns_forced_to_share_noise": identical_columns,
        "seed": seed, "gaussian_draws": draws,
        "gaussian_noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
        "maximum_standardized_noise_sha256": hashlib.sha256(maximum.tobytes()).hexdigest(),
        "psd_diagnostics": psd,
    }


def load_existing_grid() -> tuple[dict, dict, dict]:
    for relative, expected in PINNED.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Registered grid input changed: {relative}")
    protocol = json_read(ROOT / "configs/research_grid_uncertainty_v1.json")
    base = json_read(ROOT / protocol["base_protocol"])
    existing = json_read(ROOT / protocol["existing_grid_result"])
    grid = existing["block_weight_grid"]
    expected = [[1 - nn - xgb, nn, xgb] for nn in base["block_weight_grid"]
                for xgb in base["block_weight_grid"]]
    if len(grid) != 49 or [row["weights"] for row in grid] != expected:
        raise ValueError("Existing grid does not match exact registered 49-point order")
    if protocol["gaussian_draws"] != 20000 or protocol["seed"] != 20261003:
        raise ValueError("Unexpected simulation settings")
    return protocol, base, existing


def load_blocks(y: np.ndarray, folds: np.ndarray, evidence: dict, base: dict) -> list[np.ndarray]:
    """Independently rejoin just the two accepted v3 OOF blocks to frozen IDs."""
    split_path = ROOT / base["split_path"]
    if sha256(split_path) != base["split_sha256"]:
        raise ValueError("Frozen split changed")
    split = pd.read_parquet(split_path, columns=["id", "fold"])
    split = split.loc[split.fold >= 0].copy()
    ids = split.id.to_numpy()
    if (len(ids) != len(y) or not split.id.is_unique
            or not np.array_equal(split.fold.to_numpy(), folds)
            or hashlib.sha256(ids.tobytes()).hexdigest() != evidence["development_ids_sha256"]):
        raise ValueError("Independent block development identities differ")
    manifest = {row["id"]: row for row in evidence["members"]}
    blocks = []
    for name in base["blocks"][1:]:
        path = ROOT / "artifacts/runs" / name / "oof.parquet"
        if sha256(path) != manifest[name]["oof_sha256"]:
            raise ValueError("Accepted block OOF hash differs")
        frame = pd.read_parquet(path, columns=["id", "fold", "satisfaction", "prediction"])
        if len(frame) != len(ids) or not frame.id.is_unique:
            raise ValueError("Block OOF duplicate or missing IDs")
        order = pd.Index(frame.id).get_indexer(ids)
        if (order < 0).any():
            raise ValueError("Block OOF missing development IDs")
        frame = frame.iloc[order]
        if (not np.array_equal(frame.fold.to_numpy(), folds)
                or not np.array_equal(frame.satisfaction.to_numpy(), y)):
            raise ValueError("Block OOF fold or label mismatch")
        score = frame.prediction.to_numpy(dtype=np.float64)
        if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
            raise ValueError("Block OOF probability violation")
        blocks.append(score)
    if len(blocks) != 2:
        raise ValueError("Expected exactly two v3 blocks")
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/research_pass_v1/validation/grid_uncertainty.json")
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT / "artifacts/research_pass_v1/validation").resolve()):
        raise ValueError("Output must remain in isolated diagnostic directory")
    if destination.exists():
        raise FileExistsError("Preserve completed diagnostic output")
    with threadpool_limits(limits=2):
        protocol, base, existing = load_existing_grid()
        y, folds, vectors, evidence = verify_inputs()
        baseline, current = vectors["frozen_v2"], vectors["incumbent"]
        del vectors
        nn_add, xgb_add = load_blocks(y, folds, evidence, base)
        ref_auc, ref_pos, ref_neg = placements(y, current)
        grid = existing["block_weight_grid"]
        m, n, k = len(ref_pos), len(ref_neg), len(grid)
        # Two class matrices, one largest centered copy, placement/sort scratch,
        # retained score vectors, and Gaussian arrays; deliberately conservative.
        planned_bytes = 8 * ((m + n) * k + max(m, n) * k
                             + 40 * (m + n) + 6 * protocol["gaussian_draws"] * k)
        if planned_bytes > 2 * 1024 ** 3:
            raise MemoryError("Planned diagnostic arrays exceed 2 GiB")
        pos_diff, neg_diff = np.empty((m, k)), np.empty((n, k))
        deltas = np.empty(k)
        for index, row in enumerate(grid):
            a, b, c = row["weights"]
            score = a * baseline + b * nn_add + c * xgb_add
            auc, positive, negative = placements(y, score)
            deltas[index] = auc - ref_auc
            if (abs(auc - row["metrics"]["pooled_auc"]) > 2e-12
                    or abs(deltas[index] - row["pooled_delta"]) > 2e-12):
                raise ValueError("Existing grid AUC cannot be independently reproduced")
            pos_diff[:, index], neg_diff[:, index] = positive - ref_pos, negative - ref_neg
            print(f"Verified placements {index + 1}/{k}", flush=True)
        del score, positive, negative
        covariance = placement_covariance(pos_diff, neg_diff)
        del pos_diff, neg_diff
        incumbent_weights = np.asarray(base["current_block_weights"])
        tie_order = np.asarray(sorted(range(k), key=lambda index: (
            float(np.sum((np.asarray(grid[index]["weights"]) - incumbent_weights) ** 2)),
            grid[index]["weights"][1], grid[index]["weights"][2])))
        simulation = gaussian_summary(covariance, deltas, tie_order,
                                      seed=protocol["seed"], draws=protocol["gaussian_draws"],
                                      tie_tolerance=base["tie_tolerance"])
        rows = []
        for index, row in enumerate(grid):
            delta, se = float(deltas[index]), simulation["standard_error"][index]
            point_margin = simulation["pointwise_95_critical_value"] * se
            simultaneous_margin = simulation["simultaneous_95_critical_value"] * se
            rows.append({"index": index, "weights_v2_nn_xgb": row["weights"],
                         "pooled_auc": row["metrics"]["pooled_auc"],
                         "delta_grid_minus_incumbent": delta, "paired_standard_error": se,
                         "pointwise_descriptive_conditional_interval_95": [delta - point_margin, delta + point_margin],
                         "simultaneous_descriptive_conditional_fixed_grid_band_95": [delta - simultaneous_margin, delta + simultaneous_margin],
                         "perturbation_winner_count": simulation["winner_counts"][index],
                         "perturbation_winner_frequency": simulation["winner_frequencies"][index]})
        best = int(tie_order[np.argmax(deltas[tie_order] >= deltas.max() - base["tie_tolerance"])])
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contrast_orientation": "grid pooled AUC minus incumbent pooled AUC",
        "interval_interpretation": LIMITATION,
        "reference_pooled_auc": ref_auc, "positive_rows": m, "negative_rows": n,
        "input_verification": evidence, "registered_grid_input_hashes": PINNED,
        "all_49_existing_grid_aucs_independently_reproduced": True,
        "covariance_method": "Cov(positive placement differences)/n_positive + Cov(negative placement differences)/n_negative, ddof=1; same IDs and class-row order for every column",
        "covariance": covariance.tolist(), "simulation": simulation, "contrasts": rows,
        "observed_best_grid_index": best, "observed_best_grid_contrast": rows[best],
        "tie_rule": "Within 1e-12 of perturbed maximum, closest squared Euclidean weights to incumbent; then smaller NN weight; then smaller XGB weight, as in base protocol",
        "resource_estimate": {"planned_extra_array_bytes": planned_bytes,
                              "planned_extra_array_GiB": planned_bytes / 1024 ** 3,
                              "maximum_numerical_threads": 2, "measured_peak_rss": None},
        "gaussian_reproducibility_limit": "Exact noise hash is tied to the recorded NumPy/BLAS environment; eigenspace bases may differ across platforms.",
        "model_fits_executed": 0, "additional_grid_points_evaluated": 0,
        "release_or_incumbent_changed": False,
        "script_sha256": sha256(Path(__file__)),
        "test_script_sha256": sha256(Path(__file__).with_name("test_analysis_grid_uncertainty_v1.py")),
        "placement_loader_sha256": sha256(Path(__file__).with_name("analysis_auc_uncertainty_v1.py")),
        "invocation": [sys.executable, *sys.argv], "python": sys.version,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "scikit-learn", "threadpoolctl")},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"best": rows[best], "simultaneous_critical": simulation["simultaneous_95_critical_value"],
                      "output": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
