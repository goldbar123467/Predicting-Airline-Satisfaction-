"""Synthetic dataflow proof of fixed-OOF meta splitting versus full nesting.

No competition files, trained competition models or evaluation metrics are read.
This toy construction demonstrates a label-dependence path, not the sign or
magnitude of selection bias in any real ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_pass_v1/validation/nesting"


@dataclass(frozen=True)
class ToyData:
    ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    outer_fold: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.ids)
        if (self.ids.ndim != 1 or self.x.shape != (n, 2) or self.y.shape != (n,)
                or self.outer_fold.shape != (n,) or len(np.unique(self.ids)) != n
                or not np.isfinite(self.x).all() or not np.isin(self.y, [0, 1]).all()
                or set(self.outer_fold) != {0, 1, 2}):
            raise ValueError("Require unique IDs, finite (n,2) features and binary aligned labels")

    def positions(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids)
        if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("Requested IDs must be a nonempty unique vector")
        lookup = {int(value): index for index, value in enumerate(self.ids)}
        try:
            return np.asarray([lookup[int(value)] for value in ids], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("Requested ID is absent from generated data") from exc


@dataclass(frozen=True)
class MeanScoreModel:
    training_ids: tuple[int, ...]
    preprocessing_ids: tuple[int, ...]
    feature_mean: np.ndarray
    label_mean: float

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
            raise ValueError("Expected finite (n,2) query features")
        # Fixed coefficients are deliberately not optimized. Fitted means expose
        # label and preprocessing dependencies in an otherwise trivial learner.
        return 0.2 + 0.6 * self.label_mean + (x - self.feature_mean) @ np.array([.07, -.03])


def fit_base(data: ToyData, train_ids: np.ndarray,
             forbidden_fit_ids: frozenset[int] = frozenset()) -> MeanScoreModel:
    """Only permitted train rows fit either the preprocessor or label statistic."""
    positions = data.positions(train_ids)
    provenance = tuple(int(value) for value in train_ids)
    if set(provenance).intersection(forbidden_fit_ids):
        raise ValueError("Protected outer IDs entered fitting")
    return MeanScoreModel(provenance, provenance, data.x[positions].mean(axis=0),
                          float(data.y[positions].mean()))


@dataclass(frozen=True)
class FeatureTable:
    ids: np.ndarray
    values: np.ndarray
    source_training_ids: tuple[tuple[int, ...], ...]
    source_feature_means: np.ndarray

    def positions(self, ids: np.ndarray) -> np.ndarray:
        lookup = {int(value): index for index, value in enumerate(self.ids)}
        if len(lookup) != len(self.ids):
            raise ValueError("Duplicated feature-table IDs")
        try:
            return np.asarray([lookup[int(value)] for value in ids], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("Missing feature-table ID") from exc


def cross_fitted_features(data: ToyData, pool_ids: np.ndarray, folds: np.ndarray,
                         forbidden_fit_ids: frozenset[int] = frozenset()) -> FeatureTable:
    """Generate OOF features inside exactly the supplied pool of permitted IDs."""
    positions = data.positions(pool_ids)
    folds = np.asarray(folds)
    if folds.shape != pool_ids.shape or len(np.unique(folds)) < 2:
        raise ValueError("At least two aligned feature-generation folds required")
    values = np.full(len(pool_ids), np.nan)
    means = np.full((len(pool_ids), 2), np.nan)
    sources: list[tuple[int, ...] | None] = [None] * len(pool_ids)
    for fold in np.unique(folds):
        held = np.flatnonzero(folds == fold)
        training_ids = pool_ids[folds != fold]
        model = fit_base(data, training_ids, forbidden_fit_ids)
        values[held] = model.score(data.x[positions[held]])
        means[held] = model.feature_mean
        for index in held:
            sources[index] = model.training_ids
    if not np.isfinite(values).all() or any(source is None for source in sources):
        raise ValueError("Incomplete generated feature coverage")
    return FeatureTable(pool_ids.copy(), values, tuple(sources), means)


@dataclass(frozen=True)
class MetaOffset:
    training_ids: tuple[int, ...]
    offset: float

    def predict(self, base_scores: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(base_scores) + self.offset, 0.0, 1.0)


def fit_meta(data: ToyData, table: FeatureTable, train_ids: np.ndarray,
             forbidden_fit_ids: frozenset[int]) -> MetaOffset:
    """Fit one constant correction; this is a dependency probe, not an AUC optimizer."""
    if set(int(value) for value in train_ids).intersection(forbidden_fit_ids):
        raise ValueError("Protected outer labels entered meta fitting")
    positions, feature_positions = data.positions(train_ids), table.positions(train_ids)
    return MetaOffset(tuple(int(value) for value in train_ids),
                      float(np.mean(data.y[positions] - table.values[feature_positions])))


@dataclass(frozen=True)
class Procedure:
    training_ids: np.ndarray
    heldout_ids: np.ndarray
    meta_features: np.ndarray
    meta_feature_means: np.ndarray
    meta_feature_sources: tuple[tuple[int, ...], ...]
    meta_model: MetaOffset
    outer_model: MeanScoreModel
    heldout_base_scores: np.ndarray
    heldout_predictions: np.ndarray


def partition(data: ToyData, heldout_fold: int) -> tuple[np.ndarray, np.ndarray, frozenset[int]]:
    if heldout_fold not in {0, 1, 2}:
        raise ValueError("Unknown outer fold")
    training, heldout = data.ids[data.outer_fold != heldout_fold], data.ids[data.outer_fold == heldout_fold]
    return training, heldout, frozenset(int(value) for value in heldout)


def split_existing_oof(data: ToyData, heldout_fold: int) -> Procedure:
    """Ordinary row-held-out base scores, then a new split of that fixed table.

    Intentionally demonstrates the dependency problem: no protected-ID filter
    can be imposed while generating the old global OOF table.
    """
    training, heldout, forbidden = partition(data, heldout_fold)
    table = cross_fitted_features(data, data.ids, data.outer_fold)
    train_positions, held_positions = table.positions(training), table.positions(heldout)
    meta = fit_meta(data, table, training, forbidden)
    outer_model = fit_base(data, training, forbidden)
    heldout_scores = table.values[held_positions]
    return Procedure(training, heldout, table.values[train_positions], table.source_feature_means[train_positions],
                     tuple(table.source_training_ids[index] for index in train_positions),
                     meta, outer_model, heldout_scores, meta.predict(heldout_scores))


def regenerate_nested(data: ToyData, heldout_fold: int) -> Procedure:
    """Meta features regenerated using only outer-training IDs, then outer refit."""
    training, heldout, forbidden = partition(data, heldout_fold)
    # Stable generated inner partition keyed by sorted IDs, independent of labels
    # and input-row ordering. Two inner folds suffice for this tiny proof.
    inner_folds = np.argsort(np.argsort(training, kind="stable"), kind="stable") % 2
    table = cross_fitted_features(data, training, inner_folds, forbidden)
    meta = fit_meta(data, table, training, forbidden)
    outer_model = fit_base(data, training, forbidden)
    heldout_scores = outer_model.score(data.x[data.positions(heldout)])
    return Procedure(training, heldout, table.values, table.source_feature_means,
                     table.source_training_ids, meta, outer_model,
                     heldout_scores, meta.predict(heldout_scores))


def generated_data() -> ToyData:
    row = np.arange(36)
    folds = row % 3
    x = np.column_stack([np.linspace(-1, 1, len(row)), np.sin(row / 3.0)])
    y = (row % 4 < 2).astype(np.int64)
    y[folds == 0] = 0  # Deterministic one-sided perturbation has nonzero effect.
    return ToyData(1000 + 7 * row, x, y, folds)


def perturb_labels(data: ToyData, heldout_fold: int) -> ToyData:
    labels = data.y.copy()
    labels[data.outer_fold == heldout_fold] = 1
    return ToyData(data.ids.copy(), data.x.copy(), labels, data.outer_fold.copy())


def perturb_features(data: ToyData, heldout_fold: int) -> ToyData:
    features = data.x.copy()
    features[data.outer_fold == heldout_fold, 0] += 1.0
    return ToyData(data.ids.copy(), features, data.y.copy(), data.outer_fold.copy())


def maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def run_proof() -> dict:
    data = generated_data()
    label_changed, feature_changed = perturb_labels(data, 0), perturb_features(data, 0)
    old, old_label, old_feature = (split_existing_oof(item, 0) for item in (data, label_changed, feature_changed))
    nested, nested_label, nested_feature = (regenerate_nested(item, 0) for item in (data, label_changed, feature_changed))
    forbidden = set(nested.heldout_ids)
    original_oof = cross_fitted_features(data, data.ids, data.outer_fold)
    invariants = {
        "ordinary_oof_excludes_each_own_row": all(int(row) not in fit_ids for row, fit_ids in
                                                    zip(original_oof.ids, original_oof.source_training_ids)),
        "nested_all_meta_base_fit_ids_exclude_outer_ids": all(not forbidden.intersection(ids)
                                                               for ids in nested.meta_feature_sources),
        "nested_outer_refit_ids_exclude_outer_ids": not forbidden.intersection(nested.outer_model.training_ids),
        "nested_preprocessing_ids_exclude_outer_ids": not forbidden.intersection(nested.outer_model.preprocessing_ids),
        "nested_meta_training_ids_exclude_outer_ids": not forbidden.intersection(nested.meta_model.training_ids),
    }
    deltas = {
        "heldout_label_perturbation_old_meta_features": maximum_difference(old.meta_features, old_label.meta_features),
        "heldout_label_perturbation_old_predictions": maximum_difference(old.heldout_predictions, old_label.heldout_predictions),
        "heldout_label_perturbation_old_heldout_base_scores": maximum_difference(old.heldout_base_scores, old_label.heldout_base_scores),
        "heldout_label_perturbation_nested_meta_features": maximum_difference(nested.meta_features, nested_label.meta_features),
        "heldout_label_perturbation_nested_predictions": maximum_difference(nested.heldout_predictions, nested_label.heldout_predictions),
        "heldout_feature_perturbation_old_meta_preprocessing": maximum_difference(old.meta_feature_means, old_feature.meta_feature_means),
        "heldout_feature_perturbation_nested_meta_preprocessing": maximum_difference(nested.meta_feature_means, nested_feature.meta_feature_means),
        "heldout_feature_perturbation_nested_meta_features": maximum_difference(nested.meta_features, nested_feature.meta_features),
        "heldout_feature_perturbation_nested_outer_preprocessing": maximum_difference(nested.outer_model.feature_mean, nested_feature.outer_model.feature_mean),
        "heldout_feature_perturbation_nested_predictions_legitimate_query_effect": maximum_difference(nested.heldout_predictions, nested_feature.heldout_predictions),
        "in_sample_full_outer_refit_scores_vs_nested_meta_oof": maximum_difference(
            nested.outer_model.score(data.x[data.positions(nested.training_ids)]), nested.meta_features),
    }
    if not all(invariants.values()):
        raise AssertionError("Nesting lineage invariant failed")
    for key in ("heldout_label_perturbation_nested_meta_features", "heldout_label_perturbation_nested_predictions",
                "heldout_feature_perturbation_nested_meta_preprocessing", "heldout_feature_perturbation_nested_meta_features",
                "heldout_feature_perturbation_nested_outer_preprocessing", "heldout_label_perturbation_old_heldout_base_scores"):
        if deltas[key] != 0.0:
            raise AssertionError(f"Expected exact invariance: {key}")
    for key in ("heldout_label_perturbation_old_meta_features", "heldout_label_perturbation_old_predictions",
                "heldout_feature_perturbation_old_meta_preprocessing", "in_sample_full_outer_refit_scores_vs_nested_meta_oof"):
        if deltas[key] <= 0:
            raise AssertionError(f"Expected visible dependency: {key}")
    return {
        "synthetic_rows": len(data.ids), "heldout_fold": 0, "heldout_rows": len(nested.heldout_ids),
        "invariants": invariants, "maximum_absolute_changes": deltas,
        "old_meta_rows_whose_feature_model_used_outer_ids": sum(bool(forbidden.intersection(ids)) for ids in old.meta_feature_sources),
        "nested_meta_rows_whose_feature_model_used_outer_ids": sum(bool(forbidden.intersection(ids)) for ids in nested.meta_feature_sources),
        "generated_data": {"ids": data.ids.tolist(), "x": data.x.tolist(), "y": data.y.tolist(), "outer_fold": data.outer_fold.tolist()},
        "old_meta_offset_before_after_label_perturbation": [old.meta_model.offset, old_label.meta_model.offset],
        "nested_meta_offset_before_after_label_perturbation": [nested.meta_model.offset, nested_label.meta_model.offset],
        "no_performance_metric_computed": True,
        "claim": "Existence of a dependency path; neither sign nor magnitude of real-world performance bias is estimated",
    }


def main() -> None:
    output = OUT / "proof.json"
    if output.exists():
        raise FileExistsError("Preserve completed synthetic proof")
    result = run_proof()
    source = Path(__file__)
    boundary_specs = [
        (fit_base, "Every learned preprocessing statistic and base-model fit receives only allowed training IDs"),
        (cross_fitted_features, "Generate meta-training features entirely within outer-training population; own-row OOF exclusion alone is insufficient"),
        (fit_meta, "Meta labels and learned weights exclude protected outer IDs, including transitive base-feature lineage"),
        (regenerate_nested, "Refit serving base model on outer-training rows only; its in-sample training predictions cannot replace inner OOF meta features"),
    ]
    result.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                   "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "test_source_sha256": hashlib.sha256(source.with_name("test_research_nested_blend_contract_v1.py").read_bytes()).hexdigest(),
                   "python": sys.version, "numpy": np.__version__, "invocation": [sys.executable, *sys.argv],
                   "source_boundaries": [{"function": function.__name__, "line": inspect.getsourcelines(function)[1], "contract": contract}
                                         for function, contract in boundary_specs],
                   "competition_files_read": False, "production_modified": False})
    OUT.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("invariants", "maximum_absolute_changes", "claim")}, indent=2))


if __name__ == "__main__":
    main()
