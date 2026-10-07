"""Fold-local route target encoding and label-free counts for tabular boosters.

Recipe attribution: goodpjw2008's S6E10 route/original-model notebook, documented in
research/competition_evidence.md. This is a new local implementation, not copied
notebook code. Every target statistic uses only the supplied training partition.
Training rows get shuffled-KFold cross-fitted encodings; held-out/test rows get the
full supplied-training maps. This intentionally creates a small train/test mapping
sample-size difference. Counts are label-free and may include each training row.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class RouteEncoder:
    """Append route target statistics and optional counts to a writable copy.

    Configuration:
      keys: strings or lists of columns, default ['Flight Distance']; for example
            ['Flight Distance', ['Flight Distance', 'Age'], 'Age'].
      smoothing: nonnegative prior strength, default 20.
      n_splits: target-encoding KFold count, default 5.
      seed: frozen fold seed, default 20261001. Splits never depend on y.
      counts: append one label-free count per key, default True.

    Numeric key columns normalize to float32 so integer and float32 inputs agree.
    This matches the airline feature precision; callers with higher-precision keys
    must not use this encoder without revisiting that contract. Categorical strings
    and missing values have distinct typed tokens. Composite keys are JSON tuples,
    so embedded delimiter characters cannot create accidental key collisions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        raw_keys = config.get("keys", ["Flight Distance"])
        if not isinstance(raw_keys, (list, tuple)) or not raw_keys:
            raise ValueError("keys must be a nonempty list of column names or column lists")
        keys = []
        for key in raw_keys:
            columns = (key,) if isinstance(key, str) else tuple(key)
            if not columns or any(not isinstance(c, str) or not c for c in columns):
                raise ValueError("Each encoding key must contain nonempty column names")
            if len(set(columns)) != len(columns):
                raise ValueError("An encoding key must not repeat a column")
            if "satisfaction" in columns:
                raise ValueError("The target cannot be part of an encoding key")
            if columns in keys:
                raise ValueError("Encoding keys must be unique")
            keys.append(columns)
        smoothing = float(config.get("smoothing", 20.0))
        if not math.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be finite and nonnegative")
        counts = config.get("counts", True)
        if not isinstance(counts, bool):
            raise TypeError("counts must be a boolean")
        self.keys = keys
        self.config = {
            "keys": [list(key) for key in keys],
            "smoothing": smoothing,
            "n_splits": _positive_integer(config.get("n_splits", 5), "n_splits", 2),
            "seed": int(config.get("seed", 20261001)),
            "counts": counts,
        }
        self.required_columns = list(dict.fromkeys(c for key in keys for c in key))
        self.te_columns = [f"route_te_{i}" for i in range(len(keys))]
        self.count_columns = [f"route_count_{i}" for i in range(len(keys))] if counts else []

    def _validate_x(self, x: pd.DataFrame) -> None:
        if not isinstance(x, pd.DataFrame) or not x.columns.is_unique:
            raise ValueError("x must be a DataFrame with unique column names")
        missing = set(self.required_columns) - set(x.columns)
        if missing:
            raise ValueError(f"Missing encoding key columns: {sorted(missing)}")
        collisions = set(self.te_columns + self.count_columns) & set(x.columns)
        if collisions:
            raise ValueError(f"Refusing to overwrite existing columns: {sorted(collisions)}")

    @staticmethod
    def _tokens(series: pd.Series, kind: str) -> np.ndarray:
        if kind == "numeric_float32":
            values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float32, na_value=np.nan)
            if np.isinf(values).any():
                raise ValueError(f"Key column {series.name!r} contains infinity or float32 overflow")
            codes, unique = pd.factorize(values, sort=False)
            tokens = [f"n:{float(value) if value != 0 else 0.0}" for value in unique]
        elif kind == "categorical_string":
            values = series.astype("string")
            codes, unique = pd.factorize(values, sort=False)
            tokens = [f"s:{value}" for value in unique]
        else:
            raise ValueError(f"Unknown encoding key kind: {kind}")
        # The dedicated missing token cannot collide with n: or s: tokens.
        vocabulary = np.asarray([*tokens, "m:"], dtype=object)
        return vocabulary[np.where(codes < 0, len(tokens), codes)]

    def _keys(self, x: pd.DataFrame) -> list[np.ndarray]:
        columns = {c: self._tokens(x[c], self.column_kinds_[c]) for c in self.required_columns}
        result = []
        for key in self.keys:
            combinations = pd.MultiIndex.from_arrays([columns[c] for c in key])
            codes, unique = pd.factorize(combinations, sort=False)
            encoded = np.asarray(
                [json.dumps(list(parts), ensure_ascii=True, separators=(",", ":")) for parts in unique],
                dtype=object,
            )
            result.append(encoded[codes])
        return result

    def _map(self, keys: np.ndarray, labels: np.ndarray, prior: float) -> pd.DataFrame:
        grouped = pd.DataFrame({"key": keys, "target": labels}).groupby("key", sort=False)["target"]
        table = grouped.agg(["count", "sum"])
        table["value"] = (table["sum"] + self.config["smoothing"] * prior) / (
            table["count"] + self.config["smoothing"]
        )
        return table[["value", "count"]]

    @staticmethod
    def _lookup(keys: np.ndarray, table: pd.DataFrame, prior: float) -> np.ndarray:
        return pd.Series(keys).map(table["value"]).fillna(prior).to_numpy(dtype=np.float32)

    def fit_transform(self, x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit inference maps and return cross-fitted training features by position."""
        self._validate_x(x)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(x):
            raise ValueError("y must have one label per x row, in the same positional order")
        if isinstance(y, pd.Series) and not y.index.equals(x.index):
            raise ValueError("A Series target must have exactly the same index as x")
        if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
            raise ValueError("y must contain both binary classes encoded 0 and 1")
        if len(x) < self.config["n_splits"]:
            raise ValueError("There are fewer training rows than cross-fitting folds")
        labels = labels.astype(np.float64, copy=False)
        self.column_kinds_ = {
            c: "numeric_float32" if pd.api.types.is_numeric_dtype(x[c]) else "categorical_string"
            for c in self.required_columns
        }
        key_arrays = self._keys(x)
        self.prior_ = float(labels.mean())
        self.maps_ = [self._map(keys, labels, self.prior_) for keys in key_arrays]
        self.fold_ids_ = np.full(len(x), -1, dtype=np.int32)
        encoded = np.full((len(x), len(self.keys)), np.nan, dtype=np.float32)
        fold_priors = []
        splitter = KFold(n_splits=self.config["n_splits"], shuffle=True, random_state=self.config["seed"])
        for fold, (training, held_out) in enumerate(splitter.split(np.arange(len(x)))):
            if np.intersect1d(training, held_out).size or len(training) + len(held_out) != len(x):
                raise RuntimeError("Cross-fitting fold does not partition the supplied training rows")
            if (self.fold_ids_[held_out] != -1).any():
                raise RuntimeError("Cross-fitting folds contain overlapping held-out rows")
            self.fold_ids_[held_out] = fold
            prior = float(labels[training].mean())
            fold_priors.append(prior)
            for j, keys in enumerate(key_arrays):
                mapping = self._map(keys[training], labels[training], prior)
                encoded[held_out, j] = self._lookup(keys[held_out], mapping, prior)
        if (self.fold_ids_ < 0).any() or not np.isfinite(encoded).all():
            raise RuntimeError("Cross-fitting left rows without finite target encodings")
        self.fit_metadata_ = {
            "training_rows": len(x),
            "class_counts": [int((labels == 0).sum()), int((labels == 1).sum())],
            "fold_method": "shuffled_KFold_independent_of_y",
            "fold_sizes": np.bincount(self.fold_ids_).tolist(),
            "fold_priors": fold_priors,
            "fold_ids_sha256": hashlib.sha256(self.fold_ids_.astype("<i4").tobytes()).hexdigest(),
            "training_target_encoding": "cross-fitted; mapping and prior exclude held-out fold targets",
            "training_counts": "label-free counts from full supplied training x; may include self",
            "inference_target_encoding": "full supplied training map; unknown keys use full-training prior",
        }
        return self._append(x, encoded, key_arrays)

    def _append(self, x: pd.DataFrame, encoded: np.ndarray, key_arrays: list[np.ndarray]) -> pd.DataFrame:
        result = x.copy(deep=True)
        for j, name in enumerate(self.te_columns):
            result[name] = encoded[:, j]
        for j, name in enumerate(self.count_columns):
            result[name] = pd.Series(key_arrays[j]).map(self.maps_[j]["count"]).fillna(0).to_numpy(dtype=np.float32)
        values = result[self.te_columns + self.count_columns].to_numpy()
        if not np.isfinite(values).all() or ((encoded < 0) | (encoded > 1)).any():
            raise FloatingPointError("Route encoding produced invalid numerical features")
        if not result.index.equals(x.index) or len(result) != len(x):
            raise RuntimeError("Route encoding changed row identity or order")
        return result

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "maps_"):
            raise RuntimeError("RouteEncoder must be fitted or loaded before transform")
        self._validate_x(x)
        keys = self._keys(x)
        encoded = np.column_stack([self._lookup(k, m, self.prior_) for k, m in zip(keys, self.maps_, strict=True)])
        return self._append(x, encoded, keys)

    def save(self, path: Path) -> None:
        """Save a directory of parquet maps and an atomically published JSON manifest."""
        if not hasattr(self, "maps_"):
            raise RuntimeError("Cannot save an unfitted RouteEncoder")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        files = []
        for j, table in enumerate(self.maps_):
            filename = f"map-{j:02d}-{token}.parquet"
            final = directory / filename
            temporary = directory / f".{filename}.tmp"
            try:
                table.rename_axis("key").reset_index().to_parquet(temporary, index=False)
                os.replace(temporary, final)
            finally:
                temporary.unlink(missing_ok=True)
            files.append({"file": filename, "sha256": _digest(final), "rows": len(table)})
        _atomic_json(
            directory / "metadata.json",
            {
                "schema_version": 1,
                "encoder": "route_crossfit_target_count",
                "config": self.config,
                "prior": self.prior_,
                "column_kinds": self.column_kinds_,
                "te_columns": self.te_columns,
                "count_columns": self.count_columns,
                "fit_metadata": self.fit_metadata_,
                "maps": files,
                "versions": {"numpy": np.__version__, "pandas": pd.__version__},
            },
        )

    @classmethod
    def load(cls, path: Path) -> "RouteEncoder":
        directory = Path(path)
        with (directory / "metadata.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("schema_version") != 1 or metadata.get("encoder") != "route_crossfit_target_count":
            raise ValueError("Unsupported route encoding artifact")
        result = cls(metadata["config"])
        if len(metadata["maps"]) != len(result.keys):
            raise ValueError("Route encoding manifest has an inconsistent map count")
        result.prior_ = float(metadata["prior"])
        if not math.isfinite(result.prior_) or not 0 <= result.prior_ <= 1:
            raise ValueError("Route encoding manifest has an invalid prior")
        result.column_kinds_ = metadata["column_kinds"]
        if set(result.column_kinds_) != set(result.required_columns):
            raise ValueError("Route encoding manifest has inconsistent key columns")
        result.fit_metadata_ = metadata["fit_metadata"]
        result.maps_ = []
        for item in metadata["maps"]:
            filename = item["file"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Route encoding manifest contains an invalid map filename")
            file = directory / filename
            if _digest(file) != item["sha256"]:
                raise ValueError("Route encoding map hash mismatch")
            table = pd.read_parquet(file)
            if list(table.columns) != ["key", "value", "count"] or len(table) != item["rows"]:
                raise ValueError("Route encoding map schema/row count mismatch")
            if not table.key.is_unique or table.key.isna().any():
                raise ValueError("Route encoding map keys are missing or duplicated")
            if (
                not np.isfinite(table[["value", "count"]].to_numpy()).all()
                or not table.value.between(0, 1).all()
                or not (table["count"] > 0).all()
            ):
                raise ValueError("Route encoding map contains invalid statistics")
            result.maps_.append(table.set_index("key"))
        return result
