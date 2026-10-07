"""Train-only route means, with no target or validation/test fitting.

The S6E10 public route-profile feature idea is documented in
research/competition_evidence.md. This implementation deliberately uses only
the supplied training rows and 16 numerical measurements, not global train/test
profiles or means of arbitrarily coded categorical strings. Training profiles
may include the current row's features; they never use a label.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


DEFAULT_COLUMNS = (
    "Age", "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
    "Departure Delay in Minutes", "Arrival Delay in Minutes",
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RouteProfiles:
    """Append frozen group means while preserving original columns and row order.

    Config keys: ``key`` (default Flight Distance), ``columns`` (the fixed 16
    numerical fields above), and ``prefix`` (default route_profile_mean_).
    Output names are prefix + a stable zero-based two-digit position in columns.
    All fits require nonempty numerical feature rows. NaN measurements are
    ignored; an all-missing group/column uses its training global mean, then zero
    if that entire training column is missing. Missing/unseen routes always use
    training global means. Float32 route normalization matches raw CSV numeric
    values to the pipeline's parquet values; it intentionally merges values
    indistinguishable at that precision. Infinite/overflowing keys are rejected.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = {} if config is None else dict(config)
        unknown = set(config) - {"key", "columns", "prefix"}
        if unknown:
            raise ValueError(f"Unknown route-profile config keys: {sorted(unknown)}")
        key = config.get("key", "Flight Distance")
        columns = config.get("columns", list(DEFAULT_COLUMNS))
        prefix = config.get("prefix", "route_profile_mean_")
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a nonempty column name")
        if not isinstance(columns, (tuple, list)) or not columns:
            raise ValueError("columns must be a nonempty list of column names")
        if any(not isinstance(c, str) or not c for c in columns):
            raise ValueError("columns must contain nonempty strings")
        if len(set(columns)) != len(columns) or key in columns:
            raise ValueError("Profile columns must be unique and exclude the route key")
        if "satisfaction" in [key, *columns]:
            raise ValueError("The competition target cannot be used in route profiles")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a nonempty string")
        self.config = {"key": key, "columns": list(columns), "prefix": prefix}
        self.output_columns = [f"{prefix}{i:02d}" for i in range(len(columns))]

    def _validate_x(self, x: pd.DataFrame, *, fitting: bool = False) -> None:
        if not isinstance(x, pd.DataFrame) or not x.columns.is_unique:
            raise ValueError("x must be a DataFrame with unique column names")
        if any(not isinstance(c, str) for c in x.columns):
            raise ValueError("Input column names must be strings")
        needed = [self.config["key"], *self.config["columns"]]
        missing = set(needed) - set(x.columns)
        if missing:
            raise ValueError(f"Missing route-profile columns: {sorted(missing)}")
        if any(not pd.api.types.is_numeric_dtype(x[c]) for c in needed):
            raise TypeError("Route key and profile measurements must be numeric")
        if set(self.output_columns) & set(x.columns):
            raise ValueError("Route-profile output columns already exist in x")
        if fitting and not len(x):
            raise ValueError("Cannot fit route profiles on zero rows")
        if not fitting and list(x.columns) != self.input_columns_:
            raise ValueError("Input columns or order differ from the fitted schema")

    def _keys(self, x: pd.DataFrame) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            keys = x[self.config["key"]].to_numpy(dtype=np.float32, na_value=np.nan)
        if np.isinf(keys).any():
            raise ValueError("Route key contains infinity or float32 overflow")
        return keys

    def fit(self, x: pd.DataFrame) -> "RouteProfiles":
        """Learn only from x; this method intentionally has no target argument."""
        self._validate_x(x, fitting=True)
        keys = self._keys(x)
        values = x[self.config["columns"]].to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError("Profile measurements cannot contain infinity")
        global_means = np.array([
            float(column[~np.isnan(column)].mean()) if np.any(~np.isnan(column)) else 0.0
            for column in values.T
        ], dtype=np.float64)
        grouped = pd.DataFrame(values, columns=self.output_columns)
        grouped.insert(0, "key", keys)
        table = grouped.groupby("key", sort=True, dropna=True)[self.output_columns].mean()
        for i, column in enumerate(self.output_columns):
            table[column] = table[column].fillna(global_means[i])
        if not np.isfinite(global_means).all() or not np.isfinite(table.to_numpy()).all():
            raise FloatingPointError("Route means overflowed")
        with np.errstate(over="ignore"):
            means32 = global_means.astype(np.float32)
            table32 = table.astype(np.float32)
        if not np.isfinite(means32).all() or not np.isfinite(table32.to_numpy()).all():
            raise FloatingPointError("Route means exceed float32 precision range")
        self.global_means_ = means32
        self.map_ = table32
        self.input_columns_ = list(x.columns)
        self.training_rows_ = len(x)
        return self

    def fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return self.fit(x).transform(x)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "map_"):
            raise RuntimeError("RouteProfiles must be fitted or loaded before transform")
        self._validate_x(x)
        keys = self._keys(x)
        positions = self.map_.index.get_indexer(keys)
        known = positions >= 0
        means = np.broadcast_to(self.global_means_, (len(x), len(self.output_columns))).copy()
        means[known] = self.map_.to_numpy(dtype=np.float32)[positions[known]]
        result = x.copy(deep=True)
        for i, column in enumerate(self.output_columns):
            # ndarray assignment is positional, even for duplicate input indices.
            result[column] = means[:, i]
        return result

    def save(self, directory: str | Path) -> None:
        """Atomically publish JSON metadata pointing to a hashed native map."""
        if not hasattr(self, "map_"):
            raise RuntimeError("Cannot save unfitted RouteProfiles")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"map-{uuid4().hex}.parquet"
        final = directory / filename
        temporary = directory / f".{filename}.tmp"
        try:
            self.map_.reset_index().to_parquet(temporary, index=False)
            os.replace(temporary, final)
        finally:
            temporary.unlink(missing_ok=True)
        _atomic_json(directory / "metadata.json", {
            "format": "route_profiles", "version": 1, "config": self.config,
            "input_columns": self.input_columns_, "output_columns": self.output_columns,
            "key_dtype": "float32", "mean_dtype": "float32",
            "global_means": self.global_means_.tolist(), "training_rows": self.training_rows_,
            "route_count": len(self.map_), "map_file": filename, "map_sha256": _digest(final),
            "fit_scope": "supplied training feature rows only; no labels",
            "missing_route": "training global mean", "all_missing_measurement": 0.0,
        })

    @classmethod
    def load(cls, directory: str | Path) -> "RouteProfiles":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "route_profiles" or metadata.get("version") != 1:
            raise ValueError("Unsupported route-profile artifact format")
        obj = cls(metadata["config"])
        if metadata["output_columns"] != obj.output_columns:
            raise ValueError("Route-profile output schema mismatch")
        if metadata.get("key_dtype") != "float32" or metadata.get("mean_dtype") != "float32":
            raise ValueError("Unsupported route-profile numeric precision")
        filename = metadata["map_file"]
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".parquet"):
            raise ValueError("Invalid route-profile map filename")
        path = directory / filename
        if _digest(path) != metadata["map_sha256"]:
            raise ValueError("Route-profile map checksum mismatch")
        table = pd.read_parquet(path)
        if list(table.columns) != ["key", *obj.output_columns] or len(table) != metadata["route_count"]:
            raise ValueError("Route-profile map schema or row count mismatch")
        if not all(dtype == np.dtype("float32") for dtype in table.dtypes):
            raise ValueError("Route-profile map dtype mismatch")
        if not np.isfinite(table.to_numpy()).all() or table["key"].duplicated().any():
            raise ValueError("Route-profile map contains nonfinite values or duplicate keys")
        obj.global_means_ = np.asarray(metadata["global_means"], dtype=np.float32)
        if obj.global_means_.shape != (len(obj.output_columns),) or not np.isfinite(obj.global_means_).all():
            raise ValueError("Invalid route-profile global means")
        obj.input_columns_ = metadata["input_columns"]
        needed = {obj.config["key"], *obj.config["columns"]}
        if (not isinstance(obj.input_columns_, list) or
                any(not isinstance(c, str) for c in obj.input_columns_) or
                len(set(obj.input_columns_)) != len(obj.input_columns_) or
                not needed.issubset(obj.input_columns_) or
                set(obj.output_columns) & set(obj.input_columns_)):
            raise ValueError("Invalid route-profile input schema")
        obj.training_rows_ = metadata["training_rows"]
        if not isinstance(obj.training_rows_, int) or obj.training_rows_ <= 0:
            raise ValueError("Invalid route-profile training row count")
        obj.map_ = table.set_index("key")
        return obj
