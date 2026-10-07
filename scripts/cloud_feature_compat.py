"""Explicit inference semantics for the frozen pandas 2.3.3 cloud recipe.

Pandas 3 astype(str) preserves missingness; pandas 2 converted float NaN to
literal "nan". Apply after features(raw, run), before categorical encoding.
This adapter never fits, reads labels, extends a vocabulary, or changes a
finite numeric/category value. It is intentionally restricted to audited stacks.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = (
    "Age", "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness", "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
)
SOURCE_VERSIONS = {"pandas": "2.3.3", "numpy": "2.0.2"}
SUPPORTED_INFERENCE_STACKS = {("2.3.3", "2.0.2"), ("3.0.6", "2.5.3")}


def cloud_numeric_category_compat(
    features: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    run: Mapping,
    source_versions: Mapping,
) -> pd.DataFrame:
    """Return a same-index/schema frame with cloud-trained numeric NaN tokens.

    ``raw`` must contain the 17 canonical float32 measurements in row order;
    ``features`` must contain their already-created ``*_category`` twins.
    Other raw columns and feature columns, including literal string categories,
    are preserved. Learned vocabularies still control finite unseen categories.
    ``source_versions`` must come from the verified cloud execution report.
    """
    if run.get("execution_backend") != "kaggle" or run.get("categorical_twins") is not True:
        raise ValueError("Compatibility is restricted to cloud categorical-twin runs")
    if any(source_versions.get(k) != v for k, v in SOURCE_VERSIONS.items()):
        raise ValueError("Unsupported cloud source pandas/NumPy platform")
    if (pd.__version__, np.__version__) not in SUPPORTED_INFERENCE_STACKS:
        raise ValueError("Unsupported inference pandas/NumPy platform; audit before extending")
    if (not features.columns.is_unique or not raw.columns.is_unique
            or len(features) != len(raw) or not features.index.equals(raw.index)):
        raise ValueError("Feature/raw row index or unique column contract differs")
    required = set(NUMERIC_COLUMNS)
    if not required.issubset(raw.columns) or not {c + "_category" for c in required}.issubset(features.columns):
        raise ValueError("Missing canonical numeric measurements or categorical twins")
    result = features.copy(deep=False)
    for column in NUMERIC_COLUMNS:
        if raw[column].dtype != np.dtype("float32"):
            raise ValueError(f"Raw cloud measurement must be float32: {column}")
        numbers = raw[column].to_numpy(copy=False)
        if np.isinf(numbers).any():
            raise ValueError(f"Infinite cloud measurement: {column}")
        twin = column + "_category"
        if pd.api.types.is_numeric_dtype(features[twin]):
            raise ValueError(f"Categorical twin is not a string column: {twin}")
        missing = np.isnan(numbers)
        finite = ~missing
        # Guard finite spelling as well as preserving it. This rejects a schema
        # or formatting drift instead of silently rewriting previously valid data.
        current = features[twin].to_numpy(dtype=object, copy=False)
        expected = numbers[finite].astype(str)
        if any(not isinstance(value, str) for value in current[finite]) or not np.array_equal(current[finite], expected):
            raise ValueError(f"Finite cloud numeric-category spelling differs: {twin}")
        if missing.any():
            absent = current[missing]
            if any(not pd.isna(value) and value != "nan" for value in absent):
                raise ValueError(f"Unexpected missing numeric-category token: {twin}")
            updated = features[twin].copy()
            updated.iloc[np.flatnonzero(missing)] = "nan"
            result[twin] = updated
    return result
