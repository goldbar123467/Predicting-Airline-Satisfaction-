"""Prepare external-only training rows, excluding all synthetic feature matches.

No competition target values are read. Exact matching treats missing values as equal,
uses every competition feature, and preserves source row order. No fitted statistics
or data-dependent imputation are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = "https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k"
TARGET = "satisfaction"
CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Normalize equivalent CSV dtypes without changing feature values."""
    out = frame.loc[:, features].copy()
    for col in features:
        if col in CATEGORICAL:
            out[col] = out[col].astype("string")
        else:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
            if np.isinf(out[col].to_numpy()).any():
                raise ValueError(f"Infinite values in {col}")
    return out


def exact_match_mask(
    source: pd.DataFrame, other: pd.DataFrame, features: list[str]
) -> np.ndarray:
    """Return source membership in other using exact multi-column equality."""
    keys = other.loc[:, features].drop_duplicates()
    indexed = source.loc[:, features].assign(_source_row=np.arange(len(source)))
    matches = indexed.merge(keys, on=features, how="inner", validate="many_to_one")
    mask = np.zeros(len(source), dtype=bool)
    mask[matches["_source_row"].to_numpy()] = True
    return mask


def clean_original(
    original: pd.DataFrame,
    synthetic_train: pd.DataFrame,
    synthetic_test: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove conflicting labels, repeated profiles, then any synthetic overlap."""
    distinct_labels = original.drop_duplicates(features + [TARGET])
    conflicts = distinct_labels.loc[
        distinct_labels.duplicated(features, keep=False), features
    ].drop_duplicates()
    conflict_mask = exact_match_mask(original, conflicts, features)
    no_conflict = original.loc[~conflict_mask].copy()
    unique = no_conflict.drop_duplicates(features, keep="first").reset_index(drop=True)
    train_match = exact_match_mask(unique, synthetic_train, features)
    test_match = exact_match_mask(unique, synthetic_test, features)
    clean = unique.loc[~(train_match | test_match)].reset_index(drop=True)
    counts = {
        "input_rows": len(original),
        "conflicting_feature_profiles": len(conflicts),
        "conflicting_label_rows_removed": int(conflict_mask.sum()),
        "same_label_duplicate_rows_removed": len(no_conflict) - len(unique),
        "unique_nonconflicting_rows": len(unique),
        "original_profiles_matching_synthetic_train": int(train_match.sum()),
        "original_profiles_matching_synthetic_test": int(test_match.sum()),
        "original_profiles_matching_both": int((train_match & test_match).sum()),
        "overlap_rows_removed": int((train_match | test_match).sum()),
        "output_rows": len(clean),
    }
    if clean.duplicated(features).any():
        raise AssertionError("Output contains duplicate feature profiles")
    if len(clean) == 0 or clean[TARGET].nunique() != 2:
        raise ValueError("Filtered original must contain both target classes")
    return clean, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    archive = root / "research/downloaded_sources/original/binary-aviation-satisfaction-129k.zip"
    train_path, test_path = root / "data/train.csv", root / "data/test.csv"
    output = root / "data/original.parquet"
    audit_path = root / "research/original_audit.json"
    features = [c for c in pd.read_csv(test_path, nrows=0).columns if c != "id"]
    if len(features) != 21 or TARGET in features:
        raise ValueError(f"Unexpected competition feature contract: {features}")
    # Deliberately exclude the synthetic target from every data read.
    train = canonical_features(pd.read_csv(train_path, usecols=features), features)
    test = canonical_features(pd.read_csv(test_path, usecols=features), features)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.namelist() != ["data.csv"]:
            raise ValueError(f"Unexpected original archive members: {zipped.namelist()}")
        with zipped.open("data.csv") as stream:
            raw = pd.read_csv(stream)
        source_csv_sha256 = hashlib.sha256(zipped.read("data.csv")).hexdigest()
    if set(raw.columns) != set(features + [TARGET]):
        raise ValueError("Original schema differs from competition features plus target")
    if not raw[TARGET].isin([True, False, 0, 1]).all():
        raise ValueError("Original labels must be Boolean or binary 0/1")
    original = canonical_features(raw, features)
    original[TARGET] = raw[TARGET].astype("int8")
    clean, counts = clean_original(original, train, test, features)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(".tmp.parquet")
    clean.to_parquet(temp_output, index=False)
    restored = pd.read_parquet(temp_output)
    pd.testing.assert_frame_equal(restored, clean)
    for name, synthetic in [("train", train), ("test", test)]:
        if exact_match_mask(restored, synthetic, features).any():
            raise AssertionError(f"Remaining exact original/{name} overlap")
    temp_output.replace(output)
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "source_license": "CC0-1.0, Kaggle metadata retrieved 2026-10-02 UTC",
        "source_csv_member": "data.csv",
        "source_csv_sha256": source_csv_sha256,
        "archive_sha256": sha256(archive),
        "synthetic_train_sha256": sha256(train_path),
        "synthetic_test_sha256": sha256(test_path),
        "output_sha256": sha256(output),
        "features": features,
        "schema": {c: str(t) for c, t in clean.dtypes.items()},
        "label_mapping": {"False": 0, "True": 1},
        "counts": counts,
        "class_counts": {str(k): int(v) for k, v in clean[TARGET].value_counts().items()},
        "missing_counts": {c: int(n) for c, n in clean.isna().sum().items() if n},
        "feature_mapping": "Exact matching column names; reorder to competition test schema; numeric float64, categoricals string; no feature value remapping",
        "procedure": "Remove all conflicting-label feature groups; retain first of equal-label duplicates; exact label-free anti-join against all synthetic train and test feature profiles; NA equals NA",
        "verification": "Parquet round-trip equality; unique profiles; both target classes; independent repeat exact anti-joins found zero overlap",
        "limitations": "Exact-profile exclusion does not rule out approximate similarity or common original ancestry; no teacher trained or model quality claimed",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "audit": str(audit_path), **counts}, indent=2))


if __name__ == "__main__":
    main()
