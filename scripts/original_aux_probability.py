"""Observed-rating probabilities from the immutable original-only auxiliary bank.

This independently implements the conditional-score hypothesis registered in
research/second_pass_post_confirmation_review.md. No fitting occurs here.
The 13-condition log sum is a composite score, not a calibrated joint likelihood.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import xgboost as xgb

from original_aux import RAW_FEATURES, RATINGS, ROOT, load_bank

EPSILON = 1e-6
OUTPUT_COLUMNS = tuple(f"orig_aux_prob_{i:02d}" for i in range(13)) + ("orig_aux_log_score",)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _runtime(batch_size: int, n_jobs: int) -> None:
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
           for v in (batch_size, n_jobs)):
        raise ValueError("batch_size and n_jobs must be positive integers")


def _lookup(probability: np.ndarray, observed: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map actual class values, never rating-as-column-index or a missing -1 index."""
    if (probability.shape != (len(observed), len(classes)) or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("Invalid auxiliary probability matrix")
    np.testing.assert_allclose(probability.sum(axis=1, dtype=np.float64), 1., rtol=1e-5, atol=1e-6)
    if classes.ndim != 1 or len(np.unique(classes)) != len(classes) or not np.isfinite(classes).all():
        raise ValueError("Invalid explicit class-value mapping")
    positions = pd.Index(classes.astype(np.float64)).get_indexer(observed)
    supported = np.isfinite(observed) & (positions >= 0)
    result = np.full(len(observed), EPSILON, dtype=np.float32)
    rows = np.flatnonzero(supported)
    result[rows] = probability[rows, positions[rows]]
    return result, {"missing_rows": int(np.isnan(observed).sum()),
                    "unseen_rows": int((~supported & ~np.isnan(observed)).sum()),
                    "supported_log_clipped_rows": int((result[supported].astype(np.float64) < EPSILON).sum())}


def _log_score(probability: np.ndarray) -> np.ndarray:
    # Fixed 13-term float64 reduction, rounded once, is independent of row batches.
    return np.log(np.maximum(probability.astype(np.float64), EPSILON)).sum(axis=1, dtype=np.float64).astype(np.float32)


def _validate_values(frame: pd.DataFrame) -> None:
    if list(frame.columns) != list(OUTPUT_COLUMNS) or any(v != np.dtype("float32") for v in frame.dtypes):
        raise ValueError("Probability-feature schema/dtype mismatch")
    values = frame.to_numpy()
    if not np.isfinite(values).all() or ((values[:, :13] < 0) | (values[:, :13] > 1)).any():
        raise ValueError("Probability-feature values are not finite/in range")
    if not np.array_equal(values[:, 13], _log_score(values[:, :13])):
        raise ValueError("Composite score differs from fixed log contract")


def _predict(raw: pd.DataFrame, directory: Path, batch_size: int, n_jobs: int,
             *, progress: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    _runtime(batch_size, n_jobs)
    directory = Path(directory)
    processor, manifest = load_bank(directory)  # Hash-checks every native artifact.
    encoded = processor.transform(raw)  # Explicit 21-feature schema, no ID/target.
    result = np.empty((len(raw), len(OUTPUT_COLUMNS)), dtype=np.float32)
    details = []
    for i, entry in enumerate(manifest["models"]):
        start = time.monotonic()
        model = xgb.Booster(model_file=directory / entry["model_file"])
        model.set_param({"device": "cpu", "nthread": int(n_jobs)})
        native = json.loads(model.save_config())["learner"]
        classes = np.asarray(entry["classes"], dtype=np.float64)
        if (model.num_features() != 20 or model.num_boosted_rounds() != entry["rounds"]
                or native["objective"]["name"] != "multi:softprob"
                or int(native["learner_model_param"]["num_class"]) != len(classes)):
            raise ValueError("Native model class count, shape, rounds or objective mismatch")
        # Use original numeric values for support lookup, before float32 predictor
        # conversion could round a near-integer unsupported value onto a class.
        observed = pd.to_numeric(raw[entry["target"]], errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
        counts = {"missing_rows": 0, "unseen_rows": 0, "supported_log_clipped_rows": 0}
        for begin in range(0, len(raw), batch_size):
            end = min(begin + batch_size, len(raw))
            block = np.ascontiguousarray(encoded[begin:end, entry["predictor_positions"]], dtype=np.float32)
            data = xgb.DMatrix(block, feature_names=[f"f{j}" for j in range(20)], nthread=int(n_jobs))
            probability = model.predict(data, strict_shape=True)
            values, batch_counts = _lookup(probability, observed[begin:end], classes)
            result[begin:end, i] = values
            for name, count in batch_counts.items():
                counts[name] += count
        detail = {"target": entry["target"], "output_column": OUTPUT_COLUMNS[i],
                  "class_values_in_native_probability_order": entry["classes"],
                  **counts, "seconds_including_load": time.monotonic() - start}
        details.append(detail)
        if progress:
            print(json.dumps({"original_aux_probability_model": i + 1, **detail}), flush=True)
        del model
    result[:, 13] = _log_score(result[:, :13])
    frame = pd.DataFrame(result, index=raw.index.copy(), columns=OUTPUT_COLUMNS)
    _validate_values(frame)
    return frame, details


def predict_aux_probability(raw: pd.DataFrame, directory: Path = ROOT / "artifacts/original_aux",
                            batch_size: int = 32768, n_jobs: int = 4) -> pd.DataFrame:
    """Same-index float32 q_00..q_12 and composite log score, using native CPU IO.

    Extra satisfaction/ID columns are ignored. Each predicted rating is excluded
    from its own model's inputs, but its observed value chooses the class column.
    Missing/unseen observed ratings get q=1e-6. Supported q is not floored; only
    log(max(q,1e-6)) is clipped. No caller rows fit vocabularies or statistics.
    Empty frames preserve the same schema. Infinite predictor values are invalid.
    """
    return _predict(raw, directory, batch_size, n_jobs)[0]


def _valid_ids(frame: pd.DataFrame) -> None:
    if "id" not in frame or frame.id.isna().any() or not frame.id.is_unique:
        raise ValueError("Cache rows require unique nonmissing IDs")


def _publish(raw: pd.DataFrame, features: pd.DataFrame, output: Path) -> str:
    _valid_ids(raw)
    _validate_values(features)
    if not raw.index.equals(features.index):
        raise ValueError("Raw/features row indexes differ")
    if output.exists():
        raise ValueError("Refusing to overwrite an existing probability-feature cache")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = features.copy()
    frame.insert(0, "id", raw.id.to_numpy())
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        restored = pd.read_parquet(temporary)
        pd.testing.assert_frame_equal(restored, frame.reset_index(drop=True), check_exact=True)
        aligned = raw[["id"]].merge(restored, how="left", on="id", sort=False, validate="one_to_one")
        if (not np.array_equal(aligned.id.to_numpy(), raw.id.to_numpy())
                or not np.array_equal(aligned[list(OUTPUT_COLUMNS)].to_numpy(), features.to_numpy())):
            raise ValueError("Keyed cache coverage/alignment mismatch")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha(output)


def _load_inputs(root: Path, bank: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hashes = {}
    frames = []
    for kind in ["train", "test"]:
        csv = root / f"data/{kind}.csv"
        parquet = root / f"data/{kind}.parquet"
        hashes[f"{kind}_csv_sha256"] = _sha(csv)
        hashes[f"{kind}_parquet_sha256"] = _sha(parquet)
        if hashes[f"{kind}_csv_sha256"] != bank["input_hashes"][f"synthetic_{kind}_csv_sha256"]:
            raise ValueError("Raw source differs from the immutable original-bank provenance")
        # Satisfaction is never requested, including during full-cache generation.
        raw = pd.read_parquet(parquet, columns=["id", *RAW_FEATURES])
        csv_ids = pd.read_csv(csv, usecols=["id"])
        _valid_ids(raw)
        if not np.array_equal(raw.id.to_numpy(), csv_ids.id.to_numpy()):
            raise ValueError("Canonical parquet/raw CSV ID order mismatch")
        frames.append(raw)
    if len(np.intersect1d(frames[0].id, frames[1].id)):
        raise ValueError("Train/test ID overlap")
    return frames[0], frames[1], hashes


def build(root: Path = ROOT, *, directory: Path | None = None, batch_size: int = 32768, n_jobs: int = 4) -> dict:
    """Build a separate immutable keyed cache; completed replays perform no writes."""
    _runtime(batch_size, n_jobs)
    start = time.monotonic()
    root = Path(root)
    directory = root / "artifacts/original_aux" if directory is None else Path(directory)
    _, bank = load_bank(directory)
    bank_manifest_hash = _sha(directory / "manifest.json")
    train, test, input_hashes = _load_inputs(root, bank)
    raw = pd.concat([train[["id", *RAW_FEATURES]], test[["id", *RAW_FEATURES]]], ignore_index=True)
    del train, test
    _valid_ids(raw)
    output = root / "data/original_aux_probability_predictions.parquet"
    metadata_path = root / "artifacts/original_aux_probability/manifest.json"
    contract = {"format": "original_aux_probability", "version": 1, "epsilon": EPSILON,
                "missing_unseen_rule": "q=epsilon", "supported_q_rule": "native observed-class probability, not clipped",
                "log_rule": "sum(log(max(q_j,epsilon))), float64 terms/reduction then float32",
                "output_columns": list(OUTPUT_COLUMNS), "output_rows": len(raw),
                "input_hashes": input_hashes, "bank_manifest_sha256": bank_manifest_hash,
                "bank_preprocessor_sha256": bank["preprocessor_sha256"],
                "native_model_sha256": [entry["model_sha256"] for entry in bank["models"]],
                "implementation_sha256": _sha(Path(__file__)),
                "original_aux_implementation_sha256": _sha(Path(__file__).with_name("original_aux.py")),
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__},
                "id_order_sha256": hashlib.sha256(raw.id.to_numpy(dtype="<i8").tobytes()).hexdigest(),
                "inference_device": "cpu", "satisfaction_read": False, "fitting_performed": False}
    if output.exists() or metadata_path.exists():
        if not output.exists() or not metadata_path.exists():
            raise ValueError("Incomplete existing cache publication; inspect it before rebuilding")
        manifest = json.loads(metadata_path.read_text())
        if manifest.get("status") != "complete" or manifest.get("contract") != contract:
            raise ValueError("Existing probability cache contract changed")
        if _sha(output) != manifest["output_sha256"]:
            raise ValueError("Existing probability cache checksum mismatch")
        cached = pd.read_parquet(output)
        if list(cached.columns) != ["id", *OUTPUT_COLUMNS] or not np.array_equal(cached.id.to_numpy(), raw.id.to_numpy()):
            raise ValueError("Completed cache identity/schema mismatch")
        _valid_ids(cached)
        _validate_values(cached[list(OUTPUT_COLUMNS)])
        print(json.dumps({"reused_complete": str(output), "writes": 0}), flush=True)
        return manifest
    values, details = _predict(raw, directory, batch_size, n_jobs, progress=True)
    # Feature-only, deterministic probe crosses the train/test boundary and changes
    # batching. No satisfaction labels or model accuracy are examined.
    positions = np.unique(np.concatenate([np.arange(min(17, len(raw))), np.arange(max(0, len(raw) - 17), len(raw))]))
    probe = predict_aux_probability(raw.iloc[positions], directory, batch_size=1, n_jobs=n_jobs)
    np.testing.assert_array_equal(probe.to_numpy(), values.iloc[positions].to_numpy())
    if _sha(directory / "manifest.json") != bank_manifest_hash:
        raise ValueError("Original auxiliary bank changed during feature generation")
    load_bank(directory)  # Recheck all original native hashes before publication.
    output_hash = _publish(raw, values, output)
    manifest = {"status": "complete", "contract": contract, "output_file": str(output.relative_to(root)),
                "output_sha256": output_hash, "created_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - start, "batch_size": batch_size, "n_jobs": n_jobs,
                "models": details, "exact_native_reload_probe_rows": len(positions),
                "exact_native_reload_probe_batch_size": 1,
                "source": {"method_note": "research/second_pass_post_confirmation_review.md",
                           "bank_attribution": bank.get("source", {}),
                           "implementation": "Independent mathematical feature implementation; no public notebook code copied"},
                "limitations": "Composite conditional score is not a calibrated joint likelihood; original-only bank may be miscalibrated under synthetic shift"}
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, metadata_path)
    print(json.dumps({"complete": str(output), "rows": len(raw), "seconds": manifest["elapsed_seconds"], "sha256": output_hash}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    build(args.root.resolve(), directory=args.directory, batch_size=args.batch_size, n_jobs=args.n_jobs)
