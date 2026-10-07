"""Original-only rating predictors yielding 13 expected-rating features.

Recipe attribution: goodpjw2008, S6E10 RealMLP + Aux-Task Features, inspected
2026-10-02; research/second_pass_competition.md records source and limitations.
This is an independent implementation, not copied notebook code. Satisfaction
is never loaded or used, even from the original table. Each auxiliary model
predicts one rating using the other 20 raw features. Models use a fixed seed and
round count, with no competition-label tuning or evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = ("Gender", "Customer Type", "Type of Travel", "Class")
RATINGS = (
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness",
)
RAW_FEATURES = (
    "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance", *RATINGS,
    "Departure Delay in Minutes", "Arrival Delay in Minutes",
)
OUTPUT_COLUMNS = tuple(f"orig_aux_{i:02d}" for i in range(len(RATINGS)))
SOURCE_URL = "https://www.kaggle.com/code/goodpjw2008/s6e10-realmlp-aux-task-features-lb-0-96141"
SOURCE_HASH = "60055ac61cc907c87ee5045083e40aeea14aa5b55218c98567697f0dbcd79931"
FIXED_PARAMS = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 1,
                "reg_lambda": 1.0, "max_bin": 256, "tree_method": "hist", "random_state": 0}


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path, value: dict[str, Any]) -> None:
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_file(directory: Path, filename: str) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Artifact filenames must be local leaf names")
    return directory / filename


class AuxiliaryPreprocessor:
    """Original-fitted category codes; numeric float32 with native NaN missingness."""

    def __init__(self, vocabulary: dict[str, list[str]]) -> None:
        if set(vocabulary) != set(CATEGORICAL):
            raise ValueError("Auxiliary category schema mismatch")
        for values in vocabulary.values():
            if (not isinstance(values, list) or any(not isinstance(v, str) for v in values)
                    or sorted(set(values)) != values):
                raise ValueError("Category vocabularies must be sorted unique string lists")
        self.vocabulary = vocabulary

    @classmethod
    def fit(cls, original: pd.DataFrame) -> "AuxiliaryPreprocessor":
        cls._validate(original)
        if not len(original):
            raise ValueError("Cannot fit on zero original rows")
        return cls({c: sorted(original[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL})

    @staticmethod
    def _validate(raw: pd.DataFrame) -> None:
        if not isinstance(raw, pd.DataFrame) or not raw.columns.is_unique:
            raise ValueError("Raw features must be a DataFrame with unique column names")
        absent = set(RAW_FEATURES) - set(raw.columns)
        if absent:
            raise ValueError(f"Missing raw auxiliary features: {sorted(absent)}")

    def transform(self, raw: pd.DataFrame) -> np.ndarray:
        self._validate(raw)
        encoded = np.empty((len(raw), len(RAW_FEATURES)), dtype=np.float32)
        for j, column in enumerate(RAW_FEATURES):
            if column in CATEGORICAL:
                # Missing and previously unseen strings share the reserved -1 code.
                mapping = {value: number for number, value in enumerate(self.vocabulary[column])}
                encoded[:, j] = raw[column].astype("string").map(mapping).fillna(-1).to_numpy(
                    dtype=np.float32)
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    encoded[:, j] = pd.to_numeric(raw[column], errors="raise").to_numpy(
                        dtype=np.float32, na_value=np.nan)
                if np.isinf(encoded[:, j]).any():
                    raise ValueError(f"Infinite or overflowing numeric predictor {column!r}")
        return encoded

    def payload(self) -> dict:
        return {"format": "original_aux_preprocessor", "version": 1,
                "features": list(RAW_FEATURES), "vocabulary": self.vocabulary,
                "numeric_dtype": "float32", "unknown_category_code": -1,
                "numeric_missing": "NaN, handled by XGBoost; no learned imputation"}

    @classmethod
    def from_payload(cls, data: dict) -> "AuxiliaryPreprocessor":
        if (data.get("format") != "original_aux_preprocessor" or data.get("version") != 1
                or data.get("features") != list(RAW_FEATURES)
                or data.get("numeric_dtype") != "float32" or data.get("unknown_category_code") != -1):
            raise ValueError("Unsupported auxiliary preprocessing contract")
        return cls(data["vocabulary"])


def _expected(model: xgb.Booster, encoded: np.ndarray, predictors: list[int], classes: np.ndarray,
              batch_size: int, n_jobs: int) -> np.ndarray:
    if batch_size <= 0 or n_jobs <= 0:
        raise ValueError("batch_size and n_jobs must be positive")
    result = np.empty(len(encoded), dtype=np.float32)
    class_weights = classes.astype(np.float64)
    for start in range(0, len(encoded), batch_size):
        block = np.ascontiguousarray(encoded[start:start + batch_size, predictors], dtype=np.float32)
        data = xgb.DMatrix(block, feature_names=[f"f{i}" for i in range(len(predictors))], nthread=n_jobs)
        probabilities = model.predict(data, strict_shape=True)
        if probabilities.shape != (len(block), len(classes)) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid auxiliary class probability shape/values")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Auxiliary class probabilities are outside [0,1]")
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-5, atol=1e-6)
        # Float32 BLAS reductions can change with batch shape. Sum the small
        # integer-weighted class vector in float64, then round once to float32;
        # identical probabilities now produce identical downstream tree inputs.
        result[start:start + len(block)] = np.sum(
            probabilities.astype(np.float64) * class_weights, axis=1, dtype=np.float64
        ).astype(np.float32)
    if not np.isfinite(result).all() or ((result < classes[0] - 1e-5) | (result > classes[-1] + 1e-5)).any():
        raise ValueError("Expected rating outside fitted class support")
    return result


def fit_bank(original: pd.DataFrame, directory: Path, *, device: str = "cpu", n_jobs: int = 4,
             rounds: int = 400, provenance: dict | None = None) -> dict:
    """Fit/reuse all 13 models. Non-400 rounds are reserved for synthetic tests.

    Extra columns (including satisfaction, if supplied by a caller) are ignored;
    only RAW_FEATURES enter preprocessing, auxiliary targets, or hashes. Production
    build() does not load satisfaction at all. A changed contract cannot overwrite
    a partially/completely fitted bank; use a fresh directory instead.
    """
    if device not in {"cpu", "cuda"} or n_jobs < 1 or rounds < 1:
        raise ValueError("Invalid runtime or fixed-round configuration")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    processor = AuxiliaryPreprocessor.fit(original)
    encoded = processor.transform(original)
    feature_hash = hashlib.sha256(encoded.astype("<f4", copy=False).tobytes()).hexdigest()
    settings = {**FIXED_PARAMS, "n_estimators": int(rounds), "device": device, "n_jobs": int(n_jobs),
                "objective": "multi:softprob", "eval_metric": "mlogloss"}
    contract = {"format": "original_aux_bank", "version": 1, "training_rows": len(original),
                "feature_matrix_sha256": feature_hash, "preprocessing": processor.payload(),
                "params": settings, "provenance": {} if provenance is None else provenance,
                "implementation_sha256": _sha(Path(__file__))}
    contract_file = directory / "build_contract.json"
    if contract_file.exists() and _read_json(contract_file) != contract:
        raise ValueError("Existing original-aux bank contract differs; use a new directory")
    manifest_file = directory / "manifest.json"
    if manifest_file.exists() and _read_json(manifest_file).get("status") == "complete":
        _, completed = load_bank(directory)
        if (completed.get("params") != settings
                or completed.get("input_hashes") != contract["provenance"]
                or completed.get("implementation_sha256") != contract["implementation_sha256"]):
            raise ValueError("Completed original-aux manifest differs from build contract")
        # Published signatures must remain byte-for-byte and timestamp stable.
        return completed
    _json(contract_file, contract)
    preprocess_file = directory / "preprocessor.json"
    _json(preprocess_file, processor.payload())
    entries = []
    for number, target in enumerate(RATINGS):
        start = time.monotonic()
        target_position = RAW_FEATURES.index(target)
        target_values = encoded[:, target_position]
        if not np.isfinite(target_values).all() or not np.equal(target_values, np.floor(target_values)).all():
            raise ValueError(f"Auxiliary target {target!r} must be finite integer ratings")
        classes = np.unique(target_values)
        if len(classes) < 2:
            raise ValueError(f"Auxiliary target {target!r} has fewer than two observed classes")
        positions = [j for j in range(len(RAW_FEATURES)) if j != target_position]
        predictors = [RAW_FEATURES[j] for j in positions]
        model_file = directory / f"model_{number:02d}.ubj"
        entry_file = directory / f"model_{number:02d}.json"
        fixed = {"target": target, "output_column": OUTPUT_COLUMNS[number],
                 "classes": classes.tolist(), "predictors": predictors,
                 "predictor_positions": positions, "model_file": model_file.name,
                 "rounds": rounds, "contract_sha256": _sha(contract_file)}
        if entry_file.exists():
            entry = _read_json(entry_file)
            if any(entry.get(k) != v for k, v in fixed.items()) or _sha(model_file) != entry["model_sha256"]:
                raise ValueError(f"Cached auxiliary model {target!r} differs from contract")
        else:
            labels = np.searchsorted(classes, target_values).astype(np.int32)
            matrix = pd.DataFrame(np.ascontiguousarray(encoded[:, positions]),
                                  columns=[f"f{i}" for i in range(len(positions))])
            classifier = xgb.XGBClassifier(**settings, num_class=len(classes))
            classifier.fit(matrix, labels, verbose=False)
            model = classifier.get_booster()
            if model.num_boosted_rounds() != rounds:
                raise RuntimeError("Auxiliary model did not fit exactly the fixed round count")
            model.set_param({"device": "cpu", "nthread": n_jobs})
            probe = encoded[:min(len(encoded), 257)]
            reference = _expected(model, probe, positions, classes, 17, n_jobs)
            temporary = directory / f".model_{number:02d}-{uuid4().hex}.ubj"
            try:
                model.save_model(temporary)
                restored = xgb.Booster(model_file=temporary)
                restored.set_param({"device": "cpu", "nthread": n_jobs})
                actual = _expected(restored, probe, positions, classes, 31, n_jobs)
                np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-7)
                os.replace(temporary, model_file)
            finally:
                temporary.unlink(missing_ok=True)
            entry = {**fixed, "model_sha256": _sha(model_file), "fit_seconds": time.monotonic() - start,
                     "cpu_reload_max_abs_error": float(np.max(np.abs(reference - actual)))}
            _json(entry_file, entry)
            del classifier, model, restored, matrix
        entries.append(entry)
        print(json.dumps({"auxiliary_target": target, "model": number + 1, "total_models": len(RATINGS),
                          "seconds": time.monotonic() - start, "fixed_rounds": rounds}), flush=True)
    manifest = {
        "format": "original_aux_bank", "version": 1, "status": "bank_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(original), "raw_features": list(RAW_FEATURES),
        "output_columns": list(OUTPUT_COLUMNS), "models": entries,
        "preprocessor_file": preprocess_file.name, "preprocessor_sha256": _sha(preprocess_file),
        "contract_file": contract_file.name, "contract_sha256": _sha(contract_file),
        "params": settings, "input_hashes": {} if provenance is None else provenance,
        "implementation_sha256": _sha(Path(__file__)), "inference_device": "cpu",
        "source": {"url": SOURCE_URL, "notebook_sha256": SOURCE_HASH,
                   "attribution": "Independent implementation of expected-rating feature recipe; one seed instead of source's three",
                   "license_status": "New notebook license not exposed in downloaded metadata; no notebook code copied"},
        "versions": {"python": platform.python_version(), "xgboost": xgb.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "fit_scope": "Original raw feature rows only; predicted rating excluded from its 20 inputs; satisfaction never used",
        "limitations": "Exact overlap removal does not exclude near-duplicates or shared synthetic ancestry; EVs are not satisfaction probabilities",
        "synthetic_smoke_configuration": rounds != FIXED_PARAMS["n_estimators"],
    }
    _json(directory / "manifest.json", manifest)
    return manifest


def load_bank(directory: Path) -> tuple[AuxiliaryPreprocessor, dict]:
    directory = Path(directory)
    manifest = _read_json(directory / "manifest.json")
    if (manifest.get("format") != "original_aux_bank" or manifest.get("version") != 1
            or manifest.get("status") not in {"bank_complete", "complete"}
            or manifest.get("raw_features") != list(RAW_FEATURES)
            or manifest.get("output_columns") != list(OUTPUT_COLUMNS)
            or len(manifest.get("models", [])) != len(RATINGS)):
        raise ValueError("Invalid original-aux bank manifest")
    preprocess_file = _safe_file(directory, manifest["preprocessor_file"])
    contract_file = _safe_file(directory, manifest["contract_file"])
    if _sha(preprocess_file) != manifest["preprocessor_sha256"] or _sha(contract_file) != manifest["contract_sha256"]:
        raise ValueError("Auxiliary preprocessing/contract checksum mismatch")
    for i, entry in enumerate(manifest["models"]):
        expected_positions = [j for j, name in enumerate(RAW_FEATURES) if name != RATINGS[i]]
        if (entry["target"] != RATINGS[i] or entry["output_column"] != OUTPUT_COLUMNS[i]
                or entry["predictor_positions"] != expected_positions
                or entry["predictors"] != [RAW_FEATURES[j] for j in expected_positions]
                or entry["contract_sha256"] != manifest["contract_sha256"]):
            raise ValueError("Auxiliary target/predictor schema mismatch")
        classes = np.asarray(entry["classes"], dtype=np.float32)
        if (classes.ndim != 1 or len(classes) < 2 or not np.isfinite(classes).all()
                or not np.equal(classes, np.floor(classes)).all() or not np.all(np.diff(classes) > 0)):
            raise ValueError("Invalid auxiliary observed-class mapping")
        if _sha(_safe_file(directory, entry["model_file"])) != entry["model_sha256"]:
            raise ValueError("Auxiliary native model checksum mismatch")
    return AuxiliaryPreprocessor.from_payload(_read_json(preprocess_file)), manifest


def predict_aux(raw: pd.DataFrame, directory: Path = ROOT / "artifacts/original_aux", *,
                batch_size: int = 32768, n_jobs: int = 4, device: str = "cpu") -> pd.DataFrame:
    """Return same-index orig_aux_00..12 float32 EVs, requiring only raw features.

    Feature input order is normalized by names. Extra IDs/targets are ignored.
    All native artifacts are hash-checked before any prediction is produced.
    Models load sequentially; CPU is default. CUDA requires explicit device='cuda'.
    There is no mutable global model cache.
    """
    directory = Path(directory)
    if device not in {"cpu", "cuda"}:
        raise ValueError("Inference device must be cpu or cuda")
    processor, manifest = load_bank(directory)
    encoded = processor.transform(raw)
    result = np.empty((len(raw), len(RATINGS)), dtype=np.float32)
    for i, entry in enumerate(manifest["models"]):
        model = xgb.Booster(model_file=_safe_file(directory, entry["model_file"]))
        model.set_param({"device": device, "nthread": n_jobs})
        if model.num_features() != 20 or model.num_boosted_rounds() != entry["rounds"]:
            raise ValueError("Native auxiliary model shape/rounds differ from metadata")
        result[:, i] = _expected(model, encoded, entry["predictor_positions"],
                                 np.asarray(entry["classes"], dtype=np.float32), batch_size, n_jobs)
    return pd.DataFrame(result, index=raw.index.copy(), columns=OUTPUT_COLUMNS)


def benchmark_first_model(raw: pd.DataFrame, directory: Path, *, device: str = "cpu",
                          n_jobs: int = 4, batch_size: int = 32768) -> dict:
    """Time saved first-model inference; values/IDs only, no downstream target."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("Inference device must be cpu or cuda")
    start = time.monotonic()
    processor, manifest = load_bank(directory)
    encoded = processor.transform(raw)
    entry = manifest["models"][0]
    model = xgb.Booster(model_file=_safe_file(directory, entry["model_file"]))
    classes = np.asarray(entry["classes"], dtype=np.float32)
    model.set_param({"device": device, "nthread": n_jobs})
    prediction = _expected(model, encoded, entry["predictor_positions"], classes, batch_size, n_jobs)
    elapsed = time.monotonic() - start
    model.set_param({"device": "cpu", "nthread": n_jobs})
    reference = _expected(model, encoded[:257], entry["predictor_positions"], classes, 17, n_jobs)
    # Device arithmetic and batched reduction can differ at float32 precision.
    np.testing.assert_allclose(prediction[:257], reference, rtol=2e-5, atol=2e-6)
    report = {"rows": len(raw), "device": device, "first_model_seconds_including_load": elapsed,
              "estimated_13_model_seconds": elapsed * len(RATINGS),
              "estimate_limit": "Linear first-model extrapolation, not a measured complete-bank time",
              "cpu_probe_max_abs_error": float(np.max(np.abs(prediction[:257] - reference))) if len(raw) else 0.0}
    print(json.dumps({"original_aux_inference_pilot": report}), flush=True)
    return report


def _validate_ids(frame: pd.DataFrame, name: str) -> None:
    if "id" not in frame or frame["id"].isna().any() or not frame["id"].is_unique:
        raise ValueError(f"{name} must have unique nonmissing IDs")


def validate_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    # Local import keeps inference independent of preparation and shared common.py.
    from prepare_original import canonical_features, exact_match_mask

    root = Path(root)
    original_path, train_path, test_path = (root / p for p in ("data/original.parquet", "data/train.csv", "data/test.csv"))
    audit_path = root / "research/original_audit.json"
    audit = _read_json(audit_path)
    hashes = {"original_parquet_sha256": _sha(original_path), "synthetic_train_csv_sha256": _sha(train_path),
              "synthetic_test_csv_sha256": _sha(test_path), "original_audit_sha256": _sha(audit_path)}
    expected = {"original_parquet_sha256": audit["output_sha256"],
                "synthetic_train_csv_sha256": audit["synthetic_train_sha256"],
                "synthetic_test_csv_sha256": audit["synthetic_test_sha256"]}
    if any(hashes[k] != v for k, v in expected.items()) or set(audit["features"]) != set(RAW_FEATURES):
        raise ValueError("Auxiliary data differ from the original anti-overlap audit")
    # Do not request satisfaction from any file, original or synthetic.
    original = pd.read_parquet(original_path, columns=list(RAW_FEATURES))
    train = pd.read_csv(train_path, usecols=["id", *RAW_FEATURES])
    test = pd.read_csv(test_path, usecols=["id", *RAW_FEATURES])
    if len(original) != audit["counts"]["output_rows"] or original.duplicated(list(RAW_FEATURES)).any():
        raise ValueError("Original feature count or uniqueness differs from audited contract")
    for name, frame in (("train", train), ("test", test)):
        _validate_ids(frame, name)
        if exact_match_mask(original, canonical_features(frame, list(RAW_FEATURES)), list(RAW_FEATURES)).any():
            raise ValueError(f"Original/auxiliary {name} exact feature overlap")
    if np.intersect1d(train.id.to_numpy(), test.id.to_numpy()).size:
        raise ValueError("Auxiliary train/test ID overlap")
    return original, train, test, hashes


def _publish_keyed(frame: pd.DataFrame, features: pd.DataFrame, destination: Path) -> str:
    _validate_ids(frame, "Inference rows")
    if not frame.index.equals(features.index) or list(features.columns) != list(OUTPUT_COLUMNS):
        raise ValueError("Auxiliary row index or output schema mismatch")
    output = features.copy()
    output.insert(0, "id", frame.id.to_numpy())
    if output[list(OUTPUT_COLUMNS)].isna().any().any():
        raise ValueError("Auxiliary keyed output is incomplete")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp.parquet")
    try:
        output.to_parquet(temporary, index=False)
        restored = pd.read_parquet(temporary)
        pd.testing.assert_frame_equal(restored, output.reset_index(drop=True))
        aligned = frame[["id"]].merge(restored, on="id", how="left", sort=False, validate="one_to_one")
        if not np.array_equal(aligned.id.to_numpy(), frame.id.to_numpy()) or aligned[list(OUTPUT_COLUMNS)].isna().any().any():
            raise ValueError("Auxiliary output lost ID coverage or order")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha(destination)


def build(root: Path = ROOT, *, device: str = "cpu", inference_device: str = "cpu",
          n_jobs: int = 4, batch_size: int = 32768) -> dict:
    start = time.monotonic()
    root = Path(root)
    original, train, test, hashes = validate_inputs(root)
    directory = root / "artifacts/original_aux"
    manifest = fit_bank(original, directory, device=device, n_jobs=n_jobs, provenance=hashes)
    output = root / "data/original_aux_predictions.parquet"
    if manifest.get("status") == "complete":
        expected_metadata = {"output_file": "data/original_aux_predictions.parquet",
                             "output_rows": len(train) + len(test),
                             "synthetic_train_rows": len(train), "synthetic_test_rows": len(test),
                             "inference_device": inference_device,
                             "prepare_original_code_sha256": _sha(Path(__file__).with_name("prepare_original.py"))}
        if any(manifest.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("Completed auxiliary cache metadata/settings differ")
        if output.exists():
            if _sha(output) != manifest.get("output_sha256"):
                raise ValueError("Completed auxiliary cache checksum mismatch; refusing to overwrite")
            cached = pd.read_parquet(output)
            expected_ids = np.concatenate([train.id.to_numpy(), test.id.to_numpy()])
            if (list(cached.columns) != ["id", *OUTPUT_COLUMNS]
                    or len(cached) != len(expected_ids) or not cached.id.is_unique
                    or not np.array_equal(cached.id.to_numpy(), expected_ids)
                    or any(cached[c].dtype != np.dtype("float32") for c in OUTPUT_COLUMNS)
                    or not np.isfinite(cached[list(OUTPUT_COLUMNS)].to_numpy()).all()):
                raise ValueError("Completed auxiliary cache schema, IDs, order, or values differ")
            print(json.dumps({"reused_complete": str(output), "rows": len(cached),
                              "writes": 0}), flush=True)
            return manifest
        # Missing cache may be regenerated from verified native models. Changed
        # cache/native artifacts above fail closed rather than being repaired.
    pilot = benchmark_first_model(test, directory, device=inference_device, n_jobs=n_jobs, batch_size=batch_size)
    raw = pd.concat([train, test], ignore_index=True)
    features = predict_aux(raw, directory, batch_size=batch_size, n_jobs=n_jobs, device=inference_device)
    output_hash = _publish_keyed(raw, features, output)
    # A second independent load/batch boundary verifies saved-model cache alignment.
    positions = np.concatenate([np.arange(min(512, len(train))), np.arange(max(len(raw) - 257, 0), len(raw))])
    probe = raw.iloc[positions]
    restored = predict_aux(probe, directory, batch_size=17, n_jobs=n_jobs)
    np.testing.assert_allclose(restored.to_numpy(), features.iloc[positions].to_numpy(), rtol=2e-5, atol=2e-6)
    manifest.update({"status": "complete", "output_file": "data/original_aux_predictions.parquet",
                     "output_sha256": output_hash, "output_rows": len(raw), "synthetic_train_rows": len(train),
                     "synthetic_test_rows": len(test), "elapsed_seconds": time.monotonic() - start,
                     "inference_device": inference_device, "inference_pilot": pilot,
                     "synthetic_labels_read": False, "satisfaction_labels_read": False,
                     "known_id_proof": "Unique disjoint train/test IDs; full-order keyed merge, nonmissing features, exact parquet readback, native reload probe",
                     "prepare_original_code_sha256": _sha(Path(__file__).with_name("prepare_original.py"))})
    _json(directory / "manifest.json", manifest)
    print(json.dumps({"complete": str(output), "rows": len(raw), "features": len(OUTPUT_COLUMNS),
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("build", help="Build the fixed 13-model original-only bank and keyed cache")
    train.add_argument("--root", type=Path, default=ROOT)
    train.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train.add_argument("--inference-device", choices=["cpu", "cuda"], default="cpu")
    train.add_argument("--n-jobs", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=32768)
    infer = commands.add_parser("predict", help="Predict EV features for arbitrary raw rows; native CPU inference")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True, help="Destination Parquet file")
    infer.add_argument("--directory", type=Path, default=ROOT / "artifacts/original_aux")
    infer.add_argument("--n-jobs", type=int, default=4)
    infer.add_argument("--batch-size", type=int, default=32768)
    infer.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.command == "build":
        build(args.root.resolve(), device=args.device, inference_device=args.inference_device,
              n_jobs=args.n_jobs, batch_size=args.batch_size)
    else:
        columns = ["id", *RAW_FEATURES]
        raw = (pd.read_parquet(args.input, columns=columns) if args.input.suffix == ".parquet"
               else pd.read_csv(args.input, usecols=columns))
        values = predict_aux(raw, args.directory, batch_size=args.batch_size, n_jobs=args.n_jobs, device=args.device)
        digest = _publish_keyed(raw, values, args.output)
        print(json.dumps({"output": str(args.output), "rows": len(raw), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
