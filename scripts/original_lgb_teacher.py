"""Fixed original-only LightGBM teacher, with immutable native IO and keyed scores.

Recipe independently implemented from the view-B teacher documented in
research/second_pass_competition.md. Competition satisfaction labels are never
requested from disk, and neither early stopping nor synthetic-label tuning occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import ROOT, TARGET, Transform, atomic_json, load_config, sha256
from supervisor import lifetime_lock
from teacher import inputs


DIRECTORY = ROOT / "artifacts/original_lgb_teacher"
PREDICTION_COLUMN = "original_lgb_teacher_probability"
FIXED_ROUNDS = 1500
PARAMETERS = {
    "learning_rate": 0.03, "num_leaves": 63, "subsample": 0.8,
    "subsample_freq": 1, "colsample_bytree": 0.5, "min_child_samples": 20,
}
NATIVE_FILES = ("model.txt", "transform.json", "model_metadata.json")


def recipe() -> dict:
    return {"name": "original_only_fixed_lgb_teacher", "family": "lightgbm",
            "rounds": FIXED_ROUNDS, "seed": 0, "n_jobs": 4,
            "objective": "binary", "params": dict(PARAMETERS)}


def source_hash(value) -> str:
    return hashlib.sha256(inspect.getsource(value).encode("utf-8")).hexdigest()


def build_contract(original, train, test, features, input_hashes) -> dict:
    # Hash the exact reused implementation as the resume identity. Unrelated
    # additions to common.features must not invalidate this external-only bank.
    return {
        "schema_version": 1, "run": recipe(), "feature_columns": features,
        "positive_class": 1, "label_mapping": {"False": 0, "True": 1},
        "input_hashes": input_hashes,
        "rows": {"original": len(original), "train": len(train), "test": len(test)},
        "implementation": {
            "original_lgb_teacher.py": sha256(Path(__file__)),
            "Transform_source_sha256": source_hash(Transform),
            "teacher_inputs_source_sha256": source_hash(inputs),
            "prepare_original.py": sha256(Path(__file__).with_name("prepare_original.py")),
        },
        "library_sha256": {
            "lightgbm_sklearn.py": sha256(Path(inspect.getfile(lgb.LGBMClassifier))),
            "lightgbm_basic.py": sha256(Path(inspect.getfile(lgb.Booster))),
            "lightgbm_native_library": sha256(Path(lgb.basic._LIB._name)),
            "requirements.lock.txt": sha256(Path(__file__).resolve().parents[1] / "requirements.lock.txt"),
        },
        "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }


def _validate_features(original: pd.DataFrame, features: list[str]) -> None:
    if not features or len(set(features)) != len(features) or {"id", TARGET}.intersection(features):
        raise ValueError("Teacher predictors must be unique raw features without ID or satisfaction")
    if not original.columns.is_unique or not set(features + [TARGET]).issubset(original):
        raise ValueError("Invalid source feature/target schema")
    if not original[TARGET].isin([0, 1]).all() or original[TARGET].nunique() != 2:
        raise ValueError("Source satisfaction must contain binary classes zero and one")


def _fit_native(original: pd.DataFrame, features: list[str], directory: Path, run: dict) -> None:
    _validate_features(original, features)
    transform = Transform("lightgbm").fit(original.loc[:, features])
    model = lgb.LGBMClassifier(
        n_estimators=run["rounds"], objective="binary", random_state=0,
        n_jobs=4, verbosity=-1, **run["params"])
    model.fit(transform.transform(original.loc[:, features]), original[TARGET].to_numpy(dtype=np.int8))
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Native probability must represent class one")
    actual_rounds = int(model.booster_.current_iteration())
    if actual_rounds != run["rounds"]:
        raise ValueError(f"Expected {run['rounds']} fixed rounds; native model has {actual_rounds}")
    temporary = directory / "model.tmp.txt"
    model.booster_.save_model(str(temporary))
    temporary.replace(directory / "model.txt")
    transform.save(directory / "transform.json")
    atomic_json(directory / "model_metadata.json", {
        "family": "lightgbm", "rounds": actual_rounds, "run": run,
        "feature_columns": features, "positive_class": 1, "classes": [0, 1],
    })
    restored, restored_transform, metadata = _load_native(directory)
    reference = model.predict_proba(transform.transform(original.loc[:, features].head(257)))[:, 1]
    actual = _predict(restored, restored_transform, original.head(257), metadata["feature_columns"])
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=6e-8)


def _load_native(directory: Path):
    metadata = load_config(directory / "model_metadata.json")
    if metadata.get("family") != "lightgbm" or metadata.get("positive_class") != 1 or metadata.get("classes") != [0, 1]:
        raise ValueError("Invalid original LightGBM native model identity")
    transform = Transform.load(directory / "transform.json")
    if transform.family != "lightgbm" or transform.columns != metadata["feature_columns"]:
        raise ValueError("Native preprocessing does not match the feature contract")
    model = lgb.Booster(model_file=str(directory / "model.txt"))
    if model.current_iteration() != metadata["rounds"] or model.num_feature() != len(transform.columns):
        raise ValueError("Native model shape/round count differs from metadata")
    return model, transform, metadata


def _predict(model, transform: Transform, raw: pd.DataFrame, features: list[str]) -> np.ndarray:
    if not raw.columns.is_unique or not set(features).issubset(raw):
        raise ValueError("Teacher inference requires every raw feature exactly once")
    probability = np.empty(len(raw), dtype=np.float32)
    for start in range(0, len(raw), 32768):
        end = min(start + 32768, len(raw))
        encoded = transform.transform(raw.iloc[start:end].loc[:, features])
        score = np.asarray(model.predict(encoded, num_threads=4), dtype=np.float64)
        if score.shape != (end - start,) or not np.isfinite(score).all() or not ((score >= 0) & (score <= 1)).all():
            raise ValueError("Teacher returned invalid class-one probabilities")
        probability[start:end] = score
    return probability


def validate_native(directory: Path, expected_contract_hash: str | None = None) -> dict:
    checkpoint = load_config(directory / "model_complete.json")
    if expected_contract_hash is not None and checkpoint["contract_sha256"] != expected_contract_hash:
        raise ValueError("Native checkpoint belongs to another teacher contract")
    if checkpoint["contract_sha256"] != sha256(directory / "contract.json"):
        raise ValueError("Teacher contract checksum changed")
    if set(checkpoint["artifacts"]) != set(NATIVE_FILES):
        raise ValueError("Native teacher checkpoint is incomplete")
    for name, digest in checkpoint["artifacts"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Immutable teacher artifact checksum mismatch: {name}")
    return checkpoint


def predict_teacher(raw: pd.DataFrame, directory: Path = DIRECTORY) -> np.ndarray:
    """Return class-one probabilities for arbitrary raw rows, including new IDs.

    ID and satisfaction columns, when present, are ignored. Vocabularies and
    numerical preprocessing come exclusively from the saved original-row fit.
    """
    directory = Path(directory)
    validate_native(directory)
    model, transform, metadata = _load_native(directory)
    return _predict(model, transform, raw, metadata["feature_columns"])


def _check_export(path: Path, expected_ids: np.ndarray) -> None:
    frame = pd.read_parquet(path)
    if list(frame.columns) != ["id", PREDICTION_COLUMN] or not frame.id.is_unique:
        raise ValueError("Invalid teacher export schema or duplicate IDs")
    if not np.array_equal(frame.id.to_numpy(), expected_ids):
        raise ValueError("Teacher export coverage/order mismatch")
    values = frame[PREDICTION_COLUMN].to_numpy()
    if values.dtype != np.float32 or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Teacher export must contain finite float32 probabilities")


def build(root: Path = ROOT, *, validate_only: bool = False) -> dict:
    root = Path(root).resolve()
    original, train, test, features, input_hashes = inputs(root)
    contract = build_contract(original, train, test, features, input_hashes)
    _validate_features(original, features)
    if validate_only:
        return {"validated": True, "contract": contract, "synthetic_labels_read": False}
    directory = root / "artifacts/original_lgb_teacher"
    output = root / "data/original_lgb_teacher_predictions.parquet"
    directory.mkdir(parents=True, exist_ok=True)
    ids = np.concatenate([train.id.to_numpy(), test.id.to_numpy()])
    start = time.monotonic()
    with lifetime_lock(directory / "build.lock"):
        contract_path = directory / "contract.json"
        if contract_path.exists():
            if load_config(contract_path) != contract:
                raise ValueError("Immutable teacher contract changed; use a new artifact identity")
        else:
            # Historical full-file provenance is stored once. The resume key
            # above pins the actual reused preprocessing and input functions.
            atomic_json(directory / "code_provenance.json", {
                name: sha256(Path(__file__).with_name(name)) for name in
                ("original_lgb_teacher.py", "common.py", "teacher.py", "prepare_original.py")})
            atomic_json(contract_path, contract)
        contract_hash = sha256(contract_path)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = load_config(manifest_path)
            expected = {"status": "complete", "contract": contract, "run": contract["run"],
                        "input_hashes": input_hashes, "versions": contract["versions"],
                        "prediction_column": PREDICTION_COLUMN, "synthetic_labels_read": False,
                        "code_sha256": load_config(directory / "code_provenance.json")}
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError("Immutable completed teacher metadata changed")
            if manifest["contract_sha256"] != contract_hash or manifest["output_sha256"] != sha256(output):
                raise ValueError("Immutable completed teacher manifest/output changed")
            checkpoint = validate_native(directory, contract_hash)
            if manifest["native_artifacts"] != checkpoint["artifacts"]:
                raise ValueError("Completed teacher/native checkpoint changed")
            _check_export(output, ids)
            print("ORIGINAL LIGHTGBM TEACHER ALREADY COMPLETE", flush=True)
            return manifest
        if (directory / "model_complete.json").exists():
            validate_native(directory, contract_hash)
        else:
            print(f"FIT ORIGINAL LIGHTGBM rows={len(original)} rounds={FIXED_ROUNDS}", flush=True)
            _fit_native(original, features, directory, contract["run"])
            atomic_json(directory / "model_complete.json", {
                "contract_sha256": contract_hash,
                "artifacts": {name: sha256(directory / name) for name in NATIVE_FILES},
            })
        checkpoint = validate_native(directory, contract_hash)
        model, transform, metadata = _load_native(directory)
        pieces = []
        for label, frame in [("train", train), ("test", test)]:
            probability = _predict(model, transform, frame, metadata["feature_columns"])
            pieces.append(pd.DataFrame({"id": frame.id.to_numpy(), PREDICTION_COLUMN: probability}))
            print(f"PREDICT ORIGINAL LIGHTGBM {label} rows={len(frame)}", flush=True)
        temporary = output.with_suffix(".tmp.parquet")
        pd.concat(pieces, ignore_index=True).to_parquet(temporary, index=False)
        _check_export(temporary, ids)
        temporary.replace(output)
        manifest = {
            "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": time.monotonic() - start, "contract_sha256": contract_hash,
            "contract": contract, "run": contract["run"], "input_hashes": input_hashes,
            "native_artifacts": checkpoint["artifacts"], "output_sha256": sha256(output),
            "prediction_column": PREDICTION_COLUMN, "code_sha256": load_config(directory / "code_provenance.json"),
            "versions": contract["versions"], "synthetic_labels_read": False,
            "fit_policy": "Fixed original-only binary teacher; original-only category vocabularies; no validation, early stopping, or tuning",
            "source_recipe": "https://www.kaggle.com/code/goodpjw2008/s6e10-realmlp-aux-task-features-lb-0-96141",
            "limitations": "Exact profile exclusion does not remove near-duplicate source ancestry; no synthetic validation gain is claimed",
        }
        atomic_json(manifest_path, manifest)
        print(f"ORIGINAL LIGHTGBM COMPLETE {output} rows={len(ids)}", flush=True)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs; do not fit or write artifacts")
    args = parser.parse_args()
    result = build(args.root, validate_only=args.validate_only)
    print(json.dumps({"status": result.get("status", "validated"),
                      "rows": result["contract"]["rows"]}), flush=True)


if __name__ == "__main__":
    main()
