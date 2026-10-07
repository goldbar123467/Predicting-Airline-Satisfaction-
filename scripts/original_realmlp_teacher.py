"""One fixed original-only categorical RealMLP teacher and CPU logit cache.

The cleaned original source and exact-overlap audit are reused unchanged.
Synthetic satisfaction labels are never requested; there is no stopping set.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from categorical_transform import CategoricalTransform
from common import ROOT, CAT, TARGET, atomic_json, fit_model, load_config, sha256
from supervisor import lifetime_lock
from teacher import inputs


DIRECTORY = ROOT / "artifacts/original_realmlp_teacher"
PREDICTION_COLUMN = "original_realmlp_teacher_logit"
ORIGINAL_ROWS = 129859
FIXED_EPOCHS = 8
TIMEOUT_SECONDS = 600
CLIP = 1e-6
PARAMETERS = {
    "device": "cuda", "n_ens": 8, "hidden_sizes": [512, 256, 128],
    "batch_size": 256, "eval_batch_size": 2048, "learning_rate": .053,
    "weight_decay": .015, "threads": 4, "ls_eps": 0.0, "lr_sched": "flat_anneal",
}


def recipe() -> dict:
    return {"id": "original_only_fixed_realmlp_teacher", "family": "realmlp_cat",
            "seed": 0, "max_rounds": FIXED_EPOCHS, "categorical_twins": True,
            "params": json.loads(json.dumps(PARAMETERS))}


def source_hash(value) -> str:
    return hashlib.sha256(inspect.getsource(value).encode("utf-8")).hexdigest()


def feature_frame(raw: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Reproduce the raw numeric/category twins without any ID-keyed feature bank."""
    if not raw.columns.is_unique or not set(feature_columns).issubset(raw):
        raise ValueError("Teacher inference requires every raw feature exactly once")
    x = raw.loc[:, feature_columns].copy()
    for column in feature_columns:
        if column not in CAT:
            x[column] = pd.to_numeric(x[column], errors="raise").astype("float32")
            if np.isinf(x[column].to_numpy()).any():
                raise ValueError("Infinite or overflowing teacher input")
            x[column + "_category"] = x[column].astype(str)
    x["class_travel_gender"] = (x["Class"].astype(str) + "|" +
                                x["Type of Travel"].astype(str) + "|" + x["Gender"].astype(str))
    return x


def _validate_source(original: pd.DataFrame, feature_columns: list[str]) -> None:
    if (len(original) != ORIGINAL_ROWS or len(feature_columns) != 21
            or len(set(feature_columns)) != 21 or {"id", TARGET}.intersection(feature_columns)
            or list(original.columns) != feature_columns + [TARGET]
            or not original[TARGET].isin([0, 1]).all() or original[TARGET].nunique() != 2):
        raise ValueError("Unexpected cleaned original-only source contract")


def build_contract(original, train, test, feature_columns, input_hashes) -> dict:
    scripts = Path(__file__).resolve().parent
    versions = {name: importlib.metadata.version(name)
                for name in ["pytabkit", "torch", "pytorch-lightning", "numpy", "pandas", "scikit-learn"]}
    libraries = {}
    for name in versions:
        distribution = importlib.metadata.distribution(name)
        record = next((p for p in distribution.files or [] if str(p).endswith(".dist-info/RECORD")), None)
        if record is None:
            raise ValueError(f"Missing installed library RECORD: {name}")
        libraries[f"{name}_RECORD"] = sha256(Path(distribution.locate_file(record)))
    return {
        "schema_version": 1, "run": recipe(), "feature_columns": feature_columns,
        "positive_class": 1, "classes": [0, 1], "input_hashes": input_hashes,
        "rows": {"original": len(original), "train": len(train), "test": len(test)},
        "feature_rule": {"column": PREDICTION_COLUMN, "probability_clip": [CLIP, 1 - CLIP],
                         "calculation_dtype": "float64", "output_dtype": "float32", "inference_device": "cpu"},
        "implementation": {
            **{name: sha256(scripts / name) for name in ["original_realmlp_teacher.py", "realmlp.py",
                                                        "realmlp_categorical.py", "categorical_transform.py",
                                                        "prepare_original.py"]},
            "common_fit_model_source": source_hash(fit_model), "teacher_inputs_source": source_hash(inputs),
        },
        "library_record_sha256": libraries,
        "requirements_sha256": sha256(scripts.parent / "requirements.lock.txt"),
        "versions": {"python": platform.python_version(), **versions},
    }


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Teacher returned invalid class-one probabilities")
    p = np.clip(p, CLIP, 1 - CLIP)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def _load_native(directory: Path):
    import torch
    from realmlp_categorical import load_realmlp_categorical
    teacher = load_config(directory / "teacher_metadata.json")
    metadata = load_config(directory / "model/model_metadata.json")
    native = load_config(directory / "model/metadata.json")
    transform = CategoricalTransform.load(directory / "model/transform.json")
    if (metadata["family"] != "realmlp_cat" or metadata["run"] != teacher["run"]
            or metadata["rounds"] != teacher["run"]["max_rounds"]
            or native["classes"] != [0, 1] or native["validation_rows"] != 0
            or native["best_epoch"] != metadata["rounds"] or native["train_rows"] != teacher["rows"]
            or native["selection"] != "fixed epochs, all supplied rows"
            or transform.columns != teacher["expanded_feature_columns"]
            or transform.family != "realmlp_cat"):
        raise ValueError("Native original teacher identity/preprocessing/fit scope changed")
    torch.set_num_threads(int(teacher["run"]["params"]["threads"]))
    model = load_realmlp_categorical(directory / "model", device="cpu")
    if str(model.device) != "cpu" or not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Teacher must infer class-one probabilities on CPU")
    return model, transform, teacher


def _probabilities(model, transform, raw: pd.DataFrame, feature_columns: list[str], chunk_size: int = 32768) -> np.ndarray:
    if not raw.columns.is_unique or not set(feature_columns).issubset(raw):
        raise ValueError("Teacher inference requires every raw feature exactly once")
    probability = np.empty(len(raw), dtype=np.float32)
    for start in range(0, len(raw), chunk_size):
        end = min(start + chunk_size, len(raw))
        encoded = transform.transform(feature_frame(raw.iloc[start:end], feature_columns))
        predicted = np.asarray(model.predict_proba(encoded))
        if predicted.shape != (end - start, 2):
            raise ValueError("Teacher class-probability shape changed")
        probability[start:end] = predicted[:, 1]
    clipped_logit(probability)  # Finite/probability-domain validation also handles empty inputs.
    return probability


def _fit_native(original, feature_columns, directory, run):
    x = feature_frame(original, feature_columns)
    model, transform, rounds = fit_model(x, original[TARGET].to_numpy(dtype=np.int8),
                                         None, None, run, directory / "model", rounds=FIXED_EPOCHS)
    if rounds != FIXED_EPOCHS or not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Teacher did not use the fixed epoch count and binary class order")
    atomic_json(directory / "teacher_metadata.json", {
        "run": run, "rows": len(original), "feature_columns": feature_columns,
        "expanded_feature_columns": list(x), "positive_class": 1, "classes": [0, 1],
    })
    del model, transform, x
    gc.collect()
    # CPU batch parity is checked only after native export, not by comparing
    # CPU/GPU logits near saturation with an arbitrarily loosened tolerance.
    restored, transform, _ = _load_native(directory)
    probe = original.head(65)
    reference = _probabilities(restored, transform, probe, feature_columns)
    errors = {}
    for batch in [1, 17]:
        actual = _probabilities(restored, transform, probe, feature_columns, chunk_size=batch)
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-6)
        errors[str(batch)] = float(np.max(np.abs(actual - reference)))
    del restored, transform
    gc.collect()
    return {"device": "cpu", "rows": len(probe), "max_probability_abs_difference": errors,
            "rtol": 2e-5, "atol": 2e-6}


def _native_files(directory: Path) -> dict[str, str]:
    paths = [directory / "teacher_metadata.json"] + sorted(p for p in (directory / "model").rglob("*") if p.is_file())
    return {p.relative_to(directory).as_posix(): sha256(p) for p in paths}


def validate_native(directory: Path, contract_hash: str | None = None) -> dict:
    checkpoint = load_config(directory / "model_complete.json")
    expected = sha256(directory / "contract.json")
    if checkpoint["contract_sha256"] != expected or (contract_hash is not None and expected != contract_hash):
        raise ValueError("Teacher native contract checksum changed")
    if checkpoint["artifacts"] != _native_files(directory):
        raise ValueError("Immutable teacher native artifact checksum mismatch")
    return checkpoint


def predict_teacher(raw: pd.DataFrame, directory: Path = DIRECTORY, batch_size: int = 2048) -> np.ndarray:
    """Return float32 clipped logits using native CPU inference; IDs/labels are ignored."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Teacher batch_size must be a positive integer")
    directory = Path(directory)
    validate_native(directory)
    model, transform, metadata = _load_native(directory)
    return clipped_logit(_probabilities(model, transform, raw, metadata["feature_columns"], batch_size))


def _check_export(path: Path, ids: np.ndarray) -> None:
    frame = pd.read_parquet(path)
    if (list(frame) != ["id", PREDICTION_COLUMN] or not frame.id.is_unique
            or not np.array_equal(frame.id.to_numpy(), ids)):
        raise ValueError("Teacher cache schema/row identity changed")
    values = frame[PREDICTION_COLUMN].to_numpy()
    lower, upper = clipped_logit(np.array([0., 1.]))
    if values.dtype != np.float32 or not np.isfinite(values).all() or not ((values >= lower) & (values <= upper)).all():
        raise ValueError("Teacher cache must contain finite bounded float32 logits")


def build(root: Path = ROOT, *, validate_only: bool = False, prepare_only: bool = False) -> dict:
    start = time.monotonic()
    root = Path(root).resolve()
    original, train, test, feature_columns, input_hashes = inputs(root)
    _validate_source(original, feature_columns)
    contract = build_contract(original, train, test, feature_columns, input_hashes)
    if validate_only:
        return {"status": "validated", "contract": contract, "synthetic_labels_read": False}
    directory = root / "artifacts/original_realmlp_teacher"
    directory.mkdir(parents=True, exist_ok=True)
    output = root / "data/original_realmlp_teacher_predictions.parquet"
    ids = np.concatenate([train.id.to_numpy(), test.id.to_numpy()])
    with lifetime_lock(directory / "build.lock"):
        contract_path = directory / "contract.json"
        if contract_path.exists():
            if load_config(contract_path) != contract:
                raise ValueError("Immutable teacher contract changed; use a new artifact identity")
        else:
            atomic_json(directory / "code_provenance.json", {
                name: sha256(Path(__file__).with_name(name)) for name in
                ["original_realmlp_teacher.py", "common.py", "teacher.py", "realmlp.py",
                 "realmlp_categorical.py", "categorical_transform.py", "prepare_original.py"]})
            atomic_json(contract_path, contract)
        contract_hash = sha256(contract_path)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = load_config(manifest_path)
            if (manifest.get("status") != "complete" or manifest["contract"] != contract
                    or manifest["contract_sha256"] != contract_hash
                    or manifest["output_sha256"] != sha256(output)):
                raise ValueError("Immutable completed teacher manifest/cache changed")
            checkpoint = validate_native(directory, contract_hash)
            if manifest["native_artifacts"] != checkpoint["artifacts"]:
                raise ValueError("Immutable teacher manifest/native checkpoint changed")
            _check_export(output, ids)
            return manifest
        if prepare_only:
            return {"status": "prepared", "contract": contract, "contract_sha256": contract_hash}
        if not (directory / "model_complete.json").exists():
            print(f"FIT ORIGINAL REALMLP rows={len(original)} epochs={FIXED_EPOCHS}", flush=True)
            parity = _fit_native(original, feature_columns, directory, contract["run"])
            atomic_json(directory / "model_complete.json", {
                "contract_sha256": contract_hash, "artifacts": _native_files(directory), "cpu_batch_parity": parity})
        checkpoint = validate_native(directory, contract_hash)
        del original
        gc.collect()
        model, transform, metadata = _load_native(directory)
        pieces = []
        for label, frame in [("train", train), ("test", test)]:
            values = clipped_logit(_probabilities(model, transform, frame, feature_columns))
            pieces.append(pd.DataFrame({"id": frame.id.to_numpy(), PREDICTION_COLUMN: values}))
            print(f"PREDICT ORIGINAL REALMLP {label} rows={len(frame)}", flush=True)
        temporary = output.with_suffix(".tmp.parquet")
        pd.concat(pieces, ignore_index=True).to_parquet(temporary, index=False)
        _check_export(temporary, ids)
        temporary.replace(output)
        manifest = {
            "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": time.monotonic() - start, "contract_sha256": contract_hash,
            "contract": contract, "run": contract["run"], "input_hashes": input_hashes,
            "native_artifacts": checkpoint["artifacts"], "output_sha256": sha256(output),
            "prediction_column": PREDICTION_COLUMN, "cpu_batch_parity": checkpoint["cpu_batch_parity"],
            "code_sha256": load_config(directory / "code_provenance.json"), "versions": contract["versions"],
            "synthetic_labels_read": False, "fit_policy": "Original-only fixed epochs, no validation or selection",
            "source_recipe": "https://www.kaggle.com/code/goodpjw2008/s6e10-realmlp-aux-task-features-lb-0-96141",
            "limitations": "Exact overlap exclusion does not remove near-duplicate ancestry; no new validation score claimed",
        }
        atomic_json(manifest_path, manifest)
        print(f"ORIGINAL REALMLP COMPLETE rows={len(ids)}", flush=True)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    def expired():
        print("Original RealMLP teacher exceeded its fixed 600-second budget", flush=True)
        os._exit(124)

    watchdog = threading.Timer(TIMEOUT_SECONDS, expired)
    watchdog.daemon = True
    watchdog.start()
    try:
        result = build(args.root, validate_only=args.validate_only, prepare_only=args.prepare_only)
        print(json.dumps({"status": result["status"], "rows": result["contract"]["rows"]}), flush=True)
    finally:
        watchdog.cancel()


if __name__ == "__main__":
    main()
