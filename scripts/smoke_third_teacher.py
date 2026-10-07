"""Bounded development-only teacher/student integration smoke, run only when GPU is idle."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from categorical_transform import CategoricalTransform
from common import ROOT, CAT, TARGET, atomic_json, features, fit_model, load_config, predict, sha256
import original_realmlp_teacher as teacher


OUT = ROOT / "artifacts/smoke/third_pass_teacher"
REGISTRATION = ROOT / "configs/third_pass_teacher_reserve.json"
TIMEOUT_SECONDS = 120


def development_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join/filter IDs first; the subsequent label read contains only selected dev IDs."""
    split_path = ROOT / "data/splits.parquet"
    registration = load_config(REGISTRATION)
    if sha256(split_path) != registration["source_hashes"]["data/splits.parquet"]:
        raise ValueError("Frozen split hash changed")
    split = pd.read_parquet(split_path, columns=["id", "fold"])
    identifiers = pd.read_parquet(ROOT / "data/train.parquet", columns=["id"])
    joined = identifiers.merge(split, on="id", how="left", validate="one_to_one", sort=False)
    if len(joined) != len(split) or joined.fold.isna().any():
        raise ValueError("Training IDs and frozen splits differ")
    eligible = joined.loc[joined.fold >= 0].copy()
    selected = eligible.sample(n=1024, random_state=20261005).reset_index(drop=True)
    if len(selected) != 1024 or not selected.id.is_unique or not selected.fold.ge(0).all():
        raise ValueError("Smoke selected a non-development or duplicate row")
    del split, identifiers, joined, eligible
    gc.collect()
    # PyArrow predicate filtering prevents requesting an audit label, even while
    # discovering the population from metadata/ID columns above.
    raw = pd.read_parquet(ROOT / "data/train.parquet", filters=[("id", "in", selected.id.tolist())])
    if len(raw) != 1024 or not raw.id.is_unique or set(raw.id) != set(selected.id):
        raise ValueError("Filtered raw rows differ from selected development IDs")
    raw = raw.set_index("id").loc[selected.id].reset_index()
    if not raw[TARGET].isin([0, 1]).all() or raw[TARGET].nunique() != 2:
        raise ValueError("Smoke development labels must contain both binary classes")
    return raw, selected


def run() -> dict:
    start = time.monotonic()
    registration = load_config(REGISTRATION)
    registered = registration["student_run"]
    if registered["id"] != "v3_realmlp_cat_raw_aux_original_nn_teacher":
        raise ValueError("Unexpected preregistered student")
    control = registration["student_control_run"]
    if ({k: v for k, v in registered.items() if k not in {"id", "original_realmlp_teacher"}}
            != {k: v for k, v in control.items() if k != "id"}):
        raise ValueError("Registered student must differ only by ID and teacher feature")
    bank = load_config(teacher.DIRECTORY / "manifest.json")
    cache = ROOT / "data/original_realmlp_teacher_predictions.parquet"
    if bank.get("status") != "complete" or bank["output_sha256"] != sha256(cache):
        raise ValueError("Teacher bank/cache is not complete and hash verified")
    teacher.validate_native(teacher.DIRECTORY)
    raw, selected = development_rows()
    raw_features = bank["contract"]["feature_columns"]
    expected_columns = (raw_features + [c + "_category" for c in raw_features if c not in CAT]
                        + ["class_travel_gender"] + [f"orig_aux_{i:02d}" for i in range(13)]
                        + [teacher.PREDICTION_COLUMN])
    x = features(raw, registered)
    if len(expected_columns) != 53 or list(x) != expected_columns:
        raise ValueError("Student must have the exact registered 53-feature order")
    if x[teacher.PREDICTION_COLUMN].dtype != np.float32:
        raise ValueError("Teacher student feature is not float32")
    model, transform, metadata = teacher._load_native(teacher.DIRECTORY)
    probability = teacher._probabilities(model, transform, raw, raw_features)
    native_logit = teacher.clipped_logit(probability)
    cached_logit = x[teacher.PREDICTION_COLUMN].to_numpy()
    np.testing.assert_allclose(native_logit, cached_logit, rtol=2e-5, atol=2e-6)
    teacher_error = float(np.max(np.abs(native_logit - cached_logit)))
    del model, transform, metadata
    gc.collect()

    fit_rows, held_rows = train_test_split(np.arange(len(raw)), test_size=.2,
                                          random_state=20261005, stratify=raw[TARGET])
    if set(fit_rows).intersection(held_rows) or len(fit_rows) + len(held_rows) != 1024:
        raise ValueError("Smoke fit and held-out row partition changed")
    # The immutable registered recipe is copied, never changed on disk. The
    # smoke's only training change is the explicit fixed two-epoch horizon.
    smoke_run = copy.deepcopy(registered)
    smoke_run["max_rounds"] = 2
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUT / "selection.json", {
        "scope": "development IDs joined to frozen splits and fold>=0 before sampling",
        "selected_ids": selected.id.tolist(), "source_folds": selected.fold.astype(int).tolist(),
        "fit_ids": raw.id.iloc[fit_rows].tolist(), "heldout_ids": raw.id.iloc[held_rows].tolist(),
        "audit_rows": 0, "selection_seed": 20261005,
    })
    print(f"FIT TEACHER STUDENT SMOKE train={len(fit_rows)} heldout={len(held_rows)} features=53 epochs=2", flush=True)
    model, transform, rounds = fit_model(x.iloc[fit_rows], raw[TARGET].iloc[fit_rows].to_numpy(),
                                         None, None, smoke_run, OUT / "model", rounds=2)
    if rounds != 2 or not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Smoke did not use fixed two epochs and class-one probability")
    import torch
    from realmlp_categorical import load_realmlp_categorical
    model.model.to("cpu")
    model.device = torch.device("cpu")
    held = x.iloc[held_rows]
    reference = predict(model, transform, held)
    del model, transform
    gc.collect()
    torch.cuda.empty_cache()
    restored = load_realmlp_categorical(OUT / "model", device="cpu")
    restored_transform = CategoricalTransform.load(OUT / "model/transform.json")
    native = load_config(OUT / "model/metadata.json")
    if (native["train_rows"] != len(fit_rows) or native["validation_rows"] != 0
            or native["best_epoch"] != 2 or native["selection"] != "fixed epochs, all supplied rows"):
        raise ValueError("Smoke native fit scope changed")
    parity = {}
    for batch_size in [1, 17, len(held)]:
        pieces = [predict(restored, restored_transform, held.iloc[start:start + batch_size])
                  for start in range(0, len(held), batch_size)]
        actual = np.concatenate(pieces)
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-6)
        parity[str(batch_size)] = float(np.max(np.abs(actual - reference)))
    if sha256(cache) != bank["output_sha256"] or registration != load_config(REGISTRATION):
        raise ValueError("Teacher cache or preregistration changed during smoke")
    receipt = {
        "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.monotonic() - start, "audit_rows": 0, "audit_scored": False,
        "rows": 1024, "fit_rows": len(fit_rows), "heldout_rows": len(held_rows),
        "registered_run": registered, "smoke_run": smoke_run, "fixed_epochs": rounds,
        "feature_columns": expected_columns, "teacher_logit_max_abs_difference": teacher_error,
        "native_cpu_batch_max_abs_difference": parity, "rtol": 2e-5, "atol": 2e-6,
        "source_hashes": {name: sha256(ROOT / name) for name in [
            "scripts/smoke_third_teacher.py", "scripts/common.py", "scripts/original_realmlp_teacher.py",
            "scripts/realmlp.py", "scripts/realmlp_categorical.py", "scripts/categorical_transform.py",
            "configs/third_pass_teacher_reserve.json", "data/splits.parquet",
            "artifacts/original_realmlp_teacher/manifest.json"]},
        "cache_sha256": bank["output_sha256"], "selection_sha256": sha256(OUT / "selection.json"),
        "native_hashes": {p.relative_to(OUT).as_posix(): sha256(p) for p in (OUT / "model").rglob("*") if p.is_file()},
        "limitation": "Execution and serialization smoke only; no validation performance or gain claimed",
    }
    atomic_json(OUT / "smoke_receipt.json", receipt)
    print({"status": "passed", "seconds": receipt["seconds"], "native_cpu_parity": parity}, flush=True)
    return receipt


def main() -> None:
    def expired():
        print("Teacher student smoke exceeded its fixed 120-second wall-clock budget", flush=True)
        os._exit(124)

    watchdog = threading.Timer(TIMEOUT_SECONDS, expired)
    watchdog.daemon = True
    watchdog.start()
    try:
        run()
    finally:
        watchdog.cancel()


if __name__ == "__main__":
    main()
