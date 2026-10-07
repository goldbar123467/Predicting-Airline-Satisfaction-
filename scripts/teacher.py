"""Train a fixed original-only XGBoost teacher and export keyed synthetic scores.

Synthetic labels are never read. The full synthetic feature tables are used only
for exact-overlap exclusion verification and inference, not fitting or tuning.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, Transform, atomic_json, load_model, predict, sha256
from prepare_original import TARGET, canonical_features, exact_match_mask


def inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    original_path = root / "data/original.parquet"
    train_path = root / "data/train.csv"
    test_path = root / "data/test.csv"
    audit_path = root / "research/original_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    hashes = {
        "original_parquet_sha256": sha256(original_path),
        "synthetic_train_csv_sha256": sha256(train_path),
        "synthetic_test_csv_sha256": sha256(test_path),
        "original_audit_sha256": sha256(audit_path),
    }
    for actual, expected in [
        (hashes["original_parquet_sha256"], audit["output_sha256"]),
        (hashes["synthetic_train_csv_sha256"], audit["synthetic_train_sha256"]),
        (hashes["synthetic_test_csv_sha256"], audit["synthetic_test_sha256"]),
    ]:
        if actual != expected:
            raise ValueError("Input hash changed since original-data anti-overlap audit")
    features = [c for c in pd.read_csv(test_path, nrows=0).columns if c != "id"]
    if len(features) != 21 or features != audit["features"]:
        raise ValueError("Unexpected teacher feature schema")
    original = pd.read_parquet(original_path)
    if list(original.columns) != features + [TARGET]:
        raise ValueError("Original feature/target order differs from audited contract")
    if original.duplicated(features).any() or not original[TARGET].isin([0, 1]).all():
        raise ValueError("Original profiles are duplicated or labels are not binary")
    if original[TARGET].nunique() != 2:
        raise ValueError("Original data must contain both classes")
    # Crucially, never request the synthetic target column from disk.
    train = pd.read_csv(train_path, usecols=["id"] + features)
    test = pd.read_csv(test_path, usecols=["id"] + features)
    for label, synthetic in [("train", train), ("test", test)]:
        if synthetic["id"].isna().any() or not synthetic["id"].is_unique:
            raise ValueError(f"Invalid {label} identifiers")
        canonical = canonical_features(synthetic, features)
        if exact_match_mask(original, canonical, features).any():
            raise ValueError(f"Original has exact {label} feature-profile overlap")
    if np.intersect1d(train["id"].to_numpy(), test["id"].to_numpy()).size:
        raise ValueError("Train/test IDs overlap")
    return original, train, test, features, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--validate-only", action="store_true", help="Check data with no model fitting")
    args = parser.parse_args()
    root = args.root.resolve()
    start = time.monotonic()
    original, train, test, features, input_hashes = inputs(root)
    contract = {"original_rows": len(original), "synthetic_train_rows": len(train),
                "synthetic_test_rows": len(test), "features": len(features),
                "overlap_rows": 0, "synthetic_labels_read": False}
    print(json.dumps({"validated": contract}), flush=True)
    if args.validate_only:
        return

    import xgboost

    run = {
        "name": "original_only_fixed_xgb_teacher", "family": "xgboost",
        "seed": 0, "max_rounds": 600,
        "params": {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8,
                   "colsample_bytree": 0.5, "min_child_weight": 5, "reg_lambda": 5},
    }
    artifact_dir = root / "artifacts/teacher"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    transform = Transform("xgboost").fit(original[features])
    model = xgboost.XGBClassifier(
        n_estimators=600, objective="binary:logistic", eval_metric="auc",
        device="cuda", tree_method="hist", n_jobs=4, random_state=0,
        **run["params"],
    )
    model.fit(transform.transform(original[features]), original[TARGET].to_numpy(dtype=np.int8))
    rounds = int(model.get_booster().num_boosted_rounds())
    if rounds != 600:
        raise AssertionError("Teacher must use exactly 600 fixed boosting rounds")
    # Native reload defaults to CPU; use CPU for both inference paths so the
    # serialization check isolates persistence, rather than GPU/CPU summation.
    model.set_params(device="cpu")
    model.save_model(artifact_dir / "model.ubj")
    transform.save(artifact_dir / "transform.json")
    atomic_json(artifact_dir / "model_metadata.json", {"family": "xgboost", "rounds": rounds, "run": run})
    predictions = []
    for label, synthetic in [("train", train), ("test", test)]:
        scores = predict(model, transform, synthetic[features])
        predictions.append(pd.DataFrame({"id": synthetic["id"].to_numpy(),
                                         "teacher_probability": scores}))
        print(json.dumps({"predicted_split": label, "rows": len(scores)}), flush=True)
    combined = pd.concat(predictions, ignore_index=True)
    if not combined["id"].is_unique or len(combined) != len(train) + len(test):
        raise AssertionError("Teacher prediction identity contract failed")

    # Verify native saved model and preprocessing reproduce in-memory inference.
    reloaded, restored_transform = load_model(artifact_dir)
    probe = pd.concat([train.head(512), test.tail(257)], ignore_index=True)
    restored = predict(reloaded, restored_transform, probe[features])
    reference = probe[["id"]].merge(combined, on="id", validate="one_to_one", sort=False)
    if not np.array_equal(reference["id"].to_numpy(), probe["id"].to_numpy()):
        raise AssertionError("Native-model round-trip probe order changed")
    np.testing.assert_allclose(restored, reference["teacher_probability"], rtol=1e-6, atol=1e-7)

    destination = root / "data/teacher_predictions.parquet"
    temporary = destination.with_suffix(".tmp.parquet")
    combined.to_parquet(temporary, index=False)
    readback = pd.read_parquet(temporary)
    pd.testing.assert_frame_equal(readback, combined)
    for synthetic in [train, test]:
        aligned = synthetic[["id"]].merge(readback, on="id", how="left", validate="one_to_one", sort=False)
        if not np.array_equal(aligned["id"].to_numpy(), synthetic["id"].to_numpy()):
            raise AssertionError("Exported teacher IDs misaligned")
        if aligned["teacher_probability"].isna().any():
            raise AssertionError("Exported teacher predictions incomplete")
    temporary.replace(destination)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - start,
        "status": "complete", "contract": contract, "run": run,
        "effective_runtime": {"training_device": "cuda", "inference_device": "cpu", "tree_method": "hist", "rounds": rounds,
                              "objective": "binary:logistic", "n_jobs": 4},
        "input_hashes": input_hashes,
        "output_sha256": sha256(destination),
        "model_sha256": sha256(artifact_dir / "model.ubj"),
        "preprocessing_sha256": sha256(artifact_dir / "transform.json"),
        "code_sha256": {"teacher.py": sha256(Path(__file__)),
                        "common.py": sha256(Path(__file__).with_name("common.py")),
                        "prepare_original.py": sha256(Path(__file__).with_name("prepare_original.py"))},
        "feature_columns": features,
        "fit_data": "Only cleaned external original rows; source-only category vocabulary; no synthetic labels, validation set, early stopping, or tuning",
        "inference_data": "All synthetic train and test rows, addressed by validated unique IDs",
        "limitations": "Exact profile exclusion does not exclude near-duplicate ancestry; synthetic data shares original-source lineage. This teacher has no claimed synthetic validation score.",
        "verification": "Input hashes, repeated exact anti-overlap, fixed600rounds, probability bounds, native model+preprocess roundtrip, keyed coverage, parquet roundtrip",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "xgboost": xgboost.__version__},
    }
    atomic_json(artifact_dir / "manifest.json", manifest)
    print(json.dumps({"complete": str(destination), "rows": len(combined),
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
