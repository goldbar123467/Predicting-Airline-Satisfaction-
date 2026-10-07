"""Build an allowlisted private development-only cloud CV bundle; never launch."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / "cloud/third_pass_schedule"
DEST = CLOUD / "bundle"
DATASET = "clarkkitchen/s6e10-third-pass-schedule-data-20261002"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    final = CLOUD / "preparation_manifest.json"
    if final.exists():
        saved = json.loads(final.read_text())
        for name, item in saved["frozen_files"].items():
            if sha(ROOT / name) != item:
                raise ValueError(f"Completed preparation changed: {name}")
        print(json.dumps(saved, indent=2)); return saved
    if DEST.exists():
        raise ValueError("Existing incomplete preparation: inspect before replacing")
    started = time.monotonic()
    original = ROOT / "artifacts/runs/v2_realmlp_cat_raw_aux"
    contract = json.loads((original / "contract.json").read_text())
    source = original / "source"
    data_manifest = json.loads((ROOT / "data/manifest.json").read_text())
    if sha(ROOT / "data/splits.parquet") != contract["split_hash"]:
        raise ValueError("Frozen split changed")
    for name in ["train.parquet", "test.parquet", "manifest.json"]:
        if sha(ROOT / "data" / name) != contract["data_hashes"][name]:
            raise ValueError("Canonical data changed")
    aux_path = ROOT / "data/original_aux_predictions.parquet"
    bank_path = ROOT / "artifacts/original_aux/manifest.json"
    if sha(aux_path) != contract["original_aux_hash"] or sha(bank_path) != contract["original_aux_manifest_hash"]:
        raise ValueError("Frozen CPU expected-rating bank/cache changed")
    # Exclude audit identities BEFORE any request for the satisfaction column.
    split = pd.read_parquet(ROOT / "data/splits.parquet", columns=["id", "fold", "is_audit"])
    if not split.id.is_unique or not np.array_equal(split.is_audit, split.fold.lt(0)):
        raise ValueError("Invalid frozen split table")
    development = split.loc[split.fold.ge(0), ["id", "fold"]].copy()
    if len(development) != 629671:
        raise ValueError("Development count changed")
    raw_columns = data_manifest["feature_columns"]
    dataset = ds.dataset(ROOT / "data/train.parquet", format="parquet")
    predicate = ds.field("id").isin(development.id.to_numpy())
    raw = dataset.to_table(columns=["id", *raw_columns], filter=predicate, use_threads=False).to_pandas()
    labels = dataset.to_table(columns=["id", "satisfaction"], filter=predicate, use_threads=False).to_pandas()
    dev = development.merge(raw, on="id", validate="one_to_one", sort=False).merge(labels, on="id", validate="one_to_one", sort=False)
    dev = dev[["id", *raw_columns, "satisfaction", "fold"]]
    if not np.array_equal(dev.id, development.id) or not dev.satisfaction.isin([0, 1]).all():
        raise ValueError("Development target identity/encoding failed")
    del raw, labels
    test = pd.read_parquet(ROOT / "data/test.parquet", columns=["id", *raw_columns])
    if len(test) != 299844 or not test.id.is_unique or len(np.intersect1d(dev.id, test.id)):
        raise ValueError("Test identity failed")
    payload = DEST / "payload"
    for part in ["data", "scripts", "provenance", "licenses"]:
        (payload / part).mkdir(parents=True, exist_ok=True)
    dev.to_parquet(payload / "data/development.parquet", index=False)
    test.to_parquet(payload / "data/test.parquet", index=False)
    # Export only requested dev/test EV rows, in their exact concatenated order.
    aux = pd.read_parquet(aux_path)
    expected_ids = pd.DataFrame({"id": np.concatenate([dev.id.to_numpy(), test.id.to_numpy()])})
    selected_aux = expected_ids.merge(aux, on="id", how="left", validate="one_to_one", sort=False)
    if selected_aux.isna().any().any() or not np.array_equal(selected_aux.id, expected_ids.id):
        raise ValueError("Expected-rating cache key coverage failed")
    selected_aux.to_parquet(payload / "data/original_aux_predictions.parquet", index=False)
    aux_columns = list(selected_aux.columns[1:])
    del aux, selected_aux
    split_arrays = {}
    for k in range(3):
        outer = np.flatnonzero(dev.fold.to_numpy() != k)
        inner, stop = train_test_split(outer, test_size=.1, random_state=20261005 + k,
                                       stratify=dev.satisfaction.to_numpy()[outer])
        split_arrays[f"inner_{k}"], split_arrays[f"stop_{k}"] = inner, stop
    np.savez_compressed(payload / "data/inner_splits.npz", **split_arrays)
    for name in ["common.py", "categorical_transform.py", "realmlp.py", "realmlp_categorical.py", "original_aux.py"]:
        if sha(source / name) != contract["code_hashes"][name]:
            raise ValueError(f"Frozen v2 source snapshot changed: {name}")
        shutil.copyfile(source / name, payload / "scripts" / name)
    bootstrap = ROOT / "artifacts/kaggle_cloud/fold_control_bundle/payload/source/reviewed_bootstrap.py"
    prior = json.loads((bootstrap.parents[1] / "bundle-manifest.json").read_text())
    if sha(bootstrap) != prior["files"]["source/reviewed_bootstrap.py"]["sha256"]:
        raise ValueError("Reviewed installer changed")
    shutil.copyfile(bootstrap, payload / "scripts/reviewed_bootstrap.py")
    shutil.copyfile(bank_path, payload / "provenance/original_aux_manifest.json")
    shutil.copyfile(original / "contract.json", payload / "provenance/v2_contract.json")
    shutil.copyfile(ROOT / "research/third_pass_validation.md", payload / "provenance/third_pass_validation.md")
    shutil.copyfile(ROOT / ".venv/Lib/site-packages/pytabkit-1.7.3.dist-info/licenses/LICENSE.txt", payload / "licenses/pytabkit_LICENSE.txt")
    runs = []
    for limit in [4, 12]:
        run = dict(contract["run"], id=f"v3_cloud_realmlp_raw_aux_e{limit}", max_rounds=limit,
                   timeout_seconds=3600, execution_backend="kaggle")
        runs.append(run)
    image = json.loads((ROOT / "cloud/realmlp_probe/remote_metadata.json").read_text())["dockerImage"]
    protocol = {"campaign": "third_pass_cloud", "version": 1, "runs": runs, "execution_image": image,
        "control_id": runs[0]["id"], "candidate_id": runs[1]["id"], "hard_timeout_seconds": 3600,
        "selection_exclusions": [runs[0]["id"]], "comparison_controls": {runs[1]["id"]: runs[0]["id"]},
        "per_run_timeout_seconds": {runs[0]["id"]: 1800, runs[1]["id"]: 3000},
        "development_rows": len(dev), "test_rows": len(test), "raw_features": raw_columns,
        "aux_columns": aux_columns, "source_split_sha256": contract["split_hash"],
        "official_train_csv_sha256": data_manifest["raw_hashes"]["train.csv"],
        "full_aux_cache_sha256": contract["original_aux_hash"], "source_bank_manifest_sha256": contract["original_aux_manifest_hash"],
        "seed": 20261005, "inner_stop_fraction": .1, "inner_seed_rule": "seed+outer_fold",
        "inference_device": "cpu", "training_device": "cuda:0", "audit_labels_exported": 0,
        "matched_gate": {"pooled_gain_min": 0, "mean_fold_gain_min": 0, "worst_fold_gain_min": -0.00002},
        "blend_gate": {"alphas": [.05, .10, .20, .30], "pooled_gain_min": .00001, "mean_fold_gain_min": .00001, "worst_fold_gain_min": -.00002},
        "control_blend_eligible": False, "all_heldout_native_replay_required": True,
        "refit": "Explicit phase only after global freeze; fixed median rounds, same cloud image and full official labels then"}
    write_json(payload / "protocol.json", protocol)
    write_json(ROOT / "configs/third_pass_cloud.json", protocol)
    (payload / "ATTRIBUTION.md").write_text("# Private matched schedule experiment\n\n"
        "Kaggle Playground S6E10 development/test features and development labels: CC BY 4.0.\n"
        "https://www.kaggle.com/competitions/playground-series-s6e10/data\n"
        "No audit labels or audit feature rows are packaged. No fitting occurs during preparation.\n\n"
        "RealMLP/pytabkit1.7.3: https://github.com/dholzmueller/pytabkit, Apache-2.0 license retained.\n"
        "Auxiliary expected-rating recipe attribution: https://www.kaggle.com/code/goodpjw2008/s6e10-realmlp-aux-task-features-lb-0-96141\n"
        "Independent original-only local bank; notebook code not copied. Project sources remain owner-controlled; "
        "dataset license 'other' preserves mixed terms. Private use only.\n", encoding="utf-8")
    files = sorted(p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file())
    manifest = {"schema_version": 1, "dataset_id": DATASET, "private_required": True,
        "purpose": "Matched full-development RealMLP e4/e12 nested CV", "audit_labels_exported": 0,
        "source_train_parquet_sha256": contract["data_hashes"]["train.parquet"],
        "source_test_parquet_sha256": contract["data_hashes"]["test.parquet"],
        "source_split_sha256": contract["split_hash"], "source_aux_cache_sha256": contract["original_aux_hash"],
        "development_id_sha256": hashlib.sha256(dev.id.to_numpy(dtype="<i8").tobytes()).hexdigest(),
        "test_id_sha256": hashlib.sha256(test.id.to_numpy(dtype="<i8").tobytes()).hexdigest(),
        "files": {n: {"sha256": sha(payload / n), "bytes": (payload / n).stat().st_size} for n in files}}
    write_json(payload / "bundle-manifest.json", manifest)
    upload = DEST / "upload"; upload.mkdir()
    archive = upload / "payload.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in sorted([*files, "bundle-manifest.json"]):
            info = zipfile.ZipInfo("payload/" + name, (2026, 10, 2, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            with (payload / name).open("rb") as source_file, zipped.open(info, "w") as target:
                shutil.copyfileobj(source_file, target, 1024 * 1024)
    with zipfile.ZipFile(archive) as zipped:
        expected = {"payload/" + n for n in [*files, "bundle-manifest.json"]}
        if set(zipped.namelist()) != expected or len(zipped.namelist()) != len(expected):
            raise ValueError("Archive allowlist differs")
        for name in zipped.namelist():
            with zipped.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sha(DEST / name):
                    raise ValueError("Archive member changed")
    write_json(upload / "dataset-metadata.json", {"id": DATASET, "title": "S6E10 Third Pass Schedule Data 20261002",
        "licenses": [{"name": "other"}], "description": "Private development-only matched experiment. CC BY4.0 competition rows; "
        "Apache2 PyTabKit; project code owner-controlled. See ATTRIBUTION.md. No audit labels; do not publish."})
    manifest_hash = sha(payload / "bundle-manifest.json")
    runtime = (CLOUD / "runtime_template.py").read_text(encoding="utf-8").replace("MANIFEST_HASH_PLACEHOLDER", manifest_hash)
    (CLOUD / "run.py").write_text(runtime, encoding="utf-8")
    compile(runtime, "run.py", "exec")
    image = json.loads((ROOT / "cloud/realmlp_probe/remote_metadata.json").read_text())["dockerImage"]
    write_json(CLOUD / "kernel-metadata.json", {"id": "clarkkitchen/s6e10-third-pass-schedule-20261002",
        "title": "S6E10 Third Pass Schedule 20261002", "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "docker_image": image, "docker_image_pinning_type": "original", "dataset_sources": [DATASET],
        "competition_sources": [], "kernel_sources": [], "model_sources": []})
    frozen_paths = [*list(payload.rglob("*")), archive, upload / "dataset-metadata.json", CLOUD / "run.py",
                    CLOUD / "runtime_template.py", CLOUD / "kernel-metadata.json", Path(__file__), ROOT / "configs/third_pass_cloud.json"]
    prepared = {"status": "prepared_not_uploaded", "manifest_sha256": manifest_hash,
        "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size, "runtime_sha256": sha(CLOUD / "run.py"),
        "training_performed": False, "cloud_operations_performed": False, "development_rows": len(dev), "test_rows": len(test),
        "audit_labels_exported": 0, "hard_timeout_seconds": 3600, "seconds": time.monotonic() - started,
        "frozen_files": {p.relative_to(ROOT).as_posix(): sha(p) for p in frozen_paths if p.is_file()}}
    write_json(final, prepared)
    print(json.dumps({k: v for k, v in prepared.items() if k != "frozen_files"}, indent=2))
    return prepared


if __name__ == "__main__":
    prepare()
