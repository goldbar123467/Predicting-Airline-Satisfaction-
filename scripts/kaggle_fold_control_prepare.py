"""Prepare a private, frozen fold timing control; never upload or train."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/kaggle_cloud/fold_control_bundle"
CLOUD = ROOT / "cloud/fold_control"
DATASET = "clarkkitchen/s6e10-fold-control-bundle-20261002"
SOURCES = ("common.py", "categorical_transform.py", "realmlp.py", "realmlp_categorical.py")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_frozen(payload: Path, manifest: dict) -> None:
    actual = {p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"bundle-manifest.json"}:
        raise ValueError("Frozen payload allowlist changed")
    for name, info in manifest["files"].items():
        if sha256(payload / name) != info["sha256"]:
            raise ValueError(f"Frozen payload changed: {name}")


def prepare() -> dict:
    prepared_path = CLOUD / "preparation_manifest.json"
    payload = DEST / "payload"
    if prepared_path.exists():
        prepared = json.loads(prepared_path.read_text())
        if sha256(payload / "bundle-manifest.json") != prepared["manifest_sha256"]:
            raise ValueError("Completed preparation manifest changed")
        validate_frozen(payload, json.loads((payload / "bundle-manifest.json").read_text()))
        for key, path in [("archive_sha256", DEST / "upload/payload.zip"),
                          ("runtime_sha256", CLOUD / "run_fold_control.py"),
                          ("kernel_metadata_sha256", CLOUD / "kernel-metadata.json"),
                          ("dataset_metadata_sha256", DEST / "upload/dataset-metadata.json"),
                          ("preparation_code_sha256", Path(__file__))]:
            if sha256(path) != prepared[key]:
                raise ValueError(f"Completed preparation changed: {key}")
        print(json.dumps(prepared, indent=2))
        return prepared
    if payload.exists() and any(payload.rglob("*")):
        raise ValueError("Partial preparation exists; inspect it before any overwrite")
    started = time.monotonic()
    fold = ROOT / "artifacts/runs/v2_realmlp_cat_raw/fold_0/model"
    contract_path = fold.parents[1] / "contract.json"
    contract = json.loads(contract_path.read_text())
    run = contract["run"]
    if (run["id"] != "v2_realmlp_cat_raw" or run["seed"] != 20261005
            or not run["categorical_twins"] or any(run.get(k) for k in
                ["teacher", "route", "route_profiles", "encoding", "features", "original_aux", "original_lgb_teacher"])):
        raise ValueError("Only the preregistered raw categorical recipe is allowed")
    if sha256(ROOT / "data/train.parquet") != contract["data_hashes"]["train.parquet"]:
        raise ValueError("Canonical training parquet changed")
    if sha256(ROOT / "data/splits.parquet") != contract["split_hash"]:
        raise ValueError("Frozen splits changed")
    # Establish admissible IDs BEFORE requesting any target column from storage.
    split = pd.read_parquet(ROOT / "data/splits.parquet", columns=["id", "fold", "is_audit"])
    if not split.id.is_unique or not np.array_equal(split.is_audit, split.fold.lt(0)):
        raise ValueError("Invalid split identities or audit flag")
    development = split.loc[split.fold.ge(0), ["id", "fold"]].copy()
    outer = development.loc[development.fold.ne(0), ["id"]].copy()
    held = development.loc[development.fold.eq(0), ["id"]].iloc[:1024].copy()
    if len(outer) != 419780 or len(development) != 629671 or len(held) != 1024:
        raise ValueError("Frozen partition cardinality changed")
    if len(np.intersect1d(outer.id, held.id)):
        raise ValueError("Training and held-out IDs overlap")
    payload.mkdir(parents=True)
    for subdir in ["source", "data", "model", "licenses"]:
        (payload / subdir).mkdir()
    # Snapshot every implementation dependency before importing the copies.
    for name in SOURCES:
        source = ROOT / "scripts" / name
        if name != "common.py" and sha256(source) != contract["code_hashes"][name]:
            raise ValueError(f"Frozen model implementation changed: {name}")
        shutil.copyfile(source, payload / "source" / name)
    for name in ["transform.json", "model_metadata.json"]:
        shutil.copyfile(fold / name, payload / "model" / name)
    shutil.copyfile(contract_path, payload / "model/frozen_run_contract.json")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(payload / "source"))
    from categorical_transform import CategoricalTransform
    from common import features

    transform = CategoricalTransform.load(payload / "model/transform.json")
    raw_columns = [c for c in transform.columns if not c.endswith("_category") and c != "class_travel_gender"]
    if len(raw_columns) != 21 or len(transform.columns) != 39:
        raise ValueError("Unexpected frozen input feature schema")
    config = {**run["params"], "seed": 20261005, "epochs": 3, "device": "cuda:0",
              "categorical_indices": transform.categorical_indices}
    if json.loads((fold / "model_metadata.json").read_text())["rounds"] != 3:
        raise ValueError("Frozen outer fold selected epoch changed")
    write_json(payload / "model/fixed_config.json", config)
    arrays = {
        "train_X": np.lib.format.open_memmap(payload / "data/train_X.npy", mode="w+", dtype="float32", shape=(len(outer), 39)),
        "heldout_X": np.lib.format.open_memmap(payload / "data/heldout_X.npy", mode="w+", dtype="float32", shape=(len(held), 39)),
    }
    outer_index, held_index = pd.Index(outer.id), pd.Index(held.id)
    seen_outer, seen_held = np.zeros(len(outer), dtype=bool), np.zeros(len(held), dtype=bool)
    feature_start = time.monotonic()
    dataset = ds.dataset(ROOT / "data/train.parquet", format="parquet")
    # Feature batches are label-free, and the development filter precedes conversion.
    scanner = dataset.scanner(columns=["id", *raw_columns],
                              filter=ds.field("id").isin(development.id.to_numpy()),
                              batch_size=32768, use_threads=False)
    for batch in scanner.to_batches():
        raw = batch.to_pandas()
        # Join the frozen development split before selecting any fold's rows.
        raw = raw.merge(development, on="id", how="inner", validate="one_to_one", sort=False)
        for key, indices, seen, mask in [
            ("train_X", outer_index, seen_outer, raw.fold.ne(0)),
            ("heldout_X", held_index, seen_held, raw.id.isin(held_index)),
        ]:
            selected = raw.loc[mask, ["id", *raw_columns]]
            if selected.empty:
                continue
            positions = indices.get_indexer(selected.id)
            if (positions < 0).any() or seen[positions].any():
                raise ValueError("Unknown or duplicate feature row")
            arrays[key][positions] = transform.transform(features(selected, run))
            seen[positions] = True
    if not seen_outer.all() or not seen_held.all():
        raise ValueError("Missing requested feature rows")
    for array in arrays.values():
        array.flush()
    del arrays
    feature_seconds = time.monotonic() - feature_start
    # Predicate contains ONLY already-approved outer-training IDs. No held-out or
    # audit target is ever returned to Python or included in an exported array.
    labels = dataset.to_table(columns=["id", "satisfaction"],
                              filter=ds.field("id").isin(outer.id.to_numpy()), use_threads=False).to_pandas()
    labels = outer.merge(labels, on="id", how="left", validate="one_to_one", sort=False)
    if (not np.array_equal(labels.id.to_numpy(), outer.id.to_numpy()) or labels.satisfaction.isna().any()
            or not labels.satisfaction.isin([0, 1]).all()):
        raise ValueError("Outer-training label identity or binary encoding mismatch")
    np.save(payload / "data/train_y.npy", labels.satisfaction.to_numpy(dtype=np.int8), allow_pickle=False)
    np.save(payload / "data/train_ids.npy", outer.id.to_numpy(dtype=np.int64), allow_pickle=False)
    np.save(payload / "data/heldout_ids.npy", held.id.to_numpy(dtype=np.int64), allow_pickle=False)
    # Reuse the reviewed installer functions byte-for-byte; exclude its synthetic
    # training entry point and embedded model sources rather than importing it.
    reviewed_path = ROOT / "cloud/realmlp_probe/probe.py"
    reviewed = reviewed_path.read_text(encoding="utf-8")
    tree = ast.parse(reviewed)
    blocks = {n.name: ast.get_source_segment(reviewed, n) for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in {"digest", "version", "install_minimal"}}
    if set(blocks) != {"digest", "version", "install_minimal"}:
        raise ValueError("Reviewed installer source structure changed")
    requirements = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "REQUIREMENTS" for t in n.targets))
    bootstrap = ("import hashlib, importlib, importlib.metadata as md\nimport json, subprocess, sys\nfrom pathlib import Path\n"
                 + "REQUIREMENTS = " + repr(requirements) + "\n\n" + "\n\n".join(blocks.values()) + "\n")
    (payload / "source/reviewed_bootstrap.py").write_text(bootstrap, encoding="utf-8")
    shutil.copyfile(ROOT / ".venv/Lib/site-packages/pytabkit-1.7.3.dist-info/licenses/LICENSE.txt",
                    payload / "licenses/pytabkit_LICENSE.txt")
    (payload / "ATTRIBUTION.md").write_text(
        "# Private engineering timing control\n\nCompetition development data: Kaggle Playground S6E10, CC BY 4.0.\n"
        "https://www.kaggle.com/competitions/playground-series-s6e10/data\n"
        "https://creativecommons.org/licenses/by/4.0/\n\n"
        "Only transformed outer-training features/labels and 1024 label-free held-out development features are included. "
        "No audit rows or labels, held-out labels, credentials, environment, source-teacher bank or original data.\n\n"
        "Project code and artifacts remain owner-controlled; no new public license is granted. "
        "Private dataset metadata uses 'other' to preserve mixed licensing.\n"
        "RealMLP/pytabkit 1.7.3: https://github.com/dholzmueller/pytabkit (Apache-2.0 license retained).\n"
        "Recipe inspiration: https://www.kaggle.com/code/goodpjw2008/s6e10-catboost-ctr-glm-margin-xgb-lb-0-96127\n"
        "Independent local adapter; no third-party notebook execution or estimator pickle.\n", encoding="utf-8")
    files = sorted(p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file())
    schema = pq.read_schema(ROOT / "data/train.parquet")
    manifest = {
        "schema_version": 1, "purpose": "Existing raw RealMLP fold0 timing and native IO control; no model selection",
        "private_required": True, "dataset_id": DATASET, "train_rows": 419780, "heldout_rows": 1024,
        "n_features": 39, "matrix_dtype": "float32", "label_dtype": "int8", "id_dtype": "int64",
        "heldout_selection": "First 1024 fold==0 rows in frozen split order; no labels",
        "training_selection": "Join frozen development split (fold>=0), then outer fold!=0; read only their target IDs",
        "audit_rows_exported": 0, "heldout_labels_exported": 0, "no_model_score": True,
        "fixed_epochs": 3, "seed": 20261005, "hard_timeout_seconds": 1800,
        "raw_schema": {name: str(schema.field(name).type) for name in raw_columns},
        "matrix_columns": transform.nums + transform.cats, "categorical_indices": transform.categorical_indices,
        "source_train_parquet_sha256": contract["data_hashes"]["train.parquet"],
        "source_split_sha256": contract["split_hash"], "source_run_contract_sha256": sha256(contract_path),
        "reviewed_probe_sha256": sha256(reviewed_path),
        "installer_function_sha256": hashlib.sha256(blocks["install_minimal"].encode()).hexdigest(),
        "local_feature_transform_seconds": feature_seconds,
        "files": {name: {"sha256": sha256(payload / name), "bytes": (payload / name).stat().st_size} for name in files},
    }
    write_json(payload / "bundle-manifest.json", manifest)
    validate_frozen(payload, manifest)
    upload = DEST / "upload"
    upload.mkdir()
    archive = upload / "payload.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in sorted([*files, "bundle-manifest.json"]):
            info = zipfile.ZipInfo("payload/" + name, (2026, 10, 2, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with (payload / name).open("rb") as source, zipped.open(info, "w") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    with zipfile.ZipFile(archive) as zipped:
        expected = {"payload/" + name for name in [*files, "bundle-manifest.json"]}
        if set(zipped.namelist()) != expected or len(zipped.namelist()) != len(expected):
            raise ValueError("ZIP allowlist mismatch")
        for name in zipped.namelist():
            with zipped.open(name) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != sha256(DEST / name):
                raise ValueError("ZIP member checksum mismatch")
    write_json(upload / "dataset-metadata.json", {
        "title": "S6E10 Fold Control Bundle 20261002", "id": DATASET, "licenses": [{"name": "other"}],
        "description": "Private engineering control only. CC BY4.0 competition development data; Apache-2.0 PyTabKit; "
        "project-authored code remains owner-controlled. See ATTRIBUTION.md. No held-out/audit labels. Do not publish."})
    manifest_hash = sha256(payload / "bundle-manifest.json")
    template = (CLOUD / "run_fold_control_template.py").read_text(encoding="utf-8")
    (CLOUD / "run_fold_control.py").write_text(template.replace("MANIFEST_HASH_PLACEHOLDER", manifest_hash), encoding="utf-8")
    image = json.loads((ROOT / "cloud/realmlp_probe/remote_metadata.json").read_text())["dockerImage"]
    write_json(CLOUD / "kernel-metadata.json", {
        "id": "clarkkitchen/s6e10-realmlp-fold0-control-20261002", "title": "S6E10 RealMLP Fold0 Control 20261002",
        "code_file": "run_fold_control.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "docker_image": image, "docker_image_pinning_type": "original", "dataset_sources": [DATASET],
        "competition_sources": [], "kernel_sources": [], "model_sources": []})
    prepared = {"status": "prepared_not_uploaded", "training_performed": False, "external_operations_performed": False,
                "dataset_id": DATASET, "train_rows": len(outer), "heldout_feature_rows": len(held),
                "heldout_labels": 0, "audit_rows": 0, "fixed_epochs": 3, "single_gpu": True,
                "hard_timeout_seconds": 1800, "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
                "manifest_sha256": manifest_hash, "runtime_sha256": sha256(CLOUD / "run_fold_control.py"),
                "dataset_metadata_sha256": sha256(upload / "dataset-metadata.json"),
                "kernel_metadata_sha256": sha256(CLOUD / "kernel-metadata.json"),
                "preparation_code_sha256": sha256(Path(__file__)), "preparation_seconds": time.monotonic() - started,
                "local_validation_command": ".venv/Scripts/python.exe cloud/fold_control/run_fold_control.py --validate-only --bundle artifacts/kaggle_cloud/fold_control_bundle/payload --output artifacts/kaggle_cloud/fold_control_bundle/local_validation"}
    write_json(prepared_path, prepared)
    print(json.dumps(prepared, indent=2))
    return prepared


if __name__ == "__main__":
    prepare()
