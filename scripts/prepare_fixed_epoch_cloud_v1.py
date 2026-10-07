"""Prepare an immutable, allowlisted development-only Kaggle package; never upload.

Uses one Arrow CPU thread and bounded batches. Filtering frozen development IDs
precedes any train-label read. No fitting, scoring or Kaggle API calls occur here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ID = "fixed_epoch_cloud_v1"
CLOUD = "cloud/fixed_epoch_v1"
DATASET = "clarkkitchen/s6e10-fixed-epoch-bundle-20261003"
KERNEL = "clarkkitchen/s6e10-fixed-epoch-20261003"
PARENT = "artifacts/fixed_epoch_v1/registry.json"
PARENT_SHA = "3118735d18dc0bd77145bfcefd42344c3fcc896e8007e38026807cfd60c2fef0"
BOOTSTRAP = "artifacts/kaggle_cloud/fold_control_bundle/payload/source/reviewed_bootstrap.py"
BOOTSTRAP_SHA = "5886a5301b1b5d0dddcb10499844d13bc38db9b6e16048b10d9deed132865c68"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
SPLIT_SHA = "4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c"
INCUMBENT_SHA = "bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311"
DEVELOPMENT_ROWS = 629671
BUDGET = {"hard_timeout_seconds": 7200, "fit_budget_seconds": 6300, "remote_smoke_timeout_seconds": 330}
EVALUATION = {"mixture_alpha": .1, "min_pooled_gain": 1e-5, "min_macro_gain": 1e-5,
              "max_fold_regression": 2e-5, "class_order": [0, 1], "fold_ids": [0, 1, 2]}
FROZEN_SOURCE = ["common.py", "categorical_transform.py", "realmlp.py", "realmlp_categorical.py",
                 "supervisor.py", "long_local_500_policy.py"]
NEW_SOURCE = ["run_fixed_epoch_cloud_v1.py", "fixed_epoch_cloud_adapter_v1.py", "smoke_fixed_epoch_cloud_v1.py"]
POLICY = "configs/fixed_epoch_cloud_policy_v1.json"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def checked(root: Path, path: str | Path) -> Path:
    path = (root / Path(str(path).replace("\\", "/"))).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes workspace")
    return path


def relative(root: Path, path: Path) -> str:
    return checked(root, path).relative_to(root.resolve()).as_posix()


def require_hash(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"Frozen input changed: {path}")


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def copy_bound(source: Path, destination: Path, expected: str) -> None:
    require_hash(source, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    require_hash(destination, expected)


def development_split(path: Path, expected_rows: int):
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    # Only non-target split metadata is read before establishing the ID domain.
    split = pq.read_table(path, columns=["id", "fold", "is_audit"], use_threads=False)
    ids = split["id"].to_numpy()
    folds = split["fold"].to_numpy()
    audit = split["is_audit"].to_numpy()
    if (ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids)
            or not np.isin(folds, [-1, 0, 1, 2]).all() or not np.array_equal(audit, folds < 0)):
        raise ValueError("Frozen split identity/domain differs")
    development = split.filter(pc.greater_equal(split["fold"], 0)).select(["id", "fold"])
    if len(development) != expected_rows or set(development["fold"].to_pylist()) != {0, 1, 2}:
        raise ValueError("Development fold/count contract differs")
    return development


def stream_subset(source: Path, destination: Path, ids, columns: list[str], *, role: str,
                  batch_size: int = 8192) -> dict:
    """ID-filter before columns are materialized; validate exact frozen key order."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    if role not in {"train", "aux"} or batch_size < 1:
        raise ValueError("Invalid subset role/batch size")
    ids = np.asarray(ids)
    if ids.dtype.kind not in "iu" or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("Expected nonempty unique integer development IDs")
    dataset = ds.dataset(source, format="parquet")
    expected = ["id", *[f"orig_aux_{index:02d}" for index in range(13)]] if role == "aux" else columns
    if columns != expected or any(column not in dataset.schema.names for column in columns):
        raise ValueError("Subset input schema differs")
    if role == "train" and (columns[0] != "id" or columns[-1] != "satisfaction"):
        raise ValueError("Raw development target schema differs")
    if role == "aux" and dataset.schema.names != columns:
        raise ValueError("Auxiliary bank must contain only known target-free feature columns")
    scanner = dataset.scanner(columns=columns, filter=ds.field("id").isin(ids),
        batch_size=batch_size, batch_readahead=1, fragment_readahead=1, use_threads=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    cursor, batches = 0, 0
    with pq.ParquetWriter(destination, scanner.projected_schema, compression="zstd") as writer:
        for batch in scanner.to_batches():
            if len(batch) == 0:
                continue
            actual = batch.column("id").to_numpy()
            if not np.array_equal(actual, ids[cursor:cursor + len(batch)]):
                raise ValueError("Filtered subset is duplicate, missing, unexpected or out of frozen ID order")
            if role == "train":
                labels = batch.column("satisfaction").to_numpy()
                if not np.isin(labels, [0, 1]).all():
                    raise ValueError("Development target must be binary")
            else:
                for column in columns[1:]:
                    values = batch.column(column)
                    if not pa.types.is_float32(values.type) or values.null_count or not pc.all(pc.is_finite(values)).as_py():
                        raise ValueError("Auxiliary features must be finite nonnull float32")
            writer.write_batch(batch)
            cursor += len(batch)
            batches += 1
    if cursor != len(ids):
        raise ValueError("Subset did not cover every development ID")
    # Independent readback of only keys; no raw training/audit/test decoding.
    readback = pq.read_table(destination, columns=["id"], use_threads=False)["id"].to_numpy()
    if not np.array_equal(readback, ids):
        raise ValueError("Written subset key readback failed")
    return {"rows": cursor, "batches": batches, "columns": columns, "sha256": sha(destination),
            "bytes": destination.stat().st_size,
            "id_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()}


def adapter_evidence(root: Path, path: Path) -> tuple[dict, dict]:
    receipt = read(path)
    if receipt.get("status") != "passed" or receipt.get("real_data_rows") != 0:
        raise ValueError("Portable adapter requires a passing generated-data test receipt")
    if receipt.get("returncode", 0) != 0 or receipt.get("tests_passed", receipt.get("tests", 0)) != 6:
        raise ValueError("Portable adapter tests did not complete")
    sources = receipt.get("source_hashes", receipt.get("source_sha256", {}))
    normalized = {}
    for name, expected in sources.items():
        source = checked(root, name)
        require_hash(source, expected)
        normalized[relative(root, source)] = expected
    required = {"scripts/fixed_epoch_cloud_adapter_v1.py", "scripts/test_fixed_epoch_cloud_adapter_v1.py"}
    if not required.issubset(normalized):
        raise ValueError("Portable adapter receipt omits source/test binding")
    return receipt, normalized


def archive_payload(payload: Path, archive: Path, names: list[str]) -> None:
    if len(names) != len(set(names)) or any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("Invalid archive allowlist")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for name in sorted(names):
            info = zipfile.ZipInfo("payload/" + name, (2026, 10, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with (payload / name).open("rb") as source, zipped.open(info, "w") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
    with zipfile.ZipFile(archive) as zipped:
        expected = {"payload/" + name for name in names}
        if set(zipped.namelist()) != expected or len(zipped.namelist()) != len(expected):
            raise ValueError("Archive member inventory differs")
        for name in names:
            with zipped.open("payload/" + name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sha(payload / name):
                    raise ValueError("Archive member hash differs")


def prepare(root: Path, adapter_tests: Path) -> dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    start = time.monotonic()
    cloud = root / CLOUD
    payload = cloud / "bundle/payload"
    final = cloud / "preparation_manifest.json"
    if final.exists():
        saved = read(final)
        for name, expected in saved["frozen_files"].items():
            require_hash(checked(root, name), expected)
        return saved
    if payload.parent.exists() or (root / f"configs/{ID}.json").exists() or (root / f"artifacts/{ID}/registry.json").exists():
        raise FileExistsError("Existing incomplete cloud preparation requires review; nothing is overwritten")
    parent_path = root / PARENT
    require_hash(parent_path, PARENT_SHA)
    parent = read(parent_path)
    policy_path = root / POLICY
    policy = read(policy_path)
    required_policy = {"id": ID, **BUDGET, "parent_registry_sha256": PARENT_SHA,
                       "maximum_gpu_jobs": 1, "paid_compute": False, "private_required": True,
                       "cloud_namespace": CLOUD, "dataset_id": DATASET, "kernel_id": KERNEL}
    if any(policy.get(key) != value for key, value in required_policy.items()):
        raise ValueError("Cloud authorization/budget policy differs")
    policy_hash = sha(policy_path)
    if parent["evaluation"] != EVALUATION or parent["expected_fit_count"] != 12 or parent["development_rows"] != DEVELOPMENT_ROWS:
        raise ValueError("Parent fixed scientific contract differs")
    parent_names = ["data/splits.parquet", "data/train.parquet", "data/manifest.json",
                    "data/original_aux_predictions.parquet", "artifacts/original_aux/manifest.json",
                    "artifacts/third_pass/blend/frozen.json"]
    parents = {name: parent["input_hashes"][name] for name in parent_names}
    if parents["data/splits.parquet"] != SPLIT_SHA or parents["artifacts/third_pass/blend/frozen.json"] != INCUMBENT_SHA:
        raise ValueError("Original split/incumbent pin differs")
    for name, expected in parents.items():
        require_hash(root / name, expected)
    aux_manifest = read(root / "artifacts/original_aux/manifest.json")
    if (aux_manifest.get("status") != "complete" or aux_manifest.get("synthetic_labels_read") is not False
            or aux_manifest.get("satisfaction_labels_read") is not False
            or aux_manifest["output_sha256"] != parents["data/original_aux_predictions.parquet"]):
        raise ValueError("Original auxiliary bank provenance differs")
    receipt_path = checked(root, adapter_tests)
    receipt, adapter_sources = adapter_evidence(root, receipt_path)
    runtime_path = root / "scripts/run_fixed_epoch_cloud_v1.py"
    runtime = runtime_path.read_text(encoding="utf-8")
    if runtime.count("MANIFEST_HASH_PLACEHOLDER") != 1:
        raise ValueError("Runtime requires exactly one manifest placeholder")
    compile(runtime, str(runtime_path), "exec")
    source_map = {}
    for name in FROZEN_SOURCE:
        source = root / f"artifacts/fixed_epoch_v1/registered_source/scripts/{name}"
        source_map[f"scripts/{name}"] = (source, parent["source_hashes"][f"scripts/{name}"])
    for name in NEW_SOURCE:
        source = root / "scripts" / name
        source_map[f"scripts/{name}"] = (source, sha(source))
    if source_map["scripts/fixed_epoch_cloud_adapter_v1.py"][1] != adapter_sources["scripts/fixed_epoch_cloud_adapter_v1.py"]:
        raise ValueError("Portable adapter changed since its test")
    source_map["scripts/reviewed_bootstrap.py"] = (root / BOOTSTRAP, BOOTSTRAP_SHA)
    source_map["scripts/cloud_feature_compat.py"] = (root / "scripts/cloud_feature_compat.py",
        "2be5d2e0ebcfc7699f425a8d3b672500f38531eb4277e12bb3ade3bdaf09217f")
    for source, expected in source_map.values():
        require_hash(source, expected)
    development = development_split(root / "data/splits.parquet", DEVELOPMENT_ROWS)
    ids = development["id"].to_numpy()
    raw_manifest = read(root / "data/manifest.json")
    features = raw_manifest["feature_columns"]
    if len(features) != 21 or "satisfaction" in features or "id" in features:
        raise ValueError("Raw feature contract differs")
    raw_schema = pq.read_schema(root / "data/train.parquet")
    categories = {"Gender", "Customer Type", "Type of Travel", "Class"}
    if (raw_schema.names != ["id", *features, "satisfaction"]
            or not pa.types.is_int64(raw_schema.field("id").type)
            or not pa.types.is_int8(raw_schema.field("satisfaction").type)
            or any(not pa.types.is_float32(raw_schema.field(name).type) for name in features if name not in categories)
            or any(not pa.types.is_large_string(raw_schema.field(name).type) for name in categories)):
        raise ValueError("Canonical float32 numeric/string category schema differs")
    source_hashes = {name: expected for name, (_, expected) in source_map.items()}
    prep_sources = {relative(root, Path(__file__)): sha(Path(__file__)),
                    "scripts/test_prepare_fixed_epoch_cloud_v1.py": sha(root / "scripts/test_prepare_fixed_epoch_cloud_v1.py"),
                    **adapter_sources}
    claim = {"id": ID, "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
             "parent_hashes": parents, "source_hashes": source_hashes, "preparation_source_hashes": prep_sources,
             "adapter_test_receipt_sha256": sha(receipt_path), "budget": BUDGET,
             "cloud_policy_path": POLICY, "cloud_policy_sha256": policy_hash,
             "privacy_required": True, "training_performed": False, "cloud_operations_performed": False}
    write(cloud / "preparation_claim.json", claim)
    payload.mkdir(parents=True)
    for name, (source, expected) in source_map.items():
        copy_bound(source, payload / name, expected)
    (payload / "data").mkdir()
    pq.write_table(development, payload / "data/splits.parquet", compression="zstd")
    train = stream_subset(root / "data/train.parquet", payload / "data/train.parquet", ids,
                          ["id", *features, "satisfaction"], role="train")
    gc.collect()
    aux_columns = [f"orig_aux_{index:02d}" for index in range(13)]
    aux = stream_subset(root / "data/original_aux_predictions.parquet", payload / "data/original_aux_predictions.parquet",
                        ids, ["id", *aux_columns], role="aux")
    if train["id_sha256"] != aux["id_sha256"]:
        raise ValueError("Raw/auxiliary subset key domains differ")
    data_manifest = {"competition": "playground-series-s6e10", "metric": "roc_auc", "positive_label": 1,
        "feature_columns": features, "aux_columns": aux_columns, "rows": {"development": DEVELOPMENT_ROWS, "train": DEVELOPMENT_ROWS, "audit": 0, "test": 0},
        "source_split_sha256": SPLIT_SHA, "parent_hashes": parents,
        "development_id_sha256": train["id_sha256"], "train": train, "aux": aux,
        "split_sha256": sha(payload / "data/splits.parquet"), "audit_rows_exported": 0, "test_rows_exported": 0,
        "selection": "Frozen split fold>=0 established before raw train or auxiliary table filtering; exact frozen key order checked in every batch and after write",
        "preparation_did_not_fit_or_score": True}
    write(payload / "data/manifest.json", data_manifest)
    copy_bound(root / "artifacts/original_aux/manifest.json", payload / "provenance/original_aux_manifest.json", parents["artifacts/original_aux/manifest.json"])
    copy_bound(receipt_path, payload / "provenance/portable_adapter_tests.json", sha(receipt_path))
    copy_bound(policy_path, payload / "provenance/cloud_policy.json", policy_hash)
    license_path = root / ".venv/Lib/site-packages/pytabkit-1.7.3.dist-info/licenses/LICENSE.txt"
    copy_bound(license_path, payload / "licenses/pytabkit_LICENSE.txt", sha(license_path))
    attribution = "# Private fixed A/B/C experiment\n\nCompetition development rows: CC BY 4.0, https://www.kaggle.com/competitions/playground-series-s6e10/data . No audit/test rows or incumbent OOF are included.\n\nRealMLP/PyTabKit1.7.3: https://github.com/dholzmueller/pytabkit , Apache-2.0 license retained. Original-only expected-rating feature recipe: https://www.kaggle.com/code/goodpjw2008/s6e10-realmlp-aux-task-features-lb-0-96141 . Independent implementation; notebook code not copied. Project code is owner-controlled. Private use only.\n"
    with (payload / "ATTRIBUTION.md").open("x", encoding="utf-8") as stream:
        stream.write(attribution)
    inputs = {relative(payload, path): sha(path) for path in payload.rglob("*") if path.is_file() and relative(payload, path) not in source_hashes}
    registry = {"id": ID, "schema_version": 1, "registered_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": parent["recipe"], "evaluation": EVALUATION, "expected_fit_count": 12,
        "expected_outer_endpoint_count": 9, "expected_total_epochs": 120, "development_rows": DEVELOPMENT_ROWS,
        "source_hashes": source_hashes, "input_hashes": inputs, "source_split_sha256": SPLIT_SHA,
        "incumbent_selection_sha256": INCUMBENT_SHA, "parent_local_registry_sha256": PARENT_SHA,
        "parent_hashes": parents, "preparation_source_hashes": prep_sources,
        "cloud_policy_sha256": policy_hash,
        "arms": parent["arms"], "execution_order": parent["execution_order"],
        "inner_monitor_fraction": .1, "inner_partition_seeds": [20261005, 20261006, 20261007],
        "model_seed_policy": parent["model_seed_policy"], "acceptance": parent["acceptance"],
        "native_jit_optimized_execution": False, "native_cpu_threads": 4, **BUDGET,
        "deadline_policy": "Session-relative deadlines derived and durably recorded by runtime before bootstrap; no inherited local deadline",
        "execution_image": IMAGE, "dataset_id": DATASET, "kernel_id": KERNEL, "private_required": True,
        "training_device": "cuda:0", "remote_smoke_before_real_fits": True,
        "audit_rows_exported": 0, "test_rows_exported": 0, "incumbent_oof_exported": False,
        "cloud_quality_scoring": False, "post_download_local_evaluation_only": True,
        "prohibitions": parent["prohibitions"], "scope_amendment": "One bounded private Kaggle GPU job explicitly authorized; no local training, release promotion, submission or full-data refit",
        "evidence_limit": parent["evidence_limit"]}
    # Replace inherited cloud prohibition with the precisely bounded new authority.
    registry["prohibitions"] = [v for v in registry["prohibitions"] if v != "cloud jobs"] + ["additional cloud jobs"]
    write(payload / "registry.json", registry)
    wrapper = {"id": ID, "registry_path": "registry.json", "registry_sha256": sha(payload / "registry.json"),
        "output_dir": f"artifacts/{ID}", "split_path": "data/splits.parquet", "split_sha256": sha(payload / "data/splits.parquet"),
        "source_split_sha256": SPLIT_SHA, "incumbent_selection_sha256": INCUMBENT_SHA,
        "completed_manifest_path": f"artifacts/{ID}/completed_manifest.json", "completion_receipt_path": f"artifacts/{ID}/completion_receipt.json"}
    write(payload / f"configs/{ID}.json", wrapper)
    allowed = set(source_hashes) | set(inputs) | {"registry.json", f"configs/{ID}.json"}
    actual = {relative(payload, p) for p in payload.rglob("*") if p.is_file()}
    if actual != allowed:
        raise ValueError("Unexpected file in explicit payload allowlist")
    manifest = {"schema_version": 1, "id": ID, "dataset_id": DATASET, "private_required": True,
        "development_rows": DEVELOPMENT_ROWS, "audit_rows_exported": 0, "test_rows_exported": 0,
        "development_id_sha256": train["id_sha256"], "source_split_sha256": SPLIT_SHA,
        "parent_hashes": parents, "registry_sha256": wrapper["registry_sha256"],
        "files": {name: {"sha256": sha(payload / name), "bytes": (payload / name).stat().st_size} for name in sorted(allowed)}}
    write(payload / "bundle-manifest.json", manifest)
    manifest_hash = sha(payload / "bundle-manifest.json")
    upload = cloud / "bundle/upload"
    archive = upload / "payload.zip"
    archive_payload(payload, archive, [*sorted(allowed), "bundle-manifest.json"])
    write(upload / "dataset-metadata.json", {"id": DATASET, "title": "S6E10 Fixed Epoch Bundle 20261003",
        "licenses": [{"name": "other"}], "description": "PRIVATE development-only A/B/C experiment; no audit/test rows or incumbent predictions. See payload ATTRIBUTION.md for source terms. Do not publish."})
    deployed_runtime = runtime.replace("MANIFEST_HASH_PLACEHOLDER", manifest_hash)
    compile(deployed_runtime, "run.py", "exec")
    with (cloud / "run.py").open("x", encoding="utf-8") as stream:
        stream.write(deployed_runtime)
    write(cloud / "kernel-metadata.json", {"id": KERNEL, "title": "S6E10 Fixed Epoch 20261003",
        "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "docker_image": IMAGE, "docker_image_pinning_type": "original", "dataset_sources": [DATASET],
        "competition_sources": [], "kernel_sources": [], "model_sources": []})
    copy_bound(payload / "registry.json", root / f"artifacts/{ID}/registry.json", wrapper["registry_sha256"])
    copy_bound(payload / f"configs/{ID}.json", root / f"configs/{ID}.json", sha(payload / f"configs/{ID}.json"))
    # Detect source/parent drift across preparation before declaring upload readiness.
    for name, expected in parents.items():
        require_hash(root / name, expected)
    for source, expected in source_map.values():
        require_hash(source, expected)
    for name, expected in prep_sources.items():
        require_hash(root / name, expected)
    require_hash(receipt_path, claim["adapter_test_receipt_sha256"])
    require_hash(policy_path, policy_hash)
    frozen_paths = [p for p in cloud.rglob("*") if p.is_file()] + [root / f"artifacts/{ID}/registry.json", root / f"configs/{ID}.json", policy_path, Path(__file__)]
    prepared = {"id": ID, "status": "prepared_not_uploaded", "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET, "kernel_id": KERNEL, "private_required": True,
        "dataset_create_requires_public_false": True, "remote_privacy_readback_required_before_execution": True,
        "manifest_sha256": manifest_hash, "registry_sha256": wrapper["registry_sha256"],
        "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size, "runtime_sha256": sha(cloud / "run.py"),
        "development_rows": DEVELOPMENT_ROWS, "audit_rows_exported": 0, "test_rows_exported": 0,
        "training_performed": False, "cloud_operations_performed": False, "seconds": time.monotonic() - start,
        **BUDGET, "frozen_files": {relative(root, p): sha(p) for p in frozen_paths}}
    write(final, prepared)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-tests", type=Path, required=True)
    args = parser.parse_args()
    prepared = prepare(ROOT, args.adapter_tests)
    print(json.dumps({k: v for k, v in prepared.items() if k != "frozen_files"}, indent=2))


if __name__ == "__main__":
    main()
