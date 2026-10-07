"""Prepare an allowlisted private bundle. This script never uploads or launches."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "artifacts/kaggle_cloud/portability_bundle"
DATASET = "clarkkitchen/s6e10-portability-bundle-20261002"
SOURCE_NAMES = ("common.py", "categorical_transform.py", "realmlp.py", "realmlp_categorical.py")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def immutable_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to change existing prepared artifact: {path.name}")
    else:
        path.write_bytes(content)


def prepare() -> dict:
    fold = ROOT / "artifacts/runs/v2_realmlp_cat_raw/fold_0"
    model = fold / "model"
    metadata = json.loads((model / "metadata.json").read_text())
    graph_name = metadata["graph_file"]
    if Path(graph_name).name != graph_name or sha256(model / graph_name) != metadata["graph_sha256"]:
        raise ValueError("Invalid frozen native graph")
    if sha256(fold / "predictions.npz") != json.loads((fold / "done.json").read_text())["artifact_hash"]:
        raise ValueError("Frozen fold prediction checksum mismatch")
    transform = json.loads((model / "transform.json").read_text())
    raw_columns = [c for c in transform["columns"] if c not in {"class_travel_gender"}
                   and not c.endswith("_category")]
    if len(raw_columns) != 21 or "satisfaction" in raw_columns:
        raise ValueError("Unexpected raw schema")
    raw = pd.read_csv(ROOT / "data/test.csv", nrows=1024, usecols=["id", *raw_columns],
                      float_precision="round_trip")[["id", *raw_columns]]
    # Read only the ID column to prove the order used by train.py/load_data().
    parquet_ids = pd.read_parquet(ROOT / "data/test.parquet", columns=["id"])
    if not np.array_equal(raw.id.to_numpy(), parquet_ids.id.to_numpy()[:1024]):
        raise ValueError("Raw CSV and frozen pipeline test row orders differ")
    with np.load(fold / "predictions.npz", allow_pickle=False) as predictions:
        test_predictions = predictions["test"]
        if test_predictions.shape != (len(parquet_ids),):
            raise ValueError("Frozen fold test prediction count mismatch")
        expected = test_predictions[:1024].copy()
    if not raw.id.is_unique or not np.isfinite(expected).all() or ((expected < 0) | (expected > 1)).any():
        raise ValueError("Invalid test IDs or reference probabilities")
    payload = DESTINATION / "payload"
    files: dict[str, bytes] = {}
    for name in SOURCE_NAMES:
        files[f"source/{name}"] = (ROOT / "scripts" / name).read_bytes()
    for name in ("metadata.json", "model_metadata.json", "transform.json", graph_name):
        files[f"model/{name}"] = (model / name).read_bytes()
    files["data/test_1024.csv"] = raw.to_csv(index=False, float_format="%.17g").encode("utf-8")
    files["data/expected_1024.csv"] = pd.DataFrame({"id": raw.id, "expected_probability": expected}).to_csv(
        index=False, float_format="%.17g").encode("utf-8")
    files["licenses/pytabkit_LICENSE.txt"] = (ROOT / ".venv/Lib/site-packages/pytabkit-1.7.3.dist-info/licenses/LICENSE.txt").read_bytes()
    files["ATTRIBUTION.md"] = ("# Private portability bundle\n\n"
        "Competition feature rows: Predicting Airline Satisfaction, Kaggle Playground S6E10, CC BY 4.0.\n"
        "https://www.kaggle.com/competitions/playground-series-s6e10/data\n\n"
        "The model, reference predictions and pipeline code were produced in this local project for its owner. "
        "No new public license is granted for project-authored contents by this private test bundle.\n\n"
        "RealMLP uses pytabkit1.7.3; its Apache-2.0 license is retained in licenses/pytabkit_LICENSE.txt. "
        "https://github.com/dholzmueller/pytabkit\n"
        "Recipe attribution: https://www.kaggle.com/code/goodpjw2008/s6e10-catboost-ctr-glm-margin-xgb-lb-0-96127\n"
        "Independent local implementation; no public notebook or estimator pickle is included.\n\n"
        "Contains only1024 test feature rows, their model probabilities, one native graph, frozen preprocessing "
        "and four allowlisted source modules. No training/audit rows, labels, credentials, environment directory "
        "or source-teacher bank is included. This is an inference parity check, not an accuracy experiment.\n").encode()
    for name, content in files.items():
        immutable_write(payload / name, content)
    manifest = {"schema_version": 1, "purpose": "Frozen v2_realmlp_cat_raw fold0 CPU portability",
                "dataset_id": DATASET, "private_required": True, "rows": 1024,
                "raw_columns": list(raw.columns), "test_positions": {"start": 0, "stop_exclusive": 1024},
                "positive_class": 1, "rtol": 2e-5, "atol": 2e-6,
                "source_test_csv_sha256": sha256(ROOT / "data/test.csv"),
                "source_test_parquet_sha256": sha256(ROOT / "data/test.parquet"),
                "source_fold_predictions_sha256": sha256(fold / "predictions.npz"),
                "reference_contract": "predictions.npz test array follows test.parquet order; CSV prefix IDs independently checked",
                "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                          for name, content in files.items()}}
    immutable_write(payload / "bundle-manifest.json", json_bytes(manifest))
    if {p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file()} != set(files) | {"bundle-manifest.json"}:
        raise ValueError("Unexpected file in private payload")
    upload = DESTINATION / "upload"
    upload.mkdir(parents=True, exist_ok=True)
    archive = upload / "payload.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as compressed:
            for name in sorted([*files, "bundle-manifest.json"]):
                entry = zipfile.ZipInfo("payload/" + name, date_time=(2026, 10, 2, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                compressed.writestr(entry, (payload / name).read_bytes())
    with zipfile.ZipFile(archive) as compressed:
        expected_names = {"payload/" + name for name in [*files, "bundle-manifest.json"]}
        if set(compressed.namelist()) != expected_names or len(compressed.namelist()) != len(expected_names):
            raise ValueError("Archive whitelist mismatch")
        for name in compressed.namelist():
            if compressed.read(name) != (DESTINATION / name).read_bytes():
                raise ValueError("Archive contents differ from frozen payload")
    dataset_metadata = {"title": "S6E10 Portability Bundle 20261002", "id": DATASET,
                        "licenses": [{"name": "other"}],
                        "description": "Private inference test only. Competition test features are CC BY4.0; "
                        "pytabkit Apache-2.0 terms retained. Project code/model remain owner-controlled. "
                        "See ATTRIBUTION.md. No training/audit labels or credentials. Do not publish or share."}
    immutable_write(upload / "dataset-metadata.json", json_bytes(dataset_metadata))
    manifest_hash = sha256(payload / "bundle-manifest.json")
    cloud = ROOT / "cloud/portability"
    runtime = (cloud / "run_portability_template.py").read_text(encoding="utf-8").replace(
        "MANIFEST_HASH_PLACEHOLDER", manifest_hash)
    immutable_write(cloud / "run_portability.py", runtime.encode("utf-8"))
    kernel_metadata = {"id": "clarkkitchen/s6e10-raw-realmlp-portability-20261002",
                       "title": "S6E10 Raw RealMLP Portability 20261002", "code_file": "run_portability.py",
                       "language": "python", "kernel_type": "script", "is_private": True,
                       "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                       "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    probe_metadata = json.loads((ROOT / "artifacts/kaggle_cloud/probe_cpu/remote_source/kernel-metadata.json").read_text())
    probe_image = probe_metadata.get("docker_image")
    if not isinstance(probe_image, str) or not probe_image.startswith("gcr.io/kaggle-images/python@sha256:"):
        raise ValueError("Missing verified CPU probe image digest")
    kernel_metadata.update(docker_image=probe_image, docker_image_pinning_type="original")
    immutable_write(cloud / "kernel-metadata.json", json_bytes(kernel_metadata))
    prepared = {"status": "prepared_not_uploaded", "dataset_id": DATASET, "rows": 1024,
                "source_files": list(SOURCE_NAMES), "graph_file": graph_name,
                "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
                "manifest_sha256": manifest_hash, "runtime_sha256": sha256(cloud / "run_portability.py"),
                "dataset_metadata_sha256": sha256(upload / "dataset-metadata.json"),
                "kernel_metadata_sha256": sha256(cloud / "kernel-metadata.json"),
                "preparation_code_sha256": sha256(Path(__file__)),
                "visibility": "private only", "external_operations_performed": False,
                "local_command": ".venv/Scripts/python.exe cloud/portability/run_portability.py --bundle artifacts/kaggle_cloud/portability_bundle/payload --output artifacts/kaggle_cloud/portability_bundle/local_check"}
    immutable_write(cloud / "preparation_manifest.json", json_bytes(prepared))
    print(json.dumps(prepared, indent=2))
    return prepared


if __name__ == "__main__":
    prepare()
