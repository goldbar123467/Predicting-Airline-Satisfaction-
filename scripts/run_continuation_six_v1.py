"""Isolated, bounded dropout intervention experiment; never publishes a release.



The controller is lightweight and starts one fresh owned process per trajectory.

Workers filter frozen development IDs before loading labels or constructing features.

All quality scoring belongs to the separate evaluator after twelve completed fits.

"""

from __future__ import annotations



import argparse

from datetime import datetime, timedelta, timezone

import gc

import hashlib

import json

import os

from pathlib import Path

import shutil

import subprocess

import sys

import time

import signal

import zipfile

import importlib.metadata

import traceback



import psutil



ROOT = Path(__file__).resolve().parents[1]

ID = os.environ.get("CONTINUATION_STAGE_ID", "fixed_epoch_continuation_01")
if ID not in {f"fixed_epoch_continuation_0{i}" for i in range(1, 7)}:
    raise ValueError("Unregistered continuation stage")

OUT = ROOT / "artifacts" / ID

STATE = ROOT / "state" / ID

PYTHON = Path(sys.executable)

EXPECTED_BUNDLE_MANIFEST = "MANIFEST_HASH_PLACEHOLDER"





def cloud_features(frame, recipe):

    from common import features

    from cloud_feature_compat import cloud_numeric_category_compat, SOURCE_VERSIONS

    return cloud_numeric_category_compat(features(frame, recipe), frame,

        run={**recipe, "execution_backend": "kaggle"}, source_versions=SOURCE_VERSIONS)





def now() -> datetime:

    return datetime.now(timezone.utc)





def digest(path: Path) -> str:

    with path.open("rb") as stream:

        return hashlib.file_digest(stream, "sha256").hexdigest()





def read(path: Path) -> dict:

    return json.loads(path.read_text(encoding="utf-8-sig"))





def write(path: Path, value: dict, *, exclusive: bool = False) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    if exclusive:

        with path.open("x", encoding="utf-8") as stream:

            json.dump(value, stream, indent=2, allow_nan=False)

            stream.write("\n")

        return

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    with temporary.open("w", encoding="utf-8") as stream:

        json.dump(value, stream, indent=2, allow_nan=False)

        stream.flush()

        os.fsync(stream.fileno())

    temporary.replace(path)





def rel(path: Path) -> str:

    return path.resolve().relative_to(ROOT).as_posix()





def local_path(value: str) -> Path:

    path = (ROOT / value).resolve()

    if not path.is_relative_to(ROOT):

        raise ValueError("Path escaped workspace")

    return path





def config_and_registry(path: Path) -> tuple[dict, dict]:

    config = read(path)

    registry_path = local_path(config["registry_path"])

    if config["id"] != ID or digest(registry_path) != config["registry_sha256"]:

        raise ValueError("Campaign/registry identity changed")

    registry = read(registry_path)

    for section in ("source_hashes", "input_hashes"):

        for name, expected in registry[section].items():

            if digest(local_path(name)) != expected:

                raise ValueError(f"Frozen {section} changed: {name}")

    if registry["expected_fit_count"] != 12 or registry["recipe"]["id"] != "v2_realmlp_cat_raw_aux":

        raise ValueError("Unexpected recipe or work budget")

    validate_protocol(registry)

    return config, registry





def validate_protocol(registry: dict) -> None:

    import re
    from continuation_six_policy_v1 import policy_for, policy_for_stage
    if re.fullmatch(r"fixed_epoch_continuation_0[1-6]", ID) is None:
        raise ValueError("Unknown sequence stage")
    stage_policy = registry["trajectory_policies"]["C"]
    if registry["trajectory_policies"]["A"] != policy_for("control") or stage_policy != policy_for_stage(ID):
        raise ValueError("Unregistered continuation policy")
    expected = {"id": ID, "expected_fit_count": 12, "expected_outer_endpoint_count": 9,

                "common_horizon_epochs": 16, "total_executed_epochs": 192,

                "intervention_after_completed_epoch": 4, "treatment_dropout_base": .05,

                "logical_endpoints": {"A": {"trajectory": "A", "epoch": 16},

                                      "B": {"trajectory": "C", "epoch": 4},

                                      "C": {"trajectory": "C", "epoch": 16}}}

    if any(registry.get(key) != value for key, value in expected.items()):

        raise ValueError("Frozen dropout intervention protocol differs")





def pre_fit_match(control_dir: Path, treatment_dir: Path, context: dict) -> dict:

    """Reject preprocessing/partition drift before any treatment fit, using arrays not ZIP timestamps."""

    import numpy as np

    prior = read(control_dir / "context.json")

    keys = ["campaign", "fold", "phase", "training_rows", "monitor_rows",

            "training_ids_sha256", "input_feature_columns", "transform_sha256", "feature_policy"]

    checks = {key: prior.get(key) == context.get(key) and key in context for key in keys}

    checks["transform_bytes"] = digest(control_dir / "transform.json") == digest(treatment_dir / "transform.json")

    expected = {"training_ids", "monitor_ids", "outer_validation_ids"}

    with np.load(control_dir / "partitions.npz", allow_pickle=False) as left, np.load(treatment_dir / "partitions.npz", allow_pickle=False) as right:

        checks["partition_schema"] = set(left.files) == set(right.files) == expected

        for key in expected:

            checks[key] = key in left and key in right and np.array_equal(left[key], right[key])

    if not all(checks.values()):

        raise ValueError(f"A/C pre-fit identity failed: {checks}")

    return {"status": "passed", "checks": checks, "before_treatment_fit": True,

            "control_context_sha256": digest(control_dir / "context.json"),

            "control_partition_sha256": digest(control_dir / "partitions.npz"),

            "treatment_partition_sha256": digest(treatment_dir / "partitions.npz"),

            "control_transform_sha256": digest(control_dir / "transform.json"),

            "treatment_transform_sha256": digest(treatment_dir / "transform.json")}





def require_prefix_match(directory: Path) -> dict:

    receipt = read(directory / "prefix_match.json")

    if (receipt.get("status") != "passed" or receipt.get("epoch") != 4

            or receipt.get("gate_completed_before_epoch5") is not True):

        raise ValueError("Treatment did not pass its epoch-four continuation gate")

    return receipt





def require_frame_ids(frame, ids, name: str):

    import numpy as np

    import pandas as pd

    if len(frame) != len(ids) or not frame.id.is_unique:

        raise ValueError(f"{name}: duplicate or missing IDs")

    order = pd.Index(frame.id).get_indexer(ids)

    if (order < 0).any() or not np.array_equal(np.sort(frame.id.to_numpy()), np.sort(ids)):

        raise ValueError(f"{name}: identifier domain differs")

    return frame.iloc[order].reset_index(drop=True)





def read_rows(path: Path, ids, *, columns=None):

    """Arrow filters exclude all other IDs before the table is converted to pandas."""

    import pyarrow.dataset as ds

    table = ds.dataset(path, format="parquet").to_table(

        columns=columns, filter=ds.field("id").isin(ids.tolist()))

    return require_frame_ids(table.to_pandas(), ids, "filtered parquet")





def load_development(registry: dict):

    import numpy as np

    import pandas as pd

    split = pd.read_parquet(ROOT / "data/splits.parquet", columns=["id", "fold"])

    if not split.id.is_unique or not np.isin(split.fold, [-1, 0, 1, 2]).all():

        raise ValueError("Frozen split domain")

    dev = split.loc[split.fold >= 0].reset_index(drop=True)

    if len(dev) != registry["development_rows"]:

        raise ValueError("Development row count")

    frame = read_rows(ROOT / "data/train.parquet", dev.id.to_numpy())

    if not np.isin(frame.satisfaction, [0, 1]).all():

        raise ValueError("Positive class mapping")

    return frame, dev





def partition_indices(y, folds, fold: int, phase: str, seed: int):

    import numpy as np

    from sklearn.model_selection import train_test_split

    outer_train = np.flatnonzero(folds != fold)

    held = np.flatnonzero(folds == fold)

    if phase == "outer":

        return outer_train, None, held

    if phase != "inner":

        raise ValueError("Unknown phase")

    train, monitor = train_test_split(outer_train, test_size=.1,

                                      random_state=seed + fold, stratify=y[outer_train])

    if set(train) & set(monitor) or set(train) & set(held) or set(monitor) & set(held):

        raise ValueError("Partition intersection")

    return train, monitor, held





def positive_probability(model, transform, x, chunk: int):

    import numpy as np

    import torch

    result = []

    if not np.array_equal(model.classes_, [0, 1]):

        raise ValueError("Prediction positive class order")

    # The generated CUDA smoke stalls on a loaded graph's second optimized call.

    # This scoped executor workaround preserves saved operators and model bytes.

    with torch.jit.optimized_execution(False):

        for start in range(0, len(x), chunk):

            probability = model.predict_proba(transform.transform(x.iloc[start:start + chunk]))

            if probability.shape != (min(chunk, len(x) - start), 2):

                raise ValueError("Prediction class/shape contract")

            if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():

                raise ValueError("Prediction probability contract")

            if not np.allclose(probability.sum(axis=1), 1, atol=2e-6, rtol=1e-5):

                raise ValueError("Prediction simplex contract")

            result.append(probability[:, 1].astype(np.float64))

    return np.concatenate(result)





def parity(reference, loaded, *, atol=2e-6, rtol=1e-5) -> dict:

    import numpy as np

    if reference.shape != loaded.shape or not np.isfinite(reference).all() or not np.isfinite(loaded).all():

        raise ValueError("Parity alignment or finite-value failure")

    error = np.abs(reference - loaded)

    maximum = float(error.max(initial=0))

    scaled = float((error / (atol + rtol * np.abs(reference))).max(initial=0))

    if scaled > 1:

        raise ValueError(f"Native reload parity failed: {maximum}, scaled={scaled}")

    return {"atol": atol, "rtol": rtol, "max_absolute_error": maximum,

            "max_scaled_error": scaled, "parity_passed": True}





def worker(campaign: Path, fold: int, phase: str, trajectory: str) -> None:

    import numpy as np

    import pandas as pd

    import torch

    from categorical_transform import CategoricalTransform

    from run_continuation_six_v1 import cloud_features as features

    from continuation_six_adapter_v1 import fit_fixed_trajectory

    from realmlp_categorical import load_realmlp_categorical



    config, registry = config_and_registry(campaign)

    deadline = datetime.fromisoformat(read(OUT / "job_runtime.json")["fit_deadline_utc"])

    remaining = (deadline - now()).total_seconds()

    if remaining <= 0:

        raise TimeoutError("Campaign fitting deadline reached")

    deadline_monotonic = time.monotonic() + remaining

    recipe = registry["recipe"]

    torch.set_num_threads(recipe["params"]["threads"])

    directory = OUT / f"fold_{fold}" / phase / trajectory

    if (directory / "done.json").exists() or (directory / "trajectory.json").exists():

        raise FileExistsError("Preserve previous trajectory; no implicit retry")

    directory.mkdir(parents=True, exist_ok=True)

    frame, split = load_development(registry)

    ids = frame.id.to_numpy()

    y = frame.satisfaction.to_numpy(dtype=np.int64)

    folds = split.fold.to_numpy()

    train, monitor, held = partition_indices(y, folds, fold, phase, recipe["seed"])

    x = features(frame, recipe)

    # Feature construction drops the target; outer labels never reach the adapter.

    transform = CategoricalTransform("realmlp_cat").fit(x.iloc[train])

    transform.save(directory / "transform.json")

    xt = transform.transform(x.iloc[train])

    xm = None if monitor is None else transform.transform(x.iloc[monitor])

    params = {**recipe["params"], "seed": recipe["seed"], "device": "cuda",

              "categorical_indices": transform.categorical_indices}

    horizon = 16

    endpoints = [4, 16]

    partition_path = directory / "partitions.npz"

    np.savez_compressed(partition_path, training_ids=ids[train],

                        monitor_ids=np.asarray([], dtype=ids.dtype) if monitor is None else ids[monitor],

                        outer_validation_ids=ids[held])

    context = {"campaign": ID, "fold": fold, "phase": phase, "trajectory": trajectory,

               "training_rows": len(train), "monitor_rows": 0 if monitor is None else len(monitor),

               "partition_sha256": digest(partition_path),

               "training_ids_sha256": hashlib.sha256(ids[train].astype("<i8").tobytes()).hexdigest(),

               "input_feature_columns": list(x.columns),

               "transform_sha256": digest(directory / "transform.json")}

    context["feature_policy"] = "pandas2.3 numeric missing category literal nan; audited cloud compatibility"

    write(directory / "context.json", context, exclusive=True)

    if trajectory == "C":

        write(directory / "pre_fit_match.json", pre_fit_match(directory.parent / "A", directory, context), exclusive=True)

    reference_features = x.iloc[held].copy() if phase == "outer" else None

    del x, frame

    gc.collect()

    print(f"FIT {phase} fold={fold} trajectory={trajectory} H={horizon} train={len(train)} monitor={context['monitor_rows']}", flush=True)

    result = fit_fixed_trajectory(xt, y[train], xm, None if monitor is None else y[monitor],

                                  params, directory, horizon=horizon, endpoint_epochs=endpoints,

                                  context=context, deadline_monotonic=deadline_monotonic,

                                  continuation_policy=registry["trajectory_policies"][trajectory],

                                  prefix_reference_dir=None if trajectory == "A" else directory.parent / "A")

    if (result["status"] != "complete" or result["horizon"] != horizon

            or result["executed_epochs"] != horizon):

        raise ValueError(f"Adapter did not complete: {result['status']}")

    if trajectory == "C":

        require_prefix_match(directory)

        control_dir = directory.parent / "A"

        control = read(control_dir / "trajectory.json")

        common_keys = ["initial_network_sha256", "split_seed", "sub_split_seed",

                       "training_index_sha256", "updates_per_epoch", "fit_probe_indices_sha256"]

        matches = {key: control[key] == result[key] for key in common_keys}

        matches["external_transform"] = digest(control_dir / "transform.json") == digest(directory / "transform.json")

        matches["training_ids"] = read(control_dir / "context.json")["training_ids_sha256"] == context["training_ids_sha256"]

        matches["input_schema"] = (read(control_dir / "epoch_004/metadata.json")["input_schema"]

                                    == read(directory / "epoch_004/metadata.json")["input_schema"])

        write(directory / "matched_control.json", {"status": "passed" if all(matches.values()) else "failed",

              "checks": matches, "control_trajectory_sha256": digest(control_dir / "trajectory.json"),

              "continued_trajectory_sha256": digest(directory / "trajectory.json")}, exclusive=True)

        if not all(matches.values()):

            raise ValueError(f"A/C initialization or preprocessing drift: {matches}")

    del xt, xm

    gc.collect()

    torch.cuda.empty_cache()

    native_receipts = []

    prefix_native_path = None

    if phase == "outer":

        # Label-free raw feature read for independent reconstruction on all held-out IDs.

        raw_columns = ["id", *read(ROOT / "data/manifest.json")["feature_columns"]]

        raw_held = read_rows(ROOT / "data/train.parquet", ids[held], columns=raw_columns)

        if trajectory == "C":

            prefix_models = [directory.parent / "A/epoch_004", directory / "epoch_004"]

            prefix_predictions = []

            prefix_artifacts = {}

            for endpoint in prefix_models:

                model = load_realmlp_categorical(endpoint, device="cuda")

                prefix_predictions.append(positive_probability(model, transform, reference_features, 8191))

                prefix_artifacts.update({rel(path): digest(path) for path in endpoint.iterdir() if path.is_file()})

                transform_path = endpoint.parent / "transform.json"

                prefix_artifacts[rel(transform_path)] = digest(transform_path)

                del model

                gc.collect()

                torch.cuda.empty_cache()

            prefix_native_path = directory.parent / "prefix_native_verify.json"

            write(prefix_native_path, {"id": ID, "status": "passed", "fold": fold,

                  "registry_sha256": config["registry_sha256"], "verification_scope": "full_outer_fold",

                  "class_order": [0, 1], "row_count": len(held), "epoch": 4,

                  "ids_sha256": hashlib.sha256(np.sort(ids[held]).astype("<i8").tobytes()).hexdigest(),

                  "reference": "control_A4", "candidate": "treatment_C4", "chunk_rows": 8191,

                  "artifact_hashes": prefix_artifacts, "quality_metrics_computed": False,

                  **parity(prefix_predictions[0], prefix_predictions[1])}, exclusive=True)

            del prefix_predictions

        for arm, epoch in ([("A", 16)] if trajectory == "A" else [("B", 4), ("C", 16)]):

            if time.monotonic() >= deadline_monotonic:

                raise TimeoutError("Fitting/prediction deadline")

            endpoint = directory / f"epoch_{epoch:03d}"

            model = load_realmlp_categorical(endpoint, device="cuda")

            reference = positive_probability(model, transform, reference_features, 32768)

            del model

            gc.collect()

            torch.cuda.empty_cache()

            reloaded_transform = CategoricalTransform.load(directory / "transform.json")

            rebuilt_features = features(raw_held, recipe)

            reloaded = load_realmlp_categorical(endpoint, device="cuda")

            loaded = positive_probability(reloaded, reloaded_transform, rebuilt_features, 8191)

            errors = parity(reference, loaded)

            del reloaded, rebuilt_features

            gc.collect()

            torch.cuda.empty_cache()

            prediction_path = directory.parent / f"predictions_{arm}.parquet"

            if prediction_path.exists():

                raise FileExistsError(prediction_path)

            prediction = pd.DataFrame({"id": ids[held], "fold": folds[held],

                                       "satisfaction": y[held], "prediction": reference})

            prediction.to_parquet(prediction_path, index=False)

            check = pd.read_parquet(prediction_path)

            if not prediction.equals(check):

                raise ValueError("Prediction parquet readback mismatch")

            artifacts = {rel(path): digest(path) for path in endpoint.iterdir() if path.is_file()}

            artifacts[rel(directory / "transform.json")] = digest(directory / "transform.json")

            receipt = {"id": ID, "status": "passed", "fold": fold, "arm": arm,

                       "registry_sha256": config["registry_sha256"],

                       "verification_scope": "full_outer_fold", "class_order": [0, 1],

                       "row_count": len(held),

                       "ids_sha256": hashlib.sha256(np.sort(ids[held]).astype("<i8").tobytes()).hexdigest(),

                       "prediction_path": rel(prediction_path), "prediction_sha256": digest(prediction_path),

                       "model_directory": rel(endpoint), "artifact_hashes": artifacts,

                       "raw_reload_verified": True, "reference_kind": "first_native_endpoint_reload",

                       "adapter_parity_scope": "fixed_probes_at_capture", **errors,

                       "chunks_reference_reload": [32768, 8191], "created_utc": now().isoformat()}

            receipt_path = directory.parent / f"native_verify_{arm}.json"

            write(receipt_path, receipt, exclusive=True)

            native_receipts.append(rel(receipt_path))

    telemetry = directory / "curves.jsonl"

    if not telemetry.exists():

        raise FileNotFoundError("Adapter telemetry missing")

    write(directory / "done.json", {"id": ID, "status": "completed", "phase": phase,

          "fold": fold, "trajectory": trajectory, "registry_sha256": config["registry_sha256"],

          "schedule_horizon_epochs": horizon, "executed_epochs": horizon,

          "checkpoint_epochs": endpoints, "telemetry_path": rel(telemetry),

          "telemetry_sha256": digest(telemetry), "partition_path": rel(partition_path),

          "partition_sha256": digest(partition_path), "adapter_receipt_path": rel(directory / "trajectory.json"),

          "adapter_receipt_sha256": digest(directory / "trajectory.json"),

          "prefix_state_path": rel(directory / "prefix_state.json"),

          "prefix_state_sha256": digest(directory / "prefix_state.json"),

          "prefix_match_path": rel(directory / "prefix_match.json") if trajectory == "C" else None,

          "prefix_match_sha256": digest(directory / "prefix_match.json") if trajectory == "C" else None,

          "pre_fit_match_path": rel(directory / "pre_fit_match.json") if trajectory == "C" else None,

          "pre_fit_match_sha256": digest(directory / "pre_fit_match.json") if trajectory == "C" else None,

          "prefix_native_path": rel(prefix_native_path) if prefix_native_path else None,

          "prefix_native_sha256": digest(prefix_native_path) if prefix_native_path else None,

          "native_receipts": native_receipts, "completed_utc": now().isoformat()}, exclusive=True)

    print(f"COMPLETE {phase} fold={fold} trajectory={trajectory}; no outer scores computed", flush=True)





def identity(process: psutil.Process) -> dict:

    return {"pid": process.pid, "create_time": process.create_time(), "command": process.cmdline()}





def assert_no_other_training() -> None:

    active = []

    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}

    for process in psutil.process_iter(["pid", "name", "cmdline"]):

        if process.pid in own or "python" not in (process.info["name"] or "").lower():

            continue

        try:

            if Path(process.cwd()).resolve() == ROOT and any(

                    token in " ".join(process.info["cmdline"] or [])

                    for token in ("train.py", "supervisor.py", "run_fixed_epoch_v1.py", "fixed_epoch_adapter_v1.py", "run_continuation_six_v1.py", "continuation_six_adapter_v1.py")):

                active.append(identity(process))

        except (psutil.NoSuchProcess, psutil.AccessDenied):

            continue

    if active:

        raise RuntimeError(f"Another project training process is active: {active}")





def run_child(command: list[str], deadline: datetime, log_path: Path, state: dict) -> None:

    from supervisor import capture_descendants, surviving_descendants, terminate_owned

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("x", encoding="utf-8") as stream:

        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,

                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        record = identity(psutil.Process(process.pid))

        state["active_child"] = record

        state["active_log"] = rel(log_path)

        try:

            while process.poll() is None:

                capture_descendants(record)

                state["updated_utc"] = now().isoformat()

                write(STATE / "run_state.json", state)

                if (STATE / "STOP").exists() or now() >= deadline:

                    terminate_owned(record)

                    raise TimeoutError("Registered deadline or explicit STOP")

                if psutil.virtual_memory().available < 512 * 1024 ** 2:

                    terminate_owned(record)

                    raise MemoryError("System available memory below registered 512 MiB reserve")

                time.sleep(2)

            capture_descendants(record)

            if surviving_descendants(record):

                terminate_owned(record)

                raise RuntimeError("Worker redirector left live descendants")

            if process.returncode != 0:

                raise RuntimeError(f"Worker exited {process.returncode}; inspect {log_path}")

        except BaseException:

            if process.poll() is None or surviving_descendants(record):

                terminate_owned(record)

            raise

        finally:

            state["last_child"] = record

            state["active_child"] = None

            write(STATE / "run_state.json", state)





def controller(campaign: Path) -> None:

    from supervisor import lifetime_lock

    with lifetime_lock(STATE / "lock"):

        config, registry = config_and_registry(campaign)

        if (OUT / "completion_receipt.json").exists() or (STATE / "launch_receipt.json").exists():

            raise FileExistsError("This campaign was already launched; do not duplicate or silently restart")

        assert_no_other_training()

        if psutil.virtual_memory().available < 2 * 1024 ** 3:

            raise MemoryError("Require at least 2 GiB available RAM before launch")

        fit_deadline = datetime.fromisoformat(read(OUT / "job_runtime.json")["fit_deadline_utc"])

        delivery_deadline = datetime.fromisoformat(read(OUT / "job_runtime.json")["delivery_deadline_utc"])

        if now() >= fit_deadline:

            raise TimeoutError("Registration expired before launch")

        receipt = {"id": ID, "controller": identity(psutil.Process()), "started_utc": now().isoformat(),

                   "campaign_sha256": digest(campaign), "registry_sha256": config["registry_sha256"],

                   "fit_deadline_utc": fit_deadline.isoformat(),

                   "delivery_deadline_utc": delivery_deadline.isoformat()}

        write(STATE / "launch_receipt.json", receipt, exclusive=True)

        state = {**receipt, "status": "running", "completed_fits": 0, "active_child": None}

        write(STATE / "run_state.json", state)

        try:

            trajectories = []

            endpoint_records = []

            prefix_records = []

            for fold in range(3):

                for phase in ("inner", "outer"):

                    for trajectory in ("A", "C"):

                        config_and_registry(campaign)

                        state.update(fold=fold, phase=phase, trajectory=trajectory)

                        command = [str(PYTHON), str(Path(__file__)), "--campaign", str(campaign),

                                   "--worker", "--fold", str(fold), "--phase", phase,

                                   "--trajectory", trajectory]

                        log_path = ROOT / "logs" / ID / f"fold_{fold}_{phase}_{trajectory}.log"

                        run_child(command, fit_deadline, log_path, state)

                        done = OUT / f"fold_{fold}" / phase / trajectory / "done.json"

                        if read(done)["status"] != "completed":

                            raise ValueError("Worker completion receipt missing")

                        trajectories.append({"phase": phase, "fold": fold, "trajectory": trajectory,

                                             "receipt_path": rel(done), "receipt_sha256": digest(done)})

                        if trajectory == "C":

                            detail = read(done)

                            require_prefix_match(done.parent)

                            prefix_records.append({"fold": fold, "phase": phase,

                                **{key: detail[key] for key in ("prefix_match_path", "prefix_match_sha256", "pre_fit_match_path", "pre_fit_match_sha256", "prefix_native_path", "prefix_native_sha256")}})

                        if phase == "outer":

                            for arm in (["A"] if trajectory == "A" else ["B", "C"]):

                                pred = OUT / f"fold_{fold}/outer/predictions_{arm}.parquet"

                                native = pred.with_name(f"native_verify_{arm}.json")

                                endpoint_records.append({"fold": fold, "arm": arm,

                                    "prediction_path": rel(pred), "prediction_sha256": digest(pred),

                                    "native_receipt_path": rel(native), "native_receipt_sha256": digest(native)})

                        state["completed_fits"] += 1

                        write(STATE / "run_state.json", state)

            manifest = {"id": ID, "campaign_sha256": digest(campaign),

                        "registry_sha256": config["registry_sha256"],

                        "trajectories": trajectories, "endpoints": endpoint_records, "prefix_records": prefix_records}

            write(OUT / "completed_manifest.json", manifest, exclusive=True)

            write(OUT / "completion_receipt.json", {"id": ID, "status": "completed",

                  "campaign_sha256": digest(campaign), "registry_sha256": config["registry_sha256"],

                  "completed_manifest_sha256": digest(OUT / "completed_manifest.json"),

                  "completed_fit_count": 12, "completed_endpoint_count": 9,

                  "completed_prefix_count": 6, "completed_prefix_native_count": 3,

                  "evaluation_ready": True, "completed_utc": now().isoformat()}, exclusive=True)

            state.update(status="training_complete", completed_utc=now().isoformat(), phase="complete",

                         outer_metrics_computed=False, assessment_location="local after verified download")

            write(STATE / "run_state.json", state)

        except BaseException as exc:

            state.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_utc=now().isoformat())

            write(STATE / "run_state.json", state)

            raise





def verify_bundle(payload: Path, expected: str) -> dict:

    manifest_path = payload / "bundle-manifest.json"

    if digest(manifest_path) != expected:

        raise ValueError("Cloud bundle manifest hash differs")

    manifest = read(manifest_path)

    actual = {p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file()}

    if actual != set(manifest["files"]) | {"bundle-manifest.json"}:

        raise ValueError("Cloud payload file allowlist differs")

    for name, item in manifest["files"].items():

        target = (payload / name).resolve()

        if not target.is_relative_to(payload.resolve()) or digest(target) != item["sha256"]:

            raise ValueError(f"Cloud payload hash/path failure: {name}")

    return manifest





def bounded_process(command: list[str], deadline: datetime, log: Path, cwd: Path) -> None:

    """Linux session ownership isolates cancellation to this experiment's child tree."""

    from supervisor import capture_descendants, terminate_owned, surviving_descendants

    with log.open("x", encoding="utf-8") as stream:

        child = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,

                                 start_new_session=True)

        record = identity(psutil.Process(child.pid))

        try:

            while child.poll() is None:

                capture_descendants(record)

                write(log.with_suffix(".ownership.json"), record)

                if now() >= deadline:

                    raise TimeoutError(f"Cloud child deadline: {log.name}")

                time.sleep(1)

            if surviving_descendants(record):

                raise RuntimeError("Exited cloud leader left live owned descendants")

            if child.returncode:

                raise RuntimeError(f"Cloud child exited {child.returncode}: {log.name}")

        finally:

            if child.poll() is None or surviving_descendants(record):

                terminate_owned(record)

            write(log.with_suffix(".ownership.json"), record)

            if surviving_descendants(record):

                raise RuntimeError("Owned cloud descendant survived cleanup")





def save_return_bundle(payload: Path, destination: Path) -> dict:

    names = []

    for part in (f"artifacts/{ID}", f"state/{ID}", f"logs/{ID}"):

        for path in (payload / part).rglob("*"):

            if path.is_file() and path.name != "lock":

                names.append(path.relative_to(payload).as_posix())

    names.extend(["registry.json", f"configs/{ID}.json"])

    files = {name: {"sha256": digest(payload / name), "bytes": (payload / name).stat().st_size}

             for name in sorted(set(names))}

    manifest = {"id": ID, "created_utc": now().isoformat(), "files": files,

                "scope": "Cloud artifacts and receipts only; no audit/test predictions or outer quality scoring"}

    write(destination / "output-manifest.json", manifest, exclusive=True)

    with zipfile.ZipFile(destination / "results.zip", "x", compression=zipfile.ZIP_DEFLATED,

                         compresslevel=1) as archive:

        for name in files:

            archive.write(payload / name, name)

        archive.write(destination / "output-manifest.json", "output-manifest.json")

    return {"archive_sha256": digest(destination / "results.zip"),

            "manifest_sha256": digest(destination / "output-manifest.json"), "file_count": len(files)}





def install_deterministic_runtime(payload: Path, out: Path) -> None:
    registration = payload / "provenance/execution_policy.json"
    hook = payload / "scripts/continuation_six_deterministic_v1.py"
    amendment = read(registration)
    if (amendment["campaign"] != ID or digest(hook) != amendment["hook_source_sha256"]
            or amendment["execution_policy"]["torch_deterministic_algorithms"] is not True):
        raise ValueError("Frozen deterministic execution policy differs")
    hook_dir = payload.parent / "deterministic_startup"
    hook_dir.mkdir(exist_ok=False)
    evidence = out / "runtime_amendment"
    evidence.mkdir(exist_ok=False)
    shutil.copyfile(hook, hook_dir / "sitecustomize.py")
    shutil.copyfile(hook, evidence / "sitecustomize.py")
    shutil.copyfile(registration, evidence / "registration.json")
    os.environ["CONTINUATION_STAGE_ID"] = ID
    os.environ["FIXED_EPOCH_DETERMINISTIC_ROOT"] = str(payload.resolve())
    os.environ["FIXED_EPOCH_DETERMINISTIC_RECEIPTS"] = str((evidence / "processes").resolve())
    os.environ["FIXED_EPOCH_DETERMINISTIC_AMENDMENT_SHA256"] = digest(registration)
    os.environ["FIXED_EPOCH_DETERMINISTIC_HOOK_SHA256"] = digest(hook)
    os.environ["PYTHONPATH"] = str(hook_dir.resolve()) + os.pathsep + str((payload / "scripts").resolve())


def bootstrap() -> None:

    if sys.platform != "linux" or not Path("/kaggle/input").is_dir():

        raise RuntimeError("Cloud entry point runs only inside the private Kaggle job")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.environ["OMP_NUM_THREADS"] = "4"

    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    sys.dont_write_bytecode = True

    started = now()

    destination = Path("/kaggle/working/fixed_epoch_return")

    destination.mkdir(exist_ok=False)

    extract = Path("/kaggle/working/fixed_epoch_workspace")

    extract.mkdir(exist_ok=False)

    payload = extract / "payload"

    result = {"id": ID, "status": "starting", "started_utc": started.isoformat(),

              "local_training": False, "outer_metrics_computed": False}

    write(destination / "cloud_status.json", result)

    try:

        matches = [p.parent.resolve() for p in Path("/kaggle/input").rglob("bundle-manifest.json")

                   if digest(p) == EXPECTED_BUNDLE_MANIFEST]

        if len(matches) != 1:

            raise ValueError("Require exactly one hash-matched mounted private payload")

        verify_bundle(matches[0], EXPECTED_BUNDLE_MANIFEST)

        shutil.copytree(matches[0], payload)

        manifest = verify_bundle(payload, EXPECTED_BUNDLE_MANIFEST)

        # Only hash-verified payload modules may be imported by the outer entry.
        sys.path.insert(0, str(payload / "scripts"))

        registry = read(payload / "registry.json")

        if (registry["id"] != ID or registry["hard_timeout_seconds"] != 7200

                or registry["fit_budget_seconds"] != 6300 or registry["expected_fit_count"] != 12):

            raise ValueError("Cloud execution budget/protocol differs")

        validate_protocol(registry)

        out = payload / "artifacts" / ID

        out.mkdir(parents=True, exist_ok=False)

        runtime = {"started_utc": started.isoformat(),

                   "fit_deadline_utc": (started + timedelta(seconds=6300)).isoformat(),

                   "delivery_deadline_utc": (started + timedelta(seconds=7140)).isoformat(),

                   "hard_timeout_seconds": 7200, "bundle_manifest_sha256": EXPECTED_BUNDLE_MANIFEST,

                   "registry_sha256": digest(payload / "registry.json")}

        write(out / "job_runtime.json", runtime, exclusive=True)

        from reviewed_bootstrap import install_minimal

        install_minimal(out, result)

        result["versions"] = {name: importlib.metadata.version(name) for name in

            ("torch", "numpy", "pandas", "scikit-learn", "pytabkit", "pytorch-lightning")}

        if result["versions"]["numpy"] != "2.0.2" or result["versions"]["pandas"] != "2.3.3":

            raise ValueError("Unreviewed cloud preprocessing stack")

        result["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",

                                                  "--format=csv,noheader"], text=True).strip()

        write(out / "environment.json", result, exclusive=True)

        install_deterministic_runtime(payload, out)
        smoke_deadline = min(now() + timedelta(seconds=registry["remote_smoke_timeout_seconds"]),

                             datetime.fromisoformat(runtime["fit_deadline_utc"]))

        bounded_process([sys.executable, str(payload / "scripts/smoke_continuation_six_v1.py"),

                         "--device", "cuda", "--output", str(out / "smoke_cuda")],

                        smoke_deadline, out / "smoke.log", payload)

        smoke = read(out / "smoke_cuda/verification.json")

        if smoke["status"] != "passed":

            raise ValueError("Cloud full-path synthetic smoke failed")

        bounded_process([sys.executable, str(payload / "scripts/run_continuation_six_v1.py"),

                         "--execute", "--campaign", str(payload / f"configs/{ID}.json")],

                        datetime.fromisoformat(runtime["fit_deadline_utc"]) + timedelta(seconds=20),

                        out / "controller.log", payload)

        state = read(payload / "state" / ID / "run_state.json")

        if state["status"] != "training_complete" or state["completed_fits"] != 12:

            raise ValueError("Cloud experiment did not complete all twelve fits")

        result.update(status="training_complete", completed_utc=now().isoformat(), completed_fits=12)

    except BaseException as error:

        result.update(status="failed", failed_utc=now().isoformat(),

                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())

        print(result["traceback"], flush=True)

    finally:

        result["finished_utc"] = now().isoformat()

        write(destination / "cloud_status.json", result)

        if (payload / "registry.json").is_file():

            result["returned_artifacts"] = save_return_bundle(payload, destination)

            write(destination / "cloud_status.json", result)

        print(json.dumps(result, indent=2, allow_nan=False), flush=True)





def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--campaign", type=Path)

    parser.add_argument("--worker", action="store_true")

    parser.add_argument("--execute", action="store_true")

    parser.add_argument("--fold", type=int, choices=[0, 1, 2])

    parser.add_argument("--phase", choices=["inner", "outer"])

    parser.add_argument("--trajectory", choices=["A", "C"])

    args = parser.parse_args()

    if args.worker:

        if args.campaign is None or args.fold is None or args.phase is None or args.trajectory is None:

            parser.error("Worker requires campaign/fold/phase/trajectory")

        worker(args.campaign.resolve(), args.fold, args.phase, args.trajectory)

    elif args.execute:

        if args.campaign is None:

            parser.error("Controller requires campaign")

        controller(args.campaign.resolve())

    else:

        bootstrap()





if __name__ == "__main__":

    main()





