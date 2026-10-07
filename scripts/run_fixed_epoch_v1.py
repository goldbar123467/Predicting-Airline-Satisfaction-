"""Isolated, bounded schedule/duration experiment; never publishes a release.

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

import psutil

ROOT = Path(__file__).resolve().parents[1]
ID = "fixed_epoch_v1"
OUT = ROOT / "artifacts" / ID
STATE = ROOT / "state" / ID
PYTHON = ROOT / ".venv/Scripts/python.exe"


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
    return config, registry


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
    from common import features
    from fixed_epoch_adapter_v1 import fit_fixed_trajectory
    from realmlp_categorical import load_realmlp_categorical

    config, registry = config_and_registry(campaign)
    deadline = datetime.fromisoformat(registry["fit_deadline_utc"])
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
    horizon = 4 if trajectory == "A" else 16
    endpoints = [4] if trajectory == "A" else [4, 16]
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
    write(directory / "context.json", context, exclusive=True)
    print(f"FIT {phase} fold={fold} trajectory={trajectory} H={horizon} train={len(train)} monitor={context['monitor_rows']}", flush=True)
    result = fit_fixed_trajectory(xt, y[train], xm, None if monitor is None else y[monitor],
                                  params, directory, horizon=horizon, endpoint_epochs=endpoints,
                                  context=context, deadline_monotonic=deadline_monotonic)
    if (result["status"] != "complete" or result["horizon"] != horizon
            or result["executed_epochs"] != horizon):
        raise ValueError(f"Adapter did not complete: {result['status']}")
    if trajectory == "C":
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
    if phase == "outer":
        # Label-free raw feature read for independent reconstruction on all held-out IDs.
        raw_columns = ["id", *read(ROOT / "data/manifest.json")["feature_columns"]]
        raw_held = read_rows(ROOT / "data/train.parquet", ids[held], columns=raw_columns)
        for arm, epoch in ([("A", 4)] if trajectory == "A" else [("B", 4), ("C", 16)]):
            if time.monotonic() >= deadline_monotonic:
                raise TimeoutError("Fitting/prediction deadline")
            endpoint = directory / f"epoch_{epoch:03d}"
            model = load_realmlp_categorical(endpoint, device="cuda")
            reference = positive_probability(model, transform, x.iloc[held], 32768)
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
                    for token in ("train.py", "supervisor.py", "run_fixed_epoch_v1.py", "fixed_epoch_adapter_v1.py")):
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
        fit_deadline = datetime.fromisoformat(registry["fit_deadline_utc"])
        delivery_deadline = datetime.fromisoformat(registry["delivery_deadline_utc"])
        if now() >= fit_deadline:
            raise TimeoutError("Registration expired before launch")
        receipt = {"id": ID, "controller": identity(psutil.Process()), "started_utc": now().isoformat(),
                   "campaign_sha256": digest(campaign), "registry_sha256": config["registry_sha256"],
                   "fit_deadline_utc": registry["fit_deadline_utc"],
                   "delivery_deadline_utc": registry["delivery_deadline_utc"]}
        write(STATE / "launch_receipt.json", receipt, exclusive=True)
        state = {**receipt, "status": "running", "completed_fits": 0, "active_child": None}
        write(STATE / "run_state.json", state)
        try:
            trajectories = []
            endpoint_records = []
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
                        "trajectories": trajectories, "endpoints": endpoint_records}
            write(OUT / "completed_manifest.json", manifest, exclusive=True)
            write(OUT / "completion_receipt.json", {"id": ID, "status": "completed",
                  "campaign_sha256": digest(campaign), "registry_sha256": config["registry_sha256"],
                  "completed_manifest_sha256": digest(OUT / "completed_manifest.json"),
                  "completed_fit_count": 12, "completed_endpoint_count": 9,
                  "evaluation_ready": True, "completed_utc": now().isoformat()}, exclusive=True)
            state["status"] = "verifying"
            run_child([str(PYTHON), str(ROOT / "scripts/evaluate_fixed_epoch_v1.py"),
                       "--campaign", str(campaign)], delivery_deadline,
                      ROOT / "logs" / ID / "evaluation.log", state)
            state.update(status="completed", completed_utc=now().isoformat(), phase="complete")
            write(STATE / "run_state.json", state)
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_utc=now().isoformat())
            write(STATE / "run_state.json", state)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fold", type=int, choices=[0, 1, 2])
    parser.add_argument("--phase", choices=["inner", "outer"])
    parser.add_argument("--trajectory", choices=["A", "C"])
    args = parser.parse_args()
    if args.worker:
        if args.fold is None or args.phase is None or args.trajectory is None:
            parser.error("Worker requires fold, phase and trajectory")
        worker(args.campaign.resolve(), args.fold, args.phase, args.trajectory)
    else:
        controller(args.campaign.resolve())


if __name__ == "__main__":
    main()
