"""Generated-data acceptance of the full feature/transform/adapter/native path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import common
from categorical_transform import CategoricalTransform
from fixed_epoch_cloud_adapter_v1 import fit_fixed_trajectory
from realmlp_categorical import load_realmlp_categorical
from run_fixed_epoch_cloud_v1 import cloud_features, ROOT, digest, parity, positive_probability, read, rel, write


def generated_frames():
    rng = np.random.default_rng(20261003)
    rows = 512
    raw = {"id": np.arange(rows, dtype=np.int64), "Age": rng.integers(18, 85, rows),
           "Flight Distance": rng.integers(100, 1600, rows)}
    raw.update({name: rng.integers(0, 6, rows) for name in common.RATINGS})
    for name in ("Departure Delay in Minutes", "Arrival Delay in Minutes"):
        raw[name] = rng.exponential(12, rows)
    for name, categories in (("Gender", ["Female", "Male"]),
                             ("Customer Type", ["Loyal Customer", "disloyal Customer"]),
                             ("Type of Travel", ["Business travel", "Personal Travel"]),
                             ("Class", ["Eco", "Business", "Eco Plus"])):
        raw[name] = rng.choice(categories, rows)
    frame = pd.DataFrame(raw)
    frame.loc[::37, "Arrival Delay in Minutes"] = np.nan
    frame.loc[384:, "Class"] = "monitor_only_class"
    frame["satisfaction"] = ((frame["Age"] + frame["Inflight wifi service"]) % 2).astype(np.int64)
    bank = pd.DataFrame({"id": frame.id})
    for i in range(13):
        bank[f"orig_aux_{i:02d}"] = ((frame["Age"] + i * frame["Inflight wifi service"]) % 17).astype(np.float32) / 17
    return frame, bank


def build_features(frame, bank, recipe):
    # Exercise common.features' exact bank join using a target-free generated bank.
    def generated_bank(path, *args, **kwargs):
        if Path(path) != ROOT / "data/original_aux_predictions.parquet":
            raise AssertionError("Synthetic smoke attempted a non-generated data read")
        return bank.copy()
    with patch.object(common.pd, "read_parquet", side_effect=generated_bank):
        return cloud_features(frame, recipe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/fixed_epoch_cloud_v1") or output.exists():
        raise ValueError("Use a fresh isolated smoke output directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for dispatch readiness")
    recipe = read(ROOT / "registry.json")["recipe"]
    torch.set_num_threads(recipe["params"]["threads"])
    output.mkdir(parents=True)
    registered = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": args.device,
                  "generated_rows": 512, "fit_rows": 384, "monitor_rows": 128,
                  "recipe": recipe, "maximum_seconds": 300, "real_data_rows": 0,
                  "trajectories": ["inner_C_H16", "outer_A_H4", "outer_C_H16"],
                  "tolerances": {"atol": 2e-6, "rtol": 1e-5},
                  "source_hashes": {rel(path): digest(path) for path in (
                      Path(__file__), ROOT / "scripts/fixed_epoch_cloud_adapter_v1.py",
                      ROOT / "scripts/run_fixed_epoch_cloud_v1.py", ROOT / "scripts/common.py",
                      ROOT / "scripts/categorical_transform.py")}}
    write(output / "registry.json", registered, exclusive=True)
    start = time.monotonic()
    deadline = start + 300
    frame, bank = generated_frames()
    from cloud_feature_compat import NUMERIC_COLUMNS
    frame = frame.astype({name: "float32" for name in NUMERIC_COLUMNS})
    x = build_features(frame, bank, recipe)
    transformer = CategoricalTransform("realmlp_cat").fit(x.iloc[:384])
    if "monitor_only_class" in transformer.vocab["Class"]:
        raise AssertionError("External transform leaked monitor categories")
    xt = transformer.transform(x.iloc[:384])
    xm = transformer.transform(x.iloc[384:])
    y = frame.satisfaction.to_numpy()
    params = {**recipe["params"], "seed": recipe["seed"], "device": args.device,
              "categorical_indices": transformer.categorical_indices}
    records = {}
    for name, phase, horizon, endpoints in (("inner_C", "inner", 16, [4, 16]),
                                             ("outer_A", "outer", 4, [4]),
                                             ("outer_C", "outer", 16, [4, 16])):
        directory = output / name
        directory.mkdir()
        transformer.save(directory / "transform.json")
        record = fit_fixed_trajectory(xt, y[:384], xm if phase == "inner" else None,
              y[384:] if phase == "inner" else None, params, directory,
              horizon=horizon, endpoint_epochs=endpoints,
              context={"campaign": "fixed_epoch_cloud_v1", "phase": phase, "smoke": True},
              deadline_monotonic=deadline)
        if record["status"] != "complete" or record["executed_epochs"] != horizon:
            raise AssertionError("Fixed endpoint execution failed")
        native = []
        for epoch in endpoints:
            model = load_realmlp_categorical(directory / f"epoch_{epoch:03d}", device=args.device)
            reference = positive_probability(model, transformer, x.iloc[384:], 37)
            del model
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
            rebuilt = build_features(frame.iloc[384:].drop(columns=["satisfaction"]), bank, recipe)
            restored_transform = CategoricalTransform.load(directory / "transform.json")
            loaded_model = load_realmlp_categorical(directory / f"epoch_{epoch:03d}", device=args.device)
            loaded = positive_probability(loaded_model, restored_transform, rebuilt, 17)
            cpu_model = load_realmlp_categorical(directory / f"epoch_{epoch:03d}", device="cpu")
            cpu_probability = positive_probability(cpu_model, restored_transform, rebuilt, 23)
            native.append({"epoch": epoch, **parity(reference, loaded),
                           "cpu_gpu_parity": parity(reference, cpu_probability)})
            del cpu_model
            del loaded_model
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        records[name] = {"trajectory_path": rel(directory / "trajectory.json"),
                         "trajectory_sha256": digest(directory / "trajectory.json"),
                         "initial_network_sha256": record["initial_network_sha256"],
                         "training_index_sha256": record["training_index_sha256"],
                         "split_seed": record["split_seed"], "sub_split_seed": record["sub_split_seed"],
                         "native_reload": native, "memory_end": record["memory_end"]}
    for key in ("initial_network_sha256", "training_index_sha256", "split_seed", "sub_split_seed"):
        if records["outer_A"][key] != records["outer_C"][key]:
            raise AssertionError(f"A/C initialization mismatch: {key}")
    result = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "registry_sha256": digest(output / "registry.json"), "seconds": time.monotonic() - start,
              "device": args.device, "architecture": [512, 256, 128], "n_ens": 8,
              "native_jit_optimized_execution": False,
              "feature_count": len(x.columns), "categorical_count": len(transformer.cats),
              "records": records, "actual_external_transform_exclusion": True,
              "scope": "Full generated feature-bank join, actual categorical transformer, production-sized adapter and native reload. Does not establish competition quality or full-data throughput."}
    write(output / "verification.json", result, exclusive=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
