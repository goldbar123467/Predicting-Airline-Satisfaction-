"""Read-only CPU diagnosis of saved duration experiments; never fit or blend.

Reads only development OOF, split IDs, and existing metadata. Outputs are isolated
under artifacts/research_pass_v1/epochs. Schedule values are reconstructed from
the installed scheduling module, without importing the training framework.
"""
from __future__ import annotations

import hashlib
import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_pass_v1/epochs"
RUN_IDS = (
    "v2_realmlp_cat_raw_aux", "v3_realmlp_cat_raw_aux_e60",
    "v3_realmlp_cat_raw_aux_e500", "v3_cloud_realmlp_raw_aux_e4",
    "v3_cloud_realmlp_raw_aux_e12", "v3_cloud_realmlp_raw_aux_e60",
    "realmlp_te_teacher", "realmlp_te_teacher16",
)
PAIRS = (
    (RUN_IDS[0], RUN_IDS[1]), (RUN_IDS[0], RUN_IDS[2]),
    (RUN_IDS[1], RUN_IDS[2]), (RUN_IDS[3], RUN_IDS[4]),
    (RUN_IDS[3], RUN_IDS[5]), (RUN_IDS[6], RUN_IDS[7]),
)


def synthetic_optimizer_diagnostic() -> None:
    """Compile synthetic architecture and execute zero-gradient CPU steps only."""
    import torch
    from pytabkit import RealMLP_TD_Classifier
    from pytabkit.models.alg_interfaces.base import InterfaceResources, SplitIdxs
    from pytabkit.models.data.data import DictDataset, TensorInfo
    from pytabkit.models.nn_models.base import Variable
    from pytabkit.models.optim.optimizers import AdamOptimizer
    from pytabkit.models.training.coord import HyperparamManager
    from pytabkit.models.training.lightning_modules import TabNNModule
    from pytabkit.models.training.scheduling import get_schedule

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(49173)
    path = ROOT / "artifacts/runs/v3_realmlp_cat_raw_aux_e500/fold_0/inner/metadata.json"
    metadata = read_json(path)
    config = dict(metadata["constructor"], device="cpu", n_threads=1)
    merged = RealMLP_TD_Classifier(**config).get_config()
    schema = metadata["input_schema"]
    continuous_count = len(schema["continuous_indices"])
    cardinalities = schema["cardinalities"]
    synthetic_rows = 512
    tensors = {
        "x_cont": torch.randn(synthetic_rows, continuous_count),
        "x_cat": torch.stack([torch.randint(0, size, (synthetic_rows,)) for size in cardinalities], dim=1),
        "y": torch.arange(synthetic_rows).remainder(2).reshape(-1, 1),
    }
    dataset = DictDataset(tensors, {
        "x_cont": TensorInfo(feat_shape=[continuous_count]),
        "x_cat": TensorInfo(cat_sizes=cardinalities), "y": TensorInfo(cat_sizes=[2]),
    }, device="cpu")
    split = SplitIdxs(torch.arange(synthetic_rows).reshape(1, -1), None, None,
                      split_seed=49173, sub_split_seeds=[49173], split_id=0)
    module = TabNNModule(**merged)
    module.compile_model(dataset, [split], InterfaceResources(n_threads=1, gpu_devices=[]))
    param_groups = []
    for name, parameter in module.model.named_parameters():
        factors = {key: float(value) for key, value in parameter.hyper_factors.items()}
        param_groups.append({"name": name, "scope": str(parameter.context.scope),
                             "shape": list(parameter.shape), "count": parameter.numel(),
                             "factors": factors, "device": str(parameter.device)})
    # No optimizer step on the compiled architecture and no trained model saved.
    # Scalar zero-gradient steps isolate manual decoupled weight decay exactly.
    factors_to_test = sorted(set((row["factors"].get("lr", 1.0), row["factors"].get("wd", 1.0))
                                 for row in param_groups))
    checks = []
    for lr_factor, wd_factor in factors_to_test:
        variable = Variable(torch.tensor([1.0], dtype=torch.float64),
                            hyper_factors={"lr": lr_factor, "wd": wd_factor})
        variable.grad = torch.zeros_like(variable)
        optimizer = AdamOptimizer([{"params": [variable]}], HyperparamManager(lr=0.053, wd=0.015))
        optimizer.step()
        observed = float(variable.detach().item())
        once_expected = 1.0 - 0.053 * 0.015 * lr_factor * wd_factor
        twice_expected = 1.0 - 0.053 * 0.015 * (lr_factor * wd_factor) ** 2
        np.testing.assert_allclose(observed, twice_expected, rtol=0.0, atol=1e-15)
        checks.append({"lr_factor": lr_factor, "wd_factor": wd_factor,
                       "observed_parameter_after_step": observed,
                       "factor_once_expected": once_expected, "factor_twice_expected": twice_expected,
                       "matches_double_factor": True,
                       "actual_to_single_factor_decay_ratio": lr_factor * wd_factor if wd_factor else None})

    # Integrate decay-only contraction with exact schedule samples before each
    # hypothetical batch, not a claim about trained parameter magnitudes.
    lr_schedule, wd_schedule = get_schedule("flat_anneal"), get_schedule("cos_log_15")
    dose_rows = []
    for horizon, stop_epoch, updates in ((3, 3, 1639), (4, 4, 1639), (4, 3, 1475),
                                        (60, 4, 1475), (500, 4, 1475), (500, 500, 1475)):
        sample_t = np.arange(stop_epoch * updates, dtype=np.float64) / (horizon * updates)
        lr_wd = np.array([0.053 * lr_schedule.call_time_(float(t)) *
                          0.015 * wd_schedule.call_time_(float(t)) for t in sample_t])
        for lr_factor, wd_factor in factors_to_test:
            factor = lr_factor * wd_factor
            intended_log = float(np.log1p(-lr_wd * factor).sum())
            actual_log = float(np.log1p(-lr_wd * factor ** 2).sum())
            dose_rows.append({"horizon": horizon, "stop_epoch": stop_epoch,
                              "updates_per_epoch": updates, "lr_factor": lr_factor, "wd_factor": wd_factor,
                              "actual_decay_only_retention": float(np.exp(actual_log)),
                              "single_factor_decay_only_retention": float(np.exp(intended_log)),
                              "actual_log_retention": actual_log, "single_factor_log_retention": intended_log})
    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU synthetic initialization plus scalar zero-gradient optimizer tests; no real-data fit or GPU allocation",
        "synthetic_rows": synthetic_rows, "constructor_source": str(path.relative_to(ROOT)),
        "effective_config": merged, "parameter_groups": param_groups,
        "total_parameters_across_8_members": sum(r["count"] for r in param_groups),
        "scalar_checks": checks, "decay_only_doses": dose_rows,
        "decay_dose_limitation": "Hypothetical zero-gradient retention; not trained weight norms or a causal generalization result",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [
            path, ROOT / ".venv/Lib/site-packages/pytabkit/models/optim/optimizers.py",
            ROOT / ".venv/Lib/site-packages/pytabkit/models/sklearn/default_params.py",
            ROOT / ".venv/Lib/site-packages/pytabkit/models/nn_models/models.py",
            ROOT / ".venv/Lib/site-packages/pytabkit/models/nn_models/nn.py",
            ROOT / ".venv/Lib/site-packages/pytabkit/models/nn_models/activations.py", Path(__file__),
        ]},
    }
    (OUT / "synthetic_optimizer.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"scalar_checks": checks, "parameters": output["total_parameters_across_8_members"],
                      "groups": len(param_groups)}, indent=2))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_probability": float(np.mean(p)),
        "probability_sd": float(np.std(p)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    input_hashes: dict[str, str] = {}
    split_path = ROOT / "data/splits.parquet"
    splits = pd.read_parquet(split_path, columns=["id", "fold"])
    if not splits.id.is_unique:
        raise ValueError("Split IDs are not unique")
    # Exclude audit IDs before any prediction/label read or join.
    dev = splits.loc[splits.fold >= 0].sort_values("id").reset_index(drop=True)
    input_hashes[str(split_path.relative_to(ROOT))] = digest(split_path)
    predictions: dict[str, np.ndarray] = {}
    labels = None
    rows: list[dict[str, Any]] = []
    meta_by_run: dict[str, list[dict[str, Any]]] = {}
    for run_id in RUN_IDS:
        folder = ROOT / "artifacts/runs" / run_id
        result = read_json(folder / "result.json")
        oof = pd.read_parquet(folder / "oof.parquet").sort_values("id").reset_index(drop=True)
        if not oof.id.is_unique or not oof[["id", "fold"]].equals(dev):
            raise ValueError(f"OOF IDs/folds do not equal frozen development: {run_id}")
        y = oof["satisfaction"].to_numpy(dtype=np.int64)
        if labels is None:
            labels = y
        elif not np.array_equal(labels, y):
            raise ValueError(f"OOF label disagreement: {run_id}")
        p = oof.prediction.to_numpy(dtype=np.float64)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f"Invalid probabilities: {run_id}")
        if abs(roc_auc_score(y, p) - result["oof_auc"]) > 1e-12:
            raise ValueError(f"Recorded AUC not reproducible: {run_id}")
        for name in ("result.json", "oof.parquet"):
            path = folder / name
            input_hashes[str(path.relative_to(ROOT))] = digest(path)
        # Imported cloud runs preserve provenance through their result manifest
        # and source tree rather than a local-training contract.json.
        if (folder / "contract.json").exists():
            input_hashes[str((folder / "contract.json").relative_to(ROOT))] = digest(folder / "contract.json")
        predictions[run_id] = p
        folds = []
        for k in range(3):
            inner_path = folder / f"fold_{k}/inner/metadata.json"
            outer_path = folder / f"fold_{k}/model/metadata.json"
            inner, outer = read_json(inner_path), read_json(outer_path)
            for path in (inner_path, outer_path):
                input_hashes[str(path.relative_to(ROOT))] = digest(path)
            for metadata in (inner, outer):
                if metadata["constructor"]["n_ens"] != 8:
                    raise ValueError("Unexpected ensemble size")
            if inner["best_epoch"] != outer["constructor"]["n_epochs"]:
                raise ValueError("Selected epoch differs from outer schedule horizon")
            mask = dev.fold.to_numpy() == k
            if abs(roc_auc_score(y[mask], p[mask]) - result["fold_auc"][k]) > 1e-12:
                raise ValueError("Recorded fold AUC not reproducible")
            batch = inner["constructor"]["batch_size"]
            folds.append({
                "fold": k, "inner_horizon": inner["constructor"]["n_epochs"],
                "selected_epoch": inner["best_epoch"],
                "outer_horizon": outer["constructor"]["n_epochs"],
                "inner_rows": inner["train_rows"], "inner_stop_rows": inner["validation_rows"],
                "outer_rows": outer["train_rows"],
                "inner_updates_per_epoch": inner["train_rows"] // batch,
                "outer_updates_per_epoch": outer["train_rows"] // batch,
                "inner_total_updates": inner["constructor"]["n_epochs"] * (inner["train_rows"] // batch),
                "outer_total_updates": outer["constructor"]["n_epochs"] * (outer["train_rows"] // batch),
                "inner_fit_seconds": inner["elapsed_seconds"],
                "outer_fit_seconds": outer["elapsed_seconds"],
                "inner_has_history": "history" in inner,
                "inner_constructor": inner["constructor"], "outer_constructor": outer["constructor"],
                "outer_metrics": metric_row(y[mask], p[mask]),
            })
        meta_by_run[run_id] = folds
        rows.append({"id": run_id, "run": result["run"], "seconds": result["seconds"],
                     "pooled": metric_row(y, p), "mean_fold_auc": float(np.mean(result["fold_auc"])),
                     "folds": folds})

    assert labels is not None
    paired = []
    for base_id, candidate_id in PAIRS:
        b, c = predictions[base_id], predictions[candidate_id]
        fold_rows = []
        for k in range(3):
            mask = dev.fold.to_numpy() == k
            base_meta, candidate_meta = meta_by_run[base_id][k], meta_by_run[candidate_id][k]
            fold_rows.append({
                "fold": k,
                "auc_delta": float(roc_auc_score(labels[mask], c[mask]) - roc_auc_score(labels[mask], b[mask])),
                "log_loss_delta": float(log_loss(labels[mask], c[mask]) - log_loss(labels[mask], b[mask])),
                "mean_abs_probability_difference": float(np.mean(np.abs(c[mask] - b[mask]))),
                "max_abs_probability_difference": float(np.max(np.abs(c[mask] - b[mask]))),
                "outer_constructor_equal": base_meta["outer_constructor"] == candidate_meta["outer_constructor"],
            })
        paired.append({
            "base": base_id, "candidate": candidate_id,
            "pooled_auc_delta": float(roc_auc_score(labels, c) - roc_auc_score(labels, b)),
            "mean_fold_auc_delta": float(np.mean([r["auc_delta"] for r in fold_rows])),
            "probability_correlation": float(np.corrcoef(b, c)[0, 1]),
            "folds": fold_rows,
        })

    # Direct source load avoids pytabkit's package imports and never touches CUDA.
    schedule_path = ROOT / ".venv/Lib/site-packages/pytabkit/models/training/scheduling.py"
    spec = importlib.util.spec_from_file_location("epoch_diagnostic_schedule", schedule_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load installed schedule module")
    schedule_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schedule_module)
    schedules = {name: schedule_module.get_schedule(name) for name in
                 ("flat_anneal", "cos_log_15", "invsqrtp1e-3")}
    schedule_rows = []
    for horizon in (3, 4, 12, 16, 60, 500):
        for epoch_float in sorted(set([0.0, 1.0, 2.0, 3.0, 4.0, horizon * 0.6, float(horizon)])):
            if epoch_float > horizon:
                continue
            t = epoch_float / horizon
            schedule_rows.append({
                "horizon": horizon, "epoch_float": epoch_float, "fraction": t,
                "base_lr_times_schedule": float(0.053 * schedules["flat_anneal"].call_time_(t)),
                "base_wd_times_schedule": float(0.015 * schedules["cos_log_15"].call_time_(t)),
                "dropout_probability": float(0.05 * schedules["invsqrtp1e-3"].call_time_(t)),
            })
    # Values at epoch boundaries are not logged optimizer observations. The final
    # update occurs slightly before the horizon boundary due to pre-batch update.
    np.testing.assert_allclose(schedules["flat_anneal"].call_time_(0.0), 1.0)
    np.testing.assert_allclose(schedules["flat_anneal"].call_time_(1.0), 1e-5, atol=1e-15)
    pd.DataFrame(schedule_rows).to_csv(OUT / "schedule_boundaries.csv", index=False)

    tabm_rows = []
    for name in ("tabm_cat_teacher", "tabm_cat_ple_te_teacher", "tabm_cat_teacher_wd003"):
        for fold in range(3):
            path = ROOT / "artifacts/runs" / name / f"fold_{fold}/inner/metadata.json"
            meta = read_json(path)
            input_hashes[str(path.relative_to(ROOT))] = digest(path)
            history = meta["history"]
            best = next(row for row in history if row["epoch"] == meta["best_epoch"])
            last = history[-1]
            tabm_rows.append({
                "id": name, "fold": fold, "selected_epoch": meta["best_epoch"],
                "completed_epochs": meta["epochs_completed"],
                "best_valid_auc": best["valid_auc"], "last_valid_auc": last["valid_auc"],
                "auc_change_best_to_last": last["valid_auc"] - best["valid_auc"],
                "train_loss_change_best_to_last": last["train_loss"] - best["train_loss"],
            })

    source_paths = [
        Path(__file__), ROOT / "scripts/train.py", ROOT / "scripts/common.py",
        ROOT / "scripts/realmlp_categorical.py", schedule_path,
        schedule_path.with_name("lightning_modules.py"), schedule_path.with_name("lightning_callbacks.py"),
        schedule_path.with_name("nn_creator.py"),
    ]
    for path in source_paths:
        input_hashes[str(path.relative_to(ROOT))] = digest(path)
    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved development OOF and metadata only; no fitting, GPU, audit score, test predictions, or blend search",
        "row_count": len(dev), "label_source": "Saved development OOF, consistent across all eight runs",
        "update_count_method": "floor(train_rows/batch_size) from library drop_last=True; eight members run vectorized per optimizer step",
        "runs": rows, "pairs": paired, "schedule_boundaries": schedule_rows,
        "tabm_persisted_curve_summaries": tabm_rows, "input_sha256": input_hashes,
    }
    (OUT / "diagnosis.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in rows:
        print(f'{row["id"]}: pooled={row["pooled"]["auc"]:.10f} '
              f'epochs={[x["selected_epoch"] for x in row["folds"]]} seconds={row["seconds"]:.3f}')
    print(f"Saved {OUT / 'diagnosis.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-optimizer-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.synthetic_optimizer_only:
        synthetic_optimizer_diagnostic()
    else:
        main()
