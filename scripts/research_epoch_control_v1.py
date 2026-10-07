"""Synthetic CPU prototype: separate schedule horizon from training cutoff.

This is not a production training adapter. It runs only generated toy data and
preserves the installed pytabkit optimizer's historical semantics. No files from
data/, production configs, native releases, or campaign state are read/written.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import torch
import pytorch_lightning as pl
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.alg_interfaces.base import InterfaceResources, SplitIdxs
from pytabkit.models.data.data import DictDataset, TensorInfo
from pytabkit.models.training.lightning_callbacks import StopAtEpochsCallback
from pytabkit.models.training.lightning_modules import TabNNModule

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_pass_v1/epochs/control_prototype"


@dataclass(frozen=True)
class EpochPolicy:
    schedule_horizon_epochs: int
    stop_epoch: int
    snapshot_epoch: int

    def __post_init__(self) -> None:
        if not 1 <= self.snapshot_epoch <= self.stop_epoch <= self.schedule_horizon_epochs:
            raise ValueError("Require 1 <= snapshot_epoch <= stop_epoch <= schedule horizon")


def schedule_clock_comparison(inner_rows: int, outer_rows: int, batch_size: int,
                              horizon_epochs: int, selected_epoch: int) -> dict[str, float | int]:
    """Compare equal-epoch and equal-update budgets for drop_last=True loaders."""
    if min(inner_rows, outer_rows, batch_size, horizon_epochs, selected_epoch) < 1:
        raise ValueError("All clock inputs must be positive")
    if selected_epoch > horizon_epochs:
        raise ValueError("Selected epoch exceeds horizon")
    inner_updates = inner_rows // batch_size
    outer_updates = outer_rows // batch_size
    if min(inner_updates, outer_updates) == 0:
        raise ValueError("Need at least one full batch")
    return {
        "inner_updates_per_epoch": inner_updates, "outer_updates_per_epoch": outer_updates,
        "selected_inner_updates": selected_epoch * inner_updates,
        "equal_epoch_outer_updates": selected_epoch * outer_updates,
        "equal_update_outer_epochs": selected_epoch * inner_updates / outer_updates,
        "equal_epoch_update_ratio": outer_updates / inner_updates,
        "epoch_clock_selected_fraction": selected_epoch / horizon_epochs,
        "fixed_update_clock_fraction_at_equal_epoch_stop": selected_epoch * outer_updates / (horizon_epochs * inner_updates),
    }


def toy_dataset() -> DictDataset:
    generator = torch.Generator(device="cpu").manual_seed(81271)
    continuous = torch.randn(256, 6, generator=generator)
    categorical = torch.stack([torch.randint(0, n, (256,), generator=generator) for n in (4, 24)], dim=1)
    noise = torch.randn(256, generator=generator) * 0.2
    labels = (continuous[:, 0] + 0.3 * continuous[:, 1] + noise > 0).long().reshape(-1, 1)
    return DictDataset({"x_cont": continuous, "x_cat": categorical, "y": labels},
                       {"x_cont": TensorInfo(feat_shape=[6]), "x_cat": TensorInfo(cat_sizes=[4, 24]),
                        "y": TensorInfo(cat_sizes=[2])}, device="cpu")


def probabilities(module: TabNNModule, dataset: DictDataset) -> torch.Tensor:
    was_training = module.model.training
    module.model.eval()
    with torch.no_grad():
        prepared = module.creator.static_model(dataset)
        batch = {key: value.unsqueeze(0) for key, value in prepared.tensors.items()}
        result = module.model(batch)["x_cont"].softmax(-1).mean(0).clone()
    module.model.train(was_training)
    return result


class CapturePrefix(pl.Callback):
    def __init__(self, epoch: int, dataset: DictDataset):
        self.epoch = epoch
        self.dataset = dataset
        self.batch_schedule: list[dict[str, float | int]] = []
        self.state: dict[str, torch.Tensor] | None = None
        self.probabilities: torch.Tensor | None = None
        self.completed_epochs = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        # This callback follows the library's HyperparamCallback in the list.
        values = pl_module.hp_manager.get_hyper_sched_values()
        self.batch_schedule.append({
            "update_before": int(trainer.global_step),
            "epoch_float": float(pl_module.progress.epoch_float),
            "fraction": float(pl_module.progress.get_fit_progress()),
            "lr_schedule": float(values["lr"][""]),
            "wd_schedule": float(values["wd"][""]),
            "dropout_schedule": float(values["p_drop"][""]),
        })

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self.completed_epochs = int(trainer.current_epoch + 1)
        if self.completed_epochs == self.epoch:
            self.state = {key: value.detach().cpu().clone() for key, value in pl_module.model.state_dict().items()}
            self.probabilities = probabilities(pl_module, self.dataset)


def execute_toy(policy: EpochPolicy, use_library_stop: bool) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(23197)
    dataset = toy_dataset()
    config = RealMLP_TD_Classifier(
        device="cpu", n_threads=1, verbosity=0, n_epochs=policy.schedule_horizon_epochs,
        n_ens=1, hidden_sizes=[16, 8], batch_size=32, predict_batch_size=256,
        lr=0.053, wd=0.015, lr_sched="flat_anneal", wd_sched="cos_log_15",
        p_drop=0.05, p_drop_sched="invsqrtp1e-3", sq_mom=0.988,
        act="silu", first_layer_lr_factor=0.25, add_front_scale=False,
        plr_lr_factor=0.1151, ls_eps=0.0, n_cv=1, n_refit=0,
        use_early_stopping=False, use_best_mean_epoch_for_cv=True,
        val_metric_name="1-auc_ovr", ens_av_before_softmax=False,
    ).get_config()
    fit_params = [{"stop_epoch": {"1-auc_ovr": policy.stop_epoch}}] if use_library_stop else None
    module = TabNNModule(fit_params=fit_params, **config)
    split = SplitIdxs(torch.arange(256).reshape(1, -1), None, None,
                      split_seed=23197, sub_split_seeds=[23197], split_id=0)
    module.compile_model(dataset, [split], InterfaceResources(n_threads=1, gpu_devices=[]))
    if any(parameter.device.type != "cpu" for parameter in module.model.parameters()):
        raise AssertionError("Synthetic test must remain CPU-only")
    callbacks = module.create_callbacks()
    stop_callback_count = sum(isinstance(callback, StopAtEpochsCallback) for callback in callbacks)
    if stop_callback_count != int(use_library_stop):
        raise AssertionError("Unexpected library stopping callback")
    capture = CapturePrefix(policy.snapshot_epoch, dataset)
    trainer = pl.Trainer(accelerator="cpu", devices=1,
                         max_epochs=policy.schedule_horizon_epochs,
                         callbacks=[*callbacks, capture], logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0,
                         deterministic=True, log_every_n_steps=1)
    trainer.fit(module, train_dataloaders=module.train_dl, val_dataloaders=module.val_dl)
    if capture.state is None or capture.probabilities is None:
        raise AssertionError("Requested prefix was not captured")
    final_state = {key: value.detach().cpu().clone() for key, value in module.model.state_dict().items()}
    return {"state": capture.state, "probabilities": capture.probabilities,
            "final_state": final_state, "batch_schedule": capture.batch_schedule,
            "completed_epochs": capture.completed_epochs, "global_steps": trainer.global_step,
            "stop_callback_count": stop_callback_count, "policy": policy.__dict__}


def max_state_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    if left.keys() != right.keys():
        raise ValueError("State dictionaries have different keys")
    return max(float((left[key] - right[key]).abs().max()) for key in left)


def run_checks() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = {
        "registered_utc": datetime.now(timezone.utc).isoformat(), "synthetic_only": True,
        "synthetic_rows": 256, "batch_size": 32, "device": "cpu", "n_ens": 1,
        "maximum_toy_epochs": 20,
        "arms": {"reference": {"horizon": 12, "run_until": 12, "snapshot": 4},
                 "prefix_stop": {"horizon": 12, "run_until": 4, "snapshot": 4},
                 "compressed": {"horizon": 4, "run_until": 4, "snapshot": 4}},
        "acceptance": "Exact CPU equality of prefix checkpoint, probabilities and schedule trace; compressed horizon differs",
        "no_scientific_quality_claim": True,
    }
    (OUT / "registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    reference = execute_toy(EpochPolicy(12, 12, 4), use_library_stop=False)
    preserved = execute_toy(EpochPolicy(12, 4, 4), use_library_stop=True)
    compressed = execute_toy(EpochPolicy(4, 4, 4), use_library_stop=False)
    prefix_diff = max_state_difference(reference["state"], preserved["final_state"])
    compressed_diff = max_state_difference(reference["state"], compressed["final_state"])
    probability_diff = float((reference["probabilities"] - preserved["probabilities"]).abs().max())
    exact_schedule_equal = reference["batch_schedule"][:32] == preserved["batch_schedule"]
    compressed_schedule_equal = reference["batch_schedule"][:32] == compressed["batch_schedule"]
    if prefix_diff != 0.0 or probability_diff != 0.0 or not exact_schedule_equal:
        raise AssertionError("Preserved-horizon stop failed exact CPU prefix equality")
    if compressed_diff <= 1e-6 or compressed_schedule_equal:
        raise AssertionError("Compressed horizon did not differ as expected")
    if [reference["global_steps"], preserved["global_steps"], compressed["global_steps"]] != [96, 32, 32]:
        raise AssertionError("Unexpected optimizer update count")
    source_paths = [Path(__file__), ROOT / "scripts/test_research_epoch_control_v1.py",
                    ROOT / ".venv/Lib/site-packages/pytabkit/models/training/lightning_callbacks.py",
                    ROOT / ".venv/Lib/site-packages/pytabkit/models/training/nn_creator.py"]
    output = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "device": "cpu", "real_data_rows": 0,
        "reference_completed_epochs": reference["completed_epochs"],
        "preserved_completed_epochs": preserved["completed_epochs"],
        "compressed_completed_epochs": compressed["completed_epochs"],
        "preserved_state_max_abs_difference": prefix_diff,
        "preserved_probability_max_abs_difference": probability_diff,
        "compressed_state_max_abs_difference": compressed_diff,
        "compressed_probability_max_abs_difference": float((reference["probabilities"] - compressed["probabilities"]).abs().max()),
        "preserved_schedule_trace_exact": exact_schedule_equal,
        "compressed_schedule_trace_exact": compressed_schedule_equal,
        "batch07_clock_comparison": schedule_clock_comparison(377802, 419780, 256, 500, 4),
        "traces": {"reference_first4": reference["batch_schedule"][:32],
                   "preserved": preserved["batch_schedule"], "compressed": compressed["batch_schedule"]},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "limitations": [
            "Toy CPU mechanism test does not establish data-quality or competition-score gain",
            "Retaining an epoch-based horizon preserves schedule fraction per epoch, not update count when training rows change",
            "Installed StopAtEpochsCallback saves/restores model parameters and buffers, not complete optimizer/RNG/sampler state",
            "No exact training resumption or current-model continued fit was attempted",
            "A completed four-epoch flat_anneal schedule ends near zero LR; continuation requires a separately declared LR intervention",
        ],
    }
    (OUT / "verification.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    result = run_checks()
    print(json.dumps({key: value for key, value in result.items() if key not in {"traces", "source_sha256"}}, indent=2))
