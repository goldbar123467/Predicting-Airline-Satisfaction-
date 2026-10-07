"""CPU-only synthetic checks for literal endpoints and passive monitoring.

No competition files are read. The deliberately injected validation error is a
test stimulus, never a reported model-quality metric. Installed library files
and the existing epoch-control prototype remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import pytorch_lightning as pl
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.alg_interfaces.base import InterfaceResources, SplitIdxs
from pytabkit.models.training.lightning_callbacks import ModelCheckpointCallback
from pytabkit.models.training.lightning_modules import TabNNModule
from pytabkit.models.training.metrics import Metrics

from research_epoch_control_v1 import max_state_difference, toy_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/research_pass_v1/epochs/passive_monitor"
FORCED_ERRORS = (0.2, 0.1, 0.3, 0.4)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def clone_state(module: TabNNModule) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone()
            for key, value in module.model.state_dict().items()}


def ensemble_probabilities(module: TabNNModule, dataset) -> torch.Tensor:
    """Eight members share probe rows; average probabilities, not logits."""
    modes = [(layer, layer.training) for layer in module.model.modules()]
    module.model.eval()
    try:
        with torch.no_grad():
            prepared = module.creator.static_model(dataset)
            batch = {key: value.unsqueeze(0).expand(8, *value.shape)
                     for key, value in prepared.tensors.items()}
            logits = module.model(batch)["x_cont"]
            if logits.shape != (8, len(dataset.tensors["y"]), 2):
                raise AssertionError("Expected [member, row, class] probe logits")
            result = logits.softmax(-1).mean(0).clone()
            if not torch.isfinite(result).all() or not torch.allclose(
                    result.sum(-1), torch.ones(result.shape[0]), atol=1e-6, rtol=0):
                raise AssertionError("Invalid ensemble probabilities")
            return result
    finally:
        for layer, old_mode in modes:
            layer.training = old_mode


class ForcedMetricModule(TabNNModule):
    """Keep real validation forwards, inject only the final scalar scorer."""

    def on_validation_epoch_end(self) -> None:
        actual_apply = Metrics.apply
        error = FORCED_ERRORS[int(self.trainer.current_epoch)]

        def score(prediction, target, metric_name, *args, **kwargs):
            if metric_name == "1-auc_ovr":
                # Shape assertions establish that actual held-out forwards ran.
                if prediction.shape != (64, 2) or target.shape[0] != 64:
                    raise AssertionError("Expected real synthetic validation predictions")
                return error
            return actual_apply(prediction, target, metric_name, *args, **kwargs)

        with patch.object(Metrics, "apply", side_effect=score):
            super().on_validation_epoch_end()


class PassiveProbe(pl.Callback):
    """Eval-mode synthetic probe with explicit mode/RNG preservation."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.snapshot: dict[str, torch.Tensor] | None = None
        self.snapshot_initial_hashes: dict[str, str] | None = None
        self.probe_probability_hashes: list[str] = []
        self.rng_preserved: list[bool] = []

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        torch_before = torch.get_rng_state().clone()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        modes = [(layer, layer.training) for layer in pl_module.model.modules()]
        try:
            # No GPU exists in this check. fork_rng preserves the CPU generator.
            with torch.random.fork_rng(devices=[]):
                result = ensemble_probabilities(pl_module, self.dataset)
                self.probe_probability_hashes.append(tensor_hash(result))
                if trainer.current_epoch + 1 == 2:
                    self.snapshot = clone_state(pl_module)
                    self.snapshot_initial_hashes = {
                        key: tensor_hash(value) for key, value in self.snapshot.items()}
        finally:
            random.setstate(python_before)
            np.random.set_state(numpy_before)
            for layer, old_mode in modes:
                layer.training = old_mode
        self.rng_preserved.append(torch.equal(torch_before, torch.get_rng_state()))
        if not self.rng_preserved[-1]:
            raise AssertionError("Passive probe altered CPU RNG state")


class AuditTrace(pl.Callback):
    def __init__(self, deadline: float):
        self.deadline = deadline
        self.epoch_states: list[dict[str, torch.Tensor]] = []
        self.batch_trace: list[dict[str, object]] = []
        self.validation_batches = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if time.monotonic() > self.deadline:
            raise TimeoutError("Synthetic monitor check exceeded 120 seconds")
        values = pl_module.hp_manager.get_hyper_sched_values()
        self.batch_trace.append({
            "global_step": int(trainer.global_step),
            "epoch_float": float(pl_module.progress.epoch_float),
            "lr": float(values["lr"][""]),
            "wd": float(values["wd"][""]),
            "dropout": float(values["p_drop"][""]),
            "rng": tensor_hash(torch.get_rng_state()),
            "batch_x_cont": tensor_hash(batch["x_cont"]),
            "batch_x_cat": tensor_hash(batch["x_cat"]),
            "batch_y": tensor_hash(batch["y"]),
        })

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx=0) -> None:
        self.validation_batches += 1

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self.epoch_states.append(clone_state(pl_module))


@dataclass
class SyntheticRun:
    state: dict[str, torch.Tensor]
    prediction: torch.Tensor
    trace: AuditTrace
    probe: PassiveProbe | None
    checkpoint_callbacks: int
    final_rng: str
    best_epoch: int
    updates: int


def execute_run(use_best_epoch: bool, telemetry: bool, deadline: float) -> SyntheticRun:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(23197)
    np.random.seed(23197)
    torch.manual_seed(23197)
    dataset = toy_dataset()
    config = RealMLP_TD_Classifier(
        device="cpu", n_threads=1, verbosity=0, n_epochs=4,
        n_ens=8, hidden_sizes=[16, 8], batch_size=32, predict_batch_size=256,
        lr=0.053, wd=0.015, lr_sched="flat_anneal", wd_sched="cos_log_15",
        p_drop=0.05, p_drop_sched="invsqrtp1e-3", sq_mom=0.988,
        act="silu", first_layer_lr_factor=0.25, add_front_scale=False,
        plr_lr_factor=0.1151, ls_eps=0.0, n_cv=1, n_refit=0,
        use_early_stopping=False,
        use_best_mean_epoch_for_cv=True, val_metric_name="1-auc_ovr",
        ens_av_before_softmax=False,
    ).get_config()
    # The low-level creator supports this flag, but the public estimator's
    # constructor does not expose it in installed 1.7.3.
    config["use_best_epoch"] = use_best_epoch
    module = ForcedMetricModule(**config)
    split = SplitIdxs(torch.arange(192).reshape(1, -1),
                      torch.arange(192, 256).reshape(1, -1), None,
                      split_seed=23197, sub_split_seeds=[23197], split_id=0)
    module.compile_model(dataset, [split], InterfaceResources(n_threads=1, gpu_devices=[]))
    if module.val_dl is None or any(p.device.type != "cpu" for p in module.model.parameters()):
        raise AssertionError("CPU training and a real validation loader are required")
    callbacks = module.create_callbacks()
    count = sum(isinstance(callback, ModelCheckpointCallback) for callback in callbacks)
    if count != int(use_best_epoch):
        raise AssertionError("Unexpected best-checkpoint callback configuration")
    trace = AuditTrace(deadline)
    # Fixed fit-row probe. It is neither a monitor row nor an assessment set.
    probe = PassiveProbe(dataset.get_sub_dataset(torch.arange(32))) if telemetry else None
    if probe is not None:
        callbacks.append(probe)
    callbacks.append(trace)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=4,
                         callbacks=callbacks, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0, deterministic=True, log_every_n_steps=1)
    trainer.fit(module, train_dataloaders=module.train_dl, val_dataloaders=module.val_dl)
    if len(trace.epoch_states) != 4 or trainer.global_step != 24:
        raise AssertionError("Expected four complete six-update epochs")
    return SyntheticRun(clone_state(module), ensemble_probabilities(module, dataset), trace, probe,
                        count, tensor_hash(torch.get_rng_state()),
                        int(module.best_mean_val_epochs["1-auc_ovr"][0]),
                        int(trainer.global_step))


def run_checks() -> dict[str, object]:
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_only": True, "real_data_rows": 0, "device": "cpu",
        "total_rows": 256, "fit_rows": 192, "validation_rows": 64,
        "n_ens": 8, "epochs_per_run": 4, "runs": 3,
        "forced_validation_errors": FORCED_ERRORS,
        "stimulus_status": "Injected scalar errors test checkpoint behavior; not model-quality scores",
        "maximum_seconds": 120,
    }
    (OUTPUT / "registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    legacy = execute_run(True, False, started + 120)
    fixed = execute_run(False, False, started + 120)
    monitored = execute_run(False, True, started + 120)
    if monitored.probe is None or monitored.probe.snapshot is None:
        raise AssertionError("Missing passive probe snapshot")
    deltas = {
        "legacy_final_vs_epoch2": max_state_difference(legacy.state, legacy.trace.epoch_states[1]),
        "legacy_final_vs_epoch4": max_state_difference(legacy.state, legacy.trace.epoch_states[3]),
        "fixed_final_vs_epoch4": max_state_difference(fixed.state, fixed.trace.epoch_states[3]),
        "fixed_epoch2_vs_epoch4": max_state_difference(fixed.trace.epoch_states[1], fixed.trace.epoch_states[3]),
        "telemetry_final_state": max_state_difference(fixed.state, monitored.state),
        "telemetry_probabilities": float((fixed.prediction - monitored.prediction).abs().max()),
    }
    before_fit_end_equal = all(max_state_difference(a, b) == 0.0
                              for a, b in zip(legacy.trace.epoch_states, fixed.trace.epoch_states))
    snapshot_immutable = monitored.probe.snapshot_initial_hashes == {
        key: tensor_hash(value) for key, value in monitored.probe.snapshot.items()}
    passed = (deltas["legacy_final_vs_epoch2"] == 0.0
              and deltas["legacy_final_vs_epoch4"] > 1e-6
              and deltas["fixed_final_vs_epoch4"] == 0.0
              and deltas["fixed_epoch2_vs_epoch4"] > 1e-6
              and deltas["telemetry_final_state"] == 0.0
              and deltas["telemetry_probabilities"] == 0.0
              and before_fit_end_equal and snapshot_immutable
              and fixed.trace.batch_trace == monitored.trace.batch_trace
              and fixed.final_rng == monitored.final_rng
              and all(monitored.probe.rng_preserved)
              and all(run.best_epoch == 2 for run in (legacy, fixed, monitored)))
    source_paths = [Path(__file__), ROOT / "scripts/test_research_passive_monitor_v1.py",
                    ROOT / "scripts/research_epoch_control_v1.py",
                    ROOT / ".venv/Lib/site-packages/pytabkit/models/training/nn_creator.py",
                    ROOT / ".venv/Lib/site-packages/pytabkit/models/training/lightning_callbacks.py"]
    result = {
        **registry, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed", "seconds": time.monotonic() - started,
        "versions": {"torch": torch.__version__, "lightning": pl.__version__},
        "maximum_absolute_differences": deltas,
        "legacy_and_fixed_trajectory_before_fit_end_equal": before_fit_end_equal,
        "telemetry_batch_schedule_rng_path_equal": fixed.trace.batch_trace == monitored.trace.batch_trace,
        "telemetry_final_rng_equal": fixed.final_rng == monitored.final_rng,
        "snapshot_immutable_after_later_updates": snapshot_immutable,
        "validation_batches": [run.trace.validation_batches for run in (legacy, fixed, monitored)],
        "best_epochs": [run.best_epoch for run in (legacy, fixed, monitored)],
        "checkpoint_callback_counts": [run.checkpoint_callbacks for run in (legacy, fixed, monitored)],
        "trace_without_extra_telemetry": fixed.trace.batch_trace,
        "trace_with_extra_telemetry": monitored.trace.batch_trace,
        "probe_probability_hashes": monitored.probe.probe_probability_hashes,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
        "limitations": [
            "Injected validation errors establish restoration mechanics, not generalization",
            "Generated 256-row CPU data and small hidden layers; no CUDA or production adapter proof",
            "Baseline retains audit capture; comparison isolates added eval-mode probes and epoch snapshot",
            "Exact equality is verified on this installed environment only",
            "No training resumption, real-data fit, audit score or release change",
        ],
    }
    (OUTPUT / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    if not passed:
        raise AssertionError(f"Passive monitor mechanism check failed: {deltas}")
    return result


if __name__ == "__main__":
    result = run_checks()
    print(json.dumps({key: result[key] for key in ("status", "seconds", "maximum_absolute_differences")},
                     indent=2))
