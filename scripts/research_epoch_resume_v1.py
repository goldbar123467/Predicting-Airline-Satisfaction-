"""Bounded CPU-only, epoch-boundary resume test for the installed RealMLP.

Tests ordinary Lightning checkpoints and an explicitly augmented research
checkpoint. Uses generated data only. No production adapter/library is modified.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pytorch_lightning as pl
import torch
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.alg_interfaces.base import InterfaceResources, SplitIdxs
from pytabkit.models.training.lightning_modules import TabNNModule

from research_epoch_control_v1 import max_state_difference, probabilities, toy_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_pass_v1/epochs/resume_prototype"


def tensor_state(module: TabNNModule) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.model.state_dict().items()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResearchResumeModule(TabNNModule):
    """Research subclass saves states absent from ordinary installed checkpoints."""

    def __init__(self, **config: Any):
        super().__init__(**config)
        self.include_research_resume_state = False
        self.pending_resume_state: dict[str, Any] | None = None
        self.resume_applied = False

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if not self.include_research_resume_state:
            return
        wrapper = self.optimizers(use_pl_optimizer=False)
        checkpoint["research_resume_state_v1"] = {
            "inner_optimizer": copy.deepcopy(wrapper.opt.state_dict()),
            "progress": copy.deepcopy(vars(self.progress)),
            "torch_cpu_rng": torch.get_rng_state().clone(),
            "numpy_rng": copy.deepcopy(np.random.get_state()),
            "python_rng": random.getstate(),
            "static_preprocessing": copy.deepcopy(self.creator.static_model.state_dict()),
            "loader_contract": {"boundary": "completed_epoch", "rows": 256,
                                "batch_size": 32, "drop_last": True,
                                "updates_per_epoch": len(self.train_dl)},
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.pending_resume_state = checkpoint.get("research_resume_state_v1")
        if self.pending_resume_state is not None:
            # Installed wrapper load_state_dict calls an incompatible __setstate__.
            # Skip its empty state in this in-memory checkpoint dictionary only;
            # on_train_start explicitly restores the actual underlying Adam.
            checkpoint["optimizer_states"] = []

    def on_train_start(self) -> None:
        if self.pending_resume_state is None:
            return
        saved = self.pending_resume_state
        if saved["loader_contract"] != {"boundary": "completed_epoch", "rows": 256,
                                        "batch_size": 32, "drop_last": True,
                                        "updates_per_epoch": len(self.train_dl)}:
            raise ValueError("Checkpoint loader contract mismatch")
        self.optimizers(use_pl_optimizer=False).opt.load_state_dict(saved["inner_optimizer"])
        for name, value in saved["progress"].items():
            setattr(self.progress, name, value)
        self.creator.static_model.load_state_dict(saved["static_preprocessing"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        # At an epoch boundary this iterable has no pending batch/permutation.
        # It has a known length, so Lightning does not prefetch before this hook.
        torch.set_rng_state(saved["torch_cpu_rng"])
        self.hp_manager.needs_update = True
        self.pending_resume_state = None
        self.resume_applied = True


class ResumeTrace(pl.Callback):
    def __init__(self, save_paths: tuple[Path, Path] | None = None):
        self.save_paths = save_paths
        self.next_update_state: dict[str, torch.Tensor] | None = None
        self.first_resumed_batch_sha256: str | None = None
        self.next_update_schedule: dict[str, Any] | None = None
        self.completed_epochs = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if trainer.global_step == 32:
            digest = hashlib.sha256()
            for key in sorted(batch):
                digest.update(key.encode())
                digest.update(batch[key].detach().cpu().numpy().tobytes())
            self.first_resumed_batch_sha256 = digest.hexdigest()
            hypers = pl_module.hp_manager.get_hyper_sched_values()
            self.next_update_schedule = {
                "global_step_before": int(trainer.global_step),
                "epoch_float": float(pl_module.progress.epoch_float),
                "fraction": float(pl_module.progress.get_fit_progress()),
                "lr_schedule": float(hypers["lr"][""]),
                "wd_schedule": float(hypers["wd"][""]),
                "dropout_schedule": float(hypers["p_drop"][""]),
            }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step == 33:
            self.next_update_state = tensor_state(pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self.completed_epochs = int(trainer.current_epoch + 1)
        if self.save_paths is not None and self.completed_epochs == 4:
            standard_path, augmented_path = self.save_paths
            pl_module.include_research_resume_state = False
            trainer.save_checkpoint(standard_path, weights_only=False)
            pl_module.include_research_resume_state = True
            trainer.save_checkpoint(augmented_path, weights_only=False)
            pl_module.include_research_resume_state = False


def train_case(checkpoint: Path | None = None, save_paths: tuple[Path, Path] | None = None) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(23197)
    np.random.seed(23197)
    random.seed(23197)
    dataset = toy_dataset()
    config = RealMLP_TD_Classifier(
        device="cpu", n_threads=1, verbosity=0, n_epochs=12, n_ens=1,
        hidden_sizes=[16, 8], batch_size=32, predict_batch_size=256,
        lr=0.053, wd=0.015, lr_sched="flat_anneal", wd_sched="cos_log_15",
        p_drop=0.05, p_drop_sched="invsqrtp1e-3", sq_mom=0.988,
        act="silu", first_layer_lr_factor=0.25, add_front_scale=False,
        plr_lr_factor=0.1151, ls_eps=0.0, n_cv=1, n_refit=0,
        use_early_stopping=False, use_best_mean_epoch_for_cv=True,
        val_metric_name="1-auc_ovr", ens_av_before_softmax=False,
    ).get_config()
    module = ResearchResumeModule(**config)
    split = SplitIdxs(torch.arange(256).reshape(1, -1), None, None,
                      split_seed=23197, sub_split_seeds=[23197], split_id=0)
    module.compile_model(dataset, [split], InterfaceResources(n_threads=1, gpu_devices=[]))
    if any(parameter.device.type != "cpu" for parameter in module.model.parameters()):
        raise AssertionError("CPU-only experiment")
    trace = ResumeTrace(save_paths)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=12,
                         callbacks=[*module.create_callbacks(), trace], logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0,
                         deterministic=True, log_every_n_steps=1)
    # These are self-generated, hash-recorded local checkpoints, not untrusted files.
    trainer.fit(module, train_dataloaders=module.train_dl, val_dataloaders=module.val_dl,
                ckpt_path=checkpoint, weights_only=False)
    if trace.next_update_state is None:
        raise AssertionError("Did not capture update 33")
    return {"state": tensor_state(module), "probabilities": probabilities(module, dataset),
            "next_update_state": trace.next_update_state,
            "first_resumed_batch_sha256": trace.first_resumed_batch_sha256,
            "next_update_schedule": trace.next_update_schedule,
            "global_steps": int(trainer.global_step), "completed_epochs": trace.completed_epochs,
            "resume_applied": module.resume_applied,
            "inner_optimizer_state": copy.deepcopy(trainer.optimizers[0].opt.state_dict())}


def optimizer_state_max_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left["state"].keys() != right["state"].keys():
        raise AssertionError("Optimizer parameter keys differ")
    difference = 0.0
    for parameter, state in left["state"].items():
        if state.keys() != right["state"][parameter].keys():
            raise AssertionError("Optimizer state keys differ")
        for name, value in state.items():
            other = right["state"][parameter][name]
            if torch.is_tensor(value):
                difference = max(difference, float((value - other).abs().max()))
            elif value != other:
                raise AssertionError("Non-tensor optimizer state differs")
    if left["param_groups"] != right["param_groups"]:
        raise AssertionError("Final optimizer groups differ")
    return difference


def run_resume_checks() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    standard_path, augmented_path = OUT / "standard_epoch4.ckpt", OUT / "augmented_epoch4.ckpt"
    registry = {
        "registered_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu", "real_data_rows": 0,
        "synthetic_rows": 256, "interruption_boundary": "after epoch4, before epoch5 iterator",
        "horizon": 12, "reference_steps": 96, "checkpoint_steps": 32,
        "arms": ["uninterrupted", "ordinary Lightning resume", "explicitly augmented resume"],
        "acceptance": "Augmented resume matches next update, minibatch, schedule, final parameters and optimizer tensors exactly",
        "repair_attempt_limit": 3,
    }
    (OUT / "registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    reference = train_case(save_paths=(standard_path, augmented_path))
    checkpoint_hashes = {path.name: sha256(path) for path in (standard_path, augmented_path)}
    standard_checkpoint = torch.load(standard_path, map_location="cpu", weights_only=False)
    augmented_checkpoint = torch.load(augmented_path, map_location="cpu", weights_only=False)
    if any(sha256(path) != checkpoint_hashes[path.name] for path in (standard_path, augmented_path)):
        raise AssertionError("Research checkpoint changed before loading")
    ordinary_error = None
    try:
        ordinary = train_case(checkpoint=standard_path)
    except KeyError as error:
        if error.args != ("__dict__",):
            raise
        ordinary = None
        ordinary_error = {"type": "KeyError", "message": "__dict__",
                          "phase": "Lightning restore optimizer state",
                          "source": "pytabkit OptimizerBase.__setstate__ expects a pickle envelope, not torch state_dict"}
    augmented = train_case(checkpoint=augmented_path)
    results = {}
    for name, case in (("ordinary", ordinary), ("augmented", augmented)):
        if case is None:
            results[name] = {"load_failed": True, "error": ordinary_error}
            continue
        results[name] = {
            "load_failed": False,
            "next_update_state_max_abs_difference": max_state_difference(reference["next_update_state"], case["next_update_state"]),
            "final_state_max_abs_difference": max_state_difference(reference["state"], case["state"]),
            "final_probability_max_abs_difference": float((reference["probabilities"] - case["probabilities"]).abs().max()),
            "next_batch_equal": reference["first_resumed_batch_sha256"] == case["first_resumed_batch_sha256"],
            "schedule_equal": reference["next_update_schedule"] == case["next_update_schedule"],
            "next_update_schedule": case["next_update_schedule"],
            "global_steps": case["global_steps"], "completed_epochs": case["completed_epochs"],
            "explicit_extra_state_applied": case["resume_applied"],
        }
    corrected = results["augmented"]
    optimizer_difference = optimizer_state_max_difference(reference["inner_optimizer_state"], augmented["inner_optimizer_state"])
    passed = (corrected["next_update_state_max_abs_difference"] == 0.0
              and corrected["final_state_max_abs_difference"] == 0.0
              and corrected["final_probability_max_abs_difference"] == 0.0
              and corrected["next_batch_equal"] and corrected["schedule_equal"]
              and corrected["global_steps"] == 96 and optimizer_difference == 0.0)
    source_paths = [Path(__file__), ROOT / "scripts/test_research_epoch_resume_v1.py",
                    ROOT / "scripts/research_epoch_control_v1.py",
                    ROOT / ".venv/Lib/site-packages/pytabkit/models/optim/optimizers.py",
                    ROOT / ".venv/Lib/site-packages/pytabkit/models/training/lightning_modules.py",
                    ROOT / ".venv/Lib/site-packages/pytabkit/models/data/data.py"]
    output = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if passed else "not_exact",
        "device": "cpu", "real_data_rows": 0, "results": results,
        "reference_next_update_schedule": reference["next_update_schedule"],
        "ordinary_saved_optimizer_state_count": len(standard_checkpoint["optimizer_states"][0]["state"]),
        "augmented_saved_inner_optimizer_state_count": len(augmented_checkpoint["research_resume_state_v1"]["inner_optimizer"]["state"]),
        "ordinary_has_rng_state": any("rng" in key.lower() for key in standard_checkpoint),
        "ordinary_has_pytabkit_progress": "research_resume_state_v1" in standard_checkpoint,
        "augmented_final_optimizer_max_abs_difference": optimizer_difference,
        "augmented_load_policy": "Skip incompatible wrapper optimizer state in memory; explicitly load actual inner Adam state in on_train_start",
        "development_attempt_note": "First execution exposed native KeyError before diagnostic handling; one isolated repair adds expected-error recording and augmented wrapper-state bypass",
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "limits": [
            "Synthetic deterministic CPU only; does not establish GPU or cross-platform determinism",
            "Epoch-boundary resume only; no in-epoch sampler permutation/position recovery is implemented",
            "Preprocessing and generated dataset are reconstructed under exact fixed contracts before state restore",
            "No validation early-stopping/checkpoint-selection callback state is present in this fixed-epoch toy",
            "No production model/config/library source is edited and no competition predictions are created",
        ],
    }
    (OUT / "verification.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not passed:
        raise AssertionError(f"Augmented epoch-boundary resume not exact; inspect {OUT / 'verification.json'}")
    return output


if __name__ == "__main__":
    print(json.dumps(run_resume_checks(), indent=2))
