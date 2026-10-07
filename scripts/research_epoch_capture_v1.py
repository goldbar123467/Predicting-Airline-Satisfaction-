"""Synthetic CPU proof that cloned mid-training native export is passive.

Uses the actual production schema, inference graph and portability functions on
separately owned copies. This is not an integration into the production fitter.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
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

from realmlp_categorical import _fit_schema, _InputSplit, _MixedGraph, _prepare_portable_inference
from research_epoch_control_v1 import max_state_difference, probabilities, toy_dataset
from research_epoch_resume_v1 import optimizer_state_max_difference, sha256, tensor_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_pass_v1/epochs/capture_prototype"


def fingerprint(value: Any) -> str:
    """Hash tensor/scalar containers without executing pickle hooks."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, bool, int, float)):
            digest.update(repr(item).encode())
        else:
            raise TypeError(f"Unsupported fingerprint type: {type(item)}")

    visit(value)
    return digest.hexdigest()


def layer_identity(module: torch.nn.Module) -> list[dict[str, Any]]:
    methods = ("forward", "forward_tensors", "_multiple", "_binary")
    return [{"name": name, "mode": layer.training,
             "methods": {method: id(getattr(getattr(layer, method), "__func__", getattr(layer, method)))
                         for method in methods if hasattr(layer, method)}}
            for name, layer in module.named_modules()]


def live_signature(module: TabNNModule) -> dict[str, str]:
    return {
        "parameters_and_buffers": fingerprint(module.model.state_dict()),
        "static_preprocessing": fingerprint(module.creator.static_model.state_dict()),
        "inner_optimizer": fingerprint(module.optimizers(use_pl_optimizer=False).opt.state_dict()),
        "torch_cpu_rng": fingerprint(torch.get_rng_state()),
        "numpy_rng": fingerprint(np.random.get_state()),
        "python_rng": fingerprint(random.getstate()),
        "progress": fingerprint(vars(module.progress)),
        "schedule": fingerprint(module.hp_manager.get_hyper_sched_values()),
        "network_modes_and_forward_methods": fingerprint(layer_identity(module.model)),
        "static_modes_and_forward_methods": fingerprint(layer_identity(module.creator.static_model)),
    }


def owned_storage(module: torch.nn.Module) -> set[int]:
    return {value.untyped_storage().data_ptr()
            for value in list(module.parameters()) + list(module.buffers()) if value.numel()}


class CaptureExport(pl.Callback):
    def __init__(self, dataset, export_path: Path | None, deadline: float):
        self.dataset = dataset
        self.export_path = export_path
        self.deadline = deadline
        # Native code zero is represented by the adapter's reserved input -1.
        # Remaining raw categories map back to exactly their original native codes.
        self.raw = torch.cat((dataset.tensors["x_cont"], dataset.tensors["x_cat"].float() - 1), dim=1)
        self.schema = _fit_schema(self.raw.numpy(), [6, 7])
        converted = _InputSplit(self.schema)(self.raw)
        if not all(torch.equal(converted[key], dataset.tensors[key]) for key in converted):
            raise AssertionError("Synthetic production-schema round trip is not exact")
        self.batch_trace: list[dict[str, str | int]] = []
        self.capture: dict[str, Any] | None = None
        self.prefix_state: dict[str, torch.Tensor] | None = None
        self.prefix_probability: torch.Tensor | None = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if time.monotonic() > self.deadline:
            raise TimeoutError("Capture/export check exceeded 120 seconds")
        self.batch_trace.append({"step": int(trainer.global_step), "batch": fingerprint(batch),
                                 "rng": fingerprint(torch.get_rng_state()),
                                 "schedule": fingerprint(pl_module.hp_manager.get_hyper_sched_values())})

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch + 1 != 4:
            return
        self.prefix_state = tensor_state(pl_module)
        if self.export_path is None:
            return
        before = live_signature(pl_module)
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                static = copy.deepcopy(pl_module.creator.static_model)
                network = copy.deepcopy(pl_module.model)
                graph = _MixedGraph(_InputSplit(copy.deepcopy(self.schema)), static, network, n_ens=1).eval()
                live_storage = owned_storage(pl_module.model) | owned_storage(pl_module.creator.static_model)
                disjoint = live_storage.isdisjoint(owned_storage(graph))
                if not disjoint:
                    raise AssertionError("Export graph aliases live tensor storage")
                with torch.no_grad():
                    native = graph(self.raw).detach().clone()
                    self.prefix_probability = native
                    counts = _prepare_portable_inference(graph)
                    portable = graph(self.raw).detach().clone()
                    traced = torch.jit.trace(graph, self.raw[:17], check_inputs=[(self.raw[:1],), (self.raw,)],
                                             check_tolerance=1e-5)
                    torch.jit.save(traced, str(self.export_path))
                    loaded = torch.jit.load(str(self.export_path), map_location="cpu").eval()
                    export_difference = float((loaded(self.raw) - native).abs().max())
                    reverse_difference = float((loaded(self.raw.flip(0)).flip(0) - native).abs().max())
                    one_row_difference = float((loaded(self.raw[:1]) - native[:1]).abs().max())
                    if not torch.allclose(loaded(self.raw), native, atol=2e-6, rtol=1e-5):
                        raise AssertionError("Reloaded export differs from native pre-rewrite graph")
                    if reverse_difference > 2e-6 or one_row_difference > 2e-6:
                        raise AssertionError("Reloaded export is not stable across tested batch layouts")
                self.capture = {"storage_disjoint": disjoint, "portable_rewrite_counts": counts,
                                "native_portable_max_abs_difference": float((portable - native).abs().max()),
                                "native_reload_max_abs_difference": export_difference,
                                "reversed_batch_max_abs_difference": reverse_difference,
                                "one_row_max_abs_difference": one_row_difference,
                                "static_state_dict_entries": len(static.state_dict()),
                                "artifact_sha256_at_capture": sha256(self.export_path)}
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        after = live_signature(pl_module)
        if self.capture is None:
            raise AssertionError("No export capture")
        self.capture["live_state_unchanged"] = {key: before[key] == after[key] for key in before}
        self.capture["live_signatures_before"] = before
        self.capture["live_signatures_after"] = after
        if not all(self.capture["live_state_unchanged"].values()):
            raise AssertionError("Capture/export changed live training state")


def train_case(export_path: Path | None, deadline: float) -> dict[str, Any]:
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
    config["use_best_epoch"] = False
    module = TabNNModule(**config)
    split = SplitIdxs(torch.arange(256).reshape(1, -1), None, None,
                      split_seed=23197, sub_split_seeds=[23197], split_id=0)
    module.compile_model(dataset, [split], InterfaceResources(n_threads=1, gpu_devices=[]))
    if any(parameter.device.type != "cpu" for parameter in module.model.parameters()):
        raise AssertionError("CPU-only experiment")
    capture = CaptureExport(dataset, export_path, deadline)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=12,
                         callbacks=[*module.create_callbacks(), capture], logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0,
                         deterministic=True, log_every_n_steps=1)
    trainer.fit(module, train_dataloaders=module.train_dl, val_dataloaders=module.val_dl)
    return {"state": tensor_state(module), "probabilities": probabilities(module, dataset),
            "optimizer": copy.deepcopy(trainer.optimizers[0].opt.state_dict()),
            "capture": capture, "global_steps": int(trainer.global_step)}


def run_capture_checks() -> dict[str, Any]:
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    registry = {"registered_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu",
                "real_data_rows": 0, "synthetic_rows": 256, "n_ens": 1,
                "horizon": 12, "snapshot_epoch": 4, "runs": 2, "maximum_seconds": 120,
                "arms": ["uninterrupted without export", "cloned native export at epoch 4, continue to 12"],
                "acceptance": "Live state and full continuation equal; reloaded native parity and artifact hash stable"}
    (OUT / "registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    path = OUT / "epoch4_graph.pt"
    reference, exported = train_case(None, start + 120), train_case(path, start + 120)
    trace = exported["capture"]
    if trace.capture is None or trace.prefix_probability is None:
        raise AssertionError("Export capture missing")
    loaded = torch.jit.load(str(path), map_location="cpu").eval()
    with torch.no_grad():
        final_reload_difference = float((loaded(trace.raw) - trace.prefix_probability).abs().max())
    deltas = {"prefix_state": max_state_difference(reference["capture"].prefix_state, trace.prefix_state),
              "final_state": max_state_difference(reference["state"], exported["state"]),
              "final_probabilities": float((reference["probabilities"] - exported["probabilities"]).abs().max()),
              "final_optimizer": optimizer_state_max_difference(reference["optimizer"], exported["optimizer"]),
              "final_reload_vs_prefix_native": final_reload_difference,
              "continued_model_vs_prefix": max_state_difference(exported["state"], trace.prefix_state)}
    hash_stable = sha256(path) == trace.capture["artifact_sha256_at_capture"]
    batch_equal = reference["capture"].batch_trace == trace.batch_trace
    passed = (all(deltas[key] == 0.0 for key in ("prefix_state", "final_state", "final_probabilities", "final_optimizer"))
              and deltas["continued_model_vs_prefix"] > 0 and hash_stable and batch_equal
              and final_reload_difference <= 2e-6 and exported["global_steps"] == 96
              and all(trace.capture["live_state_unchanged"].values()))
    sources = [Path(__file__), ROOT / "scripts/test_research_epoch_capture_v1.py",
               ROOT / "scripts/research_epoch_control_v1.py", ROOT / "scripts/research_epoch_resume_v1.py",
               ROOT / "scripts/realmlp_categorical.py"]
    output = {**registry, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "status": "passed" if passed else "failed", "seconds": time.monotonic() - start,
              "capture": trace.capture, "maximum_absolute_differences": deltas,
              "batch_rng_and_schedule_paths_equal": batch_equal, "artifact_hash_stable_after_continuation": hash_stable,
              "artifact_sha256_after_continuation": sha256(path),
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
              "limitations": ["Synthetic CPU one-member tiny architecture only; no CUDA or cross-platform proof",
                              "Actual production schema/graph/portability helpers used, but no integration into its terminal fitter",
                              "No validation/checkpoint-selection path; the separate passive-monitor experiment covers that",
                              "Toy static state_dict is empty; learned production preprocessing ownership needs its own full-path test",
                              "Baseline includes read-only batch/state audit; comparison isolates clone, rewrite, trace, save and reload",
                              "Exported TorchScript is an inference artifact, not a resumable training checkpoint",
                              "No production code, campaign model/config, real data, audit score or release changed"]}
    (OUT / "verification.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not passed:
        raise AssertionError(f"Capture/export verification failed: {deltas}")
    return output


if __name__ == "__main__":
    result = run_capture_checks()
    print(json.dumps({key: result[key] for key in ("status", "seconds", "maximum_absolute_differences")}, indent=2))
