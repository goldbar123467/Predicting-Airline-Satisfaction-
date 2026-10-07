"""Fixed RealMLP schedule/endpoint trajectories with passive native snapshots.

The caller owns row/split provenance and external feature preprocessing. This
adapter fits native category vocabularies and model preprocessing on fit rows,
monitors only supplied inner rows, and never selects an epoch or blend weight.
Outputs are inference-only and load through load_realmlp_categorical.
"""
from __future__ import annotations

import copy
import gc
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import psutil
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.alg_interfaces.base import InterfaceResources, SplitIdxs
from pytabkit.models.data.data import DictDataset, TensorInfo
from pytabkit.models.data.splits import RandomSplitter
from pytabkit.models.training.lightning_callbacks import ModelCheckpointCallback
from pytabkit.models.training.lightning_modules import TabNNModule

from realmlp import _atomic_json, _matrix, _positive_int, _sha256
from realmlp_categorical import (
    _check_categorical, _fit_schema, _InputSplit, _MixedGraph,
    _prepare_portable_inference, load_realmlp_categorical,
)

ROOT = Path(__file__).resolve().parents[1]
RTOL, ATOL = 1e-5, 2e-6


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("Fixed trajectory exceeded its registered deadline")


def _hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        elif item is None or isinstance(item, (str, int, bool, float)):
            digest.update((type(item).__name__ + ":" + repr(item) + ";").encode())
        else:
            raise TypeError(f"Unsupported state fingerprint type {type(item)}")

    visit(value)
    return digest.hexdigest()


def _tensor_attributes(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Registered and direct/container tensors on modules, not arbitrary objects."""
    result = {}

    def walk(value: Any, path: str) -> None:
        if torch.is_tensor(value):
            result[path] = value
        elif isinstance(value, dict):
            for name, child in value.items():
                walk(child, f"{path}.{name}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}.{index}")

    for prefix, child in module.named_modules():
        for name, value in vars(child).items():
            if name != "_modules":
                walk(value, f"{prefix}.{name}")
    return result


def _modes_and_methods(module: torch.nn.Module) -> list:
    return [(name, child.training, {
        method: id(getattr(getattr(child, method), "__func__", getattr(child, method)))
        for method in ("forward", "forward_tensors", "_multiple", "_binary") if hasattr(child, method)
    }) for name, child in module.named_modules()]


def _cuda_devices(module: TabNNModule) -> list[int]:
    values = [*_tensor_attributes(module.model).values(),
              *_tensor_attributes(module.creator.static_model).values()]
    return sorted({value.device.index for value in values if value.device.type == "cuda"})


def _rng_state(devices: list[int]) -> dict:
    return {"cpu": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": {str(device): torch.cuda.get_rng_state(device) for device in devices}}


@contextmanager
def _passive(module: TabNNModule):
    owners = (module.model, module.creator.static_model)
    modes = [(child, child.training) for owner in owners for child in owner.modules()]
    devices = _cuda_devices(module)
    before = _hash(_rng_state(devices))
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for owner in owners:
                owner.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for child, mode in modes:
            child.training = mode
        if before != _hash(_rng_state(devices)):
            raise RuntimeError("Observation altered an active random generator")


def _live_signature(module: TabNNModule) -> dict[str, str]:
    return {
        "network": _hash(_tensor_attributes(module.model)),
        "static": _hash(_tensor_attributes(module.creator.static_model)),
        "optimizer": _hash(module.optimizers(use_pl_optimizer=False).opt.state_dict()),
        "rng": _hash(_rng_state(_cuda_devices(module))),
        "progress": _hash(vars(module.progress)),
        "schedule": _hash(module.hp_manager.get_hyper_sched_values()),
        "network_modes_methods": _hash(_modes_and_methods(module.model)),
        "static_modes_methods": _hash(_modes_and_methods(module.creator.static_model)),
    }


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(labels), 2) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise FloatingPointError("Invalid binary probabilities")
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0)
    selected = p[np.arange(len(labels)), labels]
    return {"auc": float(roc_auc_score(labels, p[:, 1])),
            "log_loss": float(-np.log(np.clip(selected, 1e-30, 1)).mean()),
            "brier": float(np.square(p[:, 1] - labels).mean())}


def _dataset(values: np.ndarray, schema: dict, labels: np.ndarray | None, device: str) -> DictDataset:
    raw = torch.as_tensor(values, device=device)
    tensors = _InputSplit(schema).to(device)(raw)
    infos = {"x_cont": TensorInfo(feat_shape=[len(schema["continuous_indices"])]),
             "x_cat": TensorInfo(cat_sizes=schema["cardinalities"])}
    if labels is not None:
        tensors["y"] = torch.as_tensor(labels, device=device).reshape(-1, 1)
        infos["y"] = TensorInfo(cat_sizes=[2])
    return DictDataset(tensors, infos, device=device)


def _native_probabilities(module: TabNNModule, values: np.ndarray, schema: dict, device: str) -> np.ndarray:
    dataset = _dataset(values, schema, None, device)
    with _passive(module):
        outputs = [module.predict_step(batch, index)
                   for index, batch in enumerate(module.get_predict_dataloader(dataset))]
        result = torch.cat(outputs, dim=-2).softmax(-1)
        if result.shape != (1, len(values), 2):
            raise RuntimeError("Wrong ensemble/split prediction axes")
        return result[0].cpu().numpy().copy()


def _configuration(config: dict, horizon: int) -> dict:
    params = {
        "device": "cuda", "random_state": 20261005, "n_threads": 4, "verbosity": 0,
        "n_epochs": horizon, "n_ens": 8, "batch_size": 256, "predict_batch_size": 2048,
        "lr": .053, "wd": .015, "sq_mom": .988, "lr_sched": "flat_anneal", "wd_sched": "cos_log_15",
        "first_layer_lr_factor": .25, "embedding_size": 5, "max_one_hot_cat_size": 18,
        "hidden_sizes": [512, 256, 128], "act": "silu", "p_drop": .05,
        "p_drop_sched": "invsqrtp1e-3", "plr_hidden_1": 16, "plr_hidden_2": 8,
        "plr_act_name": "gelu", "plr_lr_factor": .1151, "plr_sigma": 2.33,
        "ls_eps": 0.0, "ls_eps_sched": "sqrt_cos", "add_front_scale": False,
        "bias_init_mode": "neg-uniform-dynamic-2",
        "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
    }
    aliases = {"seed": "random_state", "threads": "n_threads", "eval_batch_size": "predict_batch_size",
               "learning_rate": "lr", "weight_decay": "wd"}
    normalized = {}
    for key, value in config.items():
        if key == "categorical_indices":
            continue
        target = aliases.get(key, key)
        if target not in params or target == "n_epochs":
            raise ValueError(f"Unsupported fixed-trajectory parameter: {key}; use horizon argument for schedules")
        if target in normalized and normalized[target] != value:
            raise ValueError(f"Conflicting aliases for {target}")
        normalized[target] = value
    params.update(normalized)
    for name in ("n_threads", "n_ens", "batch_size", "predict_batch_size"):
        params[name] = _positive_int(params[name], name)
    if params["n_ens"] != 8:
        raise ValueError("This registered adapter requires eight ensemble members")
    if params["ls_eps"] != 0 or params["lr_sched"] != "flat_anneal" or params["wd_sched"] != "cos_log_15":
        raise ValueError("Registered raw-plus-auxiliary smoothing/schedule semantics must remain fixed")
    if isinstance(params["random_state"], bool) or int(params["random_state"]) != params["random_state"]:
        raise ValueError("Seed must be an integer")
    params["random_state"] = int(params["random_state"])
    params.update(n_cv=1, n_refit=0, n_repeats=1, val_fraction=0.0, use_early_stopping=False,
                  use_best_mean_epoch_for_cv=True, val_metric_name="1-auc_ovr", ens_av_before_softmax=False)
    return params


def _parameter_inventory(module: TabNNModule) -> list[dict]:
    return [{"name": name, "scope": str(parameter.context.scope), "shape": list(parameter.shape),
             "count": parameter.numel(), "requires_grad": parameter.requires_grad,
             "lr_factor": float(parameter.hyper_factors.get("lr", 1.0)),
             "wd_factor": float(parameter.hyper_factors.get("wd", 1.0))}
            for name, parameter in module.model.named_parameters()]


def _schedule(module: TabNNModule, params: dict, inventory: list[dict]) -> dict:
    sched = module.hp_manager.get_hyper_sched_values()
    base_lr, base_wd = params["lr"] * sched["lr"][""], params["wd"] * sched["wd"][""]
    return {"epoch_float": float(module.progress.epoch_float),
            "fraction": float(module.progress.get_fit_progress()),
            "base_lr": float(base_lr), "base_wd": float(base_wd),
            "dropout": float(params["p_drop"] * sched["p_drop"][""]),
            "groups": [{"name": group["name"], "lr": float(base_lr * group["lr_factor"]),
                        "legacy_decay_coefficient": float(base_lr * base_wd * (group["lr_factor"] * group["wd_factor"]) ** 2)}
                       for group in inventory if group["requires_grad"]]}


def _memory(device: str) -> dict:
    result = {"process_rss_bytes": psutil.Process().memory_info().rss}
    if device.startswith("cuda"):
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    return result


class _FixedModule(TabNNModule):
    def __init__(self, **config):
        super().__init__(**config)
        self.monitor_curve: dict[int, dict] = {}

    def on_validation_epoch_end(self) -> None:
        prediction = self._postprocess_ens_pred(torch.cat(self.val_preds, dim=-2)).softmax(-1)
        labels = self.val_dl.val_y[::self.config["n_ens"]]
        if prediction.shape[0] != 1 or labels.shape[0] != 1:
            raise RuntimeError("Only one validation split is allowed")
        self.monitor_curve[int(self.trainer.current_epoch + 1)] = _metrics(
            labels[0, :, 0].detach().cpu().numpy(), prediction[0].detach().cpu().numpy())
        super().on_validation_epoch_end()


class _Observer(pl.Callback):
    def __init__(self, values: np.ndarray, labels: np.ndarray, schema: dict, params: dict,
                 output: Path, endpoints: tuple[int, ...], context: dict, deadline: float, sources: dict):
        self.values, self.labels, self.schema, self.params = values, labels, schema, params
        self.output, self.endpoints, self.context, self.deadline, self.sources = output, endpoints, context, deadline, sources
        self.started = time.monotonic()
        self.inventory: list[dict] = []
        self.completed = 0
        self.exported: dict[str, dict] = {}
        self.curves: list[dict] = []
        self.trace = hashlib.sha256()
        self.trace_enabled = bool(context.get("synthetic_trace", False))
        self.loss_sum = None
        self.sample_count = 0
        self.epoch_first_schedule = None
        self.epoch_last_schedule = None
        # Local generator only, deterministic fixed sample from this fit's rows.
        rng = np.random.default_rng(20261005)
        count = min(2048, len(values))
        selected = np.sort(rng.choice(len(values), size=count, replace=False))
        if len(np.unique(labels[selected])) < 2:
            raise ValueError("Fixed fit probe must contain both classes")
        self.probe_indices = selected

    def on_train_start(self, trainer, pl_module):
        self.inventory = _parameter_inventory(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        self.loss_sum = None
        self.sample_count = 0
        self.epoch_first_schedule = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        _check_deadline(self.deadline)
        if self.epoch_first_schedule is None:
            self.epoch_first_schedule = _schedule(pl_module, self.params, self.inventory)
        if batch_idx == len(pl_module.train_dl) - 1:
            self.epoch_last_schedule = _schedule(pl_module, self.params, self.inventory)
        if self.trace_enabled:
            self.trace.update(_hash({"batch": batch, "rng": _rng_state(_cuda_devices(pl_module)),
                                     "progress": vars(pl_module.progress),
                                     "schedule": pl_module.hp_manager.get_hyper_sched_values()}).encode())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not torch.is_tensor(loss):
            raise RuntimeError("Expected tensor training objective")
        rows = batch["y"].shape[-2]
        contribution = loss.detach() * rows
        self.loss_sum = contribution if self.loss_sum is None else self.loss_sum + contribution
        self.sample_count += rows

    def _export(self, module: TabNNModule, epoch: int) -> dict:
        _check_deadline(self.deadline)
        folder = self.output / f"epoch_{epoch:03d}"
        folder.mkdir(exist_ok=False)
        graph = _MixedGraph(_InputSplit(copy.deepcopy(self.schema)),
                            copy.deepcopy(module.creator.static_model), copy.deepcopy(module.model), 8)
        live = [*_tensor_attributes(module.model).values(), *_tensor_attributes(module.creator.static_model).values()]
        cloned = list(_tensor_attributes(graph).values())
        if not {x.untyped_storage().data_ptr() for x in live if x.numel()}.isdisjoint(
                {x.untyped_storage().data_ptr() for x in cloned if x.numel()}):
            raise RuntimeError("Snapshot aliases live tensor storage")
        graph = graph.cpu().eval()
        if any(x.device.type != "cpu" for x in _tensor_attributes(graph).values()):
            raise RuntimeError("Unregistered snapshot tensor did not move to CPU")
        probe = np.ascontiguousarray(self.values[self.probe_indices[:min(257, len(self.probe_indices))]])
        unknown, unseen = probe[:min(17, len(probe))].copy(), probe[:min(17, len(probe))].copy()
        unknown[:, self.schema["categorical_indices"]] = -1
        for index, vocabulary in zip(self.schema["categorical_indices"], self.schema["vocabularies"]):
            candidate = max(vocabulary, default=0) + 1
            if candidate >= 2**24:
                candidate = -2**24 + 1
            if candidate in vocabulary or candidate == -1:
                raise RuntimeError("Cannot construct unseen category probe")
            unseen[:, index] = candidate
        probes = [probe[:1], probe[:min(17, len(probe))], probe, probe[::-1].copy(), unknown, unseen]
        references = [_native_probabilities(module, value, self.schema, self.params["device"]) for value in probes]
        rewrites = _prepare_portable_inference(graph)
        traced = torch.jit.trace(graph, torch.from_numpy(probe[:min(7, len(probe))]),
                                 check_inputs=[(torch.from_numpy(x),) for x in probes], check_tolerance=RTOL)
        for node in traced.inlined_graph.nodes():
            if node.kind() == "prim::Constant":
                value = node.output().toIValue()
                if torch.is_tensor(value) and value.numel() > 1:
                    raise RuntimeError("Native graph contains immovable tensor constant")
        path = folder / "graph.pt"
        temporary = folder / ".graph.pt.tmp"
        torch.jit.save(traced, str(temporary))
        temporary.replace(path)
        metadata = {"schema_version": 1, "model_type": "realmlp_categorical", "n_features": self.values.shape[1],
                    "classes": [0, 1], "input_schema": self.schema, "constructor": self.params,
                    "requested_config": self.params, "best_epoch": epoch, "executed_epochs": epoch,
                    "schedule_horizon_epochs": self.params["n_epochs"], "train_rows": len(self.values),
                    "network_sha256": _hash(_tensor_attributes(module.model)),
                    "selection": "preregistered fixed endpoint; no metric selection", "context": self.context,
                    "graph_file": path.name, "graph_sha256": _sha256(path),
                    "native_export_tolerance": {"rtol": RTOL, "atol": ATOL},
                    "portable_inference_operations": rewrites, "source_sha256": self.sources,
                    "exact_training_resume": False, "cross_platform_determinism_guaranteed": False}
        _atomic_json(folder / "metadata.json", metadata)
        loaded = load_realmlp_categorical(folder, device="cpu")
        errors = []
        for value, reference in zip(probes, references):
            predicted = loaded.predict_proba(value)
            np.testing.assert_allclose(predicted, reference, atol=ATOL, rtol=RTOL)
            errors.append(float(np.max(np.abs(predicted - reference))))
        np.testing.assert_array_equal(loaded.predict_proba(unknown), loaded.predict_proba(unseen))
        metadata.update(native_export_max_abs_error=max(errors), native_export_probe_rows=[len(x) for x in probes],
                        native_export_probe_errors=errors, native_reference="installed prediction loader/step",
                        exported_utc=_utc())
        _atomic_json(folder / "metadata.json", metadata)
        _check_deadline(self.deadline)
        return {"path": str(folder.resolve()), "epoch": epoch, "horizon": self.params["n_epochs"],
                "network_sha256": metadata["network_sha256"],
                "graph_sha256": metadata["graph_sha256"], "metadata_sha256": _sha256(folder / "metadata.json"),
                "native_export_max_abs_error": max(errors)}

    def on_train_epoch_end(self, trainer, pl_module):
        _check_deadline(self.deadline)
        epoch = int(trainer.current_epoch + 1)
        if self.loss_sum is None or self.sample_count == 0:
            raise RuntimeError("No observed optimization steps")
        loss = float(self.loss_sum.cpu()) / self.sample_count
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite training objective")
        endpoint = epoch in self.endpoints
        before = _live_signature(pl_module) if endpoint else None
        with _passive(pl_module):
            probe_metrics = _metrics(self.labels[self.probe_indices], _native_probabilities(
                pl_module, self.values[self.probe_indices], self.schema, self.params["device"]))
            if endpoint:
                self.exported[str(epoch)] = self._export(pl_module, epoch)
        unchanged = None
        if endpoint:
            after = _live_signature(pl_module)
            unchanged = {key: before[key] == after[key] for key in before}
            if not all(unchanged.values()):
                raise RuntimeError(f"Endpoint export changed live state: {unchanged}")
        row = {"utc": _utc(), "context": self.context, "epoch": epoch, "updates": int(trainer.global_step),
               "train_rows": len(self.values), "iterated_rows": self.sample_count,
               "horizon": self.params["n_epochs"], "epoch_fraction_after": float(pl_module.progress.get_fit_progress()),
               "objective_sum_over_members_mean_rows": loss, "objective_mean_per_member": loss / 8,
               "objective_definition": "train-mode criterion, row mean then sum over 8 members; includes dropout",
               "train_eval_probe": probe_metrics, "monitor": pl_module.monitor_curve.get(epoch),
               "first_update_schedule": self.epoch_first_schedule, "last_update_schedule": self.epoch_last_schedule,
               "elapsed_seconds": time.monotonic() - self.started, "memory": _memory(self.params["device"]),
               "endpoint_live_state_unchanged": unchanged}
        with (self.output / "curves.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        self.curves.append(row)
        self.completed = epoch
        _check_deadline(self.deadline)


def fit_fixed_trajectory(X_train: np.ndarray, y_train: np.ndarray,
                         X_monitor: np.ndarray | None, y_monitor: np.ndarray | None,
                         config: dict, output_dir: Path, *, horizon: int,
                         endpoint_epochs: list[int] | tuple[int, ...], context: dict,
                         deadline_monotonic: float) -> dict:
    """Train one immutable schedule and export its predeclared literal endpoints.

    Only context phase ``inner`` may receive monitoring labels. ``synthetic`` is
    available for generated-data tests. Existing external transform.json is
    allowed; all adapter-owned files/endpoints must be absent. Partial failures
    are retained, marked failed, and never resumed or overwritten automatically.
    """
    _check_deadline(deadline_monotonic)
    if not np.isfinite(deadline_monotonic):
        raise ValueError("A finite monotonic deadline is required")
    if importlib.metadata.version("pytabkit") != "1.7.3":
        raise RuntimeError("This adapter is pinned to PyTabKit 1.7.3")
    horizon = _positive_int(horizon, "horizon")
    endpoints = tuple(_positive_int(x, "endpoint epoch") for x in endpoint_epochs)
    if not endpoints or tuple(sorted(set(endpoints))) != endpoints or endpoints[-1] > horizon:
        raise ValueError("Endpoints must be sorted unique positive epochs no later than horizon")
    if not isinstance(context, dict) or context.get("phase") not in {"inner", "outer", "synthetic"}:
        raise ValueError("context.phase must be inner, outer or synthetic")
    context = copy.deepcopy(context)
    json.dumps(context, allow_nan=False)
    if context.get("synthetic_trace", False) and context["phase"] != "synthetic":
        raise ValueError("Full batch-state tracing is restricted to synthetic checks")
    if (X_monitor is None) != (y_monitor is None):
        raise ValueError("Monitor features and labels must be paired")
    if context["phase"] == "outer" and X_monitor is not None:
        raise ValueError("Outer trajectories cannot receive monitoring rows or labels")
    if context["phase"] == "inner" and X_monitor is None:
        raise ValueError("Inner trajectory requires the registered passive monitor")
    train = _matrix(X_train, "X_train")

    def checked_labels(value: np.ndarray, rows: int, name: str) -> np.ndarray:
        labels = np.asarray(value)
        if labels.dtype != np.int64 or labels.shape != (rows,) or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{name} must have int64 binary shape ({rows},)")
        if len(np.unique(labels)) != 2:
            raise ValueError(f"{name} must contain both classes")
        return labels

    target = checked_labels(y_train, len(train), "y_train")
    monitor = None if X_monitor is None else _matrix(X_monitor, "X_monitor")
    monitor_target = None if monitor is None else checked_labels(y_monitor, len(monitor), "y_monitor")
    if monitor is not None and monitor.shape[1] != train.shape[1]:
        raise ValueError("Monitor feature count differs")
    schema = _fit_schema(train, config.get("categorical_indices"))
    if monitor is not None:
        _check_categorical(monitor, schema["categorical_indices"])
    params = _configuration(config, horizon)
    if len(train) < params["batch_size"]:
        raise ValueError("Training rows must contain a full batch")
    device = torch.device(params["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and explicitly owned CUDA are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        index = device.index if device.index is not None else 0
        params["device"] = f"cuda:{index}"
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    owned = [output / name for name in ("trajectory.json", "curves.jsonl")]
    if any(path.exists() for path in owned) or any(output.glob("epoch_*")):
        raise FileExistsError("Preserve existing trajectory artifacts; use a fresh run directory")
    sources = {str(path.relative_to(ROOT)): _sha256(path) for path in [
        Path(__file__), ROOT / "scripts/realmlp.py", ROOT / "scripts/realmlp_categorical.py",
        ROOT / ".venv/Lib/site-packages/pytabkit/models/optim/optimizers.py",
        ROOT / ".venv/Lib/site-packages/pytabkit/models/training/lightning_modules.py",
        ROOT / ".venv/Lib/site-packages/pytabkit/models/training/nn_creator.py"]}
    seed = params["random_state"]
    # Exactly the public sklearn_base.py split seed/sub-split seed mapping.
    sub_seed = int(np.random.RandomState(seed).randint(0, 2**31 - 1))
    resolved = RealMLP_TD_Classifier(**params).get_config()
    resolved["use_best_epoch"] = False
    factory = copy.deepcopy(resolved)
    if factory.get("num_emb_type") == "pbld":
        factory.update(use_plr_embeddings=True, plr_use_densenet=True, plr_use_cos_bias=True, plr_act_name="linear")
    record = {"schema_version": 1, "status": "running", "started_utc": _utc(), "horizon": horizon,
              "endpoint_epochs": list(endpoints), "executed_epochs": 0, "context": context,
              "train_rows": len(train), "monitor_rows": 0 if monitor is None else len(monitor),
              "input_schema": schema, "constructor": params, "resolved_config": resolved,
              "resolved_factory_config": factory, "split_seed": seed, "sub_split_seed": sub_seed,
              "source_sha256": sources, "input_train_sha256": _hash((train, target)),
              "input_monitor_sha256": None if monitor is None else _hash((monitor, monitor_target)),
              "versions": {name: importlib.metadata.version(name) for name in ("pytabkit", "torch", "pytorch-lightning", "numpy")},
              "device": params["device"], "clock": "epoch_fraction", "legacy_decay_semantics": True,
              "memory_start": _memory(params["device"]), "endpoints": {}}
    observer = _Observer(train, target, schema, params, output, endpoints, context, deadline_monotonic, sources)
    _atomic_json(output / "trajectory.json", record)
    old_threads = torch.get_num_threads()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    started = time.monotonic()
    try:
        torch.set_num_threads(params["n_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.set_device(index)
        with threadpool_limits(limits=params["n_threads"]):
            if monitor is None:
                values, labels = train, target
            else:
                values, labels = np.concatenate((train, monitor)), np.concatenate((target, monitor_target))
            dataset = _dataset(values, schema, labels, "cpu")
            # No-validation public fits still apply RandomSplitter(fraction=1),
            # preserving a seeded full permutation rather than input row order.
            train_indices = (torch.arange(len(train)) if monitor is not None
                             else RandomSplitter(seed, first_fraction=1.0).get_idxs(dataset)[0])
            split = SplitIdxs(train_indices.reshape(1, -1),
                              None if monitor is None else torch.arange(len(train), len(values)).reshape(1, -1),
                              None, split_seed=seed, sub_split_seeds=[sub_seed], split_id=0)
            module = _FixedModule(**resolved)
            module.compile_model(dataset, [split], InterfaceResources(
                n_threads=params["n_threads"], gpu_devices=[] if device.type == "cpu" else [params["device"]]))
            del dataset, values, labels
            callbacks = module.create_callbacks()
            if any(isinstance(callback, ModelCheckpointCallback) for callback in callbacks):
                raise RuntimeError("Fixed trajectory installed a best-checkpoint restoration callback")
            inventory = _parameter_inventory(module)
            record.update(parameter_groups=inventory, initial_network_sha256=_hash(_tensor_attributes(module.model)),
                          updates_per_epoch=len(module.train_dl),
                          training_index_sha256=_hash(train_indices),
                          training_index_policy="contiguous supplied split" if monitor is not None else "public RandomSplitter(seed, fraction=1)",
                          fit_probe_indices_sha256=_hash(observer.probe_indices), fit_probe_rows=len(observer.probe_indices),
                          fit_probe_label_sha256=_hash(target[observer.probe_indices]))
            _atomic_json(output / "trajectory.json", record)
            _check_deadline(deadline_monotonic)
            trainer = pl.Trainer(accelerator="cpu" if device.type == "cpu" else "gpu",
                                 devices=1 if device.type == "cpu" else [index], max_epochs=endpoints[-1],
                                 callbacks=[*callbacks, observer], logger=False, enable_checkpointing=False,
                                 enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                                 log_every_n_steps=1)
            trainer.fit(module, train_dataloaders=module.train_dl, val_dataloaders=module.val_dl)
            if observer.completed != endpoints[-1] or trainer.global_step != endpoints[-1] * record["updates_per_epoch"]:
                raise RuntimeError("Trajectory stopped before the fixed endpoint")
            if set(observer.exported) != {str(epoch) for epoch in endpoints}:
                raise RuntimeError("Missing fixed endpoint artifact")
            for endpoint in observer.exported.values():
                folder = Path(endpoint["path"])
                if _sha256(folder / "graph.pt") != endpoint["graph_sha256"] or _sha256(folder / "metadata.json") != endpoint["metadata_sha256"]:
                    raise RuntimeError("Saved endpoint mutated during continued training")
            record.update(status="complete", completed_utc=_utc(), executed_epochs=observer.completed,
                          optimizer_updates=int(trainer.global_step), endpoints=observer.exported,
                          final_network_sha256=_hash(_tensor_attributes(module.model)),
                          final_optimizer_sha256=_hash(trainer.optimizers[0].opt.state_dict()),
                          final_rng_sha256=_hash(_rng_state(_cuda_devices(module))),
                          synthetic_batch_trace_sha256=observer.trace.hexdigest() if observer.trace_enabled else None,
                          elapsed_seconds=time.monotonic() - started, memory_end=_memory(params["device"]),
                          curves_path=str((output / "curves.jsonl").resolve()), curves_sha256=_sha256(output / "curves.jsonl"),
                          trajectory_path=str((output / "trajectory.json").resolve()))
            _check_deadline(deadline_monotonic)
            # Drop training graph references before the caller reloads endpoints on GPU.
            del trainer, module, callbacks
            gc.collect()
        _atomic_json(output / "trajectory.json", record)
        return record
    except Exception as error:
        record.update(status="failed", failed_utc=_utc(), elapsed_seconds=time.monotonic() - started,
                      error_type=type(error).__name__, error=str(error),
                      executed_epochs=observer.completed, endpoints=observer.exported)
        _atomic_json(output / "trajectory.json", record)
        raise
    finally:
        torch.set_num_threads(old_threads)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
