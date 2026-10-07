"""Short RealMLP training with a native TorchScript inference artifact.

Contract: finite float32 matrices [rows, features], labels 0/1. The caller fits
and saves its outer preprocessing on each training partition. This numeric-only
representation does not reproduce the public notebook's raw categorical twins.
The public estimator owns optimization and fold-local numerical preprocessing.
Only the supplied inner validation may select epochs; no validation means fixed
epochs over every supplied training row. Inference requires torch, not pytabkit.

Source inspected: pytabkit 1.7.3, Apache-2.0, https://github.com/dholzmueller/pytabkit
Recipe: https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit
Export uses version-specific fitted internals, gated against the public predictor.
The graph is native executable model code; load only locally trusted artifacts.
It contains no estimator pickle and does not support optimizer-state resumption.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from uuid import uuid4

import numpy as np
import torch


def _matrix(value: np.ndarray, name: str, *, empty: bool = False) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] == 0 or (not empty and not len(result)):
        raise ValueError(f"{name} must have shape [rows, positive feature count]")
    if result.dtype != np.float32:
        raise TypeError(f"{name} must be float32")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite; fit imputation inside folds")
    return np.ascontiguousarray(result)


def _labels(value: np.ndarray, rows: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (rows,) or not np.isin(result, [0, 1]).all():
        raise ValueError(f"{name} must be binary 0/1 with shape [{rows}]")
    if len(np.unique(result)) != 2:
        raise ValueError(f"{name} must contain both classes")
    return result.astype(np.int64)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _InferenceGraph(torch.nn.Module):
    """Reproduce n_cv=1 predictor: preprocess, vectorized members, mean softmax."""

    def __init__(self, static: torch.nn.Module, network: torch.nn.Module, n_ens: int):
        super().__init__()
        self.static = static
        self.network = network
        self.n_ens = n_ens

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tensors = self.static({
            "x_cont": values,
            "x_cat": values.new_zeros((values.shape[0], 0), dtype=torch.long),
        })
        # Each ensemble member receives identical rows; axis 0 is member, 1 row.
        tensors = {key: value.unsqueeze(0).expand(self.n_ens, -1, -1)
                   for key, value in tensors.items()}
        logits = self.network(tensors)["x_cont"]
        return torch.softmax(logits, dim=-1).mean(dim=0)


class RealMLPClassifier:
    def __init__(self, model: torch.jit.ScriptModule, metadata: dict[str, Any], device: str):
        self.model = model.to(device).eval()
        self.metadata = metadata
        self.device = torch.device(device)
        self.n_features_in_ = int(metadata["n_features"])
        self.classes_ = np.array([0, 1], dtype=np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = _matrix(X, "X", empty=True)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match saved RealMLP")
        result = np.empty((len(values), 2), dtype=np.float32)
        size = int(self.metadata["constructor"]["predict_batch_size"])
        with torch.inference_mode():
            for start in range(0, len(values), size):
                batch = torch.as_tensor(values[start:start + size], device=self.device)
                result[start:start + size] = self.model(batch).cpu().numpy()
        if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
            raise FloatingPointError("RealMLP produced invalid probabilities")
        return result

    def save(self, output_path: Path) -> None:
        directory = Path(output_path)
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"graph-{uuid4().hex}.pt"
        path = directory / filename
        temporary = directory / f".{filename}.tmp"
        # Export on CPU for portable inference; restore the active execution device.
        try:
            self.model.to("cpu")
            with temporary.open("wb") as stream:
                torch.jit.save(self.model, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            self.model.to(self.device)
            temporary.unlink(missing_ok=True)
        metadata = dict(self.metadata, graph_file=filename, graph_sha256=_sha256(path))
        _atomic_json(directory / "metadata.json", metadata)
        self.metadata = metadata


def load_realmlp(output_path: Path, device: str | None = None) -> RealMLPClassifier:
    """Load a hash-checked, locally trusted native graph; never unpickle estimator."""
    directory = Path(output_path)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("model_type") != "realmlp":
        raise ValueError("Unsupported RealMLP artifact schema")
    name = metadata["graph_file"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Invalid graph filename")
    path = directory / name
    if _sha256(path) != metadata["graph_sha256"]:
        raise ValueError("RealMLP graph checksum mismatch")
    device = device or metadata["constructor"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = torch.jit.load(str(path), map_location="cpu").eval()
    return RealMLPClassifier(model, metadata, device)


def fit_realmlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    config: dict[str, Any],
    output_path: Path,
) -> tuple[RealMLPClassifier, int]:
    """Fit through the official library, export, verify, and return selected epoch."""
    from pytabkit import RealMLP_TD_Classifier

    if importlib.metadata.version("pytabkit") != "1.7.3":
        raise RuntimeError("Fitted-internal export is verified only for pytabkit 1.7.3")
    values = _matrix(X_train, "X_train")
    labels = _labels(y_train, len(values), "y_train")
    if (X_valid is None) != (y_valid is None):
        raise ValueError("Both validation arguments must be provided, or neither")
    valid = valid_labels = None
    if X_valid is not None:
        valid = _matrix(X_valid, "X_valid")
        valid_labels = _labels(y_valid, len(valid), "y_valid")
        if valid.shape[1] != values.shape[1]:
            raise ValueError("Training and validation feature counts differ")

    # Numerical twins and category handling are deliberately absent in this API.
    params = {
        "device": "cuda", "random_state": 42, "n_threads": 4, "verbosity": 0,
        "n_epochs": 4, "n_ens": 8, "batch_size": 256, "predict_batch_size": 2048,
        "lr": 0.053, "wd": 0.015, "sq_mom": 0.988, "lr_sched": "flat_anneal",
        "wd_sched": "cos_log_15", "first_layer_lr_factor": 0.25,
        "embedding_size": 5, "max_one_hot_cat_size": 18,
        "hidden_sizes": [512, 256, 128], "act": "silu", "p_drop": 0.05,
        "p_drop_sched": "invsqrtp1e-3", "plr_hidden_1": 16, "plr_hidden_2": 8,
        "plr_act_name": "gelu", "plr_lr_factor": 0.1151, "plr_sigma": 2.33,
        "ls_eps": 0.01, "ls_eps_sched": "sqrt_cos", "add_front_scale": False,
        "bias_init_mode": "neg-uniform-dynamic-2",
        "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip",
                 "embedding", "l2_normalize"],
    }
    aliases = {"seed": "random_state", "epochs": "n_epochs", "threads": "n_threads",
               "eval_batch_size": "predict_batch_size", "learning_rate": "lr",
               "weight_decay": "wd"}
    normalized = {}
    for key, value in config.items():
        target = aliases.get(key, key)
        if target not in params:
            raise ValueError(f"Unsupported RealMLP configuration field: {key}")
        if target in normalized and normalized[target] != value:
            raise ValueError(f"Conflicting aliases for {target}")
        normalized[target] = value
    params.update(normalized)
    for key in ("n_epochs", "n_ens", "batch_size", "predict_batch_size", "n_threads"):
        params[key] = _positive_int(params[key], key)
    if str(params["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    params.update(n_cv=1, n_refit=0, n_repeats=1, val_fraction=0.0,
                  use_early_stopping=False, use_best_mean_epoch_for_cv=True,
                  val_metric_name="1-auc_ovr", ens_av_before_softmax=False)
    start = time.monotonic()
    estimator = RealMLP_TD_Classifier(**params)
    estimator.fit(values, labels, X_val=valid, y_val=valid_labels,
                  cat_indicator=[False] * values.shape[1])
    if not np.array_equal(estimator.classes_, [0, 1]):
        raise RuntimeError("Unexpected fitted class order")
    selected = params["n_epochs"]
    if valid is not None:
        selected = int(estimator.fit_params_["stop_epoch"]["1-auc_ovr"])
        if not 1 <= selected <= params["n_epochs"]:
            raise RuntimeError("No finite inner-validation checkpoint was selected")

    estimator.to("cpu")
    fitted = estimator.alg_interface_.model
    if fitted.creator.n_tt_splits != 1 or fitted.creator.n_tv_splits != 1:
        raise RuntimeError("Unexpected vectorized split dimensions")
    graph = _InferenceGraph(fitted.creator.static_model, fitted.model, params["n_ens"]).eval()
    probe = np.ascontiguousarray(values[:min(len(values), 257)])
    # Changing rows and values catches traced batch-size constants and ordering bugs.
    probes = [probe[:1], probe[:min(17, len(probe))], probe, probe[::-1].copy()]
    if valid is not None:
        probes.append(np.ascontiguousarray(valid[:min(len(valid), 31)]))
    with torch.inference_mode():
        traced = torch.jit.trace(graph, torch.from_numpy(probe[:min(len(probe), 7)]),
                                 check_inputs=[(torch.from_numpy(p),) for p in probes])
    # Reject graphs with nonscalar constants that would prevent device portability.
    for node in traced.inlined_graph.nodes():
        if node.kind() == "prim::Constant":
            value = node.output().toIValue()
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                raise RuntimeError("Traced RealMLP contains an immovable tensor constant")
    max_error = 0.0
    for sample in probes:
        native = estimator.predict_proba(sample)
        with torch.inference_mode():
            exported = traced(torch.from_numpy(sample)).numpy()
        np.testing.assert_allclose(exported, native, rtol=1e-5, atol=2e-6)
        max_error = max(max_error, float(np.max(np.abs(exported - native))))
    metadata = {
        "schema_version": 1, "model_type": "realmlp", "n_features": values.shape[1],
        "classes": [0, 1], "constructor": params, "best_epoch": selected,
        "requested_config": config,
        "input_contract": "finite float32 numeric matrix; caller persists fold-local preprocessing",
        "representation_limit": "No raw categorical twins; differs from public notebook",
        "train_rows": len(values), "validation_rows": 0 if valid is None else len(valid),
        "selection": "inner ROC AUC only" if valid is not None else "fixed epochs, all supplied rows",
        "native_export_max_abs_error": max_error,
        "native_export_tolerance": {"rtol": 1e-5, "atol": 2e-6},
        "native_export_probe_rows": [len(p) for p in probes],
        "exact_training_resume": False, "cross_platform_determinism_guaranteed": False,
        "versions": {name: importlib.metadata.version(name)
                     for name in ["pytabkit", "torch", "pytorch-lightning", "numpy", "scikit-learn"]},
        "torch_cuda": torch.version.cuda,
        "code_sha256": _sha256(Path(__file__)),
        "elapsed_seconds": time.monotonic() - start,
    }
    # Trace and estimator share fitted module storage. Never call the CPU estimator
    # after moving the exported graph to CUDA: its predict() would move shared
    # weights back to CPU underneath the returned wrapper.
    reference = estimator.predict_proba(probe)
    result = RealMLPClassifier(traced, metadata, str(params["device"]))
    result.save(output_path)
    reloaded = load_realmlp(output_path, device="cpu")
    # Artifact publication is accepted only after actual disk reload predicts correctly.
    np.testing.assert_allclose(reloaded.predict_proba(probe), reference,
                               rtol=1e-5, atol=2e-6)
    return result, selected


# Drop-in naming for callers that dispatch neural families through one interface.
fit_neural = fit_realmlp
load_neural = load_realmlp


def _smoke(device: str = "cpu") -> None:
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(2)
    rng = np.random.default_rng(782)
    X = rng.normal(size=(800, 6)).astype(np.float32)
    X[:, 5] = 1.0  # Ensure train-constant columns survive export correctly.
    y = (X[:, 0] + 0.8 * X[:, 1] + rng.normal(scale=0.5, size=len(X)) > 0).astype(np.int64)
    config = {"device": device, "seed": 782, "threads": 2, "epochs": 2,
              "n_ens": 8 if device == "cuda" else 2,
              "hidden_sizes": [512, 256, 128] if device == "cuda" else [32, 16],
              "batch_size": 256 if device == "cuda" else 64,
              "eval_batch_size": 37}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="realmlp-smoke-") as temporary:
        root = Path(temporary)
        selected_model, selected = fit_realmlp(X[:600], y[:600], X[600:], y[600:], config, root / "inner")
        fixed, fixed_epoch = fit_realmlp(X[:600], y[:600], None, None,
                                        dict(config, epochs=selected), root / "fixed")
        loaded = load_realmlp(root / "fixed", device=device)
        cpu = load_realmlp(root / "fixed", device="cpu")
        preds = fixed.predict_proba(X[600:])
        restored = loaded.predict_proba(X[600:])
        if device == "cpu":
            np.testing.assert_array_equal(preds, restored)
        else:
            # CUDA/JIT graph execution may change float32 reduction ordering.
            np.testing.assert_allclose(preds, restored, rtol=1e-5, atol=2e-6)
        reload_error = float(np.max(np.abs(preds - restored)))
        assert loaded.predict_proba(X[:0]).shape == (0, 2)
        np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=2e-6)
        assert fixed_epoch == selected
        device_errors = {}
        for rows in (1, 17, 257):
            native_device = fixed.predict_proba(X[-rows:])
            cpu_device = cpu.predict_proba(X[-rows:])
            restored = loaded.predict_proba(X[-rows:])
            np.testing.assert_allclose(native_device, restored, rtol=1e-5, atol=2e-6)
            reload_error = max(reload_error, float(np.max(np.abs(native_device - restored))))
            np.testing.assert_allclose(native_device, cpu_device, rtol=2e-5, atol=2e-6)
            device_errors[rows] = float(np.max(np.abs(native_device - cpu_device)))
        print(json.dumps({"status": "passed", "selected_epoch": selected,
                          "fixed_epoch": fixed_epoch, "reload_max_abs_error": reload_error,
                          "inner_export_error": selected_model.metadata["native_export_max_abs_error"],
                          "fixed_export_error": fixed.metadata["native_export_max_abs_error"],
                          "synthetic_auc": roc_auc_score(y[600:], preds[:, 1]),
                          "device": device, "gpu_used": device == "cuda",
                          "n_ens": config["n_ens"], "hidden_sizes": config["hidden_sizes"],
                          "cpu_device_probe_errors": device_errors,
                          "seconds": time.monotonic() - started,
                          "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20
                          if device == "cuda" else 0.0}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("This module exposes fit_realmlp; use --smoke for the bounded CPU check")
    _smoke(args.device)
