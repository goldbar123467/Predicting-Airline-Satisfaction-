"""RealMLP with training-only vocabularies for numeric categorical twins.

Input is a finite float32 matrix. ``categorical_indices`` identify integer-valued
nominal columns; all other columns remain continuous. The caller must supply raw
numeric twins or training-fold category codes without scaling these columns.
The reserved input value -1 and every unseen category map to native category 0.
Vocabularies and cardinalities are learned from X_train only, including when an
inner stopping set is supplied. Keep the continuous copies in separate columns.

PyTabKit 1.7.3's public estimator owns training. A thin NNAlgInterface adapter
replaces its all-numeric input with continuous and frozen categorical tensors;
this avoids its default vocabulary fit on concatenated train+validation rows.
Inference reuses the same lookup/split module in a native TorchScript graph.
No estimator pickle is saved. Optimizer resumption is unsupported.

Library semantics: https://github.com/dholzmueller/pytabkit (Apache-2.0),
models/nn_models/categorical.py in version 1.7.3. Recipe source inspected:
https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit (2026-10-02 UTC).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import tempfile
import time
from types import MethodType
from typing import Any

import numpy as np
import torch

from realmlp import RealMLPClassifier, _labels, _matrix, _positive_int, _sha256


class _Lookup(torch.nn.Module):
    def __init__(self, vocabulary: list[float]):
        super().__init__()
        self.register_buffer("vocabulary", torch.tensor(vocabulary, dtype=torch.float32))
        self.n_categories = len(vocabulary)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # Always create a contiguous tensor. Using contiguous() only inside
        # searchsorted changes traced aliases when a one-row slice is already
        # contiguous, despite otherwise identical category semantics.
        values = values.clone(memory_format=torch.contiguous_format)
        if self.n_categories == 0:
            return torch.zeros_like(values, dtype=torch.long)
        positions = torch.searchsorted(self.vocabulary, values)
        safe = positions.clamp(max=self.n_categories - 1)
        known = (positions < self.n_categories) & (self.vocabulary[safe] == values)
        return torch.where(known, positions + 1, torch.zeros_like(positions))


class _InputSplit(torch.nn.Module):
    def __init__(self, schema: dict[str, Any]):
        super().__init__()
        self.categorical_indices = schema["categorical_indices"]
        self.register_buffer("continuous_indices", torch.tensor(
            schema["continuous_indices"], dtype=torch.long))
        self.lookups = torch.nn.ModuleList([_Lookup(v) for v in schema["vocabularies"]])

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        continuous = values.index_select(1, self.continuous_indices)
        categorical = torch.stack([
            lookup(values[:, index])
            for index, lookup in zip(self.categorical_indices, self.lookups)
        ], dim=1)
        return {"x_cont": continuous, "x_cat": categorical}


class _MixedGraph(torch.nn.Module):
    def __init__(self, splitter: _InputSplit, static: torch.nn.Module,
                 network: torch.nn.Module, n_ens: int):
        super().__init__()
        self.splitter = splitter
        self.static = static
        self.network = network
        self.n_ens = n_ens

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tensors = self.static(self.splitter(values))
        tensors = {key: value.unsqueeze(0).expand(self.n_ens, -1, -1)
                   for key, value in tensors.items()}
        return self.network(tensors)["x_cont"].softmax(dim=-1).mean(dim=0)


def _portable_onehot_multiple(self, x_cat, on_value, off_value):
    cont = x_cat.new_full((*x_cat.shape[:-1], self.cat_size), off_value, dtype=torch.float32)
    src = x_cat.new_full([1] * x_cat.dim(), on_value, dtype=torch.float32).expand_as(x_cat)
    return cont.scatter_(dim=-1, index=x_cat, src=src)


def _portable_onehot_binary(self, x_cat, values):
    # new_full inherits device dynamically; torch.as_tensor(..., device=x.device)
    # instead hardcodes the tracing device and embeds a nonportable constant.
    src = torch.stack([x_cat.new_full((), float(v), dtype=torch.float32) for v in values])
    src = src.reshape(*([1] * (x_cat.dim() - 1)), len(values))
    src = src.expand(*x_cat.shape[:-1], len(values))
    return src.gather(dim=-1, index=x_cat)


def _portable_embedding(self, tensors):
    indices = tensors["x_cat"].squeeze(-1)
    n_batch = indices.shape[-1]
    table = self.emb.reshape(-1, self.emb.shape[-1])
    flattened_size = self.emb.shape[-2]
    while indices.dim() > 1:
        n_parallel = indices.shape[-2]
        # Integer cumsum gives the same offsets as arange while inheriting device.
        offsets = indices.new_ones((n_parallel,)).cumsum(dim=0) - 1
        indices = indices + flattened_size * offsets[:, None]
        indices = indices.reshape(*indices.shape[:-2], -1)
        flattened_size *= n_parallel
    continuous = table.index_select(0, indices)
    continuous = continuous.reshape(*self.emb.shape[:-2], n_batch, self.emb.shape[-1])
    output = {key: value for key, value in tensors.items() if key != "x_cat"}
    output["x_cont"] = continuous
    return output


def _portable_encoding(self, tensors):
    x_cat = tensors["x_cat"]
    continuous = [tensors[self.enc_output_name]] if self.enc_output_name in tensors else []
    categorical = []
    for index, layer in enumerate(self.emb_layers):
        subset = {"x_cat": x_cat[..., index:index + 1]}
        if "y" in tensors:
            subset["y"] = tensors["y"]
        encoded = layer.forward_tensors(subset)
        if "x_cont" in encoded:
            continuous.append(encoded["x_cont"])
        if "x_cat" in encoded:
            categorical.append(encoded["x_cat"])
    if continuous:
        result = continuous[0] if len(continuous) == 1 else torch.cat(continuous, dim=-1)
    else:
        result = x_cat.new_zeros((*x_cat.shape[:-1], 0), dtype=torch.float32)
    output = {key: value for key, value in tensors.items() if key != "x_cat"}
    output[self.enc_output_name] = result
    if categorical:
        output["x_cat"] = torch.cat(categorical, dim=-1)
    return output


def _prepare_portable_inference(module: torch.nn.Module) -> dict[str, int]:
    """Replace only device-allocating inference operations after native training.

    These are exact equivalents of the pinned library's category transforms,
    preserving fitted tensors, category order, binary encoding, and unknowns.
    Cached unmodified native predictions are the export verification reference.
    """
    from pytabkit.models.nn_models.categorical import EncodingLayer, SingleEmbeddingLayer, SingleOneHotLayer
    counts = {"onehot": 0, "embedding": 0, "encoding": 0}
    for child in module.modules():
        if isinstance(child, SingleOneHotLayer):
            child._multiple = MethodType(_portable_onehot_multiple, child)
            child._binary = MethodType(_portable_onehot_binary, child)
            counts["onehot"] += 1
        elif isinstance(child, SingleEmbeddingLayer):
            child.forward_tensors = MethodType(_portable_embedding, child)
            counts["embedding"] += 1
        elif isinstance(child, EncodingLayer):
            child.forward_tensors = MethodType(_portable_encoding, child)
            counts["encoding"] += 1
    return counts


def _check_categorical(values: np.ndarray, indices: list[int]) -> None:
    categorical = values[:, indices]
    if not np.equal(categorical, np.rint(categorical)).all():
        raise ValueError("Categorical columns must be integer values, without standardization")
    if np.any(np.abs(categorical) >= 2**24):
        raise ValueError("Category values must be exactly representable float32 integers")


def _fit_schema(values: np.ndarray, indices: Any) -> dict[str, Any]:
    if not isinstance(indices, list) or not indices:
        raise ValueError("categorical_indices must be a nonempty list")
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < values.shape[1]
           for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("categorical_indices contains an invalid or duplicate index")
    indices = list(indices)
    continuous = [i for i in range(values.shape[1]) if i not in indices]
    if not continuous:
        raise ValueError("This mixed-input model requires at least one continuous column")
    _check_categorical(values, indices)
    vocabularies = []
    for index in indices:
        unique = np.unique(values[:, index])
        vocabularies.append(unique[unique != -1].tolist())
    return {
        "n_features": values.shape[1], "categorical_indices": indices,
        "continuous_indices": continuous, "vocabularies": vocabularies,
        "cardinalities": [len(v) + 1 for v in vocabularies],
        "unknown_input_value": -1, "unknown_native_code": 0,
        "fit_scope": "X_train only; validation and inference rows never expand vocabularies",
    }


class CategoricalRealMLPClassifier(RealMLPClassifier):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = _matrix(X, "X", empty=True)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match categorical RealMLP")
        _check_categorical(values, self.metadata["input_schema"]["categorical_indices"])
        return super().predict_proba(values)


def load_realmlp_categorical(output_path: Path, device: str | None = None) -> CategoricalRealMLPClassifier:
    directory = Path(output_path)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("model_type") != "realmlp_categorical":
        raise ValueError("Unsupported categorical RealMLP artifact")
    name = metadata["graph_file"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Invalid graph filename")
    path = directory / name
    if _sha256(path) != metadata["graph_sha256"]:
        raise ValueError("Categorical RealMLP graph checksum mismatch")
    device = device or metadata["constructor"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    graph = torch.jit.load(str(path), map_location="cpu").eval()
    return CategoricalRealMLPClassifier(graph, metadata, device)


def fit_realmlp_categorical(
    X_train: np.ndarray, y_train: np.ndarray,
    X_valid: np.ndarray | None, y_valid: np.ndarray | None,
    config: dict[str, Any], output_path: Path,
) -> tuple[CategoricalRealMLPClassifier, int]:
    from pytabkit import RealMLP_TD_Classifier
    from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface
    from pytabkit.models.data.data import DictDataset, TensorInfo

    if importlib.metadata.version("pytabkit") != "1.7.3":
        raise RuntimeError("The adapter and native export are verified only for pytabkit 1.7.3")
    values = _matrix(X_train, "X_train")
    labels = _labels(y_train, len(values), "y_train")
    schema = _fit_schema(values, config.get("categorical_indices"))
    if (X_valid is None) != (y_valid is None):
        raise ValueError("Supply both validation arguments or neither")
    valid = valid_labels = None
    if X_valid is not None:
        valid = _matrix(X_valid, "X_valid")
        valid_labels = _labels(y_valid, len(valid), "y_valid")
        if valid.shape[1] != values.shape[1]:
            raise ValueError("Training and validation feature counts differ")
        _check_categorical(valid, schema["categorical_indices"])

    class MixedInterface(NNAlgInterface):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.splitter = _InputSplit(schema)

        def convert(self, ds):
            # The public converter sees all numeric columns, so it cannot learn a
            # category vocabulary from the concatenated inner validation rows.
            tensors = dict(ds.tensors)
            self.splitter.to(ds.device)
            tensors.update(self.splitter(tensors["x_cont"]))
            infos = dict(ds.tensor_infos)
            infos.update(x_cont=TensorInfo(feat_shape=[len(schema["continuous_indices"])]),
                         x_cat=TensorInfo(cat_sizes=schema["cardinalities"]))
            return DictDataset(tensors=tensors, tensor_infos=infos, device=ds.device)

        def fit(self, ds, *args, **kwargs):
            return super().fit(self.convert(ds), *args, **kwargs)

        def predict(self, ds):
            return super().predict(self.convert(ds))

    class MixedEstimator(RealMLP_TD_Classifier):
        def _create_alg_interface(self, n_cv):
            return MixedInterface(**self.get_config())

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
        "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
    }
    aliases = {"seed": "random_state", "epochs": "n_epochs", "threads": "n_threads",
               "eval_batch_size": "predict_batch_size", "learning_rate": "lr", "weight_decay": "wd"}
    normalized = {}
    for key, value in config.items():
        if key == "categorical_indices":
            continue
        target = aliases.get(key, key)
        if target not in params:
            raise ValueError(f"Unsupported categorical RealMLP parameter: {key}")
        if target in normalized and normalized[target] != value:
            raise ValueError(f"Conflicting parameter aliases: {target}")
        normalized[target] = value
    params.update(normalized)
    for key in ("n_epochs", "n_ens", "batch_size", "predict_batch_size", "n_threads"):
        params[key] = _positive_int(params[key], key)
    if str(params["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    params.update(n_cv=1, n_refit=0, n_repeats=1, val_fraction=0.0,
                  use_early_stopping=False, use_best_mean_epoch_for_cv=True,
                  val_metric_name="1-auc_ovr", ens_av_before_softmax=False)
    started = time.monotonic()
    estimator = MixedEstimator(**params)
    estimator.fit(values, labels, X_val=valid, y_val=valid_labels,
                  cat_indicator=[False] * values.shape[1])
    if not np.array_equal(estimator.classes_, [0, 1]):
        raise RuntimeError("Unexpected fitted class order")
    selected = params["n_epochs"] if valid is None else int(
        estimator.fit_params_["stop_epoch"]["1-auc_ovr"])
    if not 1 <= selected <= params["n_epochs"]:
        raise RuntimeError("No finite checkpoint selected")

    estimator.to("cpu")
    interface = estimator.alg_interface_
    fitted = interface.model
    if fitted.creator.n_tt_splits != 1 or fitted.creator.n_tv_splits != 1:
        raise RuntimeError("Unexpected vectorized split dimensions")
    graph = _MixedGraph(interface.splitter.to("cpu"), fitted.creator.static_model,
                        fitted.model, params["n_ens"]).eval()
    probe = np.ascontiguousarray(values[:min(257, len(values))])
    unknown = probe[:min(17, len(probe))].copy()
    unseen = unknown.copy()
    unknown[:, schema["categorical_indices"]] = -1
    for index, vocabulary in zip(schema["categorical_indices"], schema["vocabularies"]):
        candidate = (max(vocabulary) if vocabulary else 0) + 1
        if candidate >= 2**24:
            candidate = -2**24 + 1
        unseen[:, index] = candidate
    probes = [probe[:1], probe[:min(17, len(probe))], probe, probe[::-1].copy(), unknown, unseen]
    if valid is not None:
        probes.append(np.ascontiguousarray(valid[:min(257, len(valid))]))
    # Save predictions before replacing any library inference operation.
    native_references = [estimator.predict_proba(sample) for sample in probes]
    portable_operations = _prepare_portable_inference(graph)
    # Verify that arbitrary novel values and the explicit sentinel encode equally.
    for key, tensor in interface.splitter(torch.from_numpy(unknown)).items():
        if key == "x_cat":
            assert torch.count_nonzero(tensor).item() == 0
    with torch.inference_mode():
        traced = torch.jit.trace(graph, torch.from_numpy(probe[:min(7, len(probe))]),
                                 check_inputs=[(torch.from_numpy(p),) for p in probes])
    for node in traced.inlined_graph.nodes():
        if node.kind() == "prim::Constant":
            tensor = node.output().toIValue()
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 1:
                raise RuntimeError("Categorical export contains an immovable tensor constant")
    max_error = 0.0
    for sample, reference in zip(probes, native_references):
        with torch.inference_mode():
            exported = traced(torch.from_numpy(sample)).numpy()
        np.testing.assert_allclose(exported, reference, rtol=1e-5, atol=2e-6)
        max_error = max(max_error, float(np.max(np.abs(reference - exported))))
    with torch.inference_mode():
        np.testing.assert_array_equal(traced(torch.from_numpy(unknown)).numpy(),
                                      traced(torch.from_numpy(unseen)).numpy())
    # Cache references before moving shared traced/estimator modules to CUDA.
    reference = native_references[2]
    metadata = {
        "schema_version": 1, "model_type": "realmlp_categorical", "n_features": values.shape[1],
        "classes": [0, 1], "input_schema": schema, "constructor": params,
        "requested_config": config, "best_epoch": selected, "train_rows": len(values),
        "validation_rows": 0 if valid is None else len(valid),
        "selection": "inner ROC AUC only" if valid is not None else "fixed epochs, all supplied rows",
        "native_export_max_abs_error": max_error, "native_export_probe_rows": [len(p) for p in probes],
        "native_export_tolerance": {"rtol": 1e-5, "atol": 2e-6},
        "portable_inference_operations": portable_operations,
        "exact_training_resume": False, "cross_platform_determinism_guaranteed": False,
        "versions": {name: importlib.metadata.version(name)
                     for name in ["pytabkit", "torch", "pytorch-lightning", "numpy", "scikit-learn"]},
        "code_sha256": _sha256(Path(__file__)),
        "shared_realmlp_code_sha256": _sha256(Path(__file__).with_name("realmlp.py")),
        "elapsed_seconds": time.monotonic() - started,
    }
    result = CategoricalRealMLPClassifier(traced, metadata, str(params["device"]))
    result.save(output_path)
    loaded = load_realmlp_categorical(output_path, device="cpu")
    np.testing.assert_allclose(loaded.predict_proba(probe), reference, rtol=1e-5, atol=2e-6)
    return result, selected


fit_neural = fit_realmlp_categorical
load_neural = load_realmlp_categorical


def _smoke(device: str) -> None:
    torch.set_num_threads(2)
    rng = np.random.default_rng(3201)
    X = rng.normal(size=(900, 9)).astype(np.float32)
    X[:, 4] = rng.integers(0, 3, len(X))
    X[:, 5] = rng.integers(0, 100, len(X))
    X[:, 6] = -1  # Exercise a training column with no known categories.
    X[:, 7] = rng.integers(0, 2, len(X))  # Native signed binary one-hot path.
    X[:, 8] = 4  # Single known category plus unknown exercises cardinality two.
    X[650:700, 5] = 555  # Validation-only category must not enter fitted vocabulary.
    y = (X[:, 0] + X[:, 1] + 0.5 * (X[:, 4] == 2) > 0).astype(np.int64)
    config = {"device": device, "seed": 3201, "threads": 2, "epochs": 2,
              "n_ens": 8 if device == "cuda" else 2,
              "hidden_sizes": [512, 256, 128] if device == "cuda" else [32, 16],
              "batch_size": 256 if device == "cuda" else 64, "eval_batch_size": 37,
              "categorical_indices": [4, 5, 6, 7, 8]}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="realmlp-categorical-smoke-") as tmp:
        root = Path(tmp)
        inner, selected = fit_realmlp_categorical(X[:600], y[:600], X[600:], y[600:], config, root / "inner")
        assert 555 not in inner.metadata["input_schema"]["vocabularies"][1]
        assert inner.metadata["input_schema"]["cardinalities"] == [4, 101, 1, 3, 2]
        fixed, epoch = fit_realmlp_categorical(X[:600], y[:600], None, None,
                                              dict(config, epochs=selected), root / "fixed")
        loaded = load_realmlp_categorical(root / "fixed", device=device)
        cpu = load_realmlp_categorical(root / "fixed", device="cpu")
        reload_error = cpu_error = 0.0
        for rows in [1, 17, 257]:
            sample = X[-rows:]
            actual, restored, cpu_values = fixed.predict_proba(sample), loaded.predict_proba(sample), cpu.predict_proba(sample)
            np.testing.assert_allclose(actual, restored, rtol=1e-5, atol=2e-6)
            np.testing.assert_allclose(actual, cpu_values, rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(actual.sum(axis=1), 1.0, atol=2e-6)
            reload_error = max(reload_error, float(np.max(np.abs(actual - restored))))
            cpu_error = max(cpu_error, float(np.max(np.abs(actual - cpu_values))))
        a = X[:17].copy()
        b = a.copy()
        a[:, config["categorical_indices"]] = -1
        b[:, config["categorical_indices"]] = 555
        np.testing.assert_array_equal(loaded.predict_proba(a), loaded.predict_proba(b))
        assert loaded.predict_proba(X[:0]).shape == (0, 2)
        assert epoch == selected
        print(json.dumps({"status": "passed", "device": device, "n_ens": config["n_ens"],
                          "selected_epoch": selected, "fixed_epoch": epoch,
                          "validation_only_category_excluded": True, "unknown_mapping_verified": True,
                          "reload_max_abs_error": reload_error, "cpu_device_max_abs_error": cpu_error,
                          "native_export_max_abs_error": fixed.metadata["native_export_max_abs_error"],
                          "seconds": time.monotonic() - started,
                          "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Use the fit_realmlp_categorical API, or --smoke for bounded verification")
    _smoke(args.device)
