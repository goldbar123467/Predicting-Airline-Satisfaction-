"""Native-categorical TabM, implementation version 1.0.

Contract: X is finite float32 [rows, features]. Config categorical_indices selects
integer category-code columns, with -1 meaning unknown. All other columns are
already fold-standardized numerical features; the caller persists that transform.
Vocabularies and optional numerical PLE bins are learned from X_train only.
Known category codes map to 1..K; -1 and unseen codes map to 0. TabM's native
one-hot representation exists only inside each forward batch, never as a full
CPU training matrix. Probability columns are [P(y=0), P(y=1)].

Inspected implementation: yandex-research/tabm v0.0.3, commit
a507095893d784c5702059d737ddfbd1299c41dd, TabM.make/forward/_OneHotEncoding.
https://github.com/yandex-research/tabm/blob/a507095893d784c5702059d737ddfbd1299c41dd/tabm.py
Native state dictionaries plus JSON support inference, not exact optimizer resume.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
import tabm
import torch
from torch.nn import functional as F

from neural import (
    NeuralClassifier,
    _atomic_json,
    _device,
    _embedding_metadata,
    _labels,
    _make_model,
    _matrix,
    _positive_int,
    _sha256,
)

IMPLEMENTATION_VERSION = "1.0"


def _indices(values: Any, n_features: int, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(values, (tuple, list)) or (not values and not allow_empty):
        raise ValueError("Feature indices must be a nonempty list of integers")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= n_features for i in values):
        raise ValueError("Feature indices must be integer positions within the input matrix")
    if len(set(values)) != len(values):
        raise ValueError("Feature indices must be unique")
    return list(values)


def _raw_codes(values: np.ndarray) -> np.ndarray:
    if ((values < -1) | (values > 2**24) | (values != np.floor(values))).any():
        raise ValueError("Categorical inputs must be exact float32 integers in {-1, 0, ..., 2**24}")
    return values.astype(np.int64, copy=False)


def _vocabulary_digest(indices: list[int], vocabularies: list[list[int]]) -> str:
    payload = json.dumps([indices, vocabularies], separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class NativeCategoricalClassifier(NeuralClassifier):
    """Reuse atomic native-weight saving; route compact inputs to TabM's two inputs."""

    def __init__(self, model: tabm.TabM, metadata: dict[str, Any], device: torch.device) -> None:
        self.model = model
        self.metadata = metadata
        self.device = device
        self.n_features_in_ = int(metadata["n_features"])
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.categorical_indices = _indices(metadata["categorical_indices"], self.n_features_in_)
        self.numerical_indices = _indices(metadata["numerical_indices"], self.n_features_in_, allow_empty=True)
        if set(self.categorical_indices) & set(self.numerical_indices):
            raise ValueError("Numerical and categorical feature positions overlap")
        vocabularies = metadata["categorical_vocabularies"]
        if len(vocabularies) != len(self.categorical_indices):
            raise ValueError("Categorical vocabulary count does not match feature positions")
        for vocabulary in vocabularies:
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > 2**24 for v in vocabulary):
                raise ValueError("Invalid category vocabulary value")
            if vocabulary != sorted(set(vocabulary)):
                raise ValueError("Category vocabularies must be sorted and unique")
        if _vocabulary_digest(self.categorical_indices, vocabularies) != metadata["vocabulary_sha256"]:
            raise ValueError("Categorical vocabulary hash mismatch")
        if [len(v) + 1 for v in vocabularies] != metadata["constructor"]["cat_cardinalities"]:
            raise ValueError("Categorical cardinalities differ from the saved vocabularies")
        self._vocabularies = [np.asarray(vocabulary, dtype=np.int64) for vocabulary in vocabularies]

    def _arrays(self, matrix: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        numerical = (
            np.ascontiguousarray(matrix[:, self.numerical_indices]) if self.numerical_indices else None
        )
        categorical = np.zeros((len(matrix), len(self.categorical_indices)), dtype=np.int64)
        for j, (index, vocabulary) in enumerate(zip(self.categorical_indices, self._vocabularies, strict=True)):
            raw = _raw_codes(matrix[:, index])
            if not len(vocabulary):
                continue
            position = np.searchsorted(vocabulary, raw)
            possible = np.flatnonzero(position < len(vocabulary))
            known = possible[vocabulary[position[possible]] == raw[possible]]
            categorical[known, j] = position[known] + 1
        return numerical, categorical

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        matrix = _matrix(X, "X", allow_empty=True)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} input features, got {matrix.shape[1]}")
        result = np.empty((len(matrix), 2), dtype=np.float32)
        batch_size = int(self.metadata["training"]["eval_batch_size"])
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(matrix), batch_size):
                stop = min(start + batch_size, len(matrix))
                numerical, categorical = self._arrays(matrix[start:stop])
                x_num = None if numerical is None else torch.as_tensor(numerical, device=self.device)
                x_cat = torch.as_tensor(categorical, device=self.device)
                logits = self.model(x_num, x_cat).squeeze(-1).float()
                positive = logits.sigmoid().mean(dim=1).cpu().numpy()
                result[start:stop, 1] = positive
                result[start:stop, 0] = 1.0 - positive
        if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
            raise FloatingPointError("Native-categorical TabM returned invalid probabilities")
        return result


def load_neural(output_path: Path) -> NativeCategoricalClassifier:
    directory = Path(output_path)
    with (directory / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema_version") != 1 or metadata.get("model_type") != "tabm_native_categorical":
        raise ValueError("Unsupported native-categorical TabM artifact")
    filename = metadata["weights_file"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Invalid native weights filename")
    weights = directory / filename
    if _sha256(weights) != metadata["weights_sha256"]:
        raise ValueError("Native categorical TabM weights hash mismatch")
    requested = metadata["training"]["device"]
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    device = _device(requested)
    model = _make_model(metadata["constructor"], metadata.get("numerical_embeddings"))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    return NativeCategoricalClassifier(model, metadata, device)


def fit_neural(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    config: dict[str, Any],
    output_path: Path,
) -> tuple[NativeCategoricalClassifier, int]:
    """Fit a native-categorical model; return selected model and 1-based epoch.

    Validation is exclusively for inner stopping. Without validation, fit the fixed
    configured epoch count. Outer OOF rows must never appear in these fit arguments.
    The only full-size tensor storage is compact numeric values and integer codes;
    categorical one-hot tensors are allocated by TabM for the current batch only.
    """
    fit_start = time.perf_counter()
    matrix = _matrix(X_train, "X_train")
    labels = _labels(y_train, len(matrix), "y_train")
    if (X_valid is None) != (y_valid is None):
        raise ValueError("X_valid and y_valid must both be supplied or both be None")
    validation = validation_labels = None
    if X_valid is not None and y_valid is not None:
        validation = _matrix(X_valid, "X_valid")
        validation_labels = _labels(y_valid, len(validation), "y_valid")
        if validation.shape[1] != matrix.shape[1]:
            raise ValueError("Train and validation feature counts differ")
    categorical_indices = _indices(config.get("categorical_indices"), matrix.shape[1])
    source_numerical_indices = [i for i in range(matrix.shape[1]) if i not in categorical_indices]
    vocabularies = []
    for index in categorical_indices:
        raw = _raw_codes(matrix[:, index])
        vocabularies.append(np.unique(raw[raw >= 0]).tolist())
        if validation is not None:
            _raw_codes(validation[:, index])
    use_embeddings = config.get("num_embeddings", False)
    if not isinstance(use_embeddings, bool):
        raise TypeError("num_embeddings must be boolean")
    embeddings = None
    numerical_indices = source_numerical_indices
    if use_embeddings:
        if not source_numerical_indices:
            raise ValueError("Numerical PLE requested, but every input column is categorical")
        embeddings = _embedding_metadata(np.ascontiguousarray(matrix[:, source_numerical_indices]), config)
        numerical_indices = [source_numerical_indices[i] for i in embeddings["input_indices"]]

    seed = int(config["seed"])
    epochs = _positive_int(config, "epochs", 40)
    patience = _positive_int(config, "patience", 6)
    batch_size = _positive_int(config, "batch_size", 512)
    eval_batch_size = _positive_int(config, "eval_batch_size", 512)
    device = _device(str(config.get("device", "cuda")))
    constructor = {
        "n_num_features": len(numerical_indices),
        "cat_cardinalities": [len(v) + 1 for v in vocabularies],
        "d_out": 1,
        "arch_type": str(config.get("arch_type", "tabm-mini")),
        "k": _positive_int(config, "k", 16),
        "d_block": _positive_int(config, "d_block", 128),
        "n_blocks": _positive_int(config, "n_blocks", 2),
        "dropout": float(config.get("dropout", 0.1)),
    }
    if constructor["arch_type"] not in {"tabm", "tabm-mini"}:
        raise ValueError("Supported architectures are tabm and tabm-mini")
    if not 0 <= constructor["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    if "learning_rate" in config and "lr" in config and config["learning_rate"] != config["lr"]:
        raise ValueError("Conflicting learning_rate and lr aliases")
    learning_rate = float(config.get("learning_rate", config.get("lr", 0.002)))
    weight_decay = float(config.get("weight_decay", 0.0003))
    grad_clip = float(config.get("grad_clip", 1.0))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("grad_clip must be finite and positive")
    amp = config.get("amp", False)
    if not isinstance(amp, bool):
        raise TypeError("amp must be boolean")
    if amp and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 AMP requires a compatible CUDA device")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = _make_model(constructor, embeddings).to(device)
    categorical_width = sum(constructor["cat_cardinalities"])
    numerical_width = len(numerical_indices) * (embeddings["d_embedding"] if embeddings else 1)
    flat_width = categorical_width + numerical_width
    parameter_count = sum(p.numel() for p in model.parameters())
    metadata = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION_VERSION,
        "model_type": "tabm_native_categorical",
        "n_features": int(matrix.shape[1]),
        "classes": [0, 1],
        "constructor": constructor,
        "categorical_indices": categorical_indices,
        "source_numerical_indices": source_numerical_indices,
        "numerical_indices": numerical_indices,
        "categorical_vocabularies": vocabularies,
        "vocabulary_sha256": _vocabulary_digest(categorical_indices, vocabularies),
        "category_contract": "exact float32 integer codes; train-known values -> 1..K; -1/unseen -> 0",
        "category_fit_scope": "X_train only; validation and inference never expand vocabulary",
        "preprocessing": "numerical standardization and source categorical coding supplied/persisted by caller",
        "training": {
            "seed": seed, "device": str(device), "epochs": epochs, "patience": patience,
            "batch_size": batch_size, "eval_batch_size": eval_batch_size,
            "learning_rate": learning_rate, "weight_decay": weight_decay, "grad_clip": grad_clip,
            "optimizer": "AdamW", "optimizer_betas": [0.9, 0.999], "optimizer_eps": 1e-8,
            "amp": amp, "amp_dtype": "bfloat16" if amp else None,
            "loss": "binary_cross_entropy_with_logits_per_member",
            "selection_metric": "roc_auc" if validation is not None else None,
            "train_rows": len(matrix), "validation_rows": len(validation) if validation is not None else 0,
            "exact_training_resume": False,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cross_platform_determinism_guaranteed": False,
        },
        "memory_estimate": {
            "categorical_onehot_width": categorical_width,
            "flat_network_input_width": flat_width,
            "parameters": parameter_count,
            "fp32_model_gradient_adam_bytes": parameter_count * 16,
            "one_expanded_training_input_bytes": batch_size * constructor["k"] * flat_width * 4,
            "one_expanded_evaluation_input_bytes": eval_batch_size * constructor["k"] * flat_width * 4,
            "compact_training_tensor_bytes": len(matrix) * (len(numerical_indices) * 4 + len(categorical_indices) * 8 + 4),
            "is_peak_estimate": False,
            "note": "Additional gradients, one-hot int64 temporaries, allocator caches and workspaces are not included",
        },
        "requested_config": json.loads(json.dumps(config)),
        "versions": {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "tabm": tabm.__version__, "numpy": np.__version__},
        "helper_source_sha256": _sha256(Path(__file__).with_name("neural.py")),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    if embeddings is not None:
        metadata["numerical_embeddings"] = embeddings
    wrapper = NativeCategoricalClassifier(model, metadata, device)
    numerical, categorical = wrapper._arrays(matrix)
    train_num = None if numerical is None else torch.as_tensor(numerical, device=device)
    train_cat = torch.as_tensor(categorical, device=device)
    train_y = torch.as_tensor(labels, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_auc, best_epoch, waiting = -math.inf, 0, 0
    history = []
    print(
        f"Native TabM v{IMPLEMENTATION_VERSION} rows={len(matrix)} numeric={len(numerical_indices)} "
        f"cat_cardinalities={constructor['cat_cardinalities']} k={constructor['k']} "
        f"width={constructor['d_block']} batch={batch_size} "
        f"expanded_input_MiB={metadata['memory_estimate']['one_expanded_training_input_bytes'] / 2**20:.1f}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        permutation = torch.randperm(len(matrix), generator=generator, device=device)
        for indices in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp else nullcontext()
            with context:
                logits = model(None if train_num is None else train_num[indices], train_cat[indices]).squeeze(-1)
                target = train_y[indices, None].expand_as(logits)
                loss = F.binary_cross_entropy_with_logits(logits.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.detach() * len(indices)
        train_loss = float((loss_sum / len(matrix)).cpu())
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        valid_auc = None
        if validation is not None and validation_labels is not None:
            valid_auc = float(roc_auc_score(validation_labels, wrapper.predict_proba(validation)[:, 1]))
        improved = valid_auc is None or valid_auc > best_auc
        if improved:
            best_epoch, waiting = epoch, 0
            if valid_auc is not None:
                best_auc = valid_auc
        else:
            waiting += 1
        event = {
            "epoch": epoch, "train_loss": train_loss, "valid_auc": valid_auc,
            "seconds": time.perf_counter() - epoch_start,
            "elapsed_seconds": time.perf_counter() - fit_start, "improved": improved,
        }
        history.append(event)
        if improved:
            wrapper.metadata.update(best_epoch=best_epoch, best_valid_auc=valid_auc, history=list(history), fit_complete=False)
            wrapper.save(output_path)
        _atomic_json(Path(output_path) / "progress.json", event)
        print(
            f"Native TabM epoch={epoch}/{epochs} loss={train_loss:.6f} "
            f"valid_auc={valid_auc if valid_auc is not None else 'fixed-refit'} "
            f"seconds={event['seconds']:.2f} best_epoch={best_epoch}",
            flush=True,
        )
        if validation is not None and waiting >= patience:
            break
    with (Path(output_path) / "metadata.json").open(encoding="utf-8") as stream:
        selected = json.load(stream)
    state_path = Path(output_path) / selected["weights_file"]
    if _sha256(state_path) != selected["weights_sha256"]:
        raise ValueError("Selected checkpoint hash mismatch")
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    selected.update(history=history, epochs_completed=len(history), elapsed_seconds=time.perf_counter() - fit_start, fit_complete=True)
    wrapper.metadata = selected
    _atomic_json(Path(output_path) / "metadata.json", selected)
    return wrapper, best_epoch


def _smoke(device: str, num_embeddings: bool) -> None:
    rng = np.random.default_rng(731)
    train_rows, valid_rows = 6203, 79
    numeric = rng.normal(size=(train_rows + valid_rows, 3)).astype(np.float32)
    numeric[:, 2] = 1.0  # A constant numeric feature tests PLE column exclusion.
    route = (np.arange(train_rows + valid_rows) % 6000).astype(np.float32)
    rating = (np.arange(train_rows + valid_rows) % 6).astype(np.float32)
    matrix = np.column_stack([numeric, route, rating, np.full(len(route), -1, dtype=np.float32)])
    labels = (numeric[:, 0] + 0.2 * (rating % 2) > 0).astype(np.int8)
    matrix[train_rows, 3] = 9000  # Validation-only positive category must be unknown.
    matrix[train_rows + 1, 3] = -1
    matrix[train_rows, 1] = 100.0  # Must not affect optional numerical bin boundaries.
    config = {
        "seed": 411, "device": device, "epochs": 2, "patience": 2,
        "categorical_indices": [3, 4, 5], "batch_size": 257, "eval_batch_size": 29,
        "k": 2, "d_block": 16, "n_blocks": 2, "dropout": 0.0,
        "num_embeddings": num_embeddings, "n_bins": 8, "d_embedding": 4,
    }
    with tempfile.TemporaryDirectory(prefix="native-tabm-smoke-") as temporary:
        directory = Path(temporary)
        fitted, epoch = fit_neural(matrix[:train_rows], labels[:train_rows], matrix[train_rows:], labels[train_rows:], config, directory / "fit")
        before = fitted.predict_proba(matrix[train_rows:])
        restored = load_neural(directory / "fit")
        after = restored.predict_proba(matrix[train_rows:])
        np.testing.assert_allclose(before, after, rtol=0, atol=1e-6)
        np.testing.assert_allclose(after.sum(axis=1), 1.0, rtol=0, atol=1e-7)
        assert after.shape == (valid_rows, 2) and 1 <= epoch <= 2
        assert restored.metadata["constructor"]["cat_cardinalities"] == [6001, 7, 1]
        assert 9000 not in restored.metadata["categorical_vocabularies"][0]
        _, codes = restored._arrays(matrix[train_rows:train_rows + 2])
        np.testing.assert_array_equal(codes[:, 0], [0, 0])
        assert codes.dtype == np.int64 and codes.shape == (2, 3)
        _, known_codes = restored._arrays(matrix[:1])
        assert known_codes[0, 0] == 1  # Raw code zero is a known category, not unknown.
        assert restored.predict_proba(matrix[:0]).shape == (0, 2)
        np.testing.assert_allclose(restored.predict_proba(matrix[train_rows:train_rows + 1]), after[:1], rtol=0, atol=2e-6)
        if num_embeddings:
            assert restored.metadata["numerical_indices"] == [0, 1]
            assert restored.metadata["numerical_embeddings"]["bins"][1][-1] == float(matrix[:train_rows, 1].max())
        for invalid in [-2.0, 1.5]:
            bad = matrix[:1].copy()
            bad[0, 3] = invalid
            try:
                restored.predict_proba(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid raw category code was accepted")
        # Fixed-epoch training checks the second fitting mode without rereading validation.
        fixed, fixed_epoch = fit_neural(matrix[:31], labels[:31], None, None, {**config, "epochs": 1}, directory / "fixed")
        assert fixed_epoch == 1 and fixed.metadata["training"]["validation_rows"] == 0
        print(
            f"SMOKE PASS device={device} PLE={num_embeddings}: 6000-category fit, "
            "unknown/known mapping, train-only vocab/bins, native reload, partial/single/empty batches, fixed-refit, invalid-code rejection",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-embeddings", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Standalone execution requires --smoke")
    torch.set_num_threads(2)
    _smoke(args.device, args.num_embeddings)
