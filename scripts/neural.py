"""Bounded TabM binary classification with native, atomic inference artifacts.

Inputs are finite float32 matrices [row, feature], already transformed exclusively
from the appropriate training partition. The caller owns and persists preprocessing.
Labels are 0/1, and predict_proba columns are [P(y=0), P(y=1)]. Checkpoints support
inference and completed-epoch recovery, not exact optimizer-state continuation.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from uuid import uuid4
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
import tabm
import torch
from torch import Tensor
from torch.nn import functional as F


def _matrix(value: np.ndarray, name: str, *, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [rows, positive feature count]")
    if array.dtype != np.float32:
        raise TypeError(f"{name} must be float32, received {array.dtype}")
    if not allow_empty and not len(array):
        raise ValueError(f"{name} must contain at least one row")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity; preprocess inside folds")
    return np.ascontiguousarray(array)


def _labels(value: np.ndarray, rows: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != rows:
        raise ValueError(f"{name} must have shape [{rows}]")
    if not np.isin(array, [0, 1]).all() or len(np.unique(array)) != 2:
        raise ValueError(f"{name} must contain both binary classes, encoded 0 and 1")
    return np.asarray(array, dtype=np.float32)


def _positive_int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{key} must be a positive integer")
    return int(value)


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported for this local run")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but the installed PyTorch cannot access CUDA")
    return device


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embedding_metadata(matrix: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Learn optional quantile bins solely from the supplied training matrix.

    All nonconstant columns are supported, including one-hot indicators. A binary
    column receives a linear embedding with one interval. Train-constant columns
    are dropped because the official quantile-bin API cannot fit constant columns.
    """
    import rtdl_num_embeddings

    if len(matrix) < 3:
        raise ValueError("Piecewise-linear embeddings require at least three training rows")
    requested_bins = _positive_int(config, "n_bins", 32)
    if requested_bins < 2:
        raise ValueError("n_bins must be at least two")
    d_embedding = _positive_int(config, "d_embedding", 16)
    minima = matrix.min(axis=0)
    maxima = matrix.max(axis=0)
    indices = np.flatnonzero(minima < maxima).tolist()
    if not indices:
        raise ValueError("Piecewise-linear embeddings need a nonconstant training feature")
    n_bins = min(requested_bins, len(matrix) - 1)
    values = torch.from_numpy(np.ascontiguousarray(matrix[:, indices]))
    # The library warns that binary columns have only one interval. This is an
    # intentional supported case, recorded explicitly in the artifact metadata.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"The .* feature has just two bin edges.*", module="rtdl_num_embeddings"
        )
        bins = rtdl_num_embeddings.compute_bins(values, n_bins=n_bins)
    return {
        "type": "piecewise_linear",
        "version": "B",
        "activation": False,
        "d_embedding": d_embedding,
        "n_bins_requested": requested_bins,
        "n_bins_effective": n_bins,
        "input_indices": indices,
        "dropped_constant_indices": np.flatnonzero(minima == maxima).tolist(),
        "single_interval_indices": [indices[i] for i, edges in enumerate(bins) if len(edges) == 2],
        "bins": [edges.tolist() for edges in bins],
        "bin_dtype": "float32",
        "fit_scope": "X_train only; no validation rows or labels",
        "rtdl_num_embeddings_version": rtdl_num_embeddings.__version__,
    }


def _make_model(
    constructor: dict[str, Any], embeddings: dict[str, Any] | None = None
) -> tabm.TabM:
    if embeddings is None:
        return tabm.TabM.make(**constructor)
    import rtdl_num_embeddings

    if embeddings.get("type") != "piecewise_linear" or embeddings.get("version") != "B":
        raise ValueError("Unsupported numerical embedding artifact")
    bins = [torch.tensor(edges, dtype=torch.float32) for edges in embeddings["bins"]]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"The .* feature has just two bin edges.*", module="rtdl_num_embeddings"
        )
        module = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins,
            d_embedding=int(embeddings["d_embedding"]),
            activation=bool(embeddings["activation"]),
            version="B",
        )
    return tabm.TabM.make(**constructor, num_embeddings=module)


class NeuralClassifier:
    """TabM inference wrapper; preprocessing is owned by the outer pipeline."""

    def __init__(
        self,
        model: tabm.TabM,
        metadata: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.device = device
        self.n_features_in_ = int(metadata["n_features"])
        self.classes_ = np.array([0, 1], dtype=np.int8)
        embeddings = metadata.get("numerical_embeddings")
        self.input_indices = None if embeddings is None else embeddings["input_indices"]
        if self.input_indices is not None:
            if (
                not self.input_indices
                or len(set(self.input_indices)) != len(self.input_indices)
                or any(not isinstance(i, int) or i < 0 or i >= self.n_features_in_ for i in self.input_indices)
            ):
                raise ValueError("Invalid numerical embedding feature indices")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        matrix = _matrix(X, "X", allow_empty=True)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, received {matrix.shape[1]}"
            )
        result = np.empty((len(matrix), 2), dtype=np.float32)
        self.model.eval()
        batch_size = int(self.metadata["training"]["eval_batch_size"])
        with torch.inference_mode():
            for start in range(0, len(matrix), batch_size):
                end = min(start + batch_size, len(matrix))
                values = matrix[start:end]
                if self.input_indices is not None:
                    values = np.ascontiguousarray(values[:, self.input_indices])
                batch = torch.as_tensor(values, device=self.device)
                # Model emits [row, ensemble member, one binary logit].
                logits = self.model(batch).squeeze(-1).float()
                positive = logits.sigmoid().mean(dim=1).cpu().numpy()
                result[start:end, 1] = positive
                result[start:end, 0] = 1.0 - positive
        if not np.isfinite(result).all():
            raise FloatingPointError("TabM inference generated non-finite probabilities")
        return result

    def save(self, output_path: Path) -> None:
        """Publish weights first, then atomically point metadata at that version.

        Previous complete snapshots remain readable after interruption. Only the
        metadata-referenced snapshot is active. Old weight files are deliberately
        retained as recovery artifacts until the outer pipeline releases the run.
        """
        directory = Path(output_path)
        directory.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        weights_file = f"weights-{token}.pt"
        temporary = directory / f".{weights_file}.tmp"
        final_weights = directory / weights_file
        state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in self.model.state_dict().items()
        }
        try:
            with temporary.open("wb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, final_weights)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = dict(self.metadata)
        manifest["weights_file"] = weights_file
        manifest["weights_sha256"] = _sha256(final_weights)
        _atomic_json(directory / "metadata.json", manifest)
        self.metadata = manifest


def load_neural(output_path: Path) -> NeuralClassifier:
    """Load a local native state_dict using PyTorch's restricted weights loader."""
    directory = Path(output_path)
    with (directory / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema_version") != 1 or metadata.get("model_type") != "tabm":
        raise ValueError("Unsupported neural artifact schema or model type")
    weights_file = metadata["weights_file"]
    if not isinstance(weights_file, str) or Path(weights_file).name != weights_file:
        raise ValueError("Neural manifest contains an invalid weights filename")
    weights_path = directory / weights_file
    if _sha256(weights_path) != metadata["weights_sha256"]:
        raise ValueError("Neural weights SHA-256 does not match the manifest")
    requested_device = metadata["training"]["device"]
    # A CUDA-trained model remains usable for inference on a CPU-only machine.
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = _device(requested_device)
    model = _make_model(metadata["constructor"], metadata.get("numerical_embeddings"))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return NeuralClassifier(model, metadata, device)


def fit_neural(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    config: dict[str, Any],
    output_path: Path,
) -> tuple[NeuralClassifier, int]:
    """Fit TabM and return the selected model and a one-based epoch count.

    With validation, ROC AUC selects a completed epoch, with patience 3 by default.
    Without validation, all configured epochs run and the final epoch is returned.
    The caller must use a separate inner stopping split to keep outer OOF rows clean.
    Standardization and category encoding are owned by the caller. Optional PLE
    bin boundaries are fitted here using X_train only. No labels outside these
    arguments are loaded or inspected. A failed/interrupted fit can leave a valid inference
    checkpoint but must not be promoted by the caller without a completion record.
    """
    matrix = _matrix(X_train, "X_train")
    labels = _labels(y_train, len(matrix), "y_train")
    if (X_valid is None) != (y_valid is None):
        raise ValueError("X_valid and y_valid must either both be present or both be None")
    validation = None
    validation_labels = None
    if X_valid is not None and y_valid is not None:
        validation = _matrix(X_valid, "X_valid")
        if validation.shape[1] != matrix.shape[1]:
            raise ValueError("Training and validation feature counts differ")
        validation_labels = _labels(y_valid, len(validation), "y_valid")

    seed = int(config["seed"])
    epochs = _positive_int(config, "epochs", 12)
    patience = _positive_int(config, "patience", 3)
    batch_size = _positive_int(config, "batch_size", 1024)
    eval_batch_size = _positive_int(config, "eval_batch_size", 4096)
    device = _device(str(config.get("device", "cuda")))
    constructor = {
        "n_num_features": int(matrix.shape[1]),
        "d_out": 1,
        "arch_type": str(config.get("arch_type", "tabm-mini")),
        "k": _positive_int(config, "k", 16),
        "d_block": _positive_int(config, "d_block", 128),
        "n_blocks": _positive_int(config, "n_blocks", 3),
        "dropout": float(config.get("dropout", 0.1)),
    }
    if constructor["arch_type"] not in {"tabm", "tabm-mini"}:
        raise ValueError("This bounded trainer supports tabm and tabm-mini architectures")
    if not 0.0 <= constructor["dropout"] < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if "learning_rate" in config and "lr" in config and config["learning_rate"] != config["lr"]:
        raise ValueError("learning_rate and lr aliases disagree")
    learning_rate = float(config.get("learning_rate", config.get("lr", 0.002)))
    weight_decay = float(config.get("weight_decay", 0.0003))
    grad_clip = float(config.get("grad_clip", 1.0))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("grad_clip must be finite and positive")
    amp = bool(config.get("amp", False))
    if amp and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("amp=True requires CUDA with bfloat16 support")

    use_embeddings = config.get("num_embeddings", False)
    if not isinstance(use_embeddings, bool):
        raise TypeError("num_embeddings must be a boolean")
    embeddings = _embedding_metadata(matrix, config) if use_embeddings else None
    if embeddings is not None:
        constructor["n_num_features"] = len(embeddings["input_indices"])

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = _make_model(constructor, embeddings).to(device)
    training = {
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1e-8,
        "amp": amp,
        "amp_dtype": "bfloat16" if amp else None,
        "loss": "binary_cross_entropy_with_logits_per_member",
        "selection_metric": "roc_auc" if validation is not None else None,
        "train_rows": len(matrix),
        "validation_rows": len(validation) if validation is not None else 0,
        "exact_training_resume": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cross_platform_determinism_guaranteed": False,
    }
    metadata = {
        "schema_version": 1,
        "model_type": "tabm",
        "n_features": int(matrix.shape[1]),
        "classes": [0, 1],
        "preprocessing": "supplied and persisted by caller; fold-local fit required",
        "constructor": constructor,
        "training": training,
        "requested_config": json.loads(json.dumps(config)),
        "versions": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tabm": tabm.__version__,
            "numpy": np.__version__,
        },
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    if embeddings is not None:
        metadata["numerical_embeddings"] = embeddings
    wrapper = NeuralClassifier(model, metadata, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_values = (
        matrix
        if embeddings is None
        else np.ascontiguousarray(matrix[:, embeddings["input_indices"]])
    )
    train_tensor = torch.as_tensor(train_values, device=device)
    label_tensor = torch.as_tensor(labels, device=device)
    best_auc = -math.inf
    best_epoch = 0
    waiting = 0
    history: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    print(
        f"TabM start rows={len(matrix)} features={matrix.shape[1]} device={device} "
        f"epochs={epochs} k={constructor['k']} width={constructor['d_block']} "
        f"batch={batch_size} amp={amp}",
        flush=True,
    )
    if embeddings is not None:
        print(
            f"TabM PLE columns={len(embeddings['input_indices'])} "
            f"bins={embeddings['n_bins_effective']} embedding={embeddings['d_embedding']} "
            f"single_interval={len(embeddings['single_interval_indices'])} "
            f"constants_dropped={len(embeddings['dropped_constant_indices'])}",
            flush=True,
        )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(matrix), generator=generator, device=device)
        loss_sum = torch.zeros((), device=device)
        seen = 0
        for indices in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp
                else nullcontext()
            )
            with context:
                logits: Tensor = model(train_tensor[indices]).squeeze(-1)
                targets = label_tensor[indices, None].expand_as(logits)
                loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            loss_sum += loss.detach() * len(indices)
            seen += len(indices)
        train_loss = float((loss_sum / seen).cpu())
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")

        valid_auc = None
        if validation is not None and validation_labels is not None:
            valid_auc = float(
                roc_auc_score(validation_labels, wrapper.predict_proba(validation)[:, 1])
            )
        improved = valid_auc is None or valid_auc > best_auc
        if improved:
            best_epoch = epoch
            if valid_auc is not None:
                best_auc = valid_auc
            waiting = 0
        else:
            waiting += 1
        epoch_seconds = time.perf_counter() - epoch_start
        event = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_auc": valid_auc,
            "seconds": epoch_seconds,
            "elapsed_seconds": time.perf_counter() - run_start,
            "improved": improved,
        }
        history.append(event)
        if improved:
            wrapper.metadata.update(
                {
                    "best_epoch": best_epoch,
                    "best_valid_auc": valid_auc,
                    "history": list(history),
                    "fit_complete": False,
                }
            )
            wrapper.save(output_path)
        _atomic_json(Path(output_path) / "progress.json", event)
        print(
            f"TabM epoch={epoch}/{epochs} loss={train_loss:.6f} "
            f"valid_auc={valid_auc if valid_auc is not None else 'fixed-refit'} "
            f"seconds={epoch_seconds:.2f} best_epoch={best_epoch}",
            flush=True,
        )
        if validation is not None and waiting >= patience:
            break

    # Recover the exact native checkpoint that generated the selected predictions.
    # Keep the existing architecture on device to avoid a second full GPU model.
    with (Path(output_path) / "metadata.json").open(encoding="utf-8") as stream:
        selected = json.load(stream)
    state = torch.load(
        Path(output_path) / selected["weights_file"], map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    selected.update(
        {
            "history": history,
            "epochs_completed": len(history),
            "elapsed_seconds": time.perf_counter() - run_start,
            "fit_complete": True,
        }
    )
    wrapper.metadata = selected
    _atomic_json(Path(output_path) / "metadata.json", selected)
    return wrapper, best_epoch


def _smoke(device: str, *, num_embeddings: bool = False) -> None:
    """Execute numerical contracts and artifact roundtrip, not a quality benchmark."""
    rng = np.random.default_rng(31)
    matrix = rng.normal(size=(257, 7)).astype(np.float32)
    labels = (matrix[:, 0] + 0.5 * matrix[:, 1] > 0).astype(np.int8)
    if num_embeddings:
        matrix = np.column_stack(
            [matrix, (matrix[:, 2] > 0).astype(np.float32), np.ones(len(matrix), dtype=np.float32)]
        )
        # A validation-only extreme must never appear in learned train bin edges.
        matrix[193, 6] = 100.0
    config = {
        "seed": 41,
        "device": device,
        "epochs": 2,
        "batch_size": 63,
        "eval_batch_size": 37,
        "d_block": 32,
        "n_blocks": 2,
        "k": 4,
        "dropout": 0.0,
    }
    if num_embeddings:
        config.update(num_embeddings=True, n_bins=8, d_embedding=4)
    with tempfile.TemporaryDirectory(prefix="airline-tabm-smoke-") as temporary:
        directory = Path(temporary)
        trained, epoch = fit_neural(
            matrix[:193], labels[:193], matrix[193:], labels[193:], config, directory / "fit"
        )
        before = trained.predict_proba(matrix[193:])
        restored = load_neural(directory / "fit")
        after = restored.predict_proba(matrix[193:])
        np.testing.assert_allclose(before, after, rtol=0, atol=1e-6)
        np.testing.assert_allclose(after.sum(axis=1), 1.0, rtol=0, atol=1e-7)
        assert after.shape == (64, 2) and np.isfinite(after).all()
        assert np.all((after >= 0) & (after <= 1)) and 1 <= epoch <= config["epochs"]
        assert restored.predict_proba(matrix[:0]).shape == (0, 2)
        if num_embeddings:
            embedding_metadata = restored.metadata["numerical_embeddings"]
            assert embedding_metadata["dropped_constant_indices"] == [8]
            assert 7 in embedding_metadata["single_interval_indices"]
            assert embedding_metadata["bins"][6][-1] == float(matrix[:193, 6].max())
            assert embedding_metadata["bins"][6][-1] < 100
        np.testing.assert_allclose(
            restored.predict_proba(matrix[193:194]), after[:1], rtol=0, atol=2e-6
        )
        # Repetition on one device checks that the seed is actually applied.
        repeated, _ = fit_neural(
            matrix[:193], labels[:193], matrix[193:], labels[193:], config, directory / "repeat"
        )
        np.testing.assert_allclose(
            before, repeated.predict_proba(matrix[193:]), rtol=0, atol=2e-6
        )
        fixed, fixed_epoch = fit_neural(
            matrix[:193], labels[:193], None, None, {**config, "epochs": 1}, directory / "fixed"
        )
        assert fixed_epoch == 1 and fixed.metadata["training"]["validation_rows"] == 0
        assert fixed.metadata["fit_complete"] is True
        try:
            restored.predict_proba(np.full((1, matrix.shape[1]), np.nan, dtype=np.float32))
        except ValueError:
            pass
        else:
            raise AssertionError("Non-finite prediction input was accepted")
        print(
            f"SMOKE PASS device={device} num_embeddings={num_embeddings}: fit/update, repeat seed, roundtrip, "
            "probabilities, empty/single/partial batches, fixed-epoch refit, NaN rejection",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run a bounded synthetic smoke test")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-embeddings", action="store_true", help="verify the optional PLE branch")
    arguments = parser.parse_args()
    if not arguments.smoke:
        parser.error("Use --smoke to run the standalone verification entry point")
    torch.set_num_threads(2)
    _smoke(arguments.device, num_embeddings=arguments.num_embeddings)
