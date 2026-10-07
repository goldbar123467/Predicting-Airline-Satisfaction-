"""Bounded generated-data CPU proof for mixed eight-member passive capture.

This is not the production fitter. It uses its schema/inference helpers and
the pinned library without changing them, with a real passive monitor.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pytorch_lightning as pl
import torch
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.alg_interfaces.base import InterfaceResources, SplitIdxs
from pytabkit.models.data.data import DictDataset, TensorInfo
from pytabkit.models.training.lightning_callbacks import ModelCheckpointCallback
from pytabkit.models.training.lightning_modules import TabNNModule
from pytabkit.models.training.metrics import Metrics
from threadpoolctl import threadpool_limits

from realmlp_categorical import _fit_schema, _InputSplit, _MixedGraph, _prepare_portable_inference
from research_epoch_capture_v1 import fingerprint, layer_identity, live_signature
from research_epoch_control_v1 import max_state_difference
from research_epoch_resume_v1 import optimizer_state_max_difference, sha256, tensor_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_pass_v1/epochs/mixed_capture"
SEED, ROWS, FIT_ROWS, MEMBERS, HORIZON = 23197, 384, 288, 8, 16
TFMS = ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"]


def check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("Mixed capture exceeded its 300-second bound")


def tensor_attributes(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Registered tensors and direct/container tensor attributes on all layers.

    Arbitrary non-module objects are not recursively traversed. Fitted pipeline
    BiasLayer/ScaleLayer tensors are explicitly inspected separately below.
    """
    found = {}

    def walk(value, path: str) -> None:
        if torch.is_tensor(value):
            found[path] = value
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}.{index}")

    for prefix, layer in module.named_modules():
        for name, value in vars(layer).items():
            if name != "_modules":
                walk(value, f"{prefix}.{name}")
    return found


def full_signature(module: TabNNModule, schema: dict) -> dict:
    return {**live_signature(module),
            "all_network_tensor_attributes": fingerprint(tensor_attributes(module.model)),
            "all_static_tensor_attributes": fingerprint(tensor_attributes(module.creator.static_model)),
            "schema": fingerprint(schema)}


def preprocessing_inventory(module: TabNNModule) -> list[dict]:
    result = []
    # PBLD's parallel factory keeps fitted transforms inside the vectorized
    # network rather than necessarily in creator.static_model. Inspect both.
    for owner, root in (("static", module.creator.static_model), ("network", module.model)):
        for path, layer in root.named_modules():
            scope = str(layer.context.scope) if hasattr(layer, "context") else ""
            if type(layer).__name__ not in ("BiasLayer", "ScaleLayer") or "/tfms" not in scope:
                continue
            name = "bias" if type(layer).__name__ == "BiasLayer" else "scale"
            value = getattr(layer, name)
            if not torch.is_tensor(value) or value.requires_grad:
                raise AssertionError("Expected fitted nontrainable preprocessing tensor")
            result.append({"owner": owner, "path": path, "scope": scope,
                           "layer": type(layer).__name__, "attribute": name,
                           "shape": list(value.shape), "registered": name in layer.state_dict(),
                           "tensor_sha256": fingerprint(value),
                           "nonidentity": bool(torch.any(value != (0 if name == "bias" else 1))),
                           "minimum": float(value.min()), "maximum": float(value.max())})
    if not any(x["layer"] == "BiasLayer" and x["nonidentity"] for x in result):
        raise AssertionError("Missing learned nonidentity median preprocessing")
    if not any(x["layer"] == "ScaleLayer" and x["nonidentity"] for x in result):
        raise AssertionError("Missing learned nonidentity scale preprocessing")
    return result


@contextmanager
def passive_context(module: TabNNModule):
    modes = [(layer, layer.training) for owner in (module.model, module.creator.static_model)
             for layer in owner.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            module.model.eval()
            module.creator.static_model.eval()
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        for layer, mode in modes:
            layer.training = mode


def generated_data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(81271)
    scales = torch.tensor([.4, 2., 4., 1., 3., 5.])
    offsets = torch.tensor([2., -1., .5, 3., -4., 1.])
    continuous = torch.randn(ROWS, 6, generator=generator) * scales + offsets
    categorical = torch.stack((torch.arange(ROWS) % 4, torch.arange(ROWS) % 24), dim=1).float()
    categorical[FIT_ROWS:FIT_ROWS + 12] = torch.tensor([999., 777.])
    raw = torch.cat((continuous, categorical), dim=1).contiguous()
    labels = (continuous[:, 0] - 2 + .2 * (continuous[:, 1] + 1)
              + .1 * (categorical[:, 0] == 2) + .2 * torch.randn(ROWS, generator=generator) > 0).long()
    assert raw.shape == (384, 8) and raw.dtype == torch.float32
    assert labels.dtype == torch.int64 and all(torch.unique(y).tolist() == [0, 1]
                                              for y in (labels[:FIT_ROWS], labels[FIT_ROWS:]))
    return raw, labels


def converted_dataset(raw: torch.Tensor, schema: dict, labels: torch.Tensor | None = None) -> DictDataset:
    values = _InputSplit(schema)(raw)
    infos = {"x_cont": TensorInfo(feat_shape=[6]), "x_cat": TensorInfo(cat_sizes=schema["cardinalities"])}
    if labels is not None:
        values["y"] = labels.reshape(-1, 1)
        infos["y"] = TensorInfo(cat_sizes=[2])
    return DictDataset(values, infos, device="cpu")


def native_probabilities(module: TabNNModule, raw: torch.Tensor, schema: dict) -> torch.Tensor:
    """Use installed prediction loader/step, including its ensemble postprocess."""
    dataset = converted_dataset(raw, schema)
    with passive_context(module):
        outputs = [module.predict_step(batch, index)
                   for index, batch in enumerate(module.get_predict_dataloader(dataset))]
        result = torch.cat(outputs, dim=-2).softmax(-1)[0].detach().clone()
    if result.shape != (len(raw), 2) or not torch.isfinite(result).all():
        raise AssertionError("Invalid [row, binary class] probability output")
    if not torch.allclose(result.sum(-1), torch.ones(len(raw)), atol=1e-6, rtol=0):
        raise AssertionError("Probabilities do not sum to one")
    return result


class MonitoredModule(TabNNModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.monitor_trace = []

    def on_validation_epoch_end(self) -> None:
        prediction = self._postprocess_ens_pred(torch.cat(self.val_preds, dim=-2))
        target = self.val_dl.val_y[::MEMBERS]
        if prediction.shape != (1, 96, 2) or target.shape != (1, 96, 1):
            raise AssertionError("Expected real 96-row monitor output")
        error = float(Metrics.apply(prediction[0], target[0], "1-auc_ovr"))
        self.monitor_trace.append({"epoch": int(self.trainer.current_epoch + 1),
                                   "prediction_sha256": fingerprint(prediction),
                                   "actual_monitor_error": error})
        super().on_validation_epoch_end()


def compile_module(raw: torch.Tensor, labels: torch.Tensor) -> tuple[MonitoredModule, dict]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    schema = _fit_schema(raw[:FIT_ROWS].numpy(), [6, 7])
    if schema["cardinalities"] != [5, 25] or 999 in schema["vocabularies"][0] or 777 in schema["vocabularies"][1]:
        raise AssertionError("Schema learned from monitor rows or wrong branch cardinality")
    config = RealMLP_TD_Classifier(
        device="cpu", n_threads=1, verbosity=0, n_epochs=HORIZON, n_ens=MEMBERS,
        hidden_sizes=[16, 8], batch_size=32, predict_batch_size=37,
        lr=.053, wd=.015, sq_mom=.988, lr_sched="flat_anneal", wd_sched="cos_log_15",
        p_drop=.05, p_drop_sched="invsqrtp1e-3", first_layer_lr_factor=.25,
        embedding_size=5, max_one_hot_cat_size=18, act="silu", plr_hidden_1=16,
        plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=.1151, plr_sigma=2.33,
        ls_eps=0., add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
        tfms=TFMS, n_cv=1, n_refit=0, use_early_stopping=False,
        use_best_mean_epoch_for_cv=True, val_metric_name="1-auc_ovr", ens_av_before_softmax=False,
    ).get_config()
    config["use_best_epoch"] = False
    module = MonitoredModule(**config)
    split = SplitIdxs(torch.arange(FIT_ROWS).reshape(1, -1), torch.arange(FIT_ROWS, ROWS).reshape(1, -1),
                      None, split_seed=SEED, sub_split_seeds=[SEED], split_id=0)
    module.compile_model(converted_dataset(raw, schema, labels), [split],
                         InterfaceResources(n_threads=1, gpu_devices=[]))
    if module.val_dl is None or any(v.device.type != "cpu" for v in tensor_attributes(module.model).values()):
        raise AssertionError("Expected CPU model and real monitor loader")
    return module, schema


class MixedCapture(pl.Callback):
    def __init__(self, raw: torch.Tensor, labels: torch.Tensor, schema: dict,
                 destination: Path | None, deadline: float):
        self.raw, self.labels, self.schema = raw, labels, schema
        self.destination, self.deadline = destination, deadline
        self.epoch_states, self.batch_trace, self.probes, self.observation_checks = [], [], [], []
        self.validation_batches = 0
        self.capture = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        check_deadline(self.deadline)
        self.batch_trace.append({"step": int(trainer.global_step), "batch": fingerprint(batch),
                                 "torch_rng": fingerprint(torch.get_rng_state()),
                                 "schedule": fingerprint(pl_module.hp_manager.get_hyper_sched_values()),
                                 "progress": fingerprint(vars(pl_module.progress))})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_batches += 1

    def export_prefix(self, module: MonitoredModule) -> None:
        graph = _MixedGraph(_InputSplit(copy.deepcopy(self.schema)),
                            copy.deepcopy(module.creator.static_model), copy.deepcopy(module.model), MEMBERS).eval()
        live_tensors = list(tensor_attributes(module.model).values()) + list(tensor_attributes(module.creator.static_model).values())
        clone_tensors = list(tensor_attributes(graph).values())
        live_storage = {v.untyped_storage().data_ptr() for v in live_tensors if v.numel()}
        clone_storage = {v.untyped_storage().data_ptr() for v in clone_tensors if v.numel()}
        if not live_storage.isdisjoint(clone_storage):
            raise AssertionError("Clone aliases live registered/direct tensor storage")
        unknown, unseen = self.raw[:7].clone(), self.raw[:7].clone()
        unknown[:, 6:] = -1
        unseen[:, 6:] = 12345
        splitter = graph.splitter
        if torch.count_nonzero(splitter(unknown)["x_cat"]) or torch.count_nonzero(splitter(unseen)["x_cat"]):
            raise AssertionError("Unknown/unseen category mapping failed")
        samples = {"one": self.raw[:1], "seven": self.raw[:7], "partial": self.raw[:37],
                   "monitor": self.raw[FIT_ROWS:], "all": self.raw, "reverse": self.raw.flip(0),
                   "unknown": unknown, "unseen": unseen}
        references = {name: native_probabilities(module, values, self.schema) for name, values in samples.items()}
        native = {name: graph(values).detach().clone() for name, values in samples.items()}
        rewrites = _prepare_portable_inference(graph)
        portable = {name: graph(values).detach().clone() for name, values in samples.items()}
        check_deadline(self.deadline)
        traced = torch.jit.trace(graph, self.raw[:7], check_inputs=[(x,) for x in samples.values()], check_tolerance=1e-5)
        path = self.destination / "epoch4_graph.pt"
        torch.jit.save(traced, str(path))
        loaded = torch.jit.load(str(path), map_location="cpu").eval()
        parity = {}
        for name, values in samples.items():
            restored = loaded(values)
            for kind, actual in (("native_graph", native[name]), ("portable", portable[name]), ("reload", restored)):
                torch.testing.assert_close(actual, references[name], atol=2e-6, rtol=1e-5)
                parity[f"{name}_{kind}"] = float((actual - references[name]).abs().max())
        unknown_equal = torch.equal(loaded(unknown), loaded(unseen))
        reverse_error = float((loaded(self.raw.flip(0)).flip(0) - loaded(self.raw)).abs().max())
        chunked = torch.cat([loaded(chunk) for chunk in self.raw.split(37)])
        chunk_error = float((chunked - loaded(self.raw)).abs().max())
        if not unknown_equal:
            raise AssertionError("Sentinel and unseen output parity failed")
        torch.testing.assert_close(chunked, loaded(self.raw), atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(loaded(self.raw.flip(0)).flip(0), loaded(self.raw), atol=2e-6, rtol=1e-5)
        (self.destination / "schema.json").write_text(json.dumps(self.schema, indent=2) + "\n")
        self.capture = {"path": str(path.relative_to(ROOT)), "artifact_sha256": sha256(path),
                        "schema_sha256": sha256(self.destination / "schema.json"),
                        "all_clone_tensor_storage_disjoint": True, "portable_rewrite_counts": rewrites,
                        "native_reference": "Installed get_predict_dataloader and predict_step followed by final softmax",
                        "maximum_absolute_parity_differences": parity,
                        "unknown_equals_unseen_exactly": unknown_equal,
                        "reverse_layout_max_abs_difference": reverse_error,
                        "split_batch_max_abs_difference": chunk_error,
                        "prefix_probability": loaded(self.raw).detach().clone(),
                        "cloned_static_tensor_hash": fingerprint(tensor_attributes(graph.static))}
        if min(rewrites["onehot"], rewrites["embedding"]) < 1:
            raise AssertionError("Both categorical branches must be exercised")

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_states.append(tensor_state(pl_module))
        if self.destination is None:
            return
        check_deadline(self.deadline)
        before = full_signature(pl_module, self.schema)
        with passive_context(pl_module):
            prediction = native_probabilities(pl_module, self.raw[:37], self.schema)
            loss = -prediction[torch.arange(37), self.labels[:37]].clamp_min(1e-30).log().mean()
            self.probes.append({"epoch": int(trainer.current_epoch + 1),
                                "fixed_fit_row_probability_sha256": fingerprint(prediction),
                                "eval_mean_bce": float(loss)})
            if trainer.current_epoch + 1 == 4:
                self.export_prefix(pl_module)
        after = full_signature(pl_module, self.schema)
        checks = {key: before[key] == after[key] for key in before}
        if not all(checks.values()):
            raise AssertionError(f"Observation changed live state: {checks}")
        self.observation_checks.append(checks)


def run_arm(raw: torch.Tensor, labels: torch.Tensor, destination: Path | None, deadline: float) -> dict:
    check_deadline(deadline)
    module, schema = compile_module(raw, labels)
    inventory = preprocessing_inventory(module)
    callbacks = module.create_callbacks()
    if any(isinstance(callback, ModelCheckpointCallback) for callback in callbacks):
        raise AssertionError("Passive monitor must not install restoration callback")
    observer = MixedCapture(raw, labels, schema, destination, deadline)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=HORIZON,
                         callbacks=[*callbacks, observer], logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0, deterministic=True, log_every_n_steps=1)
    trainer.fit(module, train_dataloaders=module.train_dl, val_dataloaders=module.val_dl)
    final_rng = fingerprint((torch.get_rng_state(), np.random.get_state(), random.getstate()))
    final_state = tensor_state(module)
    if module.fit_params != [{"stop_epoch": {"1-auc_ovr": 16}}] or trainer.global_step != 144:
        raise AssertionError("Expected literal H16 endpoint with 144 updates")
    if len(observer.epoch_states) != 16 or max_state_difference(final_state, observer.epoch_states[-1]) != 0:
        raise AssertionError("Final weights were not the literal epoch16 state")
    if preprocessing_inventory(module) != inventory:
        raise AssertionError("Fitted nontrainable preprocessing changed during optimization")
    return {"state": final_state, "probabilities": native_probabilities(module, raw, schema),
            "optimizer": copy.deepcopy(trainer.optimizers[0].opt.state_dict()),
            "static_tensor_hash": fingerprint(tensor_attributes(module.creator.static_model)),
            "inventory": inventory, "observer": observer, "monitor": module.monitor_trace,
            "rng": final_rng, "best_monitor_epoch": module.best_mean_val_epochs["1-auc_ovr"][0]}


def run_checks() -> dict:
    registry_path = OUT / "registry.json"
    registry = json.loads(registry_path.read_text())
    if (OUT / "verification.json").exists():
        raise FileExistsError("Preserve the completed mixed capture proof")
    attempts = len(list(OUT.glob("attempt_*.json")))
    if attempts >= 3:
        raise RuntimeError("Initial attempt plus two repairs exhausted")
    attempt = {"attempt": attempts + 1, "started_utc": datetime.now(timezone.utc).isoformat(),
               "registry_sha256": sha256(registry_path), "script_sha256": sha256(Path(__file__))}
    attempt_path = OUT / f"attempt_{attempts + 1}.json"
    attempt_path.write_text(json.dumps(attempt, indent=2) + "\n")
    destination = OUT / f"capture_attempt_{attempts + 1}"
    destination.mkdir(exist_ok=False)
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):
            raw, labels = generated_data()
            first, schema = compile_module(raw, labels)
            inventory = preprocessing_inventory(first)
            changed = raw.clone()
            changed[FIT_ROWS:, :6] = changed[FIT_ROWS:, :6] * 4 + 100
            changed[FIT_ROWS:, 6:] = torch.tensor([555., 888.])
            other, other_schema = compile_module(changed, labels)
            excluded = (schema == other_schema and fingerprint(tensor_attributes(first.model)) == fingerprint(tensor_attributes(other.model))
                        and fingerprint(tensor_attributes(first.creator.static_model)) == fingerprint(tensor_attributes(other.creator.static_model)))
            if not excluded:
                raise AssertionError("Monitor-only feature perturbation changed fitted initialization or preprocessing")
            del first, other
            reference = run_arm(raw, labels, None, start + 300)
            captured = run_arm(raw, labels, destination, start + 300)
            observation = captured["observer"]
            if observation.capture is None:
                raise AssertionError("Missing epoch4 export")
            capture = dict(observation.capture)
            prefix_probability = capture.pop("prefix_probability")
            path = ROOT / capture["path"]
            with torch.no_grad():
                restored = torch.jit.load(str(path), map_location="cpu").eval()(raw)
            deltas = {"final_model": max_state_difference(reference["state"], captured["state"]),
                      "final_probabilities": float((reference["probabilities"] - captured["probabilities"]).abs().max()),
                      "final_adam": optimizer_state_max_difference(reference["optimizer"], captured["optimizer"]),
                      "immutable_prefix_reload": float((restored - prefix_probability).abs().max()),
                      "maximum_epoch_state": max(max_state_difference(a, b) for a, b in zip(reference["observer"].epoch_states, observation.epoch_states)),
                      "endpoint_vs_prefix_parameters": max_state_difference(captured["state"], observation.epoch_states[3])}
            checks = {"train_only_monitor_feature_counterfactual": excluded,
                      "all_epoch_and_final_states_exact": all(deltas[k] == 0 for k in ("final_model", "final_probabilities", "final_adam", "immutable_prefix_reload", "maximum_epoch_state")),
                      "continued_model_changed_after_capture": deltas["endpoint_vs_prefix_parameters"] > 0,
                      "batch_rng_schedule_progress_paths_equal": reference["observer"].batch_trace == observation.batch_trace,
                      "real_monitor_prediction_metric_paths_equal": reference["monitor"] == captured["monitor"],
                      "final_all_rng_equal": reference["rng"] == captured["rng"],
                      "all_observations_passive": all(all(row.values()) for row in observation.observation_checks),
                      "learned_static_preprocessing_unchanged": reference["static_tensor_hash"] == captured["static_tensor_hash"],
                      "epoch4_artifact_hash_immutable": capture["artifact_sha256"] == sha256(path),
                      "real_validation_batches_executed": reference["observer"].validation_batches == observation.validation_batches == 48}
            if not all(checks.values()):
                raise AssertionError(f"Mixed capture acceptance failed: {checks}")
            check_deadline(start + 300)
            sources = [Path(__file__), ROOT / "scripts/test_research_mixed_capture_v1.py",
                       ROOT / "scripts/research_epoch_capture_v1.py", ROOT / "scripts/research_epoch_control_v1.py",
                       ROOT / "scripts/research_epoch_resume_v1.py", ROOT / "scripts/realmlp_categorical.py",
                       ROOT / ".venv/Lib/site-packages/pytabkit/models/training/nn_creator.py",
                       ROOT / ".venv/Lib/site-packages/pytabkit/models/nn_models/pipeline.py"]
            output = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": time.monotonic() - start, "registry": registry,
                      "registry_sha256": sha256(registry_path), "attempt": attempts + 1,
                      "schema": schema, "preprocessing_inventory": inventory,
                      "preprocessing_inventory_count": len(inventory), "checks": checks,
                      "maximum_absolute_differences": deltas, "capture": capture,
                      "validation_batch_counts": [reference["observer"].validation_batches, observation.validation_batches],
                      "best_monitor_epochs_observed_not_selected": [reference["best_monitor_epoch"], captured["best_monitor_epoch"]],
                      "real_monitor_trace": captured["monitor"], "fixed_fit_probes": observation.probes,
                      "observation_live_state_checks": observation.observation_checks,
                      "batch_trace_sha256": fingerprint(observation.batch_trace),
                      "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                      "versions": {name: importlib.metadata.version(name) for name in ("torch", "pytabkit", "pytorch-lightning", "numpy")},
                      "limits": [registry["limits"],
                                 "Tensor ownership inventory covers registered tensors and direct/container tensor attributes on modules, not arbitrary object graphs",
                                 "Monitor labels generate real synthetic AUC but those values are mechanism diagnostics, not generalization evidence",
                                 "Reference retains common read-only auditing; added probes, clone/export and reload are the tested intervention",
                                 "Installed preprocessing and optimizer semantics are preserved; no semantic correction is claimed"]}
        (OUT / "verification.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        attempt.update(status="passed", elapsed_seconds=time.monotonic() - start)
        attempt_path.write_text(json.dumps(attempt, indent=2) + "\n")
        return output
    except Exception as error:
        attempt.update(status="failed", elapsed_seconds=time.monotonic() - start,
                       error_type=type(error).__name__, error=str(error))
        attempt_path.write_text(json.dumps(attempt, indent=2) + "\n")
        raise


if __name__ == "__main__":
    result = run_checks()
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "checks", "maximum_absolute_differences")}, indent=2))
