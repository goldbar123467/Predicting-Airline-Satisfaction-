"""One-time local evaluation of one registered six-policy continuation stage.

No training, audit/test scoring, alpha search, release mutation or submission.
Cloud-produced full raw/native reload receipts are checked locally. This
evaluator does not execute native inference or decode raw train/auxiliary data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
from itertools import zip_longest

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "fixed_epoch_continuation_01"
POLICY_IDS = ("linear_lr", "decay_correction", "ema", "label_smoothing", "freeze_embeddings", "head_only")
POLICY_ID = POLICY_IDS[0]
DEVELOPMENT_ROWS = 629671
LOCAL_PROTOCOL = ROOT / f"artifacts/{CAMPAIGN_ID}/local_evaluation_protocol.json"
PREPARATION = ROOT / f"cloud/continuation_six_v1/{CAMPAIGN_ID}/experiment/preparation_manifest.json"
ASSESSMENT_WORKSPACE = f"cloud/continuation_six_v1/{CAMPAIGN_ID}/experiment/assessment_workspace"
SEQUENCE_PLAN_SHA256 = "ff35aa6dec492f37345f09ab75c2f83b7ccc0b725ab109da8a93af34b2699f69"
SEQUENCE_AUTH_SHA256 = "dc8e2502db03ce4e18fe7b656e0098cfd757ad159157898757ad6c53700df3fe"
EXECUTION_FLAGS = {"deterministic_algorithms": True, "deterministic_warn_only": False,
    "cudnn_benchmark": False, "cudnn_deterministic": True, "cuda_matmul_allow_tf32": False,
    "cublas_workspace_config": ":4096:8", "startup_cuda_initialized": False}
RUNTIME_MODULES = ("pytabkit.models.optim.optimizers", "pytabkit.models.training.lightning_modules",
                   "pytabkit.models.training.nn_creator", "pytabkit.models.training.scheduling",
                   "pytabkit.models.training.coord", "pytabkit.models.training.lightning_callbacks",
                   "pytabkit.models.data.data", "pytabkit.models.torch_utils",
                   "pytabkit.models.nn_models.nn", "pytabkit.models.nn_models.base",
                   "pytabkit.models.nn_models.categorical", "pytabkit.models.nn_models.activations")
ADAPTER_WORKSPACE_SOURCES = ("scripts/continuation_six_adapter_v1.py", "scripts/continuation_six_policy_v1.py", "scripts/realmlp.py",
    "scripts/realmlp_categorical.py", "scripts/common.py", "scripts/categorical_transform.py")
SPLIT_HASH = "4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c"
INCUMBENT_HASH = "bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311"
EVALUATION = {"mixture_alpha": .1, "min_pooled_gain": 1e-5, "min_macro_gain": 1e-5,
              "max_fold_regression": 2e-5, "class_order": [0, 1], "fold_ids": [0, 1, 2]}
LOGICAL_ENDPOINTS = {"A": {"trajectory": "A", "epoch": 16},
                     "B": {"trajectory": "C", "epoch": 4},
                     "C": {"trajectory": "C", "epoch": 16}}
INTERVENTION = {"common_horizon_epochs": 16, "total_executed_epochs": 192,
                "intervention_after_completed_epoch": 4, "treatment_dropout_base": .05,
                "logical_endpoints": LOGICAL_ENDPOINTS}
SIGNATURE_KEYS = {"network", "learned_static_preprocessing", "gradients", "optimizer",
    "cpu_rng", "cuda_rng", "numpy_rng", "python_rng", "progress", "schedule", "sampler",
    "batch_order", "schema", "input_train", "input_monitor", "probe", "modes",
    "dropout_scopes", "parameter_groups"}
LIMITATION = (
    "Fixed continuation-policy comparisons on historically reused development data. "
    "Not a nested or independent confirmation. Advancement is an engineering "
    "gate, not statistical significance or automatic release/submission."
)


def policy_for(policy_id: str) -> dict:
    """Independent canonical recipe, never imported from the training adapter."""
    extras = {
        "control": {},
        "linear_lr": {"linear_lr_end_epoch": 16},
        "decay_correction": {"decay_factor_applications_after_epoch4": 1},
        "ema": {"ema_half_life_epochs": 1.0},
        "label_smoothing": {"label_smoothing_after_epoch4": .05},
        "freeze_embeddings": {"frozen_roles": ["numeric_embedding", "categorical_embedding", "learned_preprocessing"]},
        "head_only": {"trainable_roles_after_epoch4": ["final_affine"]},
    }
    if policy_id not in extras:
        raise ValueError("Unregistered continuation policy")
    return {"id": policy_id, "after_completed_epoch": 4, "horizon": 16,
            "dropout_after_epoch4": .05, **extras[policy_id]}


def configure_stage(stage: str) -> None:
    """Select one fixed stage per evaluator process, before any artifact read."""
    global CAMPAIGN_ID, POLICY_ID, LOCAL_PROTOCOL, PREPARATION, ASSESSMENT_WORKSPACE
    if not isinstance(stage, str) or re.fullmatch(r"fixed_epoch_continuation_0[1-6]", stage) is None:
        raise ValueError("Unknown continuation stage")
    CAMPAIGN_ID = stage
    POLICY_ID = POLICY_IDS[int(stage[-2:]) - 1]
    LOCAL_PROTOCOL = ROOT / f"artifacts/{stage}/local_evaluation_protocol.json"
    PREPARATION = ROOT / f"cloud/continuation_six_v1/{stage}/experiment/preparation_manifest.json"
    ASSESSMENT_WORKSPACE = f"cloud/continuation_six_v1/{stage}/experiment/assessment_workspace"


def finite_number(value, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError("Expected a finite numeric policy value")
    if minimum is not None and value < minimum:
        raise ValueError("Policy value is below its bound")
    return float(value)


def linear_lr_multiplier(prefix_multiplier: float, epoch_clock: float) -> float:
    multiplier = finite_number(prefix_multiplier, minimum=0)
    clock = finite_number(epoch_clock)
    if not 4 <= clock <= 16:
        raise ValueError("Linear continuation clock outside [4,16]")
    return multiplier * max(0.0, (16.0 - clock) / 12.0)


def decay_shrink(lr: float, wd: float, lr_factor: float, wd_factor: float, *, applications: int) -> float:
    """Independent scalar oracle; lr/wd are scheduled bases before factors."""
    if type(applications) is not int or applications not in (1, 2):
        raise ValueError("Invalid decay-factor multiplicity")
    eta, decay, a, b = [finite_number(v, minimum=0) for v in (lr, wd, lr_factor, wd_factor)]
    return 1.0 - eta * decay * (a * b) ** applications


def ema_decay(updates_per_epoch: int) -> float:
    if type(updates_per_epoch) is not int or updates_per_epoch < 1:
        raise ValueError("EMA epoch must contain positive integral updates")
    return 2.0 ** (-1.0 / updates_per_epoch)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def checked_path(root: Path, relative: str, *, inside: Path | None = None) -> Path:
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Expected a portable POSIX-relative receipt path")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Expected a workspace-relative path without parent traversal")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or (inside is not None and not path.is_relative_to(inside.resolve())):
        raise ValueError("Artifact path escapes its allowed directory")
    return path


def require_hash(path: Path, expected: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Invalid SHA256 identity for {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return actual


def primitive_state_hash(value) -> str:
    """The adapter's typed fingerprint for JSON values/arrays; no ML imports."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key); visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        elif item is None or isinstance(item, (str, int, bool, float)):
            if isinstance(item, float) and not np.isfinite(item):
                raise ValueError("Nonfinite fingerprint value")
            digest.update((type(item).__name__ + ":" + repr(item) + ";").encode())
        else:
            raise TypeError("Unsupported portable signature value")
    visit(value)
    return digest.hexdigest()


def require_digest(value) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid portable state digest")


def intervention_for(trajectory: str) -> dict:
    return {"policy": "hold_base_after_epoch4",
            "start_epoch": 5, "base_p_drop": .05, "treatment_multiplier_after_epoch4": 1.0,
            "scope": "all registered p_drop schedules; preserve every base value and scope mapping"}


def bound_json(root: Path, record: dict, stem: str, expected: Path, output: Path) -> dict:
    path = checked_path(root, record[stem + "_path"], inside=output)
    if path != expected.resolve():
        raise ValueError(f"Unexpected {stem} path")
    require_hash(path, record[stem + "_sha256"])
    return read_json(path)


def update_rows(path: Path):
    """Stream bounded JSONL records; retain original bytes for prefix binding."""
    with path.open("rb") as stream:
        while True:
            line = stream.readline(262145)
            if not line:
                return
            if len(line) > 262144 or not line.endswith(b"\n"):
                raise ValueError("Oversized or incomplete update receipt")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Update receipt must be an object")
            yield value, line


def validate_update_pair(paths: list[Path], adapters: list[dict], prefixes: list[dict]) -> dict:
    steps = adapters[0]["updates_per_epoch"]
    scopes = adapters[0]["dropout_scopes"]
    if (type(steps) is not int or steps < 1 or adapters[1]["updates_per_epoch"] != steps
            or any(a["optimizer_updates"] != steps * 16 for a in adapters)
            or not scopes or scopes != adapters[1]["dropout_scopes"]
            or len({scope["name"] for scope in scopes}) != len(scopes)):
        raise ValueError("Paired update counts or active dropout scope inventory differ")
    for scope in scopes:
        if scope["base_value"] != .05 or not isinstance(scope["schedule_pattern"], str):
            raise ValueError("Unexpected dropout base/scope mapping")
    hashes = [hashlib.sha256(), hashlib.sha256()]
    order_hashes = [hashlib.sha256(), hashlib.sha256()]
    count = 0
    lr_anchors = None
    for count, pair in enumerate(zip_longest(*(update_rows(path) for path in paths)), 1):
        if None in pair or count > steps * 16:
            raise ValueError("Missing or extra applied-update evidence")
        rows = [item[0] for item in pair]
        epoch, batch = (count - 1) // steps + 1, (count - 1) % steps
        for index, (row, raw) in enumerate(pair):
            expected = {"epoch": epoch, "batch_index": batch, "global_step_before": count - 1,
                        "update": count, "horizon": 16, "optimizer_step": "completed",
                        "intervention_active": epoch >= 5}
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError("Applied update has wrong boundary, policy or optimizer step")
            if not np.isfinite(row["epoch_float_before"]) or not epoch - 1 <= row["epoch_float_before"] < epoch:
                raise ValueError("Invalid pre-update epoch clock")
            if primitive_state_hash(row["non_dropout_schedule"]) != row["non_dropout_sha256"]:
                raise ValueError("Non-dropout schedule digest does not match its values")
            if not {"lr", "wd"}.issubset(row["non_dropout_schedule"]) or "p_drop" in row["non_dropout_schedule"]:
                raise ValueError("Missing LR/WD or improperly separated dropout schedule")
            require_digest(row["optimizer_groups_sha256"])
            order = row["batch_order"]
            require_digest(order["indices_sha256"])
            if type(order["rows_per_member"]) is not int or order["rows_per_member"] < 1 or order["members"] != 8:
                raise ValueError("Invalid consumed batch shape")
            item = {key: order[key] for key in ("indices_sha256", "rows_per_member", "members")}
            order_hashes[index].update(primitive_state_hash({"update": count, **item}).encode())
            if order["cumulative_sha256"] != order_hashes[index].hexdigest():
                raise ValueError("Consumed row-order chain differs")
            multipliers, scheduled = row["p_drop_multipliers"], row["scheduled_p_drop_multipliers"]
            if not multipliers or set(multipliers) != set(scheduled):
                raise ValueError("Dropout schedule patterns changed")
            expected_multipliers = {key: 1.0 for key in scheduled} if epoch >= 5 else scheduled
            if multipliers != expected_multipliers:
                raise ValueError("Dropout intervention began early or violated its fixed continuation")
            actual = row["effective_scope_p_drop"]
            if set(actual) != {scope["name"] for scope in scopes}:
                raise ValueError("Applied dropout omits or adds a layer")
            for scope in scopes:
                expected_p = scope["base_value"] * multipliers[scope["schedule_pattern"]]
                if not np.isfinite(expected_p) or not 0 <= expected_p < 1 or actual[scope["name"]] != expected_p:
                    raise ValueError("Applied per-layer dropout differs from registered base and schedule")
            if epoch <= 4:
                hashes[index].update(raw)
        for field in ("epoch_float_before", "batch_order", "scheduled_p_drop_multipliers"):
            if rows[0][field] != rows[1][field]:
                raise ValueError("Paired continuation changed data order or common dropout schedule")
        left, right = [row["non_dropout_schedule"] for row in rows]
        if set(left) != set(right):
            raise ValueError("Paired continuation changed schedule inventory")
        allowed = {"lr"} if POLICY_ID == "linear_lr" else {"ls_eps"} if POLICY_ID == "label_smoothing" else set()
        for key in left:
            if key not in allowed or epoch <= 4:
                if left[key] != right[key]:
                    raise ValueError("Paired continuation changed an unrelated schedule")
        if epoch >= 5 and POLICY_ID == "linear_lr":
            if lr_anchors is None:
                if rows[0]["epoch_float_before"] != 4.0:
                    raise ValueError("Missing exact epoch-four LR anchor")
                lr_anchors = left["lr"]
            expected_lr = {key: linear_lr_multiplier(value, rows[1]["epoch_float_before"])
                           for key, value in lr_anchors.items()}
            if right["lr"] != expected_lr:
                raise ValueError("Applied LR differs from registered linear cooling")
        if epoch >= 5 and POLICY_ID == "label_smoothing":
            if right["ls_eps"] != {key: 1.0 for key in left["ls_eps"]}:
                raise ValueError("Applied label-smoothing multiplier differs")
        if POLICY_ID in {"ema", "label_smoothing"} or epoch <= 4:
            if rows[0]["optimizer_groups_sha256"] != rows[1]["optimizer_groups_sha256"]:
                raise ValueError("Unrelated scheduled optimizer-group summary changed")
        if epoch <= 4 and pair[0][1] != pair[1][1]:
            raise ValueError("Applied prefix updates are not byte-identical")
    if count != steps * 16:
        raise ValueError("Incomplete applied-update trajectory")
    for index in (0, 1):
        if (hashes[index].hexdigest() != prefixes[index]["update_schedule_prefix_sha256"]
                or order_hashes[index].hexdigest() != adapters[index]["consumed_batch_order_sha256"]):
            raise ValueError("Prefix/final update evidence does not bind its declared trajectory")
    return {"updates_per_trajectory": count, "prefix_updates": steps * 4,
            "unrelated_schedules_and_order_identical": True, "all_effective_dropout_values_verified": True,
            "continuation_policy": policy_for(POLICY_ID), "linear_lr_anchors": lr_anchors}


def validate_continuation_states(adapters: list[dict], metadata: list[dict]) -> tuple[dict, set[str]]:
    """Validate named parameter ownership and endpoint type without loading models."""
    states = [adapter["continuation_state"] for adapter in adapters]
    roles = states[0]["parameter_roles"]
    if not roles or roles != states[1]["parameter_roles"]:
        raise ValueError("Paired semantic parameter inventories differ")
    names = [item["name"] for item in roles]
    if len(names) != len(set(names)):
        raise ValueError("Aliased or duplicate semantic parameter inventory")
    permitted = {"numeric_embedding", "categorical_embedding", "learned_preprocessing", "final_affine",
                 "hidden_network", "fixed_label_distribution", "fixed_preprocessing"}
    role_map = {item["name"]: item for item in roles}
    for item in roles:
        if (item["role"] not in permitted or type(item["original_requires_grad"]) is not bool
                or not isinstance(item["scope"], str) or not isinstance(item["owner_class"], str)
                or not item["owner_class"].startswith("pytabkit.models.")
                or not item["shape"] or any(type(size) is not int or size < 1 for size in item["shape"])):
            raise ValueError("Invalid compiled semantic parameter role")
    original_trainable = {item["name"] for item in roles if item["original_requires_grad"]}
    heads = {item["name"] for item in roles if item["original_requires_grad"] and item["role"] == "final_affine"}
    if len(heads) != 2 or {name.rsplit(".", 1)[-1] for name in heads} != {"weight", "bias"}:
        raise ValueError("Head-only contract requires exactly one final affine weight and bias")
    freeze = set()
    if POLICY_ID == "freeze_embeddings":
        freeze = {item["name"] for item in roles if item["original_requires_grad"]
                  and item["role"] in policy_for(POLICY_ID)["frozen_roles"]}
    elif POLICY_ID == "head_only":
        freeze = original_trainable - heads
    if POLICY_ID in {"freeze_embeddings", "head_only"} and (not freeze or freeze == original_trainable):
        raise ValueError("Freeze policy is empty or removes every trainable parameter")
    for index, (adapter, state, endpoint_metadata) in enumerate(zip(adapters, states, metadata)):
        expected_policy = policy_for("control" if index == 0 else POLICY_ID)
        expected_frozen = freeze if index else set()
        is_ema = index == 1 and POLICY_ID == "ema"
        if (adapter.get("continuation_policy") != expected_policy or state["continuation_policy"] != expected_policy
                or state.get("applied_after_prefix") is not True
                or state["frozen_parameters"] != sorted(expected_frozen)
                or set(state["frozen_initial_sha256"]) != expected_frozen
                or set(state["frozen_final_sha256"]) != expected_frozen
                or state["frozen_parameters_unchanged"] != {name: True for name in expected_frozen}
                or state["frozen_initial_sha256"] != state["frozen_final_sha256"]):
            raise ValueError("Applied continuation/frozen-parameter policy differs")
        for digest in state["frozen_initial_sha256"].values():
            require_digest(digest)
        expected_updates = adapter["updates_per_epoch"] * 12 if is_ema else 0
        if (state["ema_updates"] != expected_updates
                or state["ema_parameters"] != (sorted(original_trainable) if is_ema else [])
                or state["ema_decay_per_update"] != (ema_decay(adapter["updates_per_epoch"]) if is_ema else None)):
            raise ValueError("EMA window, parameter set or applied update count differs")
        if is_ema:
            require_digest(state["ema_sha256"])
        elif state["ema_sha256"] is not None:
            raise ValueError("Unregistered averaging state")
        if set(state["ema_parameter_sha256"]) != (original_trainable if is_ema else set()):
            raise ValueError("EMA shadow omits or adds named trainable parameters")
        for epoch in (4, 16):
            value = endpoint_metadata[str(epoch)]
            endpoint = adapter["endpoints"][str(epoch)]
            averaged = is_ema and epoch == 16
            kind = "ema_trainable_parameters" if averaged else "raw_training_endpoint"
            if (value["continuation_policy"] != expected_policy or value["export_kind"] != kind
                    or endpoint["export_kind"] != kind
                    or value["network_sha256"] != endpoint["network_sha256"]
                    or value["raw_training_network_sha256"] != endpoint["raw_training_network_sha256"]):
                raise ValueError("Native endpoint does not bind its raw/averaged policy")
            require_digest(value["network_sha256"])
            require_digest(value["raw_training_network_sha256"])
            for key in ("parameter_sha256", "raw_training_parameter_sha256"):
                if set(value[key]) != set(role_map):
                    raise ValueError("Endpoint omits a named parameter fingerprint")
                for digest in value[key].values():
                    require_digest(digest)
            if not averaged and value["parameter_sha256"] != value["raw_training_parameter_sha256"]:
                raise ValueError("Raw endpoint parameter fingerprints differ")
            if averaged:
                for name in role_map:
                    expected_hash = state["ema_parameter_sha256"][name] if name in original_trainable else value["raw_training_parameter_sha256"][name]
                    if value["parameter_sha256"][name] != expected_hash:
                        raise ValueError("Exported EMA parameter differs from recorded shadow or fixed preprocessing")
            if not averaged and value["network_sha256"] != value["raw_training_network_sha256"]:
                raise ValueError("Raw endpoint differs from live training network")
            if epoch == 16 and value["raw_training_network_sha256"] != adapter["final_network_sha256"]:
                raise ValueError("Endpoint raw network differs from final training state")
            if epoch == 4:
                prefix_state = value["continuation_state"]
                if (prefix_state["applied_after_prefix"] is not False or prefix_state["frozen_parameters"]
                        or prefix_state["ema_updates"] != 0 or prefix_state["ema_parameters"]):
                    raise ValueError("Policy altered a prefix endpoint before its gate")
        if endpoint_metadata["16"]["continuation_state"] != state:
            raise ValueError("Endpoint and completed trajectory continuation states differ")
        for name in expected_frozen:
            if (state["frozen_initial_sha256"][name] != endpoint_metadata["4"]["raw_training_parameter_sha256"][name]
                    or state["frozen_final_sha256"][name] != endpoint_metadata["16"]["raw_training_parameter_sha256"][name]):
                raise ValueError("Frozen parameter evidence does not bind both native endpoints")
    if POLICY_ID == "ema":
        for key in ("final_network_sha256", "final_optimizer_sha256", "final_rng_sha256"):
            if adapters[0][key] != adapters[1][key]:
                raise ValueError("Inference-only EMA changed the live training trajectory")
    return role_map, freeze


def validate_smoothing_scopes(scopes: list[dict], schedule: dict, epsilon: float) -> None:
    if not scopes or len({item["name"] for item in scopes}) != len(scopes):
        raise ValueError("Missing or duplicate applied smoothing layer")
    for item in scopes:
        shape = item["distribution_shape"]
        if (not isinstance(shape, list) or not shape or shape[-1] != 2
                or any(type(size) is not int or size < 1 for size in shape) or np.prod(shape) > 4096):
            raise ValueError("Invalid bounded binary smoothing distribution")
        uniform = np.full(shape, .5, dtype=np.float32)
        digest = hashlib.sha256(str(("torch.float32", tuple(shape))).encode() + uniform.tobytes()).hexdigest()
        expected = {"base_pattern": "", "schedule_pattern": "", "base_value": epsilon,
                    "schedule_value": schedule["ls_eps"][""], "effective_ls_eps": epsilon,
                    "distribution_uniform_binary": True, "distribution_sha256": digest,
                    "negative_positive_targets": [epsilon / 2.0, 1.0 - epsilon / 2.0]}
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError("Actual smoothing getter/target distribution differs")
        if item["effective_ls_eps"] != item["base_value"] * item["schedule_value"]:
            raise ValueError("Applied smoothing value does not follow getter")


def validate_optimizer_pair(paths: list[Path], schedule_paths: list[Path], adapters: list[dict],
                            role_map: dict, freeze: set[str]) -> dict:
    """Check every recorded actual group getter and manual decay application."""
    steps = adapters[0]["updates_per_epoch"]
    inventory = adapters[0]["parameter_groups"]
    if inventory != adapters[1]["parameter_groups"]:
        raise ValueError("Initial parameter-group inventory differs")
    by_name = {item["name"]: item for item in inventory}
    if len(by_name) != len(inventory) or set(by_name) != set(role_map):
        raise ValueError("Optimizer ownership is not bound to compiled parameter inventory")
    expected_names = {name for name, item in role_map.items() if item["original_requires_grad"]}
    smoothing_inventory = None
    count = 0
    for count, four in enumerate(zip_longest(*(update_rows(path) for path in [*paths, *schedule_paths])), 1):
        if None in four or count > 16 * steps:
            raise ValueError("Missing or extra optimizer application evidence")
        rows, schedules = [item[0] for item in four[:2]], [item[0] for item in four[2:]]
        epoch = (count - 1) // steps + 1
        for index, (row, schedule) in enumerate(zip(rows, schedules)):
            policy = "control" if index == 0 else POLICY_ID
            active = index == 1 and epoch >= 5
            if (row["epoch"] != epoch or row["update"] != count or row["continuation_policy"] != policy
                    or row["epoch_float_before"] != schedule["epoch_float_before"]):
                raise ValueError("Optimizer evidence has wrong policy, clock or update")
            groups = row["groups"]
            if len(groups) != len(expected_names) or {g["name"] for g in groups} != expected_names:
                raise ValueError("Actual optimizer omits or adds parameter groups")
            for group_index, group in enumerate(groups):
                name = group["name"]
                initial = by_name[name]
                frozen = active and name in freeze
                corrected = active and POLICY_ID == "decay_correction"
                required = {"index": group_index, "scope": initial["scope"], "lr_factor": initial["lr_factor"],
                            "wd_factor": initial["wd_factor"], "frozen": frozen, "requires_grad": not frozen,
                            "decay_factor_applications": 1 if corrected else 2}
                if any(group.get(key) != value for key, value in required.items()):
                    raise ValueError("Applied parameter identity, factors or freeze mask differs")
                lr = finite_number(group["getter_lr"], minimum=0)
                wd = finite_number(group["getter_wd"], minimum=0)
                if set(group["getters"]) != {"lr", "wd"}:
                    raise ValueError("Missing applied optimizer getter scopes")
                for hyper, factor in (("lr", initial["lr_factor"]), ("wd", initial["wd_factor"])):
                    getter = group["getters"][hyper]
                    base = adapters[index]["constructor"][hyper]
                    if type(base) not in (int, float):
                        raise ValueError("Registered scalar optimizer base changed")
                    multiplier = schedule["non_dropout_schedule"][hyper][""]
                    expected_getter = {"base_pattern": "", "schedule_pattern": "", "base_value": base,
                                       "schedule_value": multiplier, "unfactored_value": base * multiplier}
                    if getter != expected_getter or group["getter_" + hyper] != base * multiplier * factor:
                        raise ValueError("Actual optimizer getter differs from registered base/schedule/factor")
                expected_coefficient = 0.0 if frozen else lr * wd * (
                    1.0 if corrected else initial["lr_factor"] * initial["wd_factor"])
                if group["decay_coefficient"] != expected_coefficient:
                    raise ValueError("Manual weight decay differs from independent scalar formula")
            compact_groups = [{"name": group["name"], "lr": group["getter_lr"],
                               "effective_decay_coefficient": group["decay_coefficient"], "frozen": group["frozen"]}
                              for group in groups]
            if evaluator_hash := schedule.get("optimizer_groups_sha256"):
                if evaluator_hash != primitive_state_hash(compact_groups):
                    raise ValueError("Schedule group digest does not bind actual optimizer applications")
            else:
                raise ValueError("Missing compact applied optimizer group digest")
            epsilon = .05 if active and POLICY_ID == "label_smoothing" else 0.0
            if not row["ls_eps"] or any(value != epsilon for value in row["ls_eps"].values()):
                raise ValueError("Label-smoothing base differs from registered target objective")
            validate_smoothing_scopes(row["label_smoothing_scopes"], schedule["non_dropout_schedule"], epsilon)
            layer_identity = [{key: item[key] for key in ("name", "scope", "base_pattern", "schedule_pattern",
                               "distribution_shape", "distribution_sha256", "distribution_uniform_binary")}
                              for item in row["label_smoothing_scopes"]]
            if smoothing_inventory is None:
                smoothing_inventory = layer_identity
            if layer_identity != smoothing_inventory:
                raise ValueError("Applied smoothing layer inventory changed after initialization")
            expected_ema = max(0, count - 4 * steps) if index == 1 and POLICY_ID == "ema" else 0
            if row["ema_updates"] != expected_ema:
                raise ValueError("EMA update sequence skips or adds averaging operations")
        if epoch <= 4:
            if {k: v for k, v in rows[0].items() if k != "continuation_policy"} != {
                    k: v for k, v in rows[1].items() if k != "continuation_policy"}:
                raise ValueError("Applied optimizer prefix differs")
        else:
            for left, right in zip(rows[0]["groups"], rows[1]["groups"]):
                if left["name"] != right["name"] or left["getter_wd"] != right["getter_wd"]:
                    raise ValueError("Unregistered optimizer ordering or WD getter changed")
                if POLICY_ID != "linear_lr" and left["getter_lr"] != right["getter_lr"]:
                    raise ValueError("Unregistered LR getter changed")
    if count != 16 * steps:
        raise ValueError("Incomplete optimizer application evidence")
    return {"verified_optimizer_updates_per_trajectory": count, "actual_group_getters_checked": True,
            "decay_formula_checked": True, "frozen_parameters": sorted(freeze)}


def integral_ids(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise ValueError("IDs must be a one-dimensional integer array")
    if values.dtype.kind == "u" and len(values) and values.max() > np.iinfo(np.int64).max:
        raise ValueError("IDs exceed int64 range")
    return values.astype(np.int64, copy=False)


def ids_sha256(ids: np.ndarray) -> str:
    return hashlib.sha256(np.sort(integral_ids(ids)).astype("<i8").tobytes()).hexdigest()


def development_split(split: pd.DataFrame, expected_rows: int | None = None) -> pd.DataFrame:
    if list(split.columns) != ["id", "fold"] or split.isna().any().any() or not split.id.is_unique:
        raise ValueError("Invalid authoritative split identity/schema")
    integral_ids(split.id.to_numpy())
    if split.fold.dtype.kind not in "iu":
        raise ValueError("Fold identities must be integral")
    # Exclude the audit before any labels or predictions are joined.
    selected = split.loc[split.fold >= 0, ["id", "fold"]].copy()
    if (set(selected.fold) != {0, 1, 2}
            or (expected_rows is not None and len(selected) != expected_rows)):
        raise ValueError("Unexpected development fold population")
    return selected


def aligned_oof(frame: pd.DataFrame, expected_ids: np.ndarray, expected_folds: np.ndarray,
                expected_labels: np.ndarray | None, *, class_order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Strict keyed join; includes no ranking or other metric operation."""
    if class_order != [0, 1]:
        raise ValueError("Required class order is [0,1], prediction is class 1")
    expected_ids = integral_ids(expected_ids)
    expected_folds = np.asarray(expected_folds)
    if (set(frame.columns) != {"id", "fold", "satisfaction", "prediction"}
            or len(frame.columns) != 4 or len(frame) != len(expected_ids)
            or not len(frame) or frame.isna().any().any() or not frame.id.is_unique
            or len(np.unique(expected_ids)) != len(expected_ids)
            or expected_folds.shape != expected_ids.shape or (expected_folds < 0).any()):
        raise ValueError("OOF schema, row count, uniqueness or development boundary violation")
    integral_ids(frame.id.to_numpy())
    if frame.fold.dtype.kind not in "iu" or (frame.fold < 0).any():
        raise ValueError("OOF must contain integral development folds only")
    order = pd.Index(frame.id).get_indexer(expected_ids)
    if (order < 0).any():
        raise ValueError("OOF has missing or unexpected IDs")
    joined = frame.iloc[order]
    if not np.array_equal(joined.fold.to_numpy(), expected_folds):
        raise ValueError("OOF fold mismatch after keyed join")
    labels = joined.satisfaction.to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("OOF target is not binary")
    labels = labels.astype(np.int8)
    if expected_labels is not None and not np.array_equal(labels, expected_labels):
        raise ValueError("OOF labels differ from verified development labels")
    for fold in np.unique(expected_folds):
        if len(np.unique(labels[expected_folds == fold])) != 2:
            raise ValueError("Every scored fold must contain both classes")
    if joined.prediction.dtype.kind not in "fiu":
        raise ValueError("Prediction must be numeric probability")
    probability = joined.prediction.to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid finite [0,1] positive-class probability")
    return probability, labels


def validate_manifest_inventory(manifest: dict) -> tuple[list[dict], list[dict]]:
    trajectories, endpoints = manifest["trajectories"], manifest["endpoints"]
    expected_trajectories = {(phase, fold, trajectory) for phase in ("inner", "outer")
                             for fold in (0, 1, 2) for trajectory in ("A", "C")}
    expected_endpoints = {(fold, arm) for fold in (0, 1, 2) for arm in ("A", "B", "C")}
    actual_trajectories = [(row["phase"], row["fold"], row["trajectory"]) for row in trajectories]
    actual_endpoints = [(row["fold"], row["arm"]) for row in endpoints]
    if any(type(row["fold"]) is not int for row in [*trajectories, *endpoints]):
        raise ValueError("Manifest folds must be JSON integers")
    if len(actual_trajectories) != 12 or set(actual_trajectories) != expected_trajectories:
        raise ValueError("Exactly all twelve completed trajectories are required")
    if len(actual_endpoints) != 9 or set(actual_endpoints) != expected_endpoints:
        raise ValueError("Exactly all nine verified outer endpoints are required")
    return trajectories, endpoints


def validate_partition(partition: dict[str, np.ndarray], ids: np.ndarray,
                       folds: np.ndarray, fold: int, phase: str) -> dict:
    if set(partition) != {"training_ids", "monitor_ids", "outer_validation_ids"}:
        raise ValueError("Unexpected saved partition identity fields")
    arrays = {name: integral_ids(value) for name, value in partition.items()}
    if any(len(np.unique(value)) != len(value) for value in arrays.values()):
        raise ValueError("Duplicate partition IDs")
    training, monitor, held = (arrays[name] for name in ("training_ids", "monitor_ids", "outer_validation_ids"))
    expected_train, expected_held = ids[folds != fold], ids[folds == fold]
    if not np.array_equal(np.sort(held), np.sort(expected_held)):
        raise ValueError("Outer validation partition IDs differ")
    if (np.intersect1d(training, monitor).size or np.intersect1d(training, held).size
            or np.intersect1d(monitor, held).size
            or not np.array_equal(np.sort(np.concatenate([training, monitor])), np.sort(expected_train))):
        raise ValueError("Training/monitor partition leaks, omits or adds rows")
    expected_monitor_count = 0 if phase == "outer" else int(np.ceil(.1 * len(expected_train)))
    if phase not in {"inner", "outer"} or len(monitor) != expected_monitor_count:
        raise ValueError("Unexpected monitor partition size")
    return {"fold": fold, "phase": phase, "training_rows": len(training), "monitor_rows": len(monitor),
            "training_ids_sha256": ids_sha256(training), "monitor_ids_sha256": ids_sha256(monitor),
            "outer_validation_ids_sha256": ids_sha256(held),
            "scope": "Exact outer membership, train/monitor disjointness and registered sizes; seed/order enforced by pinned runner and adapter"}


def validate_native_receipt(receipt: dict, endpoint: dict, expected_ids: np.ndarray,
                            registry_hash: str, root: Path, output: Path) -> dict:
    fold, arm = endpoint["fold"], endpoint["arm"]
    trajectory, epoch = ("A", 16) if arm == "A" else ("C", 4 if arm == "B" else 16)
    model_dir = output / f"fold_{fold}/outer/{trajectory}/epoch_{epoch:03d}"
    required = {
        "id": CAMPAIGN_ID, "status": "passed", "fold": fold, "arm": arm,
        "registry_sha256": registry_hash, "verification_scope": "full_outer_fold",
        "class_order": [0, 1], "row_count": len(expected_ids), "ids_sha256": ids_sha256(expected_ids),
        "prediction_path": endpoint["prediction_path"], "prediction_sha256": endpoint["prediction_sha256"],
        "reference_kind": "first_native_endpoint_reload", "adapter_parity_scope": "fixed_probes_at_capture",
        "atol": 2e-6, "rtol": 1e-5, "parity_passed": True, "raw_reload_verified": True,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Native receipt identity, scope, class or verification contract differs")
    if checked_path(root, receipt["model_directory"], inside=output) != model_dir.resolve():
        raise ValueError("Native receipt points to wrong endpoint directory")
    for field in ("max_absolute_error", "max_scaled_error"):
        value = receipt[field]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("Invalid native parity error")
    if receipt["max_scaled_error"] > 1 or receipt["max_absolute_error"] > 2e-6 + 1e-5:
        raise ValueError("Native parity exceeds fixed tolerance")
    hashes = receipt["artifact_hashes"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing native artifact identities")
    paths = []
    for relative, expected in hashes.items():
        path = checked_path(root, relative, inside=output)
        require_hash(path, expected)
        paths.append(path)
    metadata_path = model_dir / "metadata.json"
    if ((output / f"fold_{fold}/outer/{trajectory}/transform.json").resolve() not in paths
            or metadata_path.resolve() not in paths):
        raise ValueError("Native verification must bind endpoint metadata and fitted transformer")
    metadata = read_json(metadata_path)
    graph_name = metadata["graph_file"]
    if not isinstance(graph_name, str) or Path(graph_name).name != graph_name:
        raise ValueError("Native metadata graph filename must be local to endpoint")
    graph_path = model_dir / graph_name
    if graph_path.resolve() not in paths or graph_path.suffix != ".pt":
        raise ValueError("Receipt must hash the exact graph consumed by native loader")
    require_hash(graph_path, metadata["graph_sha256"])
    expected_horizon = 16
    if (metadata.get("schema_version") != 1 or metadata.get("model_type") != "realmlp_categorical"
            or metadata.get("classes") != [0, 1] or metadata.get("executed_epochs") != epoch
            or metadata.get("schedule_horizon_epochs") != expected_horizon
            or not isinstance(metadata.get("input_schema"), dict)
            or metadata["input_schema"].get("n_features") != metadata.get("n_features")
            or not isinstance(metadata.get("n_features"), int) or metadata["n_features"] < 1):
        raise ValueError("Native metadata class/schema or fixed endpoint differs")
    return {"fold": fold, "arm": arm, "row_count": len(expected_ids),
            "receipt_verification": "Cloud worker performed full raw/native reload; downloaded receipt and artifact identities checked locally; evaluator did not re-execute inference",
            "reference_kind": receipt["reference_kind"], "adapter_parity_scope": receipt["adapter_parity_scope"],
            "max_absolute_error": receipt["max_absolute_error"], "max_scaled_error": receipt["max_scaled_error"],
            "artifact_hashes": hashes}


def load_incumbent(selection: dict, ids: np.ndarray, folds: np.ndarray, root: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Reconstruct only accepted development OOF; no metrics are called here."""
    weights = selection["weights"]
    if (len(weights) != 15 or any(not np.isfinite(w) or w <= 0 for w in weights.values())
            or abs(sum(weights.values()) - 1) > 1e-12 or selection["split_hash"] != SPLIT_HASH):
        raise ValueError("Unexpected frozen incumbent selection")
    y, score, evidence = None, np.zeros(len(ids), dtype=np.float64), []
    for name, weight in weights.items():
        directory = checked_path(root, f"artifacts/runs/{name}", inside=root / "artifacts/runs")
        result_path = directory / "result.json"
        result_hash = require_hash(result_path, selection["source_result_hashes"][name])
        result = read_json(result_path)
        contract_hash = require_hash(directory / "contract.json", result["contract_hash"])
        contract = read_json(directory / "contract.json")
        if (result["id"] != name or result["split_hash"] != SPLIT_HASH
                or contract["split_hash"] != SPLIT_HASH or contract["run"] != result["run"]):
            raise ValueError("Incumbent run identity changed")
        for relative, expected in contract["code_hashes"].items():
            source = checked_path(directory / "source", relative)
            require_hash(source, expected)
        require_hash(directory / "source/requirements.lock.txt", contract["environment_hash"])
        oof_path = directory / "oof.parquet"
        oof_hash = require_hash(oof_path, result["artifacts"]["oof.parquet"])
        frame = pd.read_parquet(oof_path, columns=["id", "fold", "satisfaction", "prediction"])
        probability, y = aligned_oof(frame, ids, folds, y, class_order=[0, 1])
        score += float(weight) * probability
        evidence.append({"id": name, "weight": weight, "result_sha256": result_hash,
                         "contract_sha256": contract_hash, "oof_sha256": oof_hash})
    if y is None:
        raise ValueError("Missing incumbent labels")
    return y, score, evidence


def cloud_development_split(cloud: pd.DataFrame, canonical: pd.DataFrame,
                            expected_rows: int) -> pd.DataFrame:
    """Prove exact ID/fold membership; canonical audit keys are filtered first."""
    canonical_dev = development_split(canonical, expected_rows)
    cloud_dev = development_split(cloud, expected_rows)
    if len(cloud_dev) != len(cloud) or (cloud.fold < 0).any():
        raise ValueError("Cloud split must contain development rows only")
    order = pd.Index(cloud_dev.id).get_indexer(canonical_dev.id)
    if ((order < 0).any() or not np.array_equal(cloud_dev.iloc[order].id.to_numpy(), canonical_dev.id.to_numpy())
            or not np.array_equal(cloud_dev.iloc[order].fold.to_numpy(), canonical_dev.fold.to_numpy())):
        raise ValueError("Cloud split IDs/folds differ from canonical development split")
    return canonical_dev


def validate_runtime_sources(adapter: dict, root: Path, registry: dict, protocol: dict) -> dict:
    sources = adapter["source_sha256"]
    runtime = adapter["installed_source_provenance"]
    expected_paths = {module: f"artifacts/{CAMPAIGN_ID}/runtime_source/{module.replace('.', '/')}.py"
                      for module in RUNTIME_MODULES}
    if (len(runtime) != len(RUNTIME_MODULES) or {item["module"] for item in runtime} != set(RUNTIME_MODULES)
            or set(sources) != set(ADAPTER_WORKSPACE_SOURCES) | set(expected_paths.values())):
        raise ValueError("Runtime source inventory differs from the portable adapter contract")
    for name in ADAPTER_WORKSPACE_SOURCES:
        if sources[name] != registry["source_hashes"][name]:
            raise ValueError("Adapter did not bind the registered payload source")
    for item in runtime:
        module = item["module"]
        path = expected_paths[module]
        expected = protocol["runtime_source_sha256"][module]
        if (item["recorded_source_path"] != path or item["sha256"] != expected
                or sources[path] != expected or item["distribution"] != "pytabkit" or item["version"] != "1.7.3"
                or not isinstance(item.get("installed_source_path"), str) or not item["installed_source_path"]):
            raise ValueError("Actual imported runtime source differs from the pinned PyTabKit contract")
    for source, expected in sources.items():
        require_hash(checked_path(root, source), expected)
    if adapter["versions"].get("pytabkit") != "1.7.3":
        raise ValueError("Unexpected installed PyTabKit version")
    return {"source_sha256": sources, "versions": adapter["versions"]}


def validate_return_manifest(root: Path) -> dict:
    path = root / "output-manifest.json"
    manifest = read_json(path)
    if manifest.get("id") != CAMPAIGN_ID or not isinstance(manifest.get("files"), dict):
        raise ValueError("Missing downloaded output inventory")
    required = {"registry.json", f"configs/{CAMPAIGN_ID}.json",
                f"artifacts/{CAMPAIGN_ID}/completion_receipt.json",
                f"artifacts/{CAMPAIGN_ID}/completed_manifest.json", f"state/{CAMPAIGN_ID}/run_state.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Downloaded output inventory omits required completion files")
    actual = {"registry.json", f"configs/{CAMPAIGN_ID}.json"}
    for part in ("artifacts", "state", "logs"):
        folder = root / part / CAMPAIGN_ID
        actual.update(path.relative_to(root).as_posix() for path in folder.rglob("*") if path.is_file())
    if actual != set(manifest["files"]):
        raise ValueError("Downloaded output tree has missing or unlisted files")
    for name, item in manifest["files"].items():
        if name not in {"registry.json", f"configs/{CAMPAIGN_ID}.json"} and not any(
                name.startswith(f"{part}/{CAMPAIGN_ID}/") for part in ("artifacts", "state", "logs")):
            raise ValueError("Downloaded output member is outside the campaign allowlist")
        source = checked_path(root, name)
        require_hash(source, item["sha256"])
        if type(item["bytes"]) is not int or item["bytes"] != source.stat().st_size:
            raise ValueError("Downloaded output byte count differs")
    return {"manifest_sha256": sha256(path), "file_count": len(manifest["files"])}


def validate_prefix_pairs(records: list[dict], trajectory_records: dict, root: Path, output: Path) -> list[dict]:
    identities = [(row["phase"], row["fold"]) for row in records]
    expected = {(phase, fold) for phase in ("inner", "outer") for fold in range(3)}
    if len(identities) != 6 or set(identities) != expected or any(type(row["fold"]) is not int for row in records):
        raise ValueError("Exactly six paired prefix records are required")
    evidence = []
    for row in records:
        phase, fold = row["phase"], row["fold"]
        pair = [trajectory_records[(phase, fold, arm)] for arm in ("A", "C")]
        done = [item["done"] for item in pair]
        adapters = [item["adapter"] for item in pair]
        folders = [output / f"fold_{fold}/{phase}/{arm}" for arm in ("A", "C")]
        prefixes, paths, optimizer_paths, all_metadata = [], [], [], []
        for index, arm in enumerate(("A", "C")):
            adapter = adapters[index]
            policy = intervention_for(arm)
            if adapter.get("dropout_policy") != policy["policy"] or adapter.get("intervention") != policy:
                raise ValueError("Adapter applied an unregistered dropout intervention")
            if adapter.get("continuation_policy") != policy_for("control" if index == 0 else POLICY_ID):
                raise ValueError("Adapter applied the wrong stage continuation policy")
            prefix = bound_json(root, done[index], "prefix_state", folders[index] / "prefix_state.json", output)
            for key in ("prefix_state_path", "prefix_state_sha256"):
                if adapter[key] != done[index][key]:
                    raise ValueError("Adapter and worker bind different prefix states")
            fields = {"schema_version": 1, "signature_version": "portable_training_state_v1",
                      "epoch": 4, "horizon": 16, "dropout_policy": policy["policy"],
                      "optimizer_updates": adapter["updates_per_epoch"] * 4,
                      "method_addresses_and_walltimes_excluded": True,
                      "future_policy_excluded_from_state_signature": True,
                      "observation_live_state_unchanged": True}
            if any(prefix.get(key) != value for key, value in fields.items()) or prefix["context"] != adapter["context"]:
                raise ValueError("Prefix receipt has wrong scope, boundary or training context")
            if set(prefix["signature"]) != SIGNATURE_KEYS:
                raise ValueError("Portable prefix omits a required training-state component")
            for digest in prefix["signature"].values():
                require_digest(digest)
            if primitive_state_hash(prefix["signature"]) != prefix["signature_sha256"]:
                raise ValueError("Portable prefix signature digest differs")
            if prefix["endpoint"] != adapter["endpoints"]["4"] or prefix["endpoint"]["network_sha256"] != prefix["signature"]["network"]:
                raise ValueError("Prefix training state does not bind the exported epoch4 network")
            probabilities = np.asarray(prefix["native_probe_probabilities"], dtype=np.float32)
            if (type(prefix["probe_rows"]) is not int or not 1 <= prefix["probe_rows"] <= 257
                    or probabilities.shape != (prefix["probe_rows"], 2) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0)
                    or primitive_state_hash(probabilities) != prefix["native_probe_probability_sha256"]):
                raise ValueError("Invalid prefix native-probability receipt")
            for key in ("native_probe_input_sha256", "update_schedule_prefix_sha256"):
                require_digest(prefix[key])
            endpoint_metadata = {}
            for epoch in (4, 16):
                metadata = read_json(folders[index] / f"epoch_{epoch:03d}/metadata.json")
                endpoint_metadata[str(epoch)] = metadata
                if (metadata.get("dropout_policy") != policy["policy"] or metadata.get("intervention") != policy
                        or metadata.get("executed_epochs") != epoch or metadata.get("schedule_horizon_epochs") != 16
                        or metadata.get("input_schema") != prefix["input_schema"]
                        or metadata.get("constructor") != prefix["constructor"]):
                    raise ValueError("Native endpoint metadata differs from prefix/intervention contract")
            schedule_path = checked_path(root, adapter["update_schedule_path"], inside=output)
            if schedule_path != (folders[index] / "update_schedule.jsonl").resolve():
                raise ValueError("Unexpected applied-update receipt path")
            require_hash(schedule_path, adapter["update_schedule_sha256"])
            paths.append(schedule_path); prefixes.append(prefix)
            optimizer_path = checked_path(root, adapter["optimizer_updates_path"], inside=output)
            if optimizer_path != (folders[index] / "optimizer_updates.jsonl").resolve():
                raise ValueError("Unexpected applied-optimizer receipt path")
            require_hash(optimizer_path, adapter["optimizer_updates_sha256"])
            optimizer_paths.append(optimizer_path)
            all_metadata.append(endpoint_metadata)
        for key in ("signature", "signature_sha256", "constructor", "input_schema", "native_probe_input_sha256",
                    "native_probe_probability_sha256", "native_probe_probabilities", "update_schedule_prefix_sha256"):
            if prefixes[0][key] != prefixes[1][key]:
                raise ValueError("Control/treatment prefix states are not exactly equal")
        for stem in ("prefix_match", "pre_fit_match", "prefix_native"):
            if any(row.get(stem + suffix) != done[1].get(stem + suffix) for suffix in ("_path", "_sha256")):
                raise ValueError("Manifest and treatment worker bind different prefix evidence")
        match = bound_json(root, row, "prefix_match", folders[1] / "prefix_match.json", output)
        required = {"schema_version": 1, "status": "passed", "epoch": 4,
                    "gate_completed_before_epoch5": True, "optimizer_updates_at_gate": adapters[0]["updates_per_epoch"] * 4,
                    "components_equal": {key: True for key in SIGNATURE_KEYS}, "native_probabilities_exact": True,
                    "update_schedule_prefix_exact": True, "native_tolerance": {"rtol": 1e-5, "atol": 2e-6},
                    "probe_rows": prefixes[0]["probe_rows"], "observation_live_state_unchanged": True}
        if any(match.get(key) != value for key, value in required.items()):
            raise ValueError("Prefix gate was not completed before treatment continuation")
        for index, name in enumerate(("control", "treatment")):
            if (match[name + "_prefix_state_path"] != done[index]["prefix_state_path"]
                    or match[name + "_prefix_state_sha256"] != done[index]["prefix_state_sha256"]):
                raise ValueError("Prefix gate does not bind both checked full states")
        if (adapters[1]["prefix_match_path"] != row["prefix_match_path"]
                or adapters[1]["prefix_match_sha256"] != row["prefix_match_sha256"]):
            raise ValueError("Treatment adapter did not bind its prefix gate")
        error = match["control_native_reload_max_abs_error"]
        if not isinstance(error, (int, float)) or not np.isfinite(error) or not 0 <= error <= 1.2e-5:
            raise ValueError("Control-prefix reload parity exceeds fixed bound")
        prefit = bound_json(root, row, "pre_fit_match", folders[1] / "pre_fit_match.json", output)
        check_names = {"campaign", "fold", "phase", "training_rows", "monitor_rows", "training_ids_sha256",
                       "input_feature_columns", "transform_sha256", "feature_policy", "transform_bytes",
                       "partition_schema", "training_ids", "monitor_ids", "outer_validation_ids"}
        if (prefit.get("status") != "passed" or prefit.get("before_treatment_fit") is not True
                or prefit.get("checks") != {key: True for key in check_names}):
            raise ValueError("Pre-fit transform/partition matching failed")
        for index, name in enumerate(("control", "treatment")):
            require_hash(folders[index] / "partitions.npz", prefit[name + "_partition_sha256"])
            require_hash(folders[index] / "transform.json", prefit[name + "_transform_sha256"])
        require_hash(folders[0] / "context.json", prefit["control_context_sha256"])
        if prefit["control_transform_sha256"] != prefit["treatment_transform_sha256"]:
            raise ValueError("Paired fitted transforms differ")
        with np.load(folders[0] / "partitions.npz", allow_pickle=False) as left, np.load(folders[1] / "partitions.npz", allow_pickle=False) as right:
            if set(left.files) != set(right.files) or any(not np.array_equal(left[key], right[key]) for key in left.files):
                raise ValueError("Paired partition arrays/order differ")
        update_evidence = validate_update_pair(paths, adapters, prefixes)
        roles, freeze = validate_continuation_states(adapters, all_metadata)
        optimizer_evidence = validate_optimizer_pair(optimizer_paths, paths, adapters, roles, freeze)
        if phase == "inner" and (row.get("prefix_native_path") is not None or row.get("prefix_native_sha256") is not None):
            raise ValueError("Inner prefix cannot have outer validation evidence")
        evidence.append({**row, **update_evidence, **optimizer_evidence,
                         "full_training_state_components_equal": sorted(SIGNATURE_KEYS),
                         "prefix_state_sha256": [d["prefix_state_sha256"] for d in done]})
    return evidence


def validate_prefix_native(receipt: dict, fold: int, ids: np.ndarray, registry_hash: str,
                           root: Path, output: Path) -> dict:
    required = {"id": CAMPAIGN_ID, "status": "passed", "fold": fold, "registry_sha256": registry_hash,
                "verification_scope": "full_outer_fold", "class_order": [0, 1], "row_count": len(ids),
                "ids_sha256": ids_sha256(ids), "epoch": 4, "reference": "control_A4", "candidate": "treatment_C4",
                "chunk_rows": 8191, "quality_metrics_computed": False, "atol": 2e-6, "rtol": 1e-5,
                "parity_passed": True}
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Full-fold prefix native verification has wrong scope or identity")
    if (not np.isfinite([receipt["max_absolute_error"], receipt["max_scaled_error"]]).all()
            or not 0 <= receipt["max_scaled_error"] <= 1 or not 0 <= receipt["max_absolute_error"] <= 1.2e-5):
        raise ValueError("Full-fold prefix native parity exceeds fixed tolerance")
    expected = set()
    for arm in ("A", "C"):
        folder = output / f"fold_{fold}/outer/{arm}"
        expected.update(path.relative_to(root).as_posix() for path in
                        (folder / "transform.json", folder / "epoch_004/metadata.json", folder / "epoch_004/graph.pt"))
    if set(receipt["artifact_hashes"]) != expected:
        raise ValueError("Full-fold prefix check must bind both exact native models and transforms")
    for relative, digest in receipt["artifact_hashes"].items():
        require_hash(checked_path(root, relative, inside=output), digest)
    return {"fold": fold, "row_count": len(ids), "max_absolute_error": receipt["max_absolute_error"],
            "execution": "cloud full-fold A4/C4 reload; local receipt/hash verification only"}


def validate_provider_binding(protocol: dict, canonical_root: Path) -> dict:
    folder = f"cloud/continuation_six_v1/{CAMPAIGN_ID}/experiment"
    binding = protocol["provider_identity"]
    if binding["path"] != folder + "/provider_identity.json":
        raise ValueError("Provider evidence belongs to another stage")
    identity_path = checked_path(canonical_root, binding["path"])
    require_hash(identity_path, binding["sha256"])
    identity = read_json(identity_path)
    preparation_path = checked_path(canonical_root, protocol["preparation_path"])
    require_hash(preparation_path, protocol["preparation_sha256"])
    preparation = read_json(preparation_path)
    expected = {"status": "verified", "requested_kernel": preparation["kernel_id"], "version": 1,
                "private_verified": True, "runtime_sha256": preparation["runtime_sha256"],
                "plan_sha256": protocol["sequence_plan_sha256"]}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Private immutable provider identity differs")
    if type(identity["kernel_id"]) is not int or identity["kernel_id"] <= 0 or not isinstance(identity["actual_kernel"], str):
        raise ValueError("Missing actual provider identity")
    expected_files = {folder + "/provider_identity.json", folder + "/run.py", folder + "/push_intent.json",
                      folder + "/remote_metadata.json", folder + "/remote_source_api.py"}
    if identity["push_receipt_sha256"] is not None:
        expected_files.add(folder + "/push_receipt.json")
    else:
        recovery_path = identity.get("recovery_receipt_path")
        if not isinstance(recovery_path, str) or not recovery_path.startswith(folder + "/"):
            raise ValueError("Unknown push result lacks durable read-only reconciliation evidence")
        require_hash(checked_path(canonical_root, recovery_path), identity["recovery_receipt_sha256"])
        expected_files.add(recovery_path)
    if set(binding["files"]) != expected_files:
        raise ValueError("Provider binding omits or adds identity evidence")
    for relative, digest in binding["files"].items():
        if not relative.startswith(folder + "/"):
            raise ValueError("Provider evidence escapes its stage")
        require_hash(checked_path(canonical_root, relative), digest)
    for field, filename in (("runtime_sha256", "run.py"), ("push_intent_sha256", "push_intent.json"),
                            ("remote_metadata_sha256", "remote_metadata.json"), ("remote_source_sha256", "remote_source_api.py")):
        if identity[field] != binding["files"][folder + "/" + filename]:
            raise ValueError("Provider evidence hash does not bind its exact file")
    if identity["remote_source_sha256"] != preparation["runtime_sha256"]:
        raise ValueError("Provider source differs from reviewed rendered entry")
    if identity["push_receipt_sha256"] is not None and identity["push_receipt_sha256"] != binding["files"][folder + "/push_receipt.json"]:
        raise ValueError("Provider push receipt identity differs")
    registration = canonical_root / f"state/continuation_six_v1/registrations/{CAMPAIGN_ID}.json"
    require_hash(registration, identity["registration_sha256"])
    registered = read_json(registration)
    if registered["preparation_sha256"] != protocol["preparation_sha256"] or registered["plan_sha256"] != protocol["sequence_plan_sha256"]:
        raise ValueError("Provider used another prepared stage registration")
    return identity


def validate_local_protocol(path: Path, canonical_root: Path) -> dict:
    protocol = read_json(path)
    required = {"id": CAMPAIGN_ID, "status": "frozen_before_assessment", "evaluation": EVALUATION,
                "source_split_sha256": SPLIT_HASH, "incumbent_selection_sha256": INCUMBENT_HASH,
                "assessment_workspace": ASSESSMENT_WORKSPACE, "quality_metrics_read_when_frozen": False,
                "real_predictions_scored_when_frozen": False, "intervention": INTERVENTION,
                "trajectory_policies": {"A": policy_for("control"), "C": policy_for(POLICY_ID)},
                "sequence_plan_sha256": SEQUENCE_PLAN_SHA256, "sequence_authorization_sha256": SEQUENCE_AUTH_SHA256,
                "preparation_path": PREPARATION.relative_to(ROOT).as_posix()}
    if any(protocol.get(key) != expected for key, expected in required.items()):
        raise ValueError("Local assessment protocol differs from the fixed stage/scientific gate")
    expected_sources = {"scripts/evaluate_continuation_six_v1.py", "scripts/test_evaluate_continuation_six_v1.py",
                        "scripts/evaluate_fixed_epoch_dropout_v1.py", "scripts/test_evaluate_fixed_epoch_dropout_v1.py"}
    if set(protocol["source_sha256"]) != expected_sources or set(protocol["runtime_source_sha256"]) != set(RUNTIME_MODULES):
        raise ValueError("Local protocol source inventory differs")
    for name, digest in protocol["source_sha256"].items():
        require_hash(checked_path(canonical_root, name), digest)
    proof_binding = protocol["generated_verification"]
    proof_path = checked_path(canonical_root, proof_binding["path"])
    require_hash(proof_path, proof_binding["sha256"])
    proof = read_json(proof_path)
    proof_sources = proof.get("source_hashes", proof.get("source_sha256", {}))
    if proof.get("status") != "passed" or proof.get("returncode", proof.get("exit_code", 0)) != 0:
        raise ValueError("Bound evaluator verification did not pass")
    for name in ("scripts/evaluate_continuation_six_v1.py", "scripts/test_evaluate_continuation_six_v1.py"):
        if proof_sources.get(name) != protocol["source_sha256"][name]:
            raise ValueError("Bound generated proof uses other evaluator sources")
    require_hash(canonical_root / "configs/continuation_six_v1.json", SEQUENCE_PLAN_SHA256)
    require_hash(canonical_root / "state/continuation_six_v1/authorization.json", SEQUENCE_AUTH_SHA256)
    validate_provider_binding(protocol, canonical_root)
    return protocol


def validate_execution_amendment(root: Path, output: Path, registry: dict, protocol: dict,
                                 canonical_root: Path, torch_version: str) -> dict:
    """Check the registered symmetric strict runtime in all fourteen processes."""
    registration_path = root / "provenance/execution_policy.json"
    registration_hash = require_hash(registration_path, registry["execution_policy_sha256"])
    require_hash(output / "runtime_amendment/registration.json", registration_hash)
    hook_name = "scripts/continuation_six_deterministic_v1.py"
    hook_hash = registry["source_hashes"][hook_name]
    require_hash(output / "runtime_amendment/sitecustomize.py", hook_hash)
    amendment = read_json(registration_path)
    policy = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "torch_deterministic_algorithms": True,
              "torch_deterministic_warn_only": False, "cudnn_benchmark": False, "cudnn_deterministic": True}
    expected = {"id": CAMPAIGN_ID + "_strict_execution", "campaign": CAMPAIGN_ID,
        "sequence_plan_sha256": SEQUENCE_PLAN_SHA256, "sequence_authorization_sha256": SEQUENCE_AUTH_SHA256,
        "hook_source_sha256": hook_hash, "execution_policy": policy, "matmul_allow_tf32": False,
        "startup_failure_exit_code": 86, "required_startup_roles": 14}
    if any(amendment.get(key) != value for key, value in expected.items()):
        raise ValueError("Runtime execution policy differs from the preregistered symmetric policy")
    runtime = read_json(output / "job_runtime.json")
    if any(runtime.get(key) != value for key, value in {
            "hard_timeout_seconds": 7200, "registry_sha256": protocol["registry_sha256"],
            "bundle_manifest_sha256": protocol["bundle_manifest_sha256"]}.items()):
        raise ValueError("Runtime budget or scientific anchors differ")
    start = datetime.fromisoformat(runtime["started_utc"])
    if start.tzinfo is None or any((datetime.fromisoformat(runtime[key]) - start).total_seconds() != seconds
            for key, seconds in (("fit_deadline_utc", 6300), ("delivery_deadline_utc", 7140))):
        raise ValueError("Runtime deadlines differ from the registered execution policy")
    launch = read_json(root / f"state/{CAMPAIGN_ID}/launch_receipt.json")
    expected_launches = [{"role": "smoke", "device": "cuda", "output": f"artifacts/{CAMPAIGN_ID}/smoke_cuda"},
                        {"role": "controller", "campaign": f"configs/{CAMPAIGN_ID}.json"}]
    expected_launches.extend({"role": "worker", "campaign": f"configs/{CAMPAIGN_ID}.json",
        "phase": phase, "fold": fold, "trajectory": arm}
        for phase in ("inner", "outer") for fold in range(3) for arm in ("A", "C"))
    files = sorted((output / "runtime_amendment/processes").iterdir())
    if len(files) != 14 or any(not path.is_file() or path.suffix != ".json" for path in files):
        raise ValueError("Exactly fourteen deterministic startup receipts are required")
    identities, pids, evidence, cuda_builds = [], set(), [], set()
    for path in files:
        value = read_json(path)
        required = {"schema_version": 1, "status": "passed", "policy": "strict_deterministic_execution_v1",
            "hook_sha256": hook_hash, "amendment_sha256": registration_hash, "flags": EXECUTION_FLAGS,
            "prefix_gate_unchanged": True, "model_recipe_unchanged": False, "paired_execution_policy": True,
            "torch_version": torch_version}
        if any(value.get(key) != expected for key, expected in required.items()):
            raise ValueError("A process lacks the exact deterministic startup policy")
        pid, identity = value["pid"], value["launch"]
        if type(pid) is not int or pid < 1 or pid in pids or path.name != f"process_{pid}.json":
            raise ValueError("Duplicate or unbound deterministic process identity")
        if identity not in expected_launches or identity in identities:
            raise ValueError("Startup role/context is missing, duplicated or unregistered")
        target = "scripts/" + ("smoke_continuation_six_v1.py" if identity["role"] == "smoke" else "run_continuation_six_v1.py")
        if value["target"] != target or value["target_sha256"] != registry["source_hashes"][target]:
            raise ValueError("Startup receipt did not configure the frozen scientific entry")
        if identity["role"] == "controller" and launch["controller"]["pid"] != pid:
            raise ValueError("Startup controller differs from the actual campaign controller")
        if not isinstance(value["cuda_build_version"], str) or not value["cuda_build_version"]:
            raise ValueError("Cloud startup requires a CUDA Torch build")
        pids.add(pid); identities.append(identity); cuda_builds.add(value["cuda_build_version"])
        evidence.append({"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "pid": pid, "launch": identity})
    if len(cuda_builds) != 1:
        raise ValueError("Cloud startup processes used different CUDA builds")
    return {"runtime_amendment_sha256": registration_hash, "hook_sha256": hook_hash,
            "policy": EXECUTION_FLAGS, "processes": evidence, "all_fourteen_processes_verified": True,
            "scope": "Strict startup flags and frozen source/role coverage; prefix and applied update evidence checked separately"}


def prepare_campaign(campaign_path: Path, *, root: Path, canonical_root: Path = ROOT,
                     protocol_path: Path | None = None, preparation_path: Path | None = None) -> dict:
    """Perform all completion, provenance and row checks before any metric call."""
    root, canonical_root = root.resolve(), canonical_root.resolve()
    campaign_path = campaign_path.resolve()
    if not campaign_path.is_relative_to(root.resolve()):
        raise ValueError("Campaign wrapper must be in workspace")
    campaign = read_json(campaign_path)
    configure_stage(campaign.get("id"))
    if (campaign["id"] != CAMPAIGN_ID or campaign.get("source_split_sha256") != SPLIT_HASH
            or campaign["incumbent_selection_sha256"] != INCUMBENT_HASH):
        raise ValueError("Campaign must preserve fixed split and incumbent")
    output = checked_path(root, campaign["output_dir"])
    if output != (root / "artifacts" / CAMPAIGN_ID).resolve():
        raise ValueError("Only isolated cloud campaign output is allowed")
    if (output / "evaluation.json").exists() or (output / "evaluation_claim.json").exists():
        raise FileExistsError("Evaluation has already been claimed or completed")
    protocol_path = protocol_path or canonical_root / f"artifacts/{CAMPAIGN_ID}/local_evaluation_protocol.json"
    protocol = validate_local_protocol(protocol_path, canonical_root)
    preparation_path = preparation_path or canonical_root / PREPARATION.relative_to(ROOT)
    require_hash(preparation_path, protocol["preparation_sha256"])
    preparation = read_json(preparation_path)
    registry_path = checked_path(root, campaign["registry_path"])
    if registry_path != root / "registry.json":
        raise ValueError("Cloud wrapper must preserve payload-root registry path")
    registry_hash = require_hash(registry_path, campaign["registry_sha256"])
    registry = read_json(registry_path)
    if (registry["id"] != CAMPAIGN_ID or registry["expected_fit_count"] != 12
            or registry["evaluation"] != EVALUATION or registry["development_rows"] != DEVELOPMENT_ROWS
            or registry["source_split_sha256"] != SPLIT_HASH or registry["incumbent_selection_sha256"] != INCUMBENT_HASH
            or registry.get("audit_rows_exported") != 0 or registry.get("test_rows_exported") != 0
            or registry.get("cloud_quality_scoring") is not False):
        raise ValueError("Registered fixed evaluation rule differs")
    if any(registry.get(key) != value for key, value in INTERVENTION.items()):
        raise ValueError("Registered continuation boundary or endpoint mapping differs")
    if registry.get("trajectory_policies") != {"A": policy_for("control"), "C": policy_for(POLICY_ID)}:
        raise ValueError("Stage policy differs from its fixed sequence position")
    if (registry.get("continuation_policy") != policy_for(POLICY_ID)
            or registry.get("sequence_plan_sha256") != SEQUENCE_PLAN_SHA256
            or registry.get("authorization_sha256") != SEQUENCE_AUTH_SHA256):
        raise ValueError("Registry differs from frozen sequence authority/policy")
    sources = registry["source_hashes"]
    if not set(ADAPTER_WORKSPACE_SOURCES).issubset(sources) or "scripts/run_continuation_six_v1.py" not in sources:
        raise ValueError("Registry must bind cloud adapter, feature and worker sources")
    if (preparation["id"] != CAMPAIGN_ID or preparation["registry_sha256"] != registry_hash
            or protocol["registry_sha256"] != registry_hash
            or protocol["bundle_manifest_sha256"] != preparation["manifest_sha256"]):
        raise ValueError("Downloaded registry differs from locally prepared cloud package")
    manifest_bundle_path = root / "bundle-manifest.json"
    require_hash(manifest_bundle_path, preparation["manifest_sha256"])
    bundle = read_json(manifest_bundle_path)
    if bundle["id"] != CAMPAIGN_ID or bundle["registry_sha256"] != registry_hash:
        raise ValueError("Bundle manifest identity differs")
    payload_inventory = set(sources) | set(registry["input_hashes"]) | {"registry.json", f"configs/{CAMPAIGN_ID}.json"}
    if set(bundle["files"]) != payload_inventory:
        raise ValueError("Payload inventory differs from registered source/input allowlist")
    for relative, item in bundle["files"].items():
        path = checked_path(root, relative)
        require_hash(path, item["sha256"])
        if item["bytes"] != path.stat().st_size:
            raise ValueError("Payload file byte count differs")
    for hashes in (sources, registry["input_hashes"]):
        for relative, expected in hashes.items():
            require_hash(checked_path(root, relative), expected)
    for required in ("data/train.parquet", "data/original_aux_predictions.parquet", "data/splits.parquet", "data/manifest.json"):
        if required not in registry["input_hashes"]:
            raise ValueError("Missing cloud input hash")
    return_inventory = validate_return_manifest(root)
    wrapper_hash = sha256(campaign_path)
    completion_path = checked_path(root, campaign["completion_receipt_path"], inside=output)
    completion = read_json(completion_path)
    bindings = {"id": CAMPAIGN_ID, "campaign_sha256": wrapper_hash, "registry_sha256": registry_hash}
    complete_fields = {**bindings, "status": "completed", "completed_fit_count": 12,
                       "completed_endpoint_count": 9, "completed_prefix_count": 6,
                       "completed_prefix_native_count": 3, "evaluation_ready": True}
    if any(completion.get(key) != value for key, value in complete_fields.items()):
        raise ValueError("Campaign is not fully completed and ready for one-time evaluation")
    state_path = root / f"state/{CAMPAIGN_ID}/run_state.json"
    state = read_json(state_path)
    if any(state.get(key) != value for key, value in {**bindings, "status": "training_complete",
            "completed_fits": 12, "active_child": None, "outer_metrics_computed": False}.items()):
        raise ValueError("Cloud controller is not completely finished without quality scoring")
    manifest_path = checked_path(root, campaign["completed_manifest_path"], inside=output)
    manifest_hash = require_hash(manifest_path, completion["completed_manifest_sha256"])
    manifest = read_json(manifest_path)
    if any(manifest.get(key) != value for key, value in bindings.items()):
        raise ValueError("Completed manifest is not bound to this campaign and registry")
    trajectories, endpoints = validate_manifest_inventory(manifest)
    trajectory_evidence = []
    trajectory_records = {}
    runtime_evidence = None
    for row in trajectories:
        receipt_path = checked_path(root, row["receipt_path"], inside=output)
        require_hash(receipt_path, row["receipt_sha256"])
        receipt = read_json(receipt_path)
        horizon = 16
        required = {"id": CAMPAIGN_ID, "status": "completed", "phase": row["phase"],
                    "fold": row["fold"], "trajectory": row["trajectory"], "registry_sha256": registry_hash,
                    "schedule_horizon_epochs": horizon, "executed_epochs": horizon,
                    "checkpoint_epochs": [4, 16]}
        if any(receipt.get(key) != value for key, value in required.items()):
            raise ValueError("Incomplete or mismatched trajectory receipt")
        telemetry = checked_path(root, receipt["telemetry_path"], inside=output)
        require_hash(telemetry, receipt["telemetry_sha256"])
        partition_path = checked_path(root, receipt["partition_path"], inside=output)
        require_hash(partition_path, receipt["partition_sha256"])
        adapter_path = checked_path(root, receipt["adapter_receipt_path"], inside=output)
        require_hash(adapter_path, receipt["adapter_receipt_sha256"])
        trajectory_dir = output / f"fold_{row['fold']}/{row['phase']}/{row['trajectory']}"
        if adapter_path != (trajectory_dir / "trajectory.json").resolve() or partition_path != (trajectory_dir / "partitions.npz").resolve():
            raise ValueError("Adapter or partition receipt points to wrong trajectory")
        adapter = read_json(adapter_path)
        expected_continuation = policy_for("control" if row["trajectory"] == "A" else POLICY_ID)
        corrected = expected_continuation["id"] == "decay_correction"
        if (adapter.get("status") != "complete" or adapter.get("horizon") != horizon
                or adapter.get("executed_epochs") != horizon
                or adapter.get("endpoint_epochs") != required["checkpoint_epochs"]
                or adapter.get("clock") != "epoch_fraction" or adapter.get("legacy_decay_semantics") is not (not corrected)
                or adapter.get("curves_sha256") != receipt["telemetry_sha256"]
                or adapter.get("artifact_path_base") != "runtime_workspace"
                or set(adapter["endpoints"]) != {str(epoch) for epoch in required["checkpoint_epochs"]}):
            raise ValueError("Adapter receipt does not prove the registered completed trajectory")
        if (adapter.get("continuation_policy") != expected_continuation
                or adapter.get("decay_policy_prefix") != "legacy_double"
                or adapter.get("decay_policy_continuation") != ("single" if corrected else "legacy_double")):
            raise ValueError("Adapter decay/policy semantics differ from registered continuation")
        for key, expected in {"campaign": CAMPAIGN_ID, "fold": row["fold"], "phase": row["phase"],
                              "trajectory": row["trajectory"], "partition_sha256": receipt["partition_sha256"]}.items():
            if adapter["context"].get(key) != expected:
                raise ValueError("Adapter context differs from trajectory identity")
        for epoch, endpoint in adapter["endpoints"].items():
            endpoint_dir = trajectory_dir / f"epoch_{int(epoch):03d}"
            if checked_path(root, endpoint["path"], inside=output) != endpoint_dir.resolve():
                raise ValueError("Adapter endpoint path differs")
            require_hash(endpoint_dir / "metadata.json", endpoint["metadata_sha256"])
            require_hash(endpoint_dir / "graph.pt", endpoint["graph_sha256"])
        if (checked_path(root, adapter["curves_path"], inside=output) != telemetry
                or checked_path(root, adapter["trajectory_path"], inside=output) != adapter_path):
            raise ValueError("Portable adapter receipt paths differ")
        environment = validate_runtime_sources(adapter, root, registry, protocol)
        if runtime_evidence is None:
            runtime_evidence = environment
        elif runtime_evidence != environment:
            raise ValueError("Cloud trajectories used different source or package versions")
        trajectory_evidence.append({**row, "telemetry_sha256": receipt["telemetry_sha256"],
                                    "partition_path": receipt["partition_path"], "partition_sha256": receipt["partition_sha256"],
                                    "adapter_receipt_sha256": receipt["adapter_receipt_sha256"]})
        trajectory_records[(row["phase"], row["fold"], row["trajectory"])] = {"done": receipt, "adapter": adapter}
    execution_evidence = validate_execution_amendment(root, output, registry, protocol, canonical_root,
                                                     runtime_evidence["versions"]["torch"])
    prefix_evidence = validate_prefix_pairs(manifest["prefix_records"], trajectory_records, root, output)
    # Verify all nine file receipts before opening any prediction table.
    for endpoint in endpoints:
        path = checked_path(root, endpoint["prediction_path"], inside=output)
        expected_path = output / f"fold_{endpoint['fold']}/outer/predictions_{endpoint['arm']}.parquet"
        if path != expected_path.resolve():
            raise ValueError("Unexpected outer endpoint prediction path")
        require_hash(path, endpoint["prediction_sha256"])
        native_path = checked_path(root, endpoint["native_receipt_path"], inside=output)
        require_hash(native_path, endpoint["native_receipt_sha256"])
    split_path = checked_path(root, campaign["split_path"])
    if split_path != root / "data/splits.parquet":
        raise ValueError("Cloud split path differs from payload contract")
    require_hash(split_path, campaign["split_sha256"])
    canonical_split = canonical_root / "data/splits.parquet"
    require_hash(canonical_split, SPLIT_HASH)
    split = cloud_development_split(pd.read_parquet(split_path, columns=["id", "fold"]),
        pd.read_parquet(canonical_split, columns=["id", "fold"]), DEVELOPMENT_ROWS)
    ids, folds = split.id.to_numpy(), split.fold.to_numpy()
    partition_evidence = []
    for row in trajectory_evidence:
        with np.load(checked_path(root, row["partition_path"], inside=output), allow_pickle=False) as partition:
            checked = validate_partition(dict(partition), ids, folds, row["fold"], row["phase"])
        partition_evidence.append({"trajectory": row["trajectory"], **checked})
    prefix_native_evidence = []
    for row in manifest["prefix_records"]:
        if row["phase"] == "outer":
            path = output / f"fold_{row['fold']}/outer/prefix_native_verify.json"
            native = bound_json(root, row, "prefix_native", path, output)
            prefix_native_evidence.append(validate_prefix_native(native, row["fold"], ids[folds == row["fold"]],
                                                                 registry_hash, root, output))
    native_evidence, native_receipts = [], {}
    for endpoint in endpoints:
        native = read_json(checked_path(root, endpoint["native_receipt_path"], inside=output))
        native_evidence.append(validate_native_receipt(native, endpoint, ids[folds == endpoint["fold"]], registry_hash, root, output))
        native_receipts[(endpoint["fold"], endpoint["arm"])] = native
    selection_path = canonical_root / "artifacts/third_pass/blend/frozen.json"
    require_hash(selection_path, INCUMBENT_HASH)
    selection = read_json(selection_path)
    y, incumbent, incumbent_evidence = load_incumbent(selection, ids, folds, canonical_root)
    arms = {arm: np.full(len(ids), np.nan) for arm in ("A", "B", "C")}
    for endpoint in endpoints:
        mask = folds == endpoint["fold"]
        native = native_receipts[(endpoint["fold"], endpoint["arm"])]
        frame = pd.read_parquet(checked_path(root, endpoint["prediction_path"], inside=output))
        arms[endpoint["arm"]][mask], _ = aligned_oof(frame, ids[mask], folds[mask], y[mask], class_order=native["class_order"])
    if any(not np.isfinite(score).all() for score in arms.values()):
        raise ValueError("Incomplete assembled outer OOF vector")
    return {"output": output, "y": y, "folds": folds, "incumbent": incumbent, "arms": arms,
            "selection": selection, "campaign_sha256": wrapper_hash, "registry_sha256": registry_hash,
            "manifest_sha256": manifest_hash, "completion_sha256": sha256(completion_path),
            "source_hashes": sources, "ids_sha256": ids_sha256(ids),
            "trajectory_evidence": trajectory_evidence, "native_evidence": native_evidence,
            "partition_evidence": partition_evidence, "prefix_evidence": prefix_evidence,
            "prefix_native_evidence": prefix_native_evidence, "intervention": INTERVENTION,
            "incumbent_evidence": incumbent_evidence, "endpoint_manifest": endpoints,
            "local_protocol_sha256": sha256(protocol_path), "preparation_sha256": sha256(preparation_path),
            "return_inventory": return_inventory, "runtime_evidence": runtime_evidence,
            "execution_evidence": execution_evidence,
            "canonical_split_sha256": SPLIT_HASH, "cloud_split_sha256": campaign["split_sha256"],
            "canonical_development_rows_exactly_matched": True, "raw_cloud_train_aux_decoded": False,
            "assessment_workspace": str(root), "canonical_project_root": str(canonical_root)}


def metrics(y: np.ndarray, score: np.ndarray, folds: np.ndarray) -> dict:
    values = [float(roc_auc_score(y[folds == fold], score[folds == fold])) for fold in (0, 1, 2)]
    return {"pooled_auc": float(roc_auc_score(y, score)), "fold_auc": values,
            "mean_fold_auc": float(np.mean(values)), "log_loss": float(log_loss(y, score, labels=[0, 1])),
            "brier": float(brier_score_loss(y, score))}


def contrast(left: dict, right: dict) -> dict:
    return {"pooled_auc_delta": left["pooled_auc"] - right["pooled_auc"],
            "mean_fold_auc_delta": left["mean_fold_auc"] - right["mean_fold_auc"],
            "fold_auc_delta": (np.asarray(left["fold_auc"]) - right["fold_auc"]).tolist()}


def advancement_gate(summaries: dict[str, dict]) -> dict:
    """Require every registered C-mixture comparison, not a chosen subset."""
    comparisons = []
    for name in ("incumbent", "mixture_A", "mixture_B"):
        difference = contrast(summaries["mixture_C"], summaries[name])
        numbers = [difference["pooled_auc_delta"], difference["mean_fold_auc_delta"], *difference["fold_auc_delta"]]
        if len(difference["fold_auc_delta"]) != 3 or not np.isfinite(numbers).all():
            raise ValueError("Gate needs finite metrics for all three original folds")
        checks = {"pooled_gain_pass": difference["pooled_auc_delta"] >= EVALUATION["min_pooled_gain"],
                  "macro_gain_pass": difference["mean_fold_auc_delta"] >= EVALUATION["min_macro_gain"],
                  "fold_regression_pass": min(difference["fold_auc_delta"]) >= -EVALUATION["max_fold_regression"]}
        comparisons.append({"reference": name, "candidate": "mixture_C", **difference,
                            **checks, "passed": all(checks.values())})
    return {"advance_to_confirmation": all(row["passed"] for row in comparisons),
            "comparisons": comparisons, "rule": EVALUATION, "interpretation": LIMITATION,
            "release_promoted": False, "submission_authorized": False}


def score_fixed(y: np.ndarray, folds: np.ndarray, incumbent: np.ndarray,
                arms: dict[str, np.ndarray]) -> dict:
    if set(arms) != {"A", "B", "C"}:
        raise ValueError("Exactly fixed arms A/B/C are required")
    y, folds, incumbent = np.asarray(y), np.asarray(folds), np.asarray(incumbent)
    if y.ndim != 1 or folds.shape != y.shape or incumbent.shape != y.shape or set(folds) != {0, 1, 2}:
        raise ValueError("Score arrays must align to the three development folds")
    vectors = {"incumbent": incumbent, **arms}
    if any(np.asarray(value).shape != y.shape or not np.isfinite(value).all()
           or ((np.asarray(value) < 0) | (np.asarray(value) > 1)).any() for value in vectors.values()):
        raise ValueError("Aligned finite probability vectors required")
    if not np.isin(y, [0, 1]).all() or any(len(np.unique(y[folds == fold])) != 2 for fold in (0, 1, 2)):
        raise ValueError("Each development fold must contain both binary classes")
    for arm in ("A", "B", "C"):
        vectors[f"mixture_{arm}"] = .9 * incumbent + .1 * arms[arm]
    summaries = {name: metrics(y, np.asarray(value), folds) for name, value in vectors.items()}
    comparisons = {}
    for prefix in ("", "mixture_"):
        for left, right in (("B", "A"), ("C", "B"), ("C", "A")):
            comparisons[f"{prefix}{left}_minus_{prefix}{right}"] = contrast(summaries[prefix + left], summaries[prefix + right])
    for arm in ("A", "B", "C"):
        comparisons[f"mixture_{arm}_minus_incumbent"] = contrast(summaries[f"mixture_{arm}"], summaries["incumbent"])
    return {"metrics": summaries, "contrasts": comparisons, "advancement": advancement_gate(summaries),
            "mixture_alpha": .1, "weight_search_performed": False, "interpretation": LIMITATION}


def claim_evaluation(prepared: dict, invocation: list[str]) -> dict:
    """Create the exclusive durable scoring claim only after preparation passes."""
    claim = {"claimed_utc": datetime.now(timezone.utc).isoformat(),
             "campaign_sha256": prepared["campaign_sha256"], "registry_sha256": prepared["registry_sha256"],
             "manifest_sha256": prepared["manifest_sha256"], "local_protocol_sha256": prepared["local_protocol_sha256"],
             "invocation": invocation}
    with (prepared["output"] / "evaluation_claim.json").open("x", encoding="utf-8") as stream:
        json.dump(claim, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return claim


def freeze_protocol(stage: str, verification_receipt: Path, *, canonical_root: Path = ROOT) -> dict:
    """Bind reviewed sources and provider readback without reading predictions."""
    configure_stage(stage)
    canonical_root = canonical_root.resolve()
    destination = canonical_root / f"artifacts/{stage}/local_evaluation_protocol.json"
    if destination.exists():
        raise FileExistsError("Assessment protocol is already frozen")
    assessment_output = canonical_root / ASSESSMENT_WORKSPACE / "artifacts" / stage
    if any((assessment_output / name).exists() for name in ("evaluation.json", "evaluation_claim.json")):
        raise FileExistsError("Cannot preregister after assessment was claimed")
    source_names = ("evaluate_continuation_six_v1.py", "test_evaluate_continuation_six_v1.py",
                    "evaluate_fixed_epoch_dropout_v1.py", "test_evaluate_fixed_epoch_dropout_v1.py")
    sources = {"scripts/" + name: sha256(canonical_root / "scripts" / name) for name in source_names}
    proof = read_json(verification_receipt)
    proof_sources = proof.get("source_hashes", proof.get("source_sha256", {}))
    if proof.get("status") != "passed" or proof.get("returncode", proof.get("exit_code", 0)) != 0:
        raise ValueError("Evaluator generated verification did not pass")
    for name in source_names[:2]:
        key = "scripts/" + name
        if proof_sources.get(key) != sources[key]:
            raise ValueError("Evaluator source changed after generated verification")
    require_hash(canonical_root / "configs/continuation_six_v1.json", SEQUENCE_PLAN_SHA256)
    require_hash(canonical_root / "state/continuation_six_v1/authorization.json", SEQUENCE_AUTH_SHA256)
    preparation_path = canonical_root / PREPARATION.relative_to(ROOT)
    preparation = read_json(preparation_path)
    folder = preparation_path.parent
    registry_path = folder / "bundle/payload/registry.json"
    require_hash(registry_path, preparation["registry_sha256"])
    registry = read_json(registry_path)
    if (preparation["id"] != stage or registry["trajectory_policies"] != {"A": policy_for("control"), "C": policy_for(POLICY_ID)}
            or registry["evaluation"] != EVALUATION):
        raise ValueError("Prepared stage differs from fixed sequence policy/gate")
    require_hash(folder / "bundle/payload/bundle-manifest.json", preparation["manifest_sha256"])
    identity_path = folder / "provider_identity.json"
    identity = read_json(identity_path)
    provider_files = {identity_path, folder / "run.py", folder / "push_intent.json",
                      folder / "remote_metadata.json", folder / "remote_source_api.py"}
    if identity["push_receipt_sha256"] is not None:
        provider_files.add(folder / "push_receipt.json")
    else:
        recovery_path = identity.get("recovery_receipt_path")
        if not recovery_path:
            raise ValueError("Unknown push result requires a bound recovery receipt")
        recovery = checked_path(canonical_root, recovery_path)
        require_hash(recovery, identity["recovery_receipt_sha256"])
        provider_files.add(recovery)
    runtime_sources = {}
    for module in RUNTIME_MODULES:
        source = canonical_root / ".venv/Lib/site-packages" / (module.replace(".", "/") + ".py")
        runtime_sources[module] = sha256(source)
    protocol = {"id": stage, "status": "frozen_before_assessment", "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": EVALUATION, "intervention": INTERVENTION,
        "trajectory_policies": {"A": policy_for("control"), "C": policy_for(POLICY_ID)},
        "source_split_sha256": SPLIT_HASH, "incumbent_selection_sha256": INCUMBENT_HASH,
        "sequence_plan_sha256": SEQUENCE_PLAN_SHA256, "sequence_authorization_sha256": SEQUENCE_AUTH_SHA256,
        "assessment_workspace": ASSESSMENT_WORKSPACE, "quality_metrics_read_when_frozen": False,
        "real_predictions_scored_when_frozen": False, "source_sha256": sources,
        "runtime_source_sha256": runtime_sources,
        "preparation_path": preparation_path.relative_to(canonical_root).as_posix(),
        "preparation_sha256": sha256(preparation_path), "registry_sha256": preparation["registry_sha256"],
        "bundle_manifest_sha256": preparation["manifest_sha256"],
        "generated_verification": {"path": verification_receipt.resolve().relative_to(canonical_root).as_posix(),
                                   "sha256": sha256(verification_receipt)},
        "provider_identity": {"path": identity_path.relative_to(canonical_root).as_posix(), "sha256": sha256(identity_path),
                              "files": {path.relative_to(canonical_root).as_posix(): sha256(path) for path in provider_files}},
        "limits": "No metrics, prediction tables, models, raw features, audit or test labels read by protocol freezer."}
    validate_provider_binding(protocol, canonical_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(destination), "sha256": sha256(destination), "stage": stage,
            "status": "frozen_before_assessment", "quality_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--campaign")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--verification-receipt", type=Path)
    args = parser.parse_args()
    if args.freeze_protocol:
        if not args.stage or args.verification_receipt is None or args.workspace is not None:
            parser.error("Protocol freeze needs --stage and --verification-receipt, without --workspace")
        print(json.dumps(freeze_protocol(args.stage, args.verification_receipt.resolve()), indent=2))
        return
    if args.workspace is None or args.protocol is None:
        parser.error("Assessment requires --workspace and --protocol")
    configure_stage(read_json(args.protocol)["id"])
    with threadpool_limits(limits=2):
        workspace = args.workspace.resolve()
        allowed = ROOT / ASSESSMENT_WORKSPACE
        if workspace != allowed.resolve():
            raise ValueError("Use the isolated registered assessment workspace")
        campaign_path = checked_path(workspace, args.campaign or f"configs/{CAMPAIGN_ID}.json")
        prepared = prepare_campaign(campaign_path, root=workspace, protocol_path=args.protocol.resolve())
        claim = claim_evaluation(prepared, [sys.executable, *sys.argv])
        result = score_fixed(prepared["y"], prepared["folds"], prepared["incumbent"], prepared["arms"])
        reference = result["metrics"]["incumbent"]
        selection = prepared["selection"]
        if (abs(reference["pooled_auc"] - selection["oof_auc"]) > 2e-12
                or not np.allclose(reference["fold_auc"], selection["fold_auc"], atol=2e-12, rtol=0)):
            raise ValueError("Frozen incumbent metrics cannot be reproduced; claim retained for investigation")
    excluded = {"output", "y", "folds", "incumbent", "arms", "selection"}
    result.update({"id": CAMPAIGN_ID, "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "continuation_policy": policy_for(POLICY_ID), "sequence_plan_sha256": SEQUENCE_PLAN_SHA256,
                   "provenance": {key: value for key, value in prepared.items() if key not in excluded},
                   "claim": claim, "script_sha256": sha256(Path(__file__)),
                   "test_script_sha256": sha256(Path(__file__).with_name("test_evaluate_continuation_six_v1.py")),
                   "row_count": len(prepared["y"]), "class_order": [0, 1],
                   "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")},
                   "raw_train_audit_test_read": False, "native_inference_executed_by_evaluator": False,
                   "native_verification_execution_location": "cloud worker; receipt checked locally",
                   "release_or_incumbent_modified": False})
    destination = prepared["output"] / "evaluation.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(destination), "advance_to_confirmation": result["advancement"]["advance_to_confirmation"],
                      "metrics": result["metrics"], "contrasts": result["contrasts"]}), flush=True)


if __name__ == "__main__":
    main()
