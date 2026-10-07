"""Exact predeclared continuation policies; importing this file never loads ML."""
from __future__ import annotations

import copy

POLICY_IDS = ("linear_lr", "decay_correction", "ema", "label_smoothing", "freeze_embeddings", "head_only")
EXTRA = {
    "control": {},
    "linear_lr": {"linear_lr_end_epoch": 16},
    "decay_correction": {"decay_factor_applications_after_epoch4": 1},
    "ema": {"ema_half_life_epochs": 1.0},
    "label_smoothing": {"label_smoothing_after_epoch4": 0.05},
    "freeze_embeddings": {"frozen_roles": ["numeric_embedding", "categorical_embedding", "learned_preprocessing"]},
    "head_only": {"trainable_roles_after_epoch4": ["final_affine"]},
}


def policy_for(name: str) -> dict:
    if name not in EXTRA:
        raise ValueError("Unregistered continuation policy")
    return copy.deepcopy({"id": name, "after_completed_epoch": 4, "horizon": 16,
                          "dropout_after_epoch4": 0.05, **EXTRA[name]})


def validate_policy(value: dict) -> dict:
    if not isinstance(value, dict) or value != policy_for(value.get("id")):
        raise ValueError("Continuation policy differs from the exact registration")
    return copy.deepcopy(value)


def policy_for_stage(stage: str) -> dict:
    stages = tuple(f"fixed_epoch_continuation_{index:02d}" for index in range(1, 7))
    if stage not in stages:
        raise ValueError("Unregistered continuation stage")
    return policy_for(POLICY_IDS[stages.index(stage)])
