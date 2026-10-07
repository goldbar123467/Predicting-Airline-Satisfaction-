"""Describe locally assessed cloud fixed-epoch receipts, without new scoring.

Writes evidence.json and inner_monitor_and_lr.png to a fresh report directory.
All quality metrics and contrasts come from existing evaluation/curve receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import statistics

from summarize_fixed_epoch_v1 import (
    _curves, _finite_json, _invalid_constant, _path, _same_fields, render_plot, sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "fixed_epoch_cloud_v1"
SHARED_REPORTER_SHA256 = "0af76fecaa546b5b798a4dd742d452ab0fd79bd78b9d7f39604304f0d80e9982"
PROTOCOL_SHA256 = "1fbf6c31da6d582555026c1078f8314c7e2d46b221de43e2838bc305adcc7652"
EVALUATOR_TEST_RECEIPT_SHA256 = "093030dac0615f3c6972cf7f7a0dabde61969f6acb4af5d64e4489a7ea902018"


def _campaign_config(root: Path, campaign_path: Path | None) -> tuple[Path, dict]:
    path = (campaign_path or root / f"configs/{CAMPAIGN_ID}.json").resolve()
    if not path.is_relative_to(root / "configs"):
        raise ValueError("Campaign wrapper must be under the workspace configs directory")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_invalid_constant)
    if not isinstance(value, dict) or value.get("id") != CAMPAIGN_ID:
        raise ValueError("Expected a registered fixed-epoch campaign wrapper")
    _finite_json(value)
    return path, value


def collect_completed(root: Path, campaign_path: Path | None = None, *, canonical_root: Path = ROOT) -> dict:
    """Read only JSON/JSONL receipts; root injection supports generated IO fixtures."""
    root = root.resolve()
    canonical_root = canonical_root.resolve()
    config_path, initial_config = _campaign_config(root, campaign_path)
    campaign = initial_config["id"]
    output = root / "artifacts" / campaign
    inputs = {}
    local_inputs = {}
    shared_source = Path(__file__).with_name("summarize_fixed_epoch_v1.py")
    if sha256(shared_source) != SHARED_REPORTER_SHA256:
        raise ValueError("Shared curve/plot helper changed since reporter verification")

    def load(path: Path, expected: str | None = None) -> dict:
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Hash mismatch: {path}")
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_invalid_constant)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path}")
        _finite_json(value)
        inputs[path.relative_to(root).as_posix()] = actual
        return value

    def load_local(relative: str, expected: str | None = None) -> dict:
        path = canonical_root / relative
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Local evaluator provenance changed: {relative}")
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_invalid_constant)
        if not isinstance(value, dict):
            raise ValueError("Expected local provenance JSON object")
        _finite_json(value)
        local_inputs[relative] = actual
        return value

    protocol = load_local(f"artifacts/{CAMPAIGN_ID}/local_evaluation_protocol.json", PROTOCOL_SHA256)
    _same_fields(protocol, {"id": CAMPAIGN_ID, "status": "frozen_before_assessment",
        "quality_metrics_read_when_frozen": False, "real_predictions_scored_when_frozen": False}, "local protocol")
    test_receipt = load_local(f"artifacts/{CAMPAIGN_ID}/evaluator_smoke/verification.json", EVALUATOR_TEST_RECEIPT_SHA256)
    _same_fields(test_receipt, {"id": CAMPAIGN_ID, "status": "passed", "tests_passed": 25,
        "local_protocol_sha256": PROTOCOL_SHA256, "real_predictions_scored": False,
        "models_loaded": False, "fitting_performed": False, "gpu_used": False,
        "source_sha256": protocol["source_sha256"]}, "evaluator test receipt")
    for relative, expected in protocol["source_sha256"].items():
        path = (canonical_root / relative).resolve()
        if not path.is_relative_to(canonical_root / "scripts") or sha256(path) != expected:
            raise ValueError("Pinned evaluator source changed")
        local_inputs[relative] = expected

    state = load(root / "state" / campaign / "run_state.json")
    _same_fields(state, {"id": campaign, "status": "training_complete", "completed_fits": 12,
                         "active_child": None, "outer_metrics_computed": False}, "cloud controller completion")
    config = load(config_path, state["campaign_sha256"])
    _same_fields(config, {"id": campaign, "output_dir": f"artifacts/{campaign}"}, "campaign")
    registry_path = _path(root, config["registry_path"], root / "registry.json")
    registry = load(registry_path, config["registry_sha256"])
    registry_hash = inputs[registry_path.relative_to(root).as_posix()]
    if registry_hash != protocol["registry_sha256"]:
        raise ValueError("Cloud registry differs from pinned evaluation protocol")
    _same_fields(registry, {"id": campaign, "expected_fit_count": 12}, "registry")
    bindings = {"id": campaign, "campaign_sha256": state["campaign_sha256"], "registry_sha256": registry_hash}
    _same_fields(state, bindings, "controller identity")
    completion_path = output / "completion_receipt.json"
    completion = load(completion_path)
    _same_fields(completion, {**bindings, "status": "completed", "completed_fit_count": 12,
                             "completed_endpoint_count": 9, "evaluation_ready": True}, "completion receipt")
    manifest_path = output / "completed_manifest.json"
    manifest = load(manifest_path, completion["completed_manifest_sha256"])
    _same_fields(manifest, bindings, "manifest identity")
    evaluation = load(output / "evaluation.json")
    _same_fields(evaluation, {"id": campaign, "weight_search_performed": False,
        "class_order": [0, 1], "release_or_incumbent_modified": False,
        "script_sha256": protocol["source_sha256"]["scripts/evaluate_fixed_epoch_cloud_v1.py"],
        "test_script_sha256": protocol["source_sha256"]["scripts/test_evaluate_fixed_epoch_cloud_v1.py"],
        "native_inference_executed_by_evaluator": False,
        "native_verification_execution_location": "cloud worker; receipt checked locally"}, "local evaluation")
    if not evaluation.get("completed_utc"):
        raise ValueError("Missing evaluation completion")
    _same_fields(evaluation["provenance"], {
        "campaign_sha256": state["campaign_sha256"], "registry_sha256": registry_hash,
        "manifest_sha256": completion["completed_manifest_sha256"],
        "completion_sha256": inputs[completion_path.relative_to(root).as_posix()],
        "local_protocol_sha256": PROTOCOL_SHA256, "preparation_sha256": protocol["preparation_sha256"],
        "canonical_split_sha256": protocol["source_split_sha256"],
        "cloud_split_sha256": config["split_sha256"], "canonical_development_rows_exactly_matched": True,
        "raw_cloud_train_aux_decoded": False, "source_hashes": registry["source_hashes"]}, "evaluation provenance")
    claim = load(output / "evaluation_claim.json")
    _same_fields(claim, {"campaign_sha256": state["campaign_sha256"], "registry_sha256": registry_hash,
        "manifest_sha256": completion["completed_manifest_sha256"], "local_protocol_sha256": PROTOCOL_SHA256}, "scoring claim")
    if claim != evaluation["claim"]:
        raise ValueError("Saved scoring claim differs from completed evaluation")
    expected = {(phase, fold, arm) for phase in ("inner", "outer") for fold in range(3) for arm in ("A", "C")}
    inventory = [(row["phase"], row["fold"], row["trajectory"]) for row in manifest["trajectories"]]
    endpoints = [(row["fold"], row["arm"]) for row in manifest["endpoints"]]
    if any(type(row["fold"]) is not int for row in [*manifest["trajectories"], *manifest["endpoints"]]):
        raise ValueError("Fold identity must be an integer")
    if len(inventory) != 12 or set(inventory) != expected:
        raise ValueError("Need exactly the twelve registered trajectories")
    if len(endpoints) != 9 or set(endpoints) != {(fold, arm) for fold in range(3) for arm in ("A", "B", "C")}:
        raise ValueError("Need exactly the nine registered outer endpoints")
    summaries = []
    for entry in sorted(manifest["trajectories"], key=lambda x: (x["phase"], x["fold"], x["trajectory"])):
        phase, fold, arm = entry["phase"], entry["fold"], entry["trajectory"]
        identity = {"campaign": campaign, "phase": phase, "fold": fold, "trajectory": arm}
        horizon, epochs = (4, [4]) if arm == "A" else (16, [4, 16])
        folder = output / f"fold_{fold}" / phase / arm
        done_path = _path(root, entry["receipt_path"], folder / "done.json")
        done = load(done_path, entry["receipt_sha256"])
        _same_fields(done, {"id": campaign, "status": "completed", "phase": phase, "fold": fold,
                           "trajectory": arm, "registry_sha256": registry_hash,
                           "schedule_horizon_epochs": horizon, "executed_epochs": horizon,
                           "checkpoint_epochs": epochs}, "trajectory done receipt")
        adapter_path = _path(root, done["adapter_receipt_path"], folder / "trajectory.json")
        receipt = load(adapter_path, done["adapter_receipt_sha256"])
        _same_fields(receipt, {"status": "complete", "horizon": horizon, "executed_epochs": horizon,
                              "endpoint_epochs": epochs, "clock": "epoch_fraction",
                              "legacy_decay_semantics": True}, "adapter receipt")
        _same_fields(receipt["context"], identity, "adapter context")
        curve_path = _path(root, done["telemetry_path"], folder / "curves.jsonl")
        curve_hash = sha256(curve_path)
        if curve_hash != done["telemetry_sha256"] or curve_hash != receipt["curves_sha256"]:
            raise ValueError("Curve hash differs from trajectory/done receipt")
        inputs[curve_path.relative_to(root).as_posix()] = curve_hash
        rows = _curves(curve_path, receipt, identity)
        compact_rows = [{key: row[key] for key in (
            "epoch", "updates", "elapsed_seconds", "objective_mean_per_member", "train_eval_probe", "monitor")}
            | {"first_update": {key: row["first_update_schedule"][key] for key in (
                "epoch_float", "fraction", "base_lr", "base_wd", "dropout")},
               "last_update": {key: row["last_update_schedule"][key] for key in (
                "epoch_float", "fraction", "base_lr", "base_wd", "dropout")}}
            for row in rows]
        summaries.append({"phase": phase, "fold": fold, "trajectory": arm, "horizon_epochs": horizon,
            "executed_epochs": horizon, "optimizer_updates": receipt["optimizer_updates"],
            "elapsed_seconds": receipt["elapsed_seconds"], "train_rows": receipt["train_rows"],
            "monitor_rows": receipt["monitor_rows"], "device": receipt["device"],
            "memory_end": receipt["memory_end"], "curve_rows": compact_rows,
            "literal_endpoints": [compact_rows[epoch - 1] for epoch in epochs],
            "initial_group_schedule": rows[0]["first_update_schedule"]["groups"],
            "final_group_schedule": rows[-1]["last_update_schedule"]["groups"]})
    aggregates = []
    for phase in ("inner", "outer"):
        for arm in ("A", "C"):
            selected = [row for row in summaries if row["phase"] == phase and row["trajectory"] == arm]
            aggregates.append({"phase": phase, "trajectory": arm, "fit_count": len(selected),
                "sum_elapsed_seconds": math.fsum(row["elapsed_seconds"] for row in selected),
                "mean_elapsed_seconds": statistics.fmean(row["elapsed_seconds"] for row in selected),
                "sum_optimizer_updates": sum(row["optimizer_updates"] for row in selected),
                "mean_optimizer_updates": statistics.fmean(row["optimizer_updates"] for row in selected)})
    return {"schema_version": 1, "id": campaign, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Descriptive receipt summary only; no data, labels, models or predictions read; no metrics scored, checkpoint selected, weight search or gate changed.",
        "source_root": str(root), "input_hashes": inputs, "canonical_provenance_hashes": local_inputs,
        "registered_source_hashes": registry["source_hashes"],
        "evaluation_copied_without_recomputation": {key: evaluation[key] for key in (
            "metrics", "contrasts", "advancement", "mixture_alpha", "interpretation", "completed_utc")},
        "trajectory_summaries": summaries, "descriptive_aggregates": aggregates,
        "interpretation": {
            "arms": {"A": "H4/E4", "B": "H16/E4: prefix of C", "C": "H16/E16"},
            "B_minus_A": "Effect of schedule horizon at the fixed epoch-4 endpoint, with matched initialization and training order.",
            "C_minus_B": "Continuation from epoch 4 to epoch 16 along the common H16 schedule, including its later learning-rate/regularization phases; not a schedule-independent effect of update count.",
            "native_verification": "Cloud worker performed full raw/native reload; the local evaluator checked downloaded receipts and artifact hashes without re-executing inference. This reporter performs neither inference nor scoring.",
            "monitor": "Passive inner validation AUC of averaged member probabilities, not outer-fold AUC; never used for stopping or endpoint selection.",
            "training_probe": "Fixed subset of fitting rows, evaluation mode; not the full training set and not held-out evidence.",
            "objective": "Training-mode criterion averaged over rows and members, with changing dropout; distinct from evaluation-mode log loss.",
            "schedule": "Recorded first/last optimization-update values per epoch; not interpolated update-level curves or integrated regularization doses.",
            "time": "Adapter elapsed time includes its telemetry and export work; excludes worker imports and later full-fold native reload. Sums and arithmetic means are descriptive.",
            "statistics": "No confidence interval, independent confirmation or statistical significance is inferred from the three folds."}}


def summarize(output_dir: Path, *, root: Path, campaign_path: Path | None = None,
              canonical_root: Path = ROOT) -> dict:
    root, output_dir = root.resolve(), output_dir.resolve()
    config_path, config = _campaign_config(root, campaign_path)
    report_root = root / "artifacts" / config["id"] / "report"
    if not output_dir.is_relative_to(report_root) or output_dir == report_root:
        raise ValueError(f"Choose a fresh child directory under {report_root}")
    if output_dir.exists():
        raise FileExistsError("Preserve existing report output; choose a fresh directory")
    evidence = collect_completed(root, config_path, canonical_root=canonical_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    plot = output_dir / "inner_monitor_and_lr.png"
    render_plot(evidence, plot)
    evidence["reporter"] = {"source_sha256": sha256(Path(__file__)),
                            "shared_helper_sha256": SHARED_REPORTER_SHA256,
                            "matplotlib_version": importlib.metadata.version("matplotlib"),
                            "plot_filename": plot.name, "plot_sha256": sha256(plot)}
    with (output_dir / "evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"evidence": str(output_dir / "evidence.json"), "plot": str(plot),
            "trajectory_count": len(evidence["trajectory_summaries"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--outputdir", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace != (ROOT / "cloud/fixed_epoch_v1/assessment_workspace").resolve():
        raise ValueError("Use the isolated cloud assessment workspace")
    output = args.outputdir if args.outputdir.is_absolute() else workspace / args.outputdir
    print(json.dumps(summarize(output, root=workspace), indent=2))


if __name__ == "__main__":
    main()
