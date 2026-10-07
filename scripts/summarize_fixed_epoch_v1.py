"""Describe completed fixed-epoch receipts without reading or scoring predictions.

Writes evidence.json and inner_monitor_and_lr.png to a fresh report directory.
All quality metrics and contrasts come from existing evaluation/curve receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "configs/fixed_epoch_v1.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _invalid_constant(value: str):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _finite_json(value) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON number")
    if isinstance(value, dict):
        for child in value.values():
            _finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _finite_json(child)


def _campaign_config(root: Path, campaign_path: Path | None) -> tuple[Path, dict]:
    path = (campaign_path or root / "configs/fixed_epoch_v1.json").resolve()
    if not path.is_relative_to(root / "configs"):
        raise ValueError("Campaign wrapper must be under the workspace configs directory")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_invalid_constant)
    if not isinstance(value, dict) or not re.fullmatch(r"fixed_epoch_[a-z0-9_]+", str(value.get("id", ""))):
        raise ValueError("Expected a registered fixed-epoch campaign wrapper")
    _finite_json(value)
    return path, value


def _number(value, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Expected finite number: {name}")
    if minimum is not None and value < minimum:
        raise ValueError(f"Value below {minimum}: {name}")
    return float(value)


def _same_fields(value: dict, required: dict, name: str) -> None:
    if any(key not in value or value[key] != expected for key, expected in required.items()):
        raise ValueError(f"Incomplete or mismatched {name}")


def _path(root: Path, relative: str, expected: Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or (root / candidate).resolve() != expected:
        raise ValueError(f"Unexpected receipt path: {relative}")
    return expected


def _metrics(value: dict, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Expected metric object: {name}")
    for field in ("auc", "log_loss", "brier"):
        number = _number(value.get(field), f"{name}.{field}", minimum=0)
        if field != "log_loss" and number > 1:
            raise ValueError(f"Metric outside [0,1]: {name}.{field}")


def _schedule(value: dict, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Expected schedule: {name}")
    for field in ("epoch_float", "fraction", "base_lr", "base_wd", "dropout"):
        _number(value.get(field), f"{name}.{field}", minimum=0)
    if value["fraction"] > 1 or value["dropout"] > 1:
        raise ValueError(f"Schedule fraction/dropout outside [0,1]: {name}")
    groups = value.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"Missing parameter group schedule: {name}")
    names = []
    for group in groups:
        if not isinstance(group.get("name"), str):
            raise ValueError(f"Invalid parameter group name: {name}")
        names.append(group["name"])
        _number(group.get("lr"), f"{name}.group.lr", minimum=0)
        _number(group.get("legacy_decay_coefficient"), f"{name}.group.decay", minimum=0)
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate parameter group name: {name}")


def _curves(path: Path, receipt: dict, identity: dict) -> list[dict]:
    rows = [json.loads(line, parse_constant=_invalid_constant)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    horizon = receipt["horizon"]
    if len(rows) != horizon:
        raise ValueError(f"Missing or extra epochs: {path}")
    last_updates, last_seconds = 0, 0.0
    group_names = None
    for epoch, row in enumerate(rows, 1):
        _finite_json(row)
        if type(row.get("epoch")) is not int:
            raise ValueError("Epoch index must be an integer")
        _same_fields(row, {"epoch": epoch, "horizon": horizon, "train_rows": receipt["train_rows"]}, "curve epoch")
        _same_fields(row["context"], identity, "curve context")
        if type(row["updates"]) is not int or row["updates"] <= last_updates:
            raise ValueError("Cumulative update counts must increase each epoch")
        seconds = _number(row["elapsed_seconds"], "epoch elapsed_seconds", minimum=0)
        if seconds < last_seconds:
            raise ValueError("Elapsed epoch time decreased")
        _number(row["objective_mean_per_member"], "training objective", minimum=0)
        _metrics(row["train_eval_probe"], "fixed training probe")
        if identity["phase"] == "inner":
            _metrics(row["monitor"], "passive inner monitor")
        elif row["monitor"] is not None:
            raise ValueError("Outer curve unexpectedly contains monitor metrics")
        for field in ("first_update_schedule", "last_update_schedule"):
            _schedule(row[field], field)
            names = [group["name"] for group in row[field]["groups"]]
            if group_names is None:
                group_names = names
            if names != group_names:
                raise ValueError("Parameter groups changed within trajectory")
        last_updates, last_seconds = row["updates"], seconds
    if last_updates != receipt["optimizer_updates"]:
        raise ValueError("Final curve and trajectory update counts differ")
    if last_seconds > _number(receipt["elapsed_seconds"], "trajectory elapsed_seconds", minimum=0):
        raise ValueError("Curve elapsed time exceeds completed trajectory time")
    return rows


def collect_completed(root: Path, campaign_path: Path | None = None) -> dict:
    """Read only JSON/JSONL receipts; root injection supports generated IO fixtures."""
    root = root.resolve()
    config_path, initial_config = _campaign_config(root, campaign_path)
    campaign = initial_config["id"]
    output = root / "artifacts" / campaign
    inputs = {}

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

    state = load(root / "state" / campaign / "run_state.json")
    _same_fields(state, {"id": campaign, "status": "completed", "completed_fits": 12,
                         "active_child": None}, "controller completion")
    config = load(config_path, state["campaign_sha256"])
    _same_fields(config, {"id": campaign, "output_dir": f"artifacts/{campaign}"}, "campaign")
    registry_path = _path(root, config["registry_path"], output / "registry.json")
    registry = load(registry_path, config["registry_sha256"])
    registry_hash = inputs[registry_path.relative_to(root).as_posix()]
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
                             "class_order": [0, 1], "release_or_incumbent_modified": False}, "evaluation")
    if not evaluation.get("completed_utc"):
        raise ValueError("Missing evaluation completion")
    _same_fields(evaluation["provenance"], {
        "campaign_sha256": state["campaign_sha256"], "registry_sha256": registry_hash,
        "manifest_sha256": completion["completed_manifest_sha256"],
        "completion_sha256": inputs[completion_path.relative_to(root).as_posix()]}, "evaluation provenance")
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
        "source_root": str(root), "input_hashes": inputs,
        "registered_source_hashes": registry["source_hashes"],
        "evaluation_copied_without_recomputation": {key: evaluation[key] for key in (
            "metrics", "contrasts", "advancement", "mixture_alpha", "interpretation", "completed_utc")},
        "trajectory_summaries": summaries, "descriptive_aggregates": aggregates,
        "interpretation": {
            "arms": {"A": "H4/E4", "B": "H16/E4: prefix of C", "C": "H16/E16"},
            "monitor": "Passive inner validation AUC of averaged member probabilities, not outer-fold AUC; never used for stopping or endpoint selection.",
            "training_probe": "Fixed subset of fitting rows, evaluation mode; not the full training set and not held-out evidence.",
            "objective": "Training-mode criterion averaged over rows and members, with changing dropout; distinct from evaluation-mode log loss.",
            "schedule": "Recorded first/last optimization-update values per epoch; not interpolated update-level curves or integrated regularization doses.",
            "time": "Adapter elapsed time includes its telemetry and export work; excludes worker imports and later full-fold native reload. Sums and arithmetic means are descriptive.",
            "statistics": "No confidence interval, independent confirmation or statistical significance is inferred from the three folds."}}


def render_plot(evidence: dict, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fig = Figure(figsize=(12, 7.1), dpi=180)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3, sharex=True, sharey="row")
    colors = {"A": "#0072B2", "C": "#D55E00"}
    for fold in range(3):
        for arm in ("A", "C"):
            row = next(item for item in evidence["trajectory_summaries"]
                       if (item["phase"], item["fold"], item["trajectory"]) == ("inner", fold, arm))
            series = row["curve_rows"]
            epochs = [item["epoch"] for item in series]
            axes[0, fold].plot(epochs, [item["monitor"]["auc"] for item in series],
                               color=colors[arm], marker=".", linewidth=1.6)
            fixed = row["literal_endpoints"]
            axes[0, fold].scatter([item["epoch"] for item in fixed],
                [item["monitor"]["auc"] for item in fixed], facecolors="white", edgecolors=colors[arm], s=45, zorder=3)
            for when, style in (("first_update", "--"), ("last_update", "-")):
                axes[1, fold].plot(epochs, [item[when]["base_lr"] for item in series],
                    color=colors[arm], linestyle=style, linewidth=1.5)
        axes[0, fold].set_title(f"Fold {fold}", fontsize=11)
        for axis in axes[:, fold]:
            axis.set_xlim(.6, 16.4)
            axis.set_xticks([1, 4, 8, 12, 16])
            axis.axvline(4, color="0.65", linewidth=.8, linestyle=":")
            axis.grid(alpha=.18)
            axis.tick_params(labelsize=9)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0, fold].ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[1, fold].set_xlabel("Executed epoch")
        axes[1, fold].set_ylim(bottom=0)
        axes[1, fold].yaxis.set_major_locator(MaxNLocator(5))
    axes[0, 0].set_ylabel("Passive inner-monitor ROC AUC")
    axes[1, 0].set_ylabel("Base learning rate")
    fig.suptitle("Fixed schedule horizon versus executed duration", fontsize=14, y=.975)
    handles = [Line2D([0], [0], color=colors[arm], lw=2, label=label)
               for arm, label in (("A", "A: horizon 4"), ("C", "C: horizon 16; B is epoch 4"))]
    handles += [Line2D([0], [0], color="0.3", linestyle=style, label=label)
                for style, label in (("--", "LR at first update"), ("-", "LR at last update"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False, fontsize=9)
    fig.text(.07, .036, "Circles mark fixed endpoints. AUC is passive inner validation, not outer OOF. Lines join recorded epoch values.\n"
             "No checkpoint selection, new scoring, blend search or confidence intervals. LR lines are first/last update samples.", fontsize=9, color="0.3")
    fig.subplots_adjust(left=.075, right=.985, bottom=.15, top=.85, hspace=.24, wspace=.12)
    fig.savefig(path, dpi=180, facecolor="white", metadata={"Description": evidence["scope"]})
    fig.clear()


def summarize(output_dir: Path, *, root: Path = ROOT, campaign_path: Path | None = None) -> dict:
    root, output_dir = root.resolve(), output_dir.resolve()
    config_path, config = _campaign_config(root, campaign_path)
    report_root = root / "artifacts" / config["id"] / "report"
    if not output_dir.is_relative_to(report_root) or output_dir == report_root:
        raise ValueError(f"Choose a fresh child directory under {report_root}")
    if output_dir.exists():
        raise FileExistsError("Preserve existing report output; choose a fresh directory")
    evidence = collect_completed(root, config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    plot = output_dir / "inner_monitor_and_lr.png"
    render_plot(evidence, plot)
    evidence["reporter"] = {"source_sha256": sha256(Path(__file__)),
                            "matplotlib_version": importlib.metadata.version("matplotlib"),
                            "plot_filename": plot.name, "plot_sha256": sha256(plot)}
    with (output_dir / "evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"evidence": str(output_dir / "evidence.json"), "plot": str(plot),
            "trajectory_count": len(evidence["trajectory_summaries"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--outputdir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.outputdir, campaign_path=args.campaign), indent=2))


if __name__ == "__main__":
    main()
