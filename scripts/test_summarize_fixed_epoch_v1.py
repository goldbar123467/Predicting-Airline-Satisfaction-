"""Generated JSON fixtures only: no fitting, inference, labels or scoring."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from summarize_fixed_epoch_v1 import collect_completed, sha256, summarize


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def generated_fixture(root: Path, *, defect: str | None = None, campaign: str = "fixed_epoch_v1") -> dict:
    """Create hash-bound completed receipts with invented descriptive numbers."""
    base = root / "artifacts" / campaign
    registry = {"id": campaign, "expected_fit_count": 12,
                "source_hashes": {"synthetic_fixture_only.py": "0" * 64}}
    write(base / "registry.json", registry)
    registry_hash = sha256(base / "registry.json")
    config = {"id": campaign, "output_dir": f"artifacts/{campaign}",
              "registry_path": f"artifacts/{campaign}/registry.json", "registry_sha256": registry_hash}
    config_path = root / "configs" / f"{campaign}.json"
    write(config_path, config)
    bindings = {"id": campaign, "registry_sha256": registry_hash,
                "campaign_sha256": sha256(config_path)}
    trajectories = []
    for phase in ("inner", "outer"):
        for fold in range(3):
            for arm in ("A", "C"):
                horizon, endpoints = (4, [4]) if arm == "A" else (16, [4, 16])
                identity = {"campaign": campaign, "phase": phase, "fold": fold, "trajectory": arm}
                folder = base / f"fold_{fold}" / phase / arm
                folder.mkdir(parents=True)
                rows = []
                for epoch in range(1, horizon + 1):
                    fraction = (epoch - 1) / horizon
                    schedule = {"epoch_float": float(epoch - 1), "fraction": fraction,
                        "base_lr": .053 * (1 - fraction), "base_wd": .015 * fraction, "dropout": .05,
                        "groups": [{"name": "synthetic.weight", "lr": .053 * (1 - fraction),
                                    "legacy_decay_coefficient": .0001 * fraction}]}
                    monitor = {"auc": .8 + .01 * min(epoch, 6) - .001 * max(epoch - 6, 0) + .001 * fold,
                               "log_loss": .3, "brier": .1}
                    rows.append({"context": identity, "epoch": epoch, "horizon": horizon,
                        "updates": 2 * epoch, "train_rows": 64, "elapsed_seconds": epoch * .2,
                        "objective_mean_per_member": .2, "train_eval_probe": {"auc": .95, "log_loss": .12, "brier": .05},
                        "monitor": monitor if phase == "inner" else None,
                        "first_update_schedule": schedule, "last_update_schedule": deepcopy(schedule)})
                if (phase, fold, arm) == ("inner", 0, "A"):
                    if defect == "epoch":
                        rows[1]["epoch"] = 1
                    elif defect == "updates":
                        rows[1]["updates"] = 1
                    elif defect == "missing_monitor":
                        rows[1]["monitor"] = None
                curve_path = folder / "curves.jsonl"
                curve_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                adapter = {"status": "complete", "horizon": horizon, "executed_epochs": horizon,
                    "endpoint_epochs": endpoints, "clock": "epoch_fraction", "legacy_decay_semantics": True,
                    "context": identity, "curves_sha256": sha256(curve_path), "train_rows": 64,
                    "monitor_rows": 16 if phase == "inner" else 0, "optimizer_updates": 2 * horizon,
                    "elapsed_seconds": horizon * .2 + .1, "device": "cpu-generated-fixture",
                    "memory_end": {"process_rss_bytes": 123456}}
                adapter_path = folder / "trajectory.json"
                write(adapter_path, adapter)
                done = {"id": campaign, "status": "completed", "registry_sha256": registry_hash,
                    "phase": phase, "fold": fold, "trajectory": arm, "schedule_horizon_epochs": horizon,
                    "executed_epochs": horizon, "checkpoint_epochs": endpoints,
                    "adapter_receipt_path": adapter_path.relative_to(root).as_posix(),
                    "adapter_receipt_sha256": sha256(adapter_path),
                    "telemetry_path": curve_path.relative_to(root).as_posix(), "telemetry_sha256": sha256(curve_path)}
                done_path = folder / "done.json"
                write(done_path, done)
                trajectories.append({"phase": phase, "fold": fold, "trajectory": arm,
                    "receipt_path": done_path.relative_to(root).as_posix(), "receipt_sha256": sha256(done_path)})
    manifest = {**bindings, "trajectories": trajectories,
                "endpoints": [{"fold": fold, "arm": arm} for fold in range(3) for arm in ("A", "B", "C")]}
    write(base / "completed_manifest.json", manifest)
    completion = {**bindings, "status": "completed", "completed_fit_count": 12,
        "completed_endpoint_count": 9, "evaluation_ready": True,
        "completed_manifest_sha256": sha256(base / "completed_manifest.json")}
    write(base / "completion_receipt.json", completion)
    evaluation = {"id": campaign, "weight_search_performed": False, "class_order": [0, 1],
        "release_or_incumbent_modified": False, "completed_utc": "synthetic-fixture-time",
        "provenance": {"campaign_sha256": bindings["campaign_sha256"], "registry_sha256": registry_hash,
            "manifest_sha256": completion["completed_manifest_sha256"],
            "completion_sha256": sha256(base / "completion_receipt.json")},
        "metrics": {"A": {"pooled_auc": .81234567890123, "fold_auc": [.8, .81, .82]}},
        "contrasts": {"C_minus_B": {"pooled_auc_delta": -.000123456789}},
        "advancement": {"advance_to_confirmation": False}, "mixture_alpha": .1,
        "interpretation": "GENERATED FIXTURE: no observations or computed model metrics"}
    write(base / "evaluation.json", evaluation)
    write(root / "state" / campaign / "run_state.json", {**bindings, "status": "completed",
                                                        "completed_fits": 12, "active_child": None})
    return evaluation


class ReporterTests(unittest.TestCase):
    def test_complete_fixture_plot_and_literal_metrics(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            evaluation = generated_fixture(root)
            before = {path.relative_to(root): sha256(path) for path in root.rglob("*") if path.is_file()}
            output = root / "artifacts/fixed_epoch_v1/report/fixture"
            result = summarize(output, root=root)
            evidence = json.loads(Path(result["evidence"]).read_text())
            self.assertEqual(result["trajectory_count"], 12)
            copied = evidence["evaluation_copied_without_recomputation"]
            for key in ("metrics", "contrasts", "advancement"):
                self.assertEqual(copied[key], evaluation[key])
            self.assertEqual(sum(row["executed_epochs"] for row in evidence["trajectory_summaries"]), 120)
            self.assertEqual(sum(row["sum_optimizer_updates"] for row in evidence["descriptive_aggregates"]), 240)
            png = Path(result["plot"])
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(sha256(png), evidence["reporter"]["plot_sha256"])
            self.assertEqual(before, {path: sha256(root / path) for path in before})
            with self.assertRaises(FileExistsError):
                summarize(output, root=root)

    def test_incomplete_or_missing_evaluation_rejected_before_output(self):
        for defect in ("running", "missing_evaluation"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                generated_fixture(root)
                if defect == "running":
                    state_path = root / "state/fixed_epoch_v1/run_state.json"
                    state = json.loads(state_path.read_text())
                    state["status"] = "running"
                    write(state_path, state)
                else:
                    (root / "artifacts/fixed_epoch_v1/evaluation.json").unlink()
                output = root / "artifacts/fixed_epoch_v1/report/rejected"
                with self.assertRaises((ValueError, FileNotFoundError)):
                    summarize(output, root=root)
                self.assertFalse(output.exists())

    def test_hash_and_epoch_contracts(self):
        for defect in ("hash", "epoch", "updates", "missing_monitor"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                generated_fixture(root, defect=defect)
                if defect == "hash":
                    with (root / "artifacts/fixed_epoch_v1/fold_0/inner/A/curves.jsonl").open("a") as stream:
                        stream.write("\n")
                with self.assertRaises(ValueError):
                    collect_completed(root)

    def test_output_boundary(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            generated_fixture(root)
            for output in (root / "outside", root / "artifacts/fixed_epoch_v1/report"):
                with self.assertRaises(ValueError):
                    summarize(output, root=root)

    def test_recovery_namespace_from_wrapper(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            campaign = "fixed_epoch_v1_recovery"
            generated_fixture(root, campaign=campaign)
            wrapper = root / "configs" / f"{campaign}.json"
            evidence = collect_completed(root, wrapper)
            self.assertEqual(evidence["id"], campaign)
            self.assertTrue(all("fixed_epoch_v1/" not in path for path in evidence["input_hashes"]))
            with self.assertRaises(ValueError):
                summarize(root / "artifacts/fixed_epoch_v1/report/wrong", root=root, campaign_path=wrapper)


if __name__ == "__main__":
    unittest.main()
