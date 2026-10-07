"""Generated-data and receipt tests; never read competition data or fit models."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import evaluate_fixed_epoch_cloud_v1 as evaluator


def inventory() -> dict:
    return {"trajectories": [{"phase": phase, "fold": fold, "trajectory": trajectory}
                             for phase in ("inner", "outer") for fold in (0, 1, 2)
                             for trajectory in ("A", "C")],
            "endpoints": [{"fold": fold, "arm": arm} for fold in (0, 1, 2) for arm in ("A", "B", "C")]}


class KeyedOOFTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.arange(10, 22, dtype=np.int64)
        self.folds = np.repeat(np.arange(3), 4)
        self.y = np.tile([0, 1, 0, 1], 3).astype(np.int8)
        self.p = np.linspace(.1, .9, 12)
        self.frame = pd.DataFrame({"id": self.ids, "fold": self.folds,
                                   "satisfaction": self.y, "prediction": self.p})

    def check(self, frame):
        return evaluator.aligned_oof(frame, self.ids, self.folds, self.y, class_order=[0, 1])

    def test_permuted_row_join_is_exact(self):
        probability, labels = self.check(self.frame.sample(frac=1, random_state=19))
        np.testing.assert_array_equal(probability, self.p)
        np.testing.assert_array_equal(labels, self.y)

    def test_missing_duplicate_and_extra_ids_rejected(self):
        malformed = [self.frame.iloc[:-1].copy(), pd.concat([self.frame, self.frame.iloc[:1]])]
        duplicate = self.frame.copy(); duplicate.loc[1, "id"] = duplicate.loc[0, "id"]
        extra = self.frame.copy(); extra.loc[0, "id"] = 999
        for frame in [*malformed, duplicate, extra]:
            with self.assertRaises(ValueError):
                self.check(frame)

    def test_fold_label_class_and_dtype_rejected(self):
        for field, value in (("fold", -1), ("fold", 2), ("satisfaction", 2), ("satisfaction", 1)):
            frame = self.frame.copy(); frame.loc[0, field] = value
            with self.assertRaises(ValueError):
                self.check(frame)
        frame = self.frame.copy(); frame["id"] = frame.id.astype(str)
        with self.assertRaises(ValueError):
            self.check(frame)
        with self.assertRaises(ValueError):
            evaluator.aligned_oof(self.frame, self.ids, self.folds, self.y, class_order=[1, 0])

    def test_invalid_probabilities_rejected(self):
        for value in (np.nan, np.inf, -np.inf, -.001, 1.001):
            frame = self.frame.copy(); frame.loc[0, "prediction"] = value
            with self.assertRaises(ValueError):
                self.check(frame)

    def test_one_class_fold_rejected(self):
        frame = self.frame.copy(); frame.loc[frame.fold == 1, "satisfaction"] = 0
        with self.assertRaises(ValueError):
            evaluator.aligned_oof(frame, self.ids, self.folds, None, class_order=[0, 1])

    def test_extra_columns_rejected(self):
        frame = self.frame.copy(); frame["audit_label"] = 0
        with self.assertRaises(ValueError):
            self.check(frame)

    def test_development_filter_precedes_prediction_join(self):
        split = pd.concat([self.frame[["id", "fold"]], pd.DataFrame({"id": [900, 901], "fold": [-1, -1]})])
        selected = evaluator.development_split(split, expected_rows=12)
        np.testing.assert_array_equal(selected.id, self.ids)
        audit_prediction = self.frame.copy(); audit_prediction.loc[0, ["id", "fold"]] = [900, -1]
        with self.assertRaises(ValueError):
            self.check(audit_prediction)

    def test_id_hash_is_order_independent_and_dtype_canonical(self):
        self.assertEqual(evaluator.ids_sha256(self.ids), evaluator.ids_sha256(self.ids[::-1].astype(np.int32)))
        with self.assertRaises(ValueError):
            evaluator.ids_sha256(np.array([2.5]))


class CompletionAndNativeTests(unittest.TestCase):
    def test_partition_excludes_outer_and_audit_rows(self):
        ids = np.arange(30); folds = np.repeat([0, 1, 2], 10)
        outer = {"training_ids": ids[10:], "monitor_ids": np.array([], dtype=np.int64),
                 "outer_validation_ids": ids[:10]}
        evaluator.validate_partition(outer, ids, folds, 0, "outer")
        inner = {"training_ids": ids[12:], "monitor_ids": ids[10:12], "outer_validation_ids": ids[:10]}
        evaluator.validate_partition(inner, ids, folds, 0, "inner")
        for replacement in (1, 900):
            bad = {key: value.copy() for key, value in outer.items()}
            bad["training_ids"][0] = replacement
            with self.assertRaises(ValueError):
                evaluator.validate_partition(bad, ids, folds, 0, "outer")
        bad = {key: value.copy() for key, value in inner.items()}; bad["monitor_ids"][0] = bad["training_ids"][0]
        with self.assertRaises(ValueError):
            evaluator.validate_partition(bad, ids, folds, 0, "inner")

    def test_exact_inventory_and_partial_duplicate_wrong_fold_rejected(self):
        evaluator.validate_manifest_inventory(inventory())
        for kind in ("trajectories", "endpoints"):
            missing = inventory(); missing[kind].pop()
            duplicate = inventory(); duplicate[kind][-1] = duplicate[kind][0]
            for record in (missing, duplicate):
                with self.assertRaises(ValueError):
                    evaluator.validate_manifest_inventory(record)
        wrong = inventory(); wrong["endpoints"][0]["fold"] = -1
        with self.assertRaises(ValueError):
            evaluator.validate_manifest_inventory(wrong)

    def test_incomplete_campaign_rejected_before_prediction_read_or_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "artifacts/fixed_epoch_cloud_v1"; output.mkdir(parents=True)
            (root / "scripts").mkdir()
            source = root / "scripts/evaluate_fixed_epoch_v1.py"; source.write_text("# synthetic source fixture\n")
            registry = {"id": evaluator.CAMPAIGN_ID, "expected_fit_count": 12,
                        "evaluation": evaluator.EVALUATION,
                        "source_hashes": {"scripts/evaluate_fixed_epoch_v1.py": evaluator.sha256(source)}}
            registry_path = output / "registry.json"; registry_path.write_text(json.dumps(registry))
            campaign = {"id": evaluator.CAMPAIGN_ID, "output_dir": "artifacts/fixed_epoch_cloud_v1",
                        "split_sha256": evaluator.SPLIT_HASH, "incumbent_selection_sha256": evaluator.INCUMBENT_HASH,
                        "registry_path": "artifacts/fixed_epoch_cloud_v1/registry.json",
                        "registry_sha256": evaluator.sha256(registry_path),
                        "completion_receipt_path": "artifacts/fixed_epoch_cloud_v1/completion_receipt.json"}
            campaign_path = root / "campaign.json"; campaign_path.write_text(json.dumps(campaign))
            completion = {"id": evaluator.CAMPAIGN_ID, "status": "running", "completed_fit_count": 11}
            (output / "completion_receipt.json").write_text(json.dumps(completion))
            with patch.object(evaluator.pd, "read_parquet") as read, patch.object(evaluator, "metrics") as metric:
                with self.assertRaises(ValueError):
                    evaluator.prepare_campaign(campaign_path, root=root)
                read.assert_not_called(); metric.assert_not_called()

    def native_fixture(self, root):
        output = root / "artifacts/fixed_epoch_cloud_v1"
        model = output / "fold_0/outer/C/epoch_004"; model.mkdir(parents=True)
        (model / "model.pt").write_bytes(b"not loaded: synthetic artifact bytes")
        metadata = {"schema_version": 1, "model_type": "realmlp_categorical", "classes": [0, 1],
                    "n_features": 3, "input_schema": {"n_features": 3}, "executed_epochs": 4,
                    "schedule_horizon_epochs": 16, "graph_file": "model.pt",
                    "graph_sha256": evaluator.sha256(model / "model.pt")}
        (model / "metadata.json").write_text(json.dumps(metadata))
        transform = model.parent / "transform.json"; transform.write_text("{}")
        endpoint = {"fold": 0, "arm": "B", "prediction_path": "artifacts/fixed_epoch_cloud_v1/fold_0/outer/predictions_B.parquet",
                    "prediction_sha256": "f" * 64}
        ids = np.array([31, 14, 22, 8])
        receipt = {"id": evaluator.CAMPAIGN_ID, "status": "passed", "fold": 0, "arm": "B",
                   "registry_sha256": "r" * 64, "verification_scope": "full_outer_fold", "class_order": [0, 1],
                   "row_count": 4, "ids_sha256": evaluator.ids_sha256(ids),
                   "prediction_path": endpoint["prediction_path"], "prediction_sha256": endpoint["prediction_sha256"],
                   "model_directory": model.relative_to(root).as_posix(),
                   "artifact_hashes": {p.relative_to(root).as_posix(): evaluator.sha256(p)
                                       for p in (model / "model.pt", model / "metadata.json", transform)},
                   "reference_kind": "first_native_endpoint_reload", "adapter_parity_scope": "fixed_probes_at_capture",
                   "atol": 2e-6, "rtol": 1e-5, "max_absolute_error": 1e-7, "max_scaled_error": .02,
                   "parity_passed": True, "raw_reload_verified": True}
        return output, endpoint, ids, receipt

    def test_native_receipt_identity_parity_and_artifact_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output, endpoint, ids, receipt = self.native_fixture(root)
            evaluator.validate_native_receipt(receipt, endpoint, ids, "r" * 64, root, output)
            for field, value in (("arm", "C"), ("class_order", [1, 0]), ("row_count", 3),
                                 ("verification_scope", "probe"), ("max_scaled_error", 1.000001),
                                 ("max_absolute_error", np.nan), ("raw_reload_verified", False)):
                bad = copy.deepcopy(receipt); bad[field] = value
                with self.assertRaises(ValueError):
                    evaluator.validate_native_receipt(bad, endpoint, ids, "r" * 64, root, output)
            model_path = root / receipt["model_directory"] / "model.pt"; model_path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                evaluator.validate_native_receipt(receipt, endpoint, ids, "r" * 64, root, output)

    def test_native_transform_required_and_parent_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output, endpoint, ids, receipt = self.native_fixture(root)
            receipt["artifact_hashes"] = {path: value for path, value in receipt["artifact_hashes"].items()
                                          if not path.endswith("transform.json")}
            with self.assertRaises(ValueError):
                evaluator.validate_native_receipt(receipt, endpoint, ids, "r" * 64, root, output)
            with self.assertRaises(ValueError):
                evaluator.checked_path(root, "../outside.json")

    def test_metadata_and_exact_consumed_graph_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output, endpoint, ids, receipt = self.native_fixture(root)
            missing = copy.deepcopy(receipt)
            missing["artifact_hashes"] = {path: value for path, value in missing["artifact_hashes"].items()
                                          if not path.endswith("metadata.json")}
            with self.assertRaises(ValueError):
                evaluator.validate_native_receipt(missing, endpoint, ids, "r" * 64, root, output)
            metadata_path = root / receipt["model_directory"] / "metadata.json"
            metadata = json.loads(metadata_path.read_text()); metadata["schedule_horizon_epochs"] = 4
            metadata_path.write_text(json.dumps(metadata))
            receipt["artifact_hashes"][metadata_path.relative_to(root).as_posix()] = evaluator.sha256(metadata_path)
            with self.assertRaises(ValueError):
                evaluator.validate_native_receipt(receipt, endpoint, ids, "r" * 64, root, output)


class ScoringAndGateTests(unittest.TestCase):
    @staticmethod
    def summary(pooled, macro=None, folds=None):
        return {"pooled_auc": pooled, "mean_fold_auc": pooled if macro is None else macro,
                "fold_auc": [pooled] * 3 if folds is None else folds}

    def summaries(self):
        return {"incumbent": self.summary(.7), "mixture_A": self.summary(.70002),
                "mixture_B": self.summary(.70003), "mixture_C": self.summary(.70005)}

    def test_gate_requires_every_comparator_and_correct_direction(self):
        self.assertTrue(evaluator.advancement_gate(self.summaries())["advance_to_confirmation"])
        for name in ("incumbent", "mixture_A", "mixture_B"):
            values = self.summaries(); values[name] = self.summary(.700049)
            result = evaluator.advancement_gate(values)
            self.assertFalse(result["advance_to_confirmation"])
            self.assertEqual(len(result["comparisons"]), 3)
        values = self.summaries(); values["mixture_C"] = self.summary(.699)
        self.assertFalse(evaluator.advancement_gate(values)["advance_to_confirmation"])

    def test_macro_and_any_fold_are_mandatory(self):
        values = self.summaries(); values["mixture_C"]["mean_fold_auc"] = .700035
        self.assertFalse(evaluator.advancement_gate(values)["advance_to_confirmation"])
        for fold in range(3):
            values = self.summaries(); values["mixture_C"]["fold_auc"][fold] = .6999
            self.assertFalse(evaluator.advancement_gate(values)["advance_to_confirmation"])

    def test_literal_threshold_boundary_no_relaxation(self):
        values = {name: self.summary(0) for name in ("incumbent", "mixture_A", "mixture_B")}
        values["mixture_C"] = self.summary(1e-5, folds=[-2e-5, 0, .00005])
        self.assertTrue(evaluator.advancement_gate(values)["advance_to_confirmation"])
        values["mixture_C"]["pooled_auc"] = np.nextafter(1e-5, -np.inf)
        self.assertFalse(evaluator.advancement_gate(values)["advance_to_confirmation"])

    def test_nonfinite_gate_is_error(self):
        values = self.summaries(); values["mixture_C"]["pooled_auc"] = np.nan
        with self.assertRaises(ValueError):
            evaluator.advancement_gate(values)

    def test_only_fixed_mixtures_and_contrasts(self):
        y = np.tile([0, 1, 0, 1], 3); folds = np.repeat(np.arange(3), 4)
        anchor = np.tile([.1, .6, .8, .9], 3)
        arms = {"A": np.tile([.2, .7, .4, .8], 3), "B": np.tile([.5, .6, .3, .4], 3),
                "C": np.tile([.3, .8, .2, .9], 3)}
        before = {key: value.copy() for key, value in arms.items()}
        result = evaluator.score_fixed(y, folds, anchor, arms)
        self.assertEqual(set(result["metrics"]), {"incumbent", "A", "B", "C", "mixture_A", "mixture_B", "mixture_C"})
        self.assertEqual(len(result["contrasts"]), 9)
        for arm in arms:
            expected = roc_auc_score(y, .9 * anchor + .1 * arms[arm])
            self.assertEqual(result["metrics"][f"mixture_{arm}"]["pooled_auc"], expected)
            np.testing.assert_array_equal(arms[arm], before[arm])
        expected = result["metrics"]["B"]["pooled_auc"] - result["metrics"]["A"]["pooled_auc"]
        self.assertEqual(result["contrasts"]["B_minus_A"]["pooled_auc_delta"], expected)
        self.assertFalse(result["weight_search_performed"])
        self.assertFalse(result["advancement"]["release_promoted"])


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def complete_cloud_fixture(base: Path) -> dict:
    """Generated receipt graph; raw feature tables are deliberately not Parquet."""
    workspace, canonical = base / "workspace", base / "canonical"
    output = workspace / "artifacts" / evaluator.CAMPAIGN_ID
    (canonical / "data").mkdir(parents=True)
    (workspace / "data").mkdir(parents=True)
    ids = np.arange(12, dtype=np.int64)
    folds = np.repeat(np.arange(3), 4)
    labels = np.tile([0, 1, 0, 1], 3).astype(np.int8)
    split = pd.DataFrame({"id": ids, "fold": folds})
    pd.concat([split, pd.DataFrame({"id": [900], "fold": [-1]})]).to_parquet(canonical / "data/splits.parquet", index=False)
    split.sample(frac=1, random_state=19).to_parquet(workspace / "data/splits.parquet", index=False)
    split_hash = evaluator.sha256(canonical / "data/splits.parquet")
    selection = canonical / "artifacts/third_pass/blend/frozen.json"
    write_json(selection, {"synthetic_incumbent_fixture": True})
    selection_hash = evaluator.sha256(selection)
    local_sources = ("evaluate_fixed_epoch_cloud_v1.py", "test_evaluate_fixed_epoch_cloud_v1.py",
                     "evaluate_fixed_epoch_v1.py", "test_evaluate_fixed_epoch_v1.py")
    for name in local_sources:
        path = canonical / "scripts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((Path(evaluator.__file__).parent / name).read_bytes())
    sources = {}
    for name in (*evaluator.ADAPTER_WORKSPACE_SOURCES, "scripts/run_fixed_epoch_cloud_v1.py"):
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# generated cloud source fixture: " + name)
        sources[name] = evaluator.sha256(path)
    inputs = {}
    for name in ("data/train.parquet", "data/original_aux_predictions.parquet"):
        (workspace / name).write_bytes(b"Raw input is hashed only: decoding this file must fail.")
    write_json(workspace / "data/manifest.json", {"fixture": True})
    for name in ("data/train.parquet", "data/original_aux_predictions.parquet", "data/manifest.json", "data/splits.parquet"):
        inputs[name] = evaluator.sha256(workspace / name)
    runtime_pins, installed, adapter_sources = {}, [], {key: sources[key] for key in evaluator.ADAPTER_WORKSPACE_SOURCES}
    for module in evaluator.RUNTIME_MODULES:
        name = f"artifacts/{evaluator.CAMPAIGN_ID}/runtime_source/{module.replace('.', '/')}.py"
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# generated installed-source fixture: " + module)
        digest = evaluator.sha256(path)
        runtime_pins[module] = adapter_sources[name] = digest
        installed.append({"module": module, "recorded_source_path": name, "installed_source_path": "/cloud/site-packages/" + module,
                          "sha256": digest, "distribution": "pytabkit", "version": "1.7.3"})
    registry = {"id": evaluator.CAMPAIGN_ID, "expected_fit_count": 12, "evaluation": evaluator.EVALUATION,
        "development_rows": 12, "source_split_sha256": split_hash, "incumbent_selection_sha256": selection_hash,
        "audit_rows_exported": 0, "test_rows_exported": 0, "cloud_quality_scoring": False,
        "source_hashes": sources, "input_hashes": inputs}
    write_json(workspace / "registry.json", registry)
    registry_hash = evaluator.sha256(workspace / "registry.json")
    wrapper_path = workspace / f"configs/{evaluator.CAMPAIGN_ID}.json"
    wrapper = {"id": evaluator.CAMPAIGN_ID, "output_dir": f"artifacts/{evaluator.CAMPAIGN_ID}",
        "registry_path": "registry.json", "registry_sha256": registry_hash, "source_split_sha256": split_hash,
        "incumbent_selection_sha256": selection_hash, "split_path": "data/splits.parquet", "split_sha256": inputs["data/splits.parquet"],
        "completion_receipt_path": f"artifacts/{evaluator.CAMPAIGN_ID}/completion_receipt.json",
        "completed_manifest_path": f"artifacts/{evaluator.CAMPAIGN_ID}/completed_manifest.json"}
    write_json(wrapper_path, wrapper)
    wrapper_hash = evaluator.sha256(wrapper_path)
    payload_names = set(sources) | set(inputs) | {"registry.json", f"configs/{evaluator.CAMPAIGN_ID}.json"}
    bundle = {"id": evaluator.CAMPAIGN_ID, "registry_sha256": registry_hash,
        "files": {name: {"sha256": evaluator.sha256(workspace / name), "bytes": (workspace / name).stat().st_size}
                  for name in payload_names}}
    write_json(workspace / "bundle-manifest.json", bundle)
    prep = canonical / "cloud/fixed_epoch_v1/preparation_manifest.json"
    write_json(prep, {"id": evaluator.CAMPAIGN_ID, "registry_sha256": registry_hash,
                     "manifest_sha256": evaluator.sha256(workspace / "bundle-manifest.json")})
    protocol_path = canonical / f"artifacts/{evaluator.CAMPAIGN_ID}/local_evaluation_protocol.json"
    write_json(protocol_path, {"id": evaluator.CAMPAIGN_ID, "status": "frozen_before_assessment", "evaluation": evaluator.EVALUATION,
        "source_split_sha256": split_hash, "incumbent_selection_sha256": selection_hash,
        "quality_metrics_read_when_frozen": False, "real_predictions_scored_when_frozen": False,
        "preparation_sha256": evaluator.sha256(prep), "registry_sha256": registry_hash,
        "bundle_manifest_sha256": evaluator.sha256(workspace / "bundle-manifest.json"),
        "source_sha256": {"scripts/" + name: evaluator.sha256(canonical / "scripts" / name) for name in local_sources},
        "runtime_source_sha256": runtime_pins})
    bindings = {"id": evaluator.CAMPAIGN_ID, "campaign_sha256": wrapper_hash, "registry_sha256": registry_hash}
    trajectories, endpoints = [], []
    for phase in ("inner", "outer"):
        for fold in range(3):
            held, fit = ids[folds == fold], ids[folds != fold]
            monitor = fit[:1] if phase == "inner" else np.array([], dtype=np.int64)
            training = fit[1:] if phase == "inner" else fit
            for arm in ("A", "C"):
                horizon, epochs = (4, [4]) if arm == "A" else (16, [4, 16])
                folder = output / f"fold_{fold}" / phase / arm
                folder.mkdir(parents=True)
                partition = folder / "partitions.npz"
                np.savez_compressed(partition, training_ids=training, monitor_ids=monitor, outer_validation_ids=held)
                context = {"campaign": evaluator.CAMPAIGN_ID, "phase": phase, "fold": fold, "trajectory": arm,
                           "partition_sha256": evaluator.sha256(partition)}
                transform = folder / "transform.json"
                write_json(transform, {"generated": True})
                curve = folder / "curves.jsonl"
                curve.write_text('{"generated_curve_receipt": true}\n')
                adapter = {"status": "complete", "horizon": horizon, "executed_epochs": horizon, "endpoint_epochs": epochs,
                    "clock": "epoch_fraction", "legacy_decay_semantics": True, "context": context,
                    "artifact_path_base": "runtime_workspace", "source_sha256": adapter_sources,
                    "installed_source_provenance": installed, "versions": {"pytabkit": "1.7.3", "torch": "generated"},
                    "curves_sha256": evaluator.sha256(curve), "curves_path": curve.relative_to(workspace).as_posix(),
                    "trajectory_path": (folder / "trajectory.json").relative_to(workspace).as_posix(), "endpoints": {}}
                for epoch in epochs:
                    model = folder / f"epoch_{epoch:03d}"
                    model.mkdir()
                    (model / "graph.pt").write_bytes(b"Synthetic graph bytes: must never be loaded.")
                    write_json(model / "metadata.json", {"schema_version": 1, "model_type": "realmlp_categorical", "classes": [0, 1],
                        "n_features": 2, "input_schema": {"n_features": 2}, "executed_epochs": epoch,
                        "schedule_horizon_epochs": horizon, "graph_file": "graph.pt", "graph_sha256": evaluator.sha256(model / "graph.pt")})
                    adapter["endpoints"][str(epoch)] = {"path": model.relative_to(workspace).as_posix(),
                        "metadata_sha256": evaluator.sha256(model / "metadata.json"), "graph_sha256": evaluator.sha256(model / "graph.pt")}
                    if phase == "outer":
                        endpoint_arm = "A" if arm == "A" else ("B" if epoch == 4 else "C")
                        prediction = folder.parent / f"predictions_{endpoint_arm}.parquet"
                        pd.DataFrame({"id": held, "fold": np.full(len(held), fold), "satisfaction": labels[held],
                                      "prediction": np.array([.1, .6, .8, .9])}).to_parquet(prediction, index=False)
                        native_path = folder.parent / f"native_verify_{endpoint_arm}.json"
                        endpoint = {"fold": fold, "arm": endpoint_arm, "prediction_path": prediction.relative_to(workspace).as_posix(),
                                    "prediction_sha256": evaluator.sha256(prediction), "native_receipt_path": native_path.relative_to(workspace).as_posix()}
                        native = {"id": evaluator.CAMPAIGN_ID, "status": "passed", "fold": fold, "arm": endpoint_arm,
                            "registry_sha256": registry_hash, "verification_scope": "full_outer_fold", "class_order": [0, 1],
                            "row_count": len(held), "ids_sha256": evaluator.ids_sha256(held),
                            "prediction_path": endpoint["prediction_path"], "prediction_sha256": endpoint["prediction_sha256"],
                            "reference_kind": "first_native_endpoint_reload", "adapter_parity_scope": "fixed_probes_at_capture",
                            "atol": 2e-6, "rtol": 1e-5, "parity_passed": True, "raw_reload_verified": True,
                            "max_absolute_error": 0, "max_scaled_error": 0, "model_directory": model.relative_to(workspace).as_posix(),
                            "artifact_hashes": {path.relative_to(workspace).as_posix(): evaluator.sha256(path)
                                                for path in (model / "graph.pt", model / "metadata.json", transform)}}
                        write_json(native_path, native)
                        endpoint["native_receipt_sha256"] = evaluator.sha256(native_path)
                        endpoints.append(endpoint)
                adapter_path = folder / "trajectory.json"
                write_json(adapter_path, adapter)
                done = {"id": evaluator.CAMPAIGN_ID, "status": "completed", "phase": phase, "fold": fold,
                    "trajectory": arm, "registry_sha256": registry_hash, "schedule_horizon_epochs": horizon,
                    "executed_epochs": horizon, "checkpoint_epochs": epochs, "telemetry_path": curve.relative_to(workspace).as_posix(),
                    "telemetry_sha256": evaluator.sha256(curve), "partition_path": partition.relative_to(workspace).as_posix(),
                    "partition_sha256": evaluator.sha256(partition), "adapter_receipt_path": adapter_path.relative_to(workspace).as_posix(),
                    "adapter_receipt_sha256": evaluator.sha256(adapter_path)}
                done_path = folder / "done.json"
                write_json(done_path, done)
                trajectories.append({"phase": phase, "fold": fold, "trajectory": arm,
                                     "receipt_path": done_path.relative_to(workspace).as_posix(), "receipt_sha256": evaluator.sha256(done_path)})
    write_json(output / "completed_manifest.json", {**bindings, "trajectories": trajectories, "endpoints": endpoints})
    write_json(output / "completion_receipt.json", {**bindings, "status": "completed", "completed_fit_count": 12,
        "completed_endpoint_count": 9, "evaluation_ready": True, "completed_manifest_sha256": evaluator.sha256(output / "completed_manifest.json")})
    write_json(workspace / f"state/{evaluator.CAMPAIGN_ID}/run_state.json", {**bindings, "status": "training_complete",
        "completed_fits": 12, "active_child": None, "outer_metrics_computed": False})
    names = {"registry.json", f"configs/{evaluator.CAMPAIGN_ID}.json"}
    for part in ("artifacts", "state", "logs"):
        names.update(path.relative_to(workspace).as_posix() for path in (workspace / part / evaluator.CAMPAIGN_ID).rglob("*") if path.is_file())
    write_json(workspace / "output-manifest.json", {"id": evaluator.CAMPAIGN_ID, "files": {
        name: {"sha256": evaluator.sha256(workspace / name), "bytes": (workspace / name).stat().st_size} for name in names}})
    return {"workspace": workspace, "canonical": canonical, "wrapper": wrapper_path, "output": output,
            "split_hash": split_hash, "selection_hash": selection_hash, "ids": ids, "folds": folds, "labels": labels,
            "registry": registry, "protocol": json.loads(protocol_path.read_text()), "adapter": adapter}


class CloudBoundaryTests(unittest.TestCase):
    def test_exact_cloud_development_membership(self):
        split = pd.DataFrame({"id": np.arange(12), "fold": np.repeat([0, 1, 2], 4)})
        canonical = pd.concat([split, pd.DataFrame({"id": [99], "fold": [-1]})])
        result = evaluator.cloud_development_split(split.sample(frac=1, random_state=4), canonical, 12)
        np.testing.assert_array_equal(result.id, split.id)
        bad = split.copy(); bad.loc[0, "fold"] = 1
        for cloud in (bad, canonical, split.iloc[:-1]):
            with self.assertRaises(ValueError):
                evaluator.cloud_development_split(cloud, canonical, 12)

    def test_complete_fixture_preparation_decodes_no_raw_inputs_and_claims_once(self):
        with tempfile.TemporaryDirectory() as name:
            fixture = complete_cloud_fixture(Path(name))
            native_read = pd.read_parquet
            reads = []
            def tracked_read(path, *args, **kwargs):
                reads.append(Path(path))
                self.assertNotIn(Path(path).name, {"train.parquet", "original_aux_predictions.parquet"})
                return native_read(path, *args, **kwargs)
            with ExitStack() as stack:
                for constant, value in (("SPLIT_HASH", fixture["split_hash"]), ("INCUMBENT_HASH", fixture["selection_hash"]), ("DEVELOPMENT_ROWS", 12)):
                    stack.enter_context(patch.object(evaluator, constant, value))
                stack.enter_context(patch.object(evaluator.pd, "read_parquet", side_effect=tracked_read))
                load = stack.enter_context(patch.object(evaluator, "load_incumbent", return_value=(fixture["labels"], np.tile([.1, .6, .8, .9], 3), [])))
                metric = stack.enter_context(patch.object(evaluator, "metrics", side_effect=AssertionError("No scoring in preparation")))
                prepared = evaluator.prepare_campaign(fixture["wrapper"], root=fixture["workspace"], canonical_root=fixture["canonical"])
                self.assertTrue(prepared["canonical_development_rows_exactly_matched"])
                self.assertFalse(prepared["raw_cloud_train_aux_decoded"])
                self.assertEqual(len(prepared["trajectory_evidence"]), 12)
                self.assertEqual(len(prepared["native_evidence"]), 9)
                self.assertEqual(load.call_args.args[-1], fixture["canonical"])
                self.assertEqual(len(reads), 11)
                metric.assert_not_called()
                evaluator.claim_evaluation(prepared, ["synthetic-fixture"])
                with self.assertRaises(FileExistsError):
                    evaluator.claim_evaluation(prepared, ["second-attempt"])
                with self.assertRaises(FileExistsError):
                    evaluator.prepare_campaign(fixture["wrapper"], root=fixture["workspace"], canonical_root=fixture["canonical"])

    def test_return_inventory_rejects_unlisted_and_tampered_files(self):
        with tempfile.TemporaryDirectory() as name:
            fixture = complete_cloud_fixture(Path(name))
            evaluator.validate_return_manifest(fixture["workspace"])
            extra = fixture["output"] / "unlisted.json"
            extra.write_text("{}")
            with self.assertRaises(ValueError):
                evaluator.validate_return_manifest(fixture["workspace"])
            extra.unlink()
            (fixture["output"] / "fold_0/inner/A/curves.jsonl").write_text("changed")
            with self.assertRaises(ValueError):
                evaluator.validate_return_manifest(fixture["workspace"])

    def test_installed_source_and_versions_are_pinned(self):
        with tempfile.TemporaryDirectory() as name:
            fixture = complete_cloud_fixture(Path(name))
            evaluator.validate_runtime_sources(fixture["adapter"], fixture["workspace"], fixture["registry"], fixture["protocol"])
            for field in ("version", "sha256", "recorded_source_path"):
                bad = copy.deepcopy(fixture["adapter"])
                bad["installed_source_provenance"][0][field] = "different"
                with self.assertRaises(ValueError):
                    evaluator.validate_runtime_sources(bad, fixture["workspace"], fixture["registry"], fixture["protocol"])

    def test_preparation_anchor_drift_rejected_before_data_reads(self):
        with tempfile.TemporaryDirectory() as name:
            fixture = complete_cloud_fixture(Path(name))
            preparation = fixture["canonical"] / "cloud/fixed_epoch_v1/preparation_manifest.json"
            saved = json.loads(preparation.read_text())
            saved["registry_sha256"] = "0" * 64
            write_json(preparation, saved)
            with ExitStack() as stack:
                for constant, value in (("SPLIT_HASH", fixture["split_hash"]), ("INCUMBENT_HASH", fixture["selection_hash"]), ("DEVELOPMENT_ROWS", 12)):
                    stack.enter_context(patch.object(evaluator, constant, value))
                read = stack.enter_context(patch.object(evaluator.pd, "read_parquet"))
                metric = stack.enter_context(patch.object(evaluator, "metrics"))
                with self.assertRaises(ValueError):
                    evaluator.prepare_campaign(fixture["wrapper"], root=fixture["workspace"], canonical_root=fixture["canonical"])
                read.assert_not_called()
                metric.assert_not_called()

    def test_scientific_functions_are_identical_to_frozen_evaluator(self):
        import ast
        old = ast.parse(Path(evaluator.__file__).with_name("evaluate_fixed_epoch_v1.py").read_text())
        new = ast.parse(Path(evaluator.__file__).read_text())
        for name in ("metrics", "contrast", "advancement_gate", "score_fixed", "load_incumbent", "aligned_oof", "validate_partition"):
            before = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == name)
            after = next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(before), ast.dump(after), name)


if __name__ == "__main__":
    unittest.main()
