"""Generated-data and receipt tests; never read competition data or fit models."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import evaluate_fixed_epoch_dropout_v1 as evaluator


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
            root = Path(directory); output = root / "artifacts/fixed_epoch_dropout_v1"; output.mkdir(parents=True)
            (root / "scripts").mkdir()
            source = root / "scripts/evaluate_fixed_epoch_v1.py"; source.write_text("# synthetic source fixture\n")
            registry = {"id": evaluator.CAMPAIGN_ID, "expected_fit_count": 12,
                        "evaluation": evaluator.EVALUATION,
                        "source_hashes": {"scripts/evaluate_fixed_epoch_v1.py": evaluator.sha256(source)}}
            registry_path = output / "registry.json"; registry_path.write_text(json.dumps(registry))
            campaign = {"id": evaluator.CAMPAIGN_ID, "output_dir": "artifacts/fixed_epoch_dropout_v1",
                        "split_sha256": evaluator.SPLIT_HASH, "incumbent_selection_sha256": evaluator.INCUMBENT_HASH,
                        "registry_path": "artifacts/fixed_epoch_dropout_v1/registry.json",
                        "registry_sha256": evaluator.sha256(registry_path),
                        "completion_receipt_path": "artifacts/fixed_epoch_dropout_v1/completion_receipt.json"}
            campaign_path = root / "campaign.json"; campaign_path.write_text(json.dumps(campaign))
            completion = {"id": evaluator.CAMPAIGN_ID, "status": "running", "completed_fit_count": 11}
            (output / "completion_receipt.json").write_text(json.dumps(completion))
            with patch.object(evaluator.pd, "read_parquet") as read, patch.object(evaluator, "metrics") as metric:
                with self.assertRaises(ValueError):
                    evaluator.prepare_campaign(campaign_path, root=root)
                read.assert_not_called(); metric.assert_not_called()

    def native_fixture(self, root):
        output = root / "artifacts/fixed_epoch_dropout_v1"
        model = output / "fold_0/outer/C/epoch_004"; model.mkdir(parents=True)
        (model / "model.pt").write_bytes(b"not loaded: synthetic artifact bytes")
        metadata = {"schema_version": 1, "model_type": "realmlp_categorical", "classes": [0, 1],
                    "n_features": 3, "input_schema": {"n_features": 3}, "executed_epochs": 4,
                    "schedule_horizon_epochs": 16, "graph_file": "model.pt",
                    "graph_sha256": evaluator.sha256(model / "model.pt")}
        (model / "metadata.json").write_text(json.dumps(metadata))
        transform = model.parent / "transform.json"; transform.write_text("{}")
        endpoint = {"fold": 0, "arm": "B", "prediction_path": "artifacts/fixed_epoch_dropout_v1/fold_0/outer/predictions_B.parquet",
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

    def test_control_arm_is_epoch16_not_historical_short_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, endpoint, ids, receipt = self.native_fixture(root)
            endpoint["arm"] = receipt["arm"] = "A"
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


def generated_update_rows(arm: str, steps: int = 2) -> list[dict]:
    order = hashlib.sha256()
    result = []
    for count in range(1, steps * 16 + 1):
        epoch, batch = (count - 1) // steps + 1, (count - 1) % steps
        indices = {"indices_sha256": hashlib.sha256(str(count).encode()).hexdigest(),
                   "rows_per_member": 3, "members": 8}
        order.update(evaluator.primitive_state_hash({"update": count, **indices}).encode())
        scheduled = {"": 1.0 / count}
        applied = {"": 1.0} if arm == "C" and epoch >= 5 else scheduled
        non_dropout = {"lr": {"": .9}, "wd": {"": .8}}
        result.append({"epoch": epoch, "batch_index": batch, "global_step_before": count - 1,
            "epoch_float_before": float(epoch - 1 + batch / steps), "horizon": 16,
            "batch_order": {**indices, "cumulative_sha256": order.hexdigest()},
            "non_dropout_schedule": non_dropout, "non_dropout_sha256": evaluator.primitive_state_hash(non_dropout),
            "optimizer_groups_sha256": "a" * 64, "p_drop_multipliers": applied,
            "scheduled_p_drop_multipliers": scheduled, "effective_scope_p_drop": {"layer.dropout": .05 * applied[""]},
            "intervention_active": arm == "C" and epoch >= 5, "update": count, "optimizer_step": "completed"})
    return result


def add_generated_prefix_evidence(workspace, output, trajectories, endpoints, ids, folds, registry_hash):
    """Construct self-contained hash-bound receipt fixtures, never actual model state."""
    rows = {(r["phase"], r["fold"], r["trajectory"]): r for r in trajectories}
    records = []
    for phase in ("inner", "outer"):
        for fold in range(3):
            pair = []
            for arm in ("A", "C"):
                folder = output / f"fold_{fold}/{phase}/{arm}"
                done = evaluator.read_json(folder / "done.json")
                adapter = evaluator.read_json(folder / "trajectory.json")
                context = adapter["context"]
                write_json(folder / "context.json", context)
                policy = evaluator.intervention_for(arm)
                signature = {key: hashlib.sha256(key.encode()).hexdigest() for key in evaluator.SIGNATURE_KEYS}
                adapter.update(dropout_policy=policy["policy"], intervention=policy, updates_per_epoch=2,
                    optimizer_updates=32, dropout_scopes=[{"name": "layer.dropout", "scope": "block", "base_pattern": "",
                        "schedule_pattern": "", "base_value": .05, "effective_p_drop": .05}])
                for epoch in (4, 16):
                    metadata_path = folder / f"epoch_{epoch:03d}/metadata.json"
                    metadata = evaluator.read_json(metadata_path)
                    metadata.update(dropout_policy=policy["policy"], intervention=policy, constructor={"n_ens": 8})
                    write_json(metadata_path, metadata)
                    adapter["endpoints"][str(epoch)].update(metadata_sha256=evaluator.sha256(metadata_path),
                        epoch=epoch, horizon=16, network_sha256=signature["network"])
                updates = generated_update_rows(arm)
                raw = [(json.dumps(row, sort_keys=True) + "\n").encode() for row in updates]
                schedule = folder / "update_schedule.jsonl"
                schedule.write_bytes(b"".join(raw))
                adapter.update(update_schedule_path=schedule.relative_to(workspace).as_posix(),
                    update_schedule_sha256=evaluator.sha256(schedule),
                    consumed_batch_order_sha256=updates[-1]["batch_order"]["cumulative_sha256"])
                probability = np.array([[.25, .75], [.75, .25]], dtype=np.float32)
                prefix = {"schema_version": 1, "signature_version": "portable_training_state_v1", "epoch": 4,
                    "horizon": 16, "dropout_policy": policy["policy"], "context": context,
                    "optimizer_updates": 8, "signature": signature,
                    "signature_sha256": evaluator.primitive_state_hash(signature), "input_schema": {"n_features": 2},
                    "constructor": {"n_ens": 8}, "endpoint": adapter["endpoints"]["4"],
                    "native_probe_input_sha256": "b" * 64, "native_probe_probabilities": probability.tolist(),
                    "native_probe_probability_sha256": evaluator.primitive_state_hash(probability), "probe_rows": 2,
                    "update_schedule_prefix_sha256": hashlib.sha256(b"".join(raw[:8])).hexdigest(),
                    "method_addresses_and_walltimes_excluded": True,
                    "future_policy_excluded_from_state_signature": True, "observation_live_state_unchanged": True}
                path = folder / "prefix_state.json"
                write_json(path, prefix)
                fields = {"prefix_state_path": path.relative_to(workspace).as_posix(), "prefix_state_sha256": evaluator.sha256(path)}
                adapter.update(fields); done.update(fields)
                pair.append((folder, adapter, done))
            left, right = pair
            match = {"schema_version": 1, "status": "passed", "epoch": 4, "gate_completed_before_epoch5": True,
                "optimizer_updates_at_gate": 8, "components_equal": {key: True for key in evaluator.SIGNATURE_KEYS},
                "native_probabilities_exact": True, "update_schedule_prefix_exact": True,
                "control_native_reload_max_abs_error": 0.0, "native_tolerance": {"rtol": 1e-5, "atol": 2e-6},
                "probe_rows": 2, "observation_live_state_unchanged": True}
            for name, (_, _, done) in zip(("control", "treatment"), pair):
                match.update({name + "_" + key: done[key] for key in ("prefix_state_path", "prefix_state_sha256")})
            match_path = right[0] / "prefix_match.json"
            write_json(match_path, match)
            check_names = {"campaign", "fold", "phase", "training_rows", "monitor_rows", "training_ids_sha256",
                "input_feature_columns", "transform_sha256", "feature_policy", "transform_bytes", "partition_schema",
                "training_ids", "monitor_ids", "outer_validation_ids"}
            prefit = {"status": "passed", "checks": {key: True for key in check_names}, "before_treatment_fit": True,
                "control_context_sha256": evaluator.sha256(left[0] / "context.json")}
            for name, (folder, _, _) in zip(("control", "treatment"), pair):
                prefit.update({name + "_" + stem + "_sha256": evaluator.sha256(folder / filename)
                    for stem, filename in (("partition", "partitions.npz"), ("transform", "transform.json"))})
            prefit_path = right[0] / "pre_fit_match.json"
            write_json(prefit_path, prefit)
            record = {"phase": phase, "fold": fold, "prefix_match_path": match_path.relative_to(workspace).as_posix(),
                "prefix_match_sha256": evaluator.sha256(match_path), "pre_fit_match_path": prefit_path.relative_to(workspace).as_posix(),
                "pre_fit_match_sha256": evaluator.sha256(prefit_path), "prefix_native_path": None, "prefix_native_sha256": None}
            if phase == "outer":
                native_path = output / f"fold_{fold}/outer/prefix_native_verify.json"
                artifacts = [folder / name for folder, _, _ in pair for name in
                             ("transform.json", "epoch_004/metadata.json", "epoch_004/graph.pt")]
                held = ids[folds == fold]
                write_json(native_path, {"id": evaluator.CAMPAIGN_ID, "status": "passed", "fold": fold,
                    "registry_sha256": registry_hash, "verification_scope": "full_outer_fold", "class_order": [0, 1],
                    "row_count": len(held), "ids_sha256": evaluator.ids_sha256(held), "epoch": 4,
                    "reference": "control_A4", "candidate": "treatment_C4", "chunk_rows": 8191,
                    "quality_metrics_computed": False, "atol": 2e-6, "rtol": 1e-5,
                    "parity_passed": True, "max_absolute_error": 0.0, "max_scaled_error": 0.0,
                    "artifact_hashes": {p.relative_to(workspace).as_posix(): evaluator.sha256(p) for p in artifacts}})
                record.update(prefix_native_path=native_path.relative_to(workspace).as_posix(),
                              prefix_native_sha256=evaluator.sha256(native_path))
            for arm, (folder, adapter, done) in zip(("A", "C"), pair):
                if arm == "C":
                    done.update({key: value for key, value in record.items() if key not in {"phase", "fold"}})
                    adapter.update({key: value for key, value in record.items() if key.startswith("prefix_match_")})
                write_json(folder / "trajectory.json", adapter)
                done["adapter_receipt_sha256"] = evaluator.sha256(folder / "trajectory.json")
                write_json(folder / "done.json", done)
                rows[(phase, fold, arm)]["receipt_sha256"] = evaluator.sha256(folder / "done.json")
            records.append(record)
    for row in endpoints:
        path = workspace / row["native_receipt_path"]
        native = evaluator.read_json(path)
        native["artifact_hashes"] = {name: evaluator.sha256(workspace / name) for name in native["artifact_hashes"]}
        write_json(path, native)
        row["native_receipt_sha256"] = evaluator.sha256(path)
    return records


def add_generated_execution_evidence(workspace, canonical, output, registry, protocol_path):
    """Fourteen mock startup receipts and a byte-bound operational amendment."""
    protocol = evaluator.read_json(protocol_path)
    names = {"hook": "scripts/fixed_epoch_deterministic_execution_v1.py",
        "authorization": "state/fixed_epoch_dropout_v1/deterministic_retry_authorization.json",
        "registration": "cloud/fixed_epoch_dropout_v1_retry1/runtime_amendment.json"}
    hook = canonical / names["hook"]; hook.write_bytes(b"# generated inert startup hook\n")
    write_json(canonical / names["authorization"], {"generated": True})
    binding = {stem + "_path": name for stem, name in names.items()}
    for stem in ("hook", "authorization"):
        binding[stem + "_sha256"] = evaluator.sha256(canonical / names[stem])
    amendment = {"id": evaluator.RETRY_ID, "campaign": evaluator.CAMPAIGN_ID,
        "authorization_path": names["authorization"], "authorization_sha256": binding["authorization_sha256"],
        "scientific_registry_sha256": protocol["registry_sha256"],
        "scientific_bundle_manifest_sha256": protocol["bundle_manifest_sha256"], "scientific_payload_unchanged": True,
        "kernel_id": evaluator.RETRY_KERNEL, "provider_timeout_seconds": 6900, "fit_budget_seconds": 6300,
        "delivery_deadline_seconds": 6840, "hook_source_sha256": binding["hook_sha256"],
        "execution_policy": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "torch_deterministic_algorithms": True,
            "torch_deterministic_warn_only": False, "cudnn_benchmark": False, "cudnn_deterministic": True},
        "matmul_allow_tf32": False, "startup_failure_exit_code": 86,
        "startup_receipts": f"artifacts/{evaluator.CAMPAIGN_ID}/runtime_amendment/processes/process_<PID>.json",
        "source_hashes": {names["hook"]: binding["hook_sha256"]}}
    write_json(canonical / names["registration"], amendment)
    binding["registration_sha256"] = evaluator.sha256(canonical / names["registration"])
    folder = output / "runtime_amendment"
    folder.mkdir()
    (folder / "registration.json").write_bytes((canonical / names["registration"]).read_bytes())
    (folder / "sitecustomize.py").write_bytes(hook.read_bytes())
    protocol["execution_amendment"] = binding
    run_path = canonical / "cloud/fixed_epoch_dropout_v1_retry1/run.py"
    run_path.write_bytes(b"# generated inert cloud entry\n")
    reconciliation = canonical / "cloud/fixed_epoch_dropout_v1_retry1/provider_identity_reconciliation.json"
    write_json(reconciliation, {"status": "verified", "requested_kernel": evaluator.RETRY_KERNEL,
        "actual_kernel": evaluator.PROVIDER_KERNEL, "kernel_id": evaluator.PROVIDER_KERNEL_ID,
        "version": 1, "private_verified": True, "provider_timeout_seconds": 6900,
        "runtime_sha256": evaluator.sha256(run_path), "files": {
            run_path.relative_to(canonical).as_posix(): evaluator.sha256(run_path),
            names["registration"]: binding["registration_sha256"]}})
    protocol["provider_identity_reconciliation"] = {"path": reconciliation.relative_to(canonical).as_posix(),
                                                   "sha256": evaluator.sha256(reconciliation)}
    write_json(protocol_path, protocol)
    write_json(output / "job_runtime.json", {"started_utc": "2026-10-04T18:00:00+00:00",
        "fit_deadline_utc": "2026-10-04T19:45:00+00:00", "delivery_deadline_utc": "2026-10-04T19:54:00+00:00",
        "hard_timeout_seconds": 6900, "registered_hard_timeout_seconds": 7200,
        "registry_sha256": protocol["registry_sha256"], "bundle_manifest_sha256": protocol["bundle_manifest_sha256"]})
    write_json(workspace / f"state/{evaluator.CAMPAIGN_ID}/launch_receipt.json", {"controller": {"pid": 101}})
    launches = [{"role": "smoke", "device": "cuda", "output": f"artifacts/{evaluator.CAMPAIGN_ID}/smoke_cuda"},
                {"role": "controller", "campaign": f"configs/{evaluator.CAMPAIGN_ID}.json"}]
    launches.extend({"role": "worker", "campaign": f"configs/{evaluator.CAMPAIGN_ID}.json",
        "phase": phase, "fold": fold, "trajectory": arm}
        for phase in ("inner", "outer") for fold in range(3) for arm in ("A", "C"))
    for pid, launch in enumerate(launches, 100):
        target = "scripts/" + ("smoke_fixed_epoch_dropout_v1.py" if launch["role"] == "smoke" else "run_fixed_epoch_dropout_v1.py")
        write_json(folder / f"processes/process_{pid}.json", {"schema_version": 1, "status": "passed",
            "policy": "strict_deterministic_execution_v1", "hook_sha256": binding["hook_sha256"],
            "amendment_sha256": binding["registration_sha256"], "flags": evaluator.EXECUTION_FLAGS,
            "prefix_gate_unchanged": True, "model_recipe_unchanged": True, "torch_version": "generated",
            "cuda_build_version": "generated_cuda", "pid": pid, "launch": launch,
            "target": target, "target_sha256": registry["source_hashes"][target]})


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
    local_sources = ("evaluate_fixed_epoch_dropout_v1.py", "test_evaluate_fixed_epoch_dropout_v1.py",
                     "evaluate_fixed_epoch_cloud_v1.py", "test_evaluate_fixed_epoch_cloud_v1.py")
    for name in local_sources:
        path = canonical / "scripts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((Path(evaluator.__file__).parent / name).read_bytes())
    sources = {}
    for name in (*evaluator.ADAPTER_WORKSPACE_SOURCES, "scripts/run_fixed_epoch_dropout_v1.py", "scripts/smoke_fixed_epoch_dropout_v1.py"):
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
        **evaluator.INTERVENTION,
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
    prep = canonical / "cloud/fixed_epoch_dropout_v1/preparation_manifest.json"
    write_json(prep, {"id": evaluator.CAMPAIGN_ID, "registry_sha256": registry_hash,
                     "manifest_sha256": evaluator.sha256(workspace / "bundle-manifest.json")})
    protocol_path = canonical / f"artifacts/{evaluator.CAMPAIGN_ID}/local_evaluation_protocol.json"
    write_json(protocol_path, {"id": evaluator.CAMPAIGN_ID, "status": "frozen_before_assessment", "evaluation": evaluator.EVALUATION,
        "intervention": evaluator.INTERVENTION,
        "assessment_workspace": evaluator.ASSESSMENT_WORKSPACE,
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
                horizon, epochs = 16, [4, 16]
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
                    if phase == "outer" and not (arm == "A" and epoch == 4):
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
    prefixes = add_generated_prefix_evidence(workspace, output, trajectories, endpoints, ids, folds, registry_hash)
    write_json(output / "completed_manifest.json", {**bindings, "trajectories": trajectories, "endpoints": endpoints,
                                                  "prefix_records": prefixes})
    write_json(output / "completion_receipt.json", {**bindings, "status": "completed", "completed_fit_count": 12,
        "completed_endpoint_count": 9, "completed_prefix_count": 6, "completed_prefix_native_count": 3,
        "evaluation_ready": True, "completed_manifest_sha256": evaluator.sha256(output / "completed_manifest.json")})
    write_json(workspace / f"state/{evaluator.CAMPAIGN_ID}/run_state.json", {**bindings, "status": "training_complete",
        "completed_fits": 12, "active_child": None, "outer_metrics_computed": False})
    add_generated_execution_evidence(workspace, canonical, output, registry, protocol_path)
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
                self.assertEqual(len(prepared["prefix_evidence"]), 6)
                self.assertEqual(len(prepared["prefix_native_evidence"]), 3)
                self.assertTrue(prepared["execution_evidence"]["all_fourteen_processes_verified"])
                self.assertEqual(load.call_args.args[-1], fixture["canonical"])
                self.assertEqual(len(reads), 11)
                metric.assert_not_called()
                evaluator.claim_evaluation(prepared, ["synthetic-fixture"])
                with self.assertRaises(FileExistsError):
                    evaluator.claim_evaluation(prepared, ["second-attempt"])
                with self.assertRaises(FileExistsError):
                    evaluator.prepare_campaign(fixture["wrapper"], root=fixture["workspace"], canonical_root=fixture["canonical"])

    def test_missing_prefix_is_rejected_before_any_parquet_or_metric(self):
        with tempfile.TemporaryDirectory() as name:
            fixture = complete_cloud_fixture(Path(name))
            output, root = fixture["output"], fixture["workspace"]
            manifest = evaluator.read_json(output / "completed_manifest.json")
            manifest["prefix_records"].pop()
            write_json(output / "completed_manifest.json", manifest)
            completion = evaluator.read_json(output / "completion_receipt.json")
            completion["completed_manifest_sha256"] = evaluator.sha256(output / "completed_manifest.json")
            write_json(output / "completion_receipt.json", completion)
            returned = evaluator.read_json(root / "output-manifest.json")
            for path in (output / "completed_manifest.json", output / "completion_receipt.json"):
                returned["files"][path.relative_to(root).as_posix()] = {"sha256": evaluator.sha256(path), "bytes": path.stat().st_size}
            write_json(root / "output-manifest.json", returned)
            with ExitStack() as stack:
                for constant, value in (("SPLIT_HASH", fixture["split_hash"]), ("INCUMBENT_HASH", fixture["selection_hash"]), ("DEVELOPMENT_ROWS", 12)):
                    stack.enter_context(patch.object(evaluator, constant, value))
                read = stack.enter_context(patch.object(evaluator.pd, "read_parquet"))
                metric = stack.enter_context(patch.object(evaluator, "metrics"))
                with self.assertRaisesRegex(ValueError, "Exactly six"):
                    evaluator.prepare_campaign(fixture["wrapper"], root=root, canonical_root=fixture["canonical"])
                read.assert_not_called(); metric.assert_not_called()


class UpdateAndPrefixTests(unittest.TestCase):
    def test_provider_identity_reconciliation_is_bound_before_data_reads(self):
        with tempfile.TemporaryDirectory() as name:
            fixture = complete_cloud_fixture(Path(name))
            path = fixture["canonical"] / "cloud/fixed_epoch_dropout_v1_retry1/provider_identity_reconciliation.json"
            value = evaluator.read_json(path); value["private_verified"] = False
            write_json(path, value)
            with ExitStack() as stack:
                for constant, current in (("SPLIT_HASH", fixture["split_hash"]), ("INCUMBENT_HASH", fixture["selection_hash"]), ("DEVELOPMENT_ROWS", 12)):
                    stack.enter_context(patch.object(evaluator, constant, current))
                read = stack.enter_context(patch.object(evaluator.pd, "read_parquet"))
                metric = stack.enter_context(patch.object(evaluator, "metrics"))
                with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                    evaluator.prepare_campaign(fixture["wrapper"], root=fixture["workspace"], canonical_root=fixture["canonical"])
                read.assert_not_called(); metric.assert_not_called()

    def test_every_runtime_process_role_and_strict_flag_is_required(self):
        with tempfile.TemporaryDirectory() as name:
            fixture = complete_cloud_fixture(Path(name))
            root, output, canonical = fixture["workspace"], fixture["output"], fixture["canonical"]
            def check():
                return evaluator.validate_execution_amendment(root, output, fixture["registry"], fixture["protocol"], canonical, "generated")
            self.assertEqual(len(check()["processes"]), 14)
            path = output / "runtime_amendment/processes/process_102.json"
            original = evaluator.read_json(path)
            cases = []
            for field, value in (("deterministic_warn_only", True), ("startup_cuda_initialized", True),
                                 ("deterministic_algorithms", False), ("cublas_workspace_config", ":16:8")):
                bad = copy.deepcopy(original); bad["flags"][field] = value; cases.append(bad)
            bad = copy.deepcopy(original); bad["launch"]["trajectory"] = "C"; cases.append(bad)
            bad = copy.deepcopy(original); bad["target_sha256"] = "0" * 64; cases.append(bad)
            bad = copy.deepcopy(original); bad["amendment_sha256"] = "0" * 64; cases.append(bad)
            for bad in cases:
                write_json(path, bad)
                with self.assertRaises(ValueError): check()
            path.unlink()
            with self.assertRaisesRegex(ValueError, "fourteen"):
                check()

    def test_execution_registration_hook_and_deadline_drift_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            fixture = complete_cloud_fixture(Path(name))
            root, output, canonical = fixture["workspace"], fixture["output"], fixture["canonical"]
            def check():
                return evaluator.validate_execution_amendment(root, output, fixture["registry"], fixture["protocol"], canonical, "generated")
            path = output / "job_runtime.json"
            value = evaluator.read_json(path); value["delivery_deadline_utc"] = "2026-10-04T19:59:00+00:00"
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, "deadlines"):
                check()
            value["delivery_deadline_utc"] = "2026-10-04T19:54:00+00:00"; write_json(path, value)
            for name in ("registration.json", "sitecustomize.py"):
                path = output / "runtime_amendment" / name
                old = path.read_bytes(); path.write_bytes(old + b" ")
                with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                    check()
                path.write_bytes(old)

    def test_json_and_array_fingerprints_match_actual_adapter_function(self):
        # Compile only the pure fingerprint definition; no Torch/training import.
        import ast
        from types import SimpleNamespace
        source = ast.parse(Path(evaluator.__file__).with_name("fixed_epoch_dropout_adapter_v1.py").read_text())
        node = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_hash")
        namespace = {"Any": object, "hashlib": hashlib, "np": np,
                     "torch": SimpleNamespace(is_tensor=lambda value: False)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "adapter_fingerprint", "exec"), namespace)
        values = [{key: hashlib.sha256(key.encode()).hexdigest() for key in evaluator.SIGNATURE_KEYS},
                  np.array([[.1, .9], [.75, .25]], dtype=np.float32),
                  {"lr": {"": 1.0}, "wd": {"layer": .05}},
                  {"update": 1, "indices_sha256": "f" * 64, "rows_per_member": 256, "members": 8}]
        for value in values:
            self.assertEqual(namespace["_hash"](value), evaluator.primitive_state_hash(value))

    def update_fixture(self, folder):
        rows = [generated_update_rows(arm) for arm in ("A", "C")]
        paths = [folder / f"{arm}.jsonl" for arm in ("A", "C")]
        prefixes, adapters = [], []
        for path, records in zip(paths, rows):
            lines = [(json.dumps(row, sort_keys=True) + "\n").encode() for row in records]
            path.write_bytes(b"".join(lines))
            prefixes.append({"update_schedule_prefix_sha256": hashlib.sha256(b"".join(lines[:8])).hexdigest()})
            adapters.append({"updates_per_epoch": 2, "optimizer_updates": 32,
                "dropout_scopes": [{"name": "layer.dropout", "base_value": .05, "schedule_pattern": ""}],
                "consumed_batch_order_sha256": records[-1]["batch_order"]["cumulative_sha256"]})
        return paths, rows, adapters, prefixes

    @staticmethod
    def save_rows(path, rows):
        path.write_bytes(b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows))

    def test_full_applied_dropout_and_order_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            paths, _, adapters, prefixes = self.update_fixture(Path(name))
            result = evaluator.validate_update_pair(paths, adapters, prefixes)
            self.assertEqual(result["updates_per_trajectory"], 32)
            self.assertEqual(result["prefix_updates"], 8)
            self.assertTrue(result["non_dropout_and_order_identical"])

    def test_early_late_or_partial_dropout_change_rejected(self):
        for index, field, value in ((7, "p_drop_multipliers", {"": 1.0}),
                                    (8, "p_drop_multipliers", {"": .2}),
                                    (8, "effective_scope_p_drop", {"layer.dropout": .001}),
                                    (8, "effective_scope_p_drop", {})):
            with tempfile.TemporaryDirectory() as name:
                paths, rows, adapters, prefixes = self.update_fixture(Path(name))
                rows[1][index][field] = value
                self.save_rows(paths[1], rows[1])
                with self.assertRaisesRegex(ValueError, "Dropout|dropout"):
                    evaluator.validate_update_pair(paths, adapters, prefixes)

    def test_non_dropout_or_optimizer_change_rejected(self):
        for field in ("non_dropout_schedule", "optimizer_groups_sha256", "scheduled_p_drop_multipliers"):
            with tempfile.TemporaryDirectory() as name:
                paths, rows, adapters, prefixes = self.update_fixture(Path(name))
                row = rows[1][10]
                if field == "non_dropout_schedule":
                    row[field]["lr"][""] = .7
                    row["non_dropout_sha256"] = evaluator.primitive_state_hash(row[field])
                elif field == "optimizer_groups_sha256":
                    row[field] = "b" * 64
                else:
                    row[field] = {"": .8}
                self.save_rows(paths[1], rows[1])
                with self.assertRaisesRegex(ValueError, "Paired continuation"):
                    evaluator.validate_update_pair(paths, adapters, prefixes)

    def test_order_change_rejected_even_with_self_consistent_chain(self):
        with tempfile.TemporaryDirectory() as name:
            paths, rows, adapters, prefixes = self.update_fixture(Path(name))
            rows[1][10]["batch_order"]["indices_sha256"] = "0" * 64
            chain = hashlib.sha256()
            for row in rows[1]:
                item = {key: row["batch_order"][key] for key in ("indices_sha256", "rows_per_member", "members")}
                chain.update(evaluator.primitive_state_hash({"update": row["update"], **item}).encode())
                row["batch_order"]["cumulative_sha256"] = chain.hexdigest()
            adapters[1]["consumed_batch_order_sha256"] = chain.hexdigest()
            self.save_rows(paths[1], rows[1])
            with self.assertRaisesRegex(ValueError, "Paired continuation"):
                evaluator.validate_update_pair(paths, adapters, prefixes)

    def test_incomplete_extra_and_unbounded_updates_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            for kind in ("missing", "extra", "oversized", "truncated"):
                paths, rows, adapters, prefixes = self.update_fixture(Path(name))
                if kind == "missing":
                    self.save_rows(paths[1], rows[1][:-1])
                elif kind == "extra":
                    self.save_rows(paths[1], [*rows[1], rows[1][-1]])
                elif kind == "oversized":
                    paths[1].write_bytes(b"x" * 262145 + b"\n")
                else:
                    paths[1].write_bytes(paths[1].read_bytes()[:-1])
                with self.assertRaises(ValueError):
                    evaluator.validate_update_pair(paths, adapters, prefixes)

    def test_prefix_components_boundary_and_identity_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            fixture = complete_cloud_fixture(Path(name))
            root, output = fixture["workspace"], fixture["output"]
            manifest = evaluator.read_json(output / "completed_manifest.json")
            def records():
                return {(r["phase"], r["fold"], r["trajectory"]): {
                    "done": evaluator.read_json(root / r["receipt_path"]),
                    "adapter": evaluator.read_json((root / r["receipt_path"]).parent / "trajectory.json")}
                    for r in manifest["trajectories"]}
            evaluator.validate_prefix_pairs(manifest["prefix_records"], records(), root, output)
            path = output / "fold_0/inner/C/prefix_state.json"
            original = evaluator.read_json(path)
            for field in ("signature", "epoch", "native_probe_probability_sha256"):
                value = copy.deepcopy(original)
                if field == "signature":
                    value[field].pop("optimizer")
                    value["signature_sha256"] = evaluator.primitive_state_hash(value[field])
                elif field == "epoch":
                    value[field] = 5
                else:
                    value[field] = "0" * 64
                write_json(path, value)
                checked = records()
                for part in ("done", "adapter"):
                    checked[("inner", 0, "C")][part]["prefix_state_sha256"] = evaluator.sha256(path)
                with self.assertRaisesRegex(ValueError, "Portable prefix|Prefix receipt|native-probability"):
                    evaluator.validate_prefix_pairs(manifest["prefix_records"], checked, root, output)
            write_json(path, original)
            gate_path = output / "fold_0/inner/C/prefix_match.json"
            gate = evaluator.read_json(gate_path); gate["gate_completed_before_epoch5"] = False
            write_json(gate_path, gate)
            checked = records(); manifest["prefix_records"][0]["prefix_match_sha256"] = evaluator.sha256(gate_path)
            checked[("inner", 0, "C")]["done"]["prefix_match_sha256"] = evaluator.sha256(gate_path)
            with self.assertRaisesRegex(ValueError, "before treatment continuation"):
                evaluator.validate_prefix_pairs(manifest["prefix_records"], checked, root, output)

    def test_full_prefix_native_scope_and_both_artifacts_required(self):
        with tempfile.TemporaryDirectory() as name:
            fixture = complete_cloud_fixture(Path(name))
            root, output = fixture["workspace"], fixture["output"]
            receipt = evaluator.read_json(output / "fold_0/outer/prefix_native_verify.json")
            ids = fixture["ids"][fixture["folds"] == 0]
            evaluator.validate_prefix_native(receipt, 0, ids, receipt["registry_sha256"], root, output)
            for field, value in (("epoch", 16), ("verification_scope", "probe"), ("quality_metrics_computed", True),
                                 ("max_scaled_error", 1.01), ("artifact_hashes", {})):
                bad = copy.deepcopy(receipt); bad[field] = value
                with self.assertRaises(ValueError):
                    evaluator.validate_prefix_native(bad, 0, ids, receipt["registry_sha256"], root, output)

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
            preparation = fixture["canonical"] / "cloud/fixed_epoch_dropout_v1/preparation_manifest.json"
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
        old = ast.parse(Path(evaluator.__file__).with_name("evaluate_fixed_epoch_cloud_v1.py").read_text())
        new = ast.parse(Path(evaluator.__file__).read_text())
        for name in ("metrics", "contrast", "advancement_gate", "score_fixed", "load_incumbent", "aligned_oof", "validate_partition"):
            before = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == name)
            after = next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(before), ast.dump(after), name)


if __name__ == "__main__":
    unittest.main()
