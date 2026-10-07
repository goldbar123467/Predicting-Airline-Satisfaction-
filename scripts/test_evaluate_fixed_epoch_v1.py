"""Generated-data and receipt tests; never read competition data or fit models."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import evaluate_fixed_epoch_v1 as evaluator


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
            root = Path(directory); output = root / "artifacts/fixed_epoch_v1"; output.mkdir(parents=True)
            (root / "scripts").mkdir()
            source = root / "scripts/evaluate_fixed_epoch_v1.py"; source.write_text("# synthetic source fixture\n")
            registry = {"id": evaluator.CAMPAIGN_ID, "expected_fit_count": 12,
                        "evaluation": evaluator.EVALUATION,
                        "source_hashes": {"scripts/evaluate_fixed_epoch_v1.py": evaluator.sha256(source)}}
            registry_path = output / "registry.json"; registry_path.write_text(json.dumps(registry))
            campaign = {"id": evaluator.CAMPAIGN_ID, "output_dir": "artifacts/fixed_epoch_v1",
                        "split_sha256": evaluator.SPLIT_HASH, "incumbent_selection_sha256": evaluator.INCUMBENT_HASH,
                        "registry_path": "artifacts/fixed_epoch_v1/registry.json",
                        "registry_sha256": evaluator.sha256(registry_path),
                        "completion_receipt_path": "artifacts/fixed_epoch_v1/completion_receipt.json"}
            campaign_path = root / "campaign.json"; campaign_path.write_text(json.dumps(campaign))
            completion = {"id": evaluator.CAMPAIGN_ID, "status": "running", "completed_fit_count": 11}
            (output / "completion_receipt.json").write_text(json.dumps(completion))
            with patch.object(evaluator.pd, "read_parquet") as read, patch.object(evaluator, "metrics") as metric:
                with self.assertRaises(ValueError):
                    evaluator.prepare_campaign(campaign_path, root=root)
                read.assert_not_called(); metric.assert_not_called()

    def native_fixture(self, root):
        output = root / "artifacts/fixed_epoch_v1"
        model = output / "fold_0/outer/C/epoch_004"; model.mkdir(parents=True)
        (model / "model.pt").write_bytes(b"not loaded: synthetic artifact bytes")
        metadata = {"schema_version": 1, "model_type": "realmlp_categorical", "classes": [0, 1],
                    "n_features": 3, "input_schema": {"n_features": 3}, "executed_epochs": 4,
                    "schedule_horizon_epochs": 16, "graph_file": "model.pt",
                    "graph_sha256": evaluator.sha256(model / "model.pt")}
        (model / "metadata.json").write_text(json.dumps(metadata))
        transform = model.parent / "transform.json"; transform.write_text("{}")
        endpoint = {"fold": 0, "arm": "B", "prediction_path": "artifacts/fixed_epoch_v1/fold_0/outer/predictions_B.parquet",
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


if __name__ == "__main__":
    unittest.main()
