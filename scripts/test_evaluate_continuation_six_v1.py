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

import evaluate_continuation_six_v1 as evaluator

SUMMARY = Path(__file__).resolve().parents[1] / "artifacts/continuation_six/adapter_tests_01/generated_summary.json"
SUMMARY_SHA = "407765d47d902e57dca7ebd21bbdfab52f5beb8e7ebb2fe7d543053fd80fbfd5"


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
            root = Path(directory); output = root / "artifacts/fixed_epoch_continuation_01"; output.mkdir(parents=True)
            (root / "scripts").mkdir()
            source = root / "scripts/evaluate_fixed_epoch_v1.py"; source.write_text("# synthetic source fixture\n")
            registry = {"id": evaluator.CAMPAIGN_ID, "expected_fit_count": 12,
                        "evaluation": evaluator.EVALUATION,
                        "source_hashes": {"scripts/evaluate_fixed_epoch_v1.py": evaluator.sha256(source)}}
            registry_path = output / "registry.json"; registry_path.write_text(json.dumps(registry))
            campaign = {"id": evaluator.CAMPAIGN_ID, "output_dir": "artifacts/fixed_epoch_continuation_01",
                        "split_sha256": evaluator.SPLIT_HASH, "incumbent_selection_sha256": evaluator.INCUMBENT_HASH,
                        "registry_path": "artifacts/fixed_epoch_continuation_01/registry.json",
                        "registry_sha256": evaluator.sha256(registry_path),
                        "completion_receipt_path": "artifacts/fixed_epoch_continuation_01/completion_receipt.json"}
            campaign_path = root / "campaign.json"; campaign_path.write_text(json.dumps(campaign))
            completion = {"id": evaluator.CAMPAIGN_ID, "status": "running", "completed_fit_count": 11}
            (output / "completion_receipt.json").write_text(json.dumps(completion))
            with patch.object(evaluator.pd, "read_parquet") as read, patch.object(evaluator, "metrics") as metric:
                with self.assertRaises(ValueError):
                    evaluator.prepare_campaign(campaign_path, root=root)
                read.assert_not_called(); metric.assert_not_called()

    def native_fixture(self, root):
        output = root / "artifacts/fixed_epoch_continuation_01"
        model = output / "fold_0/outer/C/epoch_004"; model.mkdir(parents=True)
        (model / "model.pt").write_bytes(b"not loaded: synthetic artifact bytes")
        metadata = {"schema_version": 1, "model_type": "realmlp_categorical", "classes": [0, 1],
                    "n_features": 3, "input_schema": {"n_features": 3}, "executed_epochs": 4,
                    "schedule_horizon_epochs": 16, "graph_file": "model.pt",
                    "graph_sha256": evaluator.sha256(model / "model.pt")}
        (model / "metadata.json").write_text(json.dumps(metadata))
        transform = model.parent / "transform.json"; transform.write_text("{}")
        endpoint = {"fold": 0, "arm": "B", "prediction_path": "artifacts/fixed_epoch_continuation_01/fold_0/outer/predictions_B.parquet",
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


def generated_pair(folder: Path, policy: str) -> dict:
    """Reuse immutable generated CPU receipts; reconstruct toy schedule/order lines.

    No model code is imported or executed and no competition data is read.
    The applied optimizer rows and endpoint states come from the adapter proof.
    The order/schedule receipt serialization is newly generated test input.
    """
    evaluator.require_hash(SUMMARY, SUMMARY_SHA)
    summary = json.loads(SUMMARY.read_text())
    evaluator.configure_stage(f"fixed_epoch_continuation_0{evaluator.POLICY_IDS.index(policy) + 1}")
    names = ("control", policy)
    adapters = [copy.deepcopy(summary["results"][name]) for name in names]
    metadata = [copy.deepcopy(summary["endpoint_metadata"][name]) for name in names]
    optimizer_rows = [copy.deepcopy(summary["optimizer_evidence"][name]) for name in names]
    paths, schedule_paths, prefixes = [], [], []
    steps = adapters[0]["updates_per_epoch"]
    for arm_index, (name, rows, adapter) in enumerate(zip(names, optimizer_rows, adapters)):
        order = hashlib.sha256()
        generated = []
        for item in rows:
            update, epoch, clock = item["update"], item["epoch"], item["epoch_float_before"]
            common_order = {"indices_sha256": hashlib.sha256(str(update).encode()).hexdigest(),
                            "rows_per_member": 32, "members": 8}
            order.update(evaluator.primitive_state_hash({"update": update, **common_order}).encode())
            scheduled = {"": 1.0 / (1.0 + clock)}
            multipliers = {"": 1.0} if epoch >= 5 else scheduled
            non_dropout = {key: {"": item["groups"][0]["getters"][key]["schedule_value"]} for key in ("lr", "wd")}
            non_dropout["ls_eps"] = {"": item["label_smoothing_scopes"][0]["schedule_value"]}
            group_summary = [{"name": group["name"], "lr": group["getter_lr"],
                              "effective_decay_coefficient": group["decay_coefficient"], "frozen": group["frozen"]}
                             for group in item["groups"]]
            generated.append({"epoch": epoch, "batch_index": (update - 1) % steps,
                "global_step_before": update - 1, "update": update, "horizon": 16,
                "epoch_float_before": clock, "optimizer_step": "completed", "intervention_active": epoch >= 5,
                "batch_order": {**common_order, "cumulative_sha256": order.hexdigest()},
                "non_dropout_schedule": non_dropout, "non_dropout_sha256": evaluator.primitive_state_hash(non_dropout),
                "optimizer_groups_sha256": evaluator.primitive_state_hash(group_summary),
                "p_drop_multipliers": multipliers, "scheduled_p_drop_multipliers": scheduled,
                "effective_scope_p_drop": {scope["name"]: .05 * multipliers[scope["schedule_pattern"]]
                                          for scope in adapter["dropout_scopes"]}})
        raw = [json.dumps(value).encode() + b"\n" for value in generated]
        schedule = folder / f"{arm_index}_schedule.jsonl"
        schedule.write_bytes(b"".join(raw))
        optimizer = folder / f"{arm_index}_optimizer.jsonl"
        optimizer.write_text("".join(json.dumps(value) + "\n" for value in rows))
        adapter["consumed_batch_order_sha256"] = order.hexdigest()
        prefixes.append({"update_schedule_prefix_sha256": hashlib.sha256(b"".join(raw[:steps * 4])).hexdigest()})
        schedule_paths.append(schedule); paths.append(optimizer)
    return {"adapters": adapters, "metadata": metadata, "paths": paths,
            "schedules": schedule_paths, "prefixes": prefixes, "rows": optimizer_rows}


def rewrite_row(path: Path, index: int, mutate) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows[index])
    path.write_text("".join(json.dumps(value) + "\n" for value in rows))


class ContinuationEvidenceTests(unittest.TestCase):
    def tearDown(self):
        evaluator.configure_stage("fixed_epoch_continuation_01")

    def validate_pair(self, fixture):
        roles, frozen = evaluator.validate_continuation_states(fixture["adapters"], fixture["metadata"])
        schedule = evaluator.validate_update_pair(fixture["schedules"], fixture["adapters"], fixture["prefixes"])
        optimizer = evaluator.validate_optimizer_pair(fixture["paths"], fixture["schedules"], fixture["adapters"], roles, frozen)
        return schedule, optimizer

    def test_all_six_actual_generated_optimizer_and_endpoint_receipts_pass(self):
        for policy in evaluator.POLICY_IDS:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as folder:
                fixture = generated_pair(Path(folder), policy)
                schedule, optimizer = self.validate_pair(fixture)
                self.assertEqual(schedule["continuation_policy"], evaluator.policy_for(policy))
                self.assertEqual(optimizer["verified_optimizer_updates_per_trajectory"], 16 * fixture["adapters"][0]["updates_per_epoch"])

    def test_canonical_policy_mapping_and_mutation_isolation(self):
        for index, policy in enumerate(evaluator.POLICY_IDS, 1):
            evaluator.configure_stage(f"fixed_epoch_continuation_0{index}")
            self.assertEqual(evaluator.POLICY_ID, policy)
        for value in ("fixed_epoch_continuation_00", "fixed_epoch_continuation_07", "../01", None):
            with self.assertRaises(ValueError): evaluator.configure_stage(value)
        value = evaluator.policy_for("freeze_embeddings"); value["frozen_roles"].append("extra")
        self.assertNotIn("extra", evaluator.policy_for("freeze_embeddings")["frozen_roles"])

    def test_independent_scalar_oracles(self):
        self.assertEqual(evaluator.linear_lr_multiplier(.5, 4), .5)
        self.assertEqual(evaluator.linear_lr_multiplier(.5, 10), .25)
        self.assertEqual(evaluator.linear_lr_multiplier(.5, 16), 0)
        for value in (float("nan"), -1, 17):
            with self.assertRaises(ValueError): evaluator.linear_lr_multiplier(1, value)
        self.assertAlmostEqual(evaluator.ema_decay(1600) ** 1600, .5, places=12)
        self.assertEqual(evaluator.decay_shrink(.1, .2, .25, 1, applications=1), .995)
        self.assertEqual(evaluator.decay_shrink(.1, .2, .25, 1, applications=2), .99875)

    def test_wrong_stage_policy_and_endpoint_type_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = generated_pair(Path(folder), "ema")
            fixture["metadata"][1]["16"]["export_kind"] = "raw_training_endpoint"
            with self.assertRaises(ValueError): self.validate_pair(fixture)

    def test_ema_live_trajectory_and_parameter_shadow_binding_required(self):
        for mutation in ("final_network_sha256", "ema_parameter_sha256"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                fixture = generated_pair(Path(folder), "ema")
                if mutation == "final_network_sha256": fixture["adapters"][1][mutation] = "f" * 64
                else:
                    state = fixture["adapters"][1]["continuation_state"]
                    state[mutation][next(iter(state[mutation]))] = "f" * 64
                with self.assertRaises(ValueError): self.validate_pair(fixture)

    def test_freeze_inventory_and_native_endpoint_fingerprints_required(self):
        for policy in ("freeze_embeddings", "head_only"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as folder:
                fixture = generated_pair(Path(folder), policy)
                state = fixture["adapters"][1]["continuation_state"]
                name = state["frozen_parameters"][0]
                fixture["metadata"][1]["4"]["raw_training_parameter_sha256"][name] = "f" * 64
                with self.assertRaises(ValueError): self.validate_pair(fixture)

    def test_frozen_manual_decay_cannot_be_hidden_by_zero_gradient(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = generated_pair(Path(folder), "head_only")
            index = fixture["adapters"][0]["updates_per_epoch"] * 4
            rewrite_row(fixture["paths"][1], index, lambda row: row["groups"][0].update(decay_coefficient=.001))
            with self.assertRaises(ValueError): self.validate_pair(fixture)

    def test_correction_cannot_change_actual_gradient_lr_or_start_early(self):
        for index in (0, 16):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as folder:
                fixture = generated_pair(Path(folder), "decay_correction")
                rewrite_row(fixture["paths"][1], index, lambda row: row["groups"][0].update(getter_lr=.999))
                with self.assertRaises(ValueError): self.validate_pair(fixture)

    def test_actual_smoothing_getter_targets_and_uniform_distribution_required(self):
        for key, value in (("effective_ls_eps", 0.0), ("negative_positive_targets", [0.05, .95]),
                           ("distribution_sha256", "f" * 64)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                fixture = generated_pair(Path(folder), "label_smoothing")
                index = fixture["adapters"][0]["updates_per_epoch"] * 4
                rewrite_row(fixture["paths"][1], index,
                            lambda row: row["label_smoothing_scopes"][0].update({key: value}))
                with self.assertRaises(ValueError): self.validate_pair(fixture)

    def test_linear_lr_clock_formula_and_unrelated_wd_are_required(self):
        for hyper in ("lr", "wd"):
            with self.subTest(hyper=hyper), tempfile.TemporaryDirectory() as folder:
                fixture = generated_pair(Path(folder), "linear_lr")
                def mutate(row):
                    row["non_dropout_schedule"][hyper][""] = .123
                    row["non_dropout_sha256"] = evaluator.primitive_state_hash(row["non_dropout_schedule"])
                rewrite_row(fixture["schedules"][1], 20, mutate)
                with self.assertRaises(ValueError): self.validate_pair(fixture)

    def test_duplicate_missing_update_and_oversized_line_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = generated_pair(Path(folder), "ema")
            fixture["paths"][1].write_text(fixture["paths"][1].read_text().splitlines()[0] + "\n")
            with self.assertRaises(ValueError): self.validate_pair(fixture)
            fixture["paths"][1].write_bytes(b"x" * 262145 + b"\n")
            with self.assertRaises(ValueError): list(evaluator.update_rows(fixture["paths"][1]))

    def test_actual_optimizer_compact_digest_and_layer_inventory_are_bound(self):
        for mutation in ("group_digest", "smoothing_layer"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                fixture = generated_pair(Path(folder), "decay_correction")
                if mutation == "group_digest":
                    rewrite_row(fixture["schedules"][1], 20, lambda row: row.update(optimizer_groups_sha256="f" * 64))
                else:
                    rewrite_row(fixture["paths"][1], 20, lambda row: row["label_smoothing_scopes"][0].update(name="omitted_actual_layer"))
                with self.assertRaises(ValueError): self.validate_pair(fixture)

    def test_scientific_functions_remain_ast_identical_to_frozen_parent(self):
        import ast
        source = Path(evaluator.__file__).with_name("evaluate_fixed_epoch_dropout_v1.py")
        evaluator.require_hash(source, "8ff9c036ae4480b4e7bcadbecf0700adf79d8df4fc9dc36b9629ddaac535676d")
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        left, right = functions(source), functions(Path(evaluator.__file__))
        for name in ("metrics", "contrast", "advancement_gate", "score_fixed", "aligned_oof", "load_incumbent",
                     "development_split", "cloud_development_split", "validate_partition", "validate_native_receipt"):
            self.assertEqual(left[name], right[name], name)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def identity_fixture(canonical: Path, registry: dict | None = None) -> dict:
    """Opaque generated provider/source fixture. Never calls a remote service."""
    stage = evaluator.CAMPAIGN_ID
    folder = canonical / f"cloud/continuation_six_v1/{stage}/experiment"
    payload = folder / "bundle/payload"
    root = Path(evaluator.__file__).resolve().parents[1]
    for name in ("configs/continuation_six_v1.json", "state/continuation_six_v1/authorization.json",
                 "scripts/evaluate_continuation_six_v1.py", "scripts/test_evaluate_continuation_six_v1.py",
                 "scripts/evaluate_fixed_epoch_dropout_v1.py", "scripts/test_evaluate_fixed_epoch_dropout_v1.py"):
        target = canonical / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    for module in evaluator.RUNTIME_MODULES:
        target = canonical / ".venv/Lib/site-packages" / (module.replace(".", "/") + ".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Generated runtime fixture: " + module)
    if registry is None:
        registry = {"id": stage, "trajectory_policies": {"A": evaluator.policy_for("control"), "C": evaluator.policy_for(evaluator.POLICY_ID)},
                    "evaluation": evaluator.EVALUATION}
    write_json(payload / "registry.json", registry)
    if not (payload / "bundle-manifest.json").exists():
        write_json(payload / "bundle-manifest.json", {"generated": True})
    for name, content in (("run.py", "# generated exact remote source\n"),
                          ("remote_source_api.py", "# generated exact remote source\n")):
        (folder / name).write_text(content)
    write_json(folder / "push_intent.json", {"generated": True, "durable_before_push": True})
    write_json(folder / "push_receipt.json", {"generated": True, "version": 1, "kernel_id": 100})
    write_json(folder / "remote_metadata.json", {"generated": True, "is_private": True})
    preparation = {"id": stage, "kernel_id": "generated/continuation", "runtime_sha256": evaluator.sha256(folder / "run.py"),
                   "registry_sha256": evaluator.sha256(payload / "registry.json"),
                   "manifest_sha256": evaluator.sha256(payload / "bundle-manifest.json")}
    write_json(folder / "preparation_manifest.json", preparation)
    registration_path = canonical / f"state/continuation_six_v1/registrations/{stage}.json"
    write_json(registration_path, {"plan_sha256": evaluator.SEQUENCE_PLAN_SHA256,
        "preparation_sha256": evaluator.sha256(folder / "preparation_manifest.json")})
    identity = {"status": "verified", "requested_kernel": preparation["kernel_id"], "actual_kernel": "generated/continuation",
        "kernel_id": 100, "version": 1, "private_verified": True, "runtime_sha256": preparation["runtime_sha256"],
        "plan_sha256": evaluator.SEQUENCE_PLAN_SHA256, "registration_sha256": evaluator.sha256(registration_path),
        "push_intent_sha256": evaluator.sha256(folder / "push_intent.json"),
        "push_receipt_sha256": evaluator.sha256(folder / "push_receipt.json"),
        "remote_metadata_sha256": evaluator.sha256(folder / "remote_metadata.json"),
        "remote_source_sha256": evaluator.sha256(folder / "remote_source_api.py"),
        "recovery_receipt_path": None, "recovery_receipt_sha256": None}
    write_json(folder / "provider_identity.json", identity)
    proof = canonical / "artifacts/continuation_six_v1/evaluator_tests/verification.json"
    write_json(proof, {"status": "passed", "returncode": 0,
        "source_hashes": {name: evaluator.sha256(canonical / name) for name in
        ("scripts/evaluate_continuation_six_v1.py", "scripts/test_evaluate_continuation_six_v1.py")}})
    return {"folder": folder, "payload": payload, "proof": proof, "identity": identity, "registry": registry}


def startup_fixture(workspace: Path, registry: dict, protocol: dict) -> dict:
    stage = evaluator.CAMPAIGN_ID
    output = workspace / "artifacts" / stage
    hook = "scripts/continuation_six_deterministic_v1.py"
    target = workspace / hook; target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists(): target.write_text("# Generated hook, never executed")
    registry.setdefault("source_hashes", {})[hook] = evaluator.sha256(target)
    for name in ("scripts/run_continuation_six_v1.py", "scripts/smoke_continuation_six_v1.py"):
        target = workspace / name
        if not target.exists(): target.write_text("# Generated target, never executed")
        registry["source_hashes"][name] = evaluator.sha256(target)
    amendment = {"id": stage + "_strict_execution", "campaign": stage,
        "sequence_plan_sha256": evaluator.SEQUENCE_PLAN_SHA256, "sequence_authorization_sha256": evaluator.SEQUENCE_AUTH_SHA256,
        "hook_source_sha256": registry["source_hashes"][hook], "matmul_allow_tf32": False,
        "startup_failure_exit_code": 86, "required_startup_roles": 14,
        "execution_policy": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "torch_deterministic_algorithms": True,
                             "torch_deterministic_warn_only": False, "cudnn_benchmark": False, "cudnn_deterministic": True}}
    write_json(workspace / "provenance/execution_policy.json", amendment)
    registry["execution_policy_sha256"] = evaluator.sha256(workspace / "provenance/execution_policy.json")
    write_json(output / "runtime_amendment/registration.json", amendment)
    (output / "runtime_amendment/sitecustomize.py").write_bytes((workspace / hook).read_bytes())
    write_json(output / "job_runtime.json", {"started_utc": "2026-10-04T21:00:00+00:00",
        "fit_deadline_utc": "2026-10-04T22:45:00+00:00", "delivery_deadline_utc": "2026-10-04T22:59:00+00:00",
        "hard_timeout_seconds": 7200, "registry_sha256": protocol["registry_sha256"],
        "bundle_manifest_sha256": protocol["bundle_manifest_sha256"]})
    launches = [{"role": "smoke", "device": "cuda", "output": f"artifacts/{stage}/smoke_cuda"},
                {"role": "controller", "campaign": f"configs/{stage}.json"}]
    launches += [{"role": "worker", "campaign": f"configs/{stage}.json", "phase": phase, "fold": fold, "trajectory": arm}
                 for phase in ("inner", "outer") for fold in range(3) for arm in ("A", "C")]
    write_json(workspace / f"state/{stage}/launch_receipt.json", {"controller": {"pid": 101}})
    for index, launch in enumerate(launches):
        name = "scripts/" + ("smoke_continuation_six_v1.py" if launch["role"] == "smoke" else "run_continuation_six_v1.py")
        write_json(output / f"runtime_amendment/processes/process_{100 + index}.json", {
            "schema_version": 1, "status": "passed", "policy": "strict_deterministic_execution_v1",
            "hook_sha256": registry["source_hashes"][hook], "amendment_sha256": registry["execution_policy_sha256"],
            "flags": evaluator.EXECUTION_FLAGS, "prefix_gate_unchanged": True, "model_recipe_unchanged": False,
            "paired_execution_policy": True, "torch_version": "generated", "cuda_build_version": "generated_cuda",
            "pid": 100 + index, "launch": launch, "target": name, "target_sha256": registry["source_hashes"][name]})
    return {"output": output, "registry": registry, "protocol": protocol}


def full_fixture(base: Path, policy: str = "linear_lr") -> dict:
    """Complete 12/9 generated receipt graph with deliberately undecodable models/raw inputs."""
    canonical, workspace = base / "canonical", base / "workspace"
    workspace.mkdir()
    pair_folder = base / "pair"; pair_folder.mkdir()
    pair = generated_pair(pair_folder, policy)
    stage = evaluator.CAMPAIGN_ID
    identity_fixture(canonical)
    output = workspace / "artifacts" / stage
    ids, folds, labels = np.arange(12, dtype=np.int64), np.repeat(np.arange(3), 4), np.tile([0, 1, 0, 1], 3)
    split = pd.DataFrame({"id": ids, "fold": folds})
    (canonical / "data").mkdir(); (workspace / "data").mkdir()
    pd.concat([split, pd.DataFrame({"id": [900], "fold": [-1]})]).to_parquet(canonical / "data/splits.parquet", index=False)
    split.to_parquet(workspace / "data/splits.parquet", index=False)
    selection = canonical / "artifacts/third_pass/blend/frozen.json"
    write_json(selection, {"generated": True})
    split_hash, selection_hash = evaluator.sha256(canonical / "data/splits.parquet"), evaluator.sha256(selection)
    evaluator.SPLIT_HASH, evaluator.INCUMBENT_HASH, evaluator.DEVELOPMENT_ROWS = split_hash, selection_hash, 12
    sources = {}
    for name in (*evaluator.ADAPTER_WORKSPACE_SOURCES, "scripts/run_continuation_six_v1.py",
                 "scripts/smoke_continuation_six_v1.py", "scripts/continuation_six_deterministic_v1.py"):
        path = workspace / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# Generated source " + name)
        sources[name] = evaluator.sha256(path)
    installed, adapter_sources = [], {name: sources[name] for name in evaluator.ADAPTER_WORKSPACE_SOURCES}
    for module in evaluator.RUNTIME_MODULES:
        relative = f"artifacts/{stage}/runtime_source/{module.replace('.', '/')}.py"
        path = workspace / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((canonical / ".venv/Lib/site-packages" / (module.replace(".", "/") + ".py")).read_bytes())
        digest = evaluator.sha256(path); adapter_sources[relative] = digest
        installed.append({"module": module, "recorded_source_path": relative, "installed_source_path": "/generated/" + module,
                          "sha256": digest, "distribution": "pytabkit", "version": "1.7.3"})
    for name in ("data/train.parquet", "data/original_aux_predictions.parquet"):
        (workspace / name).write_bytes(b"Opaque generated raw input. Parquet decoding must fail.")
    write_json(workspace / "data/manifest.json", {"generated": True})
    registry = {"id": stage, **evaluator.INTERVENTION, "expected_fit_count": 12, "evaluation": evaluator.EVALUATION,
                "trajectory_policies": {"A": evaluator.policy_for("control"), "C": evaluator.policy_for(policy)},
                "continuation_policy": evaluator.policy_for(policy), "development_rows": 12,
                "source_split_sha256": split_hash, "incumbent_selection_sha256": selection_hash,
                "sequence_plan_sha256": evaluator.SEQUENCE_PLAN_SHA256, "authorization_sha256": evaluator.SEQUENCE_AUTH_SHA256,
                "audit_rows_exported": 0, "test_rows_exported": 0, "cloud_quality_scoring": False, "source_hashes": sources}
    startup_fixture(workspace, registry, {"registry_sha256": "0" * 64, "bundle_manifest_sha256": "0" * 64})
    registry["input_hashes"] = {name: evaluator.sha256(workspace / name) for name in
        ("data/train.parquet", "data/splits.parquet", "data/original_aux_predictions.parquet", "data/manifest.json", "provenance/execution_policy.json")}
    write_json(workspace / "registry.json", registry)
    registry_hash = evaluator.sha256(workspace / "registry.json")
    wrapper_path = workspace / f"configs/{stage}.json"
    wrapper = {"id": stage, "output_dir": f"artifacts/{stage}", "registry_path": "registry.json", "registry_sha256": registry_hash,
        "source_split_sha256": split_hash, "incumbent_selection_sha256": selection_hash,
        "split_path": "data/splits.parquet", "split_sha256": registry["input_hashes"]["data/splits.parquet"],
        "completed_manifest_path": f"artifacts/{stage}/completed_manifest.json", "completion_receipt_path": f"artifacts/{stage}/completion_receipt.json"}
    write_json(wrapper_path, wrapper)
    wrapper_hash = evaluator.sha256(wrapper_path)
    payload_names = set(sources) | set(registry["input_hashes"]) | {"registry.json", f"configs/{stage}.json"}
    bundle = {"id": stage, "registry_sha256": registry_hash, "files": {name: {
        "bytes": (workspace / name).stat().st_size, "sha256": evaluator.sha256(workspace / name)} for name in payload_names}}
    write_json(workspace / "bundle-manifest.json", bundle)
    folder = canonical / f"cloud/continuation_six_v1/{stage}/experiment"
    write_json(folder / "bundle/payload/bundle-manifest.json", bundle)
    provider = identity_fixture(canonical, registry)
    runtime_path = output / "job_runtime.json"; runtime = json.loads(runtime_path.read_text())
    runtime.update(registry_sha256=registry_hash, bundle_manifest_sha256=evaluator.sha256(workspace / "bundle-manifest.json"))
    write_json(runtime_path, runtime)
    trajectories, endpoints, prefix_records = [], [], []
    binding = {"id": stage, "campaign_sha256": wrapper_hash, "registry_sha256": registry_hash}
    steps = pair["adapters"][0]["updates_per_epoch"]
    for phase in ("inner", "outer"):
        for fold in range(3):
            held, fit = ids[folds == fold], ids[folds != fold]
            monitor = fit[:1] if phase == "inner" else np.array([], dtype=np.int64)
            training = fit[1:] if phase == "inner" else fit
            pending = []
            for index, arm in enumerate(("A", "C")):
                directory = output / f"fold_{fold}/{phase}/{arm}"; directory.mkdir(parents=True)
                np.savez_compressed(directory / "partitions.npz", training_ids=training, monitor_ids=monitor, outer_validation_ids=held)
                write_json(directory / "transform.json", {"generated": True})
                context = {"campaign": stage, "fold": fold, "phase": phase, "trajectory": arm,
                           "partition_sha256": evaluator.sha256(directory / "partitions.npz")}
                write_json(directory / "context.json", context)
                curve = directory / "curves.jsonl"; curve.write_text('{"generated":true}\n')
                for name, source in (("update_schedule.jsonl", pair["schedules"][index]), ("optimizer_updates.jsonl", pair["paths"][index])):
                    (directory / name).write_bytes(source.read_bytes())
                adapter = copy.deepcopy(pair["adapters"][index])
                adapter.update(context=context, versions={"pytabkit": "1.7.3", "torch": "generated"},
                    source_sha256=adapter_sources, installed_source_provenance=installed,
                    curves_path=curve.relative_to(workspace).as_posix(), curves_sha256=evaluator.sha256(curve),
                    trajectory_path=(directory / "trajectory.json").relative_to(workspace).as_posix())
                for stem in ("update_schedule", "optimizer_updates"):
                    adapter[stem + "_path"] = (directory / (stem + ".jsonl")).relative_to(workspace).as_posix()
                    adapter[stem + "_sha256"] = evaluator.sha256(directory / (stem + ".jsonl"))
                for epoch in (4, 16):
                    model = directory / f"epoch_{epoch:03d}"; model.mkdir()
                    (model / "graph.pt").write_bytes(b"Opaque generated graph. Must never be loaded.")
                    metadata = copy.deepcopy(pair["metadata"][index][str(epoch)])
                    metadata.update(context=context, graph_sha256=evaluator.sha256(model / "graph.pt"))
                    write_json(model / "metadata.json", metadata)
                    adapter["endpoints"][str(epoch)].update(path=model.relative_to(workspace).as_posix(),
                        graph_sha256=evaluator.sha256(model / "graph.pt"), metadata_sha256=evaluator.sha256(model / "metadata.json"))
                    if phase == "outer" and (arm != "A" or epoch != 4):
                        logical = "A" if arm == "A" else "B" if epoch == 4 else "C"
                        prediction = directory.parent / f"predictions_{logical}.parquet"
                        pd.DataFrame({"id": held, "fold": np.full(len(held), fold), "satisfaction": labels[held],
                                      "prediction": [.1, .6, .8, .9]}).to_parquet(prediction, index=False)
                        native_path = directory.parent / f"native_verify_{logical}.json"
                        endpoint = {"fold": fold, "arm": logical, "prediction_path": prediction.relative_to(workspace).as_posix(),
                            "prediction_sha256": evaluator.sha256(prediction), "native_receipt_path": native_path.relative_to(workspace).as_posix()}
                        native = {"id": stage, "status": "passed", "fold": fold, "arm": logical, "registry_sha256": registry_hash,
                            "verification_scope": "full_outer_fold", "class_order": [0, 1], "row_count": len(held),
                            "ids_sha256": evaluator.ids_sha256(held), "prediction_path": endpoint["prediction_path"],
                            "prediction_sha256": endpoint["prediction_sha256"], "reference_kind": "first_native_endpoint_reload",
                            "adapter_parity_scope": "fixed_probes_at_capture", "atol": 2e-6, "rtol": 1e-5, "parity_passed": True,
                            "raw_reload_verified": True, "max_absolute_error": 0, "max_scaled_error": 0,
                            "model_directory": model.relative_to(workspace).as_posix(), "artifact_hashes": {
                                p.relative_to(workspace).as_posix(): evaluator.sha256(p) for p in
                                (model / "metadata.json", model / "graph.pt", directory / "transform.json")}}
                        write_json(native_path, native); endpoint["native_receipt_sha256"] = evaluator.sha256(native_path); endpoints.append(endpoint)
                signature = {key: hashlib.sha256(key.encode()).hexdigest() for key in evaluator.SIGNATURE_KEYS}
                signature["network"] = adapter["endpoints"]["4"]["network_sha256"]
                probability = np.array([[.3, .7], [.7, .3]], dtype=np.float32)
                prefix = {"schema_version": 1, "signature_version": "portable_training_state_v1", "epoch": 4, "horizon": 16,
                    "dropout_policy": "hold_base_after_epoch4", "context": context, "optimizer_updates": steps * 4,
                    "signature": signature, "signature_sha256": evaluator.primitive_state_hash(signature),
                    "constructor": adapter["constructor"], "input_schema": adapter["input_schema"], "endpoint": adapter["endpoints"]["4"],
                    "native_probe_input_sha256": "b" * 64, "native_probe_probabilities": probability.tolist(),
                    "native_probe_probability_sha256": evaluator.primitive_state_hash(probability), "probe_rows": 2,
                    **pair["prefixes"][index], "method_addresses_and_walltimes_excluded": True,
                    "future_policy_excluded_from_state_signature": True, "observation_live_state_unchanged": True}
                write_json(directory / "prefix_state.json", prefix)
                adapter.update(prefix_state_path=(directory / "prefix_state.json").relative_to(workspace).as_posix(),
                               prefix_state_sha256=evaluator.sha256(directory / "prefix_state.json"))
                done = {"id": stage, "status": "completed", "phase": phase, "fold": fold, "trajectory": arm,
                    "registry_sha256": registry_hash, "schedule_horizon_epochs": 16, "executed_epochs": 16, "checkpoint_epochs": [4, 16],
                    "telemetry_path": adapter["curves_path"], "telemetry_sha256": adapter["curves_sha256"],
                    "partition_path": (directory / "partitions.npz").relative_to(workspace).as_posix(), "partition_sha256": context["partition_sha256"],
                    "adapter_receipt_path": adapter["trajectory_path"], "prefix_state_path": adapter["prefix_state_path"],
                    "prefix_state_sha256": adapter["prefix_state_sha256"]}
                pending.append((directory, adapter, done))
            left, right = pending
            match = {"schema_version": 1, "status": "passed", "epoch": 4, "gate_completed_before_epoch5": True,
                "optimizer_updates_at_gate": steps * 4, "components_equal": {key: True for key in evaluator.SIGNATURE_KEYS},
                "native_probabilities_exact": True, "update_schedule_prefix_exact": True,
                "native_tolerance": {"atol": 2e-6, "rtol": 1e-5}, "probe_rows": 2,
                "observation_live_state_unchanged": True, "control_native_reload_max_abs_error": 0.0}
            for name, (_, adapter, _) in zip(("control", "treatment"), pending):
                for key in ("prefix_state_path", "prefix_state_sha256"): match[name + "_" + key] = adapter[key]
            write_json(right[0] / "prefix_match.json", match)
            check_names = {"campaign", "fold", "phase", "training_rows", "monitor_rows", "training_ids_sha256",
                           "input_feature_columns", "transform_sha256", "feature_policy", "transform_bytes", "partition_schema",
                           "training_ids", "monitor_ids", "outer_validation_ids"}
            prefit = {"status": "passed", "before_treatment_fit": True, "checks": {key: True for key in check_names},
                      "control_context_sha256": evaluator.sha256(left[0] / "context.json")}
            for name, (directory, _, _) in zip(("control", "treatment"), pending):
                prefit[name + "_partition_sha256"] = evaluator.sha256(directory / "partitions.npz")
                prefit[name + "_transform_sha256"] = evaluator.sha256(directory / "transform.json")
            write_json(right[0] / "pre_fit_match.json", prefit)
            record = {"phase": phase, "fold": fold, "prefix_native_path": None, "prefix_native_sha256": None}
            for stem in ("prefix_match", "pre_fit_match"):
                record[stem + "_path"] = (right[0] / (stem + ".json")).relative_to(workspace).as_posix()
                record[stem + "_sha256"] = evaluator.sha256(right[0] / (stem + ".json"))
            if phase == "outer":
                native_path = right[0].parent / "prefix_native_verify.json"
                write_json(native_path, {"id": stage, "status": "passed", "fold": fold, "registry_sha256": registry_hash,
                    "verification_scope": "full_outer_fold", "class_order": [0, 1], "row_count": len(held), "ids_sha256": evaluator.ids_sha256(held),
                    "epoch": 4, "reference": "control_A4", "candidate": "treatment_C4", "chunk_rows": 8191,
                    "quality_metrics_computed": False, "atol": 2e-6, "rtol": 1e-5, "parity_passed": True,
                    "max_absolute_error": 0.0, "max_scaled_error": 0.0, "artifact_hashes": {
                        (directory / name).relative_to(workspace).as_posix(): evaluator.sha256(directory / name)
                        for directory, _, _ in pending for name in ("transform.json", "epoch_004/metadata.json", "epoch_004/graph.pt")}})
                record.update(prefix_native_path=native_path.relative_to(workspace).as_posix(), prefix_native_sha256=evaluator.sha256(native_path))
            prefix_records.append(record)
            for arm, (directory, adapter, done) in zip(("A", "C"), pending):
                if arm == "C":
                    done.update({key: value for key, value in record.items() if key not in {"phase", "fold"}})
                    adapter.update({key: value for key, value in record.items() if key.startswith("prefix_match_")})
                write_json(directory / "trajectory.json", adapter)
                done["adapter_receipt_sha256"] = evaluator.sha256(directory / "trajectory.json")
                write_json(directory / "done.json", done)
                trajectories.append({"phase": phase, "fold": fold, "trajectory": arm,
                    "receipt_path": (directory / "done.json").relative_to(workspace).as_posix(), "receipt_sha256": evaluator.sha256(directory / "done.json")})
    write_json(output / "completed_manifest.json", {**binding, "trajectories": trajectories, "endpoints": endpoints, "prefix_records": prefix_records})
    write_json(output / "completion_receipt.json", {**binding, "status": "completed", "completed_fit_count": 12,
        "completed_endpoint_count": 9, "completed_prefix_count": 6, "completed_prefix_native_count": 3, "evaluation_ready": True,
        "completed_manifest_sha256": evaluator.sha256(output / "completed_manifest.json")})
    write_json(workspace / f"state/{stage}/run_state.json", {**binding, "status": "training_complete", "completed_fits": 12,
        "active_child": None, "outer_metrics_computed": False})
    return_names = {"registry.json", f"configs/{stage}.json"}
    for part in ("artifacts", "state", "logs"):
        return_names.update(path.relative_to(workspace).as_posix() for path in (workspace / part / stage).rglob("*") if path.is_file())
    write_json(workspace / "output-manifest.json", {"id": stage, "files": {name: {
        "bytes": (workspace / name).stat().st_size, "sha256": evaluator.sha256(workspace / name)} for name in return_names}})
    frozen = evaluator.freeze_protocol(stage, provider["proof"], canonical_root=canonical)
    return {"canonical": canonical, "workspace": workspace, "wrapper": wrapper_path, "output": output,
            "protocol": Path(frozen["path"]), "ids": ids, "folds": folds, "labels": labels}


class FullPreparationTests(unittest.TestCase):
    def setUp(self):
        self.original = evaluator.SPLIT_HASH, evaluator.INCUMBENT_HASH, evaluator.DEVELOPMENT_ROWS

    def tearDown(self):
        evaluator.SPLIT_HASH, evaluator.INCUMBENT_HASH, evaluator.DEVELOPMENT_ROWS = self.original
        evaluator.configure_stage("fixed_epoch_continuation_01")

    def test_complete_12fit_9endpoint_chain_prepares_without_raw_model_or_metric_reads(self):
        for policy in ("linear_lr", "ema", "head_only"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                fixture = full_fixture(Path(directory), policy)
                real_read = pd.read_parquet
                def read(path, *args, **kwargs):
                    self.assertNotIn(Path(path).name, {"train.parquet", "original_aux_predictions.parquet"})
                    return real_read(path, *args, **kwargs)
                with patch.object(evaluator.pd, "read_parquet", side_effect=read), patch.object(evaluator, "metrics") as metric, \
                     patch.object(evaluator, "load_incumbent", return_value=(fixture["labels"], np.linspace(.1, .9, 12), [])):
                    prepared = evaluator.prepare_campaign(fixture["wrapper"], root=fixture["workspace"],
                        canonical_root=fixture["canonical"], protocol_path=fixture["protocol"])
                    self.assertEqual(len(prepared["trajectory_evidence"]), 12)
                    self.assertEqual(len(prepared["native_evidence"]), 9)
                    self.assertEqual(len(prepared["prefix_evidence"]), 6)
                    metric.assert_not_called()
                    evaluator.claim_evaluation(prepared, ["generated test"])
                    with self.assertRaises(FileExistsError): evaluator.claim_evaluation(prepared, ["generated test"])

    def test_missing_prefix_with_self_consistent_file_hashes_stops_before_any_table_read(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = full_fixture(Path(directory))
            manifest_path = fixture["output"] / "completed_manifest.json"
            manifest = json.loads(manifest_path.read_text()); manifest["prefix_records"].pop()
            write_json(manifest_path, manifest)
            completion_path = fixture["output"] / "completion_receipt.json"
            completion = json.loads(completion_path.read_text())
            completion["completed_manifest_sha256"] = evaluator.sha256(manifest_path)
            write_json(completion_path, completion)
            inventory_path = fixture["workspace"] / "output-manifest.json"
            inventory = json.loads(inventory_path.read_text())
            for path in (manifest_path, completion_path):
                inventory["files"][path.relative_to(fixture["workspace"]).as_posix()] = {
                    "bytes": path.stat().st_size, "sha256": evaluator.sha256(path)}
            write_json(inventory_path, inventory)
            with patch.object(evaluator.pd, "read_parquet") as read, patch.object(evaluator, "metrics") as metric:
                with self.assertRaises(ValueError):
                    evaluator.prepare_campaign(fixture["wrapper"], root=fixture["workspace"],
                        canonical_root=fixture["canonical"], protocol_path=fixture["protocol"])
                read.assert_not_called(); metric.assert_not_called()


class ProtocolAndRuntimeTests(unittest.TestCase):
    def setUp(self): evaluator.configure_stage("fixed_epoch_continuation_01")

    def test_source_bound_protocol_freeze_is_exclusive_and_reads_no_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            canonical = Path(directory)
            fixture = identity_fixture(canonical)
            with patch.object(evaluator.pd, "read_parquet") as parquet, patch.object(evaluator, "metrics") as metrics:
                result = evaluator.freeze_protocol(evaluator.CAMPAIGN_ID, fixture["proof"], canonical_root=canonical)
                evaluator.validate_local_protocol(Path(result["path"]), canonical)
                with self.assertRaises(FileExistsError): evaluator.freeze_protocol(evaluator.CAMPAIGN_ID, fixture["proof"], canonical_root=canonical)
                parquet.assert_not_called(); metrics.assert_not_called()

    def test_protocol_rejects_source_or_proof_drift_before_reads(self):
        for name in ("scripts/evaluate_continuation_six_v1.py", "artifacts/continuation_six_v1/evaluator_tests/verification.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                canonical = Path(directory); fixture = identity_fixture(canonical)
                result = evaluator.freeze_protocol(evaluator.CAMPAIGN_ID, fixture["proof"], canonical_root=canonical)
                with (canonical / name).open("ab") as stream: stream.write(b" ")
                with self.assertRaises(ValueError): evaluator.validate_local_protocol(Path(result["path"]), canonical)

    def test_private_provider_source_and_preparation_are_hash_bound(self):
        for name in ("provider_identity.json", "remote_source_api.py", "preparation_manifest.json", "push_intent.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                canonical = Path(directory); fixture = identity_fixture(canonical)
                result = evaluator.freeze_protocol(evaluator.CAMPAIGN_ID, fixture["proof"], canonical_root=canonical)
                with (fixture["folder"] / name).open("ab") as stream: stream.write(b" ")
                with self.assertRaises(ValueError): evaluator.validate_local_protocol(Path(result["path"]), canonical)

    def test_missing_unknown_response_reconciliation_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            canonical = Path(directory); fixture = identity_fixture(canonical)
            fixture["identity"]["push_receipt_sha256"] = None
            write_json(fixture["folder"] / "provider_identity.json", fixture["identity"])
            with self.assertRaises(ValueError): evaluator.freeze_protocol(evaluator.CAMPAIGN_ID, fixture["proof"], canonical_root=canonical)

    def test_every_startup_role_and_strict_setting_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture = startup_fixture(root, {}, {"registry_sha256": "a" * 64, "bundle_manifest_sha256": "b" * 64})
            def validate(): return evaluator.validate_execution_amendment(root, fixture["output"], fixture["registry"], fixture["protocol"], root, "generated")
            self.assertTrue(validate()["all_fourteen_processes_verified"])
            path = fixture["output"] / "runtime_amendment/processes/process_113.json"
            value = json.loads(path.read_text()); value["flags"]["deterministic_warn_only"] = True
            write_json(path, value)
            with self.assertRaises(ValueError): validate()

    def test_runtime_deadline_and_controller_identity_bound(self):
        for mutation in ("deadline", "controller"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); fixture = startup_fixture(root, {}, {"registry_sha256": "a" * 64, "bundle_manifest_sha256": "b" * 64})
                if mutation == "deadline":
                    path = fixture["output"] / "job_runtime.json"; value = json.loads(path.read_text())
                    value["fit_deadline_utc"] = "2026-10-04T22:45:01+00:00"
                else:
                    path = root / f"state/{evaluator.CAMPAIGN_ID}/launch_receipt.json"; value = {"controller": {"pid": 999}}
                write_json(path, value)
                with self.assertRaises(ValueError): evaluator.validate_execution_amendment(root, fixture["output"], fixture["registry"], fixture["protocol"], root, "generated")


if __name__ == "__main__":
    unittest.main()
