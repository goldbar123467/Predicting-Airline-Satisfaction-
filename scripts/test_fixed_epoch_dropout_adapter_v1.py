"""Generated-data CPU contracts for the isolated dropout-only intervention."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from fixed_epoch_dropout_adapter_v1 import COREROOT, _hash, fit_fixed_trajectory
from fixed_epoch_cloud_adapter_v1 import fit_fixed_trajectory as frozen_fit
from realmlp import _sha256
from realmlp_categorical import load_realmlp_categorical


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class DropoutAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fixed_epoch_dropout_test_", dir=COREROOT / "artifacts")
        cls.root = Path(cls.temporary.name)
        rng = np.random.default_rng(97531)
        cls.values = rng.normal(size=(192, 8)).astype(np.float32)
        cls.values[:, :6] = cls.values[:, :6] * np.array([.4, 2, 4, 1, 3, 5], dtype=np.float32) + 2
        cls.values[:, 6] = np.arange(192) % 4
        cls.values[:, 7] = np.arange(192) % 24
        cls.values[160:, 6:] = [999, 777]
        cls.labels = (cls.values[:, 0] + .2 * cls.values[:, 1] > 2.4).astype(np.int64)
        cls.config = {"device": "cpu", "seed": 20261005, "threads": 1, "hidden_sizes": [16, 8],
                      "batch_size": 32, "eval_batch_size": 37, "categorical_indices": [6, 7]}
        cls.deadline = time.monotonic() + 120
        cls.context = {"campaign": cls.root.name, "phase": "synthetic", "synthetic_trace": True}
        cls.control = cls.fit("control", "scheduled")
        cls.treatment = cls.fit("treatment", "hold_base_after_epoch4", cls.root / "control")
        cls.frozen = frozen_fit(cls.values[:160], cls.labels[:160], cls.values[160:], cls.labels[160:],
                                cls.config, cls.root / "frozen", horizon=16, endpoint_epochs=[4, 16],
                                context=cls.context, deadline_monotonic=cls.deadline)
        cls.prefix_control = read(cls.root / "control/prefix_state.json")
        cls.prefix_treatment = read(cls.root / "treatment/prefix_state.json")
        cls.updates = {name: [json.loads(line) for line in (cls.root / name / "update_schedule.jsonl").read_text().splitlines()]
                       for name in ("control", "treatment")}

    @classmethod
    def fit(cls, name, policy, reference=None):
        return fit_fixed_trajectory(cls.values[:160], cls.labels[:160], cls.values[160:], cls.labels[160:],
                                    cls.config, cls.root / name, horizon=16, endpoint_epochs=[4, 16],
                                    context=cls.context, deadline_monotonic=cls.deadline,
                                    dropout_policy=policy, prefix_reference_dir=reference)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_scheduled_control_matches_frozen_math_and_randomness(self):
        for key in ("initial_network_sha256", "final_network_sha256", "final_optimizer_sha256", "final_rng_sha256",
                    "synthetic_batch_trace_sha256", "training_index_sha256", "optimizer_updates", "constructor",
                    "resolved_factory_config"):
            self.assertEqual(self.control[key], self.frozen[key], key)
        for epoch in ("4", "16"):
            self.assertEqual(self.control["endpoints"][epoch]["network_sha256"], self.frozen["endpoints"][epoch]["network_sha256"])
            left = load_realmlp_categorical(COREROOT / self.control["endpoints"][epoch]["path"], device="cpu")
            right = load_realmlp_categorical(COREROOT / self.frozen["endpoints"][epoch]["path"], device="cpu")
            with torch.jit.optimized_execution(False):
                np.testing.assert_array_equal(left.predict_proba(self.values), right.predict_proba(self.values))

    def test_exact_full_prefix_and_pre_epoch5_gate(self):
        self.assertEqual(self.prefix_control["signature"], self.prefix_treatment["signature"])
        self.assertEqual(len(self.prefix_control["signature"]), 19)
        self.assertEqual(self.prefix_control["signature_sha256"], _hash(self.prefix_control["signature"]))
        self.assertEqual(self.prefix_control["native_probe_probability_sha256"], self.prefix_treatment["native_probe_probability_sha256"])
        self.assertEqual(self.prefix_control["update_schedule_prefix_sha256"], self.prefix_treatment["update_schedule_prefix_sha256"])
        match = read(self.root / "treatment/prefix_match.json")
        self.assertEqual(match["status"], "passed")
        self.assertTrue(match["gate_completed_before_epoch5"])
        self.assertEqual(match["optimizer_updates_at_gate"], 20)
        self.assertTrue(all(match["components_equal"].values()))
        for name in ("control", "treatment"):
            self.assertEqual(match[name + "_prefix_state_sha256"], _sha256(self.root / name / "prefix_state.json"))
        np.testing.assert_array_equal(self.prefix_control["native_probe_probabilities"], self.prefix_treatment["native_probe_probabilities"])

    def test_only_dropout_changes_after_four_and_actual_order_matches(self):
        control, treatment = self.updates["control"], self.updates["treatment"]
        self.assertEqual(len(control), 80)
        self.assertEqual(control[:20], treatment[:20])
        for index, (left, right) in enumerate(zip(control, treatment), start=1):
            for key in ("non_dropout_schedule", "non_dropout_sha256", "optimizer_groups_sha256", "batch_order",
                        "scheduled_p_drop_multipliers", "epoch_float_before", "horizon", "optimizer_step", "update"):
                self.assertEqual(left[key], right[key], (index, key))
            self.assertEqual(left["non_dropout_sha256"], _hash(left["non_dropout_schedule"]))
            self.assertEqual(right["update"], index)
            self.assertEqual(right["global_step_before"], index - 1)
            self.assertEqual(right["epoch"], (index - 1) // 5 + 1)
            self.assertEqual(right["batch_index"], (index - 1) % 5)
            self.assertEqual(right["optimizer_step"], "completed")
            self.assertFalse(left["intervention_active"])
            self.assertEqual(left["p_drop_multipliers"], left["scheduled_p_drop_multipliers"])
            if index > 20:
                self.assertTrue(right["intervention_active"])
                self.assertEqual(right["p_drop_multipliers"], {"": 1.0})
                self.assertTrue(all(value == .05 for value in right["effective_scope_p_drop"].values()))
                self.assertTrue(all(value < .05 for value in left["effective_scope_p_drop"].values()))
        self.assertEqual(self.control["consumed_batch_order_sha256"], self.treatment["consumed_batch_order_sha256"])
        self.assertNotEqual(self.control["final_network_sha256"], self.treatment["final_network_sha256"])
        self.assertNotEqual(self.control["final_optimizer_sha256"], self.treatment["final_optimizer_sha256"])
        self.assertTrue(self.control["legacy_decay_semantics"])

    def test_schema_and_native_endpoint_contract(self):
        for result in (self.control, self.treatment):
            self.assertNotIn(999, result["input_schema"]["vocabularies"][0])
            self.assertNotIn(777, result["input_schema"]["vocabularies"][1])
            self.assertEqual(result["constructor"]["n_ens"], 8)
            for epoch, endpoint in result["endpoints"].items():
                folder = COREROOT / endpoint["path"]
                self.assertEqual(_sha256(folder / "graph.pt"), endpoint["graph_sha256"])
                self.assertEqual(_sha256(folder / "metadata.json"), endpoint["metadata_sha256"])
                model = load_realmlp_categorical(folder, device="cpu")
                self.assertEqual(model.metadata["dropout_policy"], result["dropout_policy"])
                self.assertEqual(model.metadata["executed_epochs"], int(epoch))
                with torch.jit.optimized_execution(False):
                    predictions = model.predict_proba(self.values)
                    chunked = np.concatenate([model.predict_proba(value) for value in np.array_split(self.values, 19)])
                np.testing.assert_allclose(predictions, chunked, atol=2e-6, rtol=1e-5)
        self.assertLessEqual(read(self.root / "treatment/prefix_match.json")["control_native_reload_max_abs_error"], 2e-6)

    def test_corrupted_reference_stops_before_first_changed_update(self):
        reference = self.root / "bad_reference"
        reference.mkdir()
        prefix = copy.deepcopy(self.prefix_control)
        prefix["signature"]["cpu_rng"] = "0" * 64
        prefix["signature_sha256"] = _hash(prefix["signature"])
        (reference / "prefix_state.json").write_text(json.dumps(prefix), encoding="utf-8")
        trajectory = copy.deepcopy(self.control)
        trajectory["prefix_state_sha256"] = _sha256(reference / "prefix_state.json")
        (reference / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "full-state prefix mismatch"):
            self.fit("rejected", "hold_base_after_epoch4", reference)
        result = read(self.root / "rejected/trajectory.json")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["executed_epochs"], 4)
        updates = [json.loads(line) for line in (self.root / "rejected/update_schedule.jsonl").read_text().splitlines()]
        self.assertEqual(len(updates), 20)
        self.assertEqual(updates[-1]["epoch"], 4)
        self.assertFalse((self.root / "rejected/epoch_016").exists())
        self.assertFalse((self.root / "rejected/prefix_match.json").exists())

    def test_runtime_source_and_admission_guards(self):
        self.assertEqual(len(self.control["installed_source_provenance"]), 9)
        for source in self.control["installed_source_provenance"]:
            self.assertEqual(_sha256(Path(source["installed_source_path"])), source["sha256"])
            self.assertEqual(_sha256(COREROOT / source["recorded_source_path"]), source["sha256"])
        for key in ("prefix_state_path", "prefix_match_path", "update_schedule_path"):
            path = self.treatment[key]
            self.assertNotIn("\\", path)
            self.assertFalse(Path(path).is_absolute())
            self.assertEqual(_sha256(COREROOT / path), self.treatment[key.replace("_path", "_sha256")])
        with self.assertRaisesRegex(ValueError, "completed control prefix"):
            self.fit("missing", "hold_base_after_epoch4")
        with self.assertRaises(FileExistsError):
            self.fit("control", "scheduled")
        config = {**self.config, "p_drop": .1}
        with self.assertRaisesRegex(ValueError, "base 0.05"):
            fit_fixed_trajectory(self.values[:160], self.labels[:160], None, None, config, self.root / "changed",
                                 horizon=16, endpoint_epochs=[4, 16], context=self.context, deadline_monotonic=self.deadline)


if __name__ == "__main__":
    unittest.main()
