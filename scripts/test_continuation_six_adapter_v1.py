"""Generated-data CPU oracles for six fixed continuation policies; no real data."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from pytabkit.models.training.scheduling import Schedule

from continuation_six_adapter_v1 import (
    COREROOT, POLICY_IDS, _hash, _LinearCoolingSchedule, _changed_optimizer_step,
    _parameter_roles, _ContinuationState, fit_fixed_trajectory, policy_for, validate_policy,
)
from fixed_epoch_dropout_adapter_v1 import fit_fixed_trajectory as old_fit
from realmlp import _sha256
from realmlp_categorical import load_realmlp_categorical


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class ScalarOracles(unittest.TestCase):
    def test_policy_exact_keys_and_copy(self):
        for name in ("control", *POLICY_IDS):
            self.assertEqual(validate_policy(policy_for(name)), policy_for(name))
            bad = policy_for(name)
            bad["undeclared"] = 1
            with self.assertRaises(ValueError):
                validate_policy(bad)
        with self.assertRaises(ValueError):
            policy_for("other")
        value = policy_for("freeze_embeddings")
        value["frozen_roles"].append("unknown")
        self.assertNotEqual(value, policy_for("freeze_embeddings"))

    def test_scope_cooling_continuity_and_end(self):
        class Example(Schedule):
            def __init__(self, scale):
                self.scale, self.t = scale, 0.
            def update(self, learner):
                self.t = learner.progress.epoch_float
            def get_value(self):
                return self.scale * (1 + self.t / 10.)
        for scale in (.25, 1., 3.):
            original = Example(scale)
            wrapper = _LinearCoolingSchedule(original)
            for t in (0., 3.999999, 4., 4.25, 10., 15.9, 16., 17.):
                wrapper.update(types.SimpleNamespace(progress=types.SimpleNamespace(epoch_float=t)))
                expected = scale * (1 + t / 10.) if t < 4 else scale * 1.4 * max(0., (16-t)/12)
                self.assertAlmostEqual(wrapper.get_value(), expected, places=14)
                self.assertEqual(original.t, t, "Boundary probe must not move the live clock")
            self.assertAlmostEqual(wrapper.anchor, 1.4 * scale)

    def test_manual_decay_scalar_and_frozen_oracle(self):
        for a, b in ((.25, .5), (.1151, 1.), (1., 0.), (1., 1.)):
            for corrected in (False, True):
                for frozen in (False, True):
                    parameter = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
                    parameter.hyper_factors = {"lr": a, "wd": b}
                    parameter.grad = torch.tensor([7.], dtype=torch.float64)
                    applied = []
                    optimizer = types.SimpleNamespace(
                        hyper_mappings=[("lr", "lr", .001), ("wd", None, 0.)],
                        opt=types.SimpleNamespace(param_groups=[{"params": [parameter]}]),
                        get_hyper_values=lambda name, index: .2*a if name == "lr" else .3*b,
                        _opt_step_with_loss=lambda loss: applied.append(parameter.grad))
                    _changed_optimizer_step(optimizer, corrected=corrected,
                                            frozen_ids={id(parameter)} if frozen else set())
                    shrink = 0. if frozen else .2*.3*a*b*(1. if corrected else a*b)
                    self.assertAlmostEqual(parameter.item(), 2*(1-shrink), places=14)
                    self.assertEqual(optimizer.opt.param_groups[0]["lr"], .2*a)
                    if frozen:
                        self.assertIsNone(applied[0])

    def test_unknown_trainable_role_fails_closed(self):
        layer = torch.nn.Linear(2, 1)
        for parameter in layer.parameters():
            parameter.context = types.SimpleNamespace(scope="/unknown")
        with self.assertRaisesRegex(ValueError, "Unknown trainable parameter"):
            _parameter_roles(types.SimpleNamespace(model=layer))

    def test_ema_recurrence_one_epoch_half_life(self):
        # Exercise the same updater with a separately calculated closed form.
        decay = 2**(-1/4)
        raw = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
        state = types.SimpleNamespace(ema={"p": raw.detach().clone()}, ema_decay=decay,
                                      parameters={"p":raw}, ema_updates=0)
        sequence = [3., 7., 4., -2., 9., 6., 0., 1.]
        for count, value in enumerate(sequence, 1):
            with torch.no_grad():
                raw.fill_(value)
            _ContinuationState.after_update(state)
            expected = decay**count*2 + (1-decay)*sum(decay**(count-index)*item for index,item in enumerate(sequence[:count],1))
            self.assertAlmostEqual(state.ema["p"].item(), expected, places=14)
            self.assertEqual(state.ema_updates, count)
            self.assertEqual(raw.item(), value)
        self.assertAlmostEqual(decay**4, .5, places=15)


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fixed_epoch_continuation_test_", dir=COREROOT / "artifacts")
        cls.root = Path(cls.temporary.name)
        rng = np.random.default_rng(97531)
        cls.values = rng.normal(size=(160, 8)).astype(np.float32)
        cls.values[:, :6] = cls.values[:, :6] * np.array([.4, 2, 4, 1, 3, 5], dtype=np.float32) + 2
        cls.values[:, 6] = np.arange(160) % 4
        cls.values[:, 7] = np.arange(160) % 24
        cls.values[128:, 6:] = [999, 777]
        cls.labels = (cls.values[:, 0] + .2 * cls.values[:, 1] > 2.4).astype(np.int64)
        cls.config = {"device": "cpu", "seed": 20261005, "threads": 1, "hidden_sizes": [16, 8],
                      "batch_size": 32, "eval_batch_size": 37, "categorical_indices": [6, 7]}
        cls.deadline = time.monotonic() + 240
        cls.context = {"campaign": cls.root.name, "phase": "synthetic", "synthetic_trace": True}
        cls.results = {"control": cls.fit("control")}
        for name in POLICY_IDS:
            cls.results[name] = cls.fit(name)
        old_fit(cls.values[:128], cls.labels[:128], cls.values[128:], cls.labels[128:], cls.config,
                cls.root / "old_scheduled", horizon=16, endpoint_epochs=[4, 16], context=cls.context,
                deadline_monotonic=cls.deadline, dropout_policy="scheduled")
        cls.old_control = old_fit(cls.values[:128], cls.labels[:128], cls.values[128:], cls.labels[128:], cls.config,
                cls.root / "old_held", horizon=16, endpoint_epochs=[4, 16], context=cls.context,
                deadline_monotonic=cls.deadline, dropout_policy="hold_base_after_epoch4",
                prefix_reference_dir=cls.root / "old_scheduled")
        cls.updates = {name: rows(cls.root / name / "update_schedule.jsonl") for name in cls.results}
        cls.optimizers = {name: rows(cls.root / name / "optimizer_updates.jsonl") for name in cls.results}

    @classmethod
    def fit(cls, policy, name=None, reference=None):
        return fit_fixed_trajectory(cls.values[:128], cls.labels[:128], cls.values[128:], cls.labels[128:],
            cls.config, cls.root / (name or policy), horizon=16, endpoint_epochs=[4, 16], context=cls.context,
            deadline_monotonic=cls.deadline, continuation_policy=policy_for(policy),
            prefix_reference_dir=reference if reference is not None else (None if policy == "control" else cls.root / "control"))

    @classmethod
    def tearDownClass(cls):
        evidence = os.environ.get("CONTINUATION_SIX_TEST_EVIDENCE")
        if evidence:
            destination = Path(evidence).resolve()
            if not destination.is_relative_to((COREROOT / "artifacts").resolve()):
                raise ValueError("Generated test evidence must stay in artifact namespace")
            summary = {"scope":"generated CPU data only; not statistical evidence", "generated_rows":160,
                       "fit_rows":128,"monitor_rows":32,"real_data_rows":0,"cuda_used":False,
                       "results":cls.results,
                       "endpoint_metadata":{name:{str(epoch):read(cls.root/name/f"epoch_{epoch:03d}"/"metadata.json") for epoch in (4,16)} for name in cls.results},
                       "optimizer_evidence":cls.optimizers,
                       "update_schedule_bytes":{name:(cls.root/name/"update_schedule.jsonl").stat().st_size for name in cls.results},
                       "optimizer_update_bytes":{name:(cls.root/name/"optimizer_updates.jsonl").stat().st_size for name in cls.results}}
            with destination.open("x",encoding="utf-8") as stream:
                json.dump(summary,stream,allow_nan=False,separators=(",",":"))
        cls.temporary.cleanup()

    def metadata(self, name, epoch):
        return read(self.root / name / f"epoch_{epoch:03d}" / "metadata.json")

    def test_control_preserves_frozen_held_dropout_math_rng(self):
        control = self.results["control"]
        for key in ("initial_network_sha256", "final_network_sha256", "final_optimizer_sha256", "final_rng_sha256",
                    "synthetic_batch_trace_sha256", "training_index_sha256", "optimizer_updates", "constructor", "resolved_factory_config"):
            self.assertEqual(control[key], self.old_control[key], key)
        for epoch in ("4", "16"):
            self.assertEqual(control["endpoints"][epoch]["network_sha256"], self.old_control["endpoints"][epoch]["network_sha256"])

    def test_all_six_full_prefix_and_applied_traces_match(self):
        control = read(self.root / "control/prefix_state.json")
        self.assertEqual(len(control["signature"]), 19)
        for name in POLICY_IDS:
            prefix = read(self.root / name / "prefix_state.json")
            match = read(self.root / name / "prefix_match.json")
            for key in ("signature", "signature_sha256", "native_probe_probability_sha256", "update_schedule_prefix_sha256"):
                self.assertEqual(prefix[key], control[key], (name, key))
            self.assertTrue(match["gate_completed_before_epoch5"])
            self.assertTrue(all(match["components_equal"].values()))
            self.assertEqual(match["optimizer_updates_at_gate"], 16)
            self.assertEqual(self.updates[name][:16], self.updates["control"][:16])
            for left, right in zip(self.optimizers[name][:16], self.optimizers["control"][:16]):
                self.assertEqual({k:v for k,v in left.items() if k != "continuation_policy"},
                                 {k:v for k,v in right.items() if k != "continuation_policy"})
            self.assertEqual(self.metadata(name, 4)["parameter_sha256"], self.metadata("control", 4)["parameter_sha256"])

    def test_batch_order_dropout_and_endpoint_guard(self):
        for name, result in self.results.items():
            self.assertEqual(result["optimizer_updates"], 64)
            self.assertEqual(result["consumed_batch_order_sha256"], self.results["control"]["consumed_batch_order_sha256"])
            for left, right in zip(self.updates[name], self.updates["control"]):
                self.assertEqual(left["batch_order"], right["batch_order"])
                self.assertEqual(left["p_drop_multipliers"], right["p_drop_multipliers"])
                if left["epoch"] > 4:
                    self.assertTrue(all(value == .05 for value in left["effective_scope_p_drop"].values()))
            for curve in rows(self.root / name / "curves.jsonl"):
                if curve["epoch"] in (4, 16):
                    self.assertTrue(all(curve["endpoint_live_state_unchanged"].values()))

    def test_linear_lr_actual_group_anchor_and_decay_coupling(self):
        first = self.optimizers["linear_lr"][16]
        for row, control in zip(self.optimizers["linear_lr"], self.optimizers["control"]):
            for index, (group, baseline) in enumerate(zip(row["groups"], control["groups"])):
                self.assertEqual(group["getter_wd"], baseline["getter_wd"])
                if row["epoch"] <= 4:
                    self.assertEqual(group["getter_lr"], baseline["getter_lr"])
                else:
                    expected = first["groups"][index]["getter_lr"] * (16-row["epoch_float_before"])/12
                    self.assertAlmostEqual(group["getter_lr"], expected, places=14)
                expected_decay = group["getter_lr"]*group["getter_wd"]*group["lr_factor"]*group["wd_factor"]
                self.assertAlmostEqual(group["decay_coefficient"], expected_decay, places=15)

    def test_decay_correction_only_after_four_and_getter_mapping(self):
        for name, trace in self.optimizers.items():
            for row in trace:
                for group in row["groups"]:
                    for hyper in ("lr", "wd"):
                        getter = group["getters"][hyper]
                        self.assertAlmostEqual(getter["unfactored_value"], getter["base_value"]*getter["schedule_value"], places=14)
                        self.assertAlmostEqual(group["getter_"+hyper], getter["unfactored_value"]*group[hyper+"_factor"], places=14)
                    correction = name == "decay_correction" and row["epoch"] > 4
                    self.assertEqual(group["decay_factor_applications"], 1 if correction else 2)
                    coefficient = 0. if group["frozen"] else group["getter_lr"]*group["getter_wd"]*(1. if correction else group["lr_factor"]*group["wd_factor"])
                    self.assertAlmostEqual(group["decay_coefficient"], coefficient, places=15)

    def test_label_smoothing_applied_getter_and_uniform_targets(self):
        for name, trace in self.optimizers.items():
            for row in trace:
                scopes = row["label_smoothing_scopes"]
                self.assertTrue(scopes)
                expected = .05 if name == "label_smoothing" and row["epoch"] > 4 else 0.
                for scope in scopes:
                    self.assertEqual(scope["effective_ls_eps"], expected)
                    self.assertEqual(scope["negative_positive_targets"], [expected/2, 1-expected/2])
                    self.assertTrue(scope["distribution_uniform_binary"])
        self.assertNotEqual(self.results["label_smoothing"]["final_network_sha256"], self.results["control"]["final_network_sha256"])

    def test_freeze_exact_roles_unchanged_tensor_hashes_no_decay(self):
        for name in ("freeze_embeddings", "head_only"):
            result = self.results[name]
            state = result["continuation_state"]
            roles = {item["name"]: item for item in state["parameter_roles"]}
            frozen = set(state["frozen_parameters"])
            original = {key for key, value in roles.items() if value["original_requires_grad"]}
            expected = {key for key in original if roles[key]["role"] in policy_for(name).get("frozen_roles", [])} if name == "freeze_embeddings" else {key for key in original if roles[key]["role"] != "final_affine"}
            self.assertEqual(frozen, expected)
            self.assertTrue(frozen)
            self.assertTrue(original-frozen)
            self.assertTrue(all(state["frozen_parameters_unchanged"].values()))
            e4, e16 = self.metadata(name, 4), self.metadata(name, 16)
            for parameter in frozen:
                self.assertEqual(state["frozen_initial_sha256"][parameter], e4["parameter_sha256"][parameter])
                self.assertEqual(state["frozen_final_sha256"][parameter], e16["parameter_sha256"][parameter])
                self.assertEqual(e4["parameter_sha256"][parameter], e16["parameter_sha256"][parameter])
            self.assertTrue(any(e4["parameter_sha256"][parameter] != e16["parameter_sha256"][parameter] for parameter in original-frozen))
            for row in self.optimizers[name][16:]:
                for group in row["groups"]:
                    self.assertEqual(group["frozen"], group["name"] in frozen)
                    if group["frozen"]:
                        self.assertFalse(group["requires_grad"])
                        self.assertEqual(group["decay_coefficient"], 0.)

    def test_ema_raw_training_identical_export_is_averaged(self):
        result, control = self.results["ema"], self.results["control"]
        for key in ("final_network_sha256", "final_optimizer_sha256", "final_rng_sha256", "synthetic_batch_trace_sha256"):
            self.assertEqual(result[key], control[key], key)
        state = result["continuation_state"]
        self.assertEqual(state["ema_updates"], 48)
        self.assertEqual(state["ema_decay_per_update"], 2**(-1/4))
        metadata = self.metadata("ema", 16)
        self.assertEqual(metadata["export_kind"], "ema_trainable_parameters")
        self.assertEqual(metadata["raw_training_network_sha256"], control["final_network_sha256"])
        self.assertNotEqual(metadata["network_sha256"], metadata["raw_training_network_sha256"])
        for parameter, digest in state["ema_parameter_sha256"].items():
            self.assertEqual(metadata["parameter_sha256"][parameter], digest)
        self.assertEqual(self.metadata("ema", 4)["export_kind"], "raw_training_endpoint")

    def test_native_export_runtime_sources_and_unknown_categories(self):
        for name, result in self.results.items():
            self.assertEqual(len(result["installed_source_provenance"]), 12)
            self.assertNotIn(999, result["input_schema"]["vocabularies"][0])
            self.assertNotIn(777, result["input_schema"]["vocabularies"][1])
            self.assertIn("scripts/continuation_six_policy_v1.py", result["source_sha256"])
            for source in result["installed_source_provenance"]:
                self.assertEqual(_sha256(COREROOT/source["recorded_source_path"]), source["sha256"])
            for epoch, endpoint in result["endpoints"].items():
                folder = COREROOT / endpoint["path"]
                self.assertEqual(_sha256(folder / "metadata.json"), endpoint["metadata_sha256"])
                self.assertEqual(_sha256(folder / "graph.pt"), endpoint["graph_sha256"])
                model = load_realmlp_categorical(folder, device="cpu")
                with torch.jit.optimized_execution(False):
                    entire = model.predict_proba(self.values)
                    chunked = np.concatenate([model.predict_proba(batch) for batch in np.array_split(self.values, 13)])
                np.testing.assert_allclose(entire, chunked, atol=2e-6, rtol=1e-5)

    def test_corrupted_prefix_rejected_before_any_changed_update(self):
        reference = self.root / "bad_reference"
        reference.mkdir()
        prefix = read(self.root / "control/prefix_state.json")
        prefix["signature"]["cpu_rng"] = "0"*64
        prefix["signature_sha256"] = _hash(prefix["signature"])
        (reference / "prefix_state.json").write_text(json.dumps(prefix), encoding="utf-8")
        trajectory = copy.deepcopy(self.results["control"])
        trajectory["prefix_state_sha256"] = _sha256(reference / "prefix_state.json")
        (reference / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "full-state prefix mismatch"):
            self.fit("linear_lr", "rejected", reference)
        failed = read(self.root / "rejected/trajectory.json")
        self.assertEqual(failed["executed_epochs"], 4)
        self.assertEqual(len(rows(self.root / "rejected/optimizer_updates.jsonl")), 16)
        self.assertFalse((self.root / "rejected/epoch_016").exists())

    def test_admission_and_cuda_policy_fail_before_gpu_use(self):
        with self.assertRaises(FileExistsError):
            self.fit("control")
        with self.assertRaisesRegex(ValueError, "Only treatment"):
            fit_fixed_trajectory(self.values[:128], self.labels[:128], None, None, self.config,
                self.root / "missing", horizon=16, endpoint_epochs=[4, 16], context=self.context,
                deadline_monotonic=self.deadline, continuation_policy=policy_for("ema"))
        with patch("torch.are_deterministic_algorithms_enabled", return_value=False), patch("torch.cuda.is_available") as probe:
            with self.assertRaisesRegex(RuntimeError, "strict process-start"):
                fit_fixed_trajectory(self.values[:128], self.labels[:128], None, None, {**self.config, "device":"cuda"},
                    self.root / "gpu_reject", horizon=16, endpoint_epochs=[4,16], context=self.context,
                    deadline_monotonic=self.deadline, continuation_policy=policy_for("control"))
            probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
