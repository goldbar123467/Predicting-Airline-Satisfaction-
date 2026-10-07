"""Generated CPU checks for portable receipts and unchanged cloud training math."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch
from pytabkit.models.data.splits import RandomSplitter
from fixed_epoch_cloud_adapter_v1 import (COREROOT, fit_fixed_trajectory, _configuration,
                                        _dataset, _hash, _source_provenance)
from fixed_epoch_adapter_v1 import fit_fixed_trajectory as frozen_fit
from realmlp import _sha256
from realmlp_categorical import _fit_schema, load_realmlp_categorical


def toy_data():
    rng = np.random.default_rng(97531)
    values = rng.normal(size=(192, 8)).astype(np.float32)
    values[:, :6] = values[:, :6] * np.array([.4, 2, 4, 1, 3, 5], dtype=np.float32) + 2
    values[:, 6] = np.arange(192) % 4
    values[:, 7] = np.arange(192) % 24
    values[160:, 6:] = [999, 777]
    labels = (values[:, 0] + .2 * values[:, 1] > 2.4).astype(np.int64)
    return values, labels


class FixedEpochCloudAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fixed_epoch_cloud_test_", dir=COREROOT / "artifacts")
        cls.root = Path(cls.temporary.name)
        cls.campaign = cls.root.name
        cls.values, cls.labels = toy_data()
        cls.config = {"device": "cpu", "seed": 20261005, "threads": 1, "hidden_sizes": [16, 8],
                      "batch_size": 32, "eval_batch_size": 37, "categorical_indices": [6, 7]}
        cls.deadline = time.monotonic() + 120
        cls.results = {}
        # Both arms have identical passive curves. Only the additional prefix
        # clone/export differs, so exact final trajectory equality is meaningful.
        for name, endpoints in (("reference", [4]), ("captured", [1, 4]), ("stopped", [1])):
            cls.results[name] = fit_fixed_trajectory(
                cls.values[:160], cls.labels[:160], cls.values[160:], cls.labels[160:],
                cls.config, cls.root / name, horizon=4, endpoint_epochs=endpoints,
                context={"campaign": cls.campaign, "phase": "synthetic", "synthetic_trace": True, "case": name},
                deadline_monotonic=cls.deadline)
        (cls.root / "outer").mkdir()
        (cls.root / "outer" / "transform.json").write_text('{"external_transform":"preserve"}\n')
        cls.outer = fit_fixed_trajectory(
            cls.values[:160], cls.labels[:160], None, None, cls.config, cls.root / "outer",
            horizon=1, endpoint_epochs=[1], context={"campaign": cls.campaign, "phase": "outer"}, deadline_monotonic=cls.deadline)
        cls.baseline = frozen_fit(
            cls.values[:160], cls.labels[:160], cls.values[160:], cls.labels[160:],
            cls.config, cls.root / "frozen_reference", horizon=4, endpoint_epochs=[4],
            context={"campaign": cls.campaign, "phase": "synthetic", "synthetic_trace": True, "case": "reference"},
            deadline_monotonic=cls.deadline)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_prefix_export_and_continuation_are_exact(self):
        left, right = self.results["reference"], self.results["captured"]
        for key in ("initial_network_sha256", "final_network_sha256", "final_optimizer_sha256",
                    "final_rng_sha256", "synthetic_batch_trace_sha256", "training_index_sha256"):
            self.assertEqual(left[key], right[key], key)
        self.assertEqual(right["endpoints"]["1"]["network_sha256"], self.results["stopped"]["final_network_sha256"])
        self.assertNotEqual(right["endpoints"]["1"]["network_sha256"], right["final_network_sha256"])
        self.assertEqual(right["executed_epochs"], 4)
        self.assertEqual(right["optimizer_updates"], 20)
        self.assertEqual(self.results["stopped"]["horizon"], 4)
        self.assertEqual(self.results["stopped"]["executed_epochs"], 1)

    def test_native_loader_and_heldout_category_contract(self):
        result = self.results["captured"]
        self.assertNotIn(999, result["input_schema"]["vocabularies"][0])
        self.assertNotIn(777, result["input_schema"]["vocabularies"][1])
        model = load_realmlp_categorical(COREROOT / result["endpoints"]["1"]["path"], device="cpu")
        prediction = model.predict_proba(self.values)
        chunked = np.concatenate([model.predict_proba(x) for x in np.array_split(self.values, 19)])
        np.testing.assert_allclose(prediction, chunked, atol=2e-6, rtol=1e-5)
        np.testing.assert_allclose(prediction.sum(1), 1, atol=1e-6, rtol=0)
        self.assertEqual(model.predict_proba(self.values[:0]).shape, (0, 2))
        self.assertLessEqual(model.metadata["native_export_max_abs_error"], 2e-6)

    def test_curves_and_seed_policy(self):
        result = self.results["captured"]
        curves = [json.loads(line) for line in (COREROOT / result["curves_path"]).read_text().splitlines()]
        self.assertEqual(len(curves), 4)
        self.assertTrue(all(row["monitor"] is not None for row in curves))
        self.assertTrue(all(row["train_eval_probe"] is not None for row in curves))
        self.assertTrue(all(all(row["endpoint_live_state_unchanged"].values()) for row in curves
                            if row["endpoint_live_state_unchanged"] is not None))
        expected = int(np.random.RandomState(20261005).randint(0, 2**31 - 1))
        self.assertEqual(result["sub_split_seed"], expected)
        self.assertEqual(result["training_index_sha256"], _hash(torch.arange(160)))
        schema = _fit_schema(self.values[:160], [6, 7])
        dataset = _dataset(self.values[:160], schema, self.labels[:160], "cpu")
        permutation = RandomSplitter(20261005, first_fraction=1).get_idxs(dataset)[0]
        self.assertEqual(self.outer["training_index_sha256"], _hash(permutation))
        self.assertNotEqual(_hash(permutation), _hash(torch.arange(160)))
        self.assertFalse(result["resolved_config"]["use_best_epoch"])
        self.assertEqual(result["resolved_factory_config"]["plr_act_name"], "linear")

    def test_fail_closed_and_preserve_external_transform(self):
        self.assertEqual((self.root / "outer" / "transform.json").read_text(), '{"external_transform":"preserve"}\n')
        with self.assertRaises(FileExistsError):
            fit_fixed_trajectory(self.values[:160], self.labels[:160], None, None, self.config,
                                 self.root / "outer", horizon=1, endpoint_epochs=[1],
                                 context={"campaign": self.campaign, "phase": "outer"}, deadline_monotonic=self.deadline)
        with self.assertRaises(ValueError):
            fit_fixed_trajectory(self.values[:160], self.labels[:160], self.values[160:], self.labels[160:],
                                 self.config, self.root / "forbidden", horizon=1, endpoint_epochs=[1],
                                 context={"campaign": self.campaign, "phase": "outer"}, deadline_monotonic=self.deadline)
        with self.assertRaises(TimeoutError):
            fit_fixed_trajectory(self.values[:160], self.labels[:160], None, None, self.config,
                                 self.root / "expired", horizon=1, endpoint_epochs=[1],
                                 context={"campaign": self.campaign, "phase": "outer"}, deadline_monotonic=time.monotonic() - 1)
        self.assertFalse((self.root / "expired").exists())
        defaults = _configuration({}, 16)
        self.assertEqual(defaults["hidden_sizes"], [512, 256, 128])
        self.assertEqual(defaults["ls_eps"], 0)
        self.assertEqual(defaults["n_ens"], 8)

    def test_exact_equality_to_frozen_training(self):
        result = self.results["reference"]
        for key in ("initial_network_sha256", "final_network_sha256", "final_optimizer_sha256",
                    "final_rng_sha256", "synthetic_batch_trace_sha256", "training_index_sha256",
                    "optimizer_updates", "constructor", "resolved_factory_config"):
            self.assertEqual(self.baseline[key], result[key], key)
        frozen = load_realmlp_categorical(Path(self.baseline["endpoints"]["4"]["path"]), device="cpu")
        cloud = load_realmlp_categorical(COREROOT / result["endpoints"]["4"]["path"], device="cpu")
        np.testing.assert_array_equal(frozen.predict_proba(self.values), cloud.predict_proba(self.values))

    def test_portable_paths_actual_source_snapshot_and_mismatch(self):
        result = self.results["captured"]
        self.assertEqual(result["artifact_path_base"], "runtime_workspace")
        paths = [result["trajectory_path"], result["curves_path"],
                 *[value["path"] for value in result["endpoints"].values()], *result["source_sha256"]]
        for path in paths:
            self.assertFalse(Path(path).is_absolute(), path)
            self.assertNotIn("\\", path)
            self.assertNotIn("..", Path(path).parts)
            self.assertTrue((COREROOT / path).exists(), path)
        self.assertEqual(len(result["installed_source_provenance"]), 3)
        for source in result["installed_source_provenance"]:
            self.assertEqual(_sha256(Path(source["installed_source_path"])), source["sha256"])
            self.assertEqual(_sha256(COREROOT / source["recorded_source_path"]), source["sha256"])
        first = COREROOT / result["installed_source_provenance"][0]["recorded_source_path"]
        original = first.read_bytes()
        try:
            first.write_bytes(original + b"\n# synthetic mismatch test\n")
            with self.assertRaises(ValueError):
                _source_provenance(self.root)
        finally:
            first.write_bytes(original)
        sources, _ = _source_provenance(self.root)
        self.assertEqual(sources, result["source_sha256"])


if __name__ == "__main__":
    unittest.main()
