"""Reusable generated-data CPU checks for the production fixed-epoch adapter."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch
from pytabkit.models.data.splits import RandomSplitter
from fixed_epoch_adapter_v1 import fit_fixed_trajectory, _configuration, _dataset, _hash
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


class FixedEpochAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fixed-epoch-adapter-test-")
        cls.root = Path(cls.temporary.name)
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
                context={"phase": "synthetic", "synthetic_trace": True, "case": name},
                deadline_monotonic=cls.deadline)
        (cls.root / "outer").mkdir()
        (cls.root / "outer" / "transform.json").write_text('{"external_transform":"preserve"}\n')
        cls.outer = fit_fixed_trajectory(
            cls.values[:160], cls.labels[:160], None, None, cls.config, cls.root / "outer",
            horizon=1, endpoint_epochs=[1], context={"phase": "outer"}, deadline_monotonic=cls.deadline)

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
        model = load_realmlp_categorical(Path(result["endpoints"]["1"]["path"]), device="cpu")
        prediction = model.predict_proba(self.values)
        chunked = np.concatenate([model.predict_proba(x) for x in np.array_split(self.values, 19)])
        np.testing.assert_allclose(prediction, chunked, atol=2e-6, rtol=1e-5)
        np.testing.assert_allclose(prediction.sum(1), 1, atol=1e-6, rtol=0)
        self.assertEqual(model.predict_proba(self.values[:0]).shape, (0, 2))
        self.assertLessEqual(model.metadata["native_export_max_abs_error"], 2e-6)

    def test_curves_and_seed_policy(self):
        result = self.results["captured"]
        curves = [json.loads(line) for line in Path(result["curves_path"]).read_text().splitlines()]
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
                                 context={"phase": "outer"}, deadline_monotonic=self.deadline)
        with self.assertRaises(ValueError):
            fit_fixed_trajectory(self.values[:160], self.labels[:160], self.values[160:], self.labels[160:],
                                 self.config, self.root / "forbidden", horizon=1, endpoint_epochs=[1],
                                 context={"phase": "outer"}, deadline_monotonic=self.deadline)
        with self.assertRaises(TimeoutError):
            fit_fixed_trajectory(self.values[:160], self.labels[:160], None, None, self.config,
                                 self.root / "expired", horizon=1, endpoint_epochs=[1],
                                 context={"phase": "outer"}, deadline_monotonic=time.monotonic() - 1)
        self.assertFalse((self.root / "expired").exists())
        defaults = _configuration({}, 16)
        self.assertEqual(defaults["hidden_sizes"], [512, 256, 128])
        self.assertEqual(defaults["ls_eps"], 0)
        self.assertEqual(defaults["n_ens"], 8)


if __name__ == "__main__":
    unittest.main()
