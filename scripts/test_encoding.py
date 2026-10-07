"""Focused route-encoding regressions; no competition training or GPU use."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from encoding import RouteEncoder


class RouteEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        n = 75
        self.x = pd.DataFrame(
            {
                "Flight Distance": np.tile([100, 200, 300, 400, 500], 15),
                "Age": np.arange(n) % 9 + 20,
                "Class": np.tile(["Eco", "Business", None], 25),
                "Type of Travel": np.tile(["Personal", "Business", "__MISSING__"], 25),
                "Unrelated": np.arange(n),
            },
            index=pd.Index([f"passenger-{i * 17 % 31}" for i in range(n)], name="row_identity"),
        )
        self.y = (np.arange(n) % 7 < 3).astype(np.int8)
        self.config = {
            "keys": ["Flight Distance", ["Flight Distance", "Age"], ["Flight Distance", "Class"], "Age"],
            "smoothing": 20,
            "n_splits": 5,
            "seed": 911,
            "counts": True,
        }

    def test_same_heldout_fold_targets_cannot_affect_own_encoding(self) -> None:
        first = RouteEncoder(self.config)
        original = first.fit_transform(self.x, self.y)
        held_out = np.flatnonzero(first.fold_ids_ == 2)
        perturbed_y = self.y.copy()
        perturbed_y[held_out] = 1
        second = RouteEncoder(self.config)
        changed = second.fit_transform(self.x, perturbed_y)
        np.testing.assert_array_equal(first.fold_ids_, second.fold_ids_)
        np.testing.assert_array_equal(
            original.iloc[held_out][first.te_columns].to_numpy(),
            changed.iloc[held_out][second.te_columns].to_numpy(),
        )
        np.testing.assert_array_equal(
            original[first.count_columns].to_numpy(), changed[second.count_columns].to_numpy()
        )

    def test_singleton_keys_use_inner_training_prior(self) -> None:
        x = pd.DataFrame({"Flight Distance": np.arange(30)})
        y = (np.arange(30) % 3 == 0).astype(np.int8)
        encoder = RouteEncoder({"seed": 2, "counts": True})
        result = encoder.fit_transform(x, y)
        for fold in range(5):
            held = encoder.fold_ids_ == fold
            expected = np.float32(y[~held].mean())
            np.testing.assert_array_equal(result.loc[held, "route_te_0"].to_numpy(), expected)
        np.testing.assert_array_equal(result.route_count_0.to_numpy(), np.ones(30))

    def test_unknown_fallback_numeric_dtype_and_serialization(self) -> None:
        encoder = RouteEncoder(self.config)
        encoder.fit_transform(self.x, self.y)
        known = self.x.iloc[:9].copy()
        as_float = known.copy()
        as_float["Flight Distance"] = as_float["Flight Distance"].astype("Float32")
        as_float["Age"] = as_float["Age"].astype(np.float32)
        np.testing.assert_array_equal(
            encoder.transform(known)[encoder.te_columns + encoder.count_columns].to_numpy(),
            encoder.transform(as_float)[encoder.te_columns + encoder.count_columns].to_numpy(),
        )
        unknown = known.iloc[:1].copy()
        unknown["Flight Distance"] = 987654
        unknown["Age"] = 122
        result = encoder.transform(unknown)
        np.testing.assert_array_equal(result[encoder.te_columns].to_numpy(), np.full((1, 4), encoder.prior_, dtype=np.float32))
        np.testing.assert_array_equal(result[encoder.count_columns].to_numpy(), np.zeros((1, 4)))
        with tempfile.TemporaryDirectory(prefix="route-encoding-test-") as temporary:
            directory = Path(temporary)
            encoder.save(directory)
            restored = RouteEncoder.load(directory)
            pd.testing.assert_frame_equal(encoder.transform(as_float), restored.transform(as_float))
            pd.testing.assert_frame_equal(encoder.transform(unknown), restored.transform(unknown))

    def test_column_order_row_identity_and_writable_copy(self) -> None:
        untouched = self.x.copy(deep=True)
        encoder = RouteEncoder(self.config)
        encoded = encoder.fit_transform(self.x, self.y)
        pd.testing.assert_frame_equal(self.x, untouched)
        pd.testing.assert_frame_equal(encoded[self.x.columns], self.x)
        self.assertTrue(encoded.index.equals(self.x.index))
        reversed_x = self.x[list(reversed(self.x.columns))]
        pd.testing.assert_frame_equal(
            encoder.transform(self.x)[encoder.te_columns + encoder.count_columns],
            encoder.transform(reversed_x)[encoder.te_columns + encoder.count_columns],
        )
        encoded.iloc[0, encoded.columns.get_loc("Unrelated")] = -999
        self.assertEqual(self.x.iloc[0].Unrelated, 0)
        empty = encoder.transform(self.x.iloc[:0])
        self.assertEqual(empty.shape, (0, len(self.x.columns) + 8))

    def test_tuple_keys_and_missing_tokens_do_not_collide(self) -> None:
        x = pd.DataFrame(
            {
                "Class": ["a|b", "a", "__MISSING__", None] * 5,
                "Type of Travel": ["c", "b|c", "x", "x"] * 5,
            }
        )
        y = np.tile([0, 1, 0, 1], 5).astype(np.int8)
        encoder = RouteEncoder({"keys": [["Class", "Type of Travel"]], "smoothing": 0})
        encoder.fit_transform(x, y)
        result = encoder.transform(x.iloc[:4])
        np.testing.assert_array_equal(result.route_te_0.to_numpy(), np.array([0, 1, 0, 1]))
        self.assertEqual(len(encoder.maps_[0]), 4)

    def test_bad_labels_columns_and_infinite_keys_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RouteEncoder(self.config).fit_transform(self.x, self.y[:-1])
        with self.assertRaises(ValueError):
            RouteEncoder(self.config).fit_transform(self.x, np.full(len(self.x), 2))
        x = self.x.copy()
        x["Flight Distance"] = x["Flight Distance"].astype(float)
        x.iloc[0, x.columns.get_loc("Flight Distance")] = np.inf
        with self.assertRaises(ValueError):
            RouteEncoder(self.config).fit_transform(x, self.y)
        encoder = RouteEncoder(self.config)
        encoder.fit_transform(self.x, self.y)
        with self.assertRaises(ValueError):
            encoder.transform(self.x.assign(route_te_0=0))


if __name__ == "__main__":
    unittest.main()
