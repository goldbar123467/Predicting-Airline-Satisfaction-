"""Tiny generated Arrow fixtures; no model fitting or competition-data reads."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import prepare_fixed_epoch_cloud_v1 as prep


class CloudPreparationTests(unittest.TestCase):
    def test_development_filter_precedes_target_and_aux_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = pa.table({"id": np.arange(9, dtype=np.int64),
                "fold": [-1, 0, 1, 2, -1, 0, 1, 2, -1],
                "is_audit": [True, False, False, False, True, False, False, False, True]})
            pq.write_table(split, root / "split.parquet")
            development = prep.development_split(root / "split.parquet", 6)
            ids = development["id"].to_numpy()
            raw = pa.table({"id": np.arange(9, dtype=np.int64), "value": np.arange(9, dtype=np.float32),
                            "satisfaction": [999, 0, 1, 0, 999, 1, 0, 1, 999]})
            pq.write_table(raw, root / "raw.parquet")
            result = prep.stream_subset(root / "raw.parquet", root / "filtered.parquet", ids,
                                        ["id", "value", "satisfaction"], role="train", batch_size=2)
            self.assertEqual(result["rows"], 6)
            np.testing.assert_array_equal(pq.read_table(root / "filtered.parquet")["id"].to_numpy(), ids)
            auxiliary = {"id": np.arange(9, dtype=np.int64)}
            for index in range(13):
                values = np.arange(9, dtype=np.float32)
                values[[0, 4, 8]] = np.nan
                auxiliary[f"orig_aux_{index:02d}"] = values
            pq.write_table(pa.table(auxiliary), root / "aux.parquet")
            aux = prep.stream_subset(root / "aux.parquet", root / "aux_filtered.parquet", ids,
                                     list(auxiliary), role="aux", batch_size=2)
            self.assertEqual(aux["id_sha256"], result["id_sha256"])

    def test_subset_rejects_wrong_order_missing_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, actual in enumerate(([2, 1, 3], [1, 2], [1, 2, 2])):
                source = root / f"source{index}.parquet"
                pq.write_table(pa.table({"id": actual, "satisfaction": [0] * len(actual)}), source)
                with self.assertRaises(ValueError):
                    prep.stream_subset(source, root / f"out{index}.parquet", np.array([1, 2, 3]),
                                       ["id", "satisfaction"], role="train", batch_size=1)

    def test_written_subset_is_exclusive_and_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.parquet"
            pq.write_table(pa.table({"id": [1, 2], "satisfaction": [0, 8]}), source)
            with self.assertRaises(ValueError):
                prep.stream_subset(source, root / "bad.parquet", np.array([1, 2]), ["id", "satisfaction"], role="train")
            with self.assertRaises(FileExistsError):
                prep.stream_subset(source, root / "bad.parquet", np.array([1, 2]), ["id", "satisfaction"], role="train")

    def test_archive_has_only_explicit_files_and_roundtrip_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload"
            payload.mkdir()
            (payload / "allowed.txt").write_bytes(b"generated fixture")
            (payload / "excluded.txt").write_bytes(b"not in allowlist")
            prep.archive_payload(payload, root / "payload.zip", ["allowed.txt"])
            with zipfile.ZipFile(root / "payload.zip") as archive:
                self.assertEqual(archive.namelist(), ["payload/allowed.txt"])
                self.assertEqual(hashlib.sha256(archive.read("payload/allowed.txt")).hexdigest(), prep.sha(payload / "allowed.txt"))
            with self.assertRaises(FileExistsError):
                prep.archive_payload(payload, root / "payload.zip", ["allowed.txt"])
            with self.assertRaises(ValueError):
                prep.archive_payload(payload, root / "bad.zip", ["../outside"])


if __name__ == "__main__":
    unittest.main()
