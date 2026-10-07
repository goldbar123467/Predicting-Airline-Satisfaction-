"""Generated filesystem fixtures only; no model or competition data access."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

import register_fixed_epoch_v1 as registration


class RegistrationTests(unittest.TestCase):
    def test_receipt_requires_exact_current_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            source = root / "scripts/source.py"
            source.write_text("x = 1\n")
            receipt = root / "receipt.json"
            value = {"status": "passed", "tests_passed": 4, "real_data_rows": 0,
                     "source_sha256": {"scripts\\source.py": registration.digest(source)}}
            receipt.write_text(json.dumps(value))
            registration.validate_test_receipt(root, receipt, {"scripts/source.py"}, 4)
            with self.assertRaises(ValueError):
                registration.validate_test_receipt(root, receipt, {"scripts/missing.py"}, 4)
            source.write_text("x = 2\n")
            with self.assertRaises(ValueError):
                registration.validate_test_receipt(root, receipt, {"scripts/source.py"}, 4)

    def test_parity_fails_nonfinite_out_of_bounds_and_false(self):
        good = {"parity_passed": True, "atol": 2e-6, "rtol": 1e-5,
                "max_absolute_error": 2e-7, "max_scaled_error": .1}
        registration.validate_parity(good)
        for update in ({"max_scaled_error": 1.01}, {"max_absolute_error": float("nan")},
                       {"parity_passed": False}, {"rtol": .1}, {"max_absolute_error": -1}):
            with self.assertRaises(ValueError):
                registration.validate_parity({**good, **update})

    def test_fixed_deadlines_and_scientific_contract(self):
        start = datetime(2026, 10, 3, tzinfo=timezone.utc)
        registry = registration.registry_payload({}, start)
        self.assertEqual(datetime.fromisoformat(registry["fit_deadline_utc"]), start + timedelta(minutes=90))
        self.assertEqual(datetime.fromisoformat(registry["delivery_deadline_utc"]), start + timedelta(minutes=120))
        self.assertEqual(len(registry["execution_order"]), 12)
        self.assertEqual(registry["development_rows"], 629671)
        self.assertEqual(registry["evaluation"], registration.EVALUATION)
        self.assertEqual(registry["arms"]["B"]["prefix_of"], "C")
        with self.assertRaises(ValueError):
            registration.registry_payload({}, datetime(2026, 10, 3))

    def test_exclusive_freeze_snapshots_and_wrapper_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            source = root / "scripts/generated.py"
            source.write_bytes(b"generated = True\n")
            input_path = root / "generated_input.json"
            input_path.write_text("{}")
            collection = {"source_hashes": {"scripts/generated.py": registration.digest(source)},
                          "input_hashes": {"generated_input.json": registration.digest(input_path)}}
            result = registration.freeze(root, collection)
            self.assertEqual(result["status"], "registered")
            registry_path = root / registration.OUTPUT / "registry.json"
            wrapper = registration.read(root / registration.WRAPPER)
            self.assertEqual(wrapper["registry_sha256"], registration.digest(registry_path))
            registry = registration.read(registry_path)
            self.assertEqual(len(registry["source_snapshot_hashes"]), 1)
            registration.check_hashes(root, registry["source_snapshot_hashes"])
            original = registry_path.read_bytes()
            with self.assertRaises(FileExistsError):
                registration.freeze(root, collection)
            self.assertEqual(registry_path.read_bytes(), original)

    def test_path_escape_and_duplicate_normalized_sources_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                registration.local(root, "../outside")
            (root / "scripts").mkdir()
            source = root / "scripts/x.py"
            source.write_text("x=0")
            sha = registration.digest(source)
            with self.assertRaises(ValueError):
                registration.check_hashes(root, {"scripts/x.py": sha, "scripts\\x.py": sha})


if __name__ == "__main__":
    unittest.main()
