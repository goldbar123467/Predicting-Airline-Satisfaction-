"""One bounded generated-data integration regression; no competition data."""
import unittest

from research_mixed_capture_v1 import run_checks


class MixedCaptureTests(unittest.TestCase):
    def test_eight_member_learned_preprocessing_capture_is_passive(self):
        result = run_checks()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(result["checks"].values()))
        self.assertGreater(result["preprocessing_inventory_count"], 0)
        self.assertEqual(result["schema"]["cardinalities"], [5, 25])
        self.assertTrue(result["capture"]["unknown_equals_unseen_exactly"])
        self.assertEqual(result["validation_batch_counts"], [48, 48])
        self.assertEqual(len(result["fixed_fit_probes"]), 16)
        self.assertLess(result["elapsed_seconds"], 300)


if __name__ == "__main__":
    unittest.main()
