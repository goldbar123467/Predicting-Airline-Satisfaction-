"""Synthetic-only live-state isolation during actual native export."""
import unittest

from research_epoch_capture_v1 import run_capture_checks


class EpochCaptureTests(unittest.TestCase):
    def test_export_is_passive_and_prefix_artifact_stays_immutable(self):
        result = run_capture_checks()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(result["capture"]["live_state_unchanged"].values()))
        self.assertTrue(result["capture"]["storage_disjoint"])
        self.assertTrue(result["artifact_hash_stable_after_continuation"])
        self.assertTrue(result["batch_rng_and_schedule_paths_equal"])
        self.assertGreater(result["capture"]["portable_rewrite_counts"]["embedding"], 0)
        self.assertGreater(result["capture"]["portable_rewrite_counts"]["onehot"], 0)
        self.assertLessEqual(result["capture"]["reversed_batch_max_abs_difference"], 2e-6)
        self.assertLessEqual(result["capture"]["one_row_max_abs_difference"], 2e-6)
        self.assertEqual(result["maximum_absolute_differences"]["final_optimizer"], 0.0)


if __name__ == "__main__":
    unittest.main()
