"""Synthetic-only exact epoch-boundary resume acceptance check."""
import unittest

from research_epoch_resume_v1 import run_resume_checks


class EpochResumeTests(unittest.TestCase):
    def test_augmented_resume_and_unaugmented_failure_are_measured(self):
        result = run_resume_checks()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["ordinary_saved_optimizer_state_count"], 0)
        self.assertGreater(result["augmented_saved_inner_optimizer_state_count"], 0)
        self.assertTrue(result["results"]["ordinary"]["load_failed"])
        self.assertEqual(result["results"]["ordinary"]["error"]["message"], "__dict__")
        self.assertEqual(result["results"]["augmented"]["final_state_max_abs_difference"], 0.0)
        self.assertEqual(result["augmented_final_optimizer_max_abs_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
