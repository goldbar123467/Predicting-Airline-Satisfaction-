"""Fast synthetic tests for the isolated epoch-policy research prototype."""
import unittest

from research_epoch_control_v1 import EpochPolicy, run_checks, schedule_clock_comparison


class EpochControlTests(unittest.TestCase):
    def test_invalid_epoch_policy_is_rejected(self):
        for values in ((0, 0, 0), (4, 5, 4), (4, 4, 5), (4, 4, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                EpochPolicy(*values)

    def test_step_clock_is_distinct_from_epoch_clock(self):
        result = schedule_clock_comparison(377802, 419780, 256, 500, 4)
        self.assertEqual(result["selected_inner_updates"], 5900)
        self.assertEqual(result["equal_epoch_outer_updates"], 6556)
        self.assertAlmostEqual(result["equal_update_outer_epochs"], 5900 / 1639)
        self.assertGreater(result["fixed_update_clock_fraction_at_equal_epoch_stop"],
                           result["epoch_clock_selected_fraction"])

    def test_zero_full_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            schedule_clock_comparison(12, 20, 32, 4, 3)

    def test_installed_callback_preserves_exact_synthetic_prefix(self):
        result = run_checks()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["preserved_state_max_abs_difference"], 0.0)
        self.assertFalse(result["compressed_schedule_trace_exact"])


if __name__ == "__main__":
    unittest.main()
