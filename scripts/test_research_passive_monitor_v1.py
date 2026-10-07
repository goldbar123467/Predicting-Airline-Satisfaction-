"""Synthetic integration regressions for monitor-only endpoint controls."""
import unittest

from research_passive_monitor_v1 import run_checks


class PassiveMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_checks()

    def test_validation_restores_best_even_without_early_termination(self):
        deltas = self.result["maximum_absolute_differences"]
        self.assertEqual(deltas["legacy_final_vs_epoch2"], 0.0)
        self.assertGreater(deltas["legacy_final_vs_epoch4"], 1e-6)
        self.assertEqual(self.result["best_epochs"], [2, 2, 2])
        self.assertTrue(all(count > 0 for count in self.result["validation_batches"]))

    def test_disabling_best_epoch_preserves_literal_endpoint(self):
        self.assertEqual(self.result["maximum_absolute_differences"]["fixed_final_vs_epoch4"], 0.0)
        self.assertTrue(self.result["legacy_and_fixed_trajectory_before_fit_end_equal"])
        self.assertEqual(self.result["checkpoint_callback_counts"], [1, 0, 0])

    def test_extra_passive_telemetry_preserves_training_path_and_snapshot(self):
        self.assertEqual(self.result["maximum_absolute_differences"]["telemetry_final_state"], 0.0)
        self.assertEqual(self.result["maximum_absolute_differences"]["telemetry_probabilities"], 0.0)
        self.assertTrue(self.result["telemetry_batch_schedule_rng_path_equal"])
        self.assertTrue(self.result["telemetry_final_rng_equal"])
        self.assertTrue(self.result["snapshot_immutable_after_later_updates"])


if __name__ == "__main__":
    unittest.main()
