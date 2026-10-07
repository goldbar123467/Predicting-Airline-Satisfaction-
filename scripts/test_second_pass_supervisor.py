"""Campaign namespace and release dispatch tests; all fixtures are synthetic."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import supervisor


class QueueWaitObserved(Exception):
    pass


class SecondPassSupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="airline-second-pass-supervisor-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in {
            "ROOT": self.root, "STATE_DIR": self.root / "state",
            "STATE_PATH": self.root / "state/run_state.json",
            "STOP_PATH": self.root / "state/STOP",
            "LOG_DIR": self.root / "logs/supervisor", "STOP_REQUESTED": False,
        }.items():
            stack.enter_context(patch.object(supervisor, name, value))
        self.output = io.StringIO()
        stack.enter_context(redirect_stdout(self.output))
        self.config_path = self.root / "configs/second_pass.json"
        self.config = {
            "campaign": "second_pass", "runs": [], "wait_for_cutoff": True,
            "stop_new_runs_utc": "2026-10-02T15:45:00Z",
            "finalize_utc": "2026-10-02T16:00:00Z",
            "deadline_utc": "2026-10-02T17:00:00Z",
        }
        self.write_config()
        self.calls = []

    def write_config(self):
        supervisor.atomic_json(self.config_path, self.config)

    def at(self, hour, minute=0):
        return datetime(2026, 10, 2, hour, minute, tzinfo=timezone.utc)

    def successful_child(self, label, script, arguments, cutoff, timeout):
        self.calls.append((label, script, arguments, cutoff, timeout))
        return {"status": "succeeded", "returncode": 0}

    def test_new_campaign_ignores_original_complete_state_and_stop(self):
        original = {"status": "complete", "runs": {"v1": {}},
                    "phases": {"final_blend": {"status": "succeeded"}}, "active_child": None}
        supervisor.atomic_json(supervisor.STATE_PATH, original)
        supervisor.STOP_PATH.write_text("original stop\n", encoding="utf-8")
        original_bytes = supervisor.STATE_PATH.read_bytes()
        instance = supervisor.Supervisor(self.config_path)
        self.assertEqual(instance.state["runs"], {})
        self.assertEqual(instance.state["phases"], {})
        self.assertFalse(instance.stopping())
        instance.save("starting")
        instance.event("synthetic_event")
        self.assertEqual(instance.state_path, self.root / "state/second_pass/run_state.json")
        self.assertTrue((self.root / "logs/second_pass/supervisor/events.jsonl").exists())
        self.assertEqual(supervisor.STATE_PATH.read_bytes(), original_bytes)
        self.assertEqual(supervisor.STOP_PATH.read_text(), "original stop\n")
        instance.stop_path.touch()
        self.assertTrue(instance.stopping())

    def test_locks_are_isolated_and_second_pass_excludes_duplicates(self):
        original = supervisor.campaign_paths({})
        second = supervisor.campaign_paths(self.config)
        self.assertEqual(second["lock_path"], self.root / "state/second_pass/lock")
        with supervisor.lifetime_lock(original["lock_path"]):
            with supervisor.lifetime_lock(second["lock_path"]):
                with self.assertRaisesRegex(RuntimeError, "Another supervisor"):
                    with supervisor.lifetime_lock(second["lock_path"]):
                        self.fail("Duplicate campaign lock was acquired")

    def test_final_dispatch_freezes_only_second_pass_and_reserves_refit_time(self):
        instance = supervisor.Supervisor(self.config_path)
        with patch.object(supervisor, "utcnow", return_value=self.at(16)), \
                patch.object(instance, "child", side_effect=self.successful_child):
            self.assertEqual(instance.run(), 0)
        self.assertEqual([item[0] for item in self.calls], ["final_blend", "refit", "verify"])
        for label, script, arguments, _, _ in self.calls:
            self.assertEqual(script, "second_pass_release.py")
            self.assertEqual(arguments[:2], ["--phase", "blend" if label == "final_blend" else label])
            self.assertIn(str(self.config_path), arguments)
            self.assertNotIn("--finalize", arguments)
        self.assertIn("--freeze", self.calls[0][2])
        self.assertNotIn("--freeze", self.calls[1][2])
        self.assertEqual(self.calls[0][3:], (self.at(16, 15), 600.0))
        self.assertEqual(self.calls[1][3:], (self.at(16, 40), 1500.0))
        self.assertEqual(self.calls[2][3], self.at(16, 50))
        self.assertEqual(instance.state["status"], "complete")
        self.assertFalse(supervisor.STATE_PATH.exists())

    def test_original_campaign_keeps_its_paths_and_original_finalize_command(self):
        self.config.pop("campaign")
        self.write_config()
        instance = supervisor.Supervisor(self.config_path)
        with patch.object(supervisor, "utcnow", return_value=self.at(16)), \
                patch.object(instance, "child", side_effect=self.successful_child):
            self.assertEqual(instance.run(), 0)
        self.assertEqual(instance.state_path, supervisor.STATE_PATH)
        self.assertEqual([item[1] for item in self.calls], ["blend.py", "refit.py", "verify.py"])
        self.assertIn("--finalize", self.calls[0][2])
        self.assertEqual(self.calls[0][3], self.at(16, 40))
        self.assertEqual(self.calls[1][4], 2400)
        self.assertFalse((self.root / "state/second_pass/run_state.json").exists())

    def test_empty_queue_waits_and_rereads_new_jobs_without_training_completed_baselines(self):
        baseline = {"id": "baseline", "family": "synthetic"}
        new_run = {"id": "v2_new", "family": "synthetic"}
        self.config["runs"] = [baseline]
        self.write_config()
        for run in [baseline, {"id": "unlisted", "family": "synthetic"}]:
            supervisor.atomic_json(self.root / "artifacts/runs" / run["id"] / "result.json", {"run": run})
        instance = supervisor.Supervisor(self.config_path)
        sleeps = []

        def wait_then_edit(_):
            sleeps.append(instance.state["status"])
            if len(sleeps) == 1:
                self.config["runs"].append(new_run)
                self.write_config()
            else:
                raise QueueWaitObserved()

        def completed_child(label, script, arguments, cutoff, timeout):
            result = self.successful_child(label, script, arguments, cutoff, timeout)
            if script == "train.py":
                self.assertEqual(arguments[-2:], ["--run-id", "v2_new"])
                supervisor.atomic_json(self.root / "artifacts/runs/v2_new/result.json", {"run": new_run})
            return result

        with patch.object(supervisor, "utcnow", return_value=self.at(14)), \
                patch.object(supervisor.time, "sleep", side_effect=wait_then_edit), \
                patch.object(instance, "validate_completed", return_value={}), \
                patch.object(instance, "child", side_effect=completed_child):
            with self.assertRaises(QueueWaitObserved):
                instance.run()
        self.assertEqual(sleeps, ["waiting_for_queue_or_cutoff"] * 2)
        self.assertEqual([item[0] for item in self.calls], ["current_blend", "v2_new", "current_blend"])
        self.assertEqual(instance.state["current_blend_run_ids"], ["baseline", "v2_new"])
        self.assertNotIn("unlisted", instance.state["runs"])
        for label, script, arguments, _, _ in self.calls:
            if label == "current_blend":
                self.assertEqual(script, "second_pass_release.py")
                self.assertNotIn("--freeze", arguments)

    def test_failed_freeze_skips_refit_but_still_attempts_verification(self):
        instance = supervisor.Supervisor(self.config_path)

        def fail_freeze(label, script, arguments, cutoff, timeout):
            result = self.successful_child(label, script, arguments, cutoff, timeout)
            if label == "final_blend":
                result["status"] = "failed"
            return result

        with patch.object(supervisor, "utcnow", return_value=self.at(16)), \
                patch.object(instance, "child", side_effect=fail_freeze):
            self.assertEqual(instance.run(), 1)
        self.assertEqual([item[0] for item in self.calls], ["final_blend", "verify"])
        self.assertEqual(instance.state["status"], "needs_attention")

    def test_past_deadline_cannot_launch_a_release_process(self):
        instance = supervisor.Supervisor(self.config_path)
        with patch.object(supervisor, "utcnow", return_value=self.at(17)), \
                patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("Deadline launch")) as launch:
            self.assertEqual(instance.run(), 1)
        launch.assert_not_called()
        self.assertEqual(instance.state["phases"]["verify"]["status"], "not_started")

    def test_campaign_cannot_change_during_queue_reread(self):
        instance = supervisor.Supervisor(self.config_path)
        self.config["campaign"] = "overnight"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Cannot change campaign namespace"):
            instance.read_config()
        self.config["campaign"] = "../unsafe"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Unknown campaign"):
            supervisor.load_config(self.config_path)

    def test_status_and_dry_run_use_second_pass_namespace_without_writes(self):
        supervisor.atomic_json(supervisor.STATE_PATH, {"status": "complete"})
        for flag in ["--status", "--dry-run"]:
            self.output.seek(0)
            self.output.truncate(0)
            with patch.object(sys, "argv", ["supervisor.py", "--config", str(self.config_path), flag]):
                self.assertEqual(supervisor.main(), 0)
            output = json.loads(self.output.getvalue())
            if flag == "--status":
                self.assertEqual(output, {"status": "not_started"})
            else:
                self.assertEqual(output["campaign"], "second_pass")
                self.assertEqual(output["state_path"], str(self.root / "state/second_pass/run_state.json"))
        self.assertFalse((self.root / "state/second_pass").exists())
        self.assertFalse((self.root / "logs").exists())
        self.assertEqual(supervisor.read_json(supervisor.STATE_PATH), {"status": "complete"})


if __name__ == "__main__":
    unittest.main()
