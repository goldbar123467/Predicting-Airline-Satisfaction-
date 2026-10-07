"""Synthetic third-campaign routing and opt-in queue completion regressions."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
import hashlib
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


class ThirdPassSupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="airline-third-pass-supervisor-")
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
        self.config_path = self.root / "configs/third_pass.json"
        self.config = {
            "campaign": "third_pass", "runs": [], "wait_for_cutoff": False,
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

    def complete_run(self, run):
        # Opaque synthetic bytes exercise actual checksum/identity validation;
        # these are not real training data or readable parquet files.
        split = self.root / "data/splits.parquet"
        split.parent.mkdir(parents=True, exist_ok=True)
        split.write_bytes(b"synthetic split")
        directory = self.root / "artifacts/runs" / run["id"]
        directory.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in ["oof.parquet", "audit.parquet", "test.parquet"]:
            payload = f"synthetic {run['id']} {name}".encode()
            (directory / name).write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        supervisor.atomic_json(directory / "result.json", {
            "run": run, "split_hash": hashlib.sha256(split.read_bytes()).hexdigest(),
            "artifacts": hashes,
        })

    def successful_child(self, label, script, arguments, cutoff, timeout):
        self.calls.append((label, script, arguments, cutoff, timeout))
        if script == "train.py":
            run_id = arguments[arguments.index("--run-id") + 1]
            self.complete_run(next(run for run in self.config["runs"] if run["id"] == run_id))
        return {"status": "succeeded", "returncode": 0}

    def test_third_state_stop_and_lock_are_isolated_from_both_prior_campaigns(self):
        prior = []
        for campaign in ["overnight", "second_pass"]:
            paths = supervisor.campaign_paths({"campaign": campaign})
            supervisor.atomic_json(paths["state_path"], {"status": "complete"})
            paths["stop_path"].write_text("prior STOP\n", encoding="utf-8")
            prior.extend([(paths["state_path"], paths["state_path"].read_bytes()),
                          (paths["stop_path"], paths["stop_path"].read_bytes())])
        instance = supervisor.Supervisor(self.config_path)
        self.assertFalse(instance.stopping())
        self.assertEqual(instance.state["runs"], {})
        self.assertEqual(instance.state["phases"], {})
        instance.save("starting")
        instance.event("synthetic")
        paths = supervisor.campaign_paths(self.config)
        self.assertEqual(paths["state_path"], self.root / "state/third_pass/run_state.json")
        self.assertEqual(paths["log_dir"], self.root / "logs/third_pass/supervisor")
        with ExitStack() as locks:
            for campaign in ["overnight", "second_pass", "third_pass"]:
                locks.enter_context(supervisor.lifetime_lock(supervisor.campaign_paths({"campaign": campaign})["lock_path"]))
            with self.assertRaisesRegex(RuntimeError, "Another supervisor"):
                with supervisor.lifetime_lock(paths["lock_path"]):
                    self.fail("Duplicate third-pass lock acquired")
        for path, contents in prior:
            self.assertEqual(path.read_bytes(), contents)

    def test_completed_nonempty_queue_freezes_early_with_correct_release_commands(self):
        old = {"id": "v2_old", "family": "synthetic"}
        new = {"id": "v3_new", "family": "synthetic", "timeout_seconds": 1200}
        self.config["runs"] = [old, new]
        self.write_config()
        self.complete_run(old)
        instance = supervisor.Supervisor(self.config_path)
        with patch.object(supervisor, "utcnow", return_value=self.at(14)), \
                patch.object(instance, "child", side_effect=self.successful_child), \
                patch.object(supervisor.time, "sleep", side_effect=AssertionError("Unnecessary queue wait")):
            self.assertEqual(instance.run(), 0)
        self.assertEqual([call[0] for call in self.calls],
                         ["current_blend", "v3_new", "current_blend", "final_blend", "refit", "verify"])
        for label, script, arguments, cutoff, timeout in self.calls:
            self.assertIn(str(self.config_path), arguments)
            if label == "v3_new":
                self.assertEqual(script, "train.py")
                self.assertEqual(arguments[-2:], ["--run-id", "v3_new"])
                self.assertEqual((cutoff, timeout), (self.at(16), 1200.0))
            else:
                self.assertEqual(script, "third_pass_release.py")
                self.assertEqual(arguments[:2], ["--phase", "blend" if "blend" in label else label])
                self.assertNotIn("--finalize", arguments)
                self.assertEqual("--freeze" in arguments, label == "final_blend")
        self.assertEqual(self.calls[-3][3:], (self.at(16, 15), 600.0))
        self.assertEqual(self.calls[-2][3:], (self.at(16, 40), 1500.0))
        self.assertEqual(self.calls[-1][3], self.at(16, 50))
        self.assertEqual(instance.state["status"], "complete")
        self.assertIn("queue_complete_early_finalize", self.output.getvalue())

    def test_batch_namespace_does_not_resume_first_third_pass_release(self):
        initial = supervisor.Supervisor(self.config_path)
        initial.state["phases"]["final_blend"] = {"status": "succeeded"}
        initial.save("complete")
        initial.stop_path.touch()
        original = initial.state_path.read_bytes()
        self.config["campaign_id"] = "third_pass_batch02"
        self.write_config()
        batch = supervisor.Supervisor(self.config_path)
        self.assertFalse(batch.stopping())
        self.assertEqual(batch.state["phases"], {})
        self.assertEqual(batch.state_path, self.root / "state/third_pass_batch02/run_state.json")
        self.assertEqual(batch.log_dir, self.root / "logs/third_pass_batch02/supervisor")
        batch.save("starting")
        with supervisor.lifetime_lock(initial.state_path.parent / "lock"):
            with supervisor.lifetime_lock(batch.state_path.parent / "lock"):
                pass
        self.assertEqual(initial.state_path.read_bytes(), original)
        self.assertTrue(initial.stop_path.exists())
        self.assertEqual(batch.release_command("blend", final=True), (
            "third_pass_release.py", ["--phase", "blend", "--config", str(self.config_path), "--freeze"]))
        self.config["campaign_id"] = "third_pass_batch03"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Cannot change campaign namespace"):
            batch.read_config()

    def test_invalid_or_mismatched_batch_names_fail_before_state_access(self):
        for name in ["third_pass_batch1", "third_pass_batch001", "../third_pass", "second_pass", "third_pass/other", None, []]:
            with self.subTest(name=name):
                self.config["campaign_id"] = name
                self.write_config()
                with self.assertRaisesRegex(ValueError, "campaign_id must be"):
                    supervisor.load_config(self.config_path)
        self.config["campaign_id"] = "third_pass_batch02"
        self.config["campaign"] = "second_pass"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "supported only for third_pass"):
            supervisor.load_config(self.config_path)
        self.config["campaign"] = "third_pass"
        self.write_config()
        supervisor.atomic_json(self.root / "state/third_pass_batch02/run_state.json", {
            "campaign": "third_pass", "campaign_id": "third_pass_batch03", "runs": {}, "phases": {}, "active_child": None,
        })
        with self.assertRaisesRegex(ValueError, "different campaign namespace"):
            supervisor.Supervisor(self.config_path)

    def test_default_wait_is_preserved_for_every_campaign_and_explicit_true(self):
        for campaign, explicit_true in [("overnight", False), ("second_pass", False),
                                        ("third_pass", False), ("third_pass", True)]:
            with self.subTest(campaign=campaign, explicit_true=explicit_true):
                self.config["campaign"] = campaign
                self.config.pop("wait_for_cutoff", None)
                if explicit_true:
                    self.config["wait_for_cutoff"] = True
                run = {"id": f"{campaign}_done", "family": "synthetic"}
                self.config["runs"] = [run]
                self.write_config()
                self.complete_run(run)
                instance = supervisor.Supervisor(self.config_path)
                self.calls.clear()
                with patch.object(supervisor, "utcnow", return_value=self.at(14)), \
                        patch.object(instance, "child", side_effect=self.successful_child), \
                        patch.object(supervisor.time, "sleep", side_effect=QueueWaitObserved):
                    with self.assertRaises(QueueWaitObserved):
                        instance.run()
                self.assertEqual(instance.state["status"], "waiting_for_queue_or_cutoff")
                self.assertNotIn("final_blend", [call[0] for call in self.calls])

    def test_empty_queue_and_exhausted_failed_run_do_not_freeze_early(self):
        for failed in [False, True]:
            with self.subTest(failed=failed):
                run = {"id": "v3_failed", "family": "synthetic", "max_retries": 0}
                self.config["runs"] = [run] if failed else []
                self.write_config()
                instance = supervisor.Supervisor(self.config_path)
                if failed:
                    instance.state["runs"][run["id"]] = {"status": "failed", "attempts": [{"status": "failed"}]}
                with patch.object(supervisor, "utcnow", return_value=self.at(14)), \
                        patch.object(instance, "child", side_effect=AssertionError("Early release")), \
                        patch.object(supervisor.time, "sleep", side_effect=QueueWaitObserved):
                    with self.assertRaises(QueueWaitObserved):
                        instance.run()
                self.assertEqual(instance.state["status"], "waiting_for_queue_or_cutoff")

    def test_corrupted_completed_artifact_prevents_early_freeze(self):
        run = {"id": "v3_corrupt", "family": "synthetic"}
        self.config["runs"] = [run]
        self.write_config()
        self.complete_run(run)
        (self.root / "artifacts/runs/v3_corrupt/oof.parquet").write_bytes(b"changed")
        instance = supervisor.Supervisor(self.config_path)
        with patch.object(supervisor, "utcnow", return_value=self.at(14)), \
                patch.object(instance, "child", side_effect=AssertionError("Invalid release")):
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                instance.run()

    def test_cutoff_dispatches_release_without_starting_pending_training(self):
        self.config["wait_for_cutoff"] = True
        self.config["runs"] = [{"id": "v3_unstarted", "family": "synthetic"}]
        self.write_config()
        instance = supervisor.Supervisor(self.config_path)
        with patch.object(supervisor, "utcnow", return_value=self.at(15, 45)), \
                patch.object(instance, "child", side_effect=self.successful_child):
            self.assertEqual(instance.run(), 0)
        self.assertEqual([call[0] for call in self.calls], ["final_blend", "refit", "verify"])
        with patch.object(supervisor, "utcnow", return_value=self.at(16)), \
                patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("Training past cutoff")):
            result = instance.child("late_train", "train.py", [], self.at(16), 1200)
        self.assertEqual(result["status"], "not_started")

    def test_rejects_ambiguous_wait_values_and_campaign_names(self):
        for value in ["false", 0, None]:
            self.config["wait_for_cutoff"] = value
            self.write_config()
            with self.assertRaisesRegex(ValueError, "wait_for_cutoff must be a boolean"):
                supervisor.load_config(self.config_path)
        self.config["wait_for_cutoff"] = False
        self.config["campaign"] = "../third_pass"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Unknown campaign"):
            supervisor.load_config(self.config_path)

    def test_third_status_and_dry_run_do_not_read_prior_state_or_create_files(self):
        supervisor.atomic_json(supervisor.STATE_PATH, {"status": "complete"})
        for flag in ["--status", "--dry-run"]:
            self.output.seek(0)
            self.output.truncate(0)
            with patch.object(sys, "argv", ["supervisor.py", "--config", str(self.config_path), flag]):
                self.assertEqual(supervisor.main(), 0)
            result = json.loads(self.output.getvalue())
            if flag == "--status":
                self.assertEqual(result, {"status": "not_started"})
            else:
                self.assertEqual(result["campaign"], "third_pass")
                self.assertFalse(result["wait_for_cutoff"])
                self.assertEqual(result["state_path"], str(self.root / "state/third_pass/run_state.json"))
        self.assertFalse((self.root / "state/third_pass").exists())
        self.assertFalse((self.root / "logs").exists())


if __name__ == "__main__":
    unittest.main()
