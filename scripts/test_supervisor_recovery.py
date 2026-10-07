"""Exercise supervisor crash recovery with temporary launchers and idle workers."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from datetime import timedelta
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import psutil

import supervisor


class CoordinatorExitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="airline-supervisor-recovery-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "scripts").mkdir()
        self.real_terminate = supervisor.terminate_owned
        self.records = []
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.addCleanup(self.cleanup_processes)
        for name, value in {
            "ROOT": self.root,
            "STATE_DIR": self.root / "state",
            "STATE_PATH": self.root / "state/run_state.json",
            "STOP_PATH": self.root / "state/STOP",
            "LOG_DIR": self.root / "logs",
            "STOP_REQUESTED": False,
        }.items():
            stack.enter_context(patch.object(supervisor, name, value))
        stack.enter_context(redirect_stdout(io.StringIO()))
        self.instance = supervisor.Supervisor(self.root / "unused-config.json")
        # Keep the real venv redirector, rather than requiring a synthetic venv.
        self.instance.python = sys.executable
        self.instance.poll_seconds = 0.05
        self.instance.stop_grace_seconds = 0
        self.worker_path = self.root / "worker.json"
        self.release_path = self.root / "release"
        (self.root / "scripts/worker.py").write_text(
            "import json,psutil,time\n"
            "from pathlib import Path\n"
            "p=psutil.Process()\n"
            f"Path({str(self.worker_path)!r}).write_text(json.dumps("
            "{'pid':p.pid,'create_time':p.create_time(),'command':p.cmdline()}))\n"
            "time.sleep(60)\n", encoding="utf-8")
        (self.root / "scripts/coordinator.py").write_text(
            "import subprocess,sys,time\n"
            "from pathlib import Path\n"
            "subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('worker.py'))],"
            "creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)\n"
            f"release=Path({str(self.release_path)!r})\n"
            "deadline=time.monotonic()+15\n"
            "while not release.exists() and time.monotonic()<deadline: time.sleep(.02)\n"
            "sys.exit(int(sys.argv[1]) if release.exists() else 99)\n", encoding="utf-8")
        self.real_metrics = supervisor.owned_tree_metrics

    def cleanup_processes(self):
        for record in self.records:
            self.real_terminate(record)

    def metrics_then_release(self, record):
        metrics = self.real_metrics(record)
        if not any(item is record for item in self.records):
            self.records.append(record)
        if self.worker_path.exists():
            worker = supervisor.read_json(self.worker_path)
            if worker["pid"] in {item["pid"] for item in record.get("descendants", [])}:
                # Only let the coordinator exit after ownership is established.
                self.release_path.touch()
        return metrics

    def invoke(self, exit_code=7):
        with patch.object(supervisor, "owned_tree_metrics", side_effect=self.metrics_then_release):
            return self.instance.phase(
                "synthetic_refit", "coordinator.py", [str(exit_code)],
                supervisor.utcnow() + timedelta(seconds=25), timeout_seconds=20)

    def assert_worker_reaped(self):
        worker = supervisor.read_json(self.worker_path)
        self.assertIsNone(supervisor.owned_process(worker))
        self.assertTrue(all(not supervisor.surviving_descendants(record) for record in self.records))

    def test_abnormal_coordinator_exit_reaps_recorded_worker_before_clearing_state(self):
        self.assertFalse(self.invoke(7))
        result = self.instance.state["phases"]["synthetic_refit"]
        self.assertEqual(result["returncode"], 7)
        self.assertEqual(result["reason"], "unexpected_descendants")
        self.assertTrue(result["unexpected_survivor_pids"])
        if os.name == "nt":
            # Windows uses the venv redirector plus actual Python coordinator
            # and worker processes; this covers the production launcher shape.
            self.assertGreaterEqual(len(result["descendants"]), 2)
        self.assert_worker_reaped()
        self.assertIsNone(self.instance.state["active_child"])
        self.assertIsNone(supervisor.read_json(supervisor.STATE_PATH)["active_child"])

    def test_successful_launcher_with_orphan_is_not_a_successful_phase(self):
        self.assertFalse(self.invoke(0))
        result = self.instance.state["phases"]["synthetic_refit"]
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "unexpected_descendants")
        self.assert_worker_reaped()

    def test_failed_cleanup_preserves_evidence_and_aborts_phase_until_recovery(self):
        with patch.object(supervisor, "terminate_owned", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Owned descendants remain"):
                self.invoke()
        saved = supervisor.read_json(supervisor.STATE_PATH)
        self.assertEqual(saved["status"], "cleanup_failed")
        self.assertTrue(saved["active_child"]["descendants"])
        self.assertIn("cleanup_error", saved["active_child"])
        self.assertNotIn("synthetic_refit", saved["phases"])
        self.assertIsNotNone(supervisor.owned_process(supervisor.read_json(self.worker_path)))
        restarted = supervisor.Supervisor(self.root / "unused-config.json")
        with patch.object(supervisor, "terminate_owned", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "refusing to start another phase"):
                restarted.recover()
        self.assertIsNotNone(restarted.state["active_child"])
        restarted.recover()
        self.assertIsNone(restarted.state["active_child"])
        self.assert_worker_reaped()

    def test_denied_inspection_does_not_claim_descendant_is_gone(self):
        record = {"pid": 123, "create_time": 5, "command": ["synthetic"]}
        with patch.object(supervisor.psutil, "Process", side_effect=psutil.AccessDenied(123)):
            self.assertIsNone(supervisor.owned_process(record))
            with self.assertRaisesRegex(RuntimeError, "Cannot inspect recorded process"):
                supervisor.surviving_descendants({"descendants": [record]})


if __name__ == "__main__":
    unittest.main()
