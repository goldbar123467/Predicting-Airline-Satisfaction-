"""Run a bounded, restartable overnight training queue on this Windows host."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import psutil
from long_local_500_policy import registered_policy as long_local_policy
from long_local_500_policy import operational_timing


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state"
STATE_PATH = STATE_DIR / "run_state.json"
STOP_PATH = STATE_DIR / "STOP"
LOG_DIR = ROOT / "logs" / "supervisor"
STOP_REQUESTED = False
DEVELOPMENT_RELEASE_SCRIPTS = {
    "second_pass": "second_pass_release.py",
    "third_pass": "third_pass_release.py",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def stamp() -> str:
    return utcnow().isoformat()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Deadline must include a UTC offset: {value!r}")
    return parsed.astimezone(timezone.utc)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    campaign_paths(config)  # Reject ambiguous namespaces before reading state.
    if not isinstance(config.get("wait_for_cutoff", True), bool):
        raise ValueError("wait_for_cutoff must be a boolean")
    runs = config.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Config must contain a runs list")
    identifiers: set[str] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Every run must be an object")
        name = run.get("id", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name):
            raise ValueError(f"Invalid run ID: {name!r}")
        if name in identifiers:
            raise ValueError(f"Duplicate run ID: {name}")
        identifiers.add(name)
        validate_backend(run, config.get("campaign", "overnight"))
        if not isinstance(run.get("family"), str):
            raise ValueError(f"Run {name} needs a family")
        if float(run.get("timeout_seconds", config.get("timeout_seconds", 7200))) <= 0:
            raise ValueError(f"Run {name} timeout_seconds must be positive")
        retries = run.get("max_retries", config.get("max_retries", 1))
        if not isinstance(retries, int) or not 0 <= retries <= 3:
            raise ValueError(f"Run {name} max_retries must be an integer from 0 to 3")
    deadlines = [parse_utc(config[key]) for key in (
        "stop_new_runs_utc", "finalize_utc", "deadline_utc"
    )]
    if not deadlines[0] <= deadlines[1] < deadlines[2] - timedelta(minutes=20):
        raise ValueError("Require stop_new_runs <= finalize < deadline minus 20 minutes")
    if config.get("campaign") in DEVELOPMENT_RELEASE_SCRIPTS:
        long_policy = long_local_policy(config)
        maximum_refit = 14400 if long_policy else 2400
        reserve = config.get('verify_reserve_seconds', 1200)
        if reserve != (3600 if long_policy else 1200):
            raise ValueError('Verification reserve differs from registered campaign')
        if not 0 < float(config.get("refit_timeout_seconds", 1500)) <= maximum_refit:
            raise ValueError(f"Development-campaign refit timeout must be in (0, {maximum_refit}]")
    return config


def validate_backend(run: dict[str, Any], campaign: str) -> None:
    backend = run.get("execution_backend", "local")
    if backend not in {"local", "kaggle"}:
        raise ValueError(f"Unsupported execution backend for {run.get('id')}: {backend!r}")
    if backend == "kaggle" and campaign != "third_pass":
        raise ValueError("Kaggle imports are supported only in third_pass")


def campaign_paths(config: dict[str, Any]) -> dict[str, Path | str]:
    """Choose fixed namespaces; campaign names never become arbitrary paths."""
    campaign = config.get("campaign", "overnight")
    if not isinstance(campaign, str):
        raise ValueError(f"Unknown campaign: {campaign!r}")
    campaign_id = campaign
    if campaign == "third_pass":
        campaign_id = config.get("campaign_id", campaign)
        if not isinstance(campaign_id, str) or not re.fullmatch(r"third_pass(?:_batch[0-9]{2})?", campaign_id):
            raise ValueError("campaign_id must be third_pass or third_pass_batchNN")
    elif "campaign_id" in config:
        raise ValueError("campaign_id is supported only for third_pass")
    if campaign in DEVELOPMENT_RELEASE_SCRIPTS:
        state_dir = ROOT / "state" / campaign_id
        return {"campaign": campaign, "campaign_id": campaign_id, "state_dir": state_dir,
                "state_path": state_dir / "run_state.json", "stop_path": state_dir / "STOP",
                "lock_path": state_dir / "lock", "log_dir": ROOT / "logs" / campaign_id / "supervisor"}
    if campaign != "overnight":
        raise ValueError(f"Unknown campaign: {campaign!r}")
    return {"campaign": campaign, "campaign_id": campaign_id, "state_dir": STATE_DIR, "state_path": STATE_PATH,
            "stop_path": STOP_PATH, "lock_path": STATE_DIR / "supervisor.lock", "log_dir": LOG_DIR}


@contextlib.contextmanager
def lifetime_lock(path: Path):
    """An OS-held lock disappears on process death; stale files are harmless."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError("Another supervisor holds the lifetime lock") from exc
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def owned_process(record: dict[str, Any], *, require_inspection: bool = False) -> psutil.Process | None:
    """Require creation time and the full argument vector before any termination."""
    try:
        process = psutil.Process(int(record["pid"]))
        if abs(process.create_time() - float(record["create_time"])) > 0.01:
            return None
        if process.cmdline() != record["command"]:
            return None
        return process
    except psutil.AccessDenied:
        if require_inspection:
            raise RuntimeError(f"Cannot inspect recorded process {record.get('pid')}")
        return None
    except (psutil.NoSuchProcess, KeyError, ValueError):
        return None


def capture_descendants(record: dict[str, Any]) -> None:
    """Remember proven descendants before Windows' venv redirector can exit."""
    root = owned_process(record)
    if root is None:
        return
    captured = {(item["pid"], item["create_time"]): item for item in record.get("descendants", [])}
    try:
        children = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in children:
        try:
            ancestors = child.parents()
            depth = next((index + 1 for index, parent in enumerate(ancestors)
                          if parent.pid == root.pid and abs(parent.create_time() - record["create_time"]) <= 0.01), None)
            if depth is None:
                continue
            item = {
                "pid": child.pid, "create_time": child.create_time(),
                "command": child.cmdline(), "parent_pid": child.ppid(), "depth": depth,
            }
            captured[(item["pid"], item["create_time"])] = item
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    record["descendants"] = list(captured.values())


def owned_tree_metrics(record: dict[str, Any]) -> dict[str, float | int]:
    capture_descendants(record)
    cpu_seconds = 0.0
    rss_bytes = 0
    process_count = 0
    for member in [record, *record.get("descendants", [])]:
        process = owned_process(member)
        if process is None:
            continue
        try:
            cpu = process.cpu_times()
            cpu_seconds += cpu.user + cpu.system
            rss_bytes += process.memory_info().rss
            process_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"process_cpu_seconds": cpu_seconds, "rss_bytes": rss_bytes,
            "owned_process_count": process_count}


def terminate_owned(record: dict[str, Any]) -> bool:
    capture_descendants(record)
    members = sorted(record.get("descendants", []), key=lambda item: item["depth"], reverse=True)
    members.append(record)
    terminated = []
    identities = {}
    for member in members:
        process = owned_process(member)
        if process is None:
            continue
        try:
            process.terminate()
            terminated.append(process)
            identities[process.pid] = member
        except psutil.NoSuchProcess:
            continue
    if not terminated:
        return False
    _, alive = psutil.wait_procs(terminated, timeout=10)
    for process in alive:
        verified = owned_process(identities[process.pid])
        if verified is not None:
            try:
                verified.kill()
            except psutil.NoSuchProcess:
                pass
    _, still_alive = psutil.wait_procs(alive, timeout=5)
    if still_alive:
        raise RuntimeError(f"Owned processes did not terminate: {[p.pid for p in still_alive]}")
    return True


def surviving_descendants(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Inspect saved identities even after their original parent has exited."""
    return [item for item in record.get("descendants", [])
            if owned_process(item, require_inspection=True) is not None]


def on_signal(signum: int, frame: Any) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


class Supervisor:
    def __init__(self, config_path: Path, config: dict[str, Any] | None = None) -> None:
        self.config_path = config_path
        self.python = str(ROOT / ".venv" / "Scripts" / "python.exe")
        if config is None:
            config = load_config(config_path) if config_path.exists() else {}
        paths = campaign_paths(config)
        self.campaign = str(paths["campaign"])
        self.campaign_id = str(paths["campaign_id"])
        self.state_path = Path(paths["state_path"])
        self.stop_path = Path(paths["stop_path"])
        self.log_dir = Path(paths["log_dir"])
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "schema_version": 1, "runs": {}, "phases": {}, "active_child": None
        }
        if self.state.get("campaign", self.campaign) != self.campaign:
            raise ValueError("Saved state belongs to a different campaign")
        if self.state.get("campaign_id", self.campaign_id) != self.campaign_id:
            raise ValueError("Saved state belongs to a different campaign namespace")
        self.state.update({
            "campaign": self.campaign,
            "campaign_id": self.campaign_id,
            "supervisor_pid": os.getpid(),
            "supervisor_create_time": psutil.Process().create_time(),
            "config_path": str(config_path), "started_utc": stamp(),
        })
        self.poll_seconds = 10.0
        self.stop_grace_seconds = 120.0
        self.hash_cache: dict[Path, tuple[int, int, str]] = {}
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def read_config(self) -> dict[str, Any]:
        config = load_config(self.config_path)
        if config.get("campaign", "overnight") != self.campaign:
            raise ValueError("Cannot change campaign namespace while the supervisor is running")
        if campaign_paths(config)["campaign_id"] != self.campaign_id:
            raise ValueError("Cannot change campaign namespace while the supervisor is running")
        # A remote-designated entry is an imported result, never local work.
        # Validate before recovery can terminate any recorded child process.
        for run in config["runs"]:
            if run.get("execution_backend") == "kaggle":
                result_path = ROOT / "artifacts" / "runs" / run["id"] / "result.json"
                if not result_path.is_file():
                    raise ValueError(f"Cloud run {run['id']} requires a verified completed import; local training is forbidden")
                self.validate_completed(result_path, run)
        return config

    def release_command(self, phase: str, *, final: bool = False) -> tuple[str, list[str]]:
        arguments = ["--config", str(self.config_path)]
        if self.campaign in DEVELOPMENT_RELEASE_SCRIPTS:
            if phase == "blend" and final:
                arguments.append("--freeze")
            return DEVELOPMENT_RELEASE_SCRIPTS[self.campaign], ["--phase", phase, *arguments]
        if phase == "blend" and final:
            arguments.append("--finalize")
        return f"{phase}.py", arguments

    def cached_sha256(self, path: Path) -> str:
        stat = path.stat()
        cached = self.hash_cache.get(path)
        if cached is not None and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        self.hash_cache[path] = (stat.st_mtime_ns, stat.st_size, value)
        return value

    def validate_completed(self, path: Path, run: dict[str, Any]) -> dict[str, Any]:
        validate_backend(run, self.campaign)
        result = read_json(path)
        if result.get("run") != run:
            raise ValueError(f"Completed run {run['id']} differs from the queue; use a new run ID")
        if result.get("split_hash") != self.cached_sha256(ROOT / "data" / "splits.parquet"):
            raise ValueError(f"Completed run {run['id']} has a different split")
        artifacts = result.get("artifacts", {})
        cloud = run.get("execution_backend") == "kaggle"
        required = {"oof.parquet", "test.parquet"} if cloud else {"oof.parquet", "audit.parquet", "test.parquet"}
        if not isinstance(artifacts, dict) or set(artifacts) != required:
            raise ValueError(f"Completed run {run['id']} has an incomplete artifact manifest")
        for name, expected in artifacts.items():
            if self.cached_sha256(path.parent / name) != expected:
                raise ValueError(f"Completed run {run['id']} artifact hash mismatch: {name}")
        if cloud:
            self.validate_cloud_import(path, run, result)
        return result

    def validate_cloud_import(self, path: Path, run: dict[str, Any], result: dict[str, Any]) -> None:
        directory = (ROOT / "artifacts" / "runs" / run["id"]).resolve()
        if path.resolve() != directory / "result.json":
            raise ValueError("Cloud result is outside its owned run directory")
        receipt = read_json(directory / "cloud_import_verification.json")
        if (receipt.get("status") != "verified" or receipt.get("run_id") != run["id"]
                or receipt.get("run") != run or receipt.get("all_heldout_native_inference") is not True):
            raise ValueError(f"Cloud run {run['id']} lacks a verified native inference receipt")
        expected = {"result_sha256": self.cached_sha256(path),
                    "oof_sha256": result["artifacts"]["oof.parquet"],
                    "test_sha256": result["artifacts"]["test.parquet"],
                    "split_sha256": result["split_hash"]}
        for field, digest in expected.items():
            if receipt.get(field) != digest:
                raise ValueError(f"Cloud import receipt hash mismatch: {field}")
        for field in ("source_hashes", "native_hashes", "heldout_replay_receipt_hashes"):
            mapping = receipt.get(field)
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"Cloud import receipt needs a nonempty {field}")
            for relative, digest in mapping.items():
                if not isinstance(relative, str) or Path(relative).is_absolute():
                    raise ValueError(f"Cloud import {field} has an invalid relative path")
                artifact = (ROOT / relative).resolve()
                if not artifact.is_relative_to(directory) or not artifact.is_file():
                    raise ValueError(f"Cloud import {field} path is outside its owned run directory or missing: {relative}")
                if self.cached_sha256(artifact) != digest:
                    raise ValueError(f"Cloud import {field} hash mismatch: {relative}")

    def save(self, status: str | None = None) -> None:
        if status:
            self.state["status"] = status
        self.state["updated_utc"] = stamp()
        atomic_json(self.state_path, self.state)

    def event(self, event: str, **details: Any) -> None:
        entry = {"utc": stamp(), "event": event, **details}
        line = json.dumps(entry, allow_nan=False)
        print(line, flush=True)
        with (self.log_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def stopping(self) -> bool:
        return STOP_REQUESTED or self.stop_path.exists()

    def recover(self) -> None:
        previous = self.state.get("active_child")
        if previous:
            # The lock establishes that no previous supervisor remains. Stop only
            # its exactly identified orphan before resuming checkpointed work.
            terminated = terminate_owned(previous)
            if owned_process(previous, require_inspection=True) is not None or surviving_descendants(previous):
                raise RuntimeError("Previous owned processes remain; refusing to start another phase")
            self.event("recover_previous_child", terminated=terminated, child=previous)
            self.state["active_child"] = None
        for item in self.state.get("runs", {}).values():
            if item.get("status") == "running":
                item["status"] = "interrupted"
                if item.get("attempts"):
                    item["attempts"][-1]["status"] = "interrupted"
        self.save("starting")

    def child(self, label: str, script: str, arguments: list[str],
              hard_stop: datetime, timeout_seconds: float) -> dict[str, Any]:
        start = utcnow()
        if self.stopping() or start >= hard_stop - timedelta(seconds=15):
            return {"status": "not_started", "reason": "stop_or_deadline"}
        unique = f"{start.strftime('%Y%m%dT%H%M%S%fZ')}_{label}"
        stdout_path = self.log_dir / f"{unique}.stdout.log"
        stderr_path = self.log_dir / f"{unique}.stderr.log"
        command = [self.python, "-u", str(ROOT / "scripts" / script), *arguments]
        # Leave time to terminate/reap the child before the external deadline.
        deadline_cutoff = hard_stop - timedelta(seconds=15)
        cutoff = min(deadline_cutoff, start + timedelta(seconds=timeout_seconds))
        record: dict[str, Any] = {
            "label": label, "command": command, "started_utc": start.isoformat(),
            "cutoff_utc": cutoff.isoformat(), "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        stop_started: float | None = None
        reason: str | None = None
        environment = os.environ.copy()
        environment.setdefault("OMP_NUM_THREADS", "8")
        environment.setdefault("MKL_NUM_THREADS", "8")
        environment["PYTHONUNBUFFERED"] = "1"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            record.update(pid=process.pid, create_time=psutil.Process(process.pid).create_time())
            self.state["active_child"] = record
            self.save("running")
            self.event("child_started", **record)
            try:
                while process.poll() is None:
                    now = utcnow()
                    if self.stopping() and stop_started is None:
                        stop_started = time.monotonic()
                        self.event("cooperative_stop_requested", label=label)
                    if now >= cutoff:
                        reason = "deadline" if cutoff == deadline_cutoff else "timeout"
                    if stop_started is not None and time.monotonic() - stop_started >= self.stop_grace_seconds:
                        reason = "operator_stop"
                    if reason:
                        self.event("terminate_owned_child", reason=reason, label=label)
                        if not terminate_owned(record) and process.poll() is None:
                            raise RuntimeError("Child identity cannot be verified; refusing termination")
                        break
                    record.update(owned_tree_metrics(record))
                    record["observed_utc"] = now.isoformat()
                    record["available_ram_bytes"] = psutil.virtual_memory().available
                    record["stdout_bytes"] = stdout_path.stat().st_size
                    record["stderr_bytes"] = stderr_path.stat().st_size
                    record["latest_log_utc"] = datetime.fromtimestamp(
                        max(stdout_path.stat().st_mtime, stderr_path.stat().st_mtime), timezone.utc
                    ).isoformat()
                    self.save("stopping" if stop_started is not None else "running")
                    time.sleep(min(self.poll_seconds, max(0.05, (cutoff - now).total_seconds())))
                returncode = process.wait(timeout=15)
            except BaseException:
                terminate_owned(record)
                process.wait(timeout=15)
                raise
            # A dead redirector/coordinator does not imply its workers are dead.
            # Keep ownership evidence until every recorded survivor is reaped.
            try:
                survivors = surviving_descendants(record)
                if survivors:
                    reason = reason or "unexpected_descendants"
                    record["unexpected_survivor_pids"] = [item["pid"] for item in survivors]
                    self.save("cleaning_up")
                    self.event("cleanup_surviving_descendants", label=label,
                               pids=record["unexpected_survivor_pids"])
                    terminate_owned(record)
                    if surviving_descendants(record):
                        raise RuntimeError("Owned descendants remain after coordinator exit")
            except BaseException as exc:
                record["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                self.save("cleanup_failed")
                self.event("descendant_cleanup_failed", label=label, error=record["cleanup_error"])
                raise
        result = {
            **record, "returncode": returncode, "finished_utc": stamp(),
            "elapsed_seconds": (utcnow() - start).total_seconds(),
            "status": "succeeded" if returncode == 0 and reason is None else "failed",
            "reason": reason,
        }
        self.state["active_child"] = None
        self.save()
        self.event("child_finished", label=label, status=result["status"],
                   returncode=returncode, reason=reason)
        return result

    def phase(self, name: str, script: str, arguments: list[str],
              cutoff: datetime, force: bool = False, timeout_seconds: float | None = None) -> bool:
        previous = self.state["phases"].get(name, {})
        if previous.get("status") == "succeeded" and not force:
            return True
        timeout = timeout_seconds if timeout_seconds is not None else max(1, (cutoff - utcnow()).total_seconds())
        result = self.child(name, script, arguments, cutoff, timeout)
        self.state["phases"][name] = result
        self.save()
        return result["status"] == "succeeded"

    def run(self) -> int:
        config = self.read_config()
        self.poll_seconds = max(1, min(60, float(config.get("supervisor_poll_seconds", 10))))
        self.stop_grace_seconds = max(0, min(300, float(config.get("stop_grace_seconds", 120))))
        self.recover()
        args = ["--config", str(self.config_path)]
        while not self.stopping():
            config = self.read_config()  # Monitor may edit the pending queue.
            timing = operational_timing(config)
            self.state["deadlines"] = {key: timing[key] for key in (
                "stop_new_runs_utc", "finalize_utc", "deadline_utc"
            )}
            stop_new = parse_utc(timing["stop_new_runs_utc"])
            final_cutoff = parse_utc(timing["finalize_utc"])
            if utcnow() >= stop_new or self.state["phases"].get("final_blend", {}).get("status") == "succeeded":
                break
            pending = []
            completed_ids = []
            for index, run in enumerate(config["runs"]):
                name = run["id"]
                item = self.state["runs"].setdefault(name, {"attempts": [], "status": "pending"})
                result_path = ROOT / "artifacts" / "runs" / name / "result.json"
                if result_path.exists():
                    self.validate_completed(result_path, run)
                    item.update(status="succeeded", result_path=str(result_path))
                    completed_ids.append(name)
                    continue
                attempts = len(item["attempts"])
                max_retries = run.get("max_retries", config.get("max_retries", 1))
                if attempts < max_retries + 1:
                    pending.append((attempts, index, run))
            if completed_ids and completed_ids != self.state.get("current_blend_run_ids"):
                blend_script, blend_args = self.release_command("blend")
                self.phase("current_blend", blend_script, blend_args, final_cutoff, force=True,
                           timeout_seconds=float(config.get("blend_timeout_seconds", 600)))
                # Record the attempted inputs even on failure; the monitor can
                # diagnose the logs without entering an unbounded retry loop.
                self.state["current_blend_run_ids"] = completed_ids
                self.save()
                continue
            if not pending:
                # Early release is opt-in and requires a genuinely completed,
                # nonempty queue. Exhausted retries and empty staging queues
                # keep waiting for the monitor or the normal cutoff.
                if (not config.get("wait_for_cutoff", True) and config["runs"]
                        and len(completed_ids) == len(config["runs"])):
                    self.event("queue_complete_early_finalize", run_ids=completed_ids)
                    break
                self.save("waiting_for_queue_or_cutoff")
                time.sleep(min(self.poll_seconds, max(0.05, (stop_new - utcnow()).total_seconds())))
                continue
            _, _, run = min(pending, key=lambda value: (value[0], value[1]))
            name = run["id"]
            if run.get("execution_backend") == "kaggle":
                raise RuntimeError(f"Cloud run {name} requires a verified completed import; local training is forbidden")
            item = self.state["runs"][name]
            item["status"] = "running"
            item["config_sha256"] = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
            attempt = {"status": "running", "started_utc": stamp()}
            item["attempts"].append(attempt)
            self.save("running")
            result = self.child(
                name, "train.py", [*args, "--run-id", name], final_cutoff,
                float(run.get("timeout_seconds", config.get("timeout_seconds", 7200))),
            )
            result_path = ROOT / "artifacts" / "runs" / name / "result.json"
            if result["status"] == "succeeded":
                if result_path.exists():
                    self.validate_completed(result_path, run)
                    item["result_path"] = str(result_path)
                else:
                    result.update(status="failed", reason="missing_result_manifest")
            attempt.update(result)
            item["status"] = result["status"]
            self.save()
        if self.stopping():
            self.save("stopped")
            self.event("supervisor_stopped", reason="operator_request")
            return 0
        config = self.read_config()
        deadline = parse_utc(operational_timing(config)["deadline_utc"])
        verify_at = deadline - timedelta(seconds=config.get('verify_reserve_seconds', 1200))
        exit_at = deadline - timedelta(minutes=10)
        self.save("finalizing")
        # Later campaigns freeze development-only selection. They must never
        # dispatch the original audit-finalization CLI.
        refit_timeout = (float(config.get("refit_timeout_seconds", 1500))
                         if self.campaign in DEVELOPMENT_RELEASE_SCRIPTS else None)
        blend_cutoff = verify_at - timedelta(seconds=refit_timeout) if refit_timeout is not None else verify_at
        blend_script, blend_args = self.release_command("blend", final=True)
        final_ok = self.phase("final_blend", blend_script, blend_args, blend_cutoff,
                              timeout_seconds=float(config.get("blend_timeout_seconds", 600)))
        refit_ok = False
        if final_ok and not self.stopping():
            refit_script, refit_args = self.release_command("refit")
            refit_ok = self.phase("refit", refit_script, refit_args, verify_at,
                                  timeout_seconds=refit_timeout)
        if not self.stopping():
            verify_script, verify_args = self.release_command("verify")
            verified = self.phase("verify", verify_script, verify_args, exit_at, force=True)
        else:
            verified = False
        status = "complete" if final_ok and refit_ok and verified else "needs_attention"
        self.save(status)
        self.event("supervisor_finished", status=status)
        return 0 if status == "complete" else 1


def self_test() -> None:
    """Exercise lock exclusion, atomic JSON, and owned-process termination only."""
    with tempfile.TemporaryDirectory(prefix="airline-supervisor-test-") as directory:
        path = Path(directory)
        atomic_json(path / "state.json", {"ok": True})
        assert read_json(path / "state.json") == {"ok": True}
        with lifetime_lock(path / "test.lock"):
            try:
                with lifetime_lock(path / "test.lock"):
                    raise AssertionError("Second lock was incorrectly acquired")
            except RuntimeError:
                pass
        # A real child of the venv redirector spawns another worker. Checking only
        # the redirector's exit is insufficient on Windows.
        worker_path = path / "worker.json"
        launcher_code = (
            "import json,subprocess,sys,time,psutil; "
            "from pathlib import Path; "
            "worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "p=psutil.Process(worker.pid); "
            f"Path({str(worker_path)!r}).write_text(json.dumps({{'pid':p.pid,'create_time':p.create_time(),'command':p.cmdline()}})); "
            "time.sleep(60)"
        )
        command = [sys.executable, "-c", launcher_code]
        child = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            record = {"pid": child.pid, "create_time": psutil.Process(child.pid).create_time(), "command": command}
            limit = time.monotonic() + 10
            while not worker_path.exists() and time.monotonic() < limit:
                time.sleep(0.05)
            assert worker_path.exists(), "Dummy launcher did not create its worker"
            worker = read_json(worker_path)
            assert owned_process(record) is not None
            assert not terminate_owned({**record, "create_time": record["create_time"] - 100})
            assert child.poll() is None, "An unverified PID must never be terminated"
            metrics = owned_tree_metrics(record)
            assert metrics["owned_process_count"] >= 2
            assert metrics["rss_bytes"] > psutil.Process(child.pid).memory_info().rss
            assert terminate_owned(record)
            child.wait(timeout=5)
            assert owned_process(worker) is None, "Terminating a launcher left an orphan worker"
            assert all(owned_process(item) is None for item in record["descendants"])
        finally:
            if "record" in locals():
                terminate_owned(record)
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
    print("PASS: atomic state, exclusive lock, PID identity rejection, owned descendant termination and resource aggregation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "overnight.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate queue and print deadlines; launch nothing")
    parser.add_argument("--status", action="store_true", help="Print saved supervisor state without taking its lock")
    parser.add_argument("--self-test", action="store_true")
    options = parser.parse_args()
    if options.self_test:
        self_test()
        return 0
    config_path = options.config.resolve()
    config = load_config(config_path)
    paths = campaign_paths(config)
    if options.status:
        state_path = Path(paths["state_path"])
        print(json.dumps(read_json(state_path) if state_path.exists() else {"status": "not_started"}, indent=2))
        return 0
    if options.dry_run:
        timing = operational_timing(config)
        print(json.dumps({
            "config": str(config_path), "run_ids": [run["id"] for run in config["runs"]],
            "campaign": paths["campaign"],
            "campaign_id": paths["campaign_id"],
            "state_path": str(paths["state_path"]), "stop_path": str(paths["stop_path"]),
            "lock_path": str(paths["lock_path"]), "log_dir": str(paths["log_dir"]),
            "wait_for_cutoff": config.get("wait_for_cutoff", True),
            "stop_new_runs_utc": timing["stop_new_runs_utc"], "finalize_utc": timing["finalize_utc"],
            "verify_latest_start_utc": (parse_utc(timing["deadline_utc"]) - timedelta(seconds=config.get('verify_reserve_seconds', 1200))).isoformat(),
            "supervisor_exit_utc": (parse_utc(timing["deadline_utc"]) - timedelta(minutes=10)).isoformat(),
        }, indent=2))
        return 0
    long_policy = long_local_policy(config)
    if long_policy and utcnow() < parse_utc(operational_timing(config)['not_before_utc']):
        raise ValueError('Overnight training cannot start before the registered time')
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    with lifetime_lock(Path(paths["lock_path"])):
        supervisor = Supervisor(config_path, config)
        try:
            return supervisor.run()
        except Exception as exc:
            supervisor.state["last_error"] = f"{type(exc).__name__}: {exc}"
            supervisor.save("failed")
            supervisor.event("supervisor_error", error=supervisor.state["last_error"])
            raise


if __name__ == "__main__":
    raise SystemExit(main())
