"""Save a reproducible, source-bound receipt for generated infrastructure checks."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("run_continuation_six_v1.py", "smoke_continuation_six_v1.py", "continuation_six_deterministic_v1.py",
         "continuation_six_policy_v1.py", "prepare_continuation_six_v1.py", "test_continuation_six_infrastructure_v1.py",
         "continuation_six_selection_v1.py", "compose_continuation_six_release_v1.py", "test_continuation_six_selection_v1.py",
         "verify_continuation_six_infrastructure_v1.py", "test_run_fixed_epoch_dropout_v1.py", "test_fixed_epoch_deterministic_execution_v1.py")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/continuation_six_v1/infrastructure_tests_01")
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT / "artifacts/continuation_six_v1").resolve()):
        raise ValueError("Infrastructure evidence must stay inside its isolated namespace")
    destination.mkdir(parents=True, exist_ok=False)
    sources = {"scripts/" + name: sha(ROOT / "scripts" / name) for name in NAMES}
    start = datetime.now(timezone.utc).isoformat()
    elapsed = time.monotonic()
    commands, total, returncode = [], 0, 0
    for name, count in (("test_continuation_six_infrastructure_v1.py", 23), ("test_continuation_six_selection_v1.py", 4)):
        command = [sys.executable, str(ROOT / "scripts" / name), "-v"]
        with (destination / (name + ".log")).open("xb") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=90)
        commands.append(command)
        total += count if result.returncode == 0 else 0
        returncode = result.returncode or returncode
    unchanged = all(sha(ROOT / name) == digest for name, digest in sources.items())
    receipt = {"status": "passed" if returncode == 0 and unchanged else "failed", "returncode": returncode,
               "tests_passed": total, "source_hashes": sources, "source_unchanged": unchanged,
               "started_utc": start, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.monotonic() - elapsed, "commands": commands,
               "real_data_rows": 0, "cuda_used": False,
               "scope": "Generated startup/data boundary/gain/schema tests, no fits or quality assessment",
               "log_sha256": {p.name: sha(p) for p in destination.glob("*.log")}}
    with (destination / "verification.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "tests_passed": total, "path": str(destination / "verification.json")}))
    raise SystemExit(0 if receipt["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
