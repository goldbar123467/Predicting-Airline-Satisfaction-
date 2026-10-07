"""Hash the isolated research deliverables and verify preserved source contracts.

Reads no data table, model predictions, credentials or environment variables.
Each invocation writes a new timestamped snapshot; completed snapshots are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/research_pass_v1"
PINS = {
    "DEEP_RESEARCH_PASS_PLAN.md": "a52374b99626aa31e539757990fa4e940c6a38916ecfee3f5965406b124f1ef1",
    "data/splits.parquet": "4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c",
    "artifacts/third_pass/blend/frozen.json": "bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311",
    "artifacts/second_pass/blend/frozen.json": "d75c7eba29f8ed9cb3e9713e4f43b102a5d48dacdcfbba616f9c331a28bb0dc1",
    "configs/research_pass_v1.json": "24e644e49dd77923290da513f7d642d5b43c1b8f5722904c02694e0d6fbac21d",
    "artifacts/third_pass_batch07/final/release_provenance.json": "b3788575af52f6dc7e807d332c3226dd3b739c56dd7b3129da82db52e3be0690",
}
PRODUCTION = (
    "train.py", "common.py", "realmlp_categorical.py", "realmlp.py",
    "third_pass_release.py", "submit_verified_release.py", "verify.py", "refit.py",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("working", "final"), default="working")
    args = parser.parse_args()
    checks = []
    for name, expected in PINS.items():
        actual = digest(ROOT / name)
        checks.append({"kind": "frozen_input", "path": name,
                       "sha256": actual, "expected_sha256": expected,
                       "passed": actual == expected})
    provenance = json.loads((ROOT / "artifacts/third_pass_batch07/final/release_provenance.json").read_text(encoding="utf-8"))
    for name in PRODUCTION:
        live = ROOT / "scripts" / name
        archived = ROOT / "artifacts/third_pass_batch07/final/reproduction_source/scripts" / name
        actual, archived_hash = digest(live), digest(archived)
        expected = provenance["source_hashes"][relative(archived)]
        checks.append({"kind": "production_source_preserved", "path": relative(live),
                       "reference": relative(archived), "sha256": actual,
                       "archive_sha256": archived_hash, "expected_sha256": expected,
                       "passed": actual == archived_hash == expected})
    snapshot_bindings = (
        ("blend/diagnostics.json", "source_sha256", "analysis_blend_diagnostics_v1_executed.py"),
        ("blend/pair_geometry.json", "source_sha256", "analysis_pair_geometry_v1_executed.py"),
        ("blend/pair_geometry.json", "base_source_sha256", "analysis_blend_diagnostics_v1_executed.py"),
    )
    for result, key, source in snapshot_bindings:
        record = json.loads((ARTIFACTS / result).read_text(encoding="utf-8"))
        path = ARTIFACTS / "source_snapshots" / source
        actual = digest(path)
        checks.append({"kind": "executed_source_binding", "path": relative(path),
                       "result": result, "key": key, "sha256": actual,
                       "expected_sha256": record[key], "passed": actual == record[key]})

    paths = {ROOT / "DEEP_RESEARCH_PASS_PLAN.md", ROOT / "DEEP_RESEARCH_REPORT.md",
             ROOT / "research/next_epoch_blend_experiment.md",
             ROOT / "state/research_pass_v1/progress.json", Path(__file__)}
    for pattern in ("research/deep_pass*.md", "configs/research*_v1.json",
                    "scripts/analysis_*_v1.py", "scripts/research_*_v1.py",
                    "scripts/test_analysis_*_v1.py", "scripts/test_research_*_v1.py"):
        paths.update(ROOT.glob(pattern))
    paths.update(path for path in ARTIFACTS.rglob("*")
                 if path.is_file() and "manifests" not in path.relative_to(ARTIFACTS).parts)
    records = [{"path": relative(path), "bytes": path.stat().st_size,
                "sha256": digest(path)} for path in sorted(paths)]
    timestamp = datetime.now(timezone.utc)
    destination = ARTIFACTS / "manifests" / f"{timestamp:%Y%m%dT%H%M%S%fZ}_{args.stage}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    passed = all(check["passed"] for check in checks)
    result = {"created_utc": timestamp.isoformat(), "stage": args.stage,
              "status": "passed" if passed else "failed", "files": records,
              "contract_checks": checks, "invocation": [sys.executable, *sys.argv],
              "scope": "Research artifact inventory plus 17 explicit checks: six frozen inputs, eight production files and archives matched against pinned release provenance, and three executed-source bindings for two root scripts. Other source/import bindings are covered only by their separately recorded reviews, not this inventory. Hashes do not establish mathematical correctness or unexecuted tests. The stage label does not establish elapsed research duration.",
              "exclusions": "Earlier manifest snapshots excluded to avoid recursive inventories. No raw table parsing, new scoring or training."}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": relative(destination), "status": result["status"],
                      "files": len(records), "contract_checks": len(checks),
                      "sha256": digest(destination)}))
    if not passed:
        raise SystemExit("Preservation or source binding failed; inspect the saved snapshot")


if __name__ == "__main__":
    main()
