"""Revalidate the immutable v1 fallback without fitting or running native models."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, TARGET, atomic_json, load_config, sha256


V1_SUBMISSION_SHA = "02443aa5011384df8343cde5b43e73003a39a65fca28fb99c602f50c7d3e4b10"
V1_SELECTION_SHA = "ea06c3b7af6505cea9e51c97bd40c6c5c488af950e6d0b43c9ba7247c9018f48"
V1_PROVENANCE_SHA = "11297e9b248e76e659cede2d3e33f65119aa97a096d94df4a1dfc145f935111b"
V1_PUBLIC_SCORE = .96093
V1_KAGGLE_SUBMISSION_ID = 56771783


def _require_hash(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f"Immutable v1 checksum mismatch: {path}")


def _local_path(relative: str) -> Path:
    path = Path(relative)
    root = ROOT.resolve()
    if path.is_absolute() or not (root / path).resolve().is_relative_to(root):
        raise ValueError(f"Provenance path escapes the project: {relative}")
    return root / path


def _probabilities(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError(f"Invalid v1 probabilities: {name}")
    return result


def _submission(path: Path, ids: np.ndarray | None = None) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if header != ["id", TARGET]:
        raise ValueError(f"Invalid submission schema: {path}")
    frame = pd.read_csv(path, dtype={"id": str})
    if not len(frame) or not frame.id.is_unique or frame.id.isna().any():
        raise ValueError(f"Invalid submission identifiers: {path}")
    if ids is not None and not np.array_equal(frame.id.to_numpy(), ids):
        raise ValueError(f"Submission row order/identity mismatch: {path}")
    _probabilities(frame[TARGET], str(path))
    return frame


def verify_v1_fallback(reason: str) -> dict:
    """Publish a failure report only after complete stored-v1 revalidation.

    This performs hash, schema and arithmetic checks. The raw-inference claim is
    explicitly historical: its archived proof and saved all-row output are
    rechecked, but no native inference is executed by this function.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A concrete v2 failure reason is required")
    final = ROOT / "artifacts/final"
    submission_path = final / "submission.csv"
    selection_path = ROOT / "artifacts/blend/frozen.json"
    provenance_path = final / "release_provenance.json"
    _require_hash(submission_path, V1_SUBMISSION_SHA)
    _require_hash(selection_path, V1_SELECTION_SHA)
    _require_hash(provenance_path, V1_PROVENANCE_SHA)
    provenance = load_config(provenance_path)
    if provenance["submission_sha256"] != V1_SUBMISSION_SHA:
        raise ValueError("V1 provenance references a different submission")
    _require_hash(final / "manifest.json", provenance["final_manifest_sha256"])
    manifest = load_config(final / "manifest.json")
    selection = load_config(selection_path)
    if (manifest["submission_hash"] != V1_SUBMISSION_SHA
            or manifest["frozen_selection_hash"] != V1_SELECTION_SHA):
        raise ValueError("V1 manifest belongs to a different selection or submission")

    native_hashes = provenance.get("native_model_and_prediction_hashes", {})
    source_hashes = provenance.get("source_snapshot_hashes", {})
    if not native_hashes or not source_hashes:
        raise ValueError("Missing v1 native or source-snapshot provenance")
    for relative, digest in {**native_hashes, **source_hashes}.items():
        _require_hash(_local_path(relative), digest)

    sample = _submission(ROOT / "data/sample_submission.csv")
    ids = sample.id.to_numpy()
    test_ids = pd.read_parquet(ROOT / "data/test.parquet", columns=["id"]).id.astype(str).to_numpy()
    if not np.array_equal(test_ids, ids):
        raise ValueError("Sample identifiers differ from test identifiers")
    submitted = _submission(submission_path, ids)
    values = _probabilities(submitted[TARGET], "immutable submission")
    selected = {name: float(weight) for name, weight in selection["weights"].items() if weight > 0}
    if (not selected or not np.isfinite(list(selected.values())).all()
            or abs(sum(selected.values()) - 1.) > 1e-9):
        raise ValueError("Invalid v1 blend weights")
    members = manifest.get("members", [])
    member_ids = [member["id"] for member in members]
    if len(member_ids) != len(set(member_ids)) or set(member_ids) != set(selected):
        raise ValueError("V1 manifest and frozen members differ")
    expected = np.zeros(len(ids), dtype=np.float64)
    for member in members:
        name = member["id"]
        if Path(name).name != name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Invalid v1 member identifier")
        directory = final / name
        if (Path(member["path"]).resolve() != (directory / "model").resolve()
                or member["weight"] != selected[name]):
            raise ValueError("V1 manifest model path or weight differs from selection")
        actual = {p.relative_to(ROOT).as_posix() for p in directory.rglob("*") if p.is_file()}
        recorded = {p for p in native_hashes if p.startswith(directory.relative_to(ROOT).as_posix() + "/")}
        if actual != recorded or not any("/model/" in p for p in actual):
            raise ValueError(f"Incomplete or unrecorded v1 member inventory: {name}")
        prediction_path = directory / "test.parquet"
        frame = pd.read_parquet(prediction_path, columns=["id", "prediction"])
        if len(frame) != len(ids) or not frame.id.is_unique or frame.id.isna().any():
            raise ValueError(f"Invalid v1 prediction identifiers: {name}")
        order = pd.Index(frame.id.astype(str)).get_indexer(ids)
        if (order < 0).any():
            raise ValueError(f"Incomplete v1 prediction coverage: {name}")
        expected += selected[name] * _probabilities(frame.prediction, name)[order]
    np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-11,
                               err_msg="Independent v1 blend recomputation failed")

    proof_path = final / "raw_inference_verification.json"
    _require_hash(proof_path, provenance["raw_inference_verification_sha256"])
    proof = load_config(proof_path)
    if (proof["expected_submission_sha256"] != V1_SUBMISSION_SHA
            or proof["frozen_selection_sha256"] != V1_SELECTION_SHA
            or proof["rows"] != len(ids)):
        raise ValueError("Historical raw-inference proof has a different identity")
    _require_hash(ROOT / "data/test.csv", proof["raw_csv_sha256"])
    reproduced_path = final / "reproduced_from_raw.csv"
    _require_hash(reproduced_path, proof["reproduced_submission_sha256"])
    reproduced = _submission(reproduced_path, ids)
    rtol, atol = float(proof["rtol"]), float(proof["atol"])
    if not (np.isfinite([rtol, atol]).all() and 0 <= rtol <= 2e-5 and 0 <= atol <= 2e-6):
        raise ValueError("Invalid historical raw-inference tolerances")
    np.testing.assert_allclose(reproduced[TARGET], values, rtol=rtol, atol=atol,
                               err_msg="Stored historical raw-inference output differs")
    archive = ROOT / "artifacts/verified" / V1_SUBMISSION_SHA
    verification_path = archive / "verification.json"
    _require_hash(verification_path, provenance["independent_verification_sha256"])
    previous = load_config(verification_path)
    if (previous["sha256"] != V1_SUBMISSION_SHA or previous["rows"] != len(ids)
            or previous.get("independent_blend_recomputation") is not True):
        raise ValueError("Historical independent verification identity is invalid")
    # Recheck immutable roots before publishing any success/fallback assertion.
    _require_hash(submission_path, V1_SUBMISSION_SHA)
    _require_hash(selection_path, V1_SELECTION_SHA)
    _require_hash(provenance_path, V1_PROVENANCE_SHA)
    sensitivity = selection.get("audit_sensitivity", {})
    result = {
        "verified_utc": datetime.now(timezone.utc).isoformat(), "release_status": "fallback_v1",
        "v2_release_status": "failed", "failure_reason": reason.strip(),
        "submission_path": str(submission_path), "sha256": V1_SUBMISSION_SHA, "rows": len(ids),
        "independent_blend_recomputation": True, "native_inference_executed": False,
        "stored_native_checksums_revalidated": True, "native_and_prediction_files_checked": len(native_hashes),
        "source_snapshot_files_checked": len(source_hashes), "prior_all_row_raw_proof_revalidated": True,
        "prior_raw_inference_verified_utc": proof["verified_utc"],
        "prior_raw_inference_proof_sha256": provenance["raw_inference_verification_sha256"],
        "stored_raw_output_max_absolute_difference": float(np.max(np.abs(reproduced[TARGET].to_numpy() - values))),
        "historical_v1": {"oof_auc": selection["oof_auc"], "audit_auc": selection.get("audit_auc"),
                          "audit_unexposed_auc": selection.get("audit_unexposed_auc"),
                          "audit_sensitivity": sensitivity, "public_score": V1_PUBLIC_SCORE,
                          "kaggle_submission_id": V1_KAGGLE_SUBMISSION_ID},
        "audit_labels_read": False, "v2_audit_evaluated": False,
        "evaluation_note": "V2 release failed. Revalidated stored v1 checksums and prior all-row inference proof; no new native inference or audit evaluation.",
    }
    report = (
        "# FAILED V2 RELEASE: verified v1 fallback\n\n"
        f"Verified stored fallback at {result['verified_utc']}.\n\n"
        f"Failure reason: {reason.strip()}\n\n"
        f"Use `{submission_path}` ({len(ids):,} rows), SHA256 `{V1_SUBMISSION_SHA}`. "
        "The second-pass release is not being presented as successful.\n\n"
        "Rechecked the exact submission schema, sample/test identifiers and order, finite probability bounds, "
        "independent weighted prediction arithmetic, every recorded native/prediction file, and saved source-snapshot checksums. "
        "The prior all-row raw-inference proof and its saved output were hash-validated and compared again. "
        "No new native inference ran during this fallback check.\n\n"
        f"Historical v1 development OOF ROC AUC: {selection['oof_auc']:.8f}. "
        f"Historical original audit: {selection.get('audit_auc')}; sensitivity audit: {selection.get('audit_unexposed_auc')} "
        f"on {sensitivity.get('remaining_rows', 'previously documented')} rows, excluding "
        f"{sensitivity.get('excluded_rows', 'previously documented')} smoke-exposed rows. "
        "These historical audit scores assessed the development-fold ensemble, not a new v2 model or an independently evaluated full-data refit.\n\n"
        f"Historical v1 Kaggle public score: {V1_PUBLIC_SCORE:.5f}, submission {V1_KAGGLE_SUBMISSION_ID}. "
        "No submission was made by this fallback operation. No audit labels were read or scored.\n"
    )
    output = ROOT / "artifacts/second_pass/fallback_verification.json"
    report_path = ROOT / "SECOND_PASS_REPORT.md"
    if not output.resolve().is_relative_to((ROOT / "artifacts/second_pass").resolve()):
        raise ValueError("Fallback output escaped its campaign directory")
    atomic_json(output, result)
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    print(verify_v1_fallback(args.reason), flush=True)
