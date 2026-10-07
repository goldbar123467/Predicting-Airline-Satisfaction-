"""Verify the frozen deterministic retry and save its exclusive dispatch review."""
import json
from datetime import datetime, timezone
from pathlib import Path

from prepare_fixed_epoch_dropout_retry1 import ROOT, FOLDER, ORIGINAL, HOOK, render_entry, sha, read, require


def main():
    folder = ROOT / FOLDER
    final = folder / "dispatch_review.json"
    if final.exists():
        raise FileExistsError("Preserve prior dispatch review")
    prepared_path = folder / "preparation_manifest.json"
    prepared = read(prepared_path)
    independent_path = ROOT / "state/fixed_epoch_dropout_v1/independent_retry_launch_review.json"
    independent = read(independent_path)
    if independent.get("status") != "passed" or independent.get("blockers"):
        raise ValueError("Independent retry review has not passed")
    files = dict(prepared["files"])
    for name, expected in independent["source_hashes"].items():
        if name in files and files[name] != expected:
            raise ValueError("Independent review and frozen preparation differ")
        files[name] = expected
    for path in (prepared_path, independent_path, Path(__file__)):
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, expected in files.items():
        target = (ROOT / name).resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError("Reviewed path escaped workspace")
        require(target, expected)
    original = (ROOT / ORIGINAL / "run.py").read_text(encoding="utf-8")
    hook = (ROOT / HOOK).read_bytes().decode("utf-8")
    amendment = (folder / "runtime_amendment.json").read_bytes().decode("utf-8")
    expected = render_entry(original, hook, amendment)
    if (folder / "run.py").read_text(encoding="utf-8") != expected:
        raise ValueError("Prepared entry differs from reviewed exact transformation")
    before = read(ROOT / ORIGINAL / "kernel-metadata.json")
    after = read(folder / "kernel-metadata.json")
    if {k:v for k,v in before.items() if k not in {"id", "title"}} != {k:v for k,v in after.items() if k not in {"id", "title"}}:
        raise ValueError("Private compute or dataset metadata changed")
    value = {"status": "passed", "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "hard_timeout_seconds": 6900, "files": files, "new_dataset_upload": False,
        "independent_review": independent_path.relative_to(ROOT).as_posix(),
        "verification": "Exact reviewed entry transformation and metadata delta; all bound source, evidence and payload-anchor hashes verified",
        "scope": "Single private deterministic execution retry; unchanged scientific payload and exact prefix gate",
        "real_fits_or_quality_scoring_performed_locally": False}
    with final.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    from kaggle_fixed_epoch_dropout_retry1 import verify_review
    verify_review()
    print(json.dumps({k:v for k,v in value.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
