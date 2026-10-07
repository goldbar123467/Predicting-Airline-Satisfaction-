"""Read-only reconciliation/status for the provider's canonical retry identity.

The first save succeeded under the title-derived slug. Never push another job.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kaggle_cloud_control import api_client

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "cloud/fixed_epoch_dropout_v1_retry1"
ACTUAL = "clarkkitchen/s6e10-dropout-deterministic-retry-20261004"
REQUESTED = "clarkkitchen/s6e10-dropout-20261004-r2"
KERNEL_ID = 137074361


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["reconcile", "status"])
    args = parser.parse_args()
    push = read(FOLDER / "push_receipt.json")
    intent = read(FOLDER / "push_intent.json")
    if (push["ref"] != "/code/" + ACTUAL or push["kernelId"] != KERNEL_ID
            or push["versionNumber"] != 1 or push["error"]
            or any(v for k,v in push.items() if k.startswith("invalid"))
            or intent["id"] != REQUESTED or intent["timeout_seconds"] != 6900):
        raise ValueError("Provider save identity or reviewed intent differs")
    api = api_client()
    record_path = FOLDER / "provider_identity_reconciliation.json"
    if args.action == "reconcile":
        if record_path.exists():
            raise FileExistsError("Preserve existing identity reconciliation")
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = ACTUAL.split("/")
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.get_kernel(request)
        remote = response.metadata.to_dict(ignore_defaults=False)
        save(FOLDER / "remote_metadata_api.json", remote, exclusive=True)
        meta = read(FOLDER / "kernel-metadata.json")
        expected = {"id": KERNEL_ID, "ref": ACTUAL, "currentVersionNumber": 1,
            "isPrivate": True, "enableGpu": True, "enableTpu": False,
            "enableInternet": True, "machineShape": "NvidiaTeslaT4",
            "dockerImage": meta["docker_image"], "datasetDataSources": meta["dataset_sources"],
            "competitionDataSources": [], "kernelDataSources": [], "modelDataSources": []}
        if any(remote.get(k) != v for k,v in expected.items()):
            raise ValueError("Canonical provider private identity/image/data contract differs")
        # kernels_pull writes text with Windows CRLF translation. Preserve that
        # first copy and verify the API source directly as UTF-8 bytes instead.
        source_path = FOLDER / "remote_source_api.py"
        with source_path.open("xb") as stream:
            stream.write(response.blob.source.encode("utf-8"))
        if sha(source_path) != intent["source_sha256"] or sha(FOLDER / "run.py") != intent["source_sha256"]:
            raise ValueError("Canonical provider source does not match unique reviewed push")
        files = [FOLDER / "push_receipt.json", FOLDER / "push_intent.json", FOLDER / "remote_metadata_api.json",
                 FOLDER / "run.py", FOLDER / "runtime_amendment.json", FOLDER / "dispatch_review.json", source_path, Path(__file__)]
        value = {"status": "verified", "verified_utc": datetime.now(timezone.utc).isoformat(),
            "requested_kernel": REQUESTED, "actual_kernel": ACTUAL, "kernel_id": KERNEL_ID, "version": 1,
            "private_verified": True, "runtime_sha256": intent["source_sha256"], "provider_timeout_seconds": 6900,
            "reason": "Provider returned the title-derived canonical slug in the successful immutable version1 save; no second push occurred",
            "source_verification": "API source encoded as UTF-8 matches reviewed bytes exactly; initial Windows-CRLF kernels_pull copy preserved separately",
            "files": {p.relative_to(ROOT).as_posix():sha(p) for p in files}, "cloud_mutations_performed": 0}
        save(record_path, value, exclusive=True)
    reconciliation = read(record_path)
    if reconciliation["status"] != "verified" or reconciliation["actual_kernel"] != ACTUAL:
        raise ValueError("Canonical identity is not verified")
    for name, expected in reconciliation["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError("Reconciled source or receipt changed")
    status = api.kernels_status(ACTUAL)
    value = {"checked_utc": datetime.now(timezone.utc).isoformat(), "ref": ACTUAL,
        "kernel_id": KERNEL_ID, "version": 1, "status": str(status.status), "failure_message": status.failure_message,
        "identity_reconciliation_sha256": sha(record_path)}
    save(FOLDER / "latest_status.json", value)
    print(json.dumps(value))


if __name__ == "__main__":
    main()
