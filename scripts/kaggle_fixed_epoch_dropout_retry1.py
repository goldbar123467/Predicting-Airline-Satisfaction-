"""Private deterministic-runtime retry using the unchanged dataset; no upload or local fitting."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from kaggle_cloud_control import api_client

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "cloud/fixed_epoch_dropout_v1_retry1"
DATASET = "clarkkitchen/s6e10-dropout-bundle-20261004"
KERNEL = "clarkkitchen/s6e10-dropout-20261004-r2"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=str)
        f.write("\n")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require_absent(api, kind):
    # Missing owned slugs return ambiguous403 on direct status. The official,
    # authenticated own-resource listing supplies the complete creation check.
    target = DATASET if kind == "dataset" else KERNEL
    token, seen, count, pages = None, set(), 0, []
    for _ in range(100):
        method = api.dataset_list_with_response if kind == "dataset" else api.kernels_list_with_response
        response = method(mine=True, page_size=100, page_token=token)
        entries = response.datasets if kind == "dataset" else response.kernels
        for item in entries:
            ref = str(item.ref).removeprefix("/datasets/").removeprefix("/code/")
            if ref == target:
                raise FileExistsError("Immutable cloud slug already exists; reconcile without a new version")
        count += len(entries)
        pages.append(hashlib.sha256(response.to_json(ignore_defaults=False).encode()).hexdigest())
        token = response.next_page_token
        if not token:
            write(FOLDER / f"{kind}_absence.json", {"checked_utc": stamp(), "target": target,
                  "method": "exhausted authenticated mine listing", "entries": count,
                  "page_hashes": pages, "absent": True})
            return
        if token in seen:
            raise RuntimeError("Repeated resource-list page token")
        seen.add(token)
    raise RuntimeError("Own-resource listing exceeded bounded pagination")


def dataset_readback(api):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    request = ApiGetDatasetRequest()
    request.owner_slug, request.dataset_slug = DATASET.split("/")
    with api.build_kaggle_client() as client:
        response = client.datasets.dataset_api_client.get_dataset(request)
        value = response.to_dict(ignore_defaults=False)
    write(FOLDER / "dataset_remote_metadata.json", value)
    # Official GetDataset returns privacy/ref at the top level on this SDK.
    if value.get("isPrivate") is not True or value.get("ref") != DATASET:
        raise ValueError("Remote dataset is not explicitly private")
    status = api.dataset_status(DATASET, format="json")
    result = {"checked_utc": stamp(), "dataset": DATASET, "metadata_sha256": sha(FOLDER / "dataset_remote_metadata.json"),
              "status": json.loads(status), "private_verified": True}
    write(FOLDER / "dataset_readback.json", result)
    return result


def kernel_readback(api):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = KERNEL.split("/")
    with api.build_kaggle_client() as client:
        result = client.kernels.kernels_api_client.get_kernel(request)
    if result.metadata.is_private is not True:
        raise ValueError("Remote kernel is not explicitly private")
    write(FOLDER / "remote_metadata.json", result.metadata.to_dict(ignore_defaults=False))
    destination = FOLDER / "remote_source"
    api.kernels_pull(KERNEL, str(destination), metadata=True, quiet=True)
    names = [p for p in destination.iterdir() if p.suffix == ".py"]
    if len(names) != 1 or sha(names[0]) != sha(FOLDER / "run.py"):
        raise ValueError("Remote kernel source differs from reviewed dispatch")
    return {"private_verified": True, "runtime_sha256": sha(names[0])}


def verify_review():
    review = read(FOLDER / "dispatch_review.json")
    if review.get("status") != "passed" or review.get("hard_timeout_seconds") != 6900:
        raise ValueError("Reviewed cloud dispatch contract is missing")
    for name, expected in review["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError(f"Reviewed dispatch bytes changed: {name}")
    meta = read(FOLDER / "kernel-metadata.json")
    if (meta.get("id") != KERNEL or meta.get("is_private") is not True
            or meta.get("enable_gpu") is not True or meta.get("enable_tpu") is not False
            or meta.get("enable_internet") is not True or meta.get("machine_shape") != "NvidiaTeslaT4"
            or meta.get("dataset_sources") != [DATASET] or meta.get("competition_sources") != []
            or meta.get("kernel_sources") != [] or meta.get("model_sources") != []):
        raise ValueError("Private cloud metadata changed")
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["dataset-status", "push", "status"])
    args = parser.parse_args()
    api = api_client()
    if args.action == "dataset-status":
        print(json.dumps(dataset_readback(api), default=str))
        return
    if args.action == "push":
        verify_review()
        if (FOLDER / "push_intent.json").exists():
            raise FileExistsError("Existing kernel intent; reconcile remote state")
        require_absent(api, "kernel")
        dataset = dataset_readback(api)
        if dataset["status"]["status"] != "ready" or dataset["status"].get("current_version_number") != 1:
            raise ValueError("Require ready private immutable dataset version1")
        quota = api.quota_view()
        remaining = (quota.gpu_quota.total_time_allowed - quota.gpu_quota.time_used - quota.gpu_quota.time_reserved).total_seconds()
        if (remaining < 6900 or quota.gpu_quota.time_reserved.total_seconds() != 0
                or quota.gpu_quota.is_pay_to_scale_enabled):
            raise ValueError("Require sufficient free allowance, no reserved GPU session and disabled paid scaling")
        write(FOLDER / "quota_at_dispatch.json", {"checked_utc": stamp(), "quota": quota.to_dict(ignore_defaults=False)}, exclusive=True)
        write(FOLDER / "push_intent.json", {"created_utc": stamp(), "id": KERNEL, "timeout_seconds": 6900,
              "source_sha256": sha(FOLDER / "run.py"), "metadata_sha256": sha(FOLDER / "kernel-metadata.json")}, exclusive=True)
        response = api.kernels_push(str(FOLDER), timeout="6900", acc="NvidiaTeslaT4")
        receipt = response.to_dict(ignore_defaults=False)
        write(FOLDER / "push_receipt.json", receipt, exclusive=True)
        if (response.error or str(response.ref) not in {KERNEL, f"/code/{KERNEL}"} or response.version_number != 1
                or any(v for k, v in receipt.items() if k.startswith("invalid"))):
            raise RuntimeError("Kernel save failed or immutable version1 contract differs; reconcile intent")
        print(json.dumps({"receipt": receipt, "readback": kernel_readback(api)}, default=str))
        return
    receipt = read(FOLDER / "push_receipt.json")
    if receipt["versionNumber"] != 1:
        raise ValueError("Only immutable version1 readback is supported")
    status = api.kernels_status(KERNEL)
    value = {"checked_utc": stamp(), "ref": KERNEL, "version": 1,
             "status": str(status.status), "failure_message": status.failure_message}
    write(FOLDER / "latest_status.json", value)
    print(json.dumps(value))
    return


if __name__ == "__main__":
    main()
