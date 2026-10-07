"""Bounded private Kaggle probes, with durable intent and response receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / "artifacts" / "kaggle_cloud"


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def api_client():
    logging.disable(logging.CRITICAL)
    if not os.environ.get("KAGGLE_API_TOKEN"):
        credential = Path.home() / "Documents" / "ENVEDA CASMI" / "kaggle api key.txt"
        os.environ["KAGGLE_API_TOKEN"] = credential.read_text(encoding="utf-8").strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def package(kind: str) -> Path:
    folder = CLOUD / f"probe_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "scripts" / "kaggle_compute_probe.py").read_text(encoding="utf-8")
    assert source.count("EXPECT_GPU = False") == 1
    if kind == "gpu":
        source = source.replace("EXPECT_GPU = False", "EXPECT_GPU = True")
    compile(source, "probe.py", "exec")
    (folder / "probe.py").write_text(source, encoding="utf-8")
    title = f"S6E10 Compute Probe {kind.upper()} 20261002"
    meta = {"id": f"clarkkitchen/{title.lower().replace(' ', '-')}", "title": title,
            "code_file": "probe.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": kind == "gpu", "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [], "kernel_sources": [],
            "competition_sources": ["playground-series-s6e10"]}
    if kind == "gpu":
        meta["machine_shape"] = "NvidiaTeslaT4"
    write(folder / "kernel-metadata.json", meta)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "push", "status", "output"])
    parser.add_argument("kind", choices=["cpu", "gpu"])
    args = parser.parse_args()
    folder = CLOUD / f"probe_{args.kind}"
    if args.action == "prepare":
        print(package(args.kind))
        return
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    api = api_client()
    now = datetime.now(timezone.utc).isoformat()
    if args.action == "push":
        intent = folder / "push_intent.json"
        if intent.exists():
            raise RuntimeError("Push intent already exists. Inspect remote status; do not blindly retry a mutation.")
        assert meta["is_private"] is True and meta["enable_internet"] is False
        assert meta["id"].startswith("clarkkitchen/s6e10-compute-probe-")
        write(intent, {"utc": now, "ref": meta["id"], "timeout_seconds": 600,
                       "source_sha256": hashlib.sha256((folder / "probe.py").read_bytes()).hexdigest(),
                       "metadata": meta})
        response = api.kernels_push(str(folder), timeout="600", acc="NvidiaTeslaT4" if args.kind == "gpu" else None)
        receipt = {"utc": now, "ref": str(response.ref), "url": str(response.url),
                   "version_number": response.version_number, "kernel_id": response.kernel_id,
                   "error": response.error,
                   "invalid_dataset_sources": list(response.invalid_dataset_sources or []),
                   "invalid_competition_sources": list(response.invalid_competition_sources or [])}
        write(folder / "push_receipt.json", receipt)
        print(json.dumps(receipt, default=str))
        if receipt["error"] or receipt["invalid_dataset_sources"] or receipt["invalid_competition_sources"]:
            raise RuntimeError("Kaggle rejected the kernel or required data source")
    elif args.action == "status":
        response = api.kernels_status(meta["id"])
        status = {"utc": now, "ref": meta["id"], "status": str(response.status), "failure_message": response.failure_message}
        write(folder / "latest_status.json", status)
        print(json.dumps(status))
    else:
        output = folder / "output"
        output.mkdir(exist_ok=True)
        result = api.kernels_output(meta["id"], str(output), file_pattern=r"probe_report\.json$", force=True, quiet=True)
        (folder / "kernel.log").write_text(api.kernels_logs(meta["id"]), encoding="utf-8")
        remote = folder / "remote_source"
        api.kernels_pull(meta["id"], str(remote), metadata=True, quiet=True)
        remote_meta = json.loads((remote / "kernel-metadata.json").read_text())
        if remote_meta["is_private"] is not True:
            raise RuntimeError("Remote private flag does not match requested scope")
        print({"output": str(output), "result": str(result)})


if __name__ == "__main__":
    main()
