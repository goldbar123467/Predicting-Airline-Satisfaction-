"""Transport-only repair: signed full-path listing plus bounded GET lengths.

The original retrieval/ZIP/assembly helper remains byte-identical and hash-pinned.
No fitting, scoring, native model loading or cloud mutations are implemented.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlsplit

import retrieve_fixed_epoch_cloud_v1 as base

BASE_SHA = "17fe4bed4a622ffbe14860b65bddf711bb2fcbab0520a316ebfde0f3cfeeecaf"
EVIDENCE = "cloud/fixed_epoch_v1_retry1/listing_discrepancy_01.json"


def require_original() -> None:
    if base.sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Original tested retrieval/assembly helper changed")


def stream_http_length(url: str, path: Path, *, remaining_bytes: int, get) -> dict:
    """Obtain size from the same GET response, then bound every streamed byte."""
    if urlsplit(url).scheme != "https" or type(remaining_bytes) is not int or remaining_bytes < 0:
        raise ValueError("Invalid signed output transport/remaining budget")
    cap = min(remaining_bytes, base.MAX_ARCHIVE_BYTES if path.name == "results.zip" else base.MAX_METADATA_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    start, count, hasher = time.monotonic(), 0, hashlib.sha256()
    try:
        with get(url, stream=True, timeout=(30, 90), headers={"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            if urlsplit(response.url).scheme != "https":
                raise ValueError("Output redirect left HTTPS")
            length = response.headers.get("Content-Length")
            if not isinstance(length, str) or re.fullmatch(r"[0-9]+", length) is None:
                raise ValueError("GET must supply a valid Content-Length")
            expected = int(length)
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Output transport did not honor identity encoding")
            if expected > cap or shutil.disk_usage(path.parent).free < expected + 1024**3:
                raise ValueError("GET size exceeds fixed byte/disk reserve")
            with path.open("xb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    count += len(chunk)
                    if count > expected or time.monotonic() - start > 1800:
                        raise ValueError("GET exceeded declared bytes/time budget")
                    stream.write(chunk)
                    hasher.update(chunk)
            if count != expected:
                raise ValueError("GET ended before declared byte count")
    except Exception:
        # HTTP exceptions may contain signed credentials in the URL. Keep any
        # partial file, and expose no provider exception text or chained context.
        raise RuntimeError("Bounded signed GET failed; any partial file is preserved") from None
    return {"bytes": count, "sha256": hasher.hexdigest(), "seconds": time.monotonic() - start,
            "size_source": "same GET Content-Length", "content_encoding": "identity"}


def retrieve(root: Path, destination: Path, *, include_log: bool = False) -> dict:
    require_original()
    from kaggle_cloud_control import api_client
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    import requests
    dispatch = base.local_dispatch(root)
    if destination.exists():
        raise FileExistsError("Use a fresh retrieval directory; original attempt remains unchanged")
    api = api_client()
    terminal = str(api.kernels_status(base.KERNEL).status).rsplit(".", 1)[-1]
    if terminal not in {"COMPLETE", "ERROR", "CANCELLED"}:
        raise ValueError("Provider is not terminal")
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = base.KERNEL.split("/")
    with api.build_kaggle_client() as client:
        remote = client.kernels.kernels_api_client.get_kernel(request).metadata
    if (remote.is_private is not True or remote.current_version_number != 1
            or remote.id != base.KERNEL_ID or remote.ref != base.KERNEL):
        raise ValueError("Immutable private kernel identity differs")
    destination.mkdir(parents=True, exist_ok=False)
    amendment = {"transport_source_sha256": base.sha(Path(__file__)), "original_helper_sha256": BASE_SHA,
        "listing_discrepancy_sha256": base.sha(root / EVIDENCE),
        "change": "Use authoritative signed full paths; sizes from same streamed GET response instead of incompatible basename size endpoint"}
    base.write(destination / "retrieval_claim.json", {**dispatch, **amendment,
        "created_utc": datetime.now(timezone.utc).isoformat(), "terminal_status": terminal,
        "file_pattern": base.FILE_PATTERN, "maximum_selected_bytes": base.MAX_TOTAL_BYTES,
        "signed_urls_retained": False})
    with api.build_kaggle_client() as client:
        def fetch(token):
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = base.KERNEL.split("/")
            request.page_size, request.page_token = 100, token
            return client.kernels.kernels_api_client.list_kernel_session_output(request)
        signed, pages = base.selected_pages(fetch, field="file_name")
    if not set(signed).issubset(base.SELECTED):
        raise ValueError("Unexpected selected output path")
    files, total = {}, 0
    for name in sorted(signed):
        item = stream_http_length(signed[name].url, base.checked(destination, name),
            remaining_bytes=base.MAX_TOTAL_BYTES - total, get=requests.get)
        files[name] = item
        total += item["bytes"]
    signed.clear()
    if include_log:
        log = api.kernels_logs(base.KERNEL)
        if len(log.encode("utf-8")) > base.MAX_METADATA_BYTES:
            raise ValueError("Optional log exceeds metadata size limit")
        with (destination / "kernel.log").open("x", encoding="utf-8") as stream:
            stream.write(log)
        files["kernel.log"] = {"bytes": (destination / "kernel.log").stat().st_size,
                               "sha256": base.sha(destination / "kernel.log")}
    returned = destination / "fixed_epoch_return"
    status = base.read(returned / "cloud_status.json")
    if status.get("id") != base.ID:
        raise ValueError("Returned campaign identity differs")
    verified = False
    if "returned_artifacts" in status:
        base.validate_archive(returned, status)
        verified = True
    elif (returned / "results.zip").exists() or (returned / "output-manifest.json").exists():
        raise ValueError("Returned files lack status hash binding")
    success = base.assess_success(status, terminal)
    selected_files = set(files) - {"kernel.log"}
    if success and (selected_files != base.SELECTED or not verified):
        raise ValueError("Successful status requires all three verified return files")
    result = {**dispatch, **amendment,
        "status": "retrieved_success_unscored" if success else "retrieved_terminal_failure",
        "created_utc": datetime.now(timezone.utc).isoformat(), "terminal_status": terminal,
        "cloud_status": status["status"], "archive_verified": verified,
        "file_pattern": base.FILE_PATTERN, "output_pages": pages, "files": files,
        "signed_urls_retained": False, "training_or_scoring_performed": False}
    base.write(destination / "retrieval.json", result)
    return result


def main() -> None:
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--include-log", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    download = args.download_dir.resolve()
    if not download.is_relative_to((base.ROOT / base.FOLDER).resolve()):
        raise ValueError("Downloads must remain inside retry namespace")
    result = retrieve(base.ROOT, download, include_log=args.include_log)
    if args.assemble and result["status"] == "retrieved_success_unscored":
        result["assessment"] = base.assemble_download(base.ROOT, download)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
