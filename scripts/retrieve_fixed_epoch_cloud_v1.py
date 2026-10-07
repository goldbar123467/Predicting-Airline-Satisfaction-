"""Retrieve allowlisted cloud outputs and assemble a verified, isolated workspace.

No fits, scoring, model imports or inference. Network actions occur only with the
explicit retrieve command; assemble consumes preserved local downloads only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
from urllib.parse import urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ID = "fixed_epoch_cloud_v1"
KERNEL = "clarkkitchen/s6e10-fixed-epoch-20261003-r2"
KERNEL_ID = 136851188
FOLDER = "cloud/fixed_epoch_v1_retry1"
PAYLOAD = "cloud/fixed_epoch_v1/bundle/payload"
WORKSPACE = "cloud/fixed_epoch_v1/assessment_workspace"
PREPARATION = "cloud/fixed_epoch_v1/preparation_manifest.json"
BUNDLE_SHA = "d7a5cce4652f33f3a81512d1b18681942ffc7065b931b3d014fc475e19ab494b"
REGISTRY_SHA = "2aa065324331a5eed7ad464c82bcd6d967d91d38c5a453f9a7a71ee9fd03266d"
FILE_PATTERN = r"^fixed_epoch_return/(?:cloud_status\.json|output-manifest\.json|results\.zip)$"
SELECTED = {"fixed_epoch_return/cloud_status.json", "fixed_epoch_return/output-manifest.json", "fixed_epoch_return/results.zip"}
MAX_ARCHIVE_BYTES = 8 * 1024**3
MAX_METADATA_BYTES = 32 * 1024**2
MAX_TOTAL_BYTES = MAX_ARCHIVE_BYTES + 2 * MAX_METADATA_BYTES
MAX_UNPACKED_BYTES = 16 * 1024**3
MAX_MEMBER_BYTES = 2 * 1024**3
MAX_PAGES = 100


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("JSON metadata exceeds bounded size")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def safe_name(name: str) -> str:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(ord(char) < 32 for char in name) or name.startswith("/") or name.endswith("/")):
        raise ValueError("Unsafe portable artifact name")
    parts = name.split("/")
    devices = {"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1, 10)], *[f"lpt{i}" for i in range(1, 10)]}
    if any(part in {"", ".", ".."} or part.endswith((".", " ")) or part.split(".")[0].casefold() in devices for part in parts):
        raise ValueError("Unsafe portable path component")
    if PurePosixPath(name).as_posix() != name:
        raise ValueError("Noncanonical portable path")
    return name


def checked(root: Path, name: str) -> Path:
    path = root / safe_name(name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes root")
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink in artifact path")
    return path


def inventory_names(names) -> set[str]:
    values = [safe_name(name) for name in names]
    if len(values) != len(set(values)) or len(values) != len({name.casefold() for name in values}):
        raise ValueError("Duplicate or case-colliding artifact names")
    result = set(values)
    folded = {name.casefold() for name in values}
    for name in values:
        parts = name.split("/")
        if any("/".join(parts[:end]).casefold() in folded for end in range(1, len(parts))):
            raise ValueError("File/directory prefix collision")
    return result


def validate_item(item: dict, *, maximum: int = MAX_MEMBER_BYTES) -> None:
    if (type(item.get("bytes")) is not int or not 0 <= item["bytes"] <= maximum
            or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])):
        raise ValueError("Invalid file size or SHA256 identity")


def output_allowed(name: str) -> bool:
    return name in {"registry.json", f"configs/{ID}.json"} or any(
        name.startswith(f"{part}/{ID}/") for part in ("artifacts", "state", "logs"))


def validate_archive(returned: Path, cloud_status: dict) -> dict:
    """Stream every member without extracting or interpreting model/data bytes."""
    archive_path, manifest_path = returned / "results.zip", returned / "output-manifest.json"
    binding = cloud_status["returned_artifacts"]
    if sha(archive_path) != binding["archive_sha256"] or sha(manifest_path) != binding["manifest_sha256"]:
        raise ValueError("Returned archive/manifest SHA256 differs from cloud status")
    manifest = read(manifest_path)
    if manifest.get("id") != ID or not isinstance(manifest.get("files"), dict):
        raise ValueError("Returned campaign manifest identity differs")
    names = inventory_names(manifest["files"])
    if any(not output_allowed(name) for name in names) or binding["file_count"] != len(names):
        raise ValueError("Return inventory is outside campaign scope or count differs")
    for item in manifest["files"].values():
        validate_item(item)
    if sum(item["bytes"] for item in manifest["files"].values()) > MAX_UNPACKED_BYTES:
        raise ValueError("Returned artifact scale exceeds bounded allowance")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        members = inventory_names([item.filename for item in infos])
        if members != names | {"output-manifest.json"}:
            raise ValueError("ZIP members differ from exact returned manifest")
        for info in infos:
            kind = stat.S_IFMT(info.external_attr >> 16)
            if info.is_dir() or kind not in {0, stat.S_IFREG} or info.flag_bits & 1:
                raise ValueError("Directory, symlink/special file or encrypted ZIP member")
            expected = (manifest["files"][info.filename] if info.filename != "output-manifest.json" else
                        {"bytes": manifest_path.stat().st_size, "sha256": binding["manifest_sha256"]})
            if info.file_size != expected["bytes"] or info.file_size > MAX_MEMBER_BYTES:
                raise ValueError("ZIP member size differs")
            with archive.open(info) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                    raise ValueError("ZIP member bytes differ")
    return manifest


def verify_payload(payload: Path, expected_manifest: str = BUNDLE_SHA) -> dict:
    path = payload / "bundle-manifest.json"
    if sha(path) != expected_manifest:
        raise ValueError("Prepared payload manifest changed")
    manifest = read(path)
    if manifest.get("id") != ID or not isinstance(manifest.get("files"), dict):
        raise ValueError("Prepared payload manifest identity differs")
    names = inventory_names(manifest["files"])
    actual = inventory_names([p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file() or p.is_symlink()])
    if actual != names | {"bundle-manifest.json"}:
        raise ValueError("Prepared payload tree differs from exact manifest")
    for name, item in manifest["files"].items():
        validate_item(item)
        source = checked(payload, name)
        if source.stat().st_size != item["bytes"] or sha(source) != item["sha256"]:
            raise ValueError("Prepared payload file changed")
    return manifest


def assess_success(status: dict, terminal: str) -> bool:
    return (terminal == "COMPLETE" and status.get("id") == ID and status.get("status") == "training_complete"
            and status.get("completed_fits") == 12 and status.get("outer_metrics_computed") is False
            and status.get("local_training") is False)


def assemble_workspace(payload: Path, returned: Path, destination: Path, terminal: str,
                       *, expected_bundle: str = BUNDLE_SHA, expected_registry: str = REGISTRY_SHA) -> dict:
    status = read(returned / "cloud_status.json")
    if not assess_success(status, terminal):
        raise ValueError("Failed or incomplete cloud execution cannot create an assessment workspace")
    manifest = validate_archive(returned, status)
    bundle = verify_payload(payload, expected_bundle)
    if bundle["registry_sha256"] != expected_registry or sha(payload / "registry.json") != expected_registry:
        raise ValueError("Prepared registry identity differs")
    overlaps = set(bundle["files"]) & set(manifest["files"])
    permitted = {"registry.json", f"configs/{ID}.json"}
    if overlaps != permitted:
        raise ValueError("Unexpected payload/return conflict or missing registry/wrapper")
    for name in overlaps:
        if bundle["files"][name] != manifest["files"][name]:
            raise ValueError("Returned registry/wrapper differs from prepared bytes")
    required = {f"artifacts/{ID}/completion_receipt.json", f"artifacts/{ID}/completed_manifest.json", f"state/{ID}/run_state.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Returned inventory omits completed-campaign evidence")
    # Check completion counts and bindings as metadata only. Detailed row/model
    # evidence stays the responsibility of the frozen, separately invoked evaluator.
    with zipfile.ZipFile(returned / "results.zip") as archive:
        for name in (f"artifacts/{ID}/completion_receipt.json", f"state/{ID}/run_state.json"):
            if manifest["files"][name]["bytes"] > MAX_METADATA_BYTES:
                raise ValueError("Completion/state metadata exceeds bounded size")
        complete = json.loads(archive.read(f"artifacts/{ID}/completion_receipt.json"))
        bindings = {"id": ID, "registry_sha256": expected_registry,
                    "campaign_sha256": bundle["files"][f"configs/{ID}.json"]["sha256"]}
        expected = {**bindings, "status": "completed", "completed_fit_count": 12,
                    "completed_endpoint_count": 9, "evaluation_ready": True}
        if any(complete.get(key) != value for key, value in expected.items()):
            raise ValueError("Returned completion counts or bindings differ")
        state = json.loads(archive.read(f"state/{ID}/run_state.json"))
        expected_state = {**bindings, "status": "training_complete", "completed_fits": 12,
                          "active_child": None, "outer_metrics_computed": False}
        if any(state.get(key) != value for key, value in expected_state.items()):
            raise ValueError("Returned controller did not complete cleanly")
        completed_name = f"artifacts/{ID}/completed_manifest.json"
        if manifest["files"][completed_name]["sha256"] != complete["completed_manifest_sha256"]:
            raise ValueError("Completed manifest binding differs")
    needed = sum(item["bytes"] for item in bundle["files"].values()) + sum(item["bytes"] for name, item in manifest["files"].items() if name not in overlaps)
    if shutil.disk_usage(destination.parent).free < needed + 1024**3:
        raise ValueError("Insufficient disk reserve for isolated assessment workspace")
    destination.mkdir(parents=False, exist_ok=False)
    for name in sorted(set(bundle["files"]) | {"bundle-manifest.json"}):
        source, target = checked(payload, name), checked(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    with zipfile.ZipFile(returned / "results.zip") as archive:
        for name in sorted(set(manifest["files"]) | {"output-manifest.json"}):
            target = checked(destination, name)
            if name in overlaps:
                if sha(target) != manifest["files"][name]["sha256"]:
                    raise ValueError("Copied overlap no longer matches")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    # Exact evaluator inventory; no helper receipt is inserted inside its tree.
    for name, item in {**bundle["files"], **manifest["files"]}.items():
        target = checked(destination, name)
        if target.stat().st_size != item["bytes"] or sha(target) != item["sha256"]:
            raise ValueError("Assembled workspace hash/size mismatch")
    if sha(destination / "output-manifest.json") != status["returned_artifacts"]["manifest_sha256"]:
        raise ValueError("Assembled return manifest changed")
    return {"status": "assembled_unscored", "workspace": str(destination), "output_manifest_sha256": sha(destination / "output-manifest.json"),
            "bundle_manifest_sha256": expected_bundle, "registry_sha256": expected_registry,
            "returned_file_count": len(manifest["files"]), "training_or_scoring_performed": False}


def selected_pages(fetch, *, field: str) -> tuple[dict, list[dict]]:
    """Exhaust official pagination explicitly; opaque tokens/URLs are never logged."""
    token, seen, selected, pages = "", set(), {}, []
    pattern = re.compile(FILE_PATTERN)
    for page in range(MAX_PAGES):
        response = fetch(token)
        names = []
        for item in response.files or []:
            name = getattr(item, field)
            if not pattern.fullmatch(name):
                continue
            safe_name(name)
            if name in selected:
                raise ValueError("Duplicate selected output across pages")
            selected[name] = item
            names.append(name)
        next_token = response.next_page_token or ""
        pages.append({"page": page + 1, "listed_count": len(response.files or []), "selected_names": names,
                      "has_next_page": bool(next_token)})
        if not next_token:
            return selected, pages
        if next_token in seen:
            raise ValueError("Repeated provider pagination token")
        seen.add(next_token)
        token = next_token
    raise ValueError("Provider output listing exceeded pagination bound")


def stream_download(url: str, path: Path, expected_bytes: int, *, get) -> dict:
    if urlsplit(url).scheme != "https" or type(expected_bytes) is not int or expected_bytes < 0:
        raise ValueError("Invalid signed output URL/size")
    cap = MAX_ARCHIVE_BYTES if path.name == "results.zip" else MAX_METADATA_BYTES
    if expected_bytes > cap:
        raise ValueError("Provider-listed output exceeds fixed size bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    start, count, hasher = time.monotonic(), 0, hashlib.sha256()
    try:
        with get(url, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            if urlsplit(response.url).scheme != "https":
                raise ValueError("Output redirect left HTTPS")
            header = response.headers.get("Content-Length")
            if header is not None and int(header) != expected_bytes:
                raise ValueError("Provider listing and HTTP size differ")
            with path.open("xb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    count += len(chunk)
                    if count > expected_bytes or count > cap or time.monotonic() - start > 1800:
                        raise ValueError("Streaming output exceeds listed bytes or time budget")
                    stream.write(chunk)
                    hasher.update(chunk)
    except Exception:
        # Requests exceptions can include signed URLs. Preserve partial bytes but
        # never propagate or serialize the original secret-bearing exception text.
        raise RuntimeError("Signed output stream failed; preserved any partial file for inspection") from None
    if count != expected_bytes:
        raise ValueError("Downloaded size differs from provider listing")
    return {"bytes": count, "sha256": hasher.hexdigest(), "seconds": time.monotonic() - start}


def local_dispatch(root: Path) -> dict:
    folder = root / FOLDER
    push, intent = read(folder / "push_receipt.json"), read(folder / "push_intent.json")
    if (push.get("versionNumber") != 1 or push.get("kernelId") != KERNEL_ID or push.get("error")
            or push.get("ref") != "/code/" + KERNEL or intent.get("id") != KERNEL
            or sha(folder / "run.py") != intent["source_sha256"]):
        raise ValueError("Unique dispatched kernel identity changed")
    return {"kernel": KERNEL, "kernel_id": KERNEL_ID, "version": 1,
            "push_receipt_sha256": sha(folder / "push_receipt.json"), "runtime_sha256": intent["source_sha256"]}


def retrieve(root: Path, destination: Path, *, include_log: bool = False) -> dict:
    # Imports that can authenticate or reach the network are strictly command-local.
    from kaggle_cloud_control import api_client
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    import requests
    dispatch = local_dispatch(root)
    if destination.exists():
        raise FileExistsError("Use a fresh download directory; originals are never overwritten")
    api = api_client()
    status = api.kernels_status(KERNEL)
    terminal = str(status.status).rsplit(".", 1)[-1]
    if terminal not in {"COMPLETE", "ERROR", "CANCELLED"}:
        raise ValueError("Wait for terminal provider status before retrieval")
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = KERNEL.split("/")
    with api.build_kaggle_client() as client:
        remote = client.kernels.kernels_api_client.get_kernel(request).metadata
    if (remote.is_private is not True or remote.current_version_number != 1 or remote.id != KERNEL_ID
            or remote.ref != KERNEL):
        raise ValueError("Remote private/version/kernel identity differs")
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / "retrieval_claim.json", {**dispatch, "created_utc": datetime.now(timezone.utc).isoformat(),
        "terminal_status": terminal, "file_pattern": FILE_PATTERN, "maximum_selected_bytes": MAX_TOTAL_BYTES,
        "signed_urls_retained": False, "method": "Official output listing; same file_pattern semantics as kernels_output, with bounded streaming instead of response.content"})
    sized, size_pages = selected_pages(lambda token: api.kernels_list_files(KERNEL, page_token=token, page_size=100), field="name")
    sizes = {name: item.size for name, item in sized.items()}
    if any(type(value) is not int or value < 0 for value in sizes.values()) or sum(sizes.values()) > MAX_TOTAL_BYTES:
        raise ValueError("Selected provider file sizes exceed bounded artifact scale")
    if shutil.disk_usage(destination).free < sum(sizes.values()) + 1024**3:
        raise ValueError("Insufficient disk reserve for preserved downloads")
    with api.build_kaggle_client() as client:
        def fetch(token):
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = KERNEL.split("/")
            request.page_size, request.page_token = 100, token
            # Do not set version_label: existing immutable version1 contract and
            # readback avoid the account's previously observed version-label404.
            return client.kernels.kernels_api_client.list_kernel_session_output(request)
        signed, output_pages = selected_pages(fetch, field="file_name")
    if set(signed) != set(sized) or not set(signed).issubset(SELECTED):
        raise ValueError("Size and signed output listings disagree")
    files = {}
    for name in sorted(signed):
        files[name] = stream_download(signed[name].url, checked(destination, name), sizes[name], get=requests.get)
    signed.clear()
    if include_log:
        log = api.kernels_logs(KERNEL)
        if len(log.encode("utf-8")) > MAX_METADATA_BYTES:
            raise ValueError("Optional kernel log exceeds bounded size")
        with (destination / "kernel.log").open("x", encoding="utf-8") as stream:
            stream.write(log)
        files["kernel.log"] = {"bytes": (destination / "kernel.log").stat().st_size, "sha256": sha(destination / "kernel.log")}
    returned = destination / "fixed_epoch_return"
    if not (returned / "cloud_status.json").is_file():
        raise ValueError("Terminal output lacks cloud status; originals preserved")
    cloud_status = read(returned / "cloud_status.json")
    if cloud_status.get("id") != ID:
        raise ValueError("Returned cloud status campaign differs")
    archive_verified = False
    if "returned_artifacts" in cloud_status:
        validate_archive(returned, cloud_status)
        archive_verified = True
    elif (returned / "results.zip").exists() or (returned / "output-manifest.json").exists():
        raise ValueError("Return archive lacks cloud-status hash binding")
    success = assess_success(cloud_status, terminal)
    if success and (set(sizes) != SELECTED or not archive_verified):
        raise ValueError("Successful cloud status has incomplete verified returned artifacts")
    result = {**dispatch, "status": "retrieved_success_unscored" if success else "retrieved_terminal_failure",
        "created_utc": datetime.now(timezone.utc).isoformat(), "terminal_status": terminal,
        "cloud_status": cloud_status["status"], "archive_verified": archive_verified,
        "file_pattern": FILE_PATTERN, "size_pages": size_pages, "output_pages": output_pages,
        "files": files, "signed_urls_retained": False, "training_or_scoring_performed": False}
    write(destination / "retrieval.json", result)
    return result


def assemble_download(root: Path, download: Path) -> dict:
    receipt = read(download / "retrieval.json")
    dispatch = local_dispatch(root)
    if any(receipt.get(key) != value for key, value in dispatch.items()) or receipt.get("status") != "retrieved_success_unscored":
        raise ValueError("Only this verified successful immutable cloud return may be assembled")
    for name, item in receipt["files"].items():
        path = checked(download, name)
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("Preserved original download changed")
    result = assemble_workspace(root / PAYLOAD, download / "fixed_epoch_return", root / WORKSPACE, receipt["terminal_status"])
    write(download / "assembly_receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["retrieve", "assemble"])
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--include-log", action="store_true")
    parser.add_argument("--assemble", action="store_true", help="After successful retrieval only, assemble the registered workspace")
    args = parser.parse_args()
    download = args.download_dir.resolve()
    if not download.is_relative_to((ROOT / FOLDER).resolve()):
        raise ValueError("Downloads must stay under the immutable retry namespace")
    if args.action == "assemble":
        result = assemble_download(ROOT, download)
    else:
        result = retrieve(ROOT, download, include_log=args.include_log)
        if args.assemble and result["status"] == "retrieved_success_unscored":
            result["assessment"] = assemble_download(ROOT, download)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
