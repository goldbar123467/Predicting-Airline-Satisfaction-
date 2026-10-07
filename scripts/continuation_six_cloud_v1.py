"""One-shot, private cloud lifecycle for six registered continuation campaigns.

No model imports, fitting, scoring or submission. Mutations require an explicit
CLI action and immutable local authorization/preparation. Unknown outcomes keep
their intent and block another launch until read-only reconciliation succeeds.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "continuation_six_v1"
STAGES = [f"fixed_epoch_continuation_{i:02d}" for i in range(1, 7)]
TERMINAL = {"COMPLETE", "ERROR", "CANCELLED"}
TRANSPORT_SHA = "274e6b8411e131c08a0eb16760debff38d4c78eeb2bc4987be8fd04207ce0bef"
MAX_JSON = 32 * 1024**2
MAX_ARCHIVE = 8 * 1024**3
MAX_UNPACKED = 16 * 1024**3


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON:
        raise ValueError("Metadata exceeds size bound")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected metadata object")
    return value


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def transport():
    source = ROOT / "scripts/retrieve_fixed_epoch_dropout_retry1.py"
    if sha(source) != TRANSPORT_SHA:
        raise ValueError("Reviewed transport helper changed")
    import retrieve_fixed_epoch_dropout_retry1 as helper
    return helper


def checked(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or name.startswith("/") or any(p in {"", ".", ".."} for p in name.split("/"))):
        raise ValueError("Expected relative portable path")
    target = root / name
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes workspace")
    current = root
    for part in name.split("/"):
        current /= part
        if current.is_symlink():
            raise ValueError("Symlink in registered path")
    return target


def require_hash(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or re.fullmatch("[a-f0-9]{64}", expected) is None or sha(path) != expected:
        raise ValueError(f"Registered bytes changed: {path.name}")


def plan_context(root: Path, plan_path: Path, stage: str, phase: str, *, mutation=False, now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    plan_path = plan_path.resolve()
    if not plan_path.is_relative_to(root.resolve()):
        raise ValueError("Plan outside workspace")
    plan = read(plan_path)
    if (plan.get("id") != SEQUENCE or [x.get("id") for x in plan.get("stages", [])] != STAGES
            or [x.get("index") for x in plan["stages"]] != list(range(1, 7)) or stage not in STAGES
            or phase not in {"experiment", "release"}):
        raise ValueError("Require six ordered, registered stages")
    authority_path = checked(root, plan.get("authority_path", f"state/{SEQUENCE}/authorization.json"))
    authority = read(authority_path)
    require_hash(plan_path, authority.get("plan_sha256"))
    start, end = (datetime.fromisoformat(authority[k]) for k in ("requested_utc", "expires_utc"))
    if (authority.get("status") != "authorized" or start.tzinfo is None or end.tzinfo is None
            or not 0 < (end - start).total_seconds() <= 86400
            or authority.get("max_gpu_seconds") != 57600
            or authority.get("experiment_timeout_seconds") != 7200
            or authority.get("release_timeout_seconds") != 2400):
        raise ValueError("Invalid isolated sequence authorization")
    if mutation and not start <= now < end:
        raise ValueError("Sequence mutation outside authorized wall window")
    suffix = ".release" if phase == "release" else ""
    registration_path = root / f"state/{SEQUENCE}/registrations/{stage}{suffix}.json"
    registration = read(registration_path)
    expected = {"status": "registered", "stage": stage, "phase": phase, "plan_sha256": sha(plan_path)}
    if any(registration.get(k) != v for k, v in expected.items()):
        raise ValueError("Operation registration differs")
    folder = root / f"cloud/{SEQUENCE}/{stage}/{phase}"
    prep_path = folder / "preparation_manifest.json"
    if checked(root, registration["preparation_path"]).resolve() != prep_path.resolve():
        raise ValueError("Prepared folder is outside registered operation")
    require_hash(prep_path, registration["preparation_sha256"])
    prep = read(prep_path)
    cap = 7200 if phase == "experiment" else 2400
    campaign_id = stage if phase == "experiment" else stage + "_release"
    if (prep.get("id") != campaign_id or prep.get("phase") != phase
            or prep.get("hard_timeout_seconds") != cap or prep.get("private_required") is not True
            or (phase == "experiment" and prep.get("fit_budget_seconds") != 6300)):
        raise ValueError("Prepared operation contract differs")
    frozen = prep.get("frozen_files")
    if not isinstance(frozen, dict) or not frozen:
        raise ValueError("Missing frozen preparation inventory")
    for name, expected_sha in frozen.items():
        require_hash(checked(root, name), expected_sha)
    required = [folder / "run.py", folder / "kernel-metadata.json",
                folder / "bundle/upload/payload.zip", folder / "bundle/upload/dataset-metadata.json",
                folder / "bundle/payload/registry.json", folder / "bundle/payload/bundle-manifest.json"]
    if any(p.relative_to(root).as_posix() not in frozen for p in required):
        raise ValueError("Preparation omits a dispatch boundary")
    for file, field in [("run.py", "runtime_sha256"), ("bundle/upload/payload.zip", "archive_sha256"),
                        ("bundle/payload/registry.json", "registry_sha256"),
                        ("bundle/payload/bundle-manifest.json", "manifest_sha256")]:
        require_hash(folder / file, prep[field])
    meta = read(folder / "kernel-metadata.json")
    expected = {"id": prep["kernel_id"], "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [prep["dataset_id"]],
                "competition_sources": [], "kernel_sources": [], "model_sources": [], "code_file": "run.py"}
    if any(meta.get(k) != v for k, v in expected.items()) or not meta.get("docker_image"):
        raise ValueError("Private immutable cloud metadata differs")
    for name in (prep["kernel_id"], prep["dataset_id"]):
        if re.fullmatch(r"clarkkitchen/[a-z0-9][a-z0-9-]+", name) is None:
            raise ValueError("Unexpected cloud owner or slug")
    if phase == "release":
        selection = checked(root, registration["release_selection_path"])
        require_hash(selection, registration["release_selection_sha256"])
        if read(selection).get("status") != "frozen_for_release":
            raise ValueError("Full-data release requires an explicit frozen selection")
    if mutation and (end - now).total_seconds() < cap:
        raise ValueError("Insufficient wall-clock time for provider cap")
    return {"root": root, "plan": plan, "plan_path": plan_path, "authority_path": authority_path,
            "authority": authority, "stage": stage, "phase": phase, "id": campaign_id,
            "folder": folder, "prep": prep, "metadata": meta, "registration_path": registration_path,
            "cap": cap, "deadline": end,
            "bindings": {plan_path: sha(plan_path), authority_path: sha(authority_path),
                         registration_path: sha(registration_path), prep_path: sha(prep_path)}}


def current_mutation_contract(ctx: dict) -> None:
    """Recheck after remote preflight latency, immediately before side effects."""
    if (ctx["deadline"] - datetime.now(timezone.utc)).total_seconds() < ctx["cap"]:
        raise ValueError("Insufficient wall-clock time for provider cap")
    for path, expected in ctx["bindings"].items():
        require_hash(path, expected)
    for name, expected in ctx["prep"]["frozen_files"].items():
        require_hash(checked(ctx["root"], name), expected)


@contextmanager
def operation_lock(root: Path):
    path = root / f"state/{SEQUENCE}/operation.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another sequence transaction is active") from exc
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def quota_snapshot(quota) -> dict:
    gpu = quota.gpu_quota
    result = {key: {"total_seconds": float(getattr(gpu, attr).total_seconds()),
                    "days": getattr(gpu, attr).days, "seconds": getattr(gpu, attr).seconds,
                    "microseconds": getattr(gpu, attr).microseconds}
              for key, attr in [("total", "total_time_allowed"), ("used", "time_used"), ("reserved", "time_reserved")]}
    for value in result.values():
        if not math.isfinite(value["total_seconds"]) or value["total_seconds"] < 0:
            raise ValueError("Invalid numeric GPU quota")
    result["remaining_seconds"] = result["total"]["total_seconds"] - result["used"]["total_seconds"] - result["reserved"]["total_seconds"]
    result["paid_scaling_enabled"] = gpu.is_pay_to_scale_enabled
    result["checked_utc"] = stamp()
    return result


def own_resources(api, kind: str) -> list:
    method = api.dataset_list_with_response if kind == "dataset" else api.kernels_list_with_response
    token, seen, result = None, set(), []
    for _ in range(100):
        response = method(mine=True, page_size=100, page_token=token)
        result.extend((response.datasets if kind == "dataset" else response.kernels) or [])
        token = response.next_page_token
        if not token:
            return result
        if token in seen:
            raise ValueError("Repeated own-resource page token")
        seen.add(token)
    raise ValueError("Resource pagination exceeded bound")


def normalized_ref(value: str) -> str:
    return str(value).removeprefix("/code/").removeprefix("/datasets/")


def require_absent(api, kind: str, ref: str) -> None:
    if any(normalized_ref(item.ref) == ref for item in own_resources(api, kind)):
        raise FileExistsError("Immutable resource already exists; reconcile, never create a new version")


def dataset_readback(api, ref: str) -> dict:
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    request = ApiGetDatasetRequest()
    request.owner_slug, request.dataset_slug = ref.split("/")
    with api.build_kaggle_client() as client:
        metadata = client.datasets.dataset_api_client.get_dataset(request).to_dict(ignore_defaults=False)
    status = api.dataset_status(ref, format="json")
    status = json.loads(status) if isinstance(status, str) else status
    if (metadata.get("ref") != ref or metadata.get("isPrivate") is not True
            or status.get("status") != "ready" or status.get("current_version_number") != 1):
        raise ValueError("Require ready private dataset version1")
    return {"metadata": metadata, "status": status, "checked_utc": stamp()}


def remote_kernel(api, ref: str) -> tuple[dict, bytes]:
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = ref.split("/")
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(request)
    return response.metadata.to_dict(ignore_defaults=False), response.blob.source.encode("utf-8")


def validate_remote(ctx: dict, ref: str, kernel_id: int, metadata: dict, source: bytes) -> None:
    local = ctx["metadata"]
    expected = {"id": kernel_id, "ref": ref, "currentVersionNumber": 1, "isPrivate": True,
                "enableGpu": True, "enableTpu": False, "enableInternet": local.get("enable_internet", False),
                "machineShape": "NvidiaTeslaT4", "dockerImage": local["docker_image"],
                "datasetDataSources": local["dataset_sources"], "competitionDataSources": [],
                "kernelDataSources": [], "modelDataSources": []}
    if (type(kernel_id) is not int or kernel_id <= 0 or not ref.startswith("clarkkitchen/")
            or any(metadata.get(k) != v for k, v in expected.items())
            or hashlib.sha256(source).hexdigest() != ctx["prep"]["runtime_sha256"]):
        raise ValueError("Canonical private source/image/data identity differs")


def require_sequence_slot(ctx: dict, api) -> None:
    root, stage = ctx["root"], ctx["stage"]
    if STAGES.index(stage):
        prior = STAGES[STAGES.index(stage) - 1]
        disposition = read(root / f"state/{SEQUENCE}/dispositions/{prior}.json")
        if (disposition.get("stage") != prior or disposition.get("plan_sha256") != sha(ctx["plan_path"])
                or disposition.get("status") not in {"completed_rejected", "completed_released", "terminal_failure"}):
            raise ValueError("Previous stage is not terminal and resolved")
        require_hash(checked(root, disposition["evidence_path"]), disposition["evidence_sha256"])
    reserved = ctx["cap"]
    for intent_path in (root / f"cloud/{SEQUENCE}").glob("*/*/push_intent.json"):
        intent = read(intent_path)
        reserved += intent["timeout_seconds"]
        identity_path = intent_path.parent / "provider_identity.json"
        if not identity_path.exists():
            raise RuntimeError("Unreconciled dispatch intent blocks the global project slot")
        identity = read(identity_path)
        if identity.get("push_intent_sha256") != sha(intent_path) or identity.get("status") != "verified":
            raise ValueError("Prior canonical identity lost its intent binding")
        state = str(api.kernels_status(identity["actual_kernel"]).status).rsplit(".", 1)[-1]
        if state not in TERMINAL:
            raise RuntimeError("Another sequence GPU job is nonterminal")
    if reserved > ctx["authority"]["max_gpu_seconds"]:
        raise ValueError("Sequence reserved provider caps exceed authorization")


def upload(ctx: dict, api) -> dict:
    folder, prep = ctx["folder"], ctx["prep"]
    intent = folder / "dataset_upload_intent.json"
    if intent.exists():
        raise FileExistsError("Existing upload intent; inspect dataset-status without retrying upload")
    require_absent(api, "dataset", prep["dataset_id"])
    upload_dir = folder / "bundle/upload"
    if {p.name for p in upload_dir.iterdir()} != {"payload.zip", "dataset-metadata.json"}:
        raise ValueError("Upload directory contains unregistered files")
    if read(upload_dir / "dataset-metadata.json").get("id") != prep["dataset_id"]:
        raise ValueError("Dataset upload identity differs")
    current_mutation_contract(ctx)
    write(intent, {"created_utc": stamp(), "dataset": prep["dataset_id"], "public": False,
                   "archive_sha256": prep["archive_sha256"], "plan_sha256": sha(ctx["plan_path"])})
    response = api.dataset_create_new(str(upload_dir), public=False, quiet=True, convert_to_csv=False, dir_mode="skip")
    receipt = response.to_dict(ignore_defaults=False)
    write(folder / "dataset_upload_receipt.json", receipt)
    return receipt


def reconcile(ctx: dict, api) -> dict:
    folder = ctx["folder"]
    intent = read(folder / "push_intent.json")
    if intent.get("source_sha256") != ctx["prep"]["runtime_sha256"]:
        raise ValueError("Dispatch intent source changed")
    identity_path = folder / "provider_identity.json"
    if identity_path.exists():
        identity = read(identity_path)
        expected = {"status": "verified", "push_intent_sha256": sha(folder / "push_intent.json"),
                    "plan_sha256": sha(ctx["plan_path"]), "registration_sha256": sha(ctx["registration_path"]),
                    "runtime_sha256": ctx["prep"]["runtime_sha256"], "requested_kernel": ctx["prep"]["kernel_id"],
                    "version": 1, "private_verified": True}
        if any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError("Canonical identity intent binding changed")
        require_hash(folder / "remote_metadata.json", identity["remote_metadata_sha256"])
        require_hash(folder / "remote_source_api.py", identity["remote_source_sha256"])
        if identity.get("push_receipt_sha256") is not None:
            require_hash(folder / "push_receipt.json", identity["push_receipt_sha256"])
        else:
            recovery_path = checked(ctx["root"], identity["recovery_receipt_path"])
            require_hash(recovery_path, identity["recovery_receipt_sha256"])
            recovery = read(recovery_path)
            if (recovery.get("status") != "recovered_readonly" or recovery.get("cloud_mutations_performed") != 0
                    or recovery.get("kernel_id") != identity["kernel_id"]
                    or recovery.get("actual_kernel") != identity["actual_kernel"]
                    or recovery.get("push_intent_sha256") != identity["push_intent_sha256"]):
                raise ValueError("Unknown-response reconciliation evidence differs")
        remote, source = remote_kernel(api, identity["actual_kernel"])
        validate_remote(ctx, identity["actual_kernel"], identity["kernel_id"], remote, source)
        return identity
    response_path = folder / "push_receipt.json"
    recovery_details = None
    if response_path.exists():
        response = read(response_path)
        if (response.get("error") or response.get("versionNumber") != 1
                or any(v for k, v in response.items() if k.startswith("invalid"))):
            raise ValueError("Rejected/ambiguous provider save requires an explicit failure disposition")
        ref, kernel_id = normalized_ref(response["ref"]), response["kernelId"]
        remote, source = remote_kernel(api, ref)
    else:
        # Recovery is read-only. A title/ref match is only a candidate; exact
        # private source/image/data/numeric-version readback remains mandatory.
        owned = own_resources(api, "kernel")
        candidates = [item for item in owned
                      if normalized_ref(item.ref) == intent["requested_kernel"]
                      or getattr(item, "title", None) == ctx["metadata"].get("title")]
        if len(candidates) != 1:
            raise RuntimeError("Unknown save outcome; cannot identify exactly one owned candidate")
        ref = normalized_ref(candidates[0].ref)
        remote, source = remote_kernel(api, ref)
        kernel_id = remote.get("id")
        recovery_details = {"owned_listing_exhausted": True, "owned_resources_observed": len(owned),
                            "matching_candidates": [{"ref": normalized_ref(x.ref), "title": getattr(x, "title", None)} for x in candidates]}
    validate_remote(ctx, ref, kernel_id, remote, source)
    # Reconciliation can resume after a local write interruption, without
    # overwriting the first saved readback or issuing another provider save.
    metadata_path, source_path = folder / "remote_metadata.json", folder / "remote_source_api.py"
    if metadata_path.exists():
        validate_remote(ctx, ref, kernel_id, read(metadata_path), source)
    else:
        write(metadata_path, remote)
    if source_path.exists():
        require_hash(source_path, ctx["prep"]["runtime_sha256"])
    else:
        with source_path.open("xb") as stream:
            stream.write(source)
    recovery_path = folder / "unknown_response_reconciliation.json"
    if recovery_details is not None:
        recovery = {"status": "recovered_readonly", "created_utc": stamp(), **recovery_details,
                    "actual_kernel": ref, "kernel_id": kernel_id, "version": 1,
                    "requested_kernel": intent["requested_kernel"], "cloud_mutations_performed": 0,
                    "push_intent_sha256": sha(folder / "push_intent.json"),
                    "remote_metadata_sha256": sha(metadata_path), "remote_source_sha256": sha(source_path),
                    "private_source_image_data_verified": True}
        if recovery_path.exists():
            saved = read(recovery_path)
            if any(saved.get(k) != recovery[k] for k in ("status", "actual_kernel", "kernel_id", "version", "push_intent_sha256", "remote_metadata_sha256", "remote_source_sha256", "cloud_mutations_performed")):
                raise ValueError("Preserved unknown-response reconciliation differs")
        else:
            write(recovery_path, recovery)
    identity = {"status": "verified", "verified_utc": stamp(), "requested_kernel": intent["requested_kernel"],
                "actual_kernel": ref, "kernel_id": kernel_id, "version": 1, "private_verified": True,
                "runtime_sha256": ctx["prep"]["runtime_sha256"], "plan_sha256": sha(ctx["plan_path"]),
                "registration_sha256": sha(ctx["registration_path"]),
                "push_intent_sha256": sha(folder / "push_intent.json"),
                "push_receipt_sha256": sha(response_path) if response_path.exists() else None,
                "recovery_receipt_path": recovery_path.relative_to(ctx["root"]).as_posix() if recovery_details is not None else None,
                "recovery_receipt_sha256": sha(recovery_path) if recovery_details is not None else None,
                "remote_metadata_sha256": sha(folder / "remote_metadata.json"),
                "remote_source_sha256": sha(folder / "remote_source_api.py")}
    write(identity_path, identity)
    return identity


def push(ctx: dict, api) -> dict:
    folder = ctx["folder"]
    if (folder / "push_intent.json").exists():
        raise FileExistsError("Existing dispatch intent; reconcile without another push")
    require_sequence_slot(ctx, api)
    require_absent(api, "kernel", ctx["prep"]["kernel_id"])
    dataset = dataset_readback(api, ctx["prep"]["dataset_id"])
    quota = quota_snapshot(api.quota_view())
    if (quota["remaining_seconds"] < ctx["cap"] or quota["reserved"]["total_seconds"] != 0
            or quota["paid_scaling_enabled"] is not False):
        raise ValueError("Require sufficient free quota, zero GPU reservation and disabled paid scaling")
    current_mutation_contract(ctx)
    write(folder / "dataset_at_dispatch.json", dataset)
    write(folder / "quota_at_dispatch.json", quota)
    write(folder / "push_intent.json", {"created_utc": stamp(), "requested_kernel": ctx["prep"]["kernel_id"],
          "stage": ctx["stage"], "phase": ctx["phase"], "timeout_seconds": ctx["cap"],
          "source_sha256": ctx["prep"]["runtime_sha256"], "metadata_sha256": sha(folder / "kernel-metadata.json"),
          "plan_sha256": sha(ctx["plan_path"]), "authority_sha256": sha(ctx["authority_path"]),
          "registration_sha256": sha(ctx["registration_path"])})
    response = api.kernels_push(str(folder), timeout=str(ctx["cap"]), acc="NvidiaTeslaT4")
    write(folder / "push_receipt.json", response.to_dict(ignore_defaults=False))
    return reconcile(ctx, api)


def status(ctx: dict, api) -> dict:
    identity = reconcile(ctx, api)
    response = api.kernels_status(identity["actual_kernel"])
    result = {"checked_utc": stamp(), "stage": ctx["stage"], "phase": ctx["phase"],
              "ref": identity["actual_kernel"], "kernel_id": identity["kernel_id"], "version": 1,
              "status": str(response.status).rsplit(".", 1)[-1], "failure_message": response.failure_message,
              "identity_sha256": sha(ctx["folder"] / "provider_identity.json")}
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    write(ctx["folder"] / "status" / name, result)
    return result


def validate_archive(returned: Path, cloud_status: dict, campaign_id: str) -> dict:
    helper = transport()
    archive, manifest_path = returned / "results.zip", returned / "output-manifest.json"
    binding = cloud_status["returned_artifacts"]
    require_hash(archive, binding["archive_sha256"])
    require_hash(manifest_path, binding["manifest_sha256"])
    manifest = read(manifest_path)
    if manifest.get("id") != campaign_id or not isinstance(manifest.get("files"), dict):
        raise ValueError("Returned manifest campaign differs")
    names = helper.inventory_names(manifest["files"])
    if binding["file_count"] != len(names) or any(not (n in {"registry.json", f"configs/{campaign_id}.json"}
            or any(n.startswith(f"{part}/{campaign_id}/") for part in ["artifacts", "state", "logs"])) for n in names):
        raise ValueError("Return inventory scope/count differs")
    for item in manifest["files"].values():
        helper.validate_item(item)
    if archive.stat().st_size > MAX_ARCHIVE or sum(x["bytes"] for x in manifest["files"].values()) > MAX_UNPACKED:
        raise ValueError("Returned archive exceeds bounded scale")
    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        if helper.inventory_names(i.filename for i in infos) != names | {"output-manifest.json"}:
            raise ValueError("ZIP members differ from exact manifest")
        for info in infos:
            if info.is_dir() or stat.S_IFMT(info.external_attr >> 16) not in {0, stat.S_IFREG} or info.flag_bits & 1:
                raise ValueError("Nonregular/encrypted ZIP member")
            expected = manifest["files"].get(info.filename, {"sha256": sha(manifest_path), "bytes": manifest_path.stat().st_size})
            if info.file_size != expected["bytes"] or info.file_size > helper.MAX_MEMBER_BYTES:
                raise ValueError("ZIP member size differs")
            with zipped.open(info) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                    raise ValueError("ZIP member bytes differ")
    return manifest


def completed_contract(ctx: dict, returned: Path, manifest: dict, remote_status: dict, terminal: str) -> bool:
    cid = ctx["id"]
    expected_status = "training_complete" if ctx["phase"] == "experiment" else "release_complete"
    if terminal != "COMPLETE" or remote_status.get("status") != expected_status:
        return False
    if remote_status.get("local_training") is not False:
        raise ValueError("Missing cloud-only execution statement")
    with zipfile.ZipFile(returned / "results.zip") as archive:
        path = f"artifacts/{cid}/completion_receipt.json"
        if manifest["files"][path]["bytes"] > MAX_JSON:
            raise ValueError("Completion receipt exceeds size bound")
        receipt = json.loads(archive.read(path))
        expected = {"id": cid, "status": "completed", "registry_sha256": ctx["prep"]["registry_sha256"]}
        if ctx["phase"] == "experiment":
            expected.update(completed_fit_count=12, completed_endpoint_count=9, completed_prefix_count=6,
                            completed_prefix_native_count=3, evaluation_ready=True)
            if remote_status.get("completed_fits") != 12 or remote_status.get("outer_metrics_computed") is not False:
                raise ValueError("Wrong experiment completion status")
            processes = [p for p in manifest["files"] if p.startswith(f"artifacts/{cid}/runtime_amendment/processes/") and p.endswith(".json")]
            if len(processes) != 14:
                raise ValueError("Missing fourteen deterministic startup receipts")
        else:
            expected.update(completed_fit_count=2, completed_prefix_count=1,
                            full_test_native_verified=True, release_ready=True)
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError("Completed operation contract differs")
    return True


def assemble(ctx: dict, returned: Path, manifest: dict, destination: Path) -> dict:
    helper = transport()
    payload = ctx["folder"] / "bundle/payload"
    require_hash(payload / "bundle-manifest.json", ctx["prep"]["manifest_sha256"])
    bundle = read(payload / "bundle-manifest.json")
    names = helper.inventory_names(bundle["files"])
    actual = helper.inventory_names(p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file() or p.is_symlink())
    if actual != names | {"bundle-manifest.json"} or bundle.get("id") != ctx["id"]:
        raise ValueError("Prepared payload inventory differs")
    overlaps = names & set(manifest["files"])
    if overlaps != {"registry.json", f"configs/{ctx['id']}.json"}:
        raise ValueError("Unexpected returned/payload overlap")
    for name, item in bundle["files"].items():
        helper.validate_item(item)
        source = helper.checked(payload, name)
        require_hash(source, item["sha256"])
        if source.stat().st_size != item["bytes"] or (name in overlaps and item != manifest["files"][name]):
            raise ValueError("Prepared/returned payload bytes differ")
    needed = sum(v["bytes"] for v in bundle["files"].values()) + sum(v["bytes"] for k, v in manifest["files"].items() if k not in overlaps)
    if shutil.disk_usage(destination.parent).free < needed + 1024**3:
        raise ValueError("Insufficient assembly disk reserve")
    destination.mkdir(exist_ok=False)
    for name in names | {"bundle-manifest.json"}:
        target = helper.checked(destination, name); target.parent.mkdir(parents=True, exist_ok=True)
        with helper.checked(payload, name).open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1024**2)
    with zipfile.ZipFile(returned / "results.zip") as archive:
        for name in set(manifest["files"]) | {"output-manifest.json"}:
            if name in overlaps:
                continue
            target = helper.checked(destination, name); target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024**2)
    for name, item in {**bundle["files"], **manifest["files"]}.items():
        target = helper.checked(destination, name)
        require_hash(target, item["sha256"])
        if target.stat().st_size != item["bytes"]:
            raise ValueError("Assembled artifact size differs")
    return {"status": "assembled_unscored", "workspace": str(destination), "returned_file_count": len(manifest["files"]),
            "registry_sha256": ctx["prep"]["registry_sha256"], "output_manifest_sha256": sha(returned / "output-manifest.json")}


def retrieve(ctx: dict, api, destination: Path, *, get=None) -> dict:
    helper = transport()
    current = status(ctx, api)
    if current["status"] not in TERMINAL:
        raise ValueError("Provider must be terminal before retrieval")
    if destination.exists():
        raise FileExistsError("Preserve previous retrieval; use fresh download directory")
    if not destination.resolve().is_relative_to(ctx["folder"].resolve()):
        raise ValueError("Download must stay inside registered operation folder")
    destination.mkdir(parents=True)
    write(destination / "retrieval_claim.json", {**current, "source_sha256": sha(Path(__file__)),
            "registration_sha256": sha(ctx["registration_path"]), "signed_urls_retained": False})
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    with api.build_kaggle_client() as client:
        def fetch(token):
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = current["ref"].split("/")
            request.page_size, request.page_token = 100, token
            return client.kernels.kernels_api_client.list_kernel_session_output(request)
        files, pages = helper.selected_pages(fetch, field="file_name")
    if get is None:
        import requests
        get = requests.get
    total, downloaded = 0, {}
    for name in sorted(files):
        value = helper.stream_http_length(files[name].url, helper.checked(destination, name),
                                         remaining_bytes=helper.MAX_TOTAL_BYTES - total, get=get)
        total += value["bytes"]; downloaded[name] = value
    files.clear()
    log = api.kernels_logs(current["ref"])
    if len(log.encode("utf-8")) > MAX_JSON:
        raise ValueError("Provider log exceeds bound")
    (destination / "kernel.log").write_text(log, encoding="utf-8")
    returned = destination / "fixed_epoch_return"
    cloud_status = read(returned / "cloud_status.json") if (returned / "cloud_status.json").exists() else {"id": ctx["id"], "status": "no_cloud_status"}
    if cloud_status.get("id") != ctx["id"]:
        raise ValueError("Returned operation identity differs")
    manifest = validate_archive(returned, cloud_status, ctx["id"]) if "returned_artifacts" in cloud_status else None
    if manifest is None and any((returned / name).exists() for name in ["results.zip", "output-manifest.json"]):
        raise ValueError("Returned files lack status hash binding")
    success = manifest is not None and completed_contract(ctx, returned, manifest, cloud_status, current["status"])
    result = {**current, "terminal_status": current["status"],
              "status": "retrieved_success_unscored" if success else "retrieved_terminal_failure",
              "cloud_status": cloud_status["status"], "archive_verified": manifest is not None,
              "files": downloaded, "output_pages": pages, "signed_urls_retained": False,
              "training_or_scoring_performed": False, "kernel_log_sha256": sha(destination / "kernel.log")}
    write(destination / "retrieval.json", result)
    if success:
        receipt = assemble(ctx, returned, manifest, ctx["folder"] / "assessment_workspace")
        write(destination / "assembly_receipt.json", receipt)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "upload", "dataset-status", "push", "reconcile", "status", "retrieve"])
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/continuation_six_v1.json")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--phase", choices=["experiment", "release"], default="experiment")
    parser.add_argument("--download-dir", type=Path)
    args = parser.parse_args()
    with operation_lock(ROOT):
        ctx = plan_context(ROOT, args.plan, args.stage, args.phase, mutation=args.action in {"upload", "push"})
        if args.action == "inspect":
            value = {"status": "registered_preparation_verified", "stage": args.stage, "phase": args.phase,
                     "plan_sha256": sha(ctx["plan_path"]), "registration_sha256": sha(ctx["registration_path"]),
                     "provider_cap_seconds": ctx["cap"], "cloud_operations_performed": False}
        else:
            from kaggle_cloud_control import api_client
            api = api_client()
            if args.action == "dataset-status":
                value = dataset_readback(api, ctx["prep"]["dataset_id"])
            elif args.action == "retrieve":
                destination = args.download_dir or ctx["folder"] / "download_01"
                value = retrieve(ctx, api, destination.resolve())
            else:
                value = {"upload": upload, "push": push, "reconcile": reconcile, "status": status}[args.action](ctx, api)
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
