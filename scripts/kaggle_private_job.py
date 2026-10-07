"""Dispatch and inspect reviewed, uniquely named private Kaggle control jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kaggle_cloud_control import ROOT, api_client, write

LGB_PROTOCOL_SHA = '80aa76eb30630dfad0512173cbf508cfa6ec16303f19de9a010d5a09cbfec6a9'
LONG_PROTOCOL_SHA = '73338bad3caf5241da5d3b520cece06459af3d9470500cbbb03ae641e6653266'
SHRINKAGE_PROTOCOL_SHA = 'ea5ce787253b52eabb8353bf5584aafbc3f736e139f88ecdc332e8e9da3d7399'


def shrinkage_protocol():
    path = ROOT / 'configs/third_pass_lgb_shrinkage.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHRINKAGE_PROTOCOL_SHA:
        raise ValueError('Registered shrinkage cloud protocol changed')
    return json.loads(path.read_text())


def long_protocol():
    path = ROOT / 'configs/third_pass_long_cloud.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != LONG_PROTOCOL_SHA:
        raise ValueError('Registered long cloud protocol changed')
    return json.loads(path.read_text())


def lgb_protocol():
    path = ROOT / 'configs/third_pass_lgb_cloud.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != LGB_PROTOCOL_SHA:
        raise ValueError('Registered LightGBM cloud protocol changed')
    return json.loads(path.read_text())


def dispatch_deadline(folder: Path, meta: dict, timeout: int, production: bool):
    if not production:
        if folder.resolve() == (ROOT / 'cloud/third_pass_lgb_shrinkage').resolve():
            protocol = shrinkage_protocol()
            prepared = json.loads((folder / 'preparation_manifest.json').read_text())
            if (timeout != 1800 or meta.get('enable_gpu') is not False
                    or meta.get('docker_image') != protocol['execution_image']
                    or meta.get('competition_sources') != []
                    or prepared.get('protocol_sha256') != SHRINKAGE_PROTOCOL_SHA
                    or prepared.get('runtime_sha256') != hashlib.sha256((folder / meta['code_file']).read_bytes()).hexdigest()):
                raise ValueError('Shrinkage CPU CV dispatch contract differs')
            return datetime(2026, 10, 2, 16, 20, tzinfo=timezone.utc)
        if folder.resolve() == (ROOT / 'cloud/third_pass_long').resolve():
            protocol = long_protocol()
            prepared = json.loads((folder / 'preparation_manifest.json').read_text())
            if (not 60 <= timeout <= 4500 or meta.get('enable_gpu') is not True
                    or meta.get('docker_image') != protocol['execution_image']
                    or meta.get('competition_sources') != []
                    or prepared.get('runtime_sha256') != hashlib.sha256((folder / meta['code_file']).read_bytes()).hexdigest()
                    or hashlib.sha256((folder / 'bundle/payload/protocol.json').read_bytes()).hexdigest() != LONG_PROTOCOL_SHA):
                raise ValueError('Long GPU CV dispatch contract differs')
            return datetime(2026, 10, 2, 16, 30, tzinfo=timezone.utc)
        if folder.resolve() == (ROOT / 'cloud/third_pass_lgb').resolve():
            protocol = lgb_protocol()
            prepared = json.loads((folder / 'preparation_manifest.json').read_text())
            if (timeout != 1200 or meta.get('enable_gpu') is not False
                    or meta.get('docker_image') != protocol['execution_image']
                    or meta.get('competition_sources') != []
                    or prepared.get('protocol_sha256') != LGB_PROTOCOL_SHA
                    or prepared.get('runtime_sha256') != hashlib.sha256((folder / meta['code_file']).read_bytes()).hexdigest()):
                raise ValueError('LightGBM CPU CV dispatch contract differs')
            return datetime(2026, 10, 2, 15, 30, tzinfo=timezone.utc)
        return datetime(2026, 10, 2, 16, tzinfo=timezone.utc)
    # Only a prepared, positively selected third-pass cloud production member
    # may use the reserved release hour. Preserve25minutes for return/verification.
    if not meta['id'].startswith('clarkkitchen/s6e10-cloud-production-'):
        raise ValueError('Production cutoff requires the dedicated production namespace')
    prepared = json.loads((folder / 'preparation_manifest.json').read_text())
    spec = json.loads((folder / 'payload/refit_spec.json').read_text())
    freeze_path = Path(prepared['freeze_file']).resolve()
    import re
    if (freeze_path.name != 'frozen.json' or freeze_path.parent.name != 'blend'
            or re.fullmatch(r'third_pass(?:_batch[0-9]{2})?', freeze_path.parent.parent.name) is None
            or freeze_path.parent.parent.parent != ROOT / 'artifacts'):
        raise ValueError('Production selection must be an isolated third-pass frozen release')
    digest = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    freeze = json.loads(freeze_path.read_text())
    weight = freeze.get('weights', {}).get(spec['run_id'], 0)
    import math
    lgb_cpu = (spec.get('protocol_id') == 'lgb_rating_probability_v1'
               or str(spec.get('run_id', '')).startswith('v3_cloud_lgb_')
               or folder.resolve().is_relative_to((ROOT / 'cloud/third_pass_lgb').resolve()))
    long_gpu = (spec.get('run_id') == 'v3_cloud_realmlp_raw_aux_e60'
                or folder.resolve().is_relative_to((ROOT / 'cloud/third_pass_long').resolve()))
    shrinkage_cpu = (spec.get('protocol_id') == 'lgb_shrinkage_v1'
                     or spec.get('run_id') == 'v3_cloud_lgb_route_teacher_aux_probability_lr01'
                     or folder.resolve().is_relative_to((ROOT / 'cloud/third_pass_lgb_shrinkage').resolve()))
    expected_timeout = 900 if shrinkage_cpu else 600 if lgb_cpu else 750 if long_gpu else 1500
    if (not math.isfinite(weight) or weight <= 0 or timeout != expected_timeout
            or spec.get('format') != 'third_pass_cloud_refit'
            or spec.get('selection_frozen_sha256') != digest
            or prepared.get('frozen_selection_hash') != digest
            or freeze.get('source_result_hashes', {}).get(spec['run_id']) != spec.get('cv_result_sha256')
            or prepared.get('runtime_sha256') != hashlib.sha256((folder / meta['code_file']).read_bytes()).hexdigest()
            or meta.get('competition_sources') != ['playground-series-s6e10']):
        raise ValueError('Production authority/configuration/source contract differs')
    if shrinkage_cpu:
        protocol = shrinkage_protocol()
        if (spec.get('protocol_id') != protocol['protocol_id']
                or meta.get('enable_gpu') is not False
                or spec.get('run_id') != protocol['candidate_id']
                or spec.get('timeout_seconds') != 900
                or spec.get('protocol_sha256') != SHRINKAGE_PROTOCOL_SHA
                or prepared.get('protocol_sha256') != SHRINKAGE_PROTOCOL_SHA
                or spec.get('execution_image') != protocol['execution_image']
                or meta.get('docker_image') != protocol['execution_image']):
            raise ValueError('Shrinkage CPU production dispatch contract differs')
        return datetime(2026, 10, 2, 16, 40, tzinfo=timezone.utc)
    if lgb_cpu:
        protocol = lgb_protocol()
        if (spec.get('protocol_id') != 'lgb_rating_probability_v1'
                or meta.get('enable_gpu') is not False
                or spec.get('run_id') != protocol['candidate_id']
                or spec.get('timeout_seconds') != 600
                or spec.get('protocol_sha256') != LGB_PROTOCOL_SHA
                or prepared.get('protocol_sha256') != LGB_PROTOCOL_SHA
                or spec.get('execution_image') != protocol['execution_image']
                or meta.get('docker_image') != protocol['execution_image']):
            raise ValueError('LightGBM CPU production dispatch contract differs')
        return datetime(2026, 10, 2, 16, tzinfo=timezone.utc)
    if long_gpu:
        protocol = long_protocol()
        if (spec.get('run_id') != protocol['candidate_id']
                or meta.get('enable_gpu') is not True
                or meta.get('docker_image') != protocol['execution_image']
                or spec.get('execution_image') != protocol['execution_image']
                or spec.get('timeout_seconds') != 750
                or datetime.now(timezone.utc) > datetime(2026, 10, 2, 16, 30, tzinfo=timezone.utc)):
            raise ValueError('Long GPU production dispatch contract differs')
        return datetime(2026, 10, 2, 16, 42, 30, tzinfo=timezone.utc)
    return datetime(2026, 10, 2, 16, 35, tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["push", "status", "output"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--production", action="store_true", help="Use the reserved release window only for a verified frozen production package")
    args = parser.parse_args()
    folder = args.folder.resolve()
    if ROOT.resolve() not in folder.parents:
        raise ValueError("Control package must be inside this project")
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    if not meta["id"].startswith("clarkkitchen/s6e10-") or meta.get("is_private") is not True:
        raise ValueError("Require explicitly private project-owned kernel")
    now = datetime.now(timezone.utc)
    api = api_client()
    if args.action == "push":
        if (folder / "push_intent.json").exists():
            raise RuntimeError("Existing dispatch intent: inspect remote state instead of retrying")
        deadline = dispatch_deadline(folder, meta, args.timeout, args.production)
        maximum = 4500 if folder == (ROOT / 'cloud/third_pass_long').resolve() and not args.production else 3600
        if not 60 <= args.timeout <= maximum or now.timestamp() + args.timeout > deadline.timestamp():
            raise ValueError("Invalid timeout or job extends beyond exploration cutoff")
        source = (folder / meta["code_file"]).resolve()
        if source.parent != folder:
            raise ValueError("Source must be a direct child of reviewed package")
        code = source.read_bytes()
        if b"KGAT_" in code or b"kaggle api key.txt" in code:
            raise ValueError("Credential marker detected in upload source")
        if meta.get("enable_gpu"):
            quota = api.quota_view().gpu_quota
            remaining = (quota.total_time_allowed-quota.time_used-quota.time_reserved).total_seconds()
            if remaining < args.timeout:
                raise ValueError("Insufficient unreserved GPU allowance for bounded control")
        write(folder / "push_intent.json", {"utc": now.isoformat(), "metadata": meta,
              "timeout_seconds": args.timeout, "source_sha256": hashlib.sha256(code).hexdigest()})
        response = api.kernels_push(str(folder), timeout=str(args.timeout), acc=meta.get("machine_shape"))
        receipt = response.to_dict(ignore_defaults=False)
        write(folder / "push_receipt.json", receipt)
        print(json.dumps(receipt, default=str))
        invalid = [k for k,v in receipt.items() if k.startswith("invalid") and v]
        if response.error or invalid or not response.version_number:
            raise RuntimeError(f"Rejected save or source attachments: {invalid}; {response.error}")
        return
    receipt = json.loads((folder / "push_receipt.json").read_text())
    # One push per unique slug. Explicit version_label returned404 from this account's
    # API despite its presence in the SDK, so use latest only under this strict contract.
    owner, slug = meta["id"].split("/")
    version = str(receipt["versionNumber"])
    if version != "1":
        raise RuntimeError("Readback requires an immutable one-version control slug")
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    with api.build_kaggle_client() as client:
        status = api.kernels_status(meta["id"])
        record = {"utc": now.isoformat(), "ref": meta["id"], "version": version,
                  "status": str(status.status), "failure_message": status.failure_message}
        write(folder / "latest_status.json", record)
        print(json.dumps(record))
        if args.action == "status":
            return
        if not str(status.status).endswith((".COMPLETE", ".ERROR", ".CANCELLED")):
            raise RuntimeError("Wait for terminal state before downloading outputs")
        get_request = ApiGetKernelRequest()
        get_request.user_name, get_request.kernel_slug = owner, slug
        remote = client.kernels.kernels_api_client.get_kernel(get_request)
        if remote.metadata.is_private is not True:
            raise RuntimeError("Remote privacy verification failed")
        write(folder / "remote_metadata.json", remote.metadata.to_dict(ignore_defaults=False))
        (folder / "kernel.log").write_text(api.kernels_logs(meta["id"]), encoding="utf-8")
    # This unique slug has a single immutable version; high-level downloader is safe here.
    output = folder / "output"
    output.mkdir(exist_ok=True)
    _, next_page = api.kernels_output(meta["id"], str(output), force=True, quiet=True)
    if next_page:
        raise RuntimeError("Output pagination requires explicit follow-up")
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
