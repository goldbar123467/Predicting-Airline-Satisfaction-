"""Submit only an incrementally qualifying, fully verified six-stage release.

No fitting, model imports or score computation. Inspect is entirely local.
Submit/reconcile call Kaggle only when explicitly requested. Every upload gets
one durable intent; an ambiguous outcome is reconciled read-only, never retried.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "continuation_six_v1"
COMPETITION = "playground-series-s6e10"
STAGES = tuple(f"fixed_epoch_continuation_{i:02d}" for i in range(1, 7))
AUTHORITY_SHA = "dc8e2502db03ce4e18fe7b656e0098cfd757ad159157898757ad6c53700df3fe"
INCUMBENT_SHA = "bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311"
INCUMBENT_SUBMISSION_SHA = "177eca4563e41b91c30c12d494284ea24d68951b6ada332edf49dc10646ccbc9"
ROWS = 299844
MAX_JSON = 32 * 1024**2


def utc():
    return datetime.now(timezone.utc)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Require a portable relative path")
    target = root / name
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes workspace")
    current = root
    for part in name.split("/"):
        current /= part
        if current.is_symlink():
            raise ValueError("Symlink in immutable release")
    return target


def read(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON:
        raise ValueError("Metadata exceeds bounded size")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Metadata must be an object")
    return value


def write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_hash(path: Path, expected: str):
    if not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None or sha(path) != expected:
        raise ValueError(f"Immutable release bytes changed: {path.name}")


def metric(value: dict) -> tuple[float, list[float]]:
    pooled, folds = value.get("oof_auc"), value.get("fold_auc")
    if (type(pooled) not in {float, int} or not math.isfinite(pooled) or not 0 <= pooled <= 1
            or not isinstance(folds, list) or len(folds) != 3
            or any(type(x) not in {float, int} or not math.isfinite(x) or not 0 <= x <= 1 for x in folds)):
        raise ValueError("Expected finite pooled and three ordered development-fold AUCs")
    return float(pooled), list(map(float, folds))


def gain_gate(candidate: dict, previous: dict) -> dict:
    pooled, folds = metric(candidate)
    old_pooled, old_folds = metric(previous)
    differences = [left-right for left, right in zip(folds, old_folds, strict=True)]
    mean = sum(differences)/3
    return {"pooled_gain": pooled-old_pooled, "macro_gain": mean, "fold_gains": differences,
            "passed": pooled-old_pooled >= 1e-5 and mean >= 1e-5 and min(differences) >= -2e-5}


def csv_contract(path: Path, expected_ids_sha256: str, *, expected_rows: int = ROWS) -> dict:
    count, ids, hasher = 0, set(), hashlib.sha256()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["id", "satisfaction"]:
            raise ValueError("Submission schema differs")
        for row in reader:
            if len(row) != 2 or re.fullmatch(r"0|[1-9][0-9]*", row[0]) is None:
                raise ValueError("Malformed submission row/ID")
            identifier = int(row[0])
            probability = float(row[1])
            if identifier in ids or identifier > 2**63-1 or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Duplicate ID or invalid probability")
            ids.add(identifier)
            hasher.update(struct.pack("<q", identifier))
            count += 1
            if count > expected_rows:
                raise ValueError("Excess submission rows")
    if count != expected_rows or hasher.hexdigest() != expected_ids_sha256:
        raise ValueError("Submission count/ordered ID hash differs")
    return {"rows": count, "ids_sha256": hasher.hexdigest(), "schema": ["id", "satisfaction"]}


def authority_context(root: Path, *, now=None, mutation=False) -> dict:
    now = now or utc()
    authority_path = root / f"state/{SEQUENCE}/authorization.json"
    require_hash(authority_path, AUTHORITY_SHA)
    authority = read(authority_path)
    plan_path = root / f"configs/{SEQUENCE}.json"
    require_hash(plan_path, authority["plan_sha256"])
    plan = read(plan_path)
    start, end = (datetime.fromisoformat(authority[key]) for key in ("requested_utc", "expires_utc"))
    if (authority.get("id") != SEQUENCE or authority.get("status") != "authorized"
            or authority.get("maximum_new_submissions") != 6 or start.tzinfo is None or end.tzinfo is None
            or not 0 < (end-start).total_seconds() <= 86400 or plan.get("id") != SEQUENCE
            or tuple(item.get("id") for item in plan.get("stages", [])) != STAGES
            or plan.get("budget", {}).get("maximum_new_submissions") != 6):
        raise ValueError("Wrong sequence authorization/plan")
    if mutation and not start <= now < end:
        raise ValueError("Submission window is closed")
    return {"root":root,"plan":plan,"authority":authority,"start":start,"end":end,
            "bindings":{authority_path:AUTHORITY_SHA,plan_path:authority["plan_sha256"]}}


def artifact_contract(ctx: dict, stage: str) -> dict:
    root, plan = ctx["root"], ctx["plan"]
    if stage not in STAGES:
        raise ValueError("Unregistered stage")
    folder = root / f"artifacts/{stage}/release"
    selection_path = root / f"artifacts/{stage}/release_selection.json"
    paths = {"submission_sha256":folder/"submission.csv", "verification_sha256":folder/"verification.json",
             "manifest_sha256":folder/"manifest.json", "selection_sha256":selection_path}
    provenance_path = folder / "release_provenance.json"
    provenance = read(provenance_path)
    bindings = {provenance_path:sha(provenance_path)}
    for key, path in paths.items():
        require_hash(path, provenance.get(key))
        bindings[path] = provenance[key]
    selection, verification = read(selection_path), read(paths["verification_sha256"])
    manifest = read(paths["manifest_sha256"])
    common = {"stage":stage,"id":stage+"_release","fixed_alpha":.1,
              "submission_sha256":provenance["submission_sha256"],"selection_sha256":provenance["selection_sha256"]}
    if (any(manifest.get(key) != value for key,value in common.items())
            or provenance.get("status") != "verified" or provenance.get("stage") != stage
            or provenance.get("fixed_alpha") != .1
            or manifest.get("registry_sha256") != verification.get("registry_sha256")):
        raise ValueError("Final manifest/provenance identity differs")
    candidate_path = checked(root,manifest["candidate_verification_path"])
    expected_candidate = root/f"cloud/{SEQUENCE}/{stage}/release/assessment_workspace/artifacts/{stage}_release/verification.json"
    if candidate_path.resolve() != expected_candidate.resolve():
        raise ValueError("Candidate verification is outside the isolated release workspace")
    require_hash(candidate_path,manifest["candidate_verification_sha256"])
    bindings[candidate_path] = manifest["candidate_verification_sha256"]
    registry_path = root/f"cloud/{SEQUENCE}/{stage}/release/assessment_workspace/registry.json"
    require_hash(registry_path,manifest["registry_sha256"])
    bindings[registry_path] = manifest["registry_sha256"]
    expected = {"status":"verified","stage":stage,"rows":ROWS,"sha256":provenance["submission_sha256"],
                "all_row_raw_inference":True,"independent_blend_recomputation":True,"audit_evaluated":False,
                "class_order":[0,1],"keyed_schema_verified":True,"native_reload_verified":True,
                "fixed_alpha":.1,"selection_sha256":provenance["selection_sha256"]}
    if any(verification.get(key) != value for key,value in expected.items()):
        raise ValueError("Missing complete native/raw/schema/blend verification")
    stage_plan = plan["stages"][STAGES.index(stage)]
    if (selection.get("status") != "frozen_for_release" or selection.get("stage") != stage
            or selection.get("plan_sha256") != ctx["authority"]["plan_sha256"]
            or selection.get("fixed_alpha") != .1 or selection.get("gain_gate_passed") is not True
            or selection.get("policy") != stage_plan["policy"] or selection.get("audit_evaluated") is not False
            or selection.get("original_incumbent_submission_sha256") != INCUMBENT_SUBMISSION_SHA):
        raise ValueError("Frozen selection contract differs")
    metric(selection)
    for field in ("evaluation", "protocol"):
        path = checked(root, selection[field+"_path"])
        require_hash(path, selection[field+"_sha256"])
        bindings[path] = selection[field+"_sha256"]
    for kind in ("source_hashes", "native_hashes"):
        mapping = provenance.get(kind)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Missing source/native inventory")
        if verification.get(kind) != mapping:
            raise ValueError("Verification/provenance source/native inventories disagree")
        if manifest.get(kind) != mapping:
            raise ValueError("Manifest source/native inventory differs")
        for name, digest in mapping.items():
            path = checked(root, name)
            require_hash(path, digest)
            bindings[path] = digest
    if "scripts/submit_continuation_six_v1.py" not in provenance["source_hashes"]:
        raise ValueError("Submitter source must be bound in final verification")
    if verification.get("ids_sha256") != verification.get("expected_test_ids_sha256"):
        raise ValueError("Verified IDs differ from frozen test order")
    csv_contract(paths["submission_sha256"], verification.get("expected_test_ids_sha256"), expected_rows=ROWS)
    return {"stage":stage,"selection":selection,"selection_path":selection_path,
            "submission_path":paths["submission_sha256"],"sha256":provenance["submission_sha256"],
            "submission_bytes":paths["submission_sha256"].stat().st_size,"bindings":bindings}


def verification_contract(root: Path, stage: str, *, now=None, mutation=False) -> dict:
    ctx = authority_context(root, now=now, mutation=mutation)
    current = artifact_contract(ctx, stage)
    ctx["bindings"].update(current["bindings"])
    ctx.update({key:value for key,value in current.items() if key != "bindings"})
    baseline = checked(root, ctx["plan"]["incumbent_selection_path"])
    require_hash(baseline, INCUMBENT_SHA)
    if ctx["plan"].get("incumbent_selection_sha256") != INCUMBENT_SHA:
        raise ValueError("Original incumbent pin differs")
    ctx["bindings"][baseline] = INCUMBENT_SHA
    original = read(baseline)
    selection = current["selection"]
    if (selection.get("original_incumbent_selection_path") != baseline.relative_to(root).as_posix()
            or selection.get("original_incumbent_selection_sha256") != INCUMBENT_SHA):
        raise ValueError("Frozen selection is not anchored to the original incumbent")
    best, best_path, best_sha = original, baseline, INCUMBENT_SHA
    for prior_stage in STAGES[:STAGES.index(stage)]:
        if (root/f"artifacts/{prior_stage}/release/verification.json").exists():
            prior = artifact_contract(ctx, prior_stage)
            ctx["bindings"].update(prior["bindings"])
            if metric(prior["selection"])[0] > metric(best)[0]:
                best, best_path = prior["selection"], prior["selection_path"]
                best_sha = sha(best_path)
    if (selection.get("best_prior_selection_path") != best_path.relative_to(root).as_posix()
            or selection.get("best_prior_selection_sha256") != best_sha):
        raise ValueError("Frozen selection no longer names the best prior verified release")
    gates = {"original":gain_gate(selection,original),"best_prior":gain_gate(selection,best)}
    if not all(value["passed"] for value in gates.values()):
        raise ValueError("No qualifying incremental development gain")
    ctx["gates"] = gates
    return ctx


def collect_submissions(fetch) -> tuple[list[dict], list[dict]]:
    token, seen_tokens, seen_refs, result, pages = None, set(), set(), [], []
    for index in range(100):
        response = fetch(token)
        submissions = response.submissions or []
        for item in submissions:
            if item is None or type(item.ref) is not int or item.ref <= 0 or item.ref in seen_refs:
                raise ValueError("Invalid/duplicate submission identity across pages")
            seen_refs.add(item.ref)
            result.append({"ref":item.ref,"description":item.description or "","file_name":item.file_name or "",
                "total_bytes":int(item.total_bytes),"status":str(item.status).rsplit(".",1)[-1],
                "public_score":item.public_score or "","date":item.date.isoformat() if item.date else None})
        token = response.next_page_token
        pages.append({"index":index,"count":len(submissions),"has_next_page":bool(token)})
        if not token:
            return result,pages
        if token in seen_tokens:
            raise ValueError("Repeated submission page token")
        seen_tokens.add(token)
    raise ValueError("Submission pagination exceeded bound")


def submission_history(api):
    from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
    with api.build_kaggle_client() as client:
        def fetch(token):
            request = ApiListSubmissionsRequest()
            request.competition_name,request.page_size,request.page_token = COMPETITION,100,token
            # The SDK defaults group=ALL and sort=DATE; retain response token.
            return client.competitions.competition_api_client.list_submissions(request)
        return collect_submissions(fetch)


def intent_paths(root: Path) -> list[Path]:
    paths = sorted((root/f"state/{SEQUENCE}/submissions").glob("*/intent.json"))
    if len(paths) > 6 or any(path.parent.name not in STAGES for path in paths):
        raise ValueError("Submission intent count/stage scope invalid")
    return paths


def validate_intent(path: Path, ctx: dict) -> dict:
    value = read(path)
    if (value.get("competition") != COMPETITION or value.get("stage") != path.parent.name
            or value.get("authority_sha256") != AUTHORITY_SHA
            or value.get("plan_sha256") != ctx["authority"]["plan_sha256"]
            or value.get("status") != "intent_recorded" or not isinstance(value.get("sha256"),str)
            or re.fullmatch(r"[a-f0-9]{64}",value["sha256"]) is None
            or value.get("description") != f"cont6 {value['stage']} sha256:{value['sha256']}"):
        raise ValueError("Durable submission intent changed")
    return value


def reconcile_intent(path: Path, ctx: dict, history: list[dict], pages: list[dict]) -> dict:
    intent = validate_intent(path,ctx)
    candidates = [row for row in history if row["description"] == intent["description"]]
    if len(candidates) != 1:
        raise RuntimeError("Submission outcome unknown; preserve intent and do not retry")
    matched = candidates[0]
    if matched["file_name"] != intent["file_name"] or matched["total_bytes"] != intent["submission_bytes"]:
        raise ValueError("Remote submission size/name differs from intent")
    response_path = path.parent/"response.json"
    if response_path.exists() and read(response_path).get("ref") not in {None,0,matched["ref"]}:
        raise ValueError("Remote history disagrees with upload response identity")
    value = {"status":"reconciled_remote","checked_utc":utc().isoformat(),"intent_sha256":sha(path),
             "submission":matched,"pages":pages,"complete_owned_history":True,"cloud_mutations_performed":0}
    name = utc().strftime("%Y%m%dT%H%M%S%fZ")+".json"
    write(path.parent/"reconciliations"/name,value)
    return value


def verify_bindings(ctx: dict, *, now=None):
    now = now or utc()
    if not ctx["start"] <= now < ctx["end"]:
        raise ValueError("Submission window is closed")
    for path,digest in ctx["bindings"].items():
        require_hash(path,digest)


def submit(ctx: dict, api, *, history_loader=submission_history, now=utc) -> dict:
    history,pages = history_loader(api)
    existing = intent_paths(ctx["root"])
    same = None
    for path in existing:
        value = validate_intent(path,ctx)
        reconciled = reconcile_intent(path,ctx,history,pages)
        if value["stage"] == ctx["stage"] or value["sha256"] == ctx["sha256"]:
            same = reconciled
    if same is not None:
        return same
    if len(existing) >= 6:
        raise RuntimeError("All six authorized submission attempts are consumed")
    if any(ctx["sha256"] in row["description"] for row in history):
        raise RuntimeError("This exact hash is already remote without this intent; reconcile manually")
    verify_bindings(ctx,now=now())
    limits = api.competition_get_submission_limits(COMPETITION)
    checked_at = now()
    if type(limits.num_allowed_now) is not int or limits.num_allowed_now <= 0:
        raise RuntimeError("Live competition daily allowance is exhausted or unavailable")
    limit_record = {"checked_utc":checked_at.isoformat(),"num_allowed_now":limits.num_allowed_now,
                    "num_today":limits.num_today,"num_total":limits.num_total,
                    "limited_by_total":limits.limited_by_total}
    verify_bindings(ctx,now=now())
    if (now()-checked_at).total_seconds() > 60:
        raise RuntimeError("Live allowance became stale during immutable verification")
    folder = ctx["root"]/f"state/{SEQUENCE}/submissions/{ctx['stage']}"
    description = f"cont6 {ctx['stage']} sha256:{ctx['sha256']}"
    intent = {"status":"intent_recorded","requested_utc":now().isoformat(),"stage":ctx["stage"],
        "competition":COMPETITION,"sha256":ctx["sha256"],"file_name":ctx["submission_path"].name,
        "submission_bytes":ctx["submission_bytes"],"file":ctx["submission_path"].relative_to(ctx["root"]).as_posix(),
        "selection_path":ctx["selection_path"].relative_to(ctx["root"]).as_posix(),
        "selection_sha256":sha(ctx["selection_path"]),"description":description,
        "authority_sha256":AUTHORITY_SHA,"plan_sha256":ctx["authority"]["plan_sha256"],
        "source_sha256":sha(Path(__file__)),"live_limits":limit_record,"preflight_history_pages":pages,
        "gates":ctx["gates"]}
    write(folder/"intent.json",intent)
    # Any exception from the provider leaves the intent unresolved. Never print
    # provider exception text, which can contain signed upload credentials.
    try:
        response = api.competition_submit(str(ctx["submission_path"]),description,COMPETITION,quiet=True)
        response_record = {"received_utc":now().isoformat(),"intent_sha256":sha(folder/"intent.json"),
                           "ref":getattr(response,"ref",None)}
        write(folder/"response.json",response_record)
    except Exception:
        raise RuntimeError("Submission outcome unknown; retain intent and reconcile read-only") from None
    history,pages = history_loader(api)
    return reconcile_intent(folder/"intent.json",ctx,history,pages)


@contextmanager
def submission_lock(root: Path):
    path = root/"state/kaggle_submission.lock"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0,2)
        if stream.tell() == 0:
            stream.write(b"0");stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["inspect","submit","reconcile"])
    parser.add_argument("--stage",choices=STAGES,required=True)
    args = parser.parse_args()
    with submission_lock(ROOT):
        if args.action == "reconcile":
            ctx = authority_context(ROOT)
            from kaggle_cloud_control import api_client
            history,pages = submission_history(api_client())
            result = reconcile_intent(ROOT/f"state/{SEQUENCE}/submissions/{args.stage}/intent.json",ctx,history,pages)
        else:
            ctx = verification_contract(ROOT,args.stage,mutation=args.action=="submit")
            if args.action == "inspect":
                result = {"status":"verified_unsubmitted","stage":args.stage,"sha256":ctx["sha256"],
                          "gates":ctx["gates"],"network_calls":0}
            else:
                from kaggle_cloud_control import api_client
                result = submit(ctx,api_client())
        print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
