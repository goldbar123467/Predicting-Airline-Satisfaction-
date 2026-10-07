"""Freeze a once-assessed qualifying fixed mixture before any full-data fit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

from continuation_six_cloud_v1 import ROOT, STAGES, checked, read, require_hash, sha, write
from continuation_six_policy_v1 import policy_for_stage


def gain_gate(candidate: dict, previous: dict) -> bool:
    values = [candidate["oof_auc"], previous["oof_auc"], *candidate["fold_auc"], *previous["fold_auc"]]
    if len(candidate["fold_auc"]) != 3 or len(previous["fold_auc"]) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid three-fold selection metrics")
    delta = [a - b for a, b in zip(candidate["fold_auc"], previous["fold_auc"], strict=True)]
    return candidate["oof_auc"] - previous["oof_auc"] >= 1e-5 and sum(delta) / 3 >= 1e-5 and min(delta) >= -2e-5


def sequence_context(root: Path, stage: str, *, mutation: bool = True) -> tuple[dict, dict]:
    if stage not in STAGES:
        raise ValueError("Unregistered stage")
    plan_path = root / "configs/continuation_six_v1.json"
    authority = read(root / "state/continuation_six_v1/authorization.json")
    require_hash(plan_path, authority["plan_sha256"])
    if authority["status"] != "authorized" or (mutation and not datetime.fromisoformat(authority["requested_utc"]) <= datetime.now(timezone.utc) < datetime.fromisoformat(authority["expires_utc"])):
        raise ValueError("Sequence authority is not active")
    plan = read(plan_path)
    if plan["stages"][STAGES.index(stage)]["policy"] != policy_for_stage(stage):
        raise ValueError("Ordered scientific policy changed")
    return plan, authority


def best_prior(root: Path, stage: str) -> tuple[Path, dict]:
    baseline_path = root / "artifacts/third_pass/blend/frozen.json"
    plan = read(root / "configs/continuation_six_v1.json")
    require_hash(baseline_path, plan["incumbent_selection_sha256"])
    best_path, best = baseline_path, read(baseline_path)
    for prior in STAGES[:STAGES.index(stage)]:
        disposition = read(root / f"state/continuation_six_v1/dispositions/{prior}.json")
        require_hash(checked(root, disposition["evidence_path"]), disposition["evidence_sha256"])
        if disposition.get("stage") != prior or disposition.get("plan_sha256") != sha(root / "configs/continuation_six_v1.json"):
            raise ValueError("Prior disposition binding changed")
        if disposition["status"] == "completed_released":
            path = checked(root, disposition["selection_path"])
            require_hash(path, disposition["selection_sha256"])
            selection = read(path)
            if selection["status"] != "frozen_for_release" or selection["stage"] != prior:
                raise ValueError("Prior release selection differs")
            if selection["oof_auc"] > best["oof_auc"]:
                best_path, best = path, selection
        elif disposition["status"] not in {"completed_rejected", "terminal_failure"}:
            raise ValueError("Previous stage remains unresolved")
    return best_path, best


def validate_assessment(root: Path, stage: str, evaluation: dict, protocol: dict, protocol_path: Path, claim_path: Path) -> None:
    required = {"id": stage, "status": "frozen_before_assessment", "quality_metrics_read_when_frozen": False,
                "real_predictions_scored_when_frozen": False,
                "sequence_plan_sha256": sha(root / "configs/continuation_six_v1.json"),
                "sequence_authorization_sha256": sha(root / "state/continuation_six_v1/authorization.json")}
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError("Assessment protocol lacks preregistered stage/authority")
    for name, digest in protocol["source_sha256"].items():
        require_hash(checked(root, name), digest)
    for field, name in (("script_sha256", "evaluate_continuation_six_v1.py"),
                        ("test_script_sha256", "test_evaluate_continuation_six_v1.py")):
        if evaluation[field] != protocol["source_sha256"]["scripts/" + name]:
            raise ValueError("Assessment used a different frozen evaluator source")
    claim = read(claim_path)
    if claim != evaluation["claim"]:
        raise ValueError("Once-only assessment claim differs from saved result")
    for field in ("campaign_sha256", "registry_sha256", "manifest_sha256", "local_protocol_sha256"):
        if claim[field] != evaluation["provenance"][field]:
            raise ValueError("Assessment claim and provenance disagree")
    workspace = root / f"cloud/continuation_six_v1/{stage}/experiment/assessment_workspace"
    if claim["registry_sha256"] != protocol["registry_sha256"]:
        raise ValueError("Assessment used another preregistered registry")
    for field, path in (("campaign_sha256", workspace / f"configs/{stage}.json"),
                        ("registry_sha256", workspace / "registry.json"),
                        ("manifest_sha256", workspace / f"artifacts/{stage}/completed_manifest.json")):
        require_hash(path, claim[field])
    if claim["local_protocol_sha256"] != sha(protocol_path):
        raise ValueError("Assessment claim binds another protocol")
    times = [datetime.fromisoformat(value) for value in (protocol["created_utc"], claim["claimed_utc"], evaluation["completed_utc"])]
    if any(value.tzinfo is None for value in times) or not times[0] <= times[1] <= times[2]:
        raise ValueError("Assessment was not claimed after protocol freeze")
    for name, digest in protocol["provider_identity"]["files"].items():
        require_hash(checked(root, name), digest)


def freeze(root: Path, stage: str) -> dict:
    plan, authority = sequence_context(root, stage)
    workspace = root / f"cloud/continuation_six_v1/{stage}/experiment/assessment_workspace"
    evaluation_path = workspace / f"artifacts/{stage}/evaluation.json"
    protocol_path = root / f"artifacts/{stage}/local_evaluation_protocol.json"
    evaluation, protocol = read(evaluation_path), read(protocol_path)
    validate_assessment(root, stage, evaluation, protocol, protocol_path,
                        workspace / f"artifacts/{stage}/evaluation_claim.json")
    evaluator = root / "scripts/evaluate_continuation_six_v1.py"
    require_hash(evaluator, evaluation["script_sha256"])
    if (evaluation.get("id") != stage or evaluation.get("row_count") != 629671
            or evaluation.get("continuation_policy") != policy_for_stage(stage)
            or evaluation.get("mixture_alpha") != .1 or evaluation.get("weight_search_performed") is not False
            or evaluation.get("raw_train_audit_test_read") is not False
            or evaluation.get("release_or_incumbent_modified") is not False
            or evaluation.get("class_order") != [0, 1]
            or evaluation["provenance"]["local_protocol_sha256"] != sha(protocol_path)):
        raise ValueError("Once-assessed scientific result differs")
    if not evaluation["advancement"]["advance_to_confirmation"]:
        return {"status": "not_qualified", "stage": stage, "reason": "Frozen advancement gate failed"}
    metric = evaluation["metrics"]["mixture_C"]
    selection_metrics = {"oof_auc": metric["pooled_auc"], "fold_auc": metric["fold_auc"]}
    for key in ("incumbent", "mixture_A", "mixture_B"):
        reference = evaluation["metrics"][key]
        if not gain_gate(selection_metrics, {"oof_auc": reference["pooled_auc"], "fold_auc": reference["fold_auc"]}):
            return {"status": "not_qualified", "stage": stage, "reason": "Saved fixed comparison gate failed"}
    original_path = root / plan["incumbent_selection_path"]
    require_hash(original_path, plan["incumbent_selection_sha256"])
    prior_path, previous = best_prior(root, stage)
    if not gain_gate(selection_metrics, read(original_path)) or not gain_gate(selection_metrics, previous):
        return {"status": "not_qualified", "stage": stage, "reason": "No qualifying incremental development gain"}
    result = {"status": "frozen_for_release", "stage": stage, "created_utc": datetime.now(timezone.utc).isoformat(),
              "plan_sha256": sha(root / "configs/continuation_six_v1.json"),
              "authority_sha256": sha(root / "state/continuation_six_v1/authorization.json"),
              "evaluation_path": evaluation_path.relative_to(root).as_posix(), "evaluation_sha256": sha(evaluation_path),
              "evaluation_claim_path": (workspace / f"artifacts/{stage}/evaluation_claim.json").relative_to(root).as_posix(),
              "evaluation_claim_sha256": sha(workspace / f"artifacts/{stage}/evaluation_claim.json"),
              "protocol_path": protocol_path.relative_to(root).as_posix(), "protocol_sha256": sha(protocol_path),
              "original_incumbent_selection_path": original_path.relative_to(root).as_posix(),
              "original_incumbent_selection_sha256": sha(original_path),
              "original_incumbent_submission_sha256": "177eca4563e41b91c30c12d494284ea24d68951b6ada332edf49dc10646ccbc9",
              "best_prior_selection_path": prior_path.relative_to(root).as_posix(), "best_prior_selection_sha256": sha(prior_path),
              "gain_gate_passed": True, "fixed_alpha": .1, "policy": policy_for_stage(stage),
              **selection_metrics, "audit_evaluated": False, "weight_search_performed": False,
              "script_sha256": sha(Path(__file__))}
    write(root / f"artifacts/{stage}/release_selection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(freeze(ROOT, args.stage), indent=2))


if __name__ == "__main__":
    main()
