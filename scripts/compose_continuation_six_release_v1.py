"""Verify a cloud refit and independently compose its fixed incumbent mixture."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from continuation_six_cloud_v1 import ROOT, checked, read, require_hash, sha, write, plan_context, validate_archive, completed_contract, validate_remote
from continuation_six_selection_v1 import best_prior, gain_gate, sequence_context

FULL_REFIT_INPUT_BINDING_SHA = "c1e58af45d2874677dde1e0f0c0861a929c98c741512150e6b09f7e367af0b2c"


def checked_vector(frame, ids, column):
    import numpy as np
    if list(frame.columns) != ["id", column] or not frame.id.is_unique or len(frame) != len(ids) or not np.array_equal(frame.id.to_numpy(), ids):
        raise ValueError("Prediction schema, identifiers or order differ")
    value = frame[column].to_numpy(dtype=float)
    if not np.isfinite(value).all() or not ((value >= 0) & (value <= 1)).all():
        raise ValueError("Invalid probabilities")
    return value


def compose(root: Path, stage: str):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    plan, _ = sequence_context(root, stage)
    selection_path = root / f"artifacts/{stage}/release_selection.json"
    selection = read(selection_path)
    prior_path, prior = best_prior(root, stage)
    if (selection["status"] != "frozen_for_release" or selection["stage"] != stage
            or selection["plan_sha256"] != sha(root / "configs/continuation_six_v1.json")
            or selection["fixed_alpha"] != .1 or not gain_gate(selection, prior)
            or selection["best_prior_selection_sha256"] != sha(prior_path)):
        raise ValueError("Frozen release no longer qualifies")
    for name in ("evaluation", "protocol", "original_incumbent_selection"):
        require_hash(checked(root, selection[name + "_path"]), selection[name + "_sha256"])
    binding_path = root / "state/continuation_six_v1/incumbent_release_binding.json"
    binding = read(binding_path)
    for name, digest in binding["files"].items():
        require_hash(checked(root, name), digest)
    incumbent_provenance = read(root / "artifacts/third_pass/final/release_provenance.json")
    incumbent_verification = read(root / "artifacts/third_pass/verification.json")
    if (incumbent_verification["rows"] != 299844 or not incumbent_verification["all_row_raw_inference"]
            or not incumbent_verification["independent_blend_recomputation"] or incumbent_verification["audit_evaluated"]
            or incumbent_verification["sha256"] != selection["original_incumbent_submission_sha256"]):
        raise ValueError("Incumbent native verification chain differs")
    maps = {field: dict(incumbent_provenance[field]) for field in ("native_hashes", "source_hashes")}
    for hashes in maps.values():
        for name, digest in hashes.items():
            require_hash(checked(root, name), digest)
    workspace = root / f"cloud/continuation_six_v1/{stage}/release/assessment_workspace"
    cid = stage + "_release"
    ctx = plan_context(root, root / "configs/continuation_six_v1.json", stage, "release", mutation=False)
    download = ctx["folder"] / "download_01"
    retrieval = read(download / "retrieval.json")
    identity_path = ctx["folder"] / "provider_identity.json"
    identity = read(identity_path)
    require_hash(identity_path, retrieval["identity_sha256"])
    if (retrieval["status"] != "retrieved_success_unscored" or retrieval["version"] != 1
            or retrieval["ref"] != identity["actual_kernel"] or retrieval["kernel_id"] != identity["kernel_id"]
            or identity["status"] != "verified" or identity["private_verified"] is not True):
        raise ValueError("Private release retrieval identity differs")
    require_hash(ctx["folder"] / "push_intent.json", identity["push_intent_sha256"])
    require_hash(ctx["registration_path"], identity["registration_sha256"])
    require_hash(ctx["folder"] / "remote_metadata.json", identity["remote_metadata_sha256"])
    require_hash(ctx["folder"] / "remote_source_api.py", identity["remote_source_sha256"])
    if identity["runtime_sha256"] != ctx["prep"]["runtime_sha256"]:
        raise ValueError("Returned provider source binding differs")
    validate_remote(ctx, identity["actual_kernel"], identity["kernel_id"],
                    read(ctx["folder"] / "remote_metadata.json"), (ctx["folder"] / "remote_source_api.py").read_bytes())
    for name, item in retrieval["files"].items():
        require_hash(checked(download, name), item["sha256"])
    returned = download / "fixed_epoch_return"
    cloud_status = read(returned / "cloud_status.json")
    manifest = validate_archive(returned, cloud_status, cid)
    if not completed_contract(ctx, returned, manifest, cloud_status, "COMPLETE"):
        raise ValueError("Preserved full-data cloud completion failed")
    for name, item in manifest["files"].items():
        path = checked(workspace, name)
        require_hash(path, item["sha256"])
        if path.stat().st_size != item["bytes"]:
            raise ValueError("Assembled return size differs")
    payload = ctx["folder"] / "bundle/payload"
    bundle = read(payload / "bundle-manifest.json")
    for name, item in bundle["files"].items():
        require_hash(checked(workspace, name), item["sha256"])
    require_hash(workspace / "registry.json", ctx["prep"]["registry_sha256"])
    candidate_out = workspace / f"artifacts/{cid}"
    candidate_path = candidate_out / "verification.json"
    candidate = read(candidate_path)
    registry_path = workspace / "registry.json"
    registry = read(registry_path)
    require_hash(selection_path, registry["selection_sha256"])
    required = {"status": "passed", "id": cid, "stage": stage, "rows": 299844, "train_rows": 699635,
                "class_order": [0, 1], "full_test_native_verified": True,
                "full_test_raw_inference": True, "audit_evaluated": False,
                "selection_sha256": sha(selection_path), "registry_sha256": sha(registry_path),
                "keyed_schema_verified": True, "native_reload_verified": True,
                "heldout_quality_metrics_computed": False, "training_probe_metrics_recorded": True,
                "atol": 2e-6, "rtol": 1e-5, "chunks": [32768, 8191]}
    if any(candidate.get(key) != value for key, value in required.items()):
        raise ValueError("Full-test candidate native verification failed")
    prediction_path = checked(workspace, candidate["test_predictions_path"])
    require_hash(prediction_path, candidate["test_predictions_sha256"])
    completion = read(candidate_out / "completion_receipt.json")
    expected_completion = {"id": cid, "status": "completed", "completed_fit_count": 2,
                           "completed_prefix_count": 1, "completed_startup_count": 3,
                           "full_test_native_verified": True, "release_ready": True,
                           "verification_sha256": sha(candidate_path), "selection_sha256": sha(selection_path),
                           "registry_sha256": sha(registry_path)}
    if any(completion.get(key) != value for key, value in expected_completion.items()):
        raise ValueError("Paired full-data release gates differ")
    prefix_path = candidate_out / "trajectory_C/prefix_match.json"
    require_hash(prefix_path, candidate["prefix_match_sha256"])
    prefix = read(prefix_path)
    expected_components = {"network", "learned_static_preprocessing", "gradients", "optimizer", "cpu_rng", "cuda_rng",
                           "numpy_rng", "python_rng", "progress", "schedule", "sampler", "batch_order", "schema",
                           "input_train", "probe", "modes", "dropout_scopes", "parameter_groups", "input_monitor"}
    if (prefix.get("status") != "passed" or prefix.get("epoch") != 4 or prefix.get("gate_completed_before_epoch5") is not True
            or set(prefix["components_equal"]) != expected_components or any(v is not True for v in prefix["components_equal"].values())
            or prefix.get("native_probabilities_exact") is not True or prefix.get("update_schedule_prefix_exact") is not True
            or prefix.get("observation_live_state_unchanged") is not True):
        raise ValueError("Full-data training-state prefix gate failed")
    for kind in ("control", "treatment"):
        require_hash(checked(workspace, prefix[kind + "_prefix_state_path"]), prefix[kind + "_prefix_state_sha256"])
    canonical_binding = root / "state/continuation_six_v1/full_refit_input_binding.json"
    require_hash(canonical_binding, FULL_REFIT_INPUT_BINDING_SHA)
    for name, digest in read(canonical_binding)["files"].items():
        require_hash(checked(root, name), digest)
    for field, name in (("raw_test_sha256", "data/test.csv"), ("sample_submission_sha256", "data/sample_submission.csv")):
        if candidate[field] != sha(root / name):
            raise ValueError("Candidate raw test/sample binding differs")
    if not np.isfinite(candidate["full_test_max_abs_error"]) or not 0 <= candidate["full_test_max_abs_error"] <= 1.2e-5:
        raise ValueError("Candidate native/raw parity exceeds tolerance")
    processes = list((candidate_out / "runtime_amendment/processes").glob("*.json"))
    roles = {read(path)["role"] for path in processes}
    if len(processes) != 3 or roles != {"release_controller", "release_worker_A", "release_worker_C"}:
        raise ValueError("Missing release deterministic startup roles")
    policy_path = workspace / "provenance/execution_policy.json"
    policy = read(policy_path)
    strict_flags = {"deterministic_algorithms": True, "deterministic_warn_only": False,
                    "cudnn_benchmark": False, "cudnn_deterministic": True, "cuda_matmul_allow_tf32": False,
                    "cublas_workspace_config": ":4096:8", "startup_cuda_initialized": False}
    for process in processes:
        record = read(process)
        if (record["status"] != "passed" or record["flags"] != strict_flags
                or record["registry_sha256"] != sha(registry_path)
                or record["execution_policy_sha256"] != sha(policy_path)
                or record["source_sha256"] != policy["source_sha256"]):
            raise ValueError("Strict release startup evidence differs")
        compatibility_path = candidate_out / f"runtime_compatibility/{record['role']}.json"
        compatibility = read(compatibility_path)
        if (compatibility.get("status") != "passed" or compatibility.get("before_full_fit") is not True
                or compatibility.get("role") != record["role"] or compatibility.get("registry_sha256") != sha(registry_path)
                or compatibility.get("versions") != registry["expected_versions"]
                or set(compatibility["installed_sources"]) != set(registry["installed_runtime_source_sha256"])
                or any(item["sha256"] != registry["installed_runtime_source_sha256"][name]
                       for name, item in compatibility["installed_sources"].items())):
            raise ValueError("Actual full-refit installed runtime differs from assessed experiment")
    for field in maps:
        if not candidate.get(field):
            raise ValueError("Missing candidate source/native inventory")
        for name, digest in candidate[field].items():
            path = checked(workspace, name)
            require_hash(path, digest)
            maps[field][path.relative_to(root).as_posix()] = digest
    with threadpool_limits(limits=2):
        sample = pd.read_csv(root / "data/sample_submission.csv")
        if list(sample.columns) != ["id", "satisfaction"] or len(sample) != 299844 or not sample.id.is_unique:
            raise ValueError("Canonical sample schema differs")
        ids = sample.id.to_numpy()
        incumbent = checked_vector(pd.read_csv(root / "artifacts/third_pass/final/submission.csv"), ids, "satisfaction")
        vector = checked_vector(pd.read_parquet(prediction_path), ids, "prediction")
        ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
        if candidate["expected_test_ids_sha256"] != ids_sha:
            raise ValueError("Candidate prediction ID digest differs from canonical test order")
        mixed = .9 * incumbent + .1 * vector
        output = root / f"artifacts/{stage}/release"
        output.mkdir(exist_ok=False)
        submitted = output / "submission.csv"
        pd.DataFrame({"id": ids, "satisfaction": mixed}).to_csv(submitted, index=False)
        reloaded = checked_vector(pd.read_csv(submitted), ids, "satisfaction")
        independent = np.sum(np.stack([incumbent * .9, vector * .1]), axis=0)
        if not np.allclose(reloaded, independent, rtol=0, atol=2e-15):
            raise ValueError("Independent mixture recomputation failed")
    for name in ("compose_continuation_six_release_v1.py", "continuation_six_selection_v1.py",
                 "continuation_six_policy_v1.py", "continuation_six_cloud_v1.py",
                 "continuation_six_release_v1.py", "submit_continuation_six_v1.py"):
        maps["source_hashes"]["scripts/" + name] = sha(root / "scripts" / name)
    manifest = {"id": cid, "stage": stage, "fixed_alpha": .1, "submission_sha256": sha(submitted),
                "selection_sha256": sha(selection_path), "registry_sha256": sha(registry_path),
                "candidate_verification_path": candidate_path.relative_to(root).as_posix(),
                "candidate_verification_sha256": sha(candidate_path), "expected_test_ids_sha256": ids_sha,
                "incumbent_binding_sha256": sha(binding_path), **maps}
    write(output / "manifest.json", manifest)
    verification = {"status": "verified", "stage": stage, "id": cid, "rows": len(ids),
                    "sha256": sha(submitted), "all_row_raw_inference": True,
                    "independent_blend_recomputation": True, "audit_evaluated": False, "class_order": [0, 1],
                    "selection_sha256": sha(selection_path), "registry_sha256": sha(registry_path),
                    "keyed_schema_verified": True, "native_reload_verified": True, "fixed_alpha": .1,
                    "expected_test_ids_sha256": ids_sha, "ids_sha256": ids_sha,
                    "parent_native_verification_reused": True,
                    "candidate_native_verification_execution_location": "Kaggle, checked locally",
                    "verified_utc": datetime.now(timezone.utc).isoformat(), **maps}
    write(output / "verification.json", verification)
    provenance = {"status": "verified", "stage": stage, "id": cid,
                  "submission_sha256": sha(submitted), "verification_sha256": sha(output / "verification.json"),
                  "manifest_sha256": sha(output / "manifest.json"), "selection_sha256": sha(selection_path),
                  "expected_test_ids_sha256": ids_sha, "keyed_schema_verified": True,
                  "native_reload_verified": True, "fixed_alpha": .1,
                  "incumbent_binding_sha256": sha(binding_path), **maps}
    write(output / "release_provenance.json", provenance)
    return verification


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    print(json.dumps(compose(ROOT, args.stage), indent=2))


if __name__ == "__main__":
    main()
