"""Generated IO and mocked provider tests; no network, training or model imports."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import zipfile

import continuation_six_cloud_v1 as controller


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def quota(total=108000, used=7181.085, reserved=0, paid=False):
    return NS(gpu_quota=NS(total_time_allowed=timedelta(seconds=total), time_used=timedelta(seconds=used),
                          time_reserved=timedelta(seconds=reserved), is_pay_to_scale_enabled=paid))


def make_context(root, stage=None, phase="experiment"):
    stage = stage or controller.STAGES[0]
    plan_path = root / "configs/continuation_six_v1.json"
    plan = {"id": controller.SEQUENCE, "stages": [{"id": s, "index": i + 1} for i, s in enumerate(controller.STAGES)]}
    dump(plan_path, plan)
    now = datetime.now(timezone.utc)
    dump(root / "state/continuation_six_v1/authorization.json", {
        "status": "authorized", "plan_sha256": controller.sha(plan_path),
        "requested_utc": (now - timedelta(seconds=10)).isoformat(),
        "expires_utc": (now + timedelta(hours=23)).isoformat(), "max_gpu_seconds": 57600,
        "experiment_timeout_seconds": 7200, "release_timeout_seconds": 2400})
    folder = root / f"cloud/continuation_six_v1/{stage}/{phase}"
    folder.mkdir(parents=True)
    cid = stage + ("_release" if phase == "release" else "")
    source = b"print('generated fixture only')\n"
    (folder / "run.py").write_bytes(source)
    meta = {"id": "clarkkitchen/requested-fixture", "title": "Canonical Fixture Title", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "docker_image": "gcr.io/kaggle-gpu-images/python@sha256:fixture",
            "dataset_sources": ["clarkkitchen/data-fixture"], "competition_sources": [], "kernel_sources": [],
            "model_sources": [], "code_file": "run.py"}
    dump(folder / "kernel-metadata.json", meta)
    upload = folder / "bundle/upload"; upload.mkdir(parents=True)
    dump(upload / "dataset-metadata.json", {"id": "clarkkitchen/data-fixture"})
    (upload / "payload.zip").write_bytes(b"fixture upload")
    payload = folder / "bundle/payload"
    dump(payload / "registry.json", {"id": cid})
    dump(payload / f"configs/{cid}.json", {"id": cid})
    files = {p.relative_to(payload).as_posix(): {"sha256": controller.sha(p), "bytes": p.stat().st_size}
             for p in payload.rglob("*") if p.is_file()}
    dump(payload / "bundle-manifest.json", {"id": cid, "files": files})
    prep = {"id": cid, "phase": phase, "hard_timeout_seconds": 7200 if phase == "experiment" else 2400,
            "fit_budget_seconds": 6300, "private_required": True, "kernel_id": meta["id"],
            "dataset_id": "clarkkitchen/data-fixture", "registry_sha256": controller.sha(payload / "registry.json"),
            "manifest_sha256": controller.sha(payload / "bundle-manifest.json"),
            "archive_sha256": controller.sha(upload / "payload.zip"), "runtime_sha256": hashlib.sha256(source).hexdigest(),
            "frozen_files": {p.relative_to(root).as_posix(): controller.sha(p) for p in folder.rglob("*") if p.is_file()}}
    dump(folder / "preparation_manifest.json", prep)
    reg = {"status": "registered", "stage": stage, "phase": phase, "plan_sha256": controller.sha(plan_path),
           "preparation_path": (folder / "preparation_manifest.json").relative_to(root).as_posix(),
           "preparation_sha256": controller.sha(folder / "preparation_manifest.json")}
    if phase == "release":
        selected = root / f"state/continuation_six_v1/{stage}_selection.json"
        dump(selected, {"status": "frozen_for_release"})
        reg.update(release_selection_path=selected.relative_to(root).as_posix(), release_selection_sha256=controller.sha(selected))
    suffix = ".release" if phase == "release" else ""
    dump(root / f"state/continuation_six_v1/registrations/{stage}{suffix}.json", reg)
    return controller.plan_context(root, plan_path, stage, phase, mutation=True)


def remote(ctx, actual="clarkkitchen/canonical-fixture", private=True):
    meta = {"id": 101, "ref": actual, "currentVersionNumber": 1, "isPrivate": private,
            "enableGpu": True, "enableTpu": False, "enableInternet": True, "machineShape": "NvidiaTeslaT4",
            "dockerImage": ctx["metadata"]["docker_image"], "datasetDataSources": ctx["metadata"]["dataset_sources"],
            "competitionDataSources": [], "kernelDataSources": [], "modelDataSources": []}
    return meta, (ctx["folder"] / "run.py").read_bytes()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ctx = make_context(self.root)
        self.api = NS(quota_view=Mock(return_value=quota()),
                      kernels_list_with_response=Mock(return_value=NS(kernels=[], next_page_token="")),
                      dataset_list_with_response=Mock(return_value=NS(datasets=[], next_page_token="")),
                      kernels_push=Mock(return_value=NS(to_dict=lambda **_: {"ref": "/code/clarkkitchen/canonical-fixture",
                                           "kernelId": 101, "versionNumber": 1, "error": ""})),
                      kernels_status=Mock(return_value=NS(status="COMPLETE", failure_message="")))
        self.dataset_patch = patch.object(controller, "dataset_readback", return_value={"private": True})
        self.dataset_patch.start(); self.addCleanup(self.dataset_patch.stop)

    def test_registration_and_no_ml_import_side_effect(self):
        self.assertEqual(self.ctx["cap"], 7200)
        self.assertEqual(self.ctx["prep"]["id"], controller.STAGES[0])
        result = subprocess.run([sys.executable, "-c",
            "import sys; import continuation_six_cloud_v1; assert not any(x in sys.modules for x in ['torch','pandas','numpy','kaggle'])"],
            cwd=Path(controller.__file__).parent, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_quota_preserves_days_and_fractional_seconds(self):
        result = controller.quota_snapshot(quota())
        self.assertEqual(result["total"]["days"], 1)
        self.assertEqual(result["total"]["seconds"], 21600)
        self.assertEqual(result["total"]["total_seconds"], 108000)
        self.assertAlmostEqual(result["remaining_seconds"], 100818.915)

    def test_changed_source_rejected_before_api(self):
        (self.ctx["folder"] / "run.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Registered bytes"):
            controller.plan_context(self.root, self.ctx["plan_path"], self.ctx["stage"], "experiment")

    def test_changed_plan_rejected(self):
        plan = controller.read(self.ctx["plan_path"]); plan["extra"] = "changed"; dump(self.ctx["plan_path"], plan)
        with self.assertRaisesRegex(ValueError, "Registered bytes"):
            controller.plan_context(self.root, self.ctx["plan_path"], self.ctx["stage"], "experiment")

    def test_expired_mutation_rejected_readonly_allowed(self):
        future = datetime.now(timezone.utc) + timedelta(days=2)
        with self.assertRaisesRegex(ValueError, "wall window"):
            controller.plan_context(self.root, self.ctx["plan_path"], self.ctx["stage"], "experiment", mutation=True, now=future)
        controller.plan_context(self.root, self.ctx["plan_path"], self.ctx["stage"], "experiment", now=future)

    def test_canonical_identity_success_no_second_push(self):
        with patch.object(controller, "remote_kernel", return_value=remote(self.ctx)):
            result = controller.push(self.ctx, self.api)
            self.assertEqual(result["actual_kernel"], "clarkkitchen/canonical-fixture")
            self.assertEqual(result["requested_kernel"], "clarkkitchen/requested-fixture")
            controller.reconcile(self.ctx, self.api)
            with self.assertRaises(FileExistsError):
                controller.push(self.ctx, self.api)
        self.api.kernels_push.assert_called_once()

    def test_unknown_outcome_keeps_intent_and_blocks_retry(self):
        self.api.kernels_push.side_effect = TimeoutError("mocked timeout")
        with self.assertRaises(TimeoutError):
            controller.push(self.ctx, self.api)
        self.assertTrue((self.ctx["folder"] / "push_intent.json").exists())
        with self.assertRaises(FileExistsError):
            controller.push(self.ctx, self.api)
        self.api.kernels_push.assert_called_once()

    def test_unknown_outcome_reconciles_only_unique_exact_source(self):
        self.api.kernels_push.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            controller.push(self.ctx, self.api)
        item = NS(ref="clarkkitchen/canonical-fixture", title="Canonical Fixture Title")
        self.api.kernels_list_with_response.return_value = NS(kernels=[item], next_page_token="")
        with patch.object(controller, "remote_kernel", return_value=remote(self.ctx)):
            result = controller.reconcile(self.ctx, self.api)
        self.assertIsNone(result["push_receipt_sha256"])
        recovery = self.root / result["recovery_receipt_path"]
        self.assertEqual(controller.sha(recovery), result["recovery_receipt_sha256"])
        self.assertEqual(controller.read(recovery)["cloud_mutations_performed"], 0)
        self.assertEqual(result["kernel_id"], 101)
        self.api.kernels_push.assert_called_once()

    def test_private_image_or_source_mismatch_rejected(self):
        metadata, source = remote(self.ctx)
        for key, value in [("isPrivate", False), ("currentVersionNumber", 2), ("dockerImage", "changed")]:
            corrupt = dict(metadata); corrupt[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                controller.validate_remote(self.ctx, metadata["ref"], 101, corrupt, source)
        with self.assertRaises(ValueError):
            controller.validate_remote(self.ctx, metadata["ref"], 101, metadata, source + b"changed")

    def test_insufficient_reserved_or_paid_quota_rejected(self):
        for q in [quota(total=7000, used=0), quota(reserved=1), quota(paid=True)]:
            self.api.quota_view.return_value = q
            with self.subTest(quota=q), self.assertRaises(ValueError):
                controller.push(self.ctx, self.api)
        self.api.kernels_push.assert_not_called()
        self.assertFalse((self.ctx["folder"] / "push_intent.json").exists())

    def test_prior_unknown_intent_blocks_global_slot(self):
        path = self.root / "cloud/continuation_six_v1/other/experiment/push_intent.json"
        dump(path, {"timeout_seconds": 7200})
        with self.assertRaisesRegex(RuntimeError, "Unreconciled"):
            controller.require_sequence_slot(self.ctx, self.api)

    def test_prior_running_job_blocks_global_slot(self):
        path = self.root / "cloud/continuation_six_v1/other/experiment/push_intent.json"
        dump(path, {"timeout_seconds": 7200})
        dump(path.parent / "provider_identity.json", {"actual_kernel": "clarkkitchen/other", "status": "verified",
                                                     "push_intent_sha256": controller.sha(path)})
        self.api.kernels_status.return_value = NS(status="RUNNING")
        with self.assertRaisesRegex(RuntimeError, "nonterminal"):
            controller.require_sequence_slot(self.ctx, self.api)

    def test_stage_order_requires_bound_disposition(self):
        ctx = make_context(self.root, controller.STAGES[1])
        with self.assertRaises(FileNotFoundError):
            controller.require_sequence_slot(ctx, self.api)
        evidence = self.root / "state/terminal_failure.json"; dump(evidence, {"status": "failed"})
        dump(self.root / f"state/continuation_six_v1/dispositions/{controller.STAGES[0]}.json",
             {"stage": controller.STAGES[0], "plan_sha256": controller.sha(ctx["plan_path"]),
              "status": "terminal_failure", "evidence_path": evidence.relative_to(self.root).as_posix(),
              "evidence_sha256": controller.sha(evidence)})
        controller.require_sequence_slot(ctx, self.api)

    def test_release_requires_frozen_selection(self):
        ctx = make_context(self.root, phase="release")
        self.assertEqual(ctx["cap"], 2400)
        reg = controller.read(ctx["registration_path"])
        dump(self.root / reg["release_selection_path"], {"status": "unfrozen"})
        with self.assertRaises(ValueError):
            controller.plan_context(self.root, ctx["plan_path"], ctx["stage"], "release")

    def test_preflight_latency_cannot_cross_dispatch_deadline(self):
        self.ctx["deadline"] = datetime.now(timezone.utc) + timedelta(seconds=60)
        with self.assertRaisesRegex(ValueError, "wall-clock"):
            controller.push(self.ctx, self.api)
        self.api.kernels_push.assert_not_called()

    def test_readback_resumes_after_partial_local_receipt_write(self):
        self.api.kernels_push.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            controller.push(self.ctx, self.api)
        metadata, source = remote(self.ctx)
        dump(self.ctx["folder"] / "remote_metadata.json", metadata)
        (self.ctx["folder"] / "remote_source_api.py").write_bytes(source)
        self.api.kernels_list_with_response.return_value = NS(kernels=[NS(ref=metadata["ref"], title=self.ctx["metadata"]["title"])], next_page_token="")
        with patch.object(controller, "remote_kernel", return_value=(metadata, source)):
            self.assertEqual(controller.reconcile(self.ctx, self.api)["status"], "verified")
        self.api.kernels_push.assert_called_once()

    def test_changed_identity_readback_file_rejected(self):
        with patch.object(controller, "remote_kernel", return_value=remote(self.ctx)):
            controller.push(self.ctx, self.api)
            (self.ctx["folder"] / "remote_source_api.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Registered bytes"):
                controller.reconcile(self.ctx, self.api)

    def test_process_lock_excludes_overlap(self):
        with controller.operation_lock(self.root):
            with self.assertRaises((RuntimeError, BlockingIOError, OSError)):
                with controller.operation_lock(self.root):
                    self.fail("overlapping lock acquired")

    def archive_fixture(self, *, count=12):
        cid = self.ctx["id"]; returned = self.root / "returned"; returned.mkdir()
        contents = {"registry.json": (self.ctx["folder"] / "bundle/payload/registry.json").read_bytes(),
                    f"configs/{cid}.json": (self.ctx["folder"] / f"bundle/payload/configs/{cid}.json").read_bytes()}
        completion = {"id": cid, "status": "completed", "registry_sha256": self.ctx["prep"]["registry_sha256"],
                      "completed_fit_count": count, "completed_endpoint_count": 9, "completed_prefix_count": 6,
                      "completed_prefix_native_count": 3, "evaluation_ready": True}
        contents[f"artifacts/{cid}/completion_receipt.json"] = json.dumps(completion).encode()
        for i in range(14):
            contents[f"artifacts/{cid}/runtime_amendment/processes/process_{i}.json"] = b"{}"
        manifest = {"id": cid, "files": {k: {"sha256": hashlib.sha256(v).hexdigest(), "bytes": len(v)} for k, v in contents.items()}}
        dump(returned / "output-manifest.json", manifest)
        with zipfile.ZipFile(returned / "results.zip", "w") as archive:
            for name, value in contents.items(): archive.writestr(name, value)
            archive.writestr("output-manifest.json", (returned / "output-manifest.json").read_bytes())
        status = {"id": cid, "status": "training_complete", "completed_fits": 12, "local_training": False,
                  "outer_metrics_computed": False, "returned_artifacts": {
                      "archive_sha256": controller.sha(returned / "results.zip"),
                      "manifest_sha256": controller.sha(returned / "output-manifest.json"), "file_count": len(contents)}}
        return returned, status, manifest

    def test_generated_archive_validates_and_assembles_without_scoring(self):
        returned, status, manifest = self.archive_fixture()
        actual = controller.validate_archive(returned, status, self.ctx["id"])
        self.assertEqual(actual, manifest)
        self.assertTrue(controller.completed_contract(self.ctx, returned, manifest, status, "COMPLETE"))
        result = controller.assemble(self.ctx, returned, manifest, self.root / "assessment")
        self.assertEqual(result["status"], "assembled_unscored")
        with self.assertRaises(FileExistsError):
            controller.assemble(self.ctx, returned, manifest, self.root / "assessment")

    def test_generated_archive_wrong_completion_count_rejected(self):
        returned, status, manifest = self.archive_fixture(count=11)
        with self.assertRaisesRegex(ValueError, "contract differs"):
            controller.completed_contract(self.ctx, returned, manifest, status, "COMPLETE")

    def test_corrupt_archive_rejected(self):
        returned, status, _ = self.archive_fixture()
        with (returned / "results.zip").open("ab") as stream: stream.write(b"corruption")
        with self.assertRaises(ValueError):
            controller.validate_archive(returned, status, self.ctx["id"])

    def test_wrong_campaign_return_rejected(self):
        returned, status, _ = self.archive_fixture()
        with self.assertRaisesRegex(ValueError, "campaign"):
            controller.validate_archive(returned, status, "wrong_campaign")

    def test_unlisted_archive_member_rejected_even_with_bound_archive(self):
        returned, status, _ = self.archive_fixture()
        with zipfile.ZipFile(returned / "results.zip", "a") as archive:
            archive.writestr("unexpected.txt", b"not allowed")
        status["returned_artifacts"]["archive_sha256"] = controller.sha(returned / "results.zip")
        with self.assertRaisesRegex(ValueError, "exact manifest"):
            controller.validate_archive(returned, status, self.ctx["id"])

    def test_upload_intent_prevents_duplicate_after_timeout(self):
        self.api.dataset_create_new = Mock(side_effect=TimeoutError())
        with self.assertRaises(TimeoutError):
            controller.upload(self.ctx, self.api)
        with self.assertRaises(FileExistsError):
            controller.upload(self.ctx, self.api)
        self.api.dataset_create_new.assert_called_once()


if __name__ == "__main__":
    unittest.main()
