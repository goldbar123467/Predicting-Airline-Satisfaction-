"""Generated receipt-only cloud reporter checks; no ML library imports or scoring."""
from __future__ import annotations

import ast
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_fixed_epoch_cloud_v1 as reporter
from test_summarize_fixed_epoch_v1 import generated_fixture, write


def cloud_fixture(base: Path) -> tuple[Path, Path, str, str, dict]:
    workspace, canonical = base / "workspace", base / "canonical"
    campaign = reporter.CAMPAIGN_ID
    generated_fixture(workspace, campaign=campaign)
    output = workspace / "artifacts" / campaign
    registry = json.loads((output / "registry.json").read_text())
    (workspace / "registry.json").write_bytes((output / "registry.json").read_bytes())
    (output / "registry.json").unlink()
    config_path = workspace / "configs" / f"{campaign}.json"
    config = json.loads(config_path.read_text())
    config.update(registry_path="registry.json", split_sha256="a" * 64)
    write(config_path, config)
    binding = {"id": campaign, "campaign_sha256": reporter.sha256(config_path),
               "registry_sha256": reporter.sha256(workspace / "registry.json")}
    state_path = workspace / "state" / campaign / "run_state.json"
    write(state_path, {**binding, "status": "training_complete", "completed_fits": 12,
                       "active_child": None, "outer_metrics_computed": False})
    manifest_path = output / "completed_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(binding)
    write(manifest_path, manifest)
    completion_path = output / "completion_receipt.json"
    completion = json.loads(completion_path.read_text())
    completion.update(binding, completed_manifest_sha256=reporter.sha256(manifest_path))
    write(completion_path, completion)
    sources = {}
    for name in ("scripts/evaluate_fixed_epoch_cloud_v1.py", "scripts/test_evaluate_fixed_epoch_cloud_v1.py"):
        path = canonical / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# generated evaluator identity fixture")
        sources[name] = reporter.sha256(path)
    protocol = {"id": campaign, "status": "frozen_before_assessment", "quality_metrics_read_when_frozen": False,
                "real_predictions_scored_when_frozen": False, "source_sha256": sources,
                "registry_sha256": binding["registry_sha256"], "preparation_sha256": "b" * 64,
                "source_split_sha256": "c" * 64}
    protocol_path = canonical / "artifacts" / campaign / "local_evaluation_protocol.json"
    write(protocol_path, protocol)
    protocol_hash = reporter.sha256(protocol_path)
    test_receipt = {"id": campaign, "status": "passed", "tests_passed": 25,
        "local_protocol_sha256": protocol_hash, "real_predictions_scored": False,
        "models_loaded": False, "fitting_performed": False, "gpu_used": False, "source_sha256": sources}
    test_path = canonical / "artifacts" / campaign / "evaluator_smoke/verification.json"
    write(test_path, test_receipt)
    claim = {"campaign_sha256": binding["campaign_sha256"], "registry_sha256": binding["registry_sha256"],
             "manifest_sha256": reporter.sha256(manifest_path), "local_protocol_sha256": protocol_hash}
    write(output / "evaluation_claim.json", claim)
    evaluation_path = output / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation.update(script_sha256=sources["scripts/evaluate_fixed_epoch_cloud_v1.py"],
        test_script_sha256=sources["scripts/test_evaluate_fixed_epoch_cloud_v1.py"],
        native_inference_executed_by_evaluator=False,
        native_verification_execution_location="cloud worker; receipt checked locally", claim=claim)
    evaluation["provenance"].update(campaign_sha256=binding["campaign_sha256"],
        registry_sha256=binding["registry_sha256"], manifest_sha256=reporter.sha256(manifest_path),
        completion_sha256=reporter.sha256(completion_path), local_protocol_sha256=protocol_hash,
        preparation_sha256=protocol["preparation_sha256"], canonical_split_sha256=protocol["source_split_sha256"],
        cloud_split_sha256=config["split_sha256"], canonical_development_rows_exactly_matched=True,
        raw_cloud_train_aux_decoded=False, source_hashes=registry["source_hashes"])
    write(evaluation_path, evaluation)
    return workspace, canonical, protocol_hash, reporter.sha256(test_path), evaluation


class CloudReporterTests(unittest.TestCase):
    def patched(self, stack: ExitStack, fixture):
        stack.enter_context(patch.object(reporter, "PROTOCOL_SHA256", fixture[2]))
        stack.enter_context(patch.object(reporter, "EVALUATOR_TEST_RECEIPT_SHA256", fixture[3]))

    def test_completed_scored_fixture_plot_literal_metrics_and_nonmutation(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            fixture = cloud_fixture(Path(name))
            workspace, canonical, _, _, evaluation = fixture
            self.patched(stack, fixture)
            before = {path: reporter.sha256(path) for path in Path(name).rglob("*") if path.is_file()}
            output = workspace / "artifacts" / reporter.CAMPAIGN_ID / "report/generated_only"
            result = reporter.summarize(output, root=workspace, canonical_root=canonical)
            evidence = json.loads(Path(result["evidence"]).read_text())
            for key in ("metrics", "contrasts", "advancement"):
                self.assertEqual(evidence["evaluation_copied_without_recomputation"][key], evaluation[key])
            self.assertEqual(len(evidence["trajectory_summaries"]), 12)
            self.assertIn("schedule horizon", evidence["interpretation"]["B_minus_A"])
            self.assertIn("not a schedule-independent", evidence["interpretation"]["C_minus_B"])
            self.assertIn("Cloud worker performed", evidence["interpretation"]["native_verification"])
            self.assertEqual(Path(result["plot"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(before, {path: reporter.sha256(path) for path in before})
            with self.assertRaises(FileExistsError):
                reporter.summarize(output, root=workspace, canonical_root=canonical)

    def test_missing_evaluation_or_unfinished_cloud_is_rejected_before_write(self):
        for defect in ("no_evaluation", "running"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name, ExitStack() as stack:
                fixture = cloud_fixture(Path(name))
                workspace, canonical = fixture[:2]
                self.patched(stack, fixture)
                if defect == "no_evaluation":
                    (workspace / "artifacts" / reporter.CAMPAIGN_ID / "evaluation.json").unlink()
                else:
                    state = workspace / "state" / reporter.CAMPAIGN_ID / "run_state.json"
                    value = json.loads(state.read_text()); value["status"] = "running"
                    write(state, value)
                output = workspace / "artifacts" / reporter.CAMPAIGN_ID / "report/rejected"
                with self.assertRaises((ValueError, FileNotFoundError)):
                    reporter.summarize(output, root=workspace, canonical_root=canonical)
                self.assertFalse(output.exists())

    def test_protocol_claim_and_curve_drift_rejected(self):
        for defect in ("protocol", "claim", "curves"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name, ExitStack() as stack:
                fixture = cloud_fixture(Path(name))
                workspace, canonical = fixture[:2]
                self.patched(stack, fixture)
                output = workspace / "artifacts" / reporter.CAMPAIGN_ID
                if defect == "protocol":
                    path = canonical / "artifacts" / reporter.CAMPAIGN_ID / "local_evaluation_protocol.json"
                elif defect == "claim":
                    path = output / "evaluation_claim.json"
                else:
                    path = output / "fold_0/inner/A/curves.jsonl"
                path.write_text("{}")
                with self.assertRaises(ValueError):
                    reporter.collect_completed(workspace, canonical_root=canonical)

    def test_reused_curve_and_plot_logic_and_no_gpu_imports(self):
        import summarize_fixed_epoch_v1 as shared
        path = Path(reporter.__file__)
        new = ast.parse(path.read_text())
        self.assertIs(reporter._curves, shared._curves)
        self.assertIs(reporter.render_plot, shared.render_plot)
        self.assertEqual(reporter.sha256(Path(shared.__file__)), reporter.SHARED_REPORTER_SHA256)
        imports = [node.module or "" for node in ast.walk(new) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(new) if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(any(name.split(".")[0] in {"torch", "pytabkit", "pytorch_lightning", "sklearn"} for name in imports))


if __name__ == "__main__":
    unittest.main()
