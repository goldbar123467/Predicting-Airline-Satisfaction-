"""Generated gain, freeze and keyed-vector boundary checks; no quality scoring."""
import copy
from datetime import datetime, timedelta, timezone
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import continuation_six_selection_v1 as selection
from compose_continuation_six_release_v1 import checked_vector


class SelectionTests(unittest.TestCase):
    def test_assessment_claim_source_and_time_must_match_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("configs/continuation_six_v1.json", "state/continuation_six_v1/authorization.json",
                         "scripts/evaluate_continuation_six_v1.py", "scripts/test_evaluate_continuation_six_v1.py"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{}")
            source_map = {name: selection.sha(root / name) for name in (
                "scripts/evaluate_continuation_six_v1.py", "scripts/test_evaluate_continuation_six_v1.py")}
            protocol = {"id": "fixed_epoch_continuation_01", "status": "frozen_before_assessment",
                "quality_metrics_read_when_frozen": False, "real_predictions_scored_when_frozen": False,
                "sequence_plan_sha256": selection.sha(root / "configs/continuation_six_v1.json"),
                "sequence_authorization_sha256": selection.sha(root / "state/continuation_six_v1/authorization.json"),
                "source_sha256": source_map, "provider_identity": {"files": {}}, "created_utc": "2026-10-04T20:00:00+00:00"}
            protocol_path, claim_path = root / "protocol.json", root / "claim.json"
            workspace = root / "cloud/continuation_six_v1/fixed_epoch_continuation_01/experiment/assessment_workspace"
            claimed_files = {"campaign_sha256": workspace / "configs/fixed_epoch_continuation_01.json",
                             "registry_sha256": workspace / "registry.json",
                             "manifest_sha256": workspace / "artifacts/fixed_epoch_continuation_01/completed_manifest.json"}
            for path in claimed_files.values():
                selection.write(path, {"generated": True})
            protocol["registry_sha256"] = selection.sha(claimed_files["registry_sha256"])
            selection.write(protocol_path, protocol)
            claim = {**{key: selection.sha(path) for key, path in claimed_files.items()},
                     "local_protocol_sha256": selection.sha(protocol_path), "claimed_utc": "2026-10-04T20:01:00+00:00"}
            selection.write(claim_path, claim)
            evaluation = {"claim": claim, "provenance": dict(claim), "completed_utc": "2026-10-04T20:02:00+00:00",
                "script_sha256": source_map["scripts/evaluate_continuation_six_v1.py"],
                "test_script_sha256": source_map["scripts/test_evaluate_continuation_six_v1.py"]}
            selection.validate_assessment(root, "fixed_epoch_continuation_01", evaluation, protocol, protocol_path, claim_path)
            for change in ({"script_sha256": "f" * 64}, {"claim": {}}, {"completed_utc": "2026-10-04T19:59:00+00:00"}):
                with self.assertRaises(ValueError):
                    selection.validate_assessment(root, "fixed_epoch_continuation_01", {**evaluation, **change}, protocol, protocol_path, claim_path)

    def test_gain_requires_pooled_macro_and_each_fold(self):
        old = {"oof_auc": .95, "fold_auc": [.94, .95, .96]}
        good = {"oof_auc": .9501, "fold_auc": [.9401, .9501, .9601]}
        self.assertTrue(selection.gain_gate(good, old))
        for bad in ({**good, "oof_auc": .950001}, {**good, "fold_auc": [.94, .95, .96]},
                    {**good, "fold_auc": [.9399, .9502, .9603]}):
            self.assertFalse(selection.gain_gate(bad, old))
        for bad in ({**good, "oof_auc": float("nan")}, {**good, "fold_auc": [.95]}):
            with self.assertRaises(ValueError):
                selection.gain_gate(bad, old)

    def test_keyed_vectors_reject_order_schema_duplicates_and_nonfinite(self):
        frame = pd.DataFrame({"id": [10, 20, 30], "prediction": [0., .5, 1.]})
        np.testing.assert_array_equal(checked_vector(frame, np.array([10, 20, 30]), "prediction"), [0, .5, 1])
        for bad in (frame.iloc[::-1], frame.assign(id=[10, 10, 30]), frame.assign(prediction=[0, np.nan, 1]),
                    frame.assign(prediction=[0, 2, 1]), frame.assign(extra=0)):
            with self.assertRaises(ValueError):
                checked_vector(bad, np.array([10, 20, 30]), "prediction")

    def test_original_and_prior_dispositions_must_remain_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "artifacts/third_pass/blend/frozen.json"
            selection.write(old, {"oof_auc": .95, "fold_auc": [.94, .95, .96]})
            plan = root / "configs/continuation_six_v1.json"
            selection.write(plan, {"incumbent_selection_sha256": selection.sha(old)})
            path, value = selection.best_prior(root, "fixed_epoch_continuation_01")
            self.assertEqual(path, old)
            proof = root / "failure.json"
            selection.write(proof, {"status": "terminal_failure"})
            disposition = root / "state/continuation_six_v1/dispositions/fixed_epoch_continuation_01.json"
            selection.write(disposition, {"stage": "fixed_epoch_continuation_01", "plan_sha256": selection.sha(plan),
                "status": "terminal_failure", "evidence_path": "failure.json", "evidence_sha256": selection.sha(proof)})
            selection.best_prior(root, "fixed_epoch_continuation_02")
            proof.write_text("{}")
            with self.assertRaises(ValueError):
                selection.best_prior(root, "fixed_epoch_continuation_02")


if __name__ == "__main__":
    unittest.main()
