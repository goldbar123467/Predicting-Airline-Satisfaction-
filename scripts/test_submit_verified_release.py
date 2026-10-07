"""Uploader regressions: temporary artifacts, fixed clock, fully mocked API."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import common
import submit_verified_release as upload


class FixedDateTime(datetime):
    fixed = datetime(2026, 10, 2, 14, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed if tz is None else cls.fixed.astimezone(tz)


def remote(ref=123, description="unrelated", date=None, score="0.99999"):
    item = SimpleNamespace(ref=ref, description=description,
                           date=date or FixedDateTime.fixed, public_score=score)
    item.to_dict = lambda: {"ref": ref, "description": description, "public_score": score}
    return item


class VerifiedUploaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="verified-uploader-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(upload, "ROOT", self.root))
        self.stack.enter_context(patch.object(upload, "datetime", FixedDateTime))
        self.stack.enter_context(patch.object(FixedDateTime, "fixed", datetime(2026, 10, 2, 14, tzinfo=timezone.utc)))
        self.api = SimpleNamespace(competition_submissions=Mock(return_value=[]),
                                   competition_submit=Mock(return_value=remote()))
        self.stack.enter_context(patch.object(upload, "api_client", return_value=self.api))
        common.atomic_json(self.root / "artifacts/second_pass/blend/frozen.json", {
            "oof_auc": .9, "fold_auc": [.9, .9, .9]})
        self.stack.enter_context(patch.object(upload, "V2_BASELINE_SHA",
            common.sha256(self.root / "artifacts/second_pass/blend/frozen.json")))
        self.publish("third_pass", .91)

    def publish(self, campaign, auc, folds=None):
        self.campaign = campaign
        out = self.root / "artifacts" / campaign
        final = out / "final"
        final.mkdir(parents=True, exist_ok=True)
        selection = {"oof_auc": auc, "fold_auc": folds or [auc] * 3,
                     "audit_evaluated": False, "weights": {"synthetic": 1.0},
                     "baseline_selection_sha256": common.sha256(self.root / "artifacts/second_pass/blend/frozen.json")}
        selection_path = out / "blend/frozen.json"
        common.atomic_json(selection_path, selection)
        submission = final / "submission.csv"
        # The verifier's signed row-count assertion is synthetic. No competition
        # matrix is read; the uploader is not responsible for repeating inference.
        submission.write_text(f"id,satisfaction\n0001,{auc}\n", encoding="utf-8")
        report = self.root / (campaign.upper() + "_REPORT.md")
        report.write_text("Synthetic verified report\n", encoding="utf-8")
        manifest = final / "manifest.json"
        common.atomic_json(manifest, {"frozen_selection_hash": common.sha256(selection_path),
                                     "submission_hash": common.sha256(submission),
                                     "members": [{"id": "synthetic", "weight": 1.0}]})
        verification = out / "verification.json"
        common.atomic_json(verification, {"sha256": common.sha256(submission), "rows": 299844,
                                         "all_row_raw_inference": True,
                                         "independent_blend_recomputation": True, "audit_evaluated": False})
        native = final / "native.bin"
        native.write_bytes(b"SYNTHETIC_NATIVE")
        source = final / "source.py"
        source.write_text("# synthetic archived source\n", encoding="utf-8")
        common.atomic_json(final / "release_provenance.json", {
            "submission_sha256": common.sha256(submission), "manifest_sha256": common.sha256(manifest),
            "selection_sha256": common.sha256(selection_path), "verification_sha256": common.sha256(verification),
            "report_sha256": common.sha256(report),
            "native_hashes": {native.relative_to(self.root).as_posix(): common.sha256(native)},
            "source_hashes": {source.relative_to(self.root).as_posix(): common.sha256(source)}})
        return submission, selection_path, selection

    def invoke(self):
        with patch.object(sys, "argv", ["submit_verified_release.py", "--campaign", self.campaign]):
            upload.main()

    def receipt_path(self, campaign=None):
        path = self.root / "artifacts" / (campaign or self.campaign) / "final/submission.csv"
        return self.root / "artifacts/kaggle" / f"submission_{common.sha256(path)[:16]}.json"

    def test_valid_release_requires_native_source_and_bound_file_hashes(self):
        upload.verification_contract(self.campaign)
        out = self.root / "artifacts" / self.campaign
        for path in [out / "final/submission.csv", out / "final/manifest.json", out / "blend/frozen.json",
                     out / "verification.json", self.root / "THIRD_PASS_REPORT.md",
                     out / "final/native.bin", out / "final/source.py"]:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    upload.verification_contract(self.campaign)
                path.write_bytes(original)
        self.api.competition_submit.assert_not_called()

    def test_gain_gate_requires_pooled_mean_and_worst_fold_improvements(self):
        prior = {"oof_auc": .9, "fold_auc": [.9] * 3}
        self.assertTrue(upload.gain_gate({"oof_auc": .9001, "fold_auc": [.9001] * 3}, prior))
        for selected in [{"oof_auc": .9, "fold_auc": [.91] * 3},
                         {"oof_auc": .91, "fold_auc": [.9] * 3},
                         {"oof_auc": .91, "fold_auc": [.899, .92, .92]}]:
            self.assertFalse(upload.gain_gate(selected, prior))

    def test_empty_native_or_archived_source_manifest_is_rejected(self):
        path = self.root / "artifacts/third_pass/final/release_provenance.json"
        original = common.load_config(path)
        for key in ["native_hashes", "source_hashes"]:
            common.atomic_json(path, {**original, key: {}})
            with self.subTest(key=key), self.assertRaises(ValueError):
                upload.verification_contract(self.campaign)
        common.atomic_json(path, original)

    def test_initial_v2_comparison_selection_must_match_frozen_anchor_hash(self):
        baseline = self.root / "artifacts/second_pass/blend/frozen.json"
        baseline.write_bytes(baseline.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "anchor|baseline|selection"):
            self.invoke()
        self.api.competition_submit.assert_not_called()

    def test_intent_is_durable_before_submit_and_remote_public_score_is_unused(self):
        self.api.competition_submissions.return_value = [remote(score="1.00000")]

        def submit(*args, **kwargs):
            intent = common.load_config(self.receipt_path())
            self.assertEqual(intent["status"], "intent_recorded")
            self.assertEqual(intent["sha256"], common.sha256(Path(args[0])))
            return remote(456, score="0.00000")

        self.api.competition_submit.side_effect = submit
        self.invoke()
        self.api.competition_submit.assert_called_once()
        receipt = common.load_config(self.receipt_path())
        self.assertEqual(receipt["response"]["ref"], 456)

    def test_daily_allowance_and_deadline_prevent_new_intent(self):
        self.api.competition_submissions.return_value = [remote(ref=x) for x in range(10)]
        with self.assertRaisesRegex(RuntimeError, "allowance"):
            self.invoke()
        self.assertFalse(self.receipt_path().exists())
        self.api.competition_submissions.return_value = []
        with patch.object(FixedDateTime, "fixed", datetime(2026, 10, 2, 17, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                self.invoke()
        self.assertFalse(self.receipt_path().exists())
        self.api.competition_submit.assert_not_called()

    def test_previous_day_submissions_do_not_exhaust_today(self):
        self.api.competition_submissions.return_value = [remote(ref=x, date=FixedDateTime.fixed - timedelta(days=1)) for x in range(10)]
        self.invoke()
        self.api.competition_submit.assert_called_once()

    def test_remote_hash_without_receipt_blocks_duplicate(self):
        path = self.root / "artifacts/third_pass/final/submission.csv"
        self.api.competition_submissions.return_value = [remote(description=f"sha256 {common.sha256(path)[:12]}")]
        with self.assertRaisesRegex(RuntimeError, "already submitted"):
            self.invoke()
        self.api.competition_submit.assert_not_called()

    def test_timeout_intent_never_blindly_retries(self):
        self.api.competition_submit.side_effect = TimeoutError("Synthetic network uncertainty")
        with self.assertRaises(TimeoutError):
            self.invoke()
        self.assertEqual(common.load_config(self.receipt_path())["status"], "intent_recorded")
        self.api.competition_submit.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "identity uncertain"):
            self.invoke()
        self.api.competition_submit.assert_not_called()

    def test_incremental_gate_uses_best_submitted_development_selection(self):
        for index, auc in enumerate([.909, .92, .91]):
            selected = self.root / f"artifacts/prior{index}.json"
            common.atomic_json(selected, {"oof_auc": auc, "fold_auc": [auc] * 3})
            common.atomic_json(self.root / f"artifacts/kaggle/submission_prior{index}.json", {
                "selection_path": selected.relative_to(self.root).as_posix(),
                "selection_sha256": common.sha256(selected), "response": {"ref": index + 1}})
        self.invoke()
        self.api.competition_submit.assert_not_called()
        self.assertFalse(self.receipt_path().exists())

    def test_reconciled_timeout_submission_remains_in_future_incremental_gate(self):
        path, selection_path, _ = self.publish("third_pass", .93)
        digest = common.sha256(path)
        common.atomic_json(self.receipt_path(), {
            "status": "intent_recorded", "sha256": digest,
            "selection_sha256": common.sha256(selection_path),
            "selection_path": selection_path.relative_to(self.root).as_posix()})
        self.api.competition_submissions.return_value = [remote(991, f"sha256 {digest[:12]}")]
        self.invoke()  # Successful reconciliation must count as a real prior submission.
        self.api.competition_submit.assert_not_called()
        self.publish("third_pass_batch02", .930005)
        self.invoke()
        self.api.competition_submit.assert_not_called()
        self.assertFalse(self.receipt_path().exists())

    def test_competition_lock_prevents_overlapping_transactions_before_api_query(self):
        with upload.submission_lock():
            with self.assertRaisesRegex(RuntimeError, "transaction is active"):
                self.invoke()
        self.api.competition_submissions.assert_not_called()
        self.api.competition_submit.assert_not_called()
        # Releasing the owning handle must make a subsequent transaction possible.
        self.invoke()
        self.api.competition_submit.assert_called_once()

    def test_changed_prior_selection_blocks_upload(self):
        selected = self.root / "artifacts/prior.json"
        common.atomic_json(selected, {"oof_auc": .905, "fold_auc": [.905] * 3})
        common.atomic_json(self.root / "artifacts/kaggle/submission_prior.json", {
            "selection_path": selected.relative_to(self.root).as_posix(),
            "selection_sha256": common.sha256(selected), "response": {"ref": 15}})
        selected.write_bytes(selected.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "Prior submitted development selection changed"):
            self.invoke()
        self.api.competition_submit.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
