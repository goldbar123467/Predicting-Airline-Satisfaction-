"""No API or submission calls: verify scope, expiry and single-attempt safety."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import submit_verified_release as submit


class BestBlendAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        permission = submit.load_config(submit.ROOT / 'state/kaggle_best_blend_authorization.json')
        self.path = self.root / 'state/kaggle_best_blend_authorization.json'
        self.path.parent.mkdir()
        self.path.write_text(json.dumps(permission))
        (self.root / 'artifacts/kaggle').mkdir(parents=True)
        self.patcher = patch.object(submit, 'ROOT', self.root)
        self.patcher.start(); self.addCleanup(self.patcher.stop)
        self.hashpatch = patch.object(submit, 'BEST_BLEND_AUTH_SHA', submit.sha256(self.path))
        self.hashpatch.start(); self.addCleanup(self.hashpatch.stop)
        self.now = datetime(2026, 10, 3, 1, tzinfo=timezone.utc)

    def test_exact_campaign_in_window(self):
        self.assertEqual(submit.submission_window('third_pass_batch07', self.now), submit.BEST_BLEND_AUTH_SHA)

    def test_other_campaign_and_expiry_rejected(self):
        with self.assertRaises(RuntimeError): submit.submission_window('third_pass_batch08', self.now)
        with self.assertRaises(RuntimeError): submit.submission_window('third_pass_batch07', datetime(2026, 10, 3, 13, tzinfo=timezone.utc))

    def test_changed_or_missing_permission_rejected(self):
        self.path.write_text('{}')
        with self.assertRaises(RuntimeError): submit.submission_window('third_pass_batch07', self.now)
        self.path.unlink()
        with self.assertRaises(RuntimeError): submit.submission_window('third_pass_batch07', self.now)

    def test_even_uncertain_prior_intent_consumes_attempt(self):
        (self.root / 'artifacts/kaggle/submission_test.json').write_text(json.dumps({
            'status': 'intent_recorded', 'renewed_authorization_sha256': submit.BEST_BLEND_AUTH_SHA}))
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            submit.submission_window('third_pass_batch07', self.now)


if __name__ == '__main__': unittest.main()
