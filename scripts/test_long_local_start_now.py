"""Timing amendment checks; no training or process launches."""
import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import long_local_500_policy as policy


class StartNowTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((policy.ROOT / 'configs/third_pass_batch07.json').read_text())
        self.protocol = policy.registered_policy(self.config)

    def test_exact_amendment_advances_all_bounds_preserving_reserves(self):
        before = copy.deepcopy(self.config)
        timing = policy.operational_timing(self.config)
        parse = datetime.fromisoformat
        start = parse(timing['not_before_utc'])
        self.assertEqual((parse(timing['deadline_utc']) - start).total_seconds(), 72000)
        for key in ['stop_new_runs_utc', 'finalize_utc', 'deadline_utc']:
            self.assertEqual(parse(self.config[key]) - parse(timing[key]),
                             parse(self.protocol['not_before_utc']) - start)
        self.assertEqual(parse(timing['deadline_utc']) - parse(timing['finalize_utc']), timedelta(hours=5, minutes=10))
        self.assertEqual(self.config, before)

    def test_changed_amendment_or_readiness_fails_closed(self):
        original = policy.digest
        for filename in ['start_now_authorization.json', 'prior_campaign_ready.json']:
            with self.subTest(filename=filename), patch.object(policy, 'digest',
                    side_effect=lambda p: 'changed' if p.name == filename else original(p)):
                with self.assertRaises(ValueError):
                    policy.operational_timing(self.config)

    def test_ordinary_campaign_timing_unaffected(self):
        config = json.loads((policy.ROOT / 'configs/third_pass_batch05.json').read_text())
        expected = {key: config[key] for key in ['stop_new_runs_utc', 'finalize_utc', 'deadline_utc']}
        self.assertEqual(policy.operational_timing(config), expected)

    def test_deleted_dispatched_amendment_cannot_extend_deadline(self):
        original = Path.exists
        with patch.object(Path, 'exists', lambda p: False if p.name == 'start_now_authorization.json' else original(p)):
            with self.assertRaisesRegex(ValueError, 'amendment is missing'):
                policy.operational_timing(self.config)


if __name__ == '__main__':
    unittest.main()
