"""No-fit checks for the exact overnight registration and resource exception."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import long_local_500_policy as policy
import supervisor
from third_pass_release import duration_comparison


class LongLocal500PolicyTests(unittest.TestCase):
    def setUp(self):
        self.config_path = policy.ROOT / 'configs/third_pass_batch07.json'
        self.config = json.loads(self.config_path.read_text())
        self.protocol = json.loads((policy.ROOT / 'configs/long_local_500.json').read_text())

    def test_registered_queue_allows_exact_reserves(self):
        self.assertEqual(policy.registered_policy(self.config), self.protocol)
        result = supervisor.load_config(self.config_path)
        self.assertEqual(result['refit_timeout_seconds'], 14400)
        self.assertEqual(result['verify_reserve_seconds'], 3600)

    def test_changed_queue_cannot_borrow_exception(self):
        for key, value in [('refit_timeout_seconds', 20000), ('campaign_id', 'third_pass_batch08'),
                           ('automatic_submission', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                policy.registered_policy({**self.config, key: value})

    def test_missing_registration_fails_closed(self):
        config = copy.deepcopy(self.config)
        config.pop('long_local_500_registration')
        with self.assertRaises(ValueError):
            policy.registered_policy(config)

    def test_changed_training_source_fails(self):
        original = policy.digest
        with patch.object(policy, 'digest', side_effect=lambda path: 'changed' if path.name == 'train.py' else original(path)):
            with self.assertRaises(ValueError):
                policy.registered_policy(self.config)

    def test_ordinary_campaign_cannot_use_long_budget(self):
        config = json.loads((policy.ROOT / 'configs/third_pass_batch05.json').read_text())
        config['refit_timeout_seconds'] = 14400
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.json'
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                supervisor.load_config(path)

    def test_500_matched_gate_and_horizon_are_explicit(self):
        before = dict(pooled=.8, mean_fold=.8, folds=[.8, .8, .8])
        candidate, control = self.protocol['candidate_run'], self.protocol['control_run']
        self.assertTrue(duration_comparison(candidate, control, before, before, horizon=500))
        with self.assertRaises(ValueError):
            duration_comparison(candidate, control, before, before)
        self.assertFalse(duration_comparison(candidate, control, {**before, 'pooled': .79}, before, horizon=500))


if __name__ == '__main__':
    unittest.main()
