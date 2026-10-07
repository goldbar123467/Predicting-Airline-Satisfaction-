"""Synthetic proof-binding tests for the separately registered cloud duration run."""
import copy
import unittest
from unittest.mock import patch

import common
import third_pass_release as release
import test_third_pass_release as fixture


class LongReleaseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.ThirdPassReleaseTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run = self.fixture.add_candidate('v3_cloud_realmlp_raw_aux_e60',
                                             family='realmlp_cat', max_rounds=60,
                                             execution_backend='kaggle')
        self.receipt = self.fixture.verify_cloud_fixture(self.run)
        self.protocol = {'protocol_id': 'realmlp_long_schedule_v1', 'runs': [self.run],
                         'execution_image': self.receipt['execution_image'],
                         'control_reuse': {'id': 'synthetic_verified_control', 'result_sha256': 'a' * 64}}
        path = self.fixture.root / 'configs/third_pass_long_cloud.json'
        common.atomic_json(path, self.protocol)
        self.fixture.stack.enter_context(patch.object(release, 'LONG_CLOUD_PROTOCOL_SHA', common.sha256(path)))
        self.receipt.update(protocol_id=self.protocol['protocol_id'],
                            protocol_sha256=common.sha256(path),
                            reused_control=copy.deepcopy(self.protocol['control_reuse']),
                            runtime_sha256=release.LONG_CLOUD_RUNTIME_SHA,
                            bundle_manifest_sha256=release.LONG_CLOUD_BUNDLE_SHA)
        self.receipt_path = self.fixture.root / 'artifacts/runs' / self.run['id'] / 'cloud_import_verification.json'

    def check(self, receipt):
        common.atomic_json(self.receipt_path, receipt)
        return release.checked_cloud_import(self.run, self.receipt['split_sha256'])

    def test_exact_registered_proof_accepted(self):
        self.assertEqual(self.check(self.receipt)['run'], self.run)

    def test_protocol_and_reused_control_must_match(self):
        for key in ['protocol_id', 'protocol_sha256', 'reused_control']:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check({**self.receipt, key: 'changed'})

    def test_runtime_and_bundle_must_match(self):
        for key in ['runtime_sha256', 'bundle_manifest_sha256']:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check({**self.receipt, key: 'changed'})

    def test_changed_registered_recipe_rejected(self):
        self.run['max_rounds'] = 61
        with self.assertRaises(ValueError):
            self.check(self.receipt)


if __name__ == '__main__':
    unittest.main()
