"""Synthetic LightGBM cloud admission/release contracts; no API or model fit."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import common
import third_pass_release as release
import test_third_pass_release as release_fixture
import test_cloud_production_gate as production_fixture


class LGBCloudReleaseTests(unittest.TestCase):
    setUp = release_fixture.ThirdPassReleaseTests.setUp
    write_config = release_fixture.ThirdPassReleaseTests.write_config
    add_candidate = release_fixture.ThirdPassReleaseTests.add_candidate
    create_native_member = release_fixture.ThirdPassReleaseTests.create_native_member
    verify_cloud_fixture = release_fixture.ThirdPassReleaseTests.verify_cloud_fixture

    def pair(self, control=None, candidate=None):
        recipe = {'family': 'lightgbm', 'execution_backend': 'kaggle', 'route': True,
                  'teacher': True, 'original_aux': True, 'seed': 20261003,
                  'max_rounds': 2600, 'params': {'num_leaves': 127, 'learning_rate': .03}}
        self.control = self.add_candidate('v3_cloud_lgb_control',
                                          np.full(60, .5) if control is None else control, **recipe)
        self.candidate = self.add_candidate('v3_cloud_lgb_probability', candidate,
                                            original_aux_probability=True, **recipe)
        self.protocol = {'protocol_id': 'lgb_rating_probability_v1',
                         'execution_image': 'synthetic-cpu-image@sha256:' + '9' * 64,
                         'runs': [self.control, self.candidate]}
        self.protocol_path = self.root / 'configs/third_pass_lgb_cloud.json'
        common.atomic_json(self.protocol_path, self.protocol)
        self.stack.enter_context(patch.object(release, 'LGB_CLOUD_PROTOCOL_SHA', common.sha256(self.protocol_path)))
        self.stack.enter_context(patch.object(release, 'LGB_CLOUD_RUNTIME_SHA', 'synthetic-runtime'))
        self.stack.enter_context(patch.object(release, 'LGB_CLOUD_BUNDLE_SHA', 'synthetic-bundle'))
        self.config['selection_exclusions'] = [self.control['id']]
        self.config['comparison_controls'] = {self.candidate['id']: self.control['id']}
        self.write_config()
        self.receipts = {r['id']: self.receipt(r) for r in self.protocol['runs']}

    def receipt(self, run):
        receipt = self.verify_cloud_fixture(run)
        receipt.update(protocol_id=self.protocol['protocol_id'],
                       protocol_sha256=release.LGB_CLOUD_PROTOCOL_SHA,
                       execution_image=self.protocol['execution_image'],
                       runtime_sha256=release.LGB_CLOUD_RUNTIME_SHA,
                       bundle_manifest_sha256=release.LGB_CLOUD_BUNDLE_SHA,
                       lightgbm_version='4.7.0', execution_threads=4, training_device='cpu',
                       all_test_fold_native_inference=True)
        self.save_receipt(run, receipt)
        return receipt

    def save_receipt(self, run, receipt):
        common.atomic_json(self.root / 'artifacts/runs' / run['id'] / 'cloud_import_verification.json', receipt)

    def check(self, run=None):
        return release.checked_cloud_import(run or self.candidate, common.sha256(self.root / 'data/splits.parquet'))

    def test_exact_registered_run_and_protocol_source_are_required(self):
        self.pair()
        self.check()
        for key, value in [('seed', 42), ('family', 'realmlp_cat'), ('max_rounds', 99),
                           ('original_aux_probability', False), ('params', {'num_leaves': 2})]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'recipe'):
                self.check({**self.candidate, key: value})
        self.protocol_path.write_bytes(self.protocol_path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'protocol'):
            self.check()

    def test_unprepared_runtime_or_bundle_pins_fail_closed(self):
        self.pair()
        for constant in ['LGB_CLOUD_RUNTIME_SHA', 'LGB_CLOUD_BUNDLE_SHA']:
            with patch.object(release, constant, None), self.assertRaisesRegex(ValueError, 'recipe'):
                self.check()

    def test_cpu_version_threads_all_test_replay_and_receipt_bindings(self):
        self.pair()
        receipt = self.receipts[self.candidate['id']]
        invalid = {'protocol_id': 'other', 'protocol_sha256': 'wrong', 'lightgbm_version': '4.6.0',
                   'execution_threads': 8, 'training_device': 'gpu', 'all_test_fold_native_inference': False,
                   'all_heldout_native_inference': False, 'heldout_rows': 59,
                   'runtime_sha256': 'wrong', 'bundle_manifest_sha256': 'wrong', 'execution_image': 'other',
                   'result_sha256': 'wrong', 'oof_sha256': 'wrong', 'test_sha256': 'wrong',
                   'split_sha256': 'wrong', 'run': {**self.candidate, 'seed': 1}}
        for key, value in invalid.items():
            self.save_receipt(self.candidate, {**receipt, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check()
        self.save_receipt(self.candidate, receipt)
        self.check()

    def test_source_native_replay_maps_are_owned_nonempty_and_checksum_bound(self):
        self.pair()
        receipt = self.receipts[self.candidate['id']]
        external = self.root / 'outside.stub'
        external.write_text('synthetic external')
        for field in ['source_hashes', 'native_hashes', 'heldout_replay_receipt_hashes']:
            for value in [{}, None, {'outside.stub': common.sha256(external)}]:
                self.save_receipt(self.candidate, {**receipt, field: value})
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.check()
            self.save_receipt(self.candidate, receipt)
            path = self.root / next(iter(receipt[field]))
            original = path.read_bytes()
            path.write_bytes(original + b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact'):
                self.check()
            path.write_bytes(original)
        self.check()

    def test_passing_candidate_selected_but_control_always_excluded(self):
        self.pair()
        release.blend(self.config_path)
        selection = common.load_config(release.BLEND / 'current.json')
        self.assertGreater(selection['weights'].get(self.candidate['id'], 0), 0)
        self.assertNotIn(self.control['id'], selection['weights'])
        self.assertFalse(selection['audit_evaluated'])

    def test_candidate_worse_than_matched_control_cannot_enter_blend(self):
        imperfect = self.y.astype(float).copy()
        imperfect[[0, 1]] = imperfect[[1, 0]]
        self.pair(control=self.y.astype(float), candidate=imperfect)
        release.blend(self.config_path)
        selection = common.load_config(release.BLEND / 'current.json')
        self.assertNotIn(self.candidate['id'], selection['weights'])
        self.assertNotIn(self.control['id'], selection['weights'])
        ledger = pd.read_csv(release.OUT / 'experiment_ledger.csv')
        self.assertFalse(bool(ledger.loc[ledger.id == self.candidate['id'], 'matched_cloud_gate'].iloc[0]))

    def test_control_must_be_excluded_same_platform(self):
        self.pair()
        self.config['selection_exclusions'] = []
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'platform control'):
            release.blend(self.config_path)
        self.config['selection_exclusions'] = [self.control['id']]
        self.control['execution_backend'] = 'local'
        self.verify_cloud_fixture(self.control)
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'platform control'):
            release.blend(self.config_path)

    def test_q_flag_is_the_only_permitted_paired_recipe_change(self):
        self.pair()
        for run, key, value in [(self.candidate, 'seed', 77),
                                 (self.candidate, 'original_aux_probability', False),
                                 (self.control, 'original_aux_probability', True)]:
            original = copy.deepcopy(run)
            run[key] = value
            common.atomic_json(self.protocol_path, self.protocol)
            with patch.object(release, 'LGB_CLOUD_PROTOCOL_SHA', common.sha256(self.protocol_path)):
                for member in self.protocol['runs']:
                    self.receipt(member)
                self.write_config()
                with self.subTest(key=key, run=run['id']), self.assertRaisesRegex(ValueError, 'probability feature block'):
                    release.blend(self.config_path)
            run.clear()
            run.update(original)

    def test_raw_lgb_route_uses_float32_numpy_string_semantics_cloud_only(self):
        raw = pd.DataFrame({'id': ['0001', '0002', '0003'],
                            'Flight Distance': [np.nan, 16777217., 1.23456789], 'Class': ['a', 'b', 'a']})
        raw.to_csv(self.root / 'data/raw.csv', index=False)
        common.atomic_json(self.root / 'data/manifest.json', {'feature_columns': ['Flight Distance', 'Class']})
        release.FINAL.mkdir(parents=True)
        runs = [{'id': 'cloud', 'family': 'lightgbm', 'execution_backend': 'kaggle', 'route': True},
                {'id': 'local', 'family': 'lightgbm', 'route': True}]
        members = [{'id': r['id'], 'run': r, 'weight': .5, 'path': r['id']} for r in runs]
        def feature_frame(frame, run):
            self.assertEqual(frame['Flight Distance'].dtype, np.dtype('float32'))
            return pd.DataFrame({'route_category': ['LOCAL_SENTINEL'] * len(frame)}, index=frame.index)
        def prediction(model, transform, frame):
            expected = raw['Flight Distance'].to_numpy(dtype=np.float32).astype(str).tolist()
            self.assertEqual(frame.route_category.tolist(), expected if str(model) == 'cloud' else ['LOCAL_SENTINEL'] * 3)
            return np.full(len(frame), .5)
        with patch.object(release, 'context', return_value=({}, [])), \
                patch.object(release, 'checked_manifest', return_value={'members': members}), \
                patch.object(release, 'features', side_effect=feature_frame), \
                patch.object(release, 'load_model', side_effect=lambda path: (path, None)), \
                patch.object(release, 'predict', side_effect=prediction), \
                patch.object(release, 'cloud_numeric_category_compat', side_effect=AssertionError('No neural adapter')):
            release.raw_inference(self.config_path, self.root / 'data/raw.csv', release.FINAL / 'raw.csv')
        actual = pd.read_csv(release.FINAL / 'raw.csv', dtype={'id': str})
        self.assertEqual(actual.id.tolist(), raw.id.tolist())


class LGBProductionReleaseTests(unittest.TestCase):
    write_proof = production_fixture.CloudProductionGateTests.write_proof
    check = production_fixture.CloudProductionGateTests.check

    def setUp(self):
        production_fixture.CloudProductionGateTests.setUp(self)
        self.run['family'] = 'lightgbm'
        self.proof.update(rtol=1e-10, atol=1e-12, full_test_max_abs_error=1e-13,
                          protocol_id='lgb_rating_probability_v1', protocol_sha256=release.LGB_CLOUD_PROTOCOL_SHA,
                          lightgbm_version='4.7.0', execution_threads=4, training_device='cpu')
        self.write_proof(self.proof)

    def test_valid_lgb_production_and_required_protocol_cpu_proof(self):
        self.check()
        for key, value in [('protocol_id', 'other'), ('protocol_sha256', 'wrong'),
                           ('lightgbm_version', '4.6.0'), ('execution_threads', 8),
                           ('training_device', 'gpu'), ('all_test_native_inference', False)]:
            self.write_proof({**self.proof, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'proof mismatch'):
                self.check()
        self.proof_path.unlink()
        with self.assertRaisesRegex(ValueError, 'Unverified cached'):
            self.check()

    def test_lgb_rejects_neural_tolerance_and_loose_native_error(self):
        for values in [{'rtol': 1e-5}, {'atol': 2e-6}, {'full_test_max_abs_error': 2e-10},
                       {'full_test_max_abs_error': float('nan')}, {'full_test_max_abs_error': -1e-13}]:
            self.write_proof({**self.proof, **values})
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.check()
        self.write_proof({**self.proof, 'full_test_max_abs_error': 1e-10 + 1e-12})
        self.check()

    def test_neural_tolerance_contract_remains_unchanged(self):
        self.run['family'] = 'realmlp_cat'
        self.write_proof({**self.proof, 'rtol': 1e-5, 'atol': 2e-6, 'full_test_max_abs_error': 1e-7})
        self.check()


if __name__ == '__main__':
    unittest.main()
