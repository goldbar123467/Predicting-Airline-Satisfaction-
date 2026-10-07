"""Synthetic long-job dispatch regressions; every API operation is mocked."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import kaggle_private_job as dispatch
import test_kaggle_production_dispatch as fixture


class LongDispatchTests(unittest.TestCase):
    setUp = fixture.ProductionDispatchTests.setUp
    sha = staticmethod(fixture.ProductionDispatchTests.sha)
    write = staticmethod(fixture.ProductionDispatchTests.write)
    bind_freeze = fixture.ProductionDispatchTests.bind_freeze
    save = fixture.ProductionDispatchTests.save
    invoke = fixture.ProductionDispatchTests.invoke
    assert_no_dispatch = fixture.ProductionDispatchTests.assert_no_dispatch

    def configure_long(self, production=False):
        self.protocol = {'candidate_id': 'v3_cloud_realmlp_raw_aux_e60',
                         'execution_image': 'gcr.io/synthetic/gpu@sha256:' + 'c' * 64}
        self.protocol_path = self.root / 'configs/third_pass_long_cloud.json'
        self.write(self.protocol_path, self.protocol)
        self.stack.enter_context(patch.object(dispatch, 'LONG_PROTOCOL_SHA', self.sha(self.protocol_path)))
        self.folder = self.root / ('cloud/third_pass_long/refits/synthetic' if production
                                   else 'cloud/third_pass_long')
        self.folder.mkdir(parents=True)
        (self.folder / 'job.py').write_text('# SYNTHETIC long-job source\n', encoding='utf-8')
        self.prepared['runtime_sha256'] = self.sha(self.folder / 'job.py')
        self.meta.update(enable_gpu=True, docker_image=self.protocol['execution_image'])
        if production:
            name = self.protocol['candidate_id']
            self.freeze = {'weights': {name: .1}, 'source_result_hashes': {name: 'a' * 64}}
            self.spec.update(run_id=name, timeout_seconds=750,
                             execution_image=self.protocol['execution_image'])
            self.bind_freeze()
        else:
            self.meta.update(id='clarkkitchen/s6e10-long-synthetic', competition_sources=[])
            payload = self.folder / 'bundle/payload/protocol.json'
            payload.parent.mkdir(parents=True)
            payload.write_bytes(self.protocol_path.read_bytes())
            self.save()
        fixture.FixedDateTime.fixed = datetime(2026, 10, 2, 15, 15, tzinfo=timezone.utc)

    def test_cv_4500_exact_latest_start_and_one_second_late(self):
        self.configure_long()
        self.assertEqual(dispatch.dispatch_deadline(self.folder, self.meta, 4500, False),
                         datetime(2026, 10, 2, 16, 30, tzinfo=timezone.utc))
        with patch.object(fixture.FixedDateTime, 'fixed', datetime(2026, 10, 2, 15, 15, 1, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, 'cutoff'):
                self.invoke(production=False, timeout=4500)
        self.assert_no_dispatch()
        self.invoke(production=False, timeout=4500)
        self.api.kernels_push.assert_called_once_with(str(self.folder), timeout='4500',
                                                     acc='NvidiaTeslaP100')

    def test_cv_later_start_requires_shorter_bounded_timeout(self):
        self.configure_long()
        fixture.FixedDateTime.fixed = datetime(2026, 10, 2, 15, 20, tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            self.invoke(production=False, timeout=4500)
        self.assert_no_dispatch()
        self.invoke(production=False, timeout=4200)
        self.api.kernels_push.assert_called_once_with(str(self.folder), timeout='4200',
                                                     acc='NvidiaTeslaP100')

    def test_cv_rejects_limits_image_gpu_attachments_and_source(self):
        self.configure_long()
        for timeout in [59, 4501]:
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, 'CV dispatch'):
                self.invoke(production=False, timeout=timeout)
        for mapping, key, value in [
                (self.meta, 'docker_image', 'other-image'), (self.meta, 'enable_gpu', False),
                (self.meta, 'enable_gpu', None),
                (self.meta, 'competition_sources', ['playground-series-s6e10']),
                (self.prepared, 'runtime_sha256', 'wrong')]:
            original = copy.deepcopy(mapping[key])
            mapping[key] = value
            self.save()
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'CV dispatch'):
                self.invoke(production=False, timeout=4500)
            mapping[key] = original
        self.save()
        (self.folder / 'job.py').write_text('# changed source\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'CV dispatch'):
            self.invoke(production=False, timeout=4500)
        self.assert_no_dispatch()

    def test_cv_binds_both_protocol_copies_by_exact_bytes(self):
        self.configure_long()
        payload = self.folder / 'bundle/payload/protocol.json'
        original = payload.read_bytes()
        payload.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'CV dispatch'):
            self.invoke(production=False, timeout=4500)
        payload.write_bytes(original)
        self.protocol_path.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            self.invoke(production=False, timeout=4500)
        self.assert_no_dispatch()

    def test_unrelated_folder_cannot_inherit_extended_cv_budget(self):
        self.configure_long()
        self.folder = self.root / 'cloud/renamed-long'
        self.folder.mkdir()
        (self.folder / 'job.py').write_text('# synthetic\n', encoding='utf-8')
        self.save()
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            self.invoke(production=False, timeout=4500)
        self.assert_no_dispatch()

    def test_production_exact_750_seconds_latest_start_and_hard_end(self):
        self.configure_long(production=True)
        fixture.FixedDateTime.fixed = datetime(2026, 10, 2, 16, 30, tzinfo=timezone.utc)
        self.assertEqual(dispatch.dispatch_deadline(self.folder, self.meta, 750, True),
                         datetime(2026, 10, 2, 16, 42, 30, tzinfo=timezone.utc))
        with patch.object(fixture.FixedDateTime, 'fixed', datetime(2026, 10, 2, 16, 30, 1, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, 'production dispatch'):
                self.invoke(timeout=750)
        for timeout in [749, 751, 1500]:
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, 'contract differs'):
                self.invoke(timeout=timeout)
        self.assert_no_dispatch()
        self.invoke(timeout=750)
        self.api.kernels_push.assert_called_once_with(str(self.folder), timeout='750',
                                                     acc='NvidiaTeslaP100')

    def test_production_requires_positive_finite_candidate_weight_and_bound_freeze(self):
        self.configure_long(production=True)
        name = self.protocol['candidate_id']
        for weight in [0, -.1, float('nan'), float('inf')]:
            self.freeze['weights'][name] = weight
            self.bind_freeze()
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, 'contract differs'):
                self.invoke(timeout=750)
        self.freeze['weights'] = {}
        self.bind_freeze()
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            self.invoke(timeout=750)
        self.freeze['weights'][name] = .1
        self.bind_freeze()
        for mapping, key in [(self.spec, 'selection_frozen_sha256'),
                             (self.prepared, 'frozen_selection_hash'), (self.spec, 'cv_result_sha256')]:
            original = mapping[key]
            mapping[key] = 'wrong'
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'contract differs'):
                self.invoke(timeout=750)
            mapping[key] = original
        self.save()
        self.freeze_path.write_bytes(self.freeze_path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            self.invoke(timeout=750)
        self.assert_no_dispatch()

    def test_production_binds_image_gpu_source_budget_and_competition(self):
        self.configure_long(production=True)
        for mapping, key, value in [
                (self.meta, 'docker_image', 'other'), (self.spec, 'execution_image', 'other'),
                (self.meta, 'enable_gpu', False), (self.spec, 'timeout_seconds', 1500),
                (self.prepared, 'runtime_sha256', 'wrong'), (self.meta, 'competition_sources', [])]:
            original = copy.deepcopy(mapping[key])
            mapping[key] = value
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'contract differs'):
                self.invoke(timeout=750)
            mapping[key] = original
        self.save()
        self.protocol_path.write_bytes(self.protocol_path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            self.invoke(timeout=750)
        self.assert_no_dispatch()

    def test_production_folder_and_run_identity_each_prevent_budget_downgrade(self):
        self.configure_long(production=True)
        self.spec['run_id'] = 'v3_renamed_cloud_candidate'
        self.freeze = {'weights': {self.spec['run_id']: .1},
                       'source_result_hashes': {self.spec['run_id']: 'a' * 64}}
        self.bind_freeze()
        for timeout in [750, 1500]:
            with self.subTest(identity='folder', timeout=timeout), self.assertRaisesRegex(ValueError, 'contract differs'):
                self.invoke(timeout=timeout)
        self.folder = self.root / 'cloud/renamed-production'
        self.folder.mkdir()
        (self.folder / 'job.py').write_text('# synthetic\n', encoding='utf-8')
        self.prepared['runtime_sha256'] = self.sha(self.folder / 'job.py')
        self.spec['run_id'] = self.protocol['candidate_id']
        self.freeze = {'weights': {self.spec['run_id']: .1},
                       'source_result_hashes': {self.spec['run_id']: 'a' * 64}}
        self.bind_freeze()
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            self.invoke(timeout=1500)
        self.assert_no_dispatch()


if __name__ == '__main__':
    unittest.main(verbosity=2)
