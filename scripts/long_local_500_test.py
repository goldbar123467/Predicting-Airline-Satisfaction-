"""Small synthetic waiter tests. No training, cloud calls, or real child launch."""
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import long_local_500_wait as waiter


class FixedTime(datetime):
    current = datetime(2026, 10, 2, 17, 5, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.current


class LongLocalWaiterTests(unittest.TestCase):
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='long-local-waiter-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = self.root / 'state/long_local_500'
        self.queue = self.root / 'configs/third_pass_batch07.json'
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(redirect_stdout(io.StringIO()))
        for field, value in [('ROOT', self.root), ('STATE', self.state), ('QUEUE', self.queue), ('datetime', FixedTime)]:
            stack.enter_context(patch.object(waiter, field, value))
        stack.enter_context(patch.object(FixedTime, 'current', datetime(2026, 10, 2, 17, 5, tzinfo=timezone.utc)))
        self.control = {'id': 'v2_realmlp_cat_raw_aux', 'family': 'realmlp_cat', 'seed': 20261005,
                        'max_rounds': 4, 'timeout_seconds': 1200, 'original_aux': True}
        self.candidate = {**self.control, 'id': 'v3_realmlp_cat_raw_aux_e500', 'max_rounds': 500, 'timeout_seconds': 53400}
        base_path = self.root / 'configs/third_pass_batch05.json'
        self.write(base_path, {'runs': [self.control]})
        source = self.root / 'scripts/source.py'; source.parent.mkdir(); source.write_text('# synthetic')
        self.protocol = {'campaign_id': 'third_pass_batch07', 'queue_path': self.queue.relative_to(self.root).as_posix(),
             'not_before_utc': '2026-10-02T17:05:00Z', 'latest_launch_utc': '2026-10-02T17:35:00Z',
             'finalize_utc': '2026-10-03T07:55:00Z', 'deadline_utc': '2026-10-03T13:05:00Z',
             'refit_timeout_seconds': 14400, 'verify_reserve_seconds': 3600, 'automatic_submission': False,
             'control_run': self.control, 'candidate_run': self.candidate, 'primary_timeout_seconds': 53400,
             'base_config_path': base_path.relative_to(self.root).as_posix(), 'base_config_sha256': waiter.sha(base_path),
             'prior_run_ids': [self.control['id']], 'source_hashes': {'scripts/source.py': waiter.sha(source)},
             'readiness_path': 'state/long_local_500/prior_campaign_ready.json', 'minimum_available_ram_gib': 3.5}
        protocol_path = self.root / 'configs/long_local_500.json'; self.write(protocol_path, self.protocol)
        self.config = {k: self.protocol[k] for k in ['campaign_id', 'finalize_utc', 'deadline_utc',
                         'refit_timeout_seconds', 'verify_reserve_seconds', 'automatic_submission']}
        self.config.update(runs=[self.control, self.candidate], max_retries=0,
             stop_new_runs_utc=self.protocol['latest_launch_utc'],
             paired_duration_controls={self.candidate['id']: self.control['id']},
             paired_duration_horizons={self.candidate['id']: 500},
             long_local_500_registration={'protocol_path': 'configs/long_local_500.json',
                                          'protocol_sha256': waiter.sha(protocol_path)})
        self.write(self.queue, self.config)
        stack.enter_context(patch.object(waiter.supervisor, 'load_config', side_effect=waiter.supervisor.read_json))
        self.write(self.root / 'artifacts/runs' / self.control['id'] / 'result.json', {'run': self.control})
        self.receipt = {'status': 'ready', 'queue_sha256': waiter.sha(self.queue),
             'resolved_utc': '2026-10-02T17:00:01Z', 'local_release_complete': True,
             'submission_attempt_resolved': True, 'all_owned_cloud_jobs_terminal': True, 'evidence': {}}
        for role, value in [('release', {'all_row_raw_inference': True, 'independent_blend_recomputation': True, 'rows': 299844}),
                            ('submission', {'status': 'resolved_no_additional_submission'}),
                            ('cloud', {'status': 'KernelStatus.COMPLETE'})]:
            path = self.root / 'evidence' / f'{role}.json'; self.write(path, value)
            self.receipt['evidence'][role] = {path.relative_to(self.root).as_posix(): waiter.sha(path)}
        self.write(self.state / 'prior_campaign_ready.json', self.receipt)

    def test_registration_checks_exact_recipe_and_pinned_inputs(self):
        self.assertEqual(waiter.registration(), (self.config, self.protocol))
        self.config['runs'][-1]['seed'] = 99; self.write(self.queue, self.config)
        with self.assertRaisesRegex(ValueError, 'Queue changed'):
            waiter.registration()
        self.config['runs'][-1]['seed'] = 20261005; self.write(self.queue, self.config)
        (self.root / 'scripts/source.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            waiter.registration()

    def test_readiness_requires_all_resolution_roles_and_checks_hashes(self):
        self.assertIsNone(waiter.readiness(self.protocol))
        self.receipt['all_owned_cloud_jobs_terminal'] = False
        self.write(self.state / 'prior_campaign_ready.json', self.receipt)
        self.assertEqual(waiter.readiness(self.protocol), 'waiting_for_all_owned_cloud_jobs_terminal')
        self.receipt['all_owned_cloud_jobs_terminal'] = True
        self.write(self.state / 'prior_campaign_ready.json', self.receipt)
        (self.root / 'evidence/cloud.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Changed cloud'):
            waiter.readiness(self.protocol)

    def test_running_cloud_status_and_incomplete_prior_prevent_launch(self):
        path = self.root / 'evidence/cloud.json'; self.write(path, {'status': 'KernelStatus.RUNNING'})
        self.receipt['evidence']['cloud']['evidence/cloud.json'] = waiter.sha(path)
        self.write(self.state / 'prior_campaign_ready.json', self.receipt)
        self.assertEqual(waiter.readiness(self.protocol), 'waiting_for_cloud_terminal_evidence')
        result = self.root / 'artifacts/runs' / self.control['id'] / 'result.json'
        result.unlink()
        self.assertEqual(waiter.incomplete_prior(self.config, self.protocol), [self.control['id']])

    def test_not_before_and_expired_window_do_not_launch(self):
        with patch.object(FixedTime, 'current', datetime(2026, 10, 2, 17, 4, tzinfo=timezone.utc)), \
                patch.object(waiter.time, 'sleep', side_effect=StopIteration), \
                patch.object(waiter, 'launch') as launch, patch('sys.argv', ['waiter']):
            with self.assertRaises(StopIteration): waiter.main()
            launch.assert_not_called()
        self.assertEqual(waiter.supervisor.read_json(self.state / 'waiter_state.json')['status'], 'waiting_for_not_before')
        with patch.object(FixedTime, 'current', datetime(2026, 10, 2, 17, 35, tzinfo=timezone.utc)), \
                patch.object(waiter, 'launch') as launch, patch('sys.argv', ['waiter']):
            self.assertEqual(waiter.main(), 1); launch.assert_not_called()

    def test_busy_owned_process_and_low_ram_do_not_launch(self):
        for workers, available, expected in [([{'pid': 123}], 8 * 2**30, 'waiting_for_project_workers'),
                                               ([], 2 * 2**30, 'waiting_for_available_memory')]:
            with patch.object(waiter, 'project_workers', return_value=workers), \
                    patch.object(waiter.psutil, 'virtual_memory', return_value=Mock(available=available)), \
                    patch.object(waiter.time, 'sleep', side_effect=StopIteration), \
                    patch.object(waiter, 'launch') as launch, patch('sys.argv', ['waiter']):
                with self.assertRaises(StopIteration): waiter.main()
                launch.assert_not_called()
            self.assertEqual(waiter.supervisor.read_json(self.state / 'waiter_state.json')['status'], expected)

    def test_launch_intent_precedes_spawn_and_blocks_duplicate(self):
        def spawn(command, **kwargs):
            self.assertEqual(waiter.supervisor.read_json(self.state / 'launch_intent.json')['command'], command)
            self.assertEqual(command[-2:], ['--config', str(self.queue)])
            self.assertNotIn('shell', kwargs)
            return Mock(pid=12345)
        with patch.object(waiter.subprocess, 'Popen', side_effect=spawn) as popen, \
                patch.object(waiter.psutil, 'Process', return_value=Mock(create_time=lambda: 1.25)):
            receipt = waiter.launch(self.config, self.protocol)
            self.assertEqual(receipt['pid'], 12345)
            self.assertEqual(receipt['create_time'], 1.25)
            with self.assertRaisesRegex(RuntimeError, 'Existing launch intent'):
                waiter.launch(self.config, self.protocol)
            popen.assert_called_once()


if __name__ == '__main__':
    unittest.main()
