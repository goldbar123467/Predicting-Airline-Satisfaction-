"""Wait for the registered local-only 500-epoch campaign; launch at most once.

This process never trains, scores, submits, calls cloud APIs, or stops another
process. The established supervisor owns training, recovery, and final release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import supervisor

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'state/long_local_500'
QUEUE = ROOT / 'configs/third_pass_batch07.json'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local_path(relative: str) -> Path:
    value = Path(relative)
    path = (ROOT / value).resolve()
    if value.is_absolute() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Evidence/source path must remain project-relative')
    return path


def registration() -> tuple[dict, dict]:
    config = supervisor.load_config(QUEUE)
    binding = config['long_local_500_registration']
    protocol_path = local_path(binding['protocol_path'])
    if sha(protocol_path) != binding['protocol_sha256']:
        raise ValueError('Long-local protocol changed')
    protocol = supervisor.read_json(protocol_path)
    if (config['campaign_id'] != 'third_pass_batch07' or protocol['campaign_id'] != config['campaign_id']
            or local_path(protocol['queue_path']) != QUEUE.resolve()
            or config.get('automatic_submission') is not False
            or protocol.get('automatic_submission') is not False):
        raise ValueError('Wrong campaign namespace/submission authority')
    for field in ['finalize_utc', 'deadline_utc', 'refit_timeout_seconds', 'verify_reserve_seconds']:
        if config.get(field) != protocol.get(field):
            raise ValueError(f'Queue differs from registered {field}')
    if config['stop_new_runs_utc'] != protocol['latest_launch_utc']:
        raise ValueError('Latest launch changed')
    control, candidate = protocol['control_run'], protocol['candidate_run']
    ignored = {'id', 'max_rounds', 'timeout_seconds'}
    if (candidate['id'] != 'v3_realmlp_cat_raw_aux_e500' or candidate['max_rounds'] != 500
            or control['id'] != 'v2_realmlp_cat_raw_aux' or control['max_rounds'] != 4
            or candidate['timeout_seconds'] != protocol['primary_timeout_seconds']
            or {k: v for k, v in candidate.items() if k not in ignored}
            != {k: v for k, v in control.items() if k not in ignored}):
        raise ValueError('The one registered horizon-only contrast changed')
    base_path = local_path(protocol['base_config_path'])
    if sha(base_path) != protocol['base_config_sha256']:
        raise ValueError('Prior campaign registration changed')
    base = supervisor.read_json(base_path)
    if config['runs'] != [*base['runs'], candidate] or protocol['prior_run_ids'] != [r['id'] for r in base['runs']]:
        raise ValueError('Queue changed prior recipes or added another experiment')
    if (config.get('paired_duration_controls', {}).get(candidate['id']) != control['id']
            or config.get('paired_duration_horizons') != {candidate['id']: 500}
            or config.get('max_retries') != 0):
        raise ValueError('Registered paired gate or no-retry policy changed')
    for name, digest in protocol['source_hashes'].items():
        if sha(local_path(name)) != digest:
            raise ValueError(f'Registered training input/source changed: {name}')
    return config, protocol


def readiness(protocol: dict) -> str | None:
    path = local_path(protocol['readiness_path'])
    if not path.exists():
        return 'waiting_for_prior_campaign_resolution'
    record = supervisor.read_json(path)
    if record.get('status') != 'ready' or record.get('queue_sha256') != sha(QUEUE):
        raise ValueError('Readiness receipt does not bind this immutable queue')
    for field in ['local_release_complete', 'submission_attempt_resolved', 'all_owned_cloud_jobs_terminal']:
        if record.get(field) is not True:
            return f'waiting_for_{field}'
    supervisor.parse_utc(record['resolved_utc'])
    for role in ['release', 'submission', 'cloud']:
        mapping = record.get('evidence', {}).get(role)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f'Missing {role} resolution evidence')
        for name, digest in mapping.items():
            artifact = local_path(name)
            if artifact.suffix != '.json' or sha(artifact) != digest:
                raise ValueError(f'Changed {role} resolution evidence')
            evidence = supervisor.read_json(artifact)
            if role == 'release' and (evidence.get('all_row_raw_inference') is not True
                    or evidence.get('independent_blend_recomputation') is not True
                    or evidence.get('rows') != 299844):
                raise ValueError('Prior local release lacks complete raw verification')
            if role == 'cloud':
                terminal = str(evidence.get('status', '')).upper().split('.')[-1]
                if terminal not in {'COMPLETE', 'COMPLETED', 'ERROR', 'CANCELLED', 'CANCELED'}:
                    return 'waiting_for_cloud_terminal_evidence'
    return None


def incomplete_prior(config: dict, protocol: dict) -> list[str]:
    expected = {r['id']: r for r in config['runs']}
    missing = []
    for name in protocol['prior_run_ids']:
        path = ROOT / 'artifacts/runs' / name / 'result.json'
        if not path.exists():
            missing.append(name)
        elif supervisor.read_json(path).get('run') != expected[name]:
            raise ValueError(f'Completed prior recipe changed: {name}')
    return missing


def project_workers() -> list[dict]:
    """Conservatively wait for any other owned project Python process."""
    ignored = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    workers = []
    for process in psutil.process_iter(['name']):
        if process.pid in ignored or 'python' not in (process.info['name'] or '').lower():
            continue
        try:
            command = process.cmdline()
            cwd = Path(process.cwd()).resolve()
            belongs = cwd == ROOT.resolve() or any(str(ROOT).casefold() in arg.casefold() for arg in command)
            if belongs:
                workers.append({'pid': process.pid, 'create_time': process.create_time(), 'command': command})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            workers.append({'pid': process.pid, 'inspection_error': 'access_denied'})
    return workers


def launch(config: dict, protocol: dict) -> dict:
    intent_path = STATE / 'launch_intent.json'
    if intent_path.exists():
        raise RuntimeError('Existing launch intent; reconcile it instead of dispatching again')
    command = [str(ROOT / '.venv/Scripts/python.exe'), '-u', str(ROOT / 'scripts/supervisor.py'),
               '--config', str(QUEUE)]
    directory = ROOT / 'logs/third_pass_batch07/supervisor'
    directory.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    record = {'status': 'intent', 'utc': supervisor.stamp(), 'queue_sha256': sha(QUEUE),
              'protocol_sha256': config['long_local_500_registration']['protocol_sha256'],
              'readiness_sha256': sha(local_path(protocol['readiness_path'])), 'command': command,
              'stdout': str(directory / f'long500-{tag}.stdout.log'),
              'stderr': str(directory / f'long500-{tag}.stderr.log')}
    supervisor.atomic_json(intent_path, record)
    with Path(record['stdout']).open('wb') as stdout, Path(record['stderr']).open('wb') as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    record.update(status='launched', pid=child.pid, create_time=psutil.Process(child.pid).create_time())
    supervisor.atomic_json(STATE / 'launch_receipt.json', record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config, protocol = registration()
    if args.dry_run:
        print({'queue': str(QUEUE), 'queue_sha256': sha(QUEUE), 'not_before_utc': protocol['not_before_utc'],
               'latest_launch_utc': protocol['latest_launch_utc'], 'automatic_submission': False})
        return 0
    with supervisor.lifetime_lock(STATE / 'waiter.lock'):
        if (STATE / 'launch_intent.json').exists():
            raise RuntimeError('Existing launch intent; inspect receipt and supervisor state')
        previous_status = None
        while True:
            config, protocol = registration()
            now = datetime.now(timezone.utc)
            if (STATE / 'STOP').exists() or (ROOT / 'state/third_pass_batch07/STOP').exists():
                reason = 'stopped_before_launch'
            elif now >= supervisor.parse_utc(protocol['latest_launch_utc']):
                reason = 'launch_window_expired'
            elif now < supervisor.parse_utc(protocol['not_before_utc']):
                reason = 'waiting_for_not_before'
            else:
                reason = readiness(protocol)
                missing = incomplete_prior(config, protocol) if reason is None else []
                workers = project_workers() if reason is None and not missing else []
                if missing:
                    reason = 'waiting_for_prior_results:' + ','.join(missing)
                elif workers:
                    reason = 'waiting_for_project_workers'
                elif reason is None and psutil.virtual_memory().available / 2**30 < protocol['minimum_available_ram_gib']:
                    reason = 'waiting_for_available_memory'
                elif reason is None:
                    # Recheck resolution immediately before the single dispatch.
                    if readiness(protocol) is not None or project_workers():
                        continue
                    record = launch(config, protocol)
                    supervisor.atomic_json(STATE / 'waiter_state.json', record)
                    print(record, flush=True)
                    return 0
            record = {'status': reason, 'utc': now.isoformat(), 'pid': os.getpid(),
                      'create_time': psutil.Process().create_time(), 'queue_sha256': sha(QUEUE)}
            supervisor.atomic_json(STATE / 'waiter_state.json', record)
            if reason != previous_status:
                print(record, flush=True)
                previous_status = reason
            if reason in {'stopped_before_launch', 'launch_window_expired'}:
                return 1
            time.sleep(30)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        supervisor.atomic_json(STATE / 'waiter_state.json', {'status': 'needs_attention',
                               'utc': supervisor.stamp(), 'error': f'{type(error).__name__}: {error}'})
        raise
