"""Exact authorization for the isolated, user-requested overnight duration test."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = '42a965cd97b3a584b39cf74810f349baaf168ba0d81e6fee446ed66d21fac318'
QUEUE_SHA = '9cbfcdf4df6fc79cc1c3338d3589fb5ff182c58b497b1e0af7e4b4f7f8f2b9ce'
CANDIDATE = 'v3_realmlp_cat_raw_aux_e500'
START_NOW_SHA = 'ce198f65392c1b1a62c5b4e906d1e9c5e658cd979e7d21b7dffb5d3702dfddce'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def registered_policy(config):
    needed = (config.get('campaign_id') == 'third_pass_batch07'
              or 'long_local_500_registration' in config
              or any(run.get('id') == CANDIDATE for run in config.get('runs', [])))
    if not needed:
        return None
    protocol_path = ROOT / 'configs/long_local_500.json'
    queue_path = ROOT / 'configs/third_pass_batch07.json'
    if digest(protocol_path) != PROTOCOL_SHA or digest(queue_path) != QUEUE_SHA:
        raise ValueError('Registered overnight protocol or queue changed')
    protocol = json.loads(protocol_path.read_text())
    if config != json.loads(queue_path.read_text()):
        raise ValueError('Overnight resource allowance requires the exact registered queue')
    for name, expected in protocol['source_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Overnight training input or source changed: {name}')
    if digest(ROOT / protocol['base_config_path']) != protocol['base_config_sha256']:
        raise ValueError('Overnight baseline configuration changed')
    if digest(ROOT / 'artifacts/runs' / protocol['control_id'] / 'result.json') != protocol['control_result_sha256']:
        raise ValueError('Overnight matched control changed')
    return protocol


def operational_timing(config):
    """Honor the user's start-now amendment without changing a training recipe."""
    fields = ('stop_new_runs_utc', 'finalize_utc', 'deadline_utc')
    timing = {key: config[key] for key in fields}
    protocol = registered_policy(config)
    if protocol is None:
        return timing
    timing['not_before_utc'] = protocol['not_before_utc']
    path = ROOT / 'state/long_local_500/start_now_authorization.json'
    if not path.exists():
        receipt = ROOT / 'state/long_local_500/launch_receipt.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('authorization_sha256'):
            raise ValueError('Dispatched start-now amendment is missing')
        return timing
    if digest(path) != START_NOW_SHA:
        raise ValueError('Start-now authorization changed')
    amendment = json.loads(path.read_text())
    if (amendment.get('status') != 'authorized'
            or amendment.get('campaign_id') != config['campaign_id']
            or amendment.get('queue_sha256') != QUEUE_SHA
            or amendment.get('protocol_sha256') != PROTOCOL_SHA
            or amendment.get('readiness_sha256') != digest(ROOT / protocol['readiness_path'])):
        raise ValueError('Start-now authorization does not bind the ready campaign')
    start = datetime.fromisoformat(amendment['not_before_utc'])
    original = datetime.fromisoformat(protocol['not_before_utc'])
    advance = original - start
    if (start.tzinfo is None or amendment['authorized_utc'] != amendment['not_before_utc']
            or not timedelta(0) < advance <= timedelta(hours=1)):
        raise ValueError('Invalid bounded start-now timing amendment')
    timing['not_before_utc'] = start.isoformat()
    for key in fields:
        timing[key] = (datetime.fromisoformat(config[key]) - advance).isoformat()
    if (datetime.fromisoformat(timing['deadline_utc']) - start).total_seconds() > protocol['maximum_campaign_seconds']:
        raise ValueError('Start-now amendment exceeds campaign ceiling')
    return timing
