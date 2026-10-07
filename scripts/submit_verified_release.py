"""Submit one verified development release under the user's explicit loop authorization."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import msvcrt
from datetime import datetime, timezone
from pathlib import Path
import re
from common import ROOT, load_config, sha256, atomic_json
from kaggle_cloud_control import api_client

COMPETITION = 'playground-series-s6e10'
V2_BASELINE_SHA = 'd75c7eba29f8ed9cb3e9713e4f43b102a5d48dacdcfbba616f9c331a28bb0dc1'
BEST_BLEND_AUTH_SHA = '530eab7d62064a8e6ff988db790e18411f6ffaf86f8c996a3a9e63f622b64104'


def submission_window(campaign, current_time):
    """The new one-release permission never broadens other campaigns or gates."""
    if current_time < datetime(2026, 10, 2, 17, tzinfo=timezone.utc):
        return None
    path = ROOT / 'state/kaggle_best_blend_authorization.json'
    if not path.exists() or sha256(path) != BEST_BLEND_AUTH_SHA:
        raise RuntimeError('Submission deadline passed without exact renewed authorization')
    permission = load_config(path)
    if (permission.get('status') != 'authorized'
            or permission.get('campaign_ids') != ['third_pass_batch07']
            or campaign not in permission['campaign_ids']
            or permission.get('maximum_new_submissions') != 1
            or permission.get('require_existing_gain_gate') is not True
            or permission.get('require_existing_full_verification_contract') is not True
            or not datetime.fromisoformat(permission['requested_utc']) <= current_time < datetime.fromisoformat(permission['expires_utc'])):
        raise RuntimeError('This campaign or time is outside renewed best-blend permission')
    for receipt in (ROOT / 'artifacts/kaggle').glob('submission_*.json'):
        if load_config(receipt).get('renewed_authorization_sha256') == BEST_BLEND_AUTH_SHA:
            raise RuntimeError('The one renewed submission attempt already exists; reconcile it')
    return BEST_BLEND_AUTH_SHA

def verification_contract(campaign: str):
    if re.fullmatch(r'third_pass(?:_batch[0-9]{2})?', campaign) is None:
        raise ValueError('Only the authorized isolated third-pass releases are accepted')
    out = ROOT / 'artifacts' / campaign
    verified = load_config(out / 'verification.json')
    provenance = load_config(out / 'final/release_provenance.json')
    selection_path = out / 'blend/frozen.json'
    selection = load_config(selection_path)
    submitted = out / 'final/submission.csv'
    checks = {'submission_sha256': submitted, 'manifest_sha256': out / 'final/manifest.json',
              'selection_sha256': selection_path, 'verification_sha256': out / 'verification.json',
              'report_sha256': ROOT / (campaign.upper() + '_REPORT.md')}
    for field, path in checks.items():
        if sha256(path) != provenance[field]:
            raise ValueError(f'Changed verified release: {field}')
    if (sha256(submitted) != verified['sha256'] or verified['rows'] != 299844
            or not verified['all_row_raw_inference'] or not verified['independent_blend_recomputation']
            or verified['audit_evaluated'] or selection['audit_evaluated']):
        raise ValueError('Missing required verification contract')
    for field in ['native_hashes', 'source_hashes']:
        if not isinstance(provenance.get(field), dict) or not provenance[field]:
            raise ValueError(f'Missing verified provenance: {field}')
    for path, digest in provenance['native_hashes'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f'Native artifact changed: {path}')
    for path, digest in provenance['source_hashes'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f'Archived source changed: {path}')
    return submitted, selection_path, selection

def gain_gate(selection, previous):
    delta = [a-b for a,b in zip(selection['fold_auc'],previous['fold_auc'],strict=True)]
    return (selection['oof_auc']-previous['oof_auc'] >= .00001
            and sum(delta)/len(delta) >= .00001 and min(delta) >= -.00002)

def _main_locked():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True)
    args=parser.parse_args()
    path, selection_path, selection = verification_contract(args.campaign)
    digest=sha256(path);receipt_path=ROOT/'artifacts/kaggle'/f'submission_{digest[:16]}.json'
    api=api_client();submissions=api.competition_submissions(COMPETITION,page_size=100)
    if receipt_path.exists():
        receipt=load_config(receipt_path)
        matches=[s for s in submissions if s.ref==receipt.get('response',{}).get('ref') or digest[:12] in str(s.description)]
        if len(matches)!=1:
            raise RuntimeError('Existing intent but remote identity uncertain; inspect, never blindly resubmit')
        receipt.setdefault('response', {})['ref'] = matches[0].ref
        receipt.update(status='reconciled_remote',processing=matches[0].to_dict(),checked_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(receipt_path,receipt);print(receipt);return
    baseline_path=ROOT/'artifacts/second_pass/blend/frozen.json'
    if sha256(baseline_path) != V2_BASELINE_SHA:
        raise ValueError('Immutable v2 baseline changed before submission')
    previous=load_config(baseline_path)
    for receipt_file in (ROOT/'artifacts/kaggle').glob('submission_*.json'):
        record=load_config(receipt_file)
        if record.get('selection_path') and record.get('response',{}).get('ref'):
            prior_path=(ROOT/record['selection_path']).resolve()
            if not prior_path.is_relative_to((ROOT/'artifacts').resolve()) or sha256(prior_path)!=record.get('selection_sha256'):
                raise ValueError('Prior submitted development selection changed')
            prior=load_config(prior_path)
            if prior['oof_auc']>previous['oof_auc']:previous=prior
    if not gain_gate(selection,previous):
        print({'status':'not_submitted','reason':'No qualifying incremental development gain','oof_auc':selection['oof_auc'],'prior_oof_auc':previous['oof_auc']});return
    if any(digest[:12] in str(s.description) for s in submissions):
        raise RuntimeError('Remote hash already submitted without expected local receipt; inspect instead of duplicating')
    today=datetime.now(timezone.utc).date()
    used=sum(s.date.date()==today for s in submissions)
    # Competition-specific Rules section2.2, inspected in research/competition_evidence.md.
    if used>=10:
        raise RuntimeError('Competition daily10-submission allowance reached; retain verified release')
    renewed_authorization = submission_window(args.campaign, datetime.now(timezone.utc))
    description=f'Local/cloud verified {args.campaign}; development OOF {selection["oof_auc"]:.8f}; sha256 {digest[:12]}'
    receipt={'competition':COMPETITION,'file':str(path),'sha256':digest,'selection_path':selection_path.relative_to(ROOT).as_posix(),
             'selection_sha256':sha256(selection_path),
             'description':description,'requested_utc':datetime.now(timezone.utc).isoformat(),
             'status':'intent_recorded','authorization':'User requested continuous local and free Kaggle cloud training and submissions until17UTC October2,2026.',
             'daily_submissions_before':used,'development_oof_auc':selection['oof_auc']}
    if renewed_authorization:
        receipt['renewed_authorization_sha256'] = renewed_authorization
        receipt['authorization'] = 'User renewed permission for one best qualifying verified500epoch campaign release; expires October3 12:47:41UTC.'
    atomic_json(receipt_path,receipt)
    response=api.competition_submit(str(path),description,COMPETITION,quiet=True)
    receipt.update(status='submitted_response',response=response.to_dict(),response_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(receipt_path,receipt);print(receipt)

@contextmanager
def submission_lock():
    path=ROOT/'state/kaggle_submission.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:
            raise RuntimeError('Another competition submission transaction is active') from exc
        try:
            yield
        finally:
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)

def main():
    with submission_lock():
        _main_locked()

if __name__=='__main__':main()
