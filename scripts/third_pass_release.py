"""Development-only third-pass selection and an isolated, reproducible release.

The previously evaluated audit is never scored here. Model selection reads labels
only from saved development OOF files after checking their frozen split identity.
Full-data fitting is allowed only after the selection has been frozen.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import (ROOT, TARGET, CAT, atomic_csv, atomic_json, features, fit_model,
                    load_config, load_data, load_model, predict, sha256)
from cloud_feature_compat import SOURCE_VERSIONS, cloud_numeric_category_compat
from long_local_500_policy import registered_policy as long_local_policy

CLOUD_FEATURE_COMPAT = Path(__file__).with_name('cloud_feature_compat.py')

OUT = ROOT / 'artifacts/third_pass'
BLEND = OUT / 'blend'
FINAL = OUT / 'final'
REPORT = ROOT / 'THIRD_PASS_REPORT.md'
ANCHOR_MANIFEST = ROOT / 'artifacts/second_pass/final/manifest.json'
ANCHOR_MANIFEST_SHA = '13b01e43cd71439a68e27e0c34f77794963f3406817e4703ffc0ea4c2aca3356'
BASELINE = ROOT / 'artifacts/second_pass/blend/frozen.json'
BASELINE_SHA = 'd75c7eba29f8ed9cb3e9713e4f43b102a5d48dacdcfbba616f9c331a28bb0dc1'
V1_PROVENANCE_SHA = '11297e9b248e76e659cede2d3e33f65119aa97a096d94df4a1dfc145f935111b'
V2_PROVENANCE_SHA = '1becb75a1613d65acf7bbe6540c037eea84a1bb2b324cf75308c9307d9cdce01'
CLOUD_DEVELOPMENT_ROWS = 629671
LGB_CLOUD_PROTOCOL_SHA = '80aa76eb30630dfad0512173cbf508cfa6ec16303f19de9a010d5a09cbfec6a9'
# Filled only after the isolated package passes preparation and review, before dispatch.
LGB_CLOUD_RUNTIME_SHA = '3927d6bf8b61e495187a66ce9e747bbf93ecead554d457cadca25403e95ac029'
LGB_CLOUD_BUNDLE_SHA = 'e4c4357fc1f2eab731b8c6fc53c78420abcc3cea399873ae233f6f034b5db6cf'
LONG_CLOUD_PROTOCOL_SHA = '73338bad3caf5241da5d3b520cece06459af3d9470500cbbb03ae641e6653266'
LONG_CLOUD_RUNTIME_SHA = '18cc3d3b4086e75611ea99dacc9218639b84b9bc285e7c0f47d3a83f4dd40bb1'
LONG_CLOUD_BUNDLE_SHA = '2e326ea44c91afae5602a7cd00d6bbb023bc5b2d12719510c4086f78fcebc1e7'
SHRINKAGE_PROTOCOL_SHA = 'ea5ce787253b52eabb8353bf5584aafbc3f736e139f88ecdc332e8e9da3d7399'
SHRINKAGE_RUNTIME_SHA = '1d6c137a56cea59ad1b9f52e715cd4259dc80c2bc45730dec28ada6602df7c4d'
SHRINKAGE_BUNDLE_SHA = 'de57c2d803375cfd2882593f8543714e666c06805d9eb5b77ca1e0a08c24d4eb'
SHRINKAGE_ID = 'v3_cloud_lgb_route_teacher_aux_probability_lr01'
REGISTERED_POLICY = {'minimum_pooled_gain': .00001, 'minimum_mean_fold_gain': .00001,
                     'maximum_fold_regression': .00002, 'alpha_grid': [.05, .1, .2, .3],
                     'max_additions': 2, 'new_candidate_prefix': 'v3_'}


def now():
    return datetime.now(timezone.utc).isoformat()


def probabilities(path, ids):
    frame = pd.read_parquet(path, columns=['id', 'prediction'])
    if len(frame) != len(ids) or not frame.id.is_unique:
        raise ValueError(f'Invalid prediction identifiers: {path}')
    order = pd.Index(frame.id).get_indexer(ids)
    if (order < 0).any():
        raise ValueError(f'Missing prediction identifiers: {path}')
    result = frame.prediction.to_numpy(dtype=np.float64)[order]
    if not np.isfinite(result).all() or not ((result >= 0) & (result <= 1)).all():
        raise ValueError(f'Invalid probabilities: {path}')
    return result


def scores(y, probability, folds):
    per_fold = [float(roc_auc_score(y[folds == k], probability[folds == k])) for k in range(3)]
    return {'pooled': float(roc_auc_score(y, probability)), 'folds': per_fold,
            'mean_fold': float(np.mean(per_fold))}


def eligible_change(before, after, policy):
    delta = np.asarray(after['folds']) - before['folds']
    return (after['pooled'] - before['pooled'] >= policy['minimum_pooled_gain']
            and after['mean_fold'] - before['mean_fold'] >= policy['minimum_mean_fold_gain']
            and delta.min() >= -policy['maximum_fold_regression'])


def duration_comparison(run, control, candidate_scores, control_scores, horizon=60):
    """Compare the registered duration contrast with its exact 4-epoch recipe."""
    ignored = {'id', 'max_rounds', 'timeout_seconds'}
    if (horizon not in {60, 500} or run.get('family') != 'realmlp_cat' or run.get('max_rounds') != horizon
            or control.get('max_rounds') != 4
            or {k: v for k, v in run.items() if k not in ignored}
            != {k: v for k, v in control.items() if k not in ignored}):
        raise ValueError('Duration comparison changed more than its registered horizon')
    return (candidate_scores['pooled'] >= control_scores['pooled']
            and candidate_scores['mean_fold'] >= control_scores['mean_fold']
            and min(np.asarray(candidate_scores['folds']) - control_scores['folds']) >= -.00002)


def anchored_search(y, folds, anchor, candidates, policy, trace=None):
    """At most three coarse convex additions; deterministic ordering breaks ties."""
    current = anchor.copy()
    current_score = scores(y, current, folds)
    anchor_score = current_score
    weights = {'anchor': 1.0}
    history = []
    unused = list(candidates)
    for _ in range(policy['max_additions']):
        best = None
        for name in unused:
            for alpha in sorted(policy['alpha_grid']):
                proposed = (1 - alpha) * current + alpha * candidates[name]
                value = scores(y, proposed, folds)
                eligible = (eligible_change(current_score, value, policy)
                            and eligible_change(anchor_score, value, policy))
                if trace is not None:
                    trace.append({'step': len(history), 'candidate': name, 'alpha': alpha,
                                  'scores': value, 'eligible': bool(eligible)})
                if eligible:
                    if (best is None or value['pooled'] > best[3]['pooled'] + 1e-12
                            or (abs(value['pooled'] - best[3]['pooled']) <= 1e-12 and alpha < best[1])):
                        best = (name, alpha, proposed, value)
        if best is None:
            break
        name, alpha, current, value = best
        history.append({'candidate': name, 'alpha': alpha, 'before': current_score, 'after': value})
        weights = {key: weight * (1 - alpha) for key, weight in weights.items()}
        weights[name] = weights.get(name, 0.0) + alpha
        current_score = value
        unused.remove(name)
    return weights, current, current_score, history


def checked_run(run, split_hash, ids, y, folds):
    directory = ROOT / 'artifacts/runs' / run['id']
    result = load_config(directory / 'result.json')
    if result['run'] != run or result['split_hash'] != split_hash:
        raise ValueError(f'Changed completed run contract: {run["id"]}')
    for name in ['oof.parquet', 'test.parquet']:
        if sha256(directory / name) != result['artifacts'][name]:
            raise ValueError(f'Changed completed predictions: {run["id"]}/{name}')
    frame = pd.read_parquet(directory / 'oof.parquet')
    if not (np.array_equal(frame.id, ids) and np.array_equal(frame[TARGET], y)
            and np.array_equal(frame.fold, folds)):
        raise ValueError(f'Development identity mismatch: {run["id"]}')
    return probabilities(directory / 'oof.parquet', ids)


def checked_cloud_feature_compat(run, receipt):
    """Bind deployment semantics to the exact adapter audited on held-out rows."""
    if run.get('execution_backend') != 'kaggle' or run.get('categorical_twins') is not True:
        return
    archive = ROOT / 'artifacts/runs' / run['id'] / 'source/cloud_feature_compat.py'
    digest = sha256(CLOUD_FEATURE_COMPAT)
    if (receipt.get('source_versions') != SOURCE_VERSIONS
            or receipt.get('feature_compat_sha256') != digest
            or not isinstance(receipt.get('source_hashes'), dict)
            or receipt['source_hashes'].get(archive.relative_to(ROOT).as_posix()) != digest
            or not archive.is_file() or sha256(archive) != digest):
        raise ValueError('Cloud feature compatibility source/version proof mismatch')


def checked_cloud_import(run, split_hash):
    directory = ROOT / 'artifacts/runs' / run['id']
    receipt = load_config(directory / 'cloud_import_verification.json')
    expected_image = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
    expected_bundle = 'a5eee31904c7700357a32f174130195f044b47d545dd77cd97579269d741bd96'
    expected_runtime = 'b534194f99c10545a50ddd52a2a7e91bd60b2e39aefbb6df1bc4de38ba169235'
    if run['id'] == 'v3_cloud_realmlp_raw_aux_e60':
        protocol_path = ROOT / 'configs/third_pass_long_cloud.json'
        if sha256(protocol_path) != LONG_CLOUD_PROTOCOL_SHA:
            raise ValueError('Changed registered long cloud protocol')
        protocol = load_config(protocol_path)
        if (not LONG_CLOUD_RUNTIME_SHA or not LONG_CLOUD_BUNDLE_SHA
                or run not in protocol['runs']
                or receipt.get('protocol_id') != protocol['protocol_id']
                or receipt.get('protocol_sha256') != LONG_CLOUD_PROTOCOL_SHA
                or receipt.get('reused_control') != protocol['control_reuse']):
            raise ValueError('Long cloud package lacks reviewed pins or exact recipe')
        expected_image = protocol['execution_image']
        expected_bundle, expected_runtime = LONG_CLOUD_BUNDLE_SHA, LONG_CLOUD_RUNTIME_SHA
    if run['family'] == 'lightgbm' or run['id'].startswith('v3_cloud_lgb_'):
        shrinkage = run['id'] == SHRINKAGE_ID
        protocol_path = ROOT / ('configs/third_pass_lgb_shrinkage.json' if shrinkage else 'configs/third_pass_lgb_cloud.json')
        protocol_sha = SHRINKAGE_PROTOCOL_SHA if shrinkage else LGB_CLOUD_PROTOCOL_SHA
        runtime_sha = SHRINKAGE_RUNTIME_SHA if shrinkage else LGB_CLOUD_RUNTIME_SHA
        bundle_sha = SHRINKAGE_BUNDLE_SHA if shrinkage else LGB_CLOUD_BUNDLE_SHA
        if sha256(protocol_path) != protocol_sha:
            raise ValueError('Changed registered LightGBM cloud protocol')
        protocol = load_config(protocol_path)
        if (not runtime_sha or not bundle_sha
                or run not in protocol['runs']
                or receipt.get('protocol_id') != protocol['protocol_id']
                or receipt.get('protocol_sha256') != protocol_sha
                or receipt.get('lightgbm_version') != '4.7.0'
                or receipt.get('execution_threads') != 4
                or receipt.get('training_device') != 'cpu'
                or receipt.get('all_test_fold_native_inference') is not True):
            raise ValueError('LightGBM cloud recipe or native return contract differs')
        if shrinkage and receipt.get('reused_control') != protocol['reused_control']:
            raise ValueError('LightGBM shrinkage control provenance differs')
        expected_image = protocol['execution_image']
        expected_bundle, expected_runtime = bundle_sha, runtime_sha
    if (receipt.get('result_sha256') != sha256(directory / 'result.json')
            or receipt.get('oof_sha256') != sha256(directory / 'oof.parquet')
            or receipt.get('test_sha256') != sha256(directory / 'test.parquet')
            or receipt.get('split_sha256') != split_hash
            or receipt.get('all_heldout_native_inference') is not True
            or receipt.get('heldout_rows') != CLOUD_DEVELOPMENT_ROWS or receipt.get('run') != run
            or receipt.get('execution_image') != expected_image
            or receipt.get('bundle_manifest_sha256') != expected_bundle
            or receipt.get('runtime_sha256') != expected_runtime):
        raise ValueError('Cloud candidate lacks full imported native/OOF verification')
    for field in ['source_hashes', 'native_hashes', 'heldout_replay_receipt_hashes']:
        mapping = receipt.get(field)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('Missing cloud native/source/replay provenance')
        for name, digest in mapping.items():
            artifact = (ROOT / name).resolve()
            if not artifact.is_relative_to(directory.resolve()) or sha256(artifact) != digest:
                raise ValueError('Changed cloud native/source/replay artifact')
    checked_cloud_feature_compat(run, receipt)
    return receipt


def blend(config_path, freeze=False):
    BLEND.mkdir(parents=True, exist_ok=True)
    frozen_path = BLEND / 'frozen.json'
    if frozen_path.exists():
        print('Third-pass selection already frozen; no reselection.', flush=True)
        return
    if sha256(BASELINE) != BASELINE_SHA:
        raise ValueError('The immutable v2 selection changed')
    config = load_config(config_path)
    long_policy = long_local_policy(config)
    if config['selection_policy'] != REGISTERED_POLICY:
        raise ValueError('The preregistered third-pass selection policy changed')
    runs = {run['id']: run for run in config['runs']}
    if len(runs) != len(config['runs']):
        raise ValueError('Duplicate run IDs')
    if any(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', name) is None for name in runs):
        raise ValueError('Invalid run identifier')
    baseline = load_config(BASELINE)
    baseline_weights = {key: value for key, value in baseline['weights'].items() if value > 0}
    split_hash = sha256(ROOT / 'data/splits.parquet')
    if baseline['split_hash'] != split_hash:
        raise ValueError('Frozen split changed')
    split = pd.read_parquet(ROOT / 'data/splits.parquet')
    dev = split.loc[split.fold >= 0]
    ids, folds = dev.id.to_numpy(), dev.fold.to_numpy()
    # Obtain only development labels. No training parquet or audit scores are read.
    first = pd.read_parquet(ROOT / 'artifacts/runs' / next(iter(baseline_weights)) / 'oof.parquet')
    if not np.array_equal(first.id, ids):
        raise ValueError('Baseline OOF IDs do not match development splits')
    y = first[TARGET].to_numpy()
    del first, dev, split
    anchor = np.zeros(len(ids), dtype=np.float64)
    source_hashes = {}
    for name, weight in baseline_weights.items():
        anchor += weight * checked_run(runs[name], split_hash, ids, y, folds)
        source_hashes[name] = sha256(ROOT / 'artifacts/runs' / name / 'result.json')
    baseline_scores = scores(y, anchor, folds)
    if abs(baseline_scores['pooled'] - baseline['oof_auc']) > 1e-12:
        raise ValueError('Fixed baseline cannot be reproduced')
    excluded = set(config.get('selection_exclusions', []))
    if not excluded.issubset(runs):
        raise ValueError('Unknown control-only selection exclusion')
    new = {}
    results = {}
    ledger = []
    for run in config['runs']:
        name = run['id']
        if not name.startswith(config['selection_policy']['new_candidate_prefix']):
            continue
        path = ROOT / 'artifacts/runs' / name / 'result.json'
        if not path.exists():
            continue
        new[name] = checked_run(run, split_hash, ids, y, folds)
        source_hashes[name] = sha256(path)
        results[name] = scores(y, new[name], folds)
        if run.get('execution_backend') == 'kaggle':
            checked_cloud_import(run, split_hash)
        recorded = load_config(path)
        control_name = (config.get('comparison_controls', {}).get(name)
                        or config.get('paired_duration_controls', {}).get(name))
        control = None
        if control_name:
            if control_name not in runs:
                raise ValueError('Unknown matched control in experiment ledger')
            control_path = ROOT / 'artifacts/runs' / control_name / 'result.json'
            if control_path.exists():
                control = load_config(control_path)
        item = {'id': name, 'family': run['family'], 'control': control_name,
                'oof_auc': results[name]['pooled'], 'mean_fold_auc': results[name]['mean_fold'],
                'seconds': recorded.get('seconds'), 'rounds': str(recorded.get('rounds')),
                'pooled_delta': None if control is None else results[name]['pooled'] - control['oof_auc'],
                'mean_fold_delta': None if control is None else results[name]['mean_fold'] - float(np.mean(control['fold_auc']))}
        for k in range(3):
            item[f'fold_{k}_auc'] = results[name]['folds'][k]
            item[f'fold_{k}_delta'] = None if control is None else results[name]['folds'][k] - control['fold_auc'][k]
        item['selection_excluded'] = name in excluded
        duration_control = config.get('paired_duration_controls', {}).get(name)
        if name in {'v3_realmlp_cat_raw_aux_e60', 'v3_realmlp_cat_raw_aux_e500'} and not duration_control:
            raise ValueError('Local long-duration candidate requires its registered paired gate')
        if duration_control:
            if duration_control != control_name or control is None:
                raise ValueError('Duration comparison requires a completed registered control')
            control_prediction = checked_run(runs[duration_control], split_hash, ids, y, folds)
            horizon = 500 if name == 'v3_realmlp_cat_raw_aux_e500' else 60
            if horizon == 500 and (not long_policy or run != long_policy['candidate_run']
                                   or duration_control != long_policy['control_id']):
                raise ValueError('Unregistered overnight duration comparison')
            matched = duration_comparison(run, runs[duration_control], results[name],
                                          scores(y, control_prediction, folds), horizon=horizon)
            item['matched_duration_gate'] = bool(matched)
            if not matched:
                excluded.add(name)
                item['selection_excluded'] = True
        if run.get('execution_backend') == 'kaggle' and name not in excluded:
            if not control_name or control is None:
                raise ValueError('Cloud candidate needs its preregistered platform control')
            if runs[control_name].get('execution_backend') != 'kaggle' or control_name not in excluded:
                raise ValueError('Cloud comparison requires its excluded cloud platform control')
            candidate_receipt = checked_cloud_import(run, split_hash)
            control_receipt = checked_cloud_import(runs[control_name], split_hash)
            if candidate_receipt['execution_image'] != control_receipt['execution_image']:
                raise ValueError('Cloud candidate/control images differ')
            if name == 'v3_cloud_realmlp_raw_aux_e12':
                ignored = {'id', 'max_rounds'}
                if ({k:v for k,v in run.items() if k not in ignored}
                        != {k:v for k,v in runs[control_name].items() if k not in ignored}
                        or run['max_rounds'] != 12 or runs[control_name]['max_rounds'] != 4):
                    raise ValueError('Longer-schedule experiment changed more than its registered horizon')
            if name == 'v3_cloud_realmlp_raw_aux_e60':
                # Recipe identity is separate from, and precedes, score admission.
                if control_name != load_config(ROOT / 'configs/third_pass_long_cloud.json')['control_id']:
                    raise ValueError('Long cloud candidate requires the exact preregistered control')
                duration_comparison(run, runs[control_name], results[name], results[name])
            if name == SHRINKAGE_ID:
                protocol = load_config(ROOT / 'configs/third_pass_lgb_shrinkage.json')
                if control_name != protocol['control_id'] or runs[control_name] != protocol['reused_control']['run']:
                    raise ValueError('Shrinkage comparison requires the exact registered verified control')
            elif run['family'] == 'lightgbm':
                ignored = {'id', 'original_aux_probability'}
                if ({k:v for k,v in run.items() if k not in ignored}
                        != {k:v for k,v in runs[control_name].items() if k not in ignored}
                        or run.get('original_aux_probability') is not True
                        or runs[control_name].get('original_aux_probability', False)):
                    raise ValueError('LightGBM comparison changed more than its registered probability feature block')
            control_prediction = checked_run(runs[control_name], split_hash, ids, y, folds)
            control_scores = scores(y, control_prediction, folds)
            matched = (results[name]['pooled'] >= control_scores['pooled']
                       and results[name]['mean_fold'] >= control_scores['mean_fold']
                       and min(np.asarray(results[name]['folds']) - control_scores['folds']) >= -.00002)
            item['matched_cloud_gate'] = bool(matched)
            if not matched:
                excluded.add(name)
                item['selection_excluded'] = True
        ledger.append(item)
    group_members = set()
    candidates = {}
    expansion = {}
    for name, members in config.get('blend_groups', {}).items():
        if not any(member.startswith('v3_') for member in members):
            continue
        if (name == 'anchor' or name in runs or not all(member.startswith('v3_') for member in members)
                or set(members).intersection(excluded)
                or len(members) < 2 or len(set(members)) != len(members) or not set(members).issubset(runs)):
            raise ValueError(f'Invalid fixed seed group {name}')
        if group_members.intersection(members):
            raise ValueError('Overlapping fixed seed groups')
        group_members.update(members)
        # Wait for the entire declared group. Never select its fortunate seed alone.
        if set(members).issubset(new):
            candidates[name] = sum(new[member] for member in members) / len(members)
            expansion[name] = {member: 1 / len(members) for member in members}
    for name, probability in new.items():
        if name not in group_members and name not in excluded:
            candidates[name] = probability
            expansion[name] = {name: 1.0}
    grid_trace = []
    weights, probability, value, history = anchored_search(y, folds, anchor, candidates, config['selection_policy'], grid_trace)
    expanded = {name: weight * weights['anchor'] for name, weight in baseline_weights.items()}
    for candidate, weight in weights.items():
        if candidate == 'anchor':
            continue
        for member, fraction in expansion[candidate].items():
            expanded[member] = expanded.get(member, 0.0) + weight * fraction
    test_ids = pd.read_parquet(ROOT / 'data/test.parquet', columns=['id']).id.to_numpy()
    test = sum(weight * probabilities(ROOT / 'artifacts/runs' / name / 'test.parquet', test_ids)
               for name, weight in expanded.items())
    atomic_csv(pd.DataFrame({'id': test_ids, TARGET: test}), BLEND / 'submission_current.csv')
    pd.DataFrame({'id': ids, 'prediction': probability}).to_parquet(BLEND / 'oof.parquet', index=False)
    selection = {'created_utc': now(), 'campaign': config.get('campaign_id', 'third_pass'), 'weights': expanded,
                 'candidate_ids': list(new), 'candidate_scores': results, 'anchor_weights': baseline_weights,
                 'baseline_selection_sha256': BASELINE_SHA, 'baseline_scores': baseline_scores,
                 'oof_auc': value['pooled'], 'fold_auc': value['folds'], 'mean_fold_auc': value['mean_fold'],
                 'anchor_and_addition_weights': weights, 'addition_history': history,
                 'grid_evaluations': grid_trace,
                 'selection_policy': config['selection_policy'], 'split_hash': split_hash,
                 'source_result_hashes': source_hashes, 'submission_hash': sha256(BLEND / 'submission_current.csv'),
                 'audit_evaluated': False,
                 'evaluation_note': 'Adaptively reused development OOF selection estimates. Original audit excluded from v3 selection and scoring.'}
    if ledger:
        atomic_csv(pd.DataFrame(ledger), OUT / 'experiment_ledger.csv')
        selection['experiment_ledger_sha256'] = sha256(OUT / 'experiment_ledger.csv')
    atomic_json(BLEND / 'current.json', selection)
    if freeze:
        shutil.copyfile(BLEND / 'submission_current.csv', BLEND / 'submission_fallback.csv')
        atomic_json(frozen_path, selection)
    print({'oof_auc': value['pooled'], 'baseline_oof_auc': baseline_scores['pooled'],
           'completed_new': len(new), 'additions': weights, 'frozen': freeze}, flush=True)


def context(config_path):
    if sha256(BASELINE) != BASELINE_SHA:
        raise ValueError('Immutable anchor selection changed after freeze')
    selection = load_config(BLEND / 'frozen.json')
    config = load_config(config_path)
    if selection['baseline_selection_sha256'] != BASELINE_SHA:
        raise ValueError('Frozen selection anchor hash mismatch')
    runs = {run['id']: run for run in config['runs']}
    if selection['split_hash'] != sha256(ROOT / 'data/splits.parquet'):
        raise ValueError('Release split changed')
    weights = selection['weights']
    if not weights or any(not np.isfinite(w) or w <= 0 for w in weights.values()) or abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError('Invalid release weights')
    specs = []
    for name, weight in weights.items():
        path = ROOT / 'artifacts/runs' / name / 'result.json'
        if sha256(path) != selection['source_result_hashes'][name]:
            raise ValueError(f'Changed source result: {name}')
        result = load_config(path)
        if result['run'] != runs[name]:
            raise ValueError(f'Changed selected config: {name}')
        specs.append({'id': name, 'weight': weight, 'run': runs[name],
                      'rounds': max(1, int(np.median(result['rounds'])))})
    return selection, specs


def member_path(spec):
    if sha256(BASELINE) != BASELINE_SHA:
        raise ValueError('Immutable anchor selection changed after freeze')
    if spec['id'] in load_config(BASELINE)['weights']:
        if sha256(ANCHOR_MANIFEST) != ANCHOR_MANIFEST_SHA:
            raise ValueError('Changed verified anchor manifest')
        matches = [m for m in load_config(ANCHOR_MANIFEST)['members'] if m['id'] == spec['id']]
        if len(matches) != 1 or matches[0]['run'] != spec['run'] or matches[0]['rounds'] != spec['rounds']:
            raise ValueError('Anchor member identity/configuration mismatch')
        directory = Path(matches[0]['path']).resolve().parent
        allowed = [ROOT / 'artifacts/final', ROOT / 'artifacts/second_pass/final']
        if directory.parent not in allowed:
            raise ValueError('Unexpected anchor member directory')
        return directory
    return FINAL / spec['id']


def validate_member(spec, ids):
    directory = member_path(spec)
    done = load_config(directory / 'done.json')
    metadata = load_config(directory / 'model/model_metadata.json')
    if (done['id'] != spec['id'] or done['rounds'] != spec['rounds']
            or done['rows'] != load_config(ROOT / 'data/manifest.json')['rows']['train']
            or metadata['run'] != spec['run'] or metadata['rounds'] != spec['rounds']
            or sha256(directory / 'test.parquet') != done['test_hash']):
        raise ValueError(f'Invalid full-data member: {spec["id"]}')
    if directory.parent != FINAL:
        # Reuse immutable v1 models only after verifying their archived checksums.
        provenance_path = directory.parent / 'release_provenance.json'
        pinned = {str(ROOT / 'artifacts/final'): V1_PROVENANCE_SHA,
                  str(ROOT / 'artifacts/second_pass/final'): V2_PROVENANCE_SHA}
        if sha256(provenance_path) != pinned[str(directory.parent)]:
            raise ValueError('Changed anchor member provenance')
        provenance = load_config(provenance_path)
        prefix = directory.relative_to(ROOT).as_posix() + '/'
        checks = {p: digest for p, digest in provenance.get('native_model_and_prediction_hashes', provenance.get('native_hashes', {})).items()
                  if p.replace('\\', '/').startswith(prefix)}
        if not checks:
            raise ValueError('Missing v1 native-model provenance')
        for path, digest in checks.items():
            if sha256(ROOT / path) != digest:
                raise ValueError(f'Changed v1 member artifact: {path}')
    else:
        if done.get('frozen_selection_hash') != sha256(BLEND / 'frozen.json'):
            raise ValueError('New refit belongs to another selection')
        checks = done.get('native_hashes', {})
        actual_files = {p.relative_to(directory).as_posix() for p in (directory / 'model').rglob('*') if p.is_file()}
        if not checks or set(checks) != actual_files:
            raise ValueError('Missing new native-model provenance')
        for name, digest in checks.items():
            if sha256(directory / name) != digest:
                raise ValueError(f'Changed native member artifact: {name}')
    if spec['run'].get('execution_backend') == 'kaggle':
        validate_cloud_production(spec, directory, done)
    return probabilities(directory / 'test.parquet', ids)


def validate_cloud_production(spec, directory, done):
    cv_dir = ROOT / 'artifacts/runs' / spec['id']
    cv = checked_cloud_import(spec['run'], sha256(ROOT / 'data/splits.parquet'))
    proof_path = directory / 'cloud_production_verification.json'
    if not proof_path.exists() or done.get('cloud_production_verification_sha256') != sha256(proof_path):
        raise ValueError('Unverified cached cloud production member')
    proof = load_config(proof_path)
    tolerance = (1e-10, 1e-12) if spec['run']['family'] == 'lightgbm' else (1e-5, 2e-6)
    expected = {'status': 'verified', 'id': spec['id'], 'run': spec['run'], 'rows': 699635,
                'rounds': spec['rounds'], 'test_rows': 299844,
                'frozen_selection_hash': sha256(BLEND / 'frozen.json'),
                'cv_result_sha256': sha256(cv_dir / 'result.json'),
                'cv_import_receipt_sha256': sha256(cv_dir / 'cloud_import_verification.json'),
                'execution_image': cv['execution_image'], 'runtime_sha256': cv['runtime_sha256'],
                'all_test_native_inference': True, 'local_inference_device': 'cpu',
                'test_hash': sha256(directory / 'test.parquet'),
                'cloud_test_sha256': sha256(directory / 'test.parquet'), 'rtol': tolerance[0], 'atol': tolerance[1]}
    if spec['run']['family'] == 'lightgbm':
        shrinkage = spec['run']['id'] == SHRINKAGE_ID
        expected.update(protocol_id='lgb_shrinkage_v1' if shrinkage else 'lgb_rating_probability_v1',
                        protocol_sha256=SHRINKAGE_PROTOCOL_SHA if shrinkage else LGB_CLOUD_PROTOCOL_SHA,
                        lightgbm_version='4.7.0', execution_threads=4, training_device='cpu')
    if spec['run'].get('categorical_twins') is True:
        expected.update(source_versions=cv['source_versions'], feature_compat_sha256=cv['feature_compat_sha256'])
    if any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError('Cloud production freeze/platform/row/test proof mismatch')
    error = proof.get('full_test_max_abs_error', float('inf'))
    if not np.isfinite(error) or not 0 <= error <= sum(tolerance):
        raise ValueError('Cloud production native replay tolerance failed')
    native = proof.get('native_hashes')
    actual = {p.relative_to(ROOT).as_posix() for p in (directory / 'model').rglob('*') if p.is_file()}
    if not isinstance(native, dict) or set(native) != actual:
        raise ValueError('Cloud production native file set changed')
    provenance = proof.get('production_provenance_hashes')
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError('Missing cloud production input/source/remote provenance')
    if spec['run'].get('categorical_twins') is True:
        if provenance.get('scripts/cloud_feature_compat.py') != cv['feature_compat_sha256']:
            raise ValueError('Missing cloud production feature compatibility provenance')
    if set(native).intersection(provenance):
        raise ValueError('Overlapping cloud native/provenance maps')
    for name, digest in list(native.items()) + list(provenance.items()):
        artifact = (ROOT / name).resolve()
        if not artifact.is_relative_to(ROOT.resolve()) or sha256(artifact) != digest:
            raise ValueError('Changed cloud production native/provenance artifact')
    return proof


def checked_manifest(specs):
    manifest = load_config(FINAL / 'manifest.json')
    if manifest['frozen_selection_hash'] != sha256(BLEND / 'frozen.json'):
        raise ValueError('Manifest selection mismatch')
    members = manifest.get('members', [])
    if len(members) != len(specs) or {m['id'] for m in members} != {s['id'] for s in specs}:
        raise ValueError('Manifest member identity mismatch')
    expected = {s['id']: s for s in specs}
    for member in members:
        spec = expected[member['id']]
        if any(member.get(key) != value for key, value in spec.items()):
            raise ValueError('Manifest member configuration or weight mismatch')
        directory = member_path(spec).resolve()
        if (Path(member['path']).resolve() != directory / 'model'
                or Path(member['predictions']).resolve() != directory / 'test.parquet'):
            raise ValueError('Manifest member path mismatch')
    return manifest


def refit(config_path, run_id=None):
    selection, specs = context(config_path)
    frozen_hash = sha256(BLEND / 'frozen.json')
    ids = pd.read_parquet(ROOT / 'data/test.parquet', columns=['id']).id.to_numpy()
    if run_id:
        spec = next(s for s in specs if s['id'] == run_id)
        directory = member_path(spec)
        if (directory / 'done.json').exists():
            validate_member(spec, ids)
            return
        if directory.parent != FINAL:
            raise ValueError('Missing v1 artifact; refuse to overwrite original release')
        if spec['run'].get('execution_backend') == 'kaggle':
            raise RuntimeError('Selected cloud member requires a fixed-round same-image cloud refit after freeze; local retraining is forbidden')
        train, test, _ = load_data()
        x, xt = features(train, spec['run']), features(test, spec['run'])
        model, transform, rounds = fit_model(x, train[TARGET].to_numpy(), None, None,
                                             spec['run'], directory / 'model', rounds=spec['rounds'])
        if rounds != spec['rounds'] or sha256(BLEND / 'frozen.json') != frozen_hash:
            raise ValueError('Refit round count or selection changed')
        pd.DataFrame({'id': ids, 'prediction': predict(model, transform, xt)}).to_parquet(directory / 'test.parquet', index=False)
        atomic_json(directory / 'done.json', {'id': run_id, 'rounds': rounds, 'rows': len(train),
                    'run': spec['run'], 'frozen_selection_hash': frozen_hash,
                    'native_hashes': {p.relative_to(directory).as_posix(): sha256(p)
                                      for p in (directory / 'model').rglob('*') if p.is_file()},
                    'test_hash': sha256(directory / 'test.parquet')})
        validate_member(spec, ids)
        return
    FINAL.mkdir(parents=True, exist_ok=True)
    probability = np.zeros(len(ids), dtype=np.float64)
    members = []
    for spec in specs:
        directory = member_path(spec)
        if not (directory / 'done.json').exists():
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--phase', 'refit',
                       '--config', str(config_path), '--run-id', spec['id']]
            subprocess.run(command, cwd=ROOT, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        probability += spec['weight'] * validate_member(spec, ids)
        members.append({**spec, 'path': str(directory / 'model'), 'predictions': str(directory / 'test.parquet')})
        print(f'Packaged {spec["id"]}', flush=True)
    if sha256(BLEND / 'frozen.json') != frozen_hash:
        raise ValueError('Selection changed during packaging')
    atomic_csv(pd.DataFrame({'id': ids, TARGET: probability}), FINAL / 'submission.csv')
    atomic_json(FINAL / 'manifest.json', {'members': members, 'frozen_selection_hash': frozen_hash,
                'submission_hash': sha256(FINAL / 'submission.csv'), 'oof_auc': selection['oof_auc'],
                'audit_evaluated': False, 'evaluation_note': selection['evaluation_note']})


def raw_inference(config_path, input_path, output_path):
    output_path = Path(output_path).resolve()
    if (output_path.parent not in {OUT.resolve(), FINAL.resolve()}
            or output_path == (FINAL / 'submission.csv').resolve()
            or output_path.is_relative_to(BLEND.resolve())
            or output_path.is_relative_to((OUT / 'verified').resolve())
            or output_path.is_relative_to((FINAL / 'reproduction_source').resolve())
            or output_path == Path(input_path).resolve()
            or output_path.suffix.lower() != '.csv'):
        raise ValueError('Inference output must be a separate CSV under artifacts/third_pass')
    selection, specs = context(config_path)
    manifest = checked_manifest(specs)
    expected_ids = pd.read_parquet(ROOT / 'data/test.parquet', columns=['id']).id.to_numpy()
    for spec in specs:
        validate_member(spec, expected_ids)
    contract = load_config(ROOT / 'data/manifest.json')
    with Path(input_path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError('Duplicate raw input column names')
    raw = pd.read_csv(input_path, dtype={'id': str, **{column: str for column in CAT}})
    required = ['id'] + contract['feature_columns']
    if not set(required).issubset(raw) or not raw.id.is_unique or raw.id.isna().any() or not len(raw):
        raise ValueError('Invalid raw input schema or IDs')
    raw = raw[required].copy()
    for column in contract['feature_columns']:
        if column not in CAT:
            raw[column] = pd.to_numeric(raw[column], errors='raise').astype('float32')
            if np.isinf(raw[column].to_numpy()).any():
                raise ValueError('Nonfinite or overflowing numeric input')
    teacher_probability, auxiliary, lgb_teacher_probability, auxiliary_probability, neural_teacher_logit = None, None, None, None, None
    probability = np.zeros(len(raw), dtype=np.float64)
    for member in manifest['members']:
        run = member['run']
        x = features(raw, {**run, 'teacher': False, 'original_aux': False, 'original_lgb_teacher': False, 'original_aux_probability': False, 'original_realmlp_teacher': False})
        if run.get('execution_backend') == 'kaggle' and run.get('family') == 'lightgbm' and run.get('route'):
            # Reproduce the separately registered cloud engine's explicit float32 string semantics.
            x['route_category'] = pd.Series(raw['Flight Distance'].to_numpy().astype(str), index=x.index, dtype=object)
        if run.get('execution_backend') == 'kaggle' and run.get('categorical_twins') is True:
            receipt = load_config(ROOT / 'artifacts/runs' / run['id'] / 'cloud_import_verification.json')
            checked_cloud_feature_compat(run, receipt)
            x = cloud_numeric_category_compat(x, raw, run=run, source_versions=receipt['source_versions'])
        if run.get('teacher'):
            if teacher_probability is None:
                model, transform = load_model(ROOT / 'artifacts/teacher')
                teacher_probability = predict(model, transform, raw[contract['feature_columns']])
                del model, transform
                gc.collect()
            x['teacher_probability'] = teacher_probability
        if run.get('original_aux'):
            if auxiliary is None:
                from original_aux import predict_aux
                auxiliary = predict_aux(raw[contract['feature_columns']])
            for column in auxiliary:
                x[column] = auxiliary[column].to_numpy()
        if run.get('original_aux_probability'):
            if auxiliary_probability is None:
                from original_aux_probability import predict_aux_probability
                auxiliary_probability = predict_aux_probability(raw[contract['feature_columns']])
            for column in auxiliary_probability:
                x[column] = auxiliary_probability[column].to_numpy()
        if run.get('original_realmlp_teacher'):
            if neural_teacher_logit is None:
                from original_realmlp_teacher import predict_teacher
                neural_teacher_logit = predict_teacher(raw[contract['feature_columns']])
            x['original_realmlp_teacher_logit'] = neural_teacher_logit
        if run.get('original_lgb_teacher'):
            if lgb_teacher_probability is None:
                from original_lgb_teacher import predict_teacher
                lgb_teacher_probability = predict_teacher(raw[contract['feature_columns']])
            x['original_lgb_teacher_probability'] = lgb_teacher_probability
        model, transform = load_model(Path(member['path']))
        probability += member['weight'] * predict(model, transform, x)
        del model, transform, x
        gc.collect()
        print(f'Raw inference complete: {member["id"]}', flush=True)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError('Invalid raw inference probabilities')
    atomic_csv(pd.DataFrame({'id': raw.id, TARGET: probability}), output_path)


def verify(config_path):
    selection, specs = context(config_path)
    manifest = checked_manifest(specs)
    path = FINAL / 'submission.csv'
    if manifest['submission_hash'] != sha256(path) or manifest['frozen_selection_hash'] != sha256(BLEND / 'frozen.json'):
        raise ValueError('Release hash mismatch')
    sample = pd.read_csv(ROOT / 'data/sample_submission.csv', dtype={'id': str})
    submitted = pd.read_csv(path, dtype={'id': str})
    ids = pd.read_parquet(ROOT / 'data/test.parquet', columns=['id']).id.to_numpy()
    if list(submitted) != ['id', TARGET] or not submitted.id.is_unique or not np.array_equal(submitted.id, sample.id):
        raise ValueError('Submission schema or row identity mismatch')
    values = submitted[TARGET].to_numpy()
    if len(values) != len(ids) or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('Invalid submission probabilities')
    expected = sum(spec['weight'] * validate_member(spec, ids) for spec in specs)
    np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-11)
    # Fresh inference process releases each library's training allocations first.
    reproduced = FINAL / 'reproduced_from_raw.csv'
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--phase', 'predict',
               '--config', str(config_path), '--input', str(ROOT / 'data/test.csv'), '--output', str(reproduced)]
    subprocess.run(command, cwd=ROOT, check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    actual = pd.read_csv(reproduced, dtype={'id': str})
    if not np.array_equal(actual.id, submitted.id):
        raise ValueError('Raw inference identifiers differ')
    np.testing.assert_allclose(actual[TARGET], values, rtol=2e-5, atol=2e-6)
    result = {'verified_utc': now(), 'submission_path': str(path), 'sha256': sha256(path),
              'rows': len(values), 'all_row_raw_inference': True,
              'raw_inference_sha256': sha256(reproduced),
              'raw_inference_max_absolute_difference': float(np.max(np.abs(actual[TARGET] - values))),
              'independent_blend_recomputation': True, 'oof_auc': selection['oof_auc'],
              'baseline_oof_auc': selection['baseline_scores']['pooled'], 'audit_evaluated': False}
    atomic_json(OUT / 'verification.json', result)
    snapshot = FINAL / 'reproduction_source'
    sources = list((ROOT / 'scripts').glob('*.py')) + [config_path, ROOT / 'requirements.lock.txt', ROOT / 'THIRD_PASS_PLAN.md',
              ROOT / 'data/manifest.json', ROOT / 'data/splits.parquet', ROOT / 'data/audit_smoke_exclusions.json',
              ROOT / 'research/original_audit.json', BASELINE, ANCHOR_MANIFEST,
              ROOT / 'artifacts/second_pass/verification.json',
              ROOT / 'artifacts/second_pass/final/release_provenance.json']
    if long_local_policy(load_config(config_path)):
        sources += [ROOT / 'configs/long_local_500.json', ROOT / 'configs/third_pass_batch05.json',
                    ROOT / 'LONG_LOCAL_500_PLAN.md',
                    ROOT / 'state/long_local_500/prior_campaign_ready.json']
        amendment = ROOT / 'state/long_local_500/start_now_authorization.json'
        if amendment.exists():
            sources.append(amendment)
    source_hashes = {}
    for source in sources:
        destination = snapshot / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hashes[destination.relative_to(ROOT).as_posix()] = sha256(destination)
    native_hashes = {}
    directories = [Path(member['path']) for member in manifest['members']] + [ROOT / 'artifacts/teacher']
    if any(member['run'].get('original_aux') or member['run'].get('original_aux_probability') for member in manifest['members']):
        directories.append(ROOT / 'artifacts/original_aux')
    if any(member['run'].get('original_lgb_teacher') for member in manifest['members']):
        directories.append(ROOT / 'artifacts/original_lgb_teacher')
    if any(member['run'].get('original_realmlp_teacher') for member in manifest['members']):
        directories.append(ROOT / 'artifacts/original_realmlp_teacher')
    for directory in directories:
        for native in directory.rglob('*'):
            if native.is_file():
                native_hashes[native.relative_to(ROOT).as_posix()] = sha256(native)
    for member in manifest['members']:
        directory = Path(member['path']).parent
        for name in ['done.json', 'test.parquet', 'cloud_production_verification.json']:
            artifact = directory / name
            if artifact.exists():
                native_hashes[artifact.relative_to(ROOT).as_posix()] = sha256(artifact)
        if member['run'].get('execution_backend') == 'kaggle':
            proof = load_config(directory / 'cloud_production_verification.json')
            native_hashes.update(proof['production_provenance_hashes'])
    baseline = load_config(ROOT / 'artifacts/blend/frozen.json')
    report = f'# Airline satisfaction third-pass result\n\nVerified {now()}.\n\n'
    report += f'Development OOF ROC AUC: **{selection["oof_auc"]:.8f}**, fixed v2 baseline **{selection["baseline_scores"]["pooled"]:.8f}**. '
    report += f'{len(selection["candidate_ids"])} completed new configurations; {len(specs)} positive-weight configurations.\n\n'
    report += 'Development fold AUC: ' + ' / '.join(f'{score:.8f}' for score in selection['fold_auc']) + '.\n\n'
    report += f'Per-run paired controls, fold scores, fit lengths and runtimes: [experiment ledger]({OUT.relative_to(ROOT).as_posix()}/experiment_ledger.csv). Full search grid: [frozen selection]({BLEND.relative_to(ROOT).as_posix()}/frozen.json).\n\n'
    report += 'These are adaptively reused development selection scores, not a fresh unbiased assessment. The exposed original audit was excluded from all third-pass selection and scoring. Full-data refits include all labels after selection freeze.\n\n'
    report += f'Historical v1 only: original audit {baseline["audit_auc"]:.8f}; unexposed sensitivity audit {baseline["audit_unexposed_auc"]:.8f} on 69,396 rows, excluding 568 smoke-exposed rows. These historical scores do not evaluate v2 or v3. V1 public score was0.96093; v2 scored0.96122. Submission receipts are separate from this frozen report.\n\n'
    report += f'Submission: `{path.relative_to(ROOT).as_posix()}`, {len(values):,} rows. SHA256 `{result["sha256"]}`.\n\n'
    report += f'Verified exact IDs/order/schema, finite probability bounds, all-row independent blend arithmetic, native-model reload and raw inference on all rows. Maximum raw inference difference: {result["raw_inference_max_absolute_difference"]:.3g}. GPU training is not guaranteed bitwise reproducible.\n\n'
    report += 'Weights:\n\n' + ''.join(f'- {spec["id"]}: {spec["weight"]:.8f}\n' for spec in specs)
    report += f'\nReproduce:\n\n```powershell\n.venv/Scripts/python.exe scripts/third_pass_release.py --phase predict --config {config_path.relative_to(ROOT).as_posix()} --input data/test.csv --output {OUT.relative_to(ROOT).as_posix()}/reproduced.csv\n```\n'
    REPORT.write_text(report, encoding='utf-8')
    atomic_json(FINAL / 'release_provenance.json', {'created_utc': now(), 'submission_sha256': sha256(path),
                'manifest_sha256': sha256(FINAL / 'manifest.json'), 'selection_sha256': sha256(BLEND / 'frozen.json'),
                'source_hashes': source_hashes, 'native_hashes': native_hashes,
                'experiment_ledger_sha256': selection.get('experiment_ledger_sha256'),
                'verification_sha256': sha256(OUT / 'verification.json'), 'report_sha256': sha256(REPORT)})
    archive = OUT / 'verified' / result['sha256']
    archive.mkdir(parents=True, exist_ok=True)
    for source in [path, BLEND / 'frozen.json', FINAL / 'manifest.json', FINAL / 'release_provenance.json', OUT / 'verification.json']:
        shutil.copyfile(source, archive / source.name)
    print(result, flush=True)


def configure_campaign(config):
    global OUT, BLEND, FINAL, REPORT
    identity = config.get('campaign_id', 'third_pass')
    if re.fullmatch(r'third_pass(?:_batch[0-9]{2})?', identity) is None:
        raise ValueError('Invalid third-pass campaign identifier')
    OUT = ROOT / 'artifacts' / identity
    BLEND, FINAL = OUT / 'blend', OUT / 'final'
    REPORT = ROOT / (identity.upper() + '_REPORT.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=['blend', 'refit', 'verify', 'predict'], required=True)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    configure_campaign(load_config(config_path))
    if load_config(config_path).get('campaign') != 'third_pass':
        raise ValueError('This entry point requires the isolated third_pass campaign')
    if args.phase == 'blend':
        blend(config_path, args.freeze)
    elif args.phase == 'refit':
        refit(config_path, args.run_id)
    elif args.phase == 'verify':
        verify(config_path)

    else:
        if args.input is None or args.output is None:
            parser.error('predict requires --input and --output')
        raw_inference(config_path, args.input, args.output)


if __name__ == '__main__':
    main()
