"""Development-only second-pass selection and an isolated, reproducible release.

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

OUT = ROOT / 'artifacts/second_pass'
BLEND = OUT / 'blend'
FINAL = OUT / 'final'
BASELINE = ROOT / 'artifacts/blend/frozen.json'
BASELINE_SHA = 'ea06c3b7af6505cea9e51c97bd40c6c5c488af950e6d0b43c9ba7247c9018f48'
REGISTERED_POLICY = {'minimum_pooled_gain': .00001, 'minimum_mean_fold_gain': .00001,
                     'maximum_fold_regression': .00002, 'alpha_grid': [.05, .1, .2, .3],
                     'max_additions': 3, 'new_candidate_prefix': 'v2_'}


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


def blend(config_path, freeze=False):
    BLEND.mkdir(parents=True, exist_ok=True)
    frozen_path = BLEND / 'frozen.json'
    if frozen_path.exists():
        print('Second-pass selection already frozen; no reselection.', flush=True)
        return
    if sha256(BASELINE) != BASELINE_SHA:
        raise ValueError('The immutable v1 selection changed')
    config = load_config(config_path)
    if config['selection_policy'] != REGISTERED_POLICY:
        raise ValueError('The preregistered second-pass selection policy changed')
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
        recorded = load_config(path)
        control_name = config.get('comparison_controls', {}).get(name)
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
        ledger.append(item)
    group_members = set()
    candidates = {}
    expansion = {}
    for name, members in config.get('blend_groups', {}).items():
        if not any(member.startswith('v2_') for member in members):
            continue
        if (name == 'anchor' or name in runs or not all(member.startswith('v2_') for member in members)
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
        if name not in group_members:
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
    selection = {'created_utc': now(), 'campaign': 'second_pass', 'weights': expanded,
                 'candidate_ids': list(new), 'candidate_scores': results, 'anchor_weights': baseline_weights,
                 'baseline_selection_sha256': BASELINE_SHA, 'baseline_scores': baseline_scores,
                 'oof_auc': value['pooled'], 'fold_auc': value['folds'], 'mean_fold_auc': value['mean_fold'],
                 'anchor_and_addition_weights': weights, 'addition_history': history,
                 'grid_evaluations': grid_trace,
                 'selection_policy': config['selection_policy'], 'split_hash': split_hash,
                 'source_result_hashes': source_hashes, 'submission_hash': sha256(BLEND / 'submission_current.csv'),
                 'audit_evaluated': False,
                 'evaluation_note': 'Adaptively reused development OOF selection estimates. Original audit excluded from v2 selection and scoring.'}
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
    selection = load_config(BLEND / 'frozen.json')
    config = load_config(config_path)
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
    return (ROOT / 'artifacts/final' if spec['id'] in load_config(BASELINE)['weights']
            else FINAL) / spec['id']


def validate_member(spec, ids):
    directory = member_path(spec)
    done = load_config(directory / 'done.json')
    metadata = load_config(directory / 'model/model_metadata.json')
    if (done['id'] != spec['id'] or done['rounds'] != spec['rounds']
            or done['rows'] != load_config(ROOT / 'data/manifest.json')['rows']['train']
            or metadata['run'] != spec['run'] or metadata['rounds'] != spec['rounds']
            or sha256(directory / 'test.parquet') != done['test_hash']):
        raise ValueError(f'Invalid full-data member: {spec["id"]}')
    if directory.parent == ROOT / 'artifacts/final':
        # Reuse immutable v1 models only after verifying their archived checksums.
        provenance = load_config(ROOT / 'artifacts/final/release_provenance.json')
        prefix = directory.relative_to(ROOT).as_posix() + '/'
        checks = {p: digest for p, digest in provenance['native_model_and_prediction_hashes'].items()
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
    return probabilities(directory / 'test.parquet', ids)


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
        raise ValueError('Inference output must be a separate CSV under artifacts/second_pass')
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
    teacher_probability, auxiliary, lgb_teacher_probability = None, None, None
    probability = np.zeros(len(raw), dtype=np.float64)
    for member in manifest['members']:
        run = member['run']
        x = features(raw, {**run, 'teacher': False, 'original_aux': False, 'original_lgb_teacher': False})
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
    sources = list((ROOT / 'scripts').glob('*.py')) + [config_path, ROOT / 'requirements.lock.txt', ROOT / 'SECOND_PASS_PLAN.md',
              ROOT / 'data/manifest.json', ROOT / 'data/splits.parquet', ROOT / 'data/audit_smoke_exclusions.json',
              ROOT / 'research/original_audit.json', BASELINE]
    source_hashes = {}
    for source in sources:
        destination = snapshot / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hashes[destination.relative_to(ROOT).as_posix()] = sha256(destination)
    native_hashes = {}
    directories = [Path(member['path']) for member in manifest['members']] + [ROOT / 'artifacts/teacher']
    if any(member['run'].get('original_aux') for member in manifest['members']):
        directories.append(ROOT / 'artifacts/original_aux')
    if any(member['run'].get('original_lgb_teacher') for member in manifest['members']):
        directories.append(ROOT / 'artifacts/original_lgb_teacher')
    for directory in directories:
        for native in directory.rglob('*'):
            if native.is_file():
                native_hashes[native.relative_to(ROOT).as_posix()] = sha256(native)
    baseline = load_config(BASELINE)
    report = f'# Airline satisfaction second-pass result\n\nVerified {now()}.\n\n'
    report += f'Development OOF ROC AUC: **{selection["oof_auc"]:.8f}**, fixed v1 baseline **{selection["baseline_scores"]["pooled"]:.8f}**. '
    report += f'{len(selection["candidate_ids"])} completed new configurations; {len(specs)} positive-weight configurations.\n\n'
    report += 'Development fold AUC: ' + ' / '.join(f'{score:.8f}' for score in selection['fold_auc']) + '.\n\n'
    report += 'Per-run paired controls, all fold scores, fit lengths and runtimes: [experiment ledger](artifacts/second_pass/experiment_ledger.csv). Full blend-search grid, including rejected mixtures: [frozen selection](artifacts/second_pass/blend/frozen.json).\n\n'
    report += 'These are adaptively reused development selection scores, not a fresh unbiased assessment. The exposed original audit was excluded from all second-pass selection and scoring. Full-data refits include all labels after selection freeze.\n\n'
    report += f'Historical v1 only: original audit {baseline["audit_auc"]:.8f}; unexposed sensitivity audit {baseline["audit_unexposed_auc"]:.8f} on 69,396 rows, excluding 568 smoke-exposed rows. These scores do not evaluate v2. V1 public score was 0.96093; v2 has not been submitted.\n\n'
    report += f'Submission: `{path.relative_to(ROOT).as_posix()}`, {len(values):,} rows. SHA256 `{result["sha256"]}`.\n\n'
    report += f'Verified exact IDs/order/schema, finite probability bounds, all-row independent blend arithmetic, native-model reload and raw inference on all rows. Maximum raw inference difference: {result["raw_inference_max_absolute_difference"]:.3g}. GPU training is not guaranteed bitwise reproducible.\n\n'
    report += 'Weights:\n\n' + ''.join(f'- {spec["id"]}: {spec["weight"]:.8f}\n' for spec in specs)
    report += '\nReproduce:\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts/second_pass_release.py --phase predict --config configs/second_pass.json --input data/test.csv --output artifacts/second_pass/reproduced.csv\n```\n'
    (ROOT / 'SECOND_PASS_REPORT.md').write_text(report, encoding='utf-8')
    atomic_json(FINAL / 'release_provenance.json', {'created_utc': now(), 'submission_sha256': sha256(path),
                'manifest_sha256': sha256(FINAL / 'manifest.json'), 'selection_sha256': sha256(BLEND / 'frozen.json'),
                'source_hashes': source_hashes, 'native_hashes': native_hashes,
                'experiment_ledger_sha256': selection.get('experiment_ledger_sha256'),
                'verification_sha256': sha256(OUT / 'verification.json'), 'report_sha256': sha256(ROOT / 'SECOND_PASS_REPORT.md')})
    archive = OUT / 'verified' / result['sha256']
    archive.mkdir(parents=True, exist_ok=True)
    for source in [path, BLEND / 'frozen.json', FINAL / 'manifest.json', FINAL / 'release_provenance.json', OUT / 'verification.json']:
        shutil.copyfile(source, archive / source.name)
    print(result, flush=True)


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
    if load_config(config_path).get('campaign') != 'second_pass':
        raise ValueError('This entry point requires the isolated second_pass campaign')
    if args.phase == 'blend':
        blend(config_path, args.freeze)
    elif args.phase == 'refit':
        refit(config_path, args.run_id)
    elif args.phase == 'verify':
        try:
            verify(config_path)
        except Exception as exc:
            from second_pass_fallback import verify_v1_fallback
            verify_v1_fallback(f'Second-pass release verification failed: {type(exc).__name__}: {exc}')
            # Preserve a verified fallback, but do not let the coordinator call
            # an incomplete or invalid second-pass release successful.
            raise
    else:
        if args.input is None or args.output is None:
            parser.error('predict requires --input and --output')
        raw_inference(config_path, args.input, args.output)


if __name__ == '__main__':
    main()
