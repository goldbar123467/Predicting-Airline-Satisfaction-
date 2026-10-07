"""Bounded, saved-prediction ensemble screen; never fits or submits."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
import numpy as np
import pandas as pd
from common import ROOT, TARGET, atomic_json, sha256
from third_pass_release import checked_run, checked_cloud_import, scores, eligible_change, now

PROTOCOL = ROOT / 'configs/checkpoint_blend_v1.json'
PROTOCOL_SHA = '5a9f69b4433362a6b025c1957c0305a9daace85a9897ee00257facbe9451940c'
OUT = ROOT / 'artifacts/checkpoint_blend_v1'


def mix(predictions, weights):
    if not weights or any(not np.isfinite(w) or w <= 0 for w in weights.values()) or abs(sum(weights.values()) - 1) > 1e-12:
        raise ValueError('Invalid fixed mixture weights')
    return sum(w * predictions[name] for name, w in weights.items())


def main():
    if sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('Changed registered screen')
    config = json.loads(PROTOCOL.read_text())
    OUT.mkdir(exist_ok=True)
    if (OUT / 'screen.json').exists():
        raise ValueError('Screen already completed; preserve it')
    anchor_path = ROOT / config['anchor_selection_path']
    if sha256(anchor_path) != config['anchor_selection_sha256'] or sha256(ROOT / 'data/splits.parquet') != config['split_sha256']:
        raise ValueError('Anchor or frozen split changed')
    for path, field in [('artifacts/third_pass/final/submission.csv', 'anchor_submission_sha256'),
                        ('artifacts/third_pass/final/release_provenance.json', 'anchor_provenance_sha256')]:
        if sha256(ROOT / path) != config[field]:
            raise ValueError('Verified submitted anchor changed')
    selection = json.loads(anchor_path.read_text())
    split = pd.read_parquet(ROOT / 'data/splits.parquet', columns=['id', 'fold'])
    split = split.loc[split.fold >= 0]
    ids, folds = split.id.to_numpy(), split.fold.to_numpy()
    first = pd.read_parquet(ROOT / 'artifacts/runs' / next(iter(selection['weights'])) / 'oof.parquet')
    if not np.array_equal(first.id, ids):
        raise ValueError('Development identity differs')
    y = first[TARGET].to_numpy()
    del first, split
    predictions = {}
    for name, spec in config['runs'].items():
        if sha256(ROOT / 'artifacts/runs' / name / 'result.json') != spec['result_sha256']:
            raise ValueError('Changed registered model result')
        predictions[name] = checked_run(spec['run'], config['split_sha256'], ids, y, folds)
        if spec['run'].get('execution_backend') == 'kaggle':
            checked_cloud_import(spec['run'], config['split_sha256'])
    anchor = mix(predictions, selection['weights'])
    anchor_score = scores(y, anchor, folds)
    if abs(anchor_score['pooled'] - selection['oof_auc']) > 1e-12:
        raise ValueError('Submitted development anchor cannot be reproduced')
    pools = {name: mix(predictions, weights) for name, weights in config['pools'].items()}
    pool_scores = {name: scores(y, probability, folds) for name, probability in pools.items()}
    pool_gates = {}
    for name, control in config['pool_controls'].items():
        control_score = pool_scores[control] if control in pools else scores(y, predictions[control], folds)
        pool_gates[name] = {'control': control, 'control_scores': control_score, 'scores': pool_scores[name],
                            'passed': bool(eligible_change(control_score, pool_scores[name], config['pool_gate']))}
    evaluations = []
    for name in config['pools']:
        for alpha in config['alpha_grid']:
            value = scores(y, (1 - alpha) * anchor + alpha * pools[name], folds)
            incremental = bool(eligible_change(anchor_score, value, config['blend_gate']))
            evaluations.append({'pool': name, 'alpha': alpha, 'scores': value,
                                'pooled_gain': value['pooled'] - anchor_score['pooled'],
                                'mean_fold_gain': value['mean_fold'] - anchor_score['mean_fold'],
                                'fold_gains': (np.asarray(value['folds']) - anchor_score['folds']).tolist(),
                                'incremental_gate': incremental, 'eligible': incremental and pool_gates[name]['passed']})
    assert len(evaluations) == config['number_of_blend_tests'] == 12
    qualified = sorted([x for x in evaluations if x['eligible']],
                       key=lambda x: (-x['scores']['pooled'], -x['scores']['mean_fold'], x['alpha'], x['pool']))
    result = {'utc': now(), 'status': 'qualified_for_native_release' if qualified else 'no_qualifying_improvement',
              'protocol_sha256': PROTOCOL_SHA, 'screen_source_sha256': sha256(Path(__file__)),
              'anchor_scores': anchor_score, 'pool_gates': pool_gates, 'evaluations': evaluations,
              'winner': qualified[0] if qualified else None, 'training_performed': False,
              'audit_evaluated': False, 'submission_performed': False,
              'evaluation_note': config['evaluation_note']}
    atomic_json(OUT / 'screen.json', result)
    diagnostic = max(evaluations, key=lambda x: x['scores']['pooled'])
    print(json.dumps({'status': result['status'], 'winner': result['winner'],
                      'best_diagnostic': diagnostic, 'pool_gates': {k:v['passed'] for k,v in pool_gates.items()}}))


if __name__ == '__main__':
    main()
