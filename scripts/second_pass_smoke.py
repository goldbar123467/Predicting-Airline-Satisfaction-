"""Bounded development-only smoke for RealMLP + route profiles composition."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from common import ROOT, features, fit_model, load_config, load_model, predict


def main():
    split = pd.read_parquet(ROOT / 'data/splits.parquet')
    ids = split.loc[split.fold >= 0, 'id']
    # Filter by frozen development IDs before sampling or accessing labels.
    data = pd.read_parquet(ROOT / 'data/train.parquet', filters=[('id', 'in', ids.tolist())])
    assert set(data.id).issubset(set(ids))
    sample = data.sample(1024, random_state=620102)
    del data, split, ids
    run = dict(next(r for r in load_config(ROOT / 'configs/second_pass.json')['runs']
                    if r['id'] == 'v2_realmlp_cat_te_teacher_profiles'))
    run.update(max_rounds=1, params={'device': 'cuda', 'n_ens': 2,
               'hidden_sizes': [32, 16], 'batch_size': 128,
               'eval_batch_size': 128, 'threads': 2, 'ls_eps': 0.0})
    x = features(sample, run)
    y = sample.satisfaction.to_numpy()
    with tempfile.TemporaryDirectory(prefix='v2-profile-smoke-') as directory:
        path = Path(directory)
        model, transform, rounds = fit_model(x.iloc[:900], y[:900], x.iloc[900:], y[900:], run, path)
        assert rounds == 1
        before = predict(model, transform, x.iloc[900:])
        restored, preprocessing = load_model(path)
        after = predict(restored, preprocessing, x.iloc[900:])
        np.testing.assert_allclose(after, before, rtol=1e-5, atol=2e-6)
        assert np.isfinite(after).all() and ((after >= 0) & (after <= 1)).all()
        print('PASS: development-only RealMLP/profile/TE CUDA fit and native reload', flush=True)


if __name__ == '__main__':
    main()
