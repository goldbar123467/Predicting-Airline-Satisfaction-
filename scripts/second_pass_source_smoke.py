"""Development-only source-feature integration and raw/native parity smoke."""
from pathlib import Path
import gc
import tempfile

import numpy as np
import pandas as pd

from common import ROOT, features, fit_model, load_config, load_model, predict
from original_aux import predict_aux
from original_lgb_teacher import predict_teacher


def main():
    splits = pd.read_parquet(ROOT / 'data/splits.parquet')
    ids = splits.loc[splits.fold >= 0, 'id']
    # Select only development IDs before sampling or accessing targets.
    train = pd.read_parquet(ROOT / 'data/train.parquet', filters=[('id', 'in', ids.tolist())])
    sample = train.sample(1024, random_state=620103)
    del train, splits, ids
    original = next(r for r in load_config(ROOT / 'configs/second_pass.json')['runs']
                    if r['id'] == 'realmlp_cat_te_teacher')
    run = {**original, 'original_aux': True, 'original_lgb_teacher': True, 'max_rounds': 1}
    run['params'] = {'device': 'cuda', 'n_ens': 2, 'hidden_sizes': [32, 16],
                     'batch_size': 128, 'eval_batch_size': 128, 'threads': 2, 'ls_eps': 0.0}
    x = features(sample, run)
    auxiliary = predict_aux(sample, device='cpu')
    np.testing.assert_allclose(x[list(auxiliary)], auxiliary, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x['original_lgb_teacher_probability'], predict_teacher(sample), rtol=1e-6, atol=6e-8)
    y = sample.satisfaction.to_numpy()
    with tempfile.TemporaryDirectory(prefix='v2-source-smoke-') as directory:
        path = Path(directory)
        model, transform, rounds = fit_model(x.iloc[:900], y[:900], x.iloc[900:], y[900:], run, path)
        assert rounds == 1
        before = predict(model, transform, x.iloc[900:])
        restored, preprocessing = load_model(path)
        after = predict(restored, preprocessing, x.iloc[900:])
        np.testing.assert_allclose(after, before, rtol=1e-5, atol=2e-6)
        assert np.isfinite(after).all() and ((after >= 0) & (after <= 1)).all()
        print('PASS: keyed source caches, raw native feature reconstruction, CUDA RealMLP integration and native reload', flush=True)
        del model, transform, restored, preprocessing
        gc.collect()


if __name__ == '__main__':
    main()
