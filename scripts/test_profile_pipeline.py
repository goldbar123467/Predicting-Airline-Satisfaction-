"""Synthetic end-to-end checks for composed route-profile/target-encoding IO."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from common import CAT, RATINGS, fit_model, load_model, predict
from route_profiles import DEFAULT_COLUMNS


class ProfilePipelineTests(unittest.TestCase):
    def test_native_reload_and_frozen_validation_profiles(self):
        rng = np.random.default_rng(9180)
        x = pd.DataFrame({c: rng.integers(0, 6, 384).astype('float32') for c in RATINGS})
        x['Age'] = rng.integers(18, 90, len(x)).astype('float32')
        x['Flight Distance'] = rng.choice([100., 200., 300.], len(x)).astype('float32')
        for c in ['Departure Delay in Minutes', 'Arrival Delay in Minutes']:
            x[c] = rng.exponential(5, len(x)).astype('float32')
        for c in CAT:
            x[c] = rng.choice(['a', 'b'], len(x))
        y = ((x['Age'] > 40) ^ (x[RATINGS[0]] > 3)).to_numpy(dtype='int8')
        valid = x.iloc[:48].copy()
        valid['Flight Distance'] = 9999.
        valid['Age'] = 1e6
        for family in ['lightgbm', 'xgboost']:
            with self.subTest(family=family), tempfile.TemporaryDirectory(prefix='profile-pipeline-') as directory:
                run = {'id': 'synthetic_profiles', 'family': family, 'route_profiles': True,
                       'encoding': {'keys': ['Flight Distance'], 'n_splits': 3},
                       'seed': 9180, 'max_rounds': 8, 'patience': 3,
                       'params': {'learning_rate': .1}}
                model, transform, rounds = fit_model(x, y, valid, y[:48], run, Path(directory))
                expected = np.asarray([x[c].mean() for c in DEFAULT_COLUMNS], dtype='float32')
                actual = transform.encoder.transform(valid)[transform.encoder.output_columns].to_numpy()
                np.testing.assert_allclose(actual, np.tile(expected, (len(valid), 1)), rtol=1e-6)
                self.assertNotIn(9999., transform.encoder.map_.index)
                before = predict(model, transform, valid)
                restored, preprocessing = load_model(Path(directory))
                np.testing.assert_allclose(predict(restored, preprocessing, valid), before, atol=1e-7)
                self.assertGreater(rounds, 0)
                self.assertTrue(np.isfinite(before).all())
                # Reversed rows and duplicate indexes exercise every composed stage.
                reordered = valid.iloc[[8, 1, 8, 3]]
                np.testing.assert_allclose(predict(restored, preprocessing, reordered), before[[8, 1, 8, 3]], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
