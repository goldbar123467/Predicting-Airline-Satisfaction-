"""Fixed seed-average constraints, using arrays and synthetic release fixtures."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import blend
import common
import test_release


class GroupProjectionTests(unittest.TestCase):
    def test_complete_group_has_one_column_and_equal_expanded_weights(self):
        completed=['a','b','c','d']
        projection,active,incomplete,candidates=blend.group_projection(
            completed,{'seeds':['a','c']},completed)
        np.testing.assert_array_equal(projection,[[.5,0,0],[0,1,0],[.5,0,0],[0,0,1]])
        self.assertEqual(active,{'seeds':['a','c']})
        self.assertEqual(incomplete,{})
        self.assertEqual([item['candidate_id'] for item in candidates],['group:seeds','run:b','run:d'])
        weights=projection@np.array([.6,.3,.1])
        np.testing.assert_allclose(weights,[.3,.3,.3,.1],rtol=0,atol=1e-15)
        raw=np.array([[.1,.2,.6,.9],[.8,.5,.4,.2]])
        expected=.6*(raw[:,0]+raw[:,2])/2+.3*raw[:,1]+.1*raw[:,3]
        np.testing.assert_allclose(raw@weights,expected,rtol=0,atol=1e-15)

    def test_incomplete_group_preserves_individual_columns(self):
        projection,active,incomplete,candidates=blend.group_projection(
            ['a','b'],{'seeds':['a','c']},['a','b','c'])
        np.testing.assert_array_equal(projection,np.eye(2))
        self.assertEqual(active,{})
        self.assertEqual(incomplete,{'seeds':['a','c']})
        self.assertEqual([item['members'] for item in candidates],[['a'],['b']])

    def test_invalid_constraints_are_rejected(self):
        invalid=[
            None, [], {'': ['a','b']}, {'seeds':'a,b'}, {'seeds':['a']},
            {'seeds':['a','a']}, {'seeds':['a','unknown']}, {'seeds':['a',2]},
            {'first':['a','b'],'second':['b','c']},
        ]
        for groups in invalid:
            with self.subTest(groups=groups),self.assertRaises(ValueError):
                blend.group_projection(['a','b'],groups,['a','b','c'])


class GroupReleaseTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_release.ReleaseContracts()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root=self.fixture.root

    def configure(self,groups):
        self.fixture.config['blend_groups']=groups
        common.atomic_json(self.fixture.config_path,self.fixture.config)

    def current(self):
        with patch.object(sys,'argv',['blend.py','--config',str(self.fixture.config_path)]):
            blend.main()
        return common.load_config(self.root/'artifacts/blend/current.json')

    def test_current_metadata_expands_weights_and_preserves_raw_baseline(self):
        members=['synthetic_a','synthetic_b']
        self.configure({'seed_average':members})
        result=self.current()
        self.assertEqual(result['weights'],dict.fromkeys(members,.5))
        self.assertEqual(result['active_blend_groups'],{'seed_average':members})
        self.assertEqual(len(result['optimization_candidates']),1)
        raw_scores={}
        expected=np.zeros(self.fixture.test_count)
        for name in members:
            base=self.root/'artifacts/runs'/name
            frame=pd.read_parquet(base/'oof.parquet')
            raw_scores[name]=roc_auc_score(frame[common.TARGET],frame.prediction)
            expected+=.5*pd.read_parquet(base/'test.parquet').prediction.to_numpy()
        self.assertEqual(result['individual_oof_auc'],raw_scores)
        self.assertEqual(result['best_single_oof_auc'],max(raw_scores.values()))
        submission=pd.read_csv(self.root/'artifacts/blend/submission_current.csv')
        np.testing.assert_allclose(submission[common.TARGET],expected,rtol=1e-10,atol=1e-11)

    def test_no_groups_and_incomplete_groups_match_existing_optimizer(self):
        reference=self.current()
        pending={'id':'not_finished','family':'synthetic','seed':9}
        self.fixture.config['runs'].append(pending)
        self.configure({'seed_average':['synthetic_a','not_finished']})
        incomplete=self.current()
        self.assertEqual(incomplete['weights'],reference['weights'])
        self.assertEqual(incomplete['oof_auc'],reference['oof_auc'])
        self.assertEqual(incomplete['active_blend_groups'],{})
        self.assertEqual(incomplete['incomplete_blend_groups'],{'seed_average':['synthetic_a','not_finished']})

    def test_frozen_replay_ignores_changed_and_even_invalid_group_configuration(self):
        groups={'fixed_seeds':['synthetic_a','synthetic_b']}
        self.configure(groups)
        original_atomic=blend.atomic_json

        def fail_publication(path:Path,value:dict):
            if path.name=='frozen.json':
                raise test_release.SimulatedPublicationFailure('Synthetic crash before final publication')
            original_atomic(path,value)

        with patch.object(blend,'atomic_json',side_effect=fail_publication):
            with self.assertRaises(test_release.SimulatedPublicationFailure):
                self.fixture.finalize()
        selected=common.load_config(self.root/'artifacts/blend/selection.json')
        self.fixture.add_candidate('synthetic_late',seed=891,perfect=True)
        self.configure({'invalid_after_freeze':['missing']})
        with patch.object(blend,'optimize',side_effect=AssertionError('Frozen replay must not optimize')) as optimizer, \
                patch.object(blend,'group_projection',side_effect=AssertionError('Frozen replay must not use new group config')) as projection:
            self.fixture.finalize()
            optimizer.assert_not_called()
            projection.assert_not_called()
        frozen=common.load_config(self.root/'artifacts/blend/frozen.json')
        for field in ['candidate_ids','weights','active_blend_groups','incomplete_blend_groups','optimization_candidates']:
            self.assertEqual(frozen[field],selected[field])
        self.assertEqual(frozen['active_blend_groups'],groups)


if __name__=='__main__':
    unittest.main(verbosity=2)
