"""Isolated-refit coordinator regressions with synthetic artifacts only."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import common
import refit


class IsolatedRefitTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='airline-isolated-refit-test-')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.object(refit,'ROOT',self.root))
        stack.enter_context(patch.object(common,'ROOT',self.root))
        stack.enter_context(redirect_stdout(io.StringIO()))
        self.config_path=self.root/'config.json'
        self.runs=[{'id':'first','family':'synthetic','seed':7},
                   {'id':'second','family':'synthetic','seed':8}]
        self.config={'runs':self.runs}
        self.ids=np.array([301,107,802,456,999])
        self.predictions={'first':np.array([.1,.3,.5,.7,.9]),
                          'second':np.array([.2,.4,.6,.8,.95])}
        self.train_rows=37
        self.frozen={'weights':{'first':.25,'second':.75},'split_hash':'synthetic-split',
                     'audit_auc':.8,'candidate_ids':['first','second']}
        (self.root/'data').mkdir()
        # No train.parquet exists: the coordinator cannot load training labels
        # or features, even accidentally, in these regressions.
        pd.DataFrame({'id':self.ids}).to_parquet(self.root/'data/test.parquet',index=False)
        common.atomic_json(self.root/'data/manifest.json',{
            'rows':{'train':self.train_rows,'test':len(self.ids)},'split_hash':'synthetic-split'})
        common.atomic_json(self.root/'artifacts/blend/frozen.json',self.frozen)
        common.atomic_json(self.config_path,self.config)
        for index,run in enumerate(self.runs):
            common.atomic_json(self.root/'artifacts/runs'/run['id']/'result.json',{
                'run':run,'rounds':[2+index,4+index,6+index],'split_hash':'synthetic-split'})
        fallback=self.root/'artifacts/blend/submission_fallback.csv'
        fallback.write_text('synthetic preserved fallback\n',encoding='utf-8')
        self.fallback_hash=common.sha256(fallback)

    def specs(self):
        return refit.release_context(self.config_path)[-1]

    def complete(self,run_id):
        spec=next(value for value in self.specs() if value['id']==run_id)
        dest=self.root/'artifacts/final'/run_id
        common.atomic_json(dest/'model/model_metadata.json',{
            'family':'synthetic','run':spec['run'],'rounds':spec['rounds']})
        pd.DataFrame({'id':self.ids,'prediction':self.predictions[run_id]}).to_parquet(dest/'test.parquet',index=False)
        common.atomic_json(dest/'done.json',{
            'id':run_id,'rounds':spec['rounds'],'weight':spec['weight'],'rows':self.train_rows,
            'run':spec['run'],'cv_result_hash':spec['cv_result_hash'],
            'frozen_selection_hash':common.sha256(self.root/'artifacts/blend/frozen.json'),
            'test_hash':common.sha256(dest/'test.parquet'),
            'model_metadata_hash':common.sha256(dest/'model/model_metadata.json')})

    def invoke(self,run_id=None):
        command=['refit.py','--config',str(self.config_path)]
        if run_id:command.extend(['--run-id',run_id])
        with patch.object(sys,'argv',command):refit.main()

    def test_failed_child_preserves_fallback_and_publishes_no_final_package(self):
        with patch.object(refit.subprocess,'run',side_effect=subprocess.CalledProcessError(3,['synthetic'])) as worker, \
                patch.object(refit,'load_data',side_effect=AssertionError('Coordinator cannot load training data')):
            with self.assertRaises(subprocess.CalledProcessError):self.invoke()
            self.assertEqual(worker.call_count,1)
        self.assertFalse((self.root/'artifacts/final/manifest.json').exists())
        self.assertFalse((self.root/'artifacts/final/submission.csv').exists())
        self.assertEqual(common.sha256(self.root/'artifacts/blend/submission_fallback.csv'),self.fallback_hash)

    def test_complete_members_resume_without_starting_workers(self):
        for run in self.runs:self.complete(run['id'])
        with patch.object(refit.subprocess,'run',side_effect=AssertionError('Completed member must not refit')) as worker, \
                patch.object(refit,'load_data',side_effect=AssertionError('Coordinator cannot load training data')):
            self.invoke()
            worker.assert_not_called()
        self.assertTrue((self.root/'artifacts/final/manifest.json').exists())

    def test_only_missing_member_runs_and_package_preserves_ids_weights_and_hashes(self):
        self.complete('first')

        def synthetic_worker(command,**kwargs):
            self.assertIsInstance(command,list)
            self.assertEqual(command[0],str(Path(sys.executable).resolve()))
            self.assertEqual(command[-2:],['--run-id','second'])
            self.assertTrue(kwargs['check'])
            self.assertEqual(kwargs['cwd'],self.root)
            self.assertNotIn('shell',kwargs)
            self.assertIs(kwargs['stdout'],sys.stdout)
            self.assertIs(kwargs['stderr'],sys.stderr)
            self.complete('second')
            return subprocess.CompletedProcess(command,0)

        with patch.object(refit.subprocess,'run',side_effect=synthetic_worker) as worker, \
                patch.object(refit,'load_data',side_effect=AssertionError('Coordinator cannot load training data')):
            self.invoke()
            self.assertEqual(worker.call_count,1)
        out=self.root/'artifacts/final'
        submission=pd.read_csv(out/'submission.csv')
        np.testing.assert_array_equal(submission.id,self.ids)
        np.testing.assert_allclose(submission[common.TARGET],
                                   .25*self.predictions['first']+.75*self.predictions['second'],rtol=1e-10,atol=1e-11)
        manifest=common.load_config(out/'manifest.json')
        self.assertEqual([(item['id'],item['weight'],item['rounds']) for item in manifest['members']],
                         [('first',.25,4),('second',.75,5)])
        self.assertEqual(manifest['submission_hash'],common.sha256(out/'submission.csv'))
        self.assertEqual(manifest['frozen_selection_hash'],common.sha256(self.root/'artifacts/blend/frozen.json'))

    def test_corrupt_completion_and_reordered_predictions_are_rejected(self):
        self.complete('first');self.complete('second')
        dest=self.root/'artifacts/final/first'
        original=common.load_config(dest/'done.json')
        mutations=[{'id':'other'},{'rows':99},{'weight':.5},{'rounds':9},
                   {'run':{**self.runs[0],'seed':999}},{'test_hash':'wrong'}]
        for change in mutations:
            with self.subTest(change=change),patch.object(refit.subprocess,'run') as worker:
                common.atomic_json(dest/'done.json',{**original,**change})
                with self.assertRaises(ValueError):self.invoke()
                worker.assert_not_called()
                self.assertFalse((self.root/'artifacts/final/manifest.json').exists())
        frame=pd.read_parquet(dest/'test.parquet').iloc[::-1]
        frame.to_parquet(dest/'test.parquet',index=False)
        common.atomic_json(dest/'done.json',{**original,'test_hash':common.sha256(dest/'test.parquet')})
        with self.assertRaisesRegex(ValueError,'ID/order mismatch'):self.invoke()

    def test_worker_fits_only_requested_member_at_fixed_round_count(self):
        train=pd.DataFrame({'id':np.arange(self.train_rows),common.TARGET:np.arange(self.train_rows)%2,
                            'numeric':np.arange(self.train_rows,dtype=float)})
        test=pd.DataFrame({'id':self.ids,'numeric':np.arange(len(self.ids),dtype=float)})

        def synthetic_fit(x,y,valid,y_valid,run,out,rounds):
            self.assertEqual(run['id'],'second')
            self.assertEqual(len(x),self.train_rows)
            self.assertIsNone(valid);self.assertIsNone(y_valid)
            self.assertEqual(rounds,5)
            common.atomic_json(out/'model_metadata.json',{'family':'synthetic','run':run,'rounds':rounds})
            return object(),object(),rounds

        with patch.object(refit,'load_data',return_value=(train,test,None)), \
                patch.object(refit,'fit_model',side_effect=synthetic_fit) as fit, \
                patch.object(refit,'predict',return_value=self.predictions['second']), \
                patch.object(refit.subprocess,'run',side_effect=AssertionError('Worker must not spawn another refit')):
            self.invoke('second')
            self.assertEqual(fit.call_count,1)
        self.assertTrue((self.root/'artifacts/final/second/done.json').exists())
        self.assertFalse((self.root/'artifacts/final/first/done.json').exists())
        self.assertFalse((self.root/'artifacts/final/manifest.json').exists())

    def test_real_subprocesses_use_distinct_workers_with_synthetic_model_backend(self):
        scripts=self.root/'scripts';scripts.mkdir()
        (scripts/'refit.py').write_text(Path(refit.__file__).read_text(encoding='utf-8'),encoding='utf-8')
        source=Path(common.__file__).read_text(encoding='utf-8')
        # Override only the model backend in a temporary common.py. The actual
        # coordinator/worker CLI, subprocess handling, IO and checks run intact.
        source+='''
def fit_model(x,y,valid,yv,run,out,rounds=None):
    import psutil
    assert valid is None and yv is None
    atomic_json(out/'model_metadata.json',{'family':run['family'],'run':run,'rounds':rounds})
    process=psutil.Process()
    atomic_json(out/'worker_identity.json',{'pid':process.pid,'create_time':process.create_time()})
    return run,None,rounds

def predict(model,transform,x):
    return np.full(len(x),model['seed']/10,dtype=np.float64)
'''
        (scripts/'common.py').write_text(source,encoding='utf-8')
        train=pd.DataFrame({'id':np.arange(self.train_rows),common.TARGET:np.arange(self.train_rows)%2,
                            'numeric':np.arange(self.train_rows,dtype=float)})
        train.to_parquet(self.root/'data/train.parquet',index=False)
        pd.DataFrame({'id':train.id,'fold':np.arange(self.train_rows)%3}).to_parquet(
            self.root/'data/splits.parquet',index=False)
        pd.DataFrame({'id':self.ids,'numeric':np.arange(len(self.ids),dtype=float)}).to_parquet(
            self.root/'data/test.parquet',index=False)
        completed=subprocess.run([sys.executable,'-u',str(scripts/'refit.py'),
                                  '--config',str(self.config_path)],cwd=self.root,
                                 check=True,capture_output=True,text=True,timeout=45,
                                 creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        self.assertEqual(completed.stdout.count('FULL REFIT COMPLETE'),2,completed.stdout+completed.stderr)
        identities=[common.load_config(self.root/'artifacts/final'/run['id']/'model/worker_identity.json')
                    for run in self.runs]
        self.assertEqual(len({(item['pid'],item['create_time']) for item in identities}),2)
        submission=pd.read_csv(self.root/'artifacts/final/submission.csv')
        np.testing.assert_array_equal(submission.id,self.ids)
        np.testing.assert_allclose(submission[common.TARGET],.25*.7+.75*.8,rtol=1e-10,atol=1e-11)


if __name__=='__main__':unittest.main(verbosity=2)
