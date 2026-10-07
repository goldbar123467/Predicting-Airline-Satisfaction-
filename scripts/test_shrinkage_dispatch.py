"""Isolated CPU shrinkage dispatch tests; all remote methods are mocked."""
import copy
from datetime import datetime, timezone
import unittest
from unittest.mock import patch
import kaggle_private_job as dispatch
import test_kaggle_production_dispatch as fixture


class ShrinkageDispatchTests(unittest.TestCase):
    setUp = fixture.ProductionDispatchTests.setUp
    sha = staticmethod(fixture.ProductionDispatchTests.sha)
    write = staticmethod(fixture.ProductionDispatchTests.write)
    bind_freeze = fixture.ProductionDispatchTests.bind_freeze
    save = fixture.ProductionDispatchTests.save
    invoke = fixture.ProductionDispatchTests.invoke
    assert_no_dispatch = fixture.ProductionDispatchTests.assert_no_dispatch

    def configure(self, production=False):
        self.protocol = {'protocol_id':'lgb_shrinkage_v1',
                         'candidate_id':'v3_cloud_lgb_route_teacher_aux_probability_lr01',
                         'execution_image':'synthetic-cpu-image'}
        self.protocol_path = self.root/'configs/third_pass_lgb_shrinkage.json'
        self.write(self.protocol_path,self.protocol)
        digest=self.sha(self.protocol_path)
        self.stack.enter_context(patch.object(dispatch,'SHRINKAGE_PROTOCOL_SHA',digest))
        self.folder=self.root/('cloud/third_pass_lgb_shrinkage/refits/test' if production else 'cloud/third_pass_lgb_shrinkage')
        self.folder.mkdir(parents=True)
        (self.folder/'job.py').write_text('# SYNTHETIC ONLY\n')
        self.meta.update(enable_gpu=False,docker_image=self.protocol['execution_image'])
        self.meta.pop('machine_shape',None)
        self.prepared.update(protocol_sha256=digest,runtime_sha256=self.sha(self.folder/'job.py'))
        if production:
            name=self.protocol['candidate_id']
            self.freeze={'weights':{name:.1},'source_result_hashes':{name:'a'*64}}
            self.spec.update(run_id=name,protocol_id=self.protocol['protocol_id'],protocol_sha256=digest,
                             timeout_seconds=900,execution_image=self.protocol['execution_image'])
            self.bind_freeze()
        else:
            self.meta.update(id='clarkkitchen/s6e10-shrinkage-synthetic',competition_sources=[])
            self.save()
        fixture.FixedDateTime.fixed=datetime(2026,10,2,15,45,tzinfo=timezone.utc)

    def test_cv_latest_start_1550_exact_timeout_and_no_gpu_quota(self):
        self.configure()
        with patch.object(fixture.FixedDateTime,'fixed',datetime(2026,10,2,15,50,1,tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError,'cutoff'):self.invoke(production=False,timeout=1800)
        self.assert_no_dispatch()
        with patch.object(fixture.FixedDateTime,'fixed',datetime(2026,10,2,15,50,tzinfo=timezone.utc)):
            self.invoke(production=False,timeout=1800)
        self.api.kernels_push.assert_called_once_with(str(self.folder),timeout='1800',acc=None)
        self.api.quota_view.assert_not_called()

    def test_cv_source_protocol_and_platform_fail_closed(self):
        self.configure()
        for timeout in [1200,1799,1801]:
            with self.subTest(timeout=timeout),self.assertRaisesRegex(ValueError,'CV dispatch'):
                self.invoke(production=False,timeout=timeout)
        for mapping,key,value in [(self.meta,'enable_gpu',True),(self.meta,'docker_image','wrong'),
                                  (self.meta,'competition_sources',['playground-series-s6e10']),
                                  (self.prepared,'runtime_sha256','wrong'),(self.prepared,'protocol_sha256','wrong')]:
            old=copy.deepcopy(mapping[key]);mapping[key]=value;self.save()
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'CV dispatch'):
                self.invoke(production=False,timeout=1800)
            mapping[key]=old
        self.save();self.protocol_path.write_bytes(self.protocol_path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'protocol changed'):self.invoke(production=False,timeout=1800)
        self.assert_no_dispatch()

    def test_production_latest_start_1625_and_exact_900_seconds(self):
        self.configure(True)
        with patch.object(fixture.FixedDateTime,'fixed',datetime(2026,10,2,16,25,1,tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError,'cutoff'):self.invoke(timeout=900)
        for timeout in [600,899,901,1500]:
            with self.subTest(timeout=timeout),self.assertRaisesRegex(ValueError,'contract differs'):self.invoke(timeout=timeout)
        self.assert_no_dispatch()
        with patch.object(fixture.FixedDateTime,'fixed',datetime(2026,10,2,16,25,tzinfo=timezone.utc)):
            self.invoke(timeout=900)
        self.api.kernels_push.assert_called_once_with(str(self.folder),timeout='900',acc=None)
        self.api.quota_view.assert_not_called()

    def test_production_freeze_platform_and_scope_tamper(self):
        self.configure(True)
        for mapping,key,value in [(self.spec,'protocol_id','lgb_rating_probability_v1'),
                (self.spec,'protocol_sha256','wrong'),(self.spec,'execution_image','wrong'),
                (self.spec,'timeout_seconds',600),(self.prepared,'frozen_selection_hash','wrong'),
                (self.spec,'selection_frozen_sha256','wrong'),(self.spec,'cv_result_sha256','wrong'),
                (self.prepared,'runtime_sha256','wrong'),(self.meta,'enable_gpu',True)]:
            old=copy.deepcopy(mapping[key]);mapping[key]=value;self.save()
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'contract differs'):self.invoke(timeout=900)
            mapping[key]=old
        for weight in [0,-1,float('nan'),float('inf')]:
            self.freeze['weights'][self.protocol['candidate_id']]=weight;self.bind_freeze()
            with self.subTest(weight=weight),self.assertRaisesRegex(ValueError,'contract differs'):self.invoke(timeout=900)
        self.assert_no_dispatch()

    def test_folder_prevents_protocol_and_candidate_downgrade(self):
        self.configure(True)
        self.spec.update(run_id='v3_renamed',protocol_id='generic')
        self.freeze={'weights':{'v3_renamed':.1},'source_result_hashes':{'v3_renamed':'a'*64}}
        self.bind_freeze()
        for timeout in [600,900,1500]:
            with self.subTest(timeout=timeout),self.assertRaisesRegex(ValueError,'contract differs'):self.invoke(timeout=timeout)
        self.assert_no_dispatch()


if __name__=='__main__':unittest.main()
