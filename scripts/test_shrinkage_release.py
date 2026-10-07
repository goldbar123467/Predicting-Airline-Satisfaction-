"""Synthetic cloud import pins and reused-control proofs; no fitting or API."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import common
import third_pass_release as release
import test_third_pass_release as fixture


class ShrinkageReleaseTests(unittest.TestCase):
    setUp = fixture.ThirdPassReleaseTests.setUp
    write_config = fixture.ThirdPassReleaseTests.write_config
    add_candidate = fixture.ThirdPassReleaseTests.add_candidate
    create_native_member = fixture.ThirdPassReleaseTests.create_native_member
    verify_cloud_fixture = fixture.ThirdPassReleaseTests.verify_cloud_fixture

    def configure(self):
        real=Path(__file__).resolve().parents[1]/'configs/third_pass_lgb_shrinkage.json'
        self.protocol=json.loads(real.read_text())
        recipe=self.protocol['runs'][0]
        self.run=self.add_candidate(recipe['id'],**{k:v for k,v in recipe.items() if k!='id'})
        self.receipt=self.verify_cloud_fixture(self.run)
        self.protocol_path=self.root/'configs/third_pass_lgb_shrinkage.json'
        common.atomic_json(self.protocol_path,self.protocol)
        digest=common.sha256(self.protocol_path)
        for key,value in [('SHRINKAGE_PROTOCOL_SHA',digest),('SHRINKAGE_RUNTIME_SHA','synthetic-runtime'),
                          ('SHRINKAGE_BUNDLE_SHA','synthetic-bundle')]:
            self.stack.enter_context(patch.object(release,key,value))
        self.receipt.update(protocol_id=self.protocol['protocol_id'],protocol_sha256=digest,
            execution_image=self.protocol['execution_image'],runtime_sha256='synthetic-runtime',
            bundle_manifest_sha256='synthetic-bundle',reused_control=copy.deepcopy(self.protocol['reused_control']),
            lightgbm_version='4.7.0',execution_threads=4,training_device='cpu',all_test_fold_native_inference=True)
        self.receipt_path=self.root/'artifacts/runs'/self.run['id']/'cloud_import_verification.json'
        common.atomic_json(self.receipt_path,self.receipt)
        self.split_hash=common.sha256(self.root/'data/splits.parquet')

    def check(self):return release.checked_cloud_import(self.run,self.split_hash)

    def test_positive_exact_pins_control_and_native_proof(self):
        self.configure();self.assertEqual(self.check(),self.receipt)

    def test_protocol_control_platform_and_native_test_fields_bound(self):
        self.configure()
        changes={'protocol_id':'other','protocol_sha256':'wrong','reused_control':{},
                 'lightgbm_version':'4.6.0','execution_threads':8,'training_device':'gpu',
                 'all_test_fold_native_inference':False,'runtime_sha256':'wrong',
                 'bundle_manifest_sha256':'wrong','execution_image':'wrong','heldout_rows':59}
        for key,value in changes.items():
            common.atomic_json(self.receipt_path,{**self.receipt,key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
        common.atomic_json(self.receipt_path,self.receipt)
        self.protocol_path.write_bytes(self.protocol_path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'protocol'):self.check()

    def test_absent_reviewed_pins_or_changed_recipe_rejected(self):
        self.configure()
        for key in ['SHRINKAGE_RUNTIME_SHA','SHRINKAGE_BUNDLE_SHA']:
            with patch.object(release,key,None),self.subTest(key=key),self.assertRaises(ValueError):self.check()
        for key,value in [('seed',1),('max_rounds',2600),('patience',150)]:
            old=self.run[key];self.run[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            self.run[key]=old

    def test_source_native_replay_tampering_cannot_reuse_receipt(self):
        self.configure()
        for field in ['native_hashes','source_hashes','heldout_replay_receipt_hashes']:
            common.atomic_json(self.receipt_path,{**self.receipt,field:{}})
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'provenance'):self.check()
        common.atomic_json(self.receipt_path,self.receipt)
        path=self.root/next(iter(self.receipt['native_hashes']))
        path.write_text('tampered')
        with self.assertRaisesRegex(ValueError,'Changed cloud'):self.check()


if __name__=='__main__':unittest.main()
