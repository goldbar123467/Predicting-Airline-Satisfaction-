"""Audit-incident assessment contracts using synthetic data only."""
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import blend
import common
import test_release


class AuditSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_release.ReleaseContracts();self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root=self.fixture.root
        self.path=self.root/'data/audit_smoke_exclusions.json'
        common.atomic_json(self.path,{'excluded_audit_ids':[60,61],'reason':'synthetic smoke incident'})

    def test_assessment_is_preregistered_before_audit_scoring(self):
        with patch.object(sys,'argv',['blend.py','--config',str(self.fixture.config_path)]):
            blend.main()
        current=common.load_config(self.root/'artifacts/blend/current.json')
        self.assertEqual(current['audit_sensitivity']['remaining_rows'],58)
        self.assertNotIn('audit_auc',current)
        self.assertNotIn('audit_unexposed_auc',current)
        self.fixture.finalize()
        frozen=common.load_config(self.root/'artifacts/blend/frozen.json')
        prediction=pd.read_parquet(self.root/'artifacts/blend/audit.parquet')
        y=self.fixture.labels[self.fixture.dev_count:]
        self.assertAlmostEqual(frozen['audit_unexposed_auc'],roc_auc_score(y[2:],prediction.prediction.to_numpy()[2:]))
        self.assertEqual(current['weights'],frozen['weights'])

    def test_frozen_incident_cannot_be_replaced(self):
        ids=np.arange(60,120)
        mask,policy=blend.audit_sensitivity_mask(ids)
        np.testing.assert_array_equal(mask[:3],[False,False,True])
        common.atomic_json(self.path,{'excluded_audit_ids':[60,62],'reason':'changed after selection'})
        with self.assertRaisesRegex(ValueError,'changed'):
            blend.audit_sensitivity_mask(ids,{'audit_sensitivity':policy})
        keep,policy=blend.audit_sensitivity_mask(ids,{})
        self.assertTrue(keep.all());self.assertIsNone(policy)

    def test_non_audit_and_duplicate_ids_rejected(self):
        for excluded in [[0],[60,60]]:
            common.atomic_json(self.path,{'excluded_audit_ids':excluded,'reason':'invalid'})
            with self.assertRaisesRegex(ValueError,'unique reserved audit IDs'):
                blend.audit_sensitivity_mask(np.arange(60,120))


if __name__=='__main__':unittest.main()
