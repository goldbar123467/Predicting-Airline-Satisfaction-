"""Freeze raw hashes and disjoint development/audit/outer fold identities."""
import json
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from common import ROOT, TARGET, atomic_json, sha256

def main():
    tr=pd.read_csv(ROOT/'data/train.csv')
    te=pd.read_csv(ROOT/'data/test.csv')
    sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    assert tr.id.is_unique and te.id.is_unique
    assert set(tr.id).isdisjoint(te.id)
    assert list(sample.columns)==['id',TARGET] and np.array_equal(sample.id,te.id)
    assert tr[TARGET].isin([False,True]).all()
    assert list(tr.drop(columns=TARGET).columns)==list(te.columns)
    feature_cols=[c for c in tr if c not in ['id',TARGET]]
    assert not tr[feature_cols].duplicated().any(), 'Need grouped folds for duplicate features'
    tr[TARGET]=tr[TARGET].astype('int8')
    for df in [tr,te]:
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(df[c]): df[c]=df[c].astype('float32')
    dest=ROOT/'data/splits.parquet'
    assert not dest.exists(), 'Frozen split already exists; do not overwrite'
    dev,audit=train_test_split(np.arange(len(tr)),test_size=.1,random_state=8675309,stratify=tr[TARGET])
    fold=np.full(len(tr),-1,dtype=np.int8)
    skf=StratifiedKFold(3,shuffle=True,random_state=20261001)
    for k,(_,val) in enumerate(skf.split(dev,tr[TARGET].to_numpy()[dev])): fold[dev[val]]=k
    split=pd.DataFrame({'id':tr.id,'fold':fold,'is_audit':fold<0})
    tr.to_parquet(ROOT/'data/train.parquet',index=False)
    te.to_parquet(ROOT/'data/test.parquet',index=False)
    split.to_parquet(dest,index=False)
    atomic_json(ROOT/'data/manifest.json',{'competition':'playground-series-s6e10','metric':'roc_auc','positive_label':True,'rows':{'train':len(tr),'test':len(te),'development':len(dev),'audit':len(audit)},'feature_columns':feature_cols,'raw_hashes':{p.name:sha256(p) for p in (ROOT/'data').glob('*.csv')},'split_hash':sha256(dest),'audit_seed':8675309,'fold_seed':20261001,'python':platform.python_version(),'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','catboost','xgboost']}})
    print(json.dumps({'prepared':True,'dev':len(dev),'audit':len(audit)}),flush=True)

if __name__=='__main__': main()
