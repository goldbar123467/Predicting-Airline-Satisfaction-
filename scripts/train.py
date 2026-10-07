"""One resumable experiment: inner stopping, outer refit, and honest OOF."""
from __future__ import annotations
import argparse
import gc
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from common import ROOT,TARGET,atomic_json,features,fit_model,load_config,load_data,predict,sha256

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    run=next(r for r in config['runs'] if r['id']==args.run_id)
    out=ROOT/'artifacts/runs'/run['id']
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        complete=load_config(out/'result.json')
        assert complete['run']==run, 'Completed run config changed; use a new run id'
        assert complete['split_hash']==sha256(ROOT/'data/splits.parquet')
        for name,digest in complete['artifacts'].items():assert sha256(out/name)==digest, 'Completed prediction changed'
        print('Already complete',flush=True); return
    code_paths=[ROOT/'scripts/train.py',ROOT/'scripts/common.py']
    if run['family']=='tabm': code_paths.append(ROOT/'scripts/neural.py')
    if run['family']=='realmlp':code_paths.append(ROOT/'scripts/realmlp.py')
    if run['family'] in {'tabm_cat','realmlp_cat'}:
        module='neural_categorical.py' if run['family']=='tabm_cat' else 'realmlp_categorical.py'
        base_module='neural.py' if run['family']=='tabm_cat' else 'realmlp.py'
        code_paths.extend([ROOT/'scripts'/module,ROOT/'scripts'/base_module,ROOT/'scripts/categorical_transform.py'])
    if run.get('encoding'):code_paths.append(ROOT/'scripts/encoding.py')
    if run.get('route_profiles'):code_paths.append(ROOT/'scripts/route_profiles.py')
    for flag in ['original_aux', 'original_lgb_teacher', 'original_aux_probability', 'original_realmlp_teacher']:
        if run.get(flag):code_paths.append(ROOT/'scripts'/f'{flag}.py')
    signature={'version':2,'run':run,'split_hash':sha256(ROOT/'data/splits.parquet'),
               'data_hashes':{p.name:sha256(p) for p in [ROOT/'data/train.parquet',ROOT/'data/test.parquet',ROOT/'data/manifest.json']},
               'code_hashes':{p.name:sha256(p) for p in code_paths},
               'environment_hash':sha256(ROOT/'requirements.lock.txt')}
    if run.get('teacher',False):signature['teacher_hash']=sha256(ROOT/'data/teacher_predictions.parquet')
    for flag in ['original_aux', 'original_lgb_teacher', 'original_aux_probability', 'original_realmlp_teacher']:
        if run.get(flag):
            bank_path=ROOT/'artifacts'/flag/'manifest.json'
            bank=load_config(bank_path)
            signature[flag+'_hash']=sha256(ROOT/'data'/f'{flag}_predictions.parquet')
            if bank.get('status')!='complete' or bank['output_sha256']!=signature[flag+'_hash']:
                raise ValueError(f'Incomplete or changed {flag} feature bank')
            signature[flag+'_manifest_hash']=sha256(bank_path)
    if (out/'contract.json').exists(): assert load_config(out/'contract.json')==signature,'Changed run configuration; use new id'
    else:
        atomic_json(out/'contract.json',signature)
        (out/'source').mkdir(exist_ok=True)
        for p in code_paths+[ROOT/'requirements.lock.txt']:shutil.copy2(p,out/'source'/p.name)
    tr,te,split=load_data()
    x=features(tr,run); xtest=features(te,run)
    y=tr[TARGET].to_numpy()
    fold=split.fold.to_numpy()
    audit=np.flatnonzero(fold<0)
    dev=np.flatnonzero(fold>=0)
    oof=np.full(len(tr),np.nan,dtype=np.float64)
    pa=np.zeros(len(audit)); pt=np.zeros(len(te))
    scores=[]; rounds=[]
    start=time.monotonic()
    for k in range(3):
        fd=out/f'fold_{k}'
        cache=fd/'predictions.npz'
        val=np.flatnonzero(fold==k)
        train=np.flatnonzero((fold>=0)&(fold!=k))
        if cache.exists() and (fd/'done.json').exists():
            done=load_config(fd/'done.json')
            assert sha256(cache)==done['artifact_hash'],'Cached fold checksum mismatch'
            pred=np.load(cache,allow_pickle=False)
            assert np.array_equal(pred['validation_ids'],tr.id.to_numpy()[val])
            for key,size in [('valid',len(val)),('audit',len(audit)),('test',len(te))]:
                assert pred[key].shape==(size,) and np.isfinite(pred[key]).all()
                assert ((pred[key]>=0)&(pred[key]<=1)).all()
            oof[val]=pred['valid']; pa+=pred['audit']/3; pt+=pred['test']/3
            pred.close()
            scores.append(done['auc']); rounds.append(done['rounds'])
            print(f'Resumed {run["id"]} fold {k}',flush=True); continue
        inner,stop=train_test_split(train,test_size=.1,random_state=run.get('seed',20261001)+k,stratify=y[train])
        print(f'START {run["id"]} fold {k} inner={len(inner)} stop={len(stop)} outer={len(val)}',flush=True)
        model,transform,n=fit_model(x.iloc[inner],y[inner],x.iloc[stop],y[stop],run,fd/'inner')
        del model,transform; gc.collect()
        print(f'REFIT {run["id"]} fold {k} fixed_rounds={n}',flush=True)
        model,transform,_=fit_model(x.iloc[train],y[train],None,None,run,fd/'model',rounds=n)
        pval=predict(model,transform,x.iloc[val]); p_audit=predict(model,transform,x.iloc[audit]); p_test=predict(model,transform,xtest)
        score=float(roc_auc_score(y[val],pval))
        tmp=fd/'predictions.tmp.npz'
        np.savez_compressed(tmp,validation_ids=tr.id.to_numpy()[val],valid=pval,audit=p_audit,test=p_test)
        tmp.replace(cache)
        atomic_json(fd/'done.json',{'auc':score,'rounds':n,'inner_train_rows':len(inner),'inner_stop_rows':len(stop),'outer_train_rows':len(train),'outer_validation_rows':len(val),'artifact_hash':sha256(cache)})
        oof[val]=pval; pa+=p_audit/3; pt+=p_test/3
        scores.append(score); rounds.append(n)
        del model,transform;gc.collect()
        print(f'COMPLETE {run["id"]} fold {k} AUC={score:.8f}',flush=True)
    assert np.isfinite(oof[dev]).all()
    pd.DataFrame({'id':tr.id.to_numpy()[dev],'fold':fold[dev],TARGET:y[dev],'prediction':oof[dev]}).to_parquet(out/'oof.parquet',index=False)
    pd.DataFrame({'id':tr.id.to_numpy()[audit],'prediction':pa}).to_parquet(out/'audit.parquet',index=False)
    pd.DataFrame({'id':te.id,'prediction':pt}).to_parquet(out/'test.parquet',index=False)
    result={'id':run['id'],'family':run['family'],'oof_auc':float(roc_auc_score(y[dev],oof[dev])),'fold_auc':scores,'rounds':rounds,'seconds':time.monotonic()-start,'run':run,'split_hash':signature['split_hash'],'contract_hash':sha256(out/'contract.json'),'artifacts':{p.name:sha256(p) for p in out.glob('*.parquet')}}
    atomic_json(out/'result.json',result)
    print(result,flush=True)

if __name__=='__main__':main()
