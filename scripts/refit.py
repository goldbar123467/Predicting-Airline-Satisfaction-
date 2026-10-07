"""Refit each frozen member in a fresh process, then package the full-data blend."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from common import ROOT,TARGET,atomic_csv,atomic_json,features,fit_model,load_config,load_data,predict,sha256


def release_context(config_path: Path):
    """Read release identities without materializing training features or labels."""
    config=load_config(config_path)
    frozen_path=ROOT/'artifacts/blend/frozen.json'
    frozen=load_config(frozen_path)
    frozen_hash=sha256(frozen_path)
    manifest=load_config(ROOT/'data/manifest.json')
    if frozen['split_hash']!=manifest['split_hash']:
        raise ValueError('Frozen selection does not match the data split')
    train_rows=int(manifest['rows']['train'])
    test_ids=pd.read_parquet(ROOT/'data/test.parquet',columns=['id']).id.to_numpy()
    if train_rows<=0 or len(test_ids)!=int(manifest['rows']['test']) or not pd.Index(test_ids).is_unique:
        raise ValueError('Invalid release row counts or test IDs')
    runs={run['id']:run for run in config['runs']}
    if len(runs)!=len(config['runs']):
        raise ValueError('Duplicate run IDs in the queue')
    weights=frozen['weights']
    if not weights or any(not np.isfinite(weight) or weight<0 for weight in weights.values()):
        raise ValueError('Frozen weights must be finite and nonnegative')
    if abs(sum(weights.values())-1)>1e-9 or not set(weights).issubset(runs):
        raise ValueError('Invalid frozen weight total or run IDs')
    specs=[]
    for run_id,weight in weights.items():
        if weight<=0:continue
        run=runs[run_id]
        cv_path=ROOT/'artifacts/runs'/run_id/'result.json'
        cv=load_config(cv_path)
        if cv['run']!=run or cv['split_hash']!=frozen['split_hash']:
            raise ValueError(f'Run {run_id} changed after OOF; use a new run ID')
        rounds=np.asarray(cv['rounds'],dtype=np.float64)
        if rounds.ndim!=1 or not len(rounds) or not np.isfinite(rounds).all() or (rounds<1).any():
            raise ValueError(f'Invalid recorded rounds for {run_id}')
        specs.append({'id':run_id,'run':run,'weight':float(weight),
                      'rounds':max(1,int(np.median(rounds))),'cv_result_hash':sha256(cv_path)})
    if not specs:
        raise ValueError('Frozen selection has no positive-weight members')
    return frozen,frozen_hash,train_rows,test_ids,specs


def validate_member(spec,train_rows,test_ids,frozen_hash):
    """Verify a complete member before resuming or aggregating its predictions."""
    dest=ROOT/'artifacts/final'/spec['id']
    done=load_config(dest/'done.json')
    expected={'id':spec['id'],'rounds':spec['rounds'],'weight':spec['weight'],'rows':train_rows}
    if any(done.get(key)!=value for key,value in expected.items()):
        raise ValueError(f'Refit completion contract mismatch for {spec["id"]}')
    # Older valid done.json files lack these extra identities. Their model run,
    # selected rounds, weight, row counts and prediction hash are still checked.
    optional={'run':spec['run'],'cv_result_hash':spec['cv_result_hash'],
              'frozen_selection_hash':frozen_hash}
    if any(key in done and done[key]!=value for key,value in optional.items()):
        raise ValueError(f'Refit source identity mismatch for {spec["id"]}')
    model_path=dest/'model/model_metadata.json'
    model=load_config(model_path)
    if model.get('run')!=spec['run'] or model.get('rounds')!=spec['rounds'] or model.get('family')!=spec['run']['family']:
        raise ValueError(f'Refit model configuration mismatch for {spec["id"]}')
    if 'model_metadata_hash' in done and sha256(model_path)!=done['model_metadata_hash']:
        raise ValueError(f'Refit model metadata checksum mismatch for {spec["id"]}')
    path=dest/'test.parquet'
    if sha256(path)!=done['test_hash']:
        raise ValueError(f'Refit prediction checksum mismatch for {spec["id"]}')
    frame=pd.read_parquet(path,columns=['id','prediction'])
    if not frame.id.is_unique or not np.array_equal(frame.id.to_numpy(),test_ids):
        raise ValueError(f'Refit prediction ID/order mismatch for {spec["id"]}')
    probability=frame.prediction.to_numpy(dtype=np.float64)
    if probability.shape!=(len(test_ids),) or not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all():
        raise ValueError(f'Invalid refit probabilities for {spec["id"]}')
    return probability


def refit_one(spec,train_rows,test_ids,frozen_hash):
    dest=ROOT/'artifacts/final'/spec['id']
    if (dest/'done.json').exists():
        validate_member(spec,train_rows,test_ids,frozen_hash)
        print(f'REFIT ALREADY COMPLETE {spec["id"]}',flush=True)
        return
    tr,te,_=load_data()
    if len(tr)!=train_rows or not np.array_equal(te.id.to_numpy(),test_ids):
        raise ValueError('Worker data does not match the release row contract')
    run=spec['run']
    print(f'FULL REFIT {spec["id"]} rounds={spec["rounds"]}, weight={spec["weight"]:.6f}',flush=True)
    x=features(tr,run);xt=features(te,run)
    model,transform,actual_rounds=fit_model(
        x,tr[TARGET].to_numpy(),None,None,run,dest/'model',rounds=spec['rounds'])
    if actual_rounds!=spec['rounds']:
        raise ValueError('Fixed-round refit returned a different round count')
    probability=predict(model,transform,xt)
    if sha256(ROOT/'artifacts/blend/frozen.json')!=frozen_hash:
        raise ValueError('Frozen selection changed while refitting a member')
    path=dest/'test.parquet';temporary=dest/'test.tmp.parquet'
    pd.DataFrame({'id':test_ids,'prediction':probability}).to_parquet(temporary,index=False)
    os.replace(temporary,path)
    atomic_json(dest/'done.json',{
        'id':spec['id'],'rounds':spec['rounds'],'weight':spec['weight'],'rows':train_rows,
        'run':run,'cv_result_hash':spec['cv_result_hash'],'frozen_selection_hash':frozen_hash,
        'test_hash':sha256(path),'model_metadata_hash':sha256(dest/'model/model_metadata.json')})
    validate_member(spec,train_rows,test_ids,frozen_hash)
    print(f'FULL REFIT COMPLETE {spec["id"]}',flush=True)


def coordinate(config_path,frozen,frozen_hash,train_rows,test_ids,specs):
    out=ROOT/'artifacts/final';out.mkdir(parents=True,exist_ok=True)
    blended=np.zeros(len(test_ids),dtype=np.float64)
    members=[]
    for spec in specs:
        if not (out/spec['id']/'done.json').exists():
            command=[str(Path(sys.executable).resolve()),'-u',str(Path(__file__).resolve()),
                     '--config',str(config_path),'--run-id',spec['id']]
            print(f'START ISOLATED REFIT {spec["id"]}',flush=True)
            # Inherit supervisor log handles. A nonzero worker exit propagates;
            # completed members remain resumable and no final manifest is issued.
            subprocess.run(command,cwd=ROOT,check=True,stdout=sys.stdout,stderr=sys.stderr,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        probability=validate_member(spec,train_rows,test_ids,frozen_hash)
        blended+=spec['weight']*probability
        members.append({'id':spec['id'],'weight':spec['weight'],'rounds':spec['rounds'],
                        'path':str(out/spec['id']/'model')})
    # Recheck identity at publication in case an external actor edited the queue
    # or frozen selection while the long-running workers were active.
    latest={run['id']:run for run in load_config(config_path)['runs']}
    if any(latest.get(spec['id'])!=spec['run'] for spec in specs):
        raise ValueError('Selected run configuration changed during refit')
    if sha256(ROOT/'artifacts/blend/frozen.json')!=frozen_hash:
        raise ValueError('Frozen selection changed during refit')
    if not np.isfinite(blended).all() or not ((blended>=0)&(blended<=1)).all():
        raise ValueError('Invalid final blend probabilities')
    atomic_csv(pd.DataFrame({'id':test_ids,TARGET:blended}),out/'submission.csv')
    atomic_json(out/'manifest.json',{
        'members':members,'submission_hash':sha256(out/'submission.csv'),
        'frozen_selection_hash':frozen_hash,'audit_auc':frozen['audit_auc'],
        'audit_unexposed_auc':frozen.get('audit_unexposed_auc'),
        'audit_sensitivity':frozen.get('audit_sensitivity'),
        'evaluation_note':'Audit measures dev-only fold ensemble; full-data refit includes audit labels only after frozen assessment.',
        'refit_execution':'one fresh subprocess per selected member'})
    print('Full-data refit and final submission complete.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--run-id',help='Refit exactly this positive-weight frozen member in this worker')
    args=parser.parse_args()
    config_path=args.config.resolve()
    frozen,frozen_hash,train_rows,test_ids,specs=release_context(config_path)
    if args.run_id:
        matching=[spec for spec in specs if spec['id']==args.run_id]
        if len(matching)!=1:
            raise ValueError('--run-id must name one positive-weight frozen member')
        refit_one(matching[0],train_rows,test_ids,frozen_hash)
    else:
        coordinate(config_path,frozen,frozen_hash,train_rows,test_ids,specs)


if __name__=='__main__':main()
