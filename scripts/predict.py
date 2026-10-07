"""Run a saved probability ensemble on raw passenger rows, without refitting."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,TARGET,features,load_config,load_model,predict,sha256

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',default=str(ROOT/'configs/overnight.json'))
    parser.add_argument('--selection',type=Path,help='Explicit saved blend manifest for reproducible inference')
    parser.add_argument('--fold-ensemble',action='store_true',help='Use evaluated fold models even if full-data refits exist')
    args=parser.parse_args()
    config=load_config(args.config)
    data_contract=load_config(ROOT/'data/manifest.json')
    raw=pd.read_csv(args.input,dtype={'id':str})
    required=['id']+data_contract['feature_columns']
    assert all(c in raw for c in required),'Input lacks required passenger columns'
    assert raw.id.notna().all() and raw.id.is_unique and len(raw)>0
    raw=raw[required].copy()
    for c in data_contract['feature_columns']:
        if pd.api.types.is_numeric_dtype(raw[c]):raw[c]=raw[c].astype('float32')
    directory=ROOT/'artifacts/blend'
    selection_path=args.selection or directory/('frozen.json' if (directory/'frozen.json').exists() else 'current.json')
    selection=load_config(selection_path)
    full=(ROOT/'artifacts/final/manifest.json').exists() and not args.fold_ensemble
    if full:
        manifest=load_config(ROOT/'artifacts/final/manifest.json')
        assert manifest['frozen_selection_hash']==sha256(selection_path)
    probability=np.zeros(len(raw),dtype=np.float64)
    teacher_probability=None
    for run_id,weight in selection['weights'].items():
        if weight<=0:continue
        run=next(r for r in config['runs'] if r['id']==run_id)
        saved=load_config(ROOT/'artifacts/runs'/run_id/'result.json')
        assert saved['run']==run,'Config changed since fitting'
        # Compute the original-data teacher from its saved native model. No cached
        # competition IDs or competition labels are needed for arbitrary new rows.
        feature_run={**run,'teacher':False}
        x=features(raw,feature_run)
        if run.get('teacher',False):
            if teacher_probability is None:
                teacher,transform=load_model(ROOT/'artifacts/teacher')
                teacher_probability=predict(teacher,transform,raw[data_contract['feature_columns']])
                del teacher,transform;gc.collect()
            x['teacher_probability']=teacher_probability
        base=ROOT/'artifacts'/('final' if full else 'runs')/run_id
        paths=[base/'model'] if full else [base/f'fold_{k}'/'model' for k in range(3)]
        for path in paths:
            model,transform=load_model(path)
            probability+=weight*predict(model,transform,x)/len(paths)
            del model,transform;gc.collect()
        print(f'Predicted {run_id}: weight={weight:.6f}',flush=True)
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'id':raw.id,TARGET:probability}).to_csv(args.output,index=False,float_format='%.12g')
    print(f'Wrote {len(raw):,} probabilities to {args.output}; sha256={sha256(args.output)}',flush=True)

if __name__=='__main__':main()
