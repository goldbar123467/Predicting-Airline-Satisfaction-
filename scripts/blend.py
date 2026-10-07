"""Constrained OOF blend; audit labels accessed for scoring only on finalization."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import ROOT,TARGET,atomic_csv,atomic_json,load_config,load_data,sha256

def checked_predictions(path, ids):
    frame=pd.read_parquet(path)
    assert frame.id.is_unique and len(frame)==len(ids)
    index=pd.Index(frame.id)
    order=index.get_indexer(ids)
    assert (order>=0).all(),f'Missing identifiers: {path}'
    p=frame.prediction.to_numpy(dtype=np.float64)[order]
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p

def optimize(y, matrix):
    individual=np.array([roc_auc_score(y,matrix[:,i]) for i in range(matrix.shape[1])])
    weights=np.zeros(matrix.shape[1]);weights[int(individual.argmax())]=1
    current=matrix@weights;score=float(individual.max())
    # Coarse convex additions limit selection flexibility. The audit is untouched.
    for alpha in [.5,.25,.1,.05,.02]:
        for _ in range(3):
            options=[float(roc_auc_score(y,(1-alpha)*current+alpha*matrix[:,i])) for i in range(matrix.shape[1])]
            winner=int(np.argmax(options))
            if options[winner] <= score+0.000002:break
            weights*=1-alpha;weights[winner]+=alpha
            current=matrix@weights;score=options[winner]
    return weights,score,individual

def group_projection(completed, configured_groups, known_run_ids):
    """Project fixed equal-average candidates back to original run columns."""
    if not isinstance(configured_groups, dict):
        raise ValueError('blend_groups must be a mapping from names to run-ID lists')
    known=set(known_run_ids)
    if len(known)!=len(known_run_ids):
        raise ValueError('Duplicate run IDs in the queue')
    if len(set(completed))!=len(completed) or not set(completed).issubset(known):
        raise ValueError('Completed run IDs must be unique members of the queue')
    claimed=set();active={};incomplete={}
    for name,members in configured_groups.items():
        if not isinstance(name,str) or not name.strip():
            raise ValueError('Blend group names must be nonempty strings')
        if not isinstance(members,list) or len(members)<2:
            raise ValueError(f'Blend group {name!r} needs at least two run IDs')
        if any(not isinstance(member,str) for member in members):
            raise ValueError(f'Blend group {name!r} contains a non-string run ID')
        if len(set(members))!=len(members):
            raise ValueError(f'Blend group {name!r} repeats a run ID')
        if not set(members).issubset(known):
            raise ValueError(f'Blend group {name!r} references an unknown run ID')
        if claimed.intersection(members):
            raise ValueError(f'Blend group {name!r} overlaps another group')
        claimed.update(members)
        destination=active if set(members).issubset(completed) else incomplete
        destination[name]=list(members)
    owner={member:name for name,members in active.items() for member in members}
    positions={name:index for index,name in enumerate(completed)}
    projection=[];candidates=[];emitted=set()
    for run_id in completed:
        if run_id in owner:
            group=owner[run_id]
            if group in emitted:continue
            emitted.add(group)
            members=active[group]
            candidate_id=f'group:{group}'
        else:
            members=[run_id]
            candidate_id=f'run:{run_id}'
        column=np.zeros(len(completed),dtype=np.float64)
        column[[positions[member] for member in members]]=1.0/len(members)
        projection.append(column)
        candidates.append({'candidate_id':candidate_id,'members':list(members)})
    return np.column_stack(projection),active,incomplete,candidates


def audit_sensitivity_mask(audit_ids, locked=None):
    """Record a smoke-test exposure incident without changing frozen splits."""
    path=ROOT/'data/audit_smoke_exclusions.json'
    recorded=None if locked is None else locked.get('audit_sensitivity')
    if locked is not None and recorded is None:
        return np.ones(len(audit_ids),dtype=bool),None
    if not path.exists():
        if recorded is not None:raise ValueError('Frozen audit sensitivity manifest is missing')
        return np.ones(len(audit_ids),dtype=bool),None
    digest=sha256(path)
    if recorded is not None and digest!=recorded['manifest_sha256']:
        raise ValueError('Frozen audit sensitivity manifest changed')
    incident=load_config(path)
    excluded=incident['excluded_audit_ids']
    if len(set(excluded))!=len(excluded) or not set(excluded).issubset(set(audit_ids)):
        raise ValueError('Audit sensitivity exclusions must be unique reserved audit IDs')
    keep=~np.isin(audit_ids,excluded)
    if not keep.any():raise ValueError('Audit sensitivity set is empty')
    return keep,{'manifest_sha256':digest,'excluded_rows':len(excluded),
                 'remaining_rows':int(keep.sum()),'reason':incident['reason']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--finalize',action='store_true');args=ap.parse_args()
    config=load_config(args.config)
    out=ROOT/'artifacts/blend';out.mkdir(parents=True,exist_ok=True)
    if args.finalize and (out/'frozen.json').exists():
        print('Frozen final selection already exists; audit will not be re-evaluated.',flush=True);return
    if not args.finalize and (out/'selection.json').exists():
        print('Selection frozen; preserving it.',flush=True);return
    locked=load_config(out/'selection.json') if (out/'selection.json').exists() else None
    tr,te,split=load_data();dev=split.fold.to_numpy()>=0;audit=~dev
    ids=tr.id.to_numpy()[dev];y=tr[TARGET].to_numpy()[dev]
    completed=[];columns=[];tests=[]
    for run in config['runs']:
        if locked is not None and run['id'] not in locked['candidate_ids']:continue
        path=ROOT/'artifacts/runs'/run['id']
        if not (path/'result.json').exists():continue
        result=load_config(path/'result.json')
        assert result['run']==run, 'Completed run config changed; use a new run id'
        assert result['split_hash']==sha256(ROOT/'data/splits.parquet')
        for name,digest in result['artifacts'].items():assert sha256(path/name)==digest, 'Completed prediction changed'
        frame=pd.read_parquet(path/'oof.parquet')
        assert np.array_equal(frame.id,ids) and np.array_equal(frame[TARGET],y) and np.array_equal(frame.fold,split.fold.to_numpy()[dev])
        columns.append(checked_predictions(path/'oof.parquet',ids))
        tests.append(checked_predictions(path/'test.parquet',te.id))
        completed.append(run['id'])
    if not completed:raise RuntimeError('No completed candidate to blend')
    matrix=np.column_stack(columns)
    # Preserve the unconstrained raw-model baseline even when a group average
    # replaces its members during optimization.
    individual=np.array([roc_auc_score(y,matrix[:,i]) for i in range(len(completed))])
    if locked is None:
        projection,active_groups,incomplete_groups,optimization_candidates=group_projection(
            completed,config.get('blend_groups',{}),[run['id'] for run in config['runs']])
        optimization_matrix=matrix@projection if active_groups else matrix
        candidate_weights,_,candidate_scores=optimize(y,optimization_matrix)
        weights=projection@candidate_weights
        auc=float(roc_auc_score(y,matrix@weights))
        for candidate,weight,score in zip(optimization_candidates,candidate_weights,candidate_scores):
            candidate.update(weight=float(weight),oof_auc=float(score))
    else:
        assert completed==locked['candidate_ids'], 'Frozen candidate set changed'
        weights=np.array([locked['weights'][r] for r in completed])
        auc=float(roc_auc_score(y,matrix@weights))
        # Current group configuration may have changed after selection. Replay
        # the saved constraints and expanded weights without selecting again.
        active_groups=locked.get('active_blend_groups',{})
        incomplete_groups=locked.get('incomplete_blend_groups',{})
        optimization_candidates=locked.get('optimization_candidates',[])
        for r in completed:assert sha256(ROOT/'artifacts/runs'/r/'test.parquet')==locked['source_predictions'][r]
    blended=matrix@weights
    test_pred=np.column_stack(tests)@weights
    folds=split.fold.to_numpy()[dev]
    audit_keep,audit_sensitivity=audit_sensitivity_mask(tr.id.to_numpy()[audit],locked)
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'candidate_ids':completed,'weights':dict(zip(completed,weights.tolist())),'oof_auc':auc,'best_single_oof_auc':float(individual.max()),'individual_oof_auc':dict(zip(completed,individual.tolist())),'fold_auc':[float(roc_auc_score(y[folds==k],blended[folds==k])) for k in range(3)],'split_hash':sha256(ROOT/'data/splits.parquet'),'selection':'coarse nonnegative convex weight search on development OOF; final audit excluded','source_predictions':{r:sha256(ROOT/'artifacts/runs'/r/'test.parquet') for r in completed}}
    metadata.update(active_blend_groups=active_groups,incomplete_blend_groups=incomplete_groups,
                    optimization_candidates=optimization_candidates,
                    blend_group_policy='Complete groups are fixed equal averages; incomplete group members remain independent')
    if audit_sensitivity is not None:metadata['audit_sensitivity']=audit_sensitivity
    pd.DataFrame({'id':ids,'prediction':blended}).to_parquet(out/'oof.parquet',index=False)
    atomic_csv(pd.DataFrame({'id':te.id,TARGET:test_pred}),out/'submission_current.csv')
    metadata['submission_hash']=sha256(out/'submission_current.csv')
    atomic_json(out/'current.json',metadata)
    if args.finalize:
        # Freeze membership, input hashes and weights before exposing any audit score.
        # A crash can repeat fixed arithmetic but cannot repeat model selection.
        if locked is None:atomic_json(out/'selection.json',metadata)
        audit_preds=np.column_stack([checked_predictions(ROOT/'artifacts/runs'/r/'audit.parquet',tr.id.to_numpy()[audit]) for r in completed])@weights
        ya=tr[TARGET].to_numpy()[audit]
        metadata['audit_auc']=float(roc_auc_score(ya,audit_preds))
        # Paired bootstrap uncertainty for fixed blend vs fixed best OOF single.
        best=int(individual.argmax())
        single=checked_predictions(ROOT/'artifacts/runs'/completed[best]/'audit.parquet',tr.id.to_numpy()[audit])
        rng=np.random.default_rng(119001);diff=[]
        for _ in range(100):
            take=rng.integers(0,len(ya),size=len(ya))
            diff.append(roc_auc_score(ya[take],audit_preds[take])-roc_auc_score(ya[take],single[take]))
        metadata['audit_best_oof_single_auc']=float(roc_auc_score(ya,single))
        metadata['audit_blend_minus_single_bootstrap_95pct']=np.quantile(diff,[.025,.975]).tolist()
        if audit_sensitivity is not None:
            metadata['audit_unexposed_auc']=float(roc_auc_score(ya[audit_keep],audit_preds[audit_keep]))
            metadata['audit_unexposed_best_oof_single_auc']=float(roc_auc_score(ya[audit_keep],single[audit_keep]))
        metadata['audit_evaluated_once']=True
        pd.DataFrame({'id':tr.id.to_numpy()[audit],'prediction':audit_preds}).to_parquet(out/'audit.parquet',index=False)
        atomic_csv(pd.DataFrame({'id':te.id,TARGET:test_pred}),out/'submission_fallback.csv')
        atomic_json(out/'frozen.json',metadata)
    print(metadata,flush=True)

if __name__=='__main__':main()
