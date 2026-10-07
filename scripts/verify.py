"""Independent keyed blend recomputation plus native model reload inference."""
import argparse
import gc
import shutil
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from common import ROOT,TARGET,atomic_json,features,load_config,load_data,load_model,predict,sha256

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    config=load_config(args.config);tr,te,_=load_data()
    final=ROOT/'artifacts/final';blend=ROOT/'artifacts/blend'
    full=(final/'manifest.json').exists()
    if full:
        final_manifest=load_config(final/'manifest.json')
        assert final_manifest['frozen_selection_hash']==sha256(blend/'frozen.json'), 'Final package belongs to a different selection'
        assert final_manifest['submission_hash']==sha256(final/'submission.csv')
    selection=load_config(blend/('frozen.json' if (blend/'frozen.json').exists() else 'current.json'))
    path=final/'submission.csv' if full else blend/('submission_fallback.csv' if (blend/'frozen.json').exists() else 'submission_current.csv')
    source_hash=sha256(path)
    assert source_hash==(final_manifest['submission_hash'] if full else selection['submission_hash']), 'Submission and selection snapshots differ'
    submission=pd.read_csv(path);sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    assert list(submission.columns)==list(sample.columns)==['id',TARGET]
    assert len(submission)==len(te) and submission.id.is_unique
    assert np.array_equal(submission.id,sample.id) and np.array_equal(submission.id,te.id)
    values=submission[TARGET].to_numpy();assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
    expected=np.zeros(len(te));reload=np.zeros(min(256,len(te)))
    for run_id,weight in selection['weights'].items():
        if weight<=0:continue
        run=next(r for r in config['runs'] if r['id']==run_id)
        base=final/run_id if full else ROOT/'artifacts/runs'/run_id
        frame=pd.read_parquet(base/'test.parquet');assert np.array_equal(frame.id,te.id)
        expected+=weight*frame.prediction.to_numpy()
        xt=features(te.iloc[:len(reload)],run)
        model_paths=[base/'model'] if full else [base/f'fold_{k}'/'model' for k in range(3)]
        member=np.zeros(len(reload))
        for model_path in model_paths:
            model,transform=load_model(model_path)
            member+=predict(model,transform,xt)/len(model_paths)
            del model,transform;gc.collect()
        assert np.allclose(member,frame.prediction.to_numpy()[:len(reload)],rtol=2e-5,atol=2e-6),f'Reload mismatch {run_id}'
        reload+=weight*member
    assert np.allclose(values,expected,rtol=1e-10,atol=1e-11),'Blend recomputation failed'
    assert np.allclose(reload,values[:len(reload)],rtol=2e-5,atol=2e-6)
    assert sha256(path)==source_hash, 'Submission changed during verification; rerun on a stable snapshot'
    result={'verified_utc':datetime.now(timezone.utc).isoformat(),'submission_path':str(path),'rows':len(submission),'sha256':sha256(path),'full_data_refit':full,'native_model_reload_rows':len(reload),'independent_blend_recomputation':True,'oof_auc':selection['oof_auc'],'audit_auc':selection.get('audit_auc'),'weights':selection['weights']}
    result.update(audit_unexposed_auc=selection.get('audit_unexposed_auc'),
                  audit_sensitivity=selection.get('audit_sensitivity'))
    # Preserve a verified byte-for-byte fallback while later blends are explored.
    # Content-addressed paths keep earlier verified packages recoverable.
    archive=ROOT/'artifacts/verified'/source_hash
    archive.mkdir(parents=True,exist_ok=True)
    temporary=archive/'submission.tmp.csv'
    shutil.copyfile(path,temporary)
    assert sha256(temporary)==source_hash, 'Submission changed while archiving verification'
    temporary.replace(archive/'submission.csv')
    atomic_json(archive/'selection.json',selection)
    result['verified_archive']=str(archive)
    atomic_json(archive/'verification.json',result)
    atomic_json(ROOT/'artifacts/verification.json',result)
    report=f"# Airline satisfaction overnight result\n\nVerified: {result['verified_utc']}\n\nSubmission: `{path}` ({len(submission):,} rows).\n\nDevelopment OOF ROC AUC: {selection['oof_auc']:.8f}.\n\n"
    report+=f"Final audit ROC AUC: {selection.get('audit_auc', 'not evaluated yet')}. Full-data refit: {full}.\n\n"
    report+=f"Completed candidate configurations: {len(selection['candidate_ids'])}. Positive-weight configurations: {sum(w>0 for w in selection['weights'].values())}. Best individual development OOF ROC AUC: {selection['best_single_oof_auc']:.8f}.\n\n"
    if selection.get('audit_sensitivity'):
        incident=selection['audit_sensitivity']
        report+=f"Audit sensitivity ROC AUC excluding {incident['excluded_rows']} smoke-test exposure rows: {selection.get('audit_unexposed_auc', 'not evaluated yet')} ({incident['remaining_rows']:,} rows). {incident['reason']}\n\n"
    report+="Weights:\n\n"+''.join(f"- {r}: {w:.6f}\n" for r,w in selection['weights'].items() if w>0)
    report+="\nVerified exact schema, identifiers and order, finite probabilities, independently recomputed blend, and saved-model reload/inference on 256 rows. No Kaggle submission or public score claimed. Audit score applies to development-only fold ensemble. Full-data refits include audit labels only after the selection was frozen. GPU reproducibility is not guaranteed bitwise.\n"
    report+="\nDevelopment OOF is the model-selection score and is optimistic after repeated comparisons. The audit evaluates fixed models and weights; the unexposed sensitivity audit excludes the recorded smoke-test rows. Full-data refit predictions have no separately measured score.\n"
    report+="\nReproduce inference from raw passenger rows in PowerShell:\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts/predict.py --input data/test.csv --output artifacts/reproduced.csv\n```\n\nAdd `--fold-ensemble` to use the evaluated development-fold models. See [RUN_PLAN.md](RUN_PLAN.md), [competition research](research/competition_evidence.md), and [method research](research/method_evidence.md) for the validation design, experiment evidence, sources, and reproduction details.\n"
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    print(result,flush=True)

if __name__=='__main__':main()
