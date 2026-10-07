# Fixed dropout retry: passive inner endpoint review

This is a descriptive review of already-saved inner metrics at epochs **4 and 16 only**, after the frozen outer assessment. No outer predictions, raw training/audit data, model files or network services were read. No AUC/log-loss computation, best-epoch search, new selection or fit occurred. The analysis uses only JSON/JSONL and Python standard-library arithmetic.

The pattern is consistent across all three folds: both arms substantially improve their fixed training-probe metrics between epochs 4 and 16 while both held-out metrics deteriorate. Holding dropout at 0.05 after epoch 4 reduces that deterioration relative to the original schedule, but does not recover the epoch-4 held-out performance. This supports **partial mitigation of overfitting under this recipe**, not a successful solution to the longer-training problem.

Each fit has 377,802 fitting rows, a fixed 2,048-row training probe, and 41,978 or 41,979 passive inner-monitor rows. The reported probe log loss is the saved evaluation-mode probability metric, not the stochastic train-mode objective. A and C have exactly equal saved epoch-4 metrics in every fold. Each curve file's hash matches its completed H16/E16 trajectory receipt; all selected endpoint observations report unchanged live state.

Unweighted means of the three saved fold metrics are below. These are descriptive macro means, not pooled AUCs.

| Endpoint | Training-probe AUC | Training-probe log loss | Inner-monitor AUC | Inner-monitor log loss |
|---|---:|---:|---:|---:|
| Shared A4/C4 | 0.965940801 | 0.213770076 | 0.961001223 | 0.222796245 |
| A16, original dropout schedule | 0.995828163 | 0.092142424 | 0.954361496 | 0.260715629 |
| C16, dropout held at 0.05 after epoch 4 | 0.993257128 | 0.111166566 | 0.955419573 | 0.253621663 |

For A, epoch 16 versus 4 lowers monitor AUC by **0.006639727** and increases monitor log loss by **0.037919383**. For C, the corresponding changes are **−0.005581650** and **+0.030825418**. C16 versus A16 improves monitor AUC by **0.001058077** and lowers monitor log loss by **0.007093965**. The treatment also fits the training probe less closely at epoch 16, consistent with a regularization effect.

| Fold | Shared A4/C4 monitor AUC | A16 monitor AUC | C16 monitor AUC | C16−A16 AUC | C16−A16 log loss |
|---|---:|---:|---:|---:|---:|
| 0 | 0.961178446 | 0.954433493 | 0.954890763 | +0.000457271 | −0.005989772 |
| 1 | 0.961841490 | 0.955297203 | 0.956326289 | +0.001029085 | −0.006767481 |
| 2 | 0.959983734 | 0.953353793 | 0.955041668 | +0.001687875 | −0.008524643 |

Both ranking and probability loss worsen on the monitor between the two endpoints, so the observed failure is not merely an AUC-versus-log-loss tradeoff. These endpoints do not identify when deterioration begins, the best stopping epoch, an optimal dropout probability, or which other mechanism would fix it. No conclusion about all longer training schedules follows. The 2,048-row in-sample probe is not an estimate of full-training performance, and its level should not be treated as population evidence.

This reuses development data and follows the frozen assessment; it is **not independent confirmation** and supplies no additional uncertainty interval or promotion decision. It cannot justify reweighting the incumbent, replacing the frozen acceptance rule, or launching another experiment. The matched intervention supports the narrower within-run comparison of C16 with A16; it does not establish a unique mechanistic explanation for every observed difference.

## Input hashes

All paths below are relative to `cloud/fixed_epoch_dropout_v1_retry1/assessment_workspace/artifacts/fixed_epoch_dropout_v1/`.

| File | SHA256 |
|---|---|
| fold_0/inner/A/curves.jsonl | `7a7d3bd21baad85ea0fa06c830ba3eeeb17cd14135a1f05f7f86e2e9d60574b3` |
| fold_0/inner/A/trajectory.json | `88c676485cee7f4517eea1ad5c5674fa5216d285549d6124662abdcdb2a43f4b` |
| fold_0/inner/C/curves.jsonl | `aef2ed7f1b2a00f8ca792f7e497fc3ba1a940839bd3a671ebb6e5bfb2974fee9` |
| fold_0/inner/C/trajectory.json | `cfba61ba1708eaf926003161724c8331832e3c3075b04ac9fa5d8bb382f0a942` |
| fold_1/inner/A/curves.jsonl | `c8b67415875a93c9c0a13afc01173d2fbfcc3500727a724f92a6ff4f7f1d1d0c` |
| fold_1/inner/A/trajectory.json | `88e048319342ae7533fcbed05c02ab0897dcc9950ef4b02683639e32b8709629` |
| fold_1/inner/C/curves.jsonl | `a06ab87e62577345dde5397a03a3bfb563cea8a757554f9b350168a1ab672399` |
| fold_1/inner/C/trajectory.json | `8f69f1f4ccc205d0a5f633d69d7eeb541f934edbce98b5099fdc95a81be1c217` |
| fold_2/inner/A/curves.jsonl | `94f73a626fa7c48f4d88e79b043482c7e0f0eab9e880147014ff07a36e1219f1` |
| fold_2/inner/A/trajectory.json | `3beacb840d779cc1e10b5ebf1814c9c0335c665f405c8e6ea057663343ae9480` |
| fold_2/inner/C/curves.jsonl | `ad938f6a6de175d060d639c0425071a8b51ed00f7091cc1bf728f8f0fdd48580` |
| fold_2/inner/C/trajectory.json | `c480d7202fae42509f92583d99f842000312e0ce10eff6125cf963f5cefe24fa` |

## Exact executed summary command

Run from `C:\Users\thecl\Documents\Predicting Airline Satisfaction` in PowerShell. It completed successfully in approximately 0.35 seconds. The output contains all twelve endpoint rows, fold deltas, means and hashes. This command parses the saved curve stream, retaining metric values only for the two specified epochs; it does not inspect any alternative endpoint for selection.

```powershell
@'
from pathlib import Path
import hashlib,json,math,statistics
root=Path.cwd()
base=root/'cloud/fixed_epoch_dropout_v1_retry1/assessment_workspace/artifacts/fixed_epoch_dropout_v1'
rows=[];hashes={}
for fold in range(3):
 for arm in ('A','C'):
  path=base/f'fold_{fold}/inner/{arm}/curves.jsonl'
  with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
  hashes[path.relative_to(root).as_posix()]=digest
  trajectory=path.with_name('trajectory.json')
  with trajectory.open('rb') as stream: hashes[trajectory.relative_to(root).as_posix()]=hashlib.file_digest(stream,'sha256').hexdigest()
  detail=json.loads(trajectory.read_text())
  assert detail['status']=='complete' and detail['executed_epochs']==16 and detail['horizon']==16 and detail['curves_sha256']==digest
  selected={}
  with path.open() as stream:
   for line in stream:
    value=json.loads(line)
    if value['epoch'] not in (4,16):continue
    assert value['epoch'] not in selected
    assert value['context']['phase']=='inner' and value['context']['fold']==fold and value['context']['trajectory']==arm
    assert value['endpoint_live_state_unchanged'] and all(value['endpoint_live_state_unchanged'].values())
    entry={'fold':fold,'trajectory':arm,'epoch':value['epoch'],'probe_rows':detail['fit_probe_rows'],'fit_rows':value['train_rows'],'monitor_rows':value['context']['monitor_rows'],'train_probe_auc':value['train_eval_probe']['auc'],'train_probe_log_loss':value['train_eval_probe']['log_loss'],'monitor_auc':value['monitor']['auc'],'monitor_log_loss':value['monitor']['log_loss']}
    assert all(math.isfinite(entry[key]) for key in ('train_probe_auc','train_probe_log_loss','monitor_auc','monitor_log_loss'))
    selected[value['epoch']]=entry
  assert set(selected)=={4,16}
  rows.extend(selected[e] for e in (4,16))
lookup={(v['fold'],v['trajectory'],v['epoch']):v for v in rows}
metrics=['train_probe_auc','train_probe_log_loss','monitor_auc','monitor_log_loss']
for fold in range(3):
 assert all(lookup[fold,'A',4][metric]==lookup[fold,'C',4][metric] for metric in metrics)
means=[{'trajectory':arm,'epoch':epoch,**{metric:statistics.fmean(lookup[fold,arm,epoch][metric] for fold in range(3)) for metric in metrics}} for arm in ('A','C') for epoch in (4,16)]
deltas=[]
for fold in range(3):
 for arm in ('A','C'):
  deltas.append({'fold':fold,'trajectory':arm,**{metric:lookup[fold,arm,16][metric]-lookup[fold,arm,4][metric] for metric in metrics}})
comparison=[{'fold':fold,**{metric:lookup[fold,'C',16][metric]-lookup[fold,'A',16][metric] for metric in metrics}} for fold in range(3)]
result={'scope':'Already-saved passive inner metrics at fixed epochs4/16 only; descriptive reuse after frozen assessment; no new scoring or best-epoch search','rows':rows,'macro_means':means,'epoch16_minus_epoch4':deltas,'C16_minus_A16':comparison,'source_hashes':hashes}
print(json.dumps(result,indent=2))
'@ | .venv/Scripts/python.exe -
```
