# Dropout continuation experiment: completed, advancement gate failed

Verified October 4, 2026. All 12 real-data fits, 192 epochs, nine outer endpoints, six exact full-state prefix gates, three full-fold native-prefix checks and fourteen deterministic startup roles passed. The frozen evaluator ran once and completed at **20:07:47 UTC**. Training succeeded; the candidate failed the registered model-quality gate. **Keep the current blend unchanged.** No further training, weight search, release, refit or submission is pending.

Holding dropout at 0.05 after epoch 4 improved the epoch-16 model relative to its matched control, but did not preserve the shared epoch-4 model's quality. Treatment pooled AUC was **0.955469095**, control **0.954543419**, and shared prefix **0.960546132**. The fixed 10% treatment mixture scored **0.961360067**, below the incumbent **0.961407501**. This intervention partially reduced continuation degradation but did not produce a useful addition under the preregistered criterion.

## Design and verified execution

Both trajectories used schedule horizon 16 and executed exactly 16 epochs. A retained the original dropout schedule. C matched it through completed epoch 4, then held the base dropout probability at 0.05 for epochs 5 through 16, preserving unrelated layer/scope multipliers. Learning-rate and weight-decay schedules, the existing duplicate decay-factor behavior, architecture 512/256/128 with eight members, optimizer, features, preprocessing, seeds and consumed batch order were held fixed. Inner monitoring was passive; no early stopping or best-checkpoint selection occurred.

The replay design used three frozen development folds, inner/outer fits and two trajectories: 12 fits and 192 real-data epochs. Logical endpoints were A=control16, B=shared prefix4 and C=treatment16. All 629,671 development rows matched canonical IDs/folds exactly; audit and test rows were excluded. There were 209,891, 209,890 and 209,890 scored rows per fold.

Each of the six matched prefixes passed exact equality for all 19 training-state components before treatment epoch 5, including network, optimizer, gradients, RNGs, preprocessing, schedules and sampler/order state. All 16 epochs' paired non-dropout schedules and consumed order matched. Effective dropout values matched the intervention at every recorded update. Each inner trajectory recorded 23,600 updates, each outer trajectory 26,224; the exact shared prefixes contained 5,900 and 6,556 updates respectively.

All three full-fold native A4/C4 comparisons had maximum absolute probability difference 0.0. All nine full-fold native reload checks passed. These compare cloud native reloads; adapter-versus-export parity was checked on fixed probes at capture. Local evaluation verified receipts and hashes without loading or running native models. Fourteen startup receipts proved strict deterministic settings and complete role coverage: smoke, controller and 12 workers. Full cross-platform determinism is not established.

The mandatory production-sized generated CUDA smoke passed in 56.21 seconds with four generated fits/64 epochs, exact paired native prefix probabilities, native reload and CPU/GPU tolerance checks. These generated fits are separate from the 12 real-data fits.

The sole successful private Kaggle job was `clarkkitchen/s6e10-dropout-deterministic-retry-20261004`, kernel **137074361**, immutable version **1**. It was dispatched at 18:41:38 UTC; the cloud wrapper ran **18:41:52.209259 to 19:54:03.412442 UTC**, or **4,331.203 seconds (72.19 minutes)** before final return packaging. The sum of the twelve adapter durations was **3,718.322 seconds (61.97 minutes)**. These are measured execution durations, not provider billing claims. It finished inside the 6,300-second setup/smoke/fit budget and 6,900-second provider cap. The provider exposed two Tesla T4 devices; all twelve fits used `cuda:0`, one worker at a time. No local GPU training occurred.

Kaggle canonicalized the requested `-r2` slug from the title. The provider reconciliation bound the actual private ref, numeric identity, unchanged image/dataset and exact API source. No duplicate push or dataset reupload occurred. The protected cloud stack remained Torch 2.10.0+cu128, NumPy 2.0.2, pandas 2.3.3 and scikit-learn 1.6.1; fitted code used PyTabKit 1.7.3 and Lightning 2.6.6. The CPU-only evaluator used NumPy 2.5.3, pandas 3.0.6 and scikit-learn 1.9.1 and reproduced incumbent AUC within the frozen tolerance.

## Once-only development results

All quality numbers below are from the single saved evaluation. Higher AUC is better; lower log loss and Brier are better. These are historically reused development OOF results, not audit, public leaderboard or independent confirmation scores.

| Endpoint | Pooled AUC | Mean-fold AUC | Log loss | Brier |
|---|---:|---:|---:|---:|
| Current blend | 0.961407501 | 0.961411871 | 0.221727291 | 0.059288590 |
| A: control, epoch 16 | 0.954543419 | 0.954542509 | 0.254055197 | 0.063778570 |
| B: shared prefix, epoch 4 | 0.960546132 | 0.960687312 | 0.223829534 | 0.059825321 |
| C: sustained dropout, epoch 16 | 0.955469095 | 0.955471336 | 0.249110342 | 0.063022271 |
| 90% incumbent + 10% A | 0.961317257 | 0.961321802 | 0.221813517 | 0.059285317 |
| 90% incumbent + 10% B | 0.961391527 | 0.961401078 | 0.221749865 | 0.059290879 |
| 90% incumbent + 10% C | 0.961360067 | 0.961364540 | 0.221768490 | 0.059281780 |

| Endpoint | Fold 0 | Fold 1 | Fold 2 |
|---|---:|---:|---:|
| Current blend | 0.960940587 | 0.961356725 | 0.961938302 |
| A: control, epoch 16 | 0.953682929 | 0.955022492 | 0.954922107 |
| B: shared prefix, epoch 4 | 0.960213534 | 0.960661080 | 0.961187323 |
| C: sustained dropout, epoch 16 | 0.954835055 | 0.955806484 | 0.955772469 |
| 90% incumbent + 10% A | 0.960865577 | 0.961279898 | 0.961819930 |
| 90% incumbent + 10% B | 0.960930560 | 0.961340225 | 0.961932450 |
| 90% incumbent + 10% C | 0.960919222 | 0.961316530 | 0.961857870 |

Treatment minus control pooled AUC was **+0.000925676**, positive in every fold (+0.001152126, +0.000783991, +0.000850363). Treatment minus shared prefix was **-0.005077037**, negative in every fold (-0.005378478, -0.004854596, -0.005414853). Treatment improved control log loss and Brier but remained worse than the shared prefix on both. The positive paired effect supports this specific dropout intervention on this recipe and seed configuration, not sustained continuation as a better predictor.

The unchanged mixture gate required treatment-mixture pooled and mean-fold AUC gains of at least 0.00001 against **each** incumbent, control mixture and prefix mixture, with no fold regression below -0.00002.

| Treatment mixture comparison | Pooled AUC delta | Mean-fold AUC delta | Worst fold delta | Gate |
|---|---:|---:|---:|---|
| Versus incumbent | -0.000047434 | -0.000047331 | -0.000080432 | Fail |
| Versus control mixture | +0.000042810 | +0.000042739 | +0.000036632 | Pass |
| Versus shared-prefix mixture | -0.000031460 | -0.000036538 | -0.000074580 | Fail |

Every treatment-mixture fold declined against the incumbent. Its slightly lower Brier does not override the preregistered AUC criterion. **Advancement to confirmation: false.** All tested 10% additions also had lower pooled AUC than the incumbent. The experiment tested only the registered 10% contribution; it does not identify an optimal alpha or establish that every possible weight is harmful. Retain the existing blend and assign this candidate no new release contribution. No adaptive weight search was performed.

## Passive inner evidence and proposed next step

A separate endpoint-only review read the already-saved inner curves at epochs 4 and 16, with no rescoring or best-epoch search. Across all three folds and both arms, the fixed 2,048-row training probe improved while held-out AUC and log loss deteriorated.

| Endpoint | Mean training-probe log loss | Mean inner-monitor log loss | Mean inner-monitor AUC |
|---|---:|---:|---:|
| Shared epoch 4 | 0.213770076 | 0.222796245 | 0.961001223 |
| Control epoch 16 | 0.092142424 | 0.260715629 | 0.954361496 |
| Treatment epoch 16 | 0.111166566 | 0.253621663 | 0.955419573 |

These are unweighted means of saved inner-fold metrics, not pooled AUCs. The probe metric is evaluated in inference mode on a small in-sample subset, not the stochastic fitting objective or a full-training estimate. The pattern supports partial mitigation of overfitting, while leaving the remaining mechanism unresolved. See `research/dropout_retry_inner_endpoints_review.md` for exact commands, twelve source hashes, fold contrasts and scope limits.

The next proposed experiment is **earlier LR cooling at the same 16-epoch duration**, with dropout held at 0.05 after epoch 4 in both arms. Reproduce an exact shared H16/E4 prefix. Keep the control's original LR schedule; linearly cool the treatment's actual E4 LR multiplier to zero by E16, preserving per-parameter factors and all other schedules, partitions, architecture and seed policy. For epoch-fraction clock t between 4 and 16, the proposed multiplier is m(4) times (16-t)/12. Its exact update-clock boundary must be frozen and tested before any fit.

This is a proposal only, with no registered new run or launch. The same 12-fit paired design, strict deterministic prefix gate, fixed E4/E16 endpoints and unchanged 10% mixture gate would keep the comparison interpretable. LR changes also change effective decoupled weight-decay dose even if the WD coefficient/schedule is held fixed, so this would test a continuation policy, not LR independent of shrinkage. Do not combine it with fixing legacy decay-factor behavior or changing capacity. The independent interpretation review, `research/dropout_retry_result_review.md`, specifies this design and its limits.

Do not search for a smaller alpha merely to rescue this failed candidate. For this completed experiment, retain the incumbent. A future question about the best blend amount requires a separately frozen selection/confirmation procedure; this one-point mixture test supplies no defensible optimal-alpha estimate.

## Preserved failure and determinism repair

The original private kernel 137072741 (`clarkkitchen/s6e10-dropout-20261004`, version 1) failed its generated CUDA exact-prefix smoke before any real-data fit. Network and Adam-state fingerprints differed at epoch 4, while the other 17 state components matched. The native generated-probe maximum difference was 1.7881393432617188e-7. The exact gate rejected it rather than widening tolerance. Its original status/manifest/archive/log remain intact under `cloud/fixed_epoch_dropout_v1/download_01/`.

One separately registered retry enabled strict deterministic Torch operations and cuBLAS configuration before CUDA initialization in all cloud processes, symmetrically for both arms. The scientific payload, data and registry remained byte-identical, and the exact prefix gate remained unchanged. Its successful exact prefixes demonstrate that the repair sufficed in this run. The specific nondeterministic CUDA kernel responsible for the original failure was not conclusively identified. The first wrapper lasted 71.61 seconds; the recorded provider quota increase was 79.767 seconds. No further retry was launched.

## Evidence, commands and limits

Terminal retrieval preserved the original cloud status, output manifest, 463,720,566-byte results archive and log. All **287** archive members passed bounded archive/path/size/hash verification. The isolated assessment workspace is `cloud/fixed_epoch_dropout_v1_retry1/assessment_workspace`. The evaluator verified source/runtime/native/partition identity and all prefix/update gates before reading new prediction quality, then exclusively claimed the assessment before metric computation. It was not rerun.

Executed commands, for provenance only. Do not repeat retrieval into the used directory or rerun the claimed evaluation:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/status_fixed_epoch_dropout_retry1.py status
.venv/Scripts/python.exe -X utf8 scripts/retrieve_fixed_epoch_dropout_retry1.py retrieve --download-dir cloud/fixed_epoch_dropout_v1_retry1/download_01 --include-log --assemble
.venv/Scripts/python.exe -X utf8 scripts/evaluate_fixed_epoch_dropout_v1.py --workspace cloud/fixed_epoch_dropout_v1_retry1/assessment_workspace
```

Key SHA256 identities:

| Artifact | SHA256 |
|---|---|
| Scientific registry | `6ec58209d08edc0b29bb330a33404bbd0c1717a1eb451e0aa533aeb77e937282` |
| Development payload manifest | `16d88ba105ce5e4f8c61cc22d45d3f7242ae83f8796a47fed2f22189cd4c9e5b` |
| Provider reconciliation | `65820c3046e06176c6f6130e10fbe1140d33c33204c97401a5fca077c1dcbc75` |
| Retry runtime amendment | `3445cf3a924294384465befbfbc833d70d832a1c8f0d80ad1894c3e07d33d4bb` |
| Runtime entry | `0de7791f1ca1de802250dfa38b285d77a649bb061e05dd02f881453d38b3ea36` |
| Frozen local evaluation protocol | `0e3c838028058b80c82d8bc63e1737aac7631405ca8c13124134c77924207a2c` |
| Frozen evaluator | `8ff9c036ae4480b4e7bcadbecf0700adf79d8df4fc9dc36b9629ddaac535676d` |
| Returned archive | `e597d8bd053f99ad446fbaf056bd0d9bbaac3cb299f590898c3c7c10614a3aa3` |
| Returned manifest | `58db0aa4d082955e5d916920950d8caf7109beaeb4bbefee5e6e1ae1d85aaa13` |
| Returned cloud status | `19a67f1acada87d93d2afac20dd0dd3dd71646f027c1d643f30afc6b84eb9741` |
| Saved evaluation | `91aede41316207b6662848efaf73f97f40c9a96b8f078c18b7faaab7e52c3d27` |
| Exclusive evaluation claim | `e2f56d09859ba7e4b7e752ea75f1760e0b23073be138977ae7ff176e6b4c2ef1` |
| Unchanged incumbent selection | `bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311` |
| Canonical split | `4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c` |

The local evaluation protocol froze at 18:46:58 UTC, before any new quality read. Implementation evidence includes six CPU adapter checks, 15 runner/preparation checks, eight deterministic-startup checks with the six adapter checks nested inside them, four retry packaging checks, 19 retrieval checks and 38 evaluator checks. Nested tests are not additional independent evidence. Source review and executed tests cannot substitute for the separately verified cloud completion and saved model-quality results.

Development folds have been reused across earlier campaigns and overlap in their fitting rows. Three consistent fold effects are descriptive support, not three independent replications or a significance claim. The run uses one registered seed recipe and one cloud stack. No new independent audit, uncertainty resampling, alpha optimization or public/private leaderboard evaluation occurred. Existing native exports support inference, not exact resumable training. Startup-role receipts bind frozen code/role coverage. Controller and last-worker PIDs have additional retained controller-state linkage; other historical worker identities rely on their typed startup receipts. These limits remain even though every registered gate executed successfully.

Three separate agent reviews completed: metadata provenance, saved-result interpretation and fixed inner-endpoint description. Root reviewed and integrated their findings without rescoring. Independent reviews and final operational closure are recorded in `state/fixed_epoch_dropout_v1/final_verification.json` and the linked review notes. The original failed attempt, all earlier campaigns, releases, audit exclusions and incumbent remain preserved. The current research heartbeat is paused on final closure; older monitors remain paused.
