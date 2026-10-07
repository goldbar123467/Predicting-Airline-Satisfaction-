# Post-confirmation development and cloud review

Review time approximately 2026-10-02 12:44-12:49 UTC. Evidence snapshot: `artifacts/second_pass/blend/current.json`, created **12:41:52.091434 UTC**, with all 17 primary new configurations plus the one registered confirmation completed. This read-only review used saved development metrics, configuration/decision records, source notes and cloud operational reports. It did not read OOF matrices, train/audit labels, audit scores or test predictions; no fits, queue/configuration changes or submissions were made. Own output: this note only.

## Decision

**Freeze the completed local second campaign early and start release refitting/verification.** The registered experiment batch and its one allowed confirmation are complete. The final constrained search accepts two additions, and no third addition clears the predeclared thresholds. The best remaining grid point, CatBoost L2=20 at alpha .10, improves only **0.000003595317 pooled** and **0.000003876762 mean-fold** over the current mixture, below both 0.00001 gates. Its fold 1 also declines slightly. This is concrete diminishing-return evidence under the current policy, satisfying the plan's early-freeze exception; it is not a claim that no undiscovered method could improve performance.

Continue cloud throughput/portability work separately from this frozen local selection. The newly proposed conditional-rating probability features are a defensible **single separate cloud hypothesis**, detailed below, but do not delay or reopen the completed local release. Free compute and spare clock time alone do not justify changing its experiment budget or selection rules.

## Independent confirmation and weight checks

The saved confirmation decision was recorded at **12:36:07.037381 UTC**, after the 17-primary-result snapshot at 12:32:30, before the extra fit. It names seed 20261021, the raw+aux recipe and its equal-weight group. The original and confirmation run objects differ **only** in ID and seed. The declared group is `v2_realmlp_cat_raw_aux_fixed_seed_average`.

I independently checked from the current JSON that:

- Each group member receives **0.135**, exactly half the **0.27** group weight.
- Every original v1 component weight is multiplied by exactly **0.63**; old internal ratios and old equal-seed groups remain locked.
- XGB profiles+aux receives **0.10**, and the expanded weights sum to one.
- The accepted sequence is 30% group plus 70% anchor, then 10% XGB plus 90% of that mixture. No individual confirmation seed is freely optimized.

Expanded positive weights:

| Member | Weight |
|---|---:|
| lgb_route | 0.029925 |
| cat_route | 0.05117175 |
| cat_route_teacher | 0.02428659228515625 |
| cat_route_teacher_d8 | 0.06476424609374999 |
| realmlp_te_teacher16 | 0.0315 |
| realmlp_cat_te_teacher | 0.09589150634765624 |
| cat_route_teacher_seed2 | 0.02428659228515625 |
| cat_route_te_teacher | 0.1151364375 |
| realmlp_cat_te_teacher_seed2 | 0.09589150634765624 |
| xgb_route_te_teacher_profiles | 0.097146369140625 |
| v2_realmlp_cat_raw_aux | 0.135 |
| v2_realmlp_cat_raw_aux_seed20261021 | 0.135 |
| v2_xgb_route_te_teacher_profiles_aux | 0.10 |

The confirmation seed alone is weaker: pooled AUC **0.960791105376**, versus first seed **0.960852059590**. Selected fold epochs change [3,3,3] -> [4,3,4]. Nonetheless, the registered equal average increases the 30%-group anchor mixture to **0.961347956867**, from **0.961327622176** for the first seed's same-weight mixture. Thus retaining the equal average is supported by the registered blend test; choosing the better standalone seed would have contradicted the protocol. This combines optimization and inner-stopping-split variability and is not an independent replication of the entire model-selection process.

## Final development evidence and diminishing returns

| Metric | Immutable anchor | Current blend | Paired gain |
|---|---:|---:|---:|
| Pooled AUC | 0.961221030186 | 0.961363719325 | +0.000142689139 |
| Mean-fold AUC | 0.961224949837 | 0.961367779436 | +0.000142829598 |
| Fold 0 | 0.960738222516 | 0.960905991249 | +0.000167768733 |
| Fold 1 | 0.961157175823 | 0.961302229898 | +0.000145054074 |
| Fold 2 | 0.961779451172 | 0.961895117160 | +0.000115665988 |

The final XGB addition contributes **+0.000015762458 pooled** beyond the group+anchor mixture, with positive changes on every fold. Relative to the earlier first-seed mixture with the same final anchor/XGB weights, the confirmed blend adds +0.000018092668 pooled. These are saved-grid arithmetic comparisons, not newly recomputed AUC estimates.

The final six primary hypotheses offer no additional qualifying blend component. Native RealMLP's second teacher improves its matched standalone pooled score +0.00002593, stronger weight decay +0.00009446, lower learning rate +0.00004382, and embedding width 10 only +0.00000671. None clears the anchor blend gate. XGB TE smoothing 100 loses -0.00006963 standalone; removing TE from the RealMLP profile+teacher recipe loses -0.00020219 and declines on all folds. This supports stopping the registered branch rather than sweeping nearby hyperparameters. The raw-versus-TE+teacher comparison still removes two factors together and must not be attributed solely to TE.

The 30 original configurations plus 18 second-pass fits and repeated grid comparisons reuse the same development labels. Fold consistency and constrained weights reduce flexibility, but the three folds are correlated through overlapping training sets, and model/weight selection remains adaptive. These gains are **not unbiased generalization estimates, not a fresh v2 audit, and not a public leaderboard improvement**. No uncertainty claim based solely on those three fold deltas is warranted.

## Cloud capabilities now actually verified

Both retrieved jobs are version 1, `COMPLETE`, with read-back metadata `isPrivate=true`.

- `cloud/realmlp_probe/output/realmlp_probe/realmlp_probe_report.json` reports **passed**. PyTabKit 1.7.3 and Lightning 2.6.6 were installed; TorchMetrics 1.9.0 was already satisfied. The protected Torch 2.10.0+cu128/CUDA and scientific packages were unchanged. A single T4 trained and exported the actual eight-member 512/256/128 categorical wrapper on synthetic rows, selected and fixed two epochs, and passed unknown-category, validation-vocabulary, batches 1/17/257 and CPU/CUDA reload checks. Maximum reload/CPU-CUDA error was **2.98023224e-7**, native-export error **1.78813934e-7**. Fit+export times were 5.34 and 2.81 seconds for 600 training rows; allocator peaks were 72.5 MiB allocated/98 MiB reserved. Those small-row figures do not predict full-fold throughput or memory.
- `cloud/portability/output/portability/portability_report.json` reports **passed** for a previously fitted local raw RealMLP native graph on 1,024 raw test-feature rows under Linux Torch 2.10.0+cpu. Batches 1/17/1024 agreed with the local reference within **1.78813934e-7**. No training, satisfaction score or PyTabKit import occurred. This establishes useful cross-version/native-inference portability for that member and sample, not all-row equivalence for the final 13-member blend.

Next operational step remains the fixed-recipe, full-size fold timing control in `cloud_realmlp_control.md`, with an 1,800-second cap and no model-selection role. Measure actual preparation, fit/export, inference and memory before parallelizing T4 work or forecasting a new three-fold campaign.

## One plausible separate-cloud hypothesis: observed-rating probability

The existing 13 original-only rating models yield full conditional class distributions but current downstream features retain only expected values. For rating j, let `q_j = P_model(R_j = observed R_j | other 20 raw features)`. Add the 13 q values plus a fixed sum of their clipped log probabilities beside the existing expected-value features in raw+aux RealMLP. This requires no new source-bank fits or synthetic-label feature training.

**Information rationale:** expected value discards distribution shape. Two distributions over [0,1,2], [.5,0,.5] and [0,1,0], both have expectation 1 but assign entirely different probability to an observed rating 1. The current raw rating plus its predicted mean cannot reconstruct that distinction. The proposed feature could distinguish unusual combinations from ordinary observations even when their mean residuals match. This is a plausible information mechanism, not measured satisfaction-prediction evidence.

**Interpretation:** the sum of overlapping conditional log scores is a composite conditional score, not a calibrated joint likelihood for a passenger row. The 13 fitted conditionals can be mutually inconsistent, and their scores are correlated. Original-domain class-probability miscalibration, rare ratings and synthetic/source shift can create misleading extreme values. Adding both q values and their log sum is one preregistered feature-block hypothesis, not permission to tune many clipping rules or feature subsets after seeing results.

**Source/licensing:** inspected `second_pass_competition.md` documents Part4's observed-category and sum-log idea, but that source fits its bank transductively on train+test features and includes 16 targets. Do not reproduce that training scope. Reuse only our fixed original-only 13-rating banks and their original-only vocabularies/overlap exclusions. The newer notebook's license was not exposed in downloaded metadata; the earlier Part3 Apache license does not establish Part4's license. Independently implement the mathematical feature with attribution rather than copying notebook code. Existing native banks and pipeline code are our own. No new license permission is inferred.

**Required fixed contract before any candidate:** use each saved model's explicit class-value array when selecting probability columns; never assume rating value equals column index. Exclude the observed rating from that bank model's inputs exactly as in current EV inference. Validate integer/nonmissing observed-rating support up front. If unsupported/missing ratings must be accepted, predeclare and report a deterministic convention, such as q=0 with a fixed log floor, rather than silently indexing -1 or fitting fallback values from synthetic/test rows. A proposed numerical convention is `log(max(q, 1e-6))`, fixed in advance, with float64 sum over the 13 terms followed by float32 output; record unsupported/clipped counts. Do not claim this floor is optimal. Check finite values, class ordering, reordered/partial batches, CPU-native reload and cache-versus-raw equality.

**Bounded evaluation:** after the full-size portability/timing control, one exact matched cloud control (raw+EV) and one raw+EV+q/log configuration, same existing seed and all unlisted params, three frozen development folds with nested inner stopping. Both run in the same cloud image and library versions; comparing only against a local control confounds the feature with environment. Preserve all audit labels outside both jobs. Register separate cloud IDs and ledger, no additional seed or hyperparameter sweep, and inspect paired fold effects plus the already-declared blend gates. Only a separately documented future release decision after full verification could admit it; it cannot retroactively modify today's frozen local release.

This is more defensible than an arbitrary larger neural model or another seed, but it still adds feature extraction, a new cache/native-inference contract, two full three-fold fits and another adaptive comparison. Tiny synthetic throughput is insufficient to promise that completion/retrieval/release checks fit today's remaining window. **Recommendation remains early local freeze; pursue this cloud hypothesis only after the realistic control and a separate fixed time budget, without making the current deliverable depend on it.**

## Remaining release risks

Three new positive-weight full-data members require fixed-length refits and native integrity checks; existing v1 members must be reused under their stored hashes. All-label production fitting is allowed only after frozen selection, with no new validation-based tuning. The expected-rating bank must reproduce cached features from raw rows with exactly the saved class ordering and float64 reductions; a tree threshold can amplify tiny feature discrepancies. Recheck the full 299,844-row weighted probabilities, exact sample IDs/order/schema, bounds, native reload, all-row raw inference and archived hashes. The small cloud probes do not replace these checks. Preserve the immutable v1 fallback if a new package fails, and keep release status distinct from model-selection scores.
