# 10/7 nightly update

Snapshot: October 7, 2026, 9:10 p.m. EDT (October 8, 01:10 UTC). Project: Kaggle `playground-series-s6e10`. Pass 4 started from main commit `f6e4e33673d6e1778c55e2bfcc9d8dc1db21c22c`.

Pass 4 is in **Phase 2 own-model expansion and rolling Phase 3 ensemble evaluation**. Twelve complete five-fold members are verified. The highest measured logistic-regression stack AUC is **0.961731463580**, from nine members. Subsequent additions did not improve that score. The next base-model priority is the registered ten-fold LightGBM copy; compute readiness is unresolved. A launched Caruana ensemble-selection run needs process/status reconciliation before its result can be assessed.

These scores are development evidence. The stack evaluations use **conditional meta-cross-validation on fixed global OOF predictions**, not an independent, fully nested assessment. No Pass 4 competition submission or release promotion has occurred.

## Current step against the plan

| Plan section | Current state |
| --- | --- |
| Phase 0: new folds and re-baseline | Three baseline members and their stack completed and verified. Calibration submission remains pending authorization. |
| Phase 1: feature store | Rich features implemented and exercised by full LightGBM/XGBoost members. Seed-42 rich LightGBM AUC 0.961389150380 exceeds the plan's 0.9608 feature gate. Optional feature/model routes remain. |
| Phase 2: own members | Twelve full members verified. M1's three five-fold seeds, M2's bounded search/final member, M3's main categorical member, M4 and M6's three n32 seeds completed. Two weaker pilots rejected. M1 ten-fold copy prepared but not launched. |
| Phase 3: stacking | LR and equal-logit comparisons verified through the twelve-member library. Caruana launched, terminal outcome unverified. Full nested base-model regeneration remains unimplemented. |
| Phase 4: public prediction libraries | Not started; user decision pending. |
| Phase 5: final selection and delivery | Not reached. No Pass 4 submission authorization inferred from earlier campaigns. |

The plan is being executed in overlapping phases. Completing a conditional stack comparison does not satisfy the plan's full nested-validation requirement.

## Work completed on October 7

All times below are EDT on October 7. The execution logs use UTC, so their October 8 entries before 04:00 UTC still belong to this local evening.

| Time | Work and outcome |
| --- | --- |
| 11:43 a.m. onward | Fetched the latest main and opened a separate Pass 4 namespace. Created immutable five-fold seed-42 assignments over all 699,635 training rows; preserved historical campaigns, splits and releases. Local baseline XGBoost and fixed-four-epoch RealMLP completed. |
| Afternoon cloud recovery, verified by 1:18 p.m. | Original Kaggle GPU job, instrumented retry and generated GPU startup probe ended in errors without usable startup evidence. A generated CPU probe succeeded, then a separately identified private CPU CatBoost baseline completed all five folds and passed import verification. GPU failure cause remains unknown. |
| 12:46 to 3:22 p.m. | Verified rich LightGBM seed 42, rich XGBoost, n32 RealMLP, full all-categorical CatBoost and LightGBM seed 43. Initial and expanded conditional stacks passed independent saved-prediction reconstruction. |
| By 4:17 p.m. | GLM-margin XGBoost pilot failed its fixed fold-0 screen: AUC 0.960588100624, deficit 0.000905554235 versus the registered reference, exceeding the 0.0008 limit. No full expansion. Eight-member stack reached 0.961731311354. |
| By 5:17 p.m. | Third/final M1 seed 44 verified. Nine-member stack reached 0.961731463580, only +0.000000152226 above eight members. Depthwise-depth-10 CatBoost pilot also failed its fixed screen: AUC 0.960522195898, deficit 0.000971458960. No full expansion. |
| 5:32 to 6:35 p.m. | M2 ran eight fixed search points, rather than the plan's possible 40-trial search. All eight passed artifact verification before selecting trial 05. A separate five-fold final member with 127 leaves, L1=15, L2=5 and max_bin=4095 completed and was verified. Its stack addition was negative. |
| 7:08 to 8:34 p.m. | Completed and verified the second and third n32 RealMLP seeds. Ten-, eleven- and twelve-member stack comparisons all failed to improve on the nine-member reference. The three-seed neural sequence is closed; no fourth seed is planned. |
| 8:47 to 9:03 p.m. | Ten-fold M1 source and generated checks were ready, but local capacity had not cleared. Prepared a private CPU environment probe and launched one bounded Caruana comparison over the frozen twelve-member library. |

The current neural input already implements the proposed raw-plus-auxiliary, no-competition-TE view. Repeating it under a new “no TE” name would duplicate existing work.

## Verified full-member inventory

Each row represents a complete five-fold member with keyed OOF and test predictions. AUC is pooled over 699,635 training rows; test predictions cover 299,844 rows. Values are rounded to twelve decimal places.

| Member | Pooled OOF AUC |
| --- | ---: |
| Baseline XGBoost, local | 0.961019843845 |
| Baseline RealMLP, n_ens=8, fixed 4 epochs | 0.961061493452 |
| Baseline CatBoost, private Kaggle CPU | 0.960996141377 |
| M1 rich LightGBM extra-trees, seed 42 | 0.961389150380 |
| M1 rich LightGBM extra-trees, seed 43 | **0.961397694408** |
| M1 rich LightGBM extra-trees, seed 44 | 0.961353553681 |
| M4 rich XGBoost, seed 42 | 0.961183801805 |
| M3 all-categorical CatBoost, seed 42 | 0.961138413724 |
| M2 tuned rich LightGBM, final v1 | 0.961238040892 |
| M6 RealMLP n_ens=32, seed 20261005, fixed 4 epochs | 0.961109313795 |
| M6 RealMLP n_ens=32, seed 20261006, fixed 4 epochs | 0.961091241424 |
| M6 RealMLP n_ens=32, seed 20261007, fixed 4 epochs | 0.961106441787 |

Seed 43 is the best measured standalone M1 seed and the selected recipe for the proposed ten-fold copy. This choice reuses development evidence; it is not independent confirmation.

## Ensemble progression

These are verified conditional LR meta-CV scores. Each expansion retained the preceding library and added the stated member(s).

| Members | Addition | Pooled AUC | Change from preceding row |
| ---: | --- | ---: | ---: |
| 3 | Baseline XGBoost, RealMLP and CPU CatBoost | 0.961641466525 | Reference |
| 5 | Rich LightGBM seed 42 and rich XGBoost | 0.961707340883 | +0.000065874358 |
| 6 | First n32 RealMLP | 0.961715241243 | +0.000007900360 |
| 7 | All-categorical CatBoost | 0.961720157439 | +0.000004916196 |
| 8 | Rich LightGBM seed 43 | 0.961731311354 | +0.000011153914 |
| 9 | Rich LightGBM seed 44 | **0.961731463580** | +0.000000152226 |
| 10 | M2 tuned LightGBM | 0.961729408772 | -0.000002054808 |
| 11 | Second n32 RealMLP | 0.961727084665 | -0.000002324107 |
| 12 | Third n32 RealMLP | 0.961726557954 | -0.000000526712 |

The nine-member stack is +0.000089997055 above the three-member re-baseline. Its advantage over eight members is below the registered 0.00001 descriptive gain threshold. The twelve-member stack is -0.000004905626 below nine members and loses on all five folds in that fixed contrast. More verified members have not automatically produced a better ensemble.

Saved coefficients, partition-specific scalers, canonical IDs, probability bounds and all-row OOF/test reconstruction passed independent checks. Maximum stack reconstruction error was 4.44e-16. The twelve-member verification checked 563 bound hashes. These checks establish artifact consistency, not unbiased generalization performance.

## Current execution and immediate next work

1. **Reconcile Caruana execution before assessing or retrying.** The registration freezes twelve members, five meta folds, 20 model bags per fold and ten greedy selections per bag, using two CPU threads and a 1,800-second total cap. The 9:03 p.m. controller snapshot says `running`; however, the 9:10 p.m. process inspection did not find recorded PID 32724 or a matching Python worker. The output directory contained only `claim.json`, and stdout/stderr were empty. No completed prediction/model artifacts or verified score were available. This is an unresolved execution status, not proof of success or a diagnosed failure. Preserve the existing intent and reconcile it before any new launch.
2. **Establish capacity for the true ten-fold M1 member.** The local adapter passed 24 generated tests and independent source review. The post-stack memory snapshot was 1.854 GiB available; the later pre-Caruana snapshot was 4.407 GiB. Neither establishes full ten-fold readiness. The memory proposal calls for a sustained 6 GiB reserve locally, explicitly an engineering threshold rather than a measured requirement. Private CPU environment probing is the next prepared fallback step; no ten-fold fit has launched.
3. **Register a separate cloud edition if the fallback is viable.** Check live free capacity, runtime versions, memory, generated native smoke and source/data/image identities before fitting. Preserve the local registration and ten-fold split. The previous CPU CatBoost success does not establish LightGBM compatibility. The historical cloud package versions differ from the frozen local contract. GPU fallback additionally needs new startup-recovery evidence.
4. **Keep ten base folds distinct from five meta folds.** The new member requires ten fresh models, honest base-fold provenance, exactly one held prediction per training row and an exact ten-model test mean. A separately reviewed stack intake is needed; existing five-fold metadata must not be relabeled.
5. **Resolve the validation gap before claiming independent improvement.** Genuine nesting requires outer-excluded base fits and regenerated inner OOF. Current LR and Caruana procedures operate on already generated global OOF.

The latest controller record has the existing 15-minute “Airline Pass 4 training and verification” heartbeat active. Older monitors remain paused. Public OOF libraries, competition submissions and TabPFN account/license acceptance remain separate pending decisions. Free private cloud fallback is authorized; paid compute and public notebook posting are not.

## Evidence and limits

- Canonical split: `data/folds_v4.parquet`, SHA256 `517803f070d13f79ca3c9a1e32dc485d36d121eb7364734a2272c4d3aabb6f16`. Historical audits were already exposed and have not been reassessed as fresh evidence. Their rows enter this separately authorized training/CV namespace.
- Tree members use scored-fold early stopping; M2 additionally selected a recipe on fold 0. Neural members use fixed four-epoch training without scored-fold neural selection. Repeated library/stack development and label-free transductive features remain relevant limitations.
- Meta-held labels can influence other global OOF rows through their base fits. Correctly excluding those labels from the combiner's direct fit does not remove that dependence. No statistical-significance, private-leaderboard gain or optimal-weight claim follows from these small deltas.
- Current-state sources: `state/pass4/progress.json` (updated October 8, 01:03:12 UTC), `PASS_4_LOG.md`, the dated additions in `PASS_4_HANDOFF.md`, and `state/pass4/caruana_v1_execution.json`. Earlier “current worker” paragraphs in the handoff are superseded by later entries.
- Result sources: member `meta.json` files under `artifacts/pass4/members/`; independent receipts under `state/pass4/`; `research/pass4_stack_comparison_results.md`; `research/pass4_stack_expansion_v2_results.md` through `pass4_stack_expansion_v7_results.md`; and the M1 readiness/Caruana review notes.
- Nine-member proof SHA256: `063aacf0a1ba8cae7384c1984f7271df061d290016024eb69142e8c9cc36067c`. Twelve-member proof SHA256: `f7d01d14392b6c0d34ec39aab4bcf319020f5e31149c7c6d836e71404e30b049`.

This commit publishes the nightly Markdown summary only. The local evidence paths identify the working research record; uncommitted Pass 4 code, logs, configurations, datasets and model artifacts are not bundled in this update. Existing verification results were read for this summary; model fitting, quality assessment and native inference were not rerun.
