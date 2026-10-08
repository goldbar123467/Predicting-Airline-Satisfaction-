# Pass 4 progress grade, October 8, 2026

Graded on 2026-10-08 against `PASS_4_PLAN.md` (commit `fe0cc83`) using the evidence in `10-7 nightly update.md` (commit `bf553d4`, snapshot 2026-10-08 01:10 UTC). The grader did not see `state/pass4/`, `PASS_4_LOG.md`, `PASS_4_HANDOFF.md` or member `meta.json` files: they live on the local machine and are not in git. Every number below is copied from those two documents or derived from them by the arithmetic shown. Tags follow `PASS_4_PLAN.md`: **[ours]**, **[field]**, **[hyp]**.

## Overall grade: B (2.9 of 4.0)

Execution is fast and the evidence is clean. On day 1 the pass finished most of the work the plan scheduled for days 1 to 4, and every full member matches or beats the field's published expectation for its recipe. The score, however, has stopped moving. The best stack gained +0.000090 over the three-member re-baseline, 45% of the +0.0002 exit target due October 11, and the last three additions each made it worse. More members from the same three families will not close the gap. The levers that can (public OOF libraries, a TabPFN member, a first calibration submission) are waiting on user decisions D-A, D-B and D-C, not on compute.

Pass 4 is on track to beat the October 2 public score of 0.96125. It is not on track for first place without those decisions.

## Scorecard

Letter-to-point scale: A 4.0, A- 3.7, B+ 3.3, B 3.0, B- 2.7, C+ 2.3, C 2.0, C- 1.7.

| Dimension | Weight | Grade | Evidence |
|---|---:|:---:|---|
| Throughput against the day plan | 20% | A- | Phase 0, the feature-store gate, M1 (3 seeds), M2, M3, M4 and M6 (3 seeds) finished on Oct 7, ahead of the Oct 8 to 11 window. Missing: calibration submission, M7, true nesting. |
| Member strength | 15% | A | All 12 members are above 0.96099. M1 seeds 0.961354 to 0.961398 against a field expectation of 0.96118 to 0.96126; M2 0.961238 against 0.9605 to 0.9610; M6 seeds 0.961091 to 0.961109 against a field n_ens=32 value of 0.961017. |
| Ensemble progress against exit gates | 25% | C | Stack +0.000089997 over re-baseline, against a +0.0002 target. Ten, eleven and twelve members all lost to nine; twelve lost on all five folds. |
| Evidence quality and honesty | 15% | A- | Conditional meta-CV is labeled as such; seed-43 selection flagged as reused evidence; reconstruction error 4.44e-16 over 563 bound hashes; two pilots stopped by pre-registered fold-0 kill rules. Minus: the plan's leave-one-out admission rule (rule 4.4) is not reported, only order-dependent sequential additions. |
| Operational reliability | 10% | C- | Caruana run in an ambiguous state (controller says `running`, no PID 32724, only `claim.json`, empty stdout/stderr). Kaggle GPU startup failure undiagnosed after three attempts. Local free memory of 1.854 to 4.407 GiB is blocking the ten-fold M1 copy. |
| Critical-path decisions | 15% | C+ | D-A (public libraries), D-B (TabPFN license) and D-C (submissions) are still open on day 2. The plan asked for them before Pass 4 started. These are mostly the user's to unblock. |

Weighted score: 0.20 x 3.7 + 0.15 x 4.0 + 0.25 x 2.0 + 0.15 x 3.7 + 0.10 x 1.7 + 0.15 x 2.3 = **2.91**.

## Exit checks from the plan's day schedule

| Window | Exit check | Target | Actual | Status |
|---|---|---|---|---|
| Oct 7 to 8 | Five-fold re-baseline stacked | 3 members stacked | 0.961641466525 | Pass |
| Oct 7 to 8 | Rich LightGBM feature gate | at least 0.9608 | 0.961389150380 (+0.000589) | Pass |
| Oct 7 to 8 | One calibration submission | 1 submission | none; authorization pending | Open |
| Oct 8 to 11 | M1 (3 seeds), M3, M4, M6 | complete | complete and verified | Pass, early |
| Oct 8 to 11 | Nested stacker live | LR, equal-logit, Caruana under nested CV | LR and equal-logit under conditional meta-CV; Caruana unresolved | Partial |
| Oct 8 to 11 | Own stack gain over Phase 0 | at least +0.0002 | +0.000089997055 (45%) | Behind; due Oct 11 |
| Oct 10 to 13 | TabPFN fold-0 decision | decision made | not started; D-B open | Not started |
| Oct 10 to 13 | First public-member stack submission | 1 submission | D-A open | Blocked |

## Where this points [hyp]

- Applying the field's calibration (public LB about 5-fold OOF minus 0.00035 [field]) to the nine-member stack: 0.961731 - 0.00035 = **0.96138 public [hyp]**. That is +0.00013 over our best LB of 0.96125 and about 0.00045 below the reported top score of 0.96183 [field, search snippet].
- On the same fold scheme, ChrisLegge reports 5-fold CV 0.96208 with 184 members, including public ones [field]. Ours is 0.000349 lower.
- Our own LB-minus-OOF offset is unmeasured until the first Pass 4 submission. Every projection above inherits that uncertainty.

## What is working

1. **The plan's diagnosis was right about base-model strength.** On the new folds, rich LightGBM extra-trees (0.961398) beats every other member family, and on the same folds it is 0.000378 above the baseline XGBoost recipe (0.961020). Pass 3's best single model (0.960872) was measured on the old 3-fold development split, so it is not directly comparable. The best stack (0.961731) is 0.000334 above the best single member, so stacking still pays.
2. **Kill rules saved compute.** The GLM-margin XGBoost pilot (deficit 0.000906) and depthwise CatBoost pilot (deficit 0.000971) both failed the 0.0008 fold-0 limit and were stopped before full expansion.
3. **Cloud fallback worked once it was simplified.** After three GPU failures, a private CPU CatBoost job completed all five folds and passed import verification.
4. **Reporting discipline.** The nightly update separates conditional meta-CV from true nesting, refuses to infer submission authority, and closes the neural seed sequence at three instead of drifting to a fourth.

## What is not working

1. **Diversity has saturated within the current families.** Nine strong members added +0.000090 in total. Members 10 to 12 (tuned LightGBM, two more n32 RealMLP seeds) were each negative. The remaining own-model queue (ten-fold copies, more seeds) is mostly more of the same families.
2. **No calibration point.** Without one Pass 4 submission there is no measured OOF-to-LB offset for this pipeline, so Phase 5 would select finals on a field prior.
3. **The highest-value items are blocked on decisions, not compute.** The plan's own priority table puts public libraries (+0.0002 to +0.0005) and TabPFN (+0.0001 to +0.0002) above ten-fold copies (+1e-5 to +1e-4) [field-derived priors].
4. **Process liveness is not verified.** A controller record said `running` for a job with no live process and no output. The same pattern can hide a dead job during Phase 5.
5. **Admission rule drift.** Rule 4.4 admits a member only if the stack *with* it beats the stack *without* it by +1e-5 pooled and on 3 of 5 folds. Sequential additions depend on order. By the sequential numbers alone, seed 44 (+1.5e-7), all-categorical CatBoost (+4.9e-6) and the first n32 RealMLP (+7.9e-6) would each fall below +1e-5. Leave-one-out ablations may tell a different story; they have not been reported.

## Fixes before Phase 5, in priority order

| # | Owner | Action | Why now |
|---:|---|---|---|
| 1 | User | Decide D-A (public libraries), D-B (TabPFN-3.5 license and token) and D-C (submissions, at most 3 per day). | They gate the two largest remaining levers and the calibration point. Every day of delay shortens Phase 4. |
| 2 | Agent | Reconcile Caruana now: check for a live process by PID, creation time and command (including a Windows Python redirector child); check `claim.json` age against the 1,800 s cap. If no process and no outputs, record a terminal `lost_no_output` state, keep `claim.json`, and relaunch only with flushed logging and a heartbeat file. | Unresolved state files block clean selection later. The field reports combiner choice moves AUC by about 1e-5 [field], so this is a hygiene fix, not a score lever. |
| 3 | Agent, after D-C | Submit the three-member re-baseline stack once as the calibration point. | Gives Phase 5 a measured offset instead of the 0.00035 field prior. |
| 4 | Agent | Report leave-one-out ablations for all 12 members (plan rule 4.4) and keep `members/ledger.csv` current. | Decides which members are admitted, independent of addition order. |
| 5 | Agent | If D-B is approved, run the TabPFN fold-0 measurement before the ten-fold M1 copy. | New model class; higher expected stack value than another LightGBM variant. |
| 6 | Agent | Size the ten-fold M1 copy from measured data: each ten-fold fit trains on 90% of rows versus 80% in the five-fold runs, so the recorded five-fold peak memory, scaled by 9/8, bounds it. Use that number instead of an unmeasured 6 GiB reserve, or run it on the private Kaggle CPU path. | Unblocks the item without guessing. |
| 7 | Agent | Do not build full nested base-model regeneration before the close. The field's published CV numbers use the same conditional procedure, so comparability is unaffected, and the cost is about five times the base fits. Phase 6 measures the actual optimism against the private leaderboard (hypothesis H6 in `PASS_4_PHASE_6_PLAN.md`). | Spends compute where it can move the score. |

## What would raise the grade

- **To B+:** calibration submission made; Caruana reconciled; leave-one-out admission ledger published.
- **To A-:** stack gain of at least +0.0002 over the re-baseline under the admission rule by October 11, with D-A and D-B resolved either way.
- **To A:** the above plus a TabPFN or public-library member admitted by ablation, and no ambiguous process state for the rest of the pass.
