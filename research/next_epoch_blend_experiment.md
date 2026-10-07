# Proposed next experiment: separate schedule, duration and ensemble value

Status: research design only, not dispatched. No training authorization is inferred from this document. The current user request is a 3–4 hour investigation. This proposal replaces the vague instruction to try still more epochs with a bounded, interpretable experiment.

## Decision and priority

First persist learning curves and separate the schedule horizon from the stopping point. Then run the three-arm experiment below if a subsequent training campaign is requested. Do not make a new 500-epoch run the next step. Do not combine a scheduler change, optimizer correction, feature change and weight search into one result.

The existing retained blend stays at `0.64*v2 + 0.16*v3_realmlp_probability + 0.20*v3_xgb_probability`. The completed weight diagnostics do not warrant a new release. Existing failed duration admission decisions and the closed checkpoint screen remain unchanged.

## Fixed data and implementation contract

- Population: the frozen 629,671 development IDs, split SHA256 `4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c`, outer folds 0/1/2. Never sample or fit before filtering `fold >= 0`. No audit labels, audit predictions or full-data production model may enter this experiment.
- Each outer training set contains 419,780 or 419,781 rows. Preserve the original inner 90% fit / 10% monitor partition using the existing `seed + fold` stratified policy. The monitor labels record learning curves only; they do not select endpoints, weights, seeds, architecture or further arms.
- Feature recipe: exact `v2_realmlp_cat_raw_aux`, including raw numerical copies, native categorical twins, unchanged original-only auxiliary expected-value banks, no teacher, no route target encoding, and no auxiliary probability block. This isolates the previously investigated raw+EV duration recipe. Testing the newer probability features is a separate representation experiment.
- Fit category dictionaries and learned continuous preprocessing only on each arm's training partition. Reuse only external bank features whose original-only provenance and overlap exclusions were already frozen. Record feature order, categorical positions, category maps and unknown behavior with each model.
- Matrix contract: rows indexed by authoritative ID; feature matrix float32 with exact numeric category codes supported by the current adapter; labels int64 in `{0,1}`, positive class satisfaction=True; output shape `(n_rows,2)` with explicit class order `[0,1]`. Persist effective resolved factory configuration, not merely constructor arguments.
- Architecture/optimization: eight vectorized members, hidden sizes `[512,256,128]`, batch size 256, base LR `.053`, WD `.015`, label smoothing `0`, existing activation/defaults and **legacy PyTabKit 1.7.3 decay behavior**. Freeze all inherited defaults and per-parameter factors. Model seed remains 20261005 in every fold/arm, matching current `common.py`; only the inner partition seed is 20261005 + fold, matching `train.py`. No new seed search or accidental per-fold model-seed change.
- Clock: choose **epoch fraction** for this experiment. Schedules use epoch progress / declared horizon. Log optimizer updates separately. Inner-to-outer training size changes steps per epoch from 1,475 to 1,639, so this is not an equal-update inner/outer refit claim. A separate step-clock experiment would require a different preregistered question.
- Code must expose independent `schedule_horizon_epochs` and `executed_epochs`. Preserve model/preprocessor export parity. The tested installed `StopAtEpochsCallback` path can retain horizon while stopping earlier; do not replace the horizon by the selected stopping epoch.

## Three fixed arms, two trajectories per training partition

| Arm | Schedule horizon | Executed epochs | Purpose |
|---|---:|---:|---|
| A | 4 | 4 | New fixed-four-epoch reference |
| B | 16 | 4 | Prefix of the longer schedule, equal training duration to A |
| C | 16 | 16 | Continue exactly B's trajectory through its declared annealing window |

Train A and C. Export B at the end of C's fourth epoch, cloning parameters/buffers so later updates cannot mutate that checkpoint. B is not a separately initialized fit. A and C use the same initial model/data-order seed on the same training partition. Horizon-dependent dropout means their RNG-dependent trajectories may diverge; that is part of the schedule intervention, not a duration-only effect.

The planned contrasts are B minus A (normalized-schedule intervention at four epochs), C minus B (more updates along one fixed trajectory, including the later schedule), and C minus A (total practical change). The old four-epoch-cap control selected three epochs in every fold, so A is **not** identical to that old member. Do not relabel it or rewrite its result.

For each outer fold:

1. Train an inner A trajectory and an inner C trajectory, saving B at epoch 4. Evaluate the fixed inner monitor at every epoch for A/C and save all scalar curves. Inner observations cannot change the registered recipe or endpoints.
2. Reinitialize and refit outer A/C on all outer-training development rows, again exporting B at epoch 4. Fixed endpoints do not depend on the inner curves.
3. After all three outer folds and integrity checks finish, compute outer endpoint metrics for A/B/C once. Do not inspect outer-fold AUC every epoch, use partial-fold success to stop, or choose additional checkpoints from those outer results.

This is twelve actual fits across the three folds: six inner fits and six outer fits. B checkpoints add no new fitting trajectory. Total epochs are 120, approximately 114 full-outer-training epoch equivalents because the inner fits use 90% as many rows. Eight vectorized members do not multiply the already measured per-epoch wall time by another eight.

**Disable best-checkpoint restoration as well as early termination.** The current adapter sets `use_best_mean_epoch_for_cv=True`, and the installed library defaults to `use_best_epoch=True`. Supplying `X_val/y_val` can therefore restore an earlier best model at fit end even when `use_early_stopping=False`. A new fixed-endpoint adapter must either explicitly expose and verify `use_best_epoch=False` or keep monitoring outside the library's checkpoint-selection path. The existing adapter rejects this new argument; it cannot be silently passed through today. Before any real fit, a synthetic nonmonotonic-monitor test must prove that endpoint 16 remains endpoint 16 when an earlier epoch has a better monitored metric. The completed no-validation prefix prototype does not establish this condition by itself.

## Telemetry and checkpoint contract

Every epoch must record timestamp, phase/fold/arm, training row count, completed optimizer updates, normalized schedule progress, effective LR/WD/dropout, mean training objective with its reduction definition, inner-monitor AUC/log loss/Brier when applicable, elapsed time and memory. Record effective parameter-group LR/decay multipliers and selected factory branches. The current PBLD branch overrides the constructor's PLR activation field; resolved configuration must make that visible.

Add an epoch-end eval-mode probe on a fixed, hash-recorded subset of that fit's training rows, sampled only after partitioning. Report its AUC and per-row BCE beside the eval-mode monitor metrics. Minibatch train-mode loss includes changing dropout and may use a sum across eight members, so it is not directly comparable to eval-mode ensemble BCE. Preserve training mode and all RNG/sampler state around probes and snapshots. Require a synthetic telemetry-on versus telemetry-off comparison to establish that monitoring does not change the continued trajectory.

The current inner selector already evaluates AUC of the eight members' averaged probabilities. Keep that ensemble definition. Log full-precision sklearn AUC separately from the library's float32 comparison value if reproducing historical selection, without letting either change the fixed endpoints. Optional member diagnostics must capture logits before `_postprocess_ens_pred` and assert shape `[8, rows, 2]`; the public `predict_proba_ensemble` output has already collapsed the eight-member axis in this configuration. Do not introduce member-specific epochs or weights. See `research/deep_pass_selection_semantics.md`.

Save native inference checkpoints at the fixed endpoints with their preprocessing, class mapping and feature schema. Checkpoint predictions on fixed synthetic probes must survive reload; actual data parity must meet the established dtype-based export tolerances. Hash all files. Verify that B's saved tensor state does not change after C continues.

The current native export path is terminal: it moves the estimator to CPU, wraps shared fitted modules and rewrites some inference operations before tracing. Do not call that path on live C at epoch 4. Export B from a separately owned clone of its state, static preprocessing and schema. A synthetic capture-and-export versus no-export test must show identical live parameters, optimizer state, RNG, schedules and continued final predictions. Preserving B's bytes after capture is necessary but does not alone prove that capture/export left C unchanged.

This mechanism has now passed a combined generated-data CPU test with eight small members, learned preprocessing, real passive validation and the production inference helpers. The active PBLD branch stores its fitted preprocessing inside the vectorized network; clone and verify that state as well as `creator.static_model`. Native/reload parity is within 1.1921e-7, and all continued epoch states and final Adam tensors are exact. This is a prototype acceptance result, not production-adapter/CUDA verification. Before dispatch, integrate the controls in an isolated new adapter and exercise its complete feature and export path under the actual device contract. See `research/deep_pass_mixed_capture.md` and its independent review.

If exact resume is added, save optimizer moments, scheduler/progress counters, RNG states for every used generator, preprocessing, batch/sampler position, and the data/config/source versions. Test the next update after resume against uninterrupted execution. The current TorchScript exports and the callback-only prototype are inference/prefix evidence, not an exact-resume implementation.

Do not claim a completed four-epoch flat-anneal run can simply keep learning for another 12 epochs: its LR ends near zero. A restart or nonzero tail LR is a separate intervention. Here B/C avoid that ambiguity by declaring H=16 from the start.

## Metrics and decision rules

Primary mechanistic metrics: paired pooled and macro-fold ROC AUC for the three fixed contrasts, plus each fold's difference. Record log loss/Brier and curves as explanations, not replacement selection objectives. Report all endpoints including failures. These remain development results from a historically reused dataset.

For an ensemble-oriented follow-up, predeclare exactly one weight per arm: `0.90*incumbent + 0.10*arm`. Evaluate all three mixtures. No alpha search, no reoptimization of the incumbent's internal weights, and no retries on alternative seeds. This tests marginal usefulness and avoids silently conflating standalone quality with ensemble contribution.

A conservative **advance-to-confirmation**, not submission, rule is that C's fixed mixture must improve pooled and mean-fold AUC by at least `1e-5` over the incumbent and over both A/B fixed mixtures, while no fold loses more than `2e-5` against any of those comparators. These are engineering gates, not statistical significance. If a different arm is promising, record that outcome; it does not silently create a new success rule or another search inside this run.

The experiment may be valuable even when no arm passes: it can show whether late improvement occurs under a known trajectory, whether A/B differ from schedule alone, and whether training loss falls while monitor discrimination degrades. Do not infer irreducible noise or a universal training limit from a negative outcome.

## Execution budget and checks before dispatch

Use this local machine only, one owned GPU process at a time, no cloud or submission. Historical raw+EV 500-cap fitting averaged roughly nine seconds per inner epoch, including some overhead in its total. The 120-epoch protocol is likely on the order of tens of minutes, **an extrapolation**, and logging/export adds work. Set a conservative 90-minute fitting ceiling plus 30 minutes for verification and report. Measure a bounded synthetic full-path smoke before any real-data dispatch; revise the resource estimate before registration, not after observing quality.

Before launch: immutable new run IDs/config/source hashes; exclusive owned-worker check; available RAM/disk/GPU; preserved current release; exact epoch-clock controls; synthetic forward/backward and export checks; checkpoint immutability; finite-value checks; thread caps; durable timeout/exit reporting. Stop on resource/integrity failures or the registered deadline. Do not introduce a futility rule that stops C before its predeclared annealing window.

## Separate conditional experiments

1. **Optimizer semantics.** After clock/logging evidence, compare legacy decay to factor-once decay under identical short fixed endpoints. Correcting the duplication increases effective decay for some parameter groups by factors of 4, about 8.69 and 10. It is a regularization intervention, not a free correctness gain. A synthetic factor-once implementation must match the intended scalar update; a dose-preserving version must match historical updates before a data experiment. Keep library files and old artifacts untouched.
2. **EMA or fixed checkpoint averaging.** Only if trajectories show complementary checkpoints worth testing. EMA can be maintained online; it does not require retaining every checkpoint. SWA averages compatible weights; arbitrary independently initialized networks cannot be averaged as if they shared a parameter basin. Predeclare accumulation times/decay or checkpoint set and judge its held-out predictions. Existing exports cannot reconstruct missing epoch history.
3. **Genuine selection-procedure evaluation.** Regenerate base OOF predictions inside each outer-training partition before fitting weights. Reusing the current OOF matrix under a new meta split does not provide that separation. With three outer folds, three inner folds and the existing stopping/refit pattern, one recipe can need nine stopping fits, nine inner refits and three outer refits. The anchor's components also need appropriate reconstruction. Historical recipe selection still prevents a claim that this erases all past adaptive reuse.

Do not upgrade the library, optimize 15 free coefficients, add nonlinear gating, search fine decimal weights, or refit exposed audit data as an alleged new holdout as part of the duration test.
