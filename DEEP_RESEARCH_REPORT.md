# Why longer training has not helped, and what the blend weights mean

**Completed research report, October 2, 2026.** Three-agent investigation: 21:01 UTC to 22:08:31 UTC, approximately 67.52 minutes of wall-clock time. This is shorter than the requested 3-4 hours. The bounded diagnostics and substantive independent reviews finished early; the remaining scientific questions need new instrumented training or properly nested evaluation. No claim is made that three hours of research elapsed. Exact completion and scope are recorded in `state/research_pass_v1/progress.json`.

The current evidence supports **keeping the existing blend and fixing the next experiment's training controls before spending more time on epochs**. No training campaign, release or submission was changed in this pass.

## What the completed duration experiments actually measured

| Recipe/platform | Epoch ceiling | Selected epochs | Development pooled AUC | Change from its matched short control |
|---|---:|---|---:|---:|
| Raw + auxiliary expected values, local | 4 | 3 / 3 / 3 | 0.9608520596 | reference |
| Same recipe, local | 60 | 3 / 4 / 4 | 0.9607842682 | -0.0000677914 |
| Same recipe, local | 500 | 4 / 4 / 4 | 0.9607352453 | -0.0001168143 |
| Raw + auxiliary expected values, cloud | 4 | 3 / 3 / 4 | 0.9608107753 | reference |
| Same recipe, cloud | 12 | 4 / 3 / 4 | 0.9607459889 | -0.0000647864 |
| Same recipe, cloud | 60 | 4 / 4 / 4 | 0.9606847043 | -0.0001260709 |
| Numerical TE + teacher, local | 4 | 4 / 4 / 4 | 0.9606333241 | reference |
| Same recipe, local | 16 | 5 / 6 / 5 | 0.9603704701 | -0.0002628540 |

All eight saved development OOF results were independently reproduced with exact ID/fold/label checks. Comparisons across platforms are not causal controls. These results argue against their registered longer-cap recipes; they do not prove that every form of longer training fails.

The 500-cap run took 13,688.797 seconds, about 3 hours 48 minutes, for its training pipeline including overhead. Each inner run traversed all 500 epochs, selected epoch 4, then discarded that fitted model for outer scoring. The outer model was initialized again and fitted four epochs. Its nominal four-epoch control actually selected and refitted three epochs.

Three separate effects are entangled:

1. **The cap changes the schedule from the beginning.** PyTabKit's schedules use normalized training progress. `flat_anneal` begins annealing at 60% of the horizon: around epoch 2.4 for a four-epoch horizon, but epoch 300 for a 500-epoch horizon. Weight decay and dropout also follow normalized schedules. The longer job does not preserve the short schedule's first four epochs.
2. **The outer refit changes the schedule again.** An epoch-4 checkpoint from a 500-epoch schedule is not the same training procedure as a fresh four-epoch schedule. `scripts/train.py` supplies selected rounds to `scripts/common.py`, which sets that count as the new RealMLP horizon.
3. **Equal epochs are not equal update counts when data size changes.** At batch size 256, the inner fit executes 1,475 updates per epoch, versus 1,639 in the outer refit. Four epochs means 5,900 versus 6,556 updates, an 11.12% difference. Eight vectorized neural members do not multiply that optimizer-step count by eight.

The completed 500-epoch inner searches selected epoch 4 under the library's float32 ensemble-AUC rule; per-epoch curves and checkpoint predictions were not retained. This does not establish that training loss kept falling after epoch 4, quantify later deterioration or recovery, or let us reconstruct an epoch ensemble. Saved TabM curves do show falling training loss with worsening inner AUC, but that is evidence about those TabM runs, not proof of the RealMLP mechanism.

An additional source audit rules out a plausible aggregation error: selection uses the AUC of the eight members' averaged probabilities, matching the inference ensemble's mathematical form. All eight members share one selected epoch; `[4,4,4]` indexes the three outer folds, not individual members. Float32 metric ties choose the latest tied epoch. There is no evidence that this numerical detail explains the observed regression. The remaining objective distinction is standalone RealMLP AUC versus its marginal contribution to the full blend: the best standalone checkpoint need not be the best blending partner. Missing checkpoint predictions prevent a retrospective test. See [selection semantics and executed mathematical example](research/deep_pass_selection_semantics.md).

The source review and a new synthetic CPU test found a usable control: the installed `StopAtEpochsCallback` can keep the original schedule horizon while stopping at a fixed earlier epoch. A 12-epoch toy trajectory's epoch-4 state and the H=12/stop=4 run matched exactly in parameters, buffers, probabilities and 32-step schedule trace. Compressing H to 4 changed the trajectory. This proves the control mechanism on the toy, not a competition-score gain or GPU determinism.

Details: [epoch diagnosis](research/deep_pass_epoch_diagnosis.md), [measured duration diagnostics](artifacts/research_pass_v1/epochs/diagnosis.json), [synthetic control verification](artifacts/research_pass_v1/epochs/control_prototype/verification.json).

## A real optimizer discrepancy, with a limited causal conclusion

The installed PyTabKit 1.7.3 applies per-parameter learning-rate and weight-decay factors twice in manual decay. Upstream [issue 40](https://github.com/dholzmueller/pytabkit/issues/40) describes the same behavior; the inspected tagged implementation matches the installed file. Six scalar zero-gradient checks reproduced it locally, and a synthetic compilation identified the parameter groups active in this recipe.

If the base rates are `lr` and `wd`, and parameter factors are `a` and `b`, the intended shrink coefficient is `lr*wd*a*b`. The installed path uses `lr*wd*a²*b²`. Relative to the factor-once rule, the first-layer weights receive one-quarter of the decay, the active PBLD group receives 0.1151 times the decay, and parametric activations receive one-tenth. Groups with factor one are unchanged; biases configured for zero decay stay at zero.

This is **not an established explanation for the longer-run loss**. Successful short runs share the implementation. Correcting the duplication while keeping global hyperparameters constant increases shrinkage substantially for those groups and invalidates the previous effective regularization dose. It needs a separate versioned ablation, after schedule/logging controls, rather than an unrecorded package patch.

Another configuration detail matters for reproducibility: the inherited PBLD factory branch overrides the constructor's PLR activation field. A record of constructor arguments alone does not fully describe the resolved network. Future telemetry should include resolved factory settings and parameter groups.

Details: [synthetic optimizer evidence](artifacts/research_pass_v1/epochs/synthetic_optimizer.json), [pinned source ledger](research/deep_pass_sources.md).

## The blend is already near a broad good region

The exact retained high-level blend is:

```text
0.64 × frozen v2 blend
+ 0.16 × v3 RealMLP with auxiliary probability features
+ 0.20 × v3 XGBoost with auxiliary probability features
```

The 16% is not mysterious: the first 20% addition is multiplied by 80% when the second 20% component is added. Internally the final 15-model weights total 47.5701% RealMLP, 32.6174% XGBoost, 17.8973% CatBoost and 1.9152% LightGBM. Registered seed pairs retain equal internal weights.

With internal neural and tree proportions held fixed, moving to an exact 50/50 neural/tree mixture changes pooled AUC from 0.9614075009 to 0.9614076323, only **+0.0000001314**. At 40% neural weight the change is -0.0000082527; at 60% it is -0.0000115076. This supports a useful neighborhood around the current balance; it does not identify a uniquely correct decimal coefficient.

A registered 49-point coarse grid varied only the two accepted v3 block weights while preserving all v2 internal ratios. Its best tested combination, 65% v2 / 20% new RealMLP / 15% new XGBoost, gained **0.0000015830** pooled AUC and **0.0000016401** macro-fold AUC, but lost 0.0000035707 on fold 3. It misses the unchanged 0.00001 operational improvement gate. The old three-pool/twelve-mixture failed-checkpoint screen was not repeated.

Selecting the fixed block grid on two existing folds and checking the third chose different weights:

| Excluded fold | Weights v2 / new NN / new XGB | Change on excluded fold |
|---|---|---:|
| 1 | 60% / 20% / 20% | +0.0000010890 |
| 2 | 70% / 15% / 15% | -0.0000012622 |
| 3 | 65% / 20% / 15% | -0.0000035707 |

This is conditional weight-stability evidence, **not nested validation**. Models generating other folds' OOF predictions can have trained on the excluded fold, and the recipes were selected using these labels previously. The procedure cannot restore independence.

For two models a and b, a blended positive-negative margin is `d_a + alpha*(d_b-d_a)`, where `d_a` is model a's positive score minus its negative score. AUC changes when such a margin crosses zero. The empirical objective is therefore a step function of alpha, not a smooth, uniquely identifiable optimum. Highly correlated components can exchange weight with almost unchanged predictions. A generic continuous optimizer or more decimal places does not supply additional evidence about the correct coefficients.

![Blend weight sensitivity](artifacts/research_pass_v1/blend/blend_weight_sensitivity.png)

## Measure conditional contribution, not just standalone score or correlation

Removing one family and renormalizing the remaining weights, without reoptimization, changes AUC as follows. The sign below is the AUC advantage of the incumbent over the removal.

| Family removed | Incumbent advantage | Interpretation |
|---|---:|---|
| RealMLP | +0.0002688131 | Strong conditional complement to trees |
| XGBoost | +0.0000657398 | Useful conditional contribution |
| CatBoost | +0.0000340430 | Smaller positive aggregate contribution |
| LightGBM | +0.0000000219 | Negligible aggregate contribution here; fold signs differ |

These removals measure contribution inside the current mixture. They are not causal feature importances, fair retrained-family comparisons, or automatic instructions to delete a model. A weak standalone model can contribute complementary rankings. Conversely, a stronger standalone model can be redundant. Existing duration admission gates remain valid historical decisions; any future ensemble-only hypothesis must be registered prospectively.

Correlation alone obscures this distinction. The incumbent and v2 scores have correlation 0.9999233, yet the incumbent repairs 105,331,719 positive-negative pair orderings and breaks 101,047,297, for a net 4,284,422 repaired pairs. Moving from current neural weight to 50% repairs 12,354,551 and breaks 12,341,690: nearly complete cancellation. The latter net is only 12,861 out of 97,858,975,630 positive-negative pairs.

These are exact strict-rank counts, checked against brute-force synthetic pair enumeration. **The 97.9 billion pairs are not independent observations.** Many pairs share each row; their count cannot justify tiny confidence intervals.

## What uncertainty can and cannot establish

For the fixed incumbent minus fixed v2 comparison, paired AUC difference is +0.0000437816, with a conditional placement-based standard error of 0.0000125446. Its descriptive 95% interval is [0.0000191946, 0.0000683686]. A separate 200-replicate paired, class-stratified bootstrap agrees on the scale: SE 0.0000118495. LightGBM removal's corresponding interval spans zero.

These calculations assume independent sampled rows and condition on the already fitted, already selected predictions. They omit training randomness, overlapping-fold training dependence, recipe/weight selection and distribution shift. They do not make the adaptively selected incumbent an independently tested result. The 1e-5 release gate is an operational threshold, not a significance test. Uncertainty for one contrast must not be reused as the standard error for a different nearby weight change.

The follow-up calculation uses the exact paired placement covariance for all 49 already registered grid points. For the observed best 65%/20%/15% mixture, the gain of 0.0000015830 has its own paired SE of **0.0000034999**. Its descriptive pointwise 95% interval is [-0.0000052767, 0.0000084427]. A 20,000-draw Gaussian approximation using the joint covariance gives a fixed-grid simultaneous reference band of [-0.0000071487, 0.0000103146]. These retain all the conditional/iid limitations above; they are not full post-selection generalization intervals.

The observed best point wins 25.88% of these perturbations among the 49 grid candidates; two neighboring points win 18.84% and 15.23%. Those frequencies describe sensitivity to estimated noise, not probabilities of being the true best model. The incumbent is not a candidate in this frequency calculation. No grid point has a positive lower reference bound. However, the best point's simultaneous upper bound slightly exceeds 1e-5, so this calculation does **not** prove equivalence or rule out a gate-sized benefit. Retaining the incumbent follows the observed improvement/fold gates and lack of stable weight evidence.

All 49 saved AUCs were independently reproduced. Eleven synthetic tests passed, including covariance/paired-SE agreement, ties, singular matrices and invalid covariance rejection. Independent review reconciled the six recorded source/input hashes, covariance diagonals, bands and 20,000 winner counts. The estimated covariance was positive definite, requiring no eigenvalue clipping or diagonal jitter. See [joint grid diagnostic](artifacts/research_pass_v1/validation/grid_uncertainty.json) and [independent review](research/deep_pass_grid_uncertainty_review.md).

Log loss, calibration and AUC also answer different questions. In the executed comparison, equal thirds of the three main blocks improves Brier score slightly while reducing AUC by 0.0000324200. A strictly increasing transform of the final score preserves AUC, but changing the scale of individual components can change their blend rankings. This is why a weight is not a model's probability of being right, and why replacing probability averaging with logit/rank averaging requires its own validation.

Details: [validation review and formulas](research/deep_pass_blend_validation.md), [paired uncertainty](artifacts/research_pass_v1/validation/paired_auc.json), [bootstrap](artifacts/research_pass_v1/validation/paired_bootstrap.json), [ranking geometry](artifacts/research_pass_v1/blend/pair_geometry.json).

## Recommended next work

The detailed proposal is [next_epoch_blend_experiment.md](research/next_epoch_blend_experiment.md). It has not been launched.

1. Persist learning curves, effective schedules and native endpoint checkpoints; separate horizon from executed epochs. Keep old artifacts and the current release intact.
2. Compare fixed H4/E4, H16/E4 and H16/E16 arms, with H16/E4 exported from the same continuing H16 trajectory. This separates the horizon intervention at four epochs from continuing the same H16 trajectory, including its later schedule. To obtain the missing curves without selecting on outer labels, include a fixed inner-monitor stage; lengths never adapt to those curves. Three folds then require 12 fits and 120 total epochs. The six outer fits alone are the cheaper 60-epoch mechanism test, but omit inner learning curves on rows disjoint from that fit's training subset.
3. If ensemble usefulness is the question, predeclare one common weight, such as 10%, for each new arm against the unchanged incumbent. Compare all fixed mixtures. Do not search 15 free coefficients or treat a failed old standalone candidate as newly qualified.
4. Test corrected decay semantics separately. Only consider EMA/SWA or fixed snapshot averaging once compatible training trajectories and state are available. There is no saved per-epoch sequence to blend retrospectively.
5. Before a stronger performance claim, evaluate the whole selection procedure with base fits regenerated inside the outer training boundary. That is substantially more expensive than re-splitting saved OOF. Even a correctly nested new procedure does not erase earlier label-informed recipe decisions; the exposed original audit cannot be reused as an untouched holdout.

The mechanism proposal uses local compute only and reserves verification time. Its 90-minute fitting / 30-minute verification caps are proposed safety bounds, not launched work or measured runtimes. Historical throughput suggests tens of minutes, but new telemetry/checkpoint overhead must be measured before dispatch.

Adversarial review identified a second endpoint trap: disabling early stopping alone does not disable the library's final restoration of the best validation checkpoint. The next fixed-endpoint adapter must explicitly disable that restoration and pass a test where an earlier monitored epoch is better than the final epoch. Training probes also need eval-mode metrics and a check that logging/snapshots preserve the continuing RNG and model state. These requirements are now explicit in the proposal; the no-validation prefix test does not cover them.

The follow-up test now covers that trap with an actual synthetic validation loader and eight small vectorized members. Injected monitor errors deliberately make epoch 2 better than epoch 4. Default behavior restores epoch 2 despite `use_early_stopping=False`; low-level `use_best_epoch=False` retains literal epoch 4. Passive probes and cloned snapshots leave batch order, schedule/RNG traces, final parameters and predictions unchanged in this CPU test. These injected errors are a checkpoint-selection stimulus, not model-quality results. See [passive-monitor verification](artifacts/research_pass_v1/epochs/passive_monitor/verification.json).

A separate resume test exposed an additional installed-library limitation: an ordinary Lightning checkpoint contains zero wrapper optimizer-state entries although the underlying toy Adam has 13, and ordinary load fails with `KeyError('__dict__')`. A research-only checkpoint explicitly restoring underlying Adam state, progress, preprocessing and Python/NumPy/Torch CPU RNG reproduces the uninterrupted next batch, next update, final parameters, optimizer tensors and probabilities exactly. This supports epoch-boundary deterministic CPU resumption under that toy's contract. It does not establish mid-epoch, validation-callback, CUDA or production-wrapper resumability. No installed library was patched. See [resume verification](artifacts/research_pass_v1/epochs/resume_prototype/verification.json).

The capture/export test uses the actual production schema, split, mixed-graph and portable-inference helpers on a separately owned clone. Native, rewritten, saved/reloaded, single-row and reversed-batch predictions agree exactly on the synthetic probes. Export leaves ten live-state fingerprints unchanged, and the continuing run's batches, schedules, final parameters, probabilities and Adam tensors match the no-export control. The saved prefix hash remains unchanged after continuation. This covers one small CPU member with an empty static-preprocessing state dictionary; learned production preprocessing, eight-member production integration and CUDA remain unverified. See [capture/export verification](artifacts/research_pass_v1/epochs/capture_prototype/verification.json).

A final combined test closes the small test's learned-preprocessing and eight-member gaps on generated CPU data. It uses the production transform list, both categorical encoding branches, 288 fit rows and a real 96-row monitor. Six fitted nontrainable preprocessing tensors reside inside the eight-member network. Changing only monitor features leaves the training schema, fitted preprocessing and initialized network unchanged. Across H16 runs with and without added probes/cloned epoch-4 export, all 16 epoch states, batch/RNG/schedule/monitor traces, final parameters, Adam state and probabilities agree exactly. Both reach literal epoch 16 although the best monitored epoch is 15. Native-library versus graph/portable/reloaded predictions differ by at most 1.1921e-7, within the preregistered tolerances; unknown/unseen handling and the saved prefix remain stable.

The first attempt stopped before optimization because the test incorrectly expected fitted preprocessing only in `creator.static_model`. Its source and failure receipt were retained, and one inventory repair passed. The active PBLD branch places these statistics inside the vectorized network; an initial source-level suspicion of repeated static preprocessing was retracted after inspecting that realized path. No preprocessing defect or correction is claimed. Two independent reviews accepted the corrected proof. Production-fitter integration, the external `common.py` feature pipeline, CUDA and arbitrary non-module state ownership remain outside this test. See [combined test and limits](research/deep_pass_mixed_capture.md), [verification](artifacts/research_pass_v1/epochs/mixed_capture/verification.json), and [independent review](research/deep_pass_mixed_capture_review.md).

Finally, the synthetic nesting proof confirms why a new split of old OOF does not fix the selection boundary. Every toy OOF row excludes its own label, yet changing only a proposed outer-held-out fold's labels changes the OOF features used to train the meta model on other rows. Regenerating inner OOF entirely inside the outer-training partition is invariant to that label change. The test also checks preprocessing exclusion and distinguishes a legitimate change in query predictions when held-out features change. It demonstrates a dependency path, not the direction or magnitude of bias in this competition. See [nesting proof and diagrams](artifacts/research_pass_v1/validation/nesting/dataflow.md).

## Verification, remaining limits and closeout

Completed: exact frozen split/result/OOF reconstruction; an independent audit of all 15 selected model contracts and archived source/environment hashes; agreement between two implementations on family ablations and equal-block scores; seven blend-contract tests; eight paired-AUC tests; four bootstrap tests; four ranking-geometry tests; six optimizer scalar checks; four schedule/prefix prototype tests. Tests use synthetic data. No audit labels/predictions or test predictions were scored by these diagnostics.

The initial pair-geometry test exposed a SciPy two-observation asymptotic edge case; the exact two-point case now bypasses that formula and tests pass. Independent review found a malformed duplicate seed-group input was not rejected; it is now rejected without changing any valid registered group or result. Executed versions of the two affected root scripts were archived under `artifacts/research_pass_v1/source_snapshots/`, so the output source hashes still resolve. Numerical results were not reselected after these hardening changes.

The next review added ten passing nesting-contract tests, three passive-monitor tests, one exact-resume acceptance test and one capture/export acceptance test. Independent source review checked the latter's evidence and prompted explicit single-row/reversed-batch assertions; the amended test passed. The [adversarial protocol review](research/deep_pass_protocol_adversarial_review.md) found no evidence to prioritize an AUC-specific training loss or calibration over fixing schedule/logging controls: the local 500-cap recipe worsens AUC, log loss and Brier together, and missing curves prevent a loss-mismatch diagnosis.

The combined eight-member test adds one passing acceptance test after one preserved pre-fit inventory failure. Its result is mechanism evidence, not model-quality evidence. The [final independent integration review](research/deep_pass_final_review.md) found no blocking issue within the stated scope. All substantive diagnostics and reviews are complete. The remaining scientific uncertainty requires newly instrumented fitting or genuinely nested evaluation, rather than another search over these same saved predictions.

Timestamped artifact inventories are under `artifacts/research_pass_v1/manifests/`. The final manifest records the delivered research files and 17 explicit preservation/binding checks: six pinned inputs including the release-provenance record, eight live/archive production sources matched to that record, and three executed-source bindings for the two root scripts hardened after execution. Other agents' source/import bindings are established by their individual recorded reviews, not merely by the inventory. The completion receipt and actual elapsed time are in `state/research_pass_v1/progress.json`.

No new real-data model was fitted, GPU/cloud campaign dispatched, audit scored, release changed or submission attempted in this research pass. Synthetic CPU fits test implementation mechanics only. The proposed A/B/C experiment, a production adapter, CUDA verification and any stronger selection-procedure evaluation remain future work; they are not silently marked implemented or successful.

Primary references and current competition observations, with inspection depth and limitations, are in [the source ledger](research/deep_pass_sources.md). It distinguishes author-reported notebook scores from executed results and notes that public OOF-meta resplitting is not full nesting. No leaderboard claim from another author's notebook is treated as our reproduced performance.
