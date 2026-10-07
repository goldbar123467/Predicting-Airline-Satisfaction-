# Adversarial review of the proposed duration/blend experiment

Reviewed October 2, 2026 UTC by methods_research. Scope: design/source review, three additional primary papers, and the explicitly authorized synthetic passive-monitor regression. No competition models were fitted, no OOF metrics were recomputed, no audit data were read or scored, and no production files were changed.

The proposed experiment is scientifically useful after the endpoint and monitoring guards described below. AUC-specific training or calibration should not move ahead of those guards on the evidence currently available. The important unresolved problem is that previous runs did not preserve the information needed to identify why training longer lost. A more elaborate loss cannot repair that experimental ambiguity.

Initial reviewed `research/next_epoch_blend_experiment.md` SHA-256: `69df849089893ab9d4fb93c5bf58f768fee8188950d8ffe89c9dbf6b8a4b594f`. After root incorporated endpoint-restoration and passive-probe objections, inspected SHA-256: `255a566f9380fc9305685d4b9d338dbd288f3a4d93b0d26b46eb80cf8d17077d`. Subsequent edits may legitimately change this working proposal; these hashes identify the reviewed versions, not immutable experiment registration.

## Design audit

| Question | Finding | Consequence |
| --- | --- | --- |
| Is A the historical short control? | No. A is a newly fitted H=4, stop=4 endpoint. The historical raw+EV control selected and refitted three epochs in all three folds. The proposal explicitly separates them. | This three-arm experiment answers a mechanism question. It does not rewrite the old four-cap control or the failed 500-cap gate. |
| Does B versus A isolate duration? | No, both execute four epochs. It changes the planned horizon and consequently LR, WD and dropout paths together. | Call it a normalized-schedule intervention. It cannot isolate which individual schedule mattered. |
| Does C versus B preserve the early path? | Yes in the intended implementation: B is a cloned checkpoint from C's epoch-four state, with identical preprocessing and member identities. | C minus B tests continuing that trajectory through twelve more epochs, including its later schedules. It is not the effect of twelve extra updates at constant LR or fixed regularization. |
| Can B be a live model reference? | No. A shallow state dictionary or shared tensor storage can mutate after C continues. | Clone parameters and buffers, archive schema/preprocessing, and verify unchanged hashes after C finishes. Probe/snapshot side effects also need separate testing. |
| Does the inner-to-outer refit reproduce an identical update sequence? | No. The chosen epoch clock changes updates per epoch from 1,475 to 1,639, approximately +11.12%, and refits preprocessing on more permitted rows. | This is acceptable as an explicitly defined procedure. Do not call it equal-update transfer or exact replay of the inner path. |
| Is all-row preprocessing prohibited? | Yes for learned dictionaries/statistics. Current `common.py` fits the external transformer on supplied training rows. The mixed adapter fixes category metadata from that schema. PyTabKit's static split is defined to need no tensors and be nontrainable; learned dynamic transforms are fitted on `train_idxs`. | An apparent `static_fitter.fit_transform(ds)` over concatenated data is not by itself evidence of statistical leakage. Preserve the train-only schema and dynamic-fit contract in the new adapter; do not substitute global vocabularies or scaling. |
| Are there 12 fits and 120 epochs? | Yes: per fold, inner A4+C16 and outer A4+C16 = 40 epoch passes across four fit calls; three folds give 120 passes and 12 calls. B adds no fit. Weighting inner passes by 0.9 gives approximately 114 outer-size epoch equivalents. | The monitor-only inner stage is an explicit diagnostic cost, not endpoint selection. Six outer fits would suffice for fixed endpoint comparison but would lose the disjoint learning-curve diagnostic. The current 12-fit proposal is internally consistent. |
| Is the 1e-5 gate statistical evidence? | No. It is an operational advance-to-confirmation rule, including fixed fold-regression bounds. | Crossing it cannot imply a significant, independent or private-leaderboard gain. The proposal correctly limits its consequence. |
| Can later nested fitting erase earlier searches? | No. Regeneration inside outer training sets fixes that new procedure's row exclusion, but not historical choice of recipes and hypotheses. | Preserve the historical-adaptation warning and never reuse the exposed audit as a fresh holdout. |

The frozen original-only auxiliary banks are an allowed reuse exception only under their recorded provenance and overlap exclusions. The new adapter should not interpret that exception as permission to fit any additional transformer to monitor, outer assessment, audit or test rows. The present review did not re-audit every bank row or fit a replacement bank.

## Objections raised and resolved in the proposal

**Fixed endpoints can silently become best-validation endpoints.** Installed `nn_creator.py:198` adds `ModelCheckpointCallback` when validation exists, fit parameters are absent, and `use_best_epoch` is true, which is the default. `lightning_callbacks.py:125–140` restores the stored best weights at fit end. Turning off early termination alone does not remove that behavior. The original prefix toy had no validation loader and therefore did not establish monitor-stage isolation. Root has now required either an explicit low-level `use_best_epoch=False` path or monitoring outside checkpoint selection. The public 1.7.3 estimator constructor and current local adapter do not expose that flag directly.

**Train-mode objective curves and eval-mode monitor curves are not directly comparable.** Dropout changes over training, minibatch objectives use the evolving model, and the implementation sums the per-member loss across eight members. The new proposal now adds fixed epoch-end eval-mode training probes, with per-row BCE and AUC alongside monitor metrics. Those probes must be sampled only from fit rows and remain fixed. Optimizer weight decay is a parameter update, not automatically a term contained in the reported BCE.

**Cloning snapshots is necessary but insufficient for passive observation.** Prediction calls can change module modes, consume RNG streams or disturb sampler iteration. Root now requires telemetry-on/off equivalence. Future CUDA/production-wrapper validation remains necessary; a passing small CPU mechanism test does not establish all backends.

No protocol quality gate was relaxed during this review. The old closed saved-checkpoint screen remains closed. No failed duration recipe was promoted.

## Executed synthetic proof

The new isolated files are `scripts/research_passive_monitor_v1.py` and `scripts/test_research_passive_monitor_v1.py`. The test uses the installed `TabNNModule`, actual 192-row training and 64-row validation loaders, eight vectorized small networks, and four full CPU epochs. Inputs are generated by the existing toy-data helper. Only the validation scorer is temporarily replaced with the explicit error sequence `[0.2,0.1,0.3,0.4]`, so epoch two is deliberately best. This is a controlled test stimulus, not a real AUC measurement. Library selection/restoration code is not mocked, and installed source is not changed.

Three tests passed using:

```powershell
.venv/Scripts/python.exe -m unittest discover -s scripts -p test_research_passive_monitor_v1.py -v
```

The successful fit/check phase took about 0.55 seconds, roughly 6.5 seconds including Python imports. The run had a 120-second CPU ceiling. Two bounded implementation repairs preceded the successful run: the flag had to be supplied to the low-level module configuration because the estimator constructor rejects it, and the eight-member probe needed explicit input replication because the reused one-member helper did not support that axis.

| Verified claim | Observed result |
| --- | --- |
| Default restoration can violate a fixed endpoint even when early stopping is off | Final state equals epoch two exactly; max absolute difference from literal epoch four is 0.3795493841 |
| Turning off best-epoch restoration preserves the endpoint | Final state equals epoch four exactly |
| The restoration flag did not change the pre-fit-end trajectory | Every recorded epoch state is identical before final restoration |
| Added eval-mode probe and cloned snapshot preserve continued training | Final state and probabilities have zero difference; every batch's data hashes, schedule values and Torch RNG hash match |
| Observation preserves final RNG and frozen snapshot | Final RNG hashes match; the snapshot's tensor hashes remain unchanged after later training |
| Validation was actually exercised | Four validation batches in each of three runs, all with the expected 64-row, two-class prediction contract |

Evidence, exact traces and source hashes are in `artifacts/research_pass_v1/epochs/passive_monitor/verification.json`. The telemetry-off baseline retains lightweight audit capture so equivalence can be measured; the comparison isolates added prediction probes and snapshotting. It is not a claim about a completely callback-free trainer. No optimizer-state resume or production export test is included in this proof.

The epoch agent independently source-reviewed this prototype without re-running it and found no blocking comparison or interpretation issue. Separate capture/export work owns production-helper and static-preprocessor mutation checks; they are outside this proof's claims.

In return, this reviewer inspected the epoch agent's `research_epoch_capture_v1.py`, its test and saved capture verification, without rerunning it. All five recorded source hashes matched the inspected files. The use of actual production schema, inference-graph and portability helpers on deep copies is appropriate; it checks disjoint tensor storage, live modes/forward methods, parameters, Adam, RNG, progress, schedules, continuation and immutable artifact bytes. All reported native/portable/reload/batch-order/single-row prediction differences were zero. The toy's static preprocessing has an empty state dictionary, so this does not empirically cover arbitrary stateful external preprocessing. A nonblocking regression-hardening suggestion was sent to the owner: explicitly assert the recorded reversed-batch and one-row parity values if they are intended to remain acceptance conditions. The observed results support the stated one-member CPU scope, not a production or CUDA integration claim.

## Three focused primary sources on loss and calibration

1. **Agarwal, JMLR 2014, [Surrogate Regret Bounds for Bipartite Ranking via Strongly Proper Losses](https://jmlr.org/papers/volume15/agarwal14b/agarwal14b.pdf).** Inspected formal setup, Theorem 13 and logistic-loss example in sections 5.1–5.2. Logistic loss is strongly proper and its population regret bounds bipartite-ranking regret. Consistency requires appropriate function-class/regularization conditions. This is theory, not an empirical RealMLP benchmark, and a bound does not make each finite SGD step monotonically improve AUC.
2. **Yuan et al., ICCV 2021, [Large-scale Robust Deep AUC Maximization](https://arxiv.org/pdf/2012.03173v2), arXiv v2 September 7, 2021.** Inspected sections 3–5 and appendix J. A min-max margin surrogate improves several image-task comparisons, especially heavy imbalance. Experiments include 1%/10% synthetic class ratios and four medical datasets; difficult tasks use CE pretraining followed by AUC optimization with additional tuned parameters. These are meaningful results, but not a plug-in objective-only change or evidence for our roughly 44% positive tabular problem. Direct CVF opening failed; the primary author arXiv version supplied the full methods and evaluation.
3. **Guo et al., ICML 2017, [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).** Inspected sections 3–5: held-out post-processing, temperature/Platt/binning methods and vision/NLP evaluation. A fitted positive scalar temperature often improves probability calibration with little complexity. Their main evaluation is calibration, not binary ensemble AUC. It does not establish that calibrating our constituents improves their blend or explain the 500-cap regression.

No new method package was installed, copied or approved for reuse. The existing Caruana ensemble-selection evidence remains the relevant source for selecting by marginal ensemble performance; no additional model-shopping literature was added.

## Does objective mismatch justify changing priority?

No, on current evidence. For a binary conditional class probability eta, expected BCE at a probability p is `-eta*log(p)-(1-eta)*log(1-p)`. Its derivative is `(p-eta)/(p*(1-p))`, so the unconstrained optimum is p=eta. Ranking by eta is Bayes-optimal for AUC. BCE and AUC are different finite-sample objectives, but they are not fundamentally incompatible at their population optima.

Our finite network, regularization, optimizer and sample do not satisfy an automatic optimality guarantee. Train BCE can fall while held-out ranking worsens, and even held-out BCE can improve while AUC falls. But the existing local 500-cap result worsened AUC, log loss and Brier together. There is no saved late trajectory showing “probability fitting continues to improve while ranking alone fails.” That weakens the claim that mismatch, rather than schedule/refit or generalization problems, is the immediate bottleneck.

Direct AUC training remains a plausible later hypothesis. It changes gradient estimation, objective scales, sampling behavior and usually tuning requirements; adding it to the clock experiment would destroy interpretability. Its eventual comparator would need the same initialization/representation, matched update and tuning budgets, untouched assessment within that future procedure, and explicit probability-output handling before blending. No expected gain is quantified here.

Calibration is similarly conditional. For one binary final score, positive-temperature scaling and positive-slope Platt scaling are strictly increasing maps, so they preserve AUC apart from numerical ties. They can improve probability estimates without repairing any rank error. Applying different calibration maps to individual members before averaging can change ensemble rankings, but then the full transformed blend is a new model requiring separate selection/evaluation. Isotonic maps can create ties; calibration is not a universal ranking improvement.

The existing equal-three-block diagnostic slightly improves Brier/log loss while losing AUC. That is direct local evidence to keep ranking as the decision metric and use probability metrics diagnostically. There is no current calibration evidence strong enough to displace the bounded schedule/endpoint test.

## Fixed-weight marginal contribution and remaining unknowns

The proposed three mixtures at alpha=0.10 are legitimate fixed comparisons. They measure whether each endpoint helps this frozen anchor at that amount, without finding a weight from the same assessment labels. A weaker standalone model can help if it repairs enough anchor ordering errors; a stronger standalone model may add redundant signal. Low prediction correlation alone cannot prove either outcome.

Fixing alpha also limits the conclusion: failure at 10% does not prove all weights are useless. Different score scales change a model's influence at the same nominal coefficient. This is part of the proposed practical prediction recipe, while standalone AUC and probability diagnostics help explain it. Searching another alpha, applying calibration, or substituting rank averaging after seeing these outcomes would be a new experiment, not a repair of the registered result.

The observations cannot yet identify which normalized schedule matters, whether useful recovery occurs after sixteen epochs, whether another seed changes the paired comparison, whether stronger calibration would improve a frozen blend, or what the full-data refit/private leaderboard effect would be. The six monitor fits will describe behavior on smaller training sets; they do not determine exact outer trajectories. Historical adaptation remains present even after fresh nested refits. These are limits to state, not reasons to spend another 500 epochs without telemetry.
