# Blend validation and paired uncertainty audit

Prepared October 2, 2026, during the registered deep research pass. This analysis preserves the submitted incumbent and does not select or publish a new blend. It uses only frozen development identities and accepted-model OOF predictions. No raw training table, audit predictions/labels, test predictions, model fitting, cloud execution, or submission was used by the diagnostic script.

## Findings that change the next decision

1. The current blend arithmetic is correct. Its meaningful high-level weights are **64% frozen v2 + 16% v3 probability-feature RealMLP + 20% v3 probability-feature XGBoost**. The first new model entered at alpha .20; the second .20 addition scales that first weight by .80, leaving .16. The original v2 ratios remain intact. Fifteen printed individual coefficients are implementation consequences of this history, not fifteen independently estimated feature importances.
2. Longer training has not produced a better accepted ingredient. The 500-horizon run selected four epochs in every inner fold and refitted each outer model for four epochs. Its recorded pooled AUC is 0.9607352453 versus 0.9608520596 for the four-horizon control; selected control lengths are [3,3,3]. The outer test is of those selected-length refits, not a 500-epoch deployed network. Schedule compression is a separate confound detailed by the epoch agent. The failed duration gates stay failed.
3. The ensemble needs both neural and tree components. Removing all RealMLP mass while preserving tree ratios loses 0.0002688131 development AUC. Removing all XGBoost loses 0.0000657398; removing CatBoost loses 0.0000340430. Removing the 1.9152% LightGBM member changes AUC by only 0.0000000219 in favor of retaining it, with mixed fold signs. This identifies a possible future simplification question, not grounds to edit the existing release.
4. The current 0.00001 acceptance threshold is an operational tolerance. It is smaller than the paired fixed-score SE of the observed incumbent-versus-v2 gain, 0.0000125446. It is not a statistical significance boundary. Repeated selection and model-training uncertainty make a campaign-level claim harder, not easier.
5. Base-model stopping is separated from outer scoring. Blend selection is not separately assessed. Re-splitting the saved OOF matrix can reveal weight instability, but cannot produce an independent meta-model test because other OOF predictors were trained using the held-out meta-fold labels. Existing project notes already acknowledge this limitation; this audit found no new evidence that audit labels entered selection.

## What was verified

The independent script [analysis_auc_uncertainty_v1.py](../scripts/analysis_auc_uncertainty_v1.py) pins hashes for the registered plan, frozen v2/incumbent selections, and split file. It first filters split rows to fold >= 0, then validates all 15 selected result/contract/OOF hashes, archived training-source and lockfile hashes, exact keyed development coverage, folds, identical binary labels, finite probabilities and each recorded pooled/fold AUC. It independently reproduces both frozen blends. It does not import training or release modules.

The development population is 629,671 rows: 279,305 positive and 350,366 negative. There are 97,858,975,630 positive-negative pairs. **These pairs are not independent observations**: one positive row participates in 350,366 comparisons. The large pair count gives fine arithmetic resolution, not a 98-billion-example effective sample size.

Eight synthetic tests passed on October 2. Tests compare pair placements and AUC with a brute-force pair matrix and sklearn, including tied/discrete/continuous scores, all-tied and identical scores, perfect/reversed rankings, paired covariance, reversed comparisons, row permutations and malformed input. Test command:

```powershell
.venv/Scripts/python.exe -m unittest discover -s scripts -p test_analysis_auc_uncertainty_v1.py -v
```

Executed diagnostic command:

```powershell
.venv/Scripts/python.exe scripts/analysis_auc_uncertainty_v1.py
```

The complete immutable record is [paired_auc.json](../artifacts/research_pass_v1/validation/paired_auc.json), including input/script/test hashes, exact invocation, dependency versions, all member weights/configurations, metric checks and fold-level comparisons. CPU numerical threads are limited to two. Exact structural-component variances answer the registered precision question without repeated full-data sorting. A subsequent independent bootstrap cross-check was executed as recorded below; it still conditions on the same selected predictions and cannot repair adaptive selection.

## Fixed comparisons and uncertainty

The difference is always incumbent AUC minus comparator AUC. Removing a family renormalizes the remaining original weights to one without optimizing them. The simple alternative gives one third each to the complete frozen v2 blend, v3 probability-feature RealMLP and v3 probability-feature XGBoost. It does not assign equal weight to each of the 15 individual models.

| Comparator | AUC difference | Paired SE | Descriptive conditional 95% interval |
|---|---:|---:|---:|
| Frozen v2 | +0.0000437816 | 0.0000125446 | [+0.0000191946, +0.0000683686] |
| Without RealMLP | +0.0002688131 | 0.0000300341 | [+0.0002099472, +0.0003276789] |
| Without CatBoost | +0.0000340430 | 0.0000091348 | [+0.0000161392, +0.0000519469] |
| Without LightGBM | +0.0000000219 | 0.0000024953 | [-0.0000048689, +0.0000049127] |
| Without XGBoost | +0.0000657398 | 0.0000179753 | [+0.0000305089, +0.0001009707] |
| Equal three blocks | +0.0000324200 | 0.0000104526 | [+0.0000119334, +0.0000529067] |

Every interval above is **descriptive, conditional on already fitted and selected scores under an independent-row approximation**. They omit overlapping-fold training dependence, model-training variance, the entire adaptive recipe/weight search, and train-test shift. They are neither post-selection confidence intervals nor new release gates. The comparisons themselves were fixed before these diagnostic scores, but the incumbent and ingredients were selected previously using the same labels. Bonferroni adjustment for these six comparisons would not account for that older adaptive search.

The frozen v2 comparison improves all three folds: +0.0000345959, +0.0000544948, +0.0000431850. Its pooled and macro-fold gains agree closely (+0.0000437816 and +0.0000440919). That supports internal consistency; three overlapping-training folds do not constitute three independent replications. LightGBM removal is mixed: incumbent-minus-removal is +0.0000037325, -0.0000010811, -0.0000023764.

Absolute AUC uncertainty is a poor shortcut for comparing nearly identical blends. The incumbent's fixed-score AUC SE is about 0.00025782, but the incumbent-v2 **paired** difference SE is about 0.00001254 because their placement covariance is very high (structural correlation 0.998817). Ignoring covariance would discard useful paired information. Conversely, pretending rows are independent after adaptive model selection understates the scope of uncertainty.

The equal-three-block blend has slightly better Brier score (0.0592751 versus 0.0592886) and essentially unchanged log loss (0.2217265 versus 0.2217273), yet worse AUC. Optimizing probability calibration, squared error and ranking are different objectives. There is no contradiction, and the competition objective should not silently switch to whichever diagnostic looks favorable.

### Bootstrap cross-check and independent root review

The separate [analysis_auc_bootstrap_v1.py](../scripts/analysis_auc_bootstrap_v1.py) executed 200 paired bootstrap replicates with seed 20261002, sampling positive and negative rows separately with replacement while preserving their observed class counts. All seven score vectors use exactly the same resampled row multiplicities in each replicate. Original model fits, folds, predictions and weights remain fixed. Cached score orderings plus integer row multiplicities compute exact weighted AUC with half-credit ties. Four additional synthetic tests passed against explicit row replication and sklearn weighted AUC, including ties, zero-count rows, all ties, class count preservation and deterministic seeds.

```powershell
.venv/Scripts/python.exe -m unittest discover -s scripts -p test_analysis_auc_bootstrap_v1.py -v
.venv/Scripts/python.exe scripts/analysis_auc_bootstrap_v1.py
```

[paired_bootstrap.json](../artifacts/research_pass_v1/validation/paired_bootstrap.json) preserves all 200 differences for each comparison, script/input hashes and the hash of the unchanged original DeLong diagnostic. Bootstrap-to-placement SE ratios range from 0.9384 to 1.0150. Incumbent-versus-v2 bootstrap SE is 0.0000118495, with a descriptive percentile interval [0.0000219515, 0.0000673389]. The LightGBM-removal interval again straddles zero: [-0.0000050484, 0.0000047206]. These agree on the practical scale of uncertainty; 200 replicates have appreciable Monte Carlo error, particularly in tail quantiles. This is an implementation/scale cross-check, not independent new evidence for model superiority.

Independent source review of root's `scripts/analysis_blend_diagnostics_v1.py` found correct key alignment, convex mixtures, seed-preserving/family removals, 49 fixed block combinations, and held-fold exclusion from its conditional weight objective. Its warnings correctly distinguish that operation from end-to-end nesting. A low-impact malformed-input gap was reported: `make_units` initially allowed duplicate names within one declared group, although all three registered groups were unique. Root's four family-removal pooled and fold AUCs and equal-three-block pooled AUC match this agent's independently computed results exactly. The three held-fold changes after conditional weight selection are +0.0000010890, -0.0000012622 and -0.0000035707; these microscopic changes do not establish a useful precision reweighting of the incumbent.

## Derivation and interpretation

For positive row i and negative row j, let h(t) = 1[t > 0] + 0.5 * 1[t = 0]. Empirical AUC is the mean of h(s_i - s_j). Define V10_i as its mean over negatives for positive row i, and V01_j as its mean over positives for negative row j. For score A minus score B, use the aligned differences d10 and d01. The paired structural-component variance is:

```text
Var_hat(AUC_A - AUC_B) = sample_var(d10) / n_positive
                       + sample_var(d01) / n_negative.
```

The script obtains exact placements by sorting each class's scores and using left/right binary searches for half-credit ties. It computes variance of differences directly instead of subtracting nearly equal marginal variances. This is the correlated-AUC structural-component approach associated with [DeLong, DeLong and Clarke-Pearson (1988), DOI 10.2307/2531595](https://pubmed.ncbi.nlm.nih.gov/3203132/). The original abstract was inspected; the implementation is independently derived and brute-force tested, not copied from third-party code.

For fixed anchor a and candidate b, each pair's blended margin is d_a + alpha * (d_b - d_a). AUC changes only when a positive-negative pair crosses zero. Its empirical weight surface is therefore a step function, often with many nearly equivalent coefficients. A decimal-heavy optimum is not evidence of decimal-heavy certainty. Correlated model columns make individual weights unstable even if blended predictions barely change.

Weight magnitude also depends on component score scale. Replacing b with c + d*b, d > 0, preserves that component's standalone AUC, but changes the blend's relative ranking influence to alpha*d / (1-alpha+alpha*d); the constant c cancels in pair differences. Hence a 20% weight is neither a probability that the model is correct nor a model-importance percentage. Compare grouped predictions, removals, broad weight regions and out-of-sample behavior.

## Source-linked design issues and smallest justified changes

| Evidence in current source | Interpretation | Smallest justified next change |
|---|---|---|
| `scripts/train.py:70` through `:94` selects stopping on inner rows, then refits and scores outer rows. `scripts/common.py:161` through `:185` fits preprocessing to the supplied training partition. | Good row-level exclusion for a prespecified base recipe. It does not isolate recipe or ensemble selection from the outer labels. | Preserve these partitions and input contracts. Avoid calling the eventual blended OOF score an independent test. |
| `scripts/common.py:186` and `:220` plus `scripts/realmlp_categorical.py:283` and `:310` make the refit epoch count its new normalized schedule horizon. | Inner-selected prefix and outer-refit learning-rate/dropout/decay paths differ. This is not outer-label leakage, but it changes what is tested. | New IDs only: separate schedule horizon from executed/selected epochs, log actual schedules, then compare matched pipelines. Do not alter completed duration artifacts. |
| `scripts/third_pass_release.py:101` through `:132`, called at `:383`, selects weights and reports their score on identical OOF labels. | Deliberate adaptive development selection. The 1e-5/2e-5 gates constrain search; they do not create independent assessment. | Keep coarse constraints; register a full outer evaluation of the weighting procedure before future expensive comparisons. |
| `scripts/third_pass_release.py:88` through `:98` requires a longer-duration model to match the control standalone. | Standalone improvement and marginal ensemble value are different hypotheses. A worse standalone model can in principle improve a mixture. | Preserve failed prior gates. A future ensemble-focused duration experiment must predeclare its own ensemble hypothesis and budget before fitting, with independent evaluation of the full selection procedure. |
| `scripts/third_pass_release.py:433` through `:439` uses median selected lengths for full-data production refits. | OOF evaluates fold-specific training sets and lengths; production uses more rows and a median length. Native/raw verification proves inference consistency, not equal generalization. | Record this deployment difference and evaluate that refit rule inside a future outer procedure. |

The stale `anchored_search` docstring says “at most three” although the registered current policy limits additions to two. This is a documentation defect, not a changed executed search bound; the loop uses the policy. No production source was edited in this pass.

## What would constitute stronger evaluation

For a prospective outer fold H, every model used to build meta-training predictions must be trained entirely in T = development minus H. Generate an additional layer of cross-fitted base predictions within T, with stopping and preprocessing fitted inside those training subsets. Fit the small nonnegative weighting rule on those inner OOF predictions. Refit its selected base recipes on T using the predeclared length/schedule rule. Only then predict H, once, with the complete selected pipeline. Repeat for the frozen outer folds and compare paired outer predictions.

The saved OOF matrix cannot substitute for those inner fits: predictions on other folds were produced by models that included H in their base training. Existing full-data native models are also unsuitable for this assessment. Frozen original-only feature banks may be reused because they do not fit synthetic satisfaction labels, subject to their recorded source-overlap limitations.

Even that future correctly separated procedure remains conditioned on a short list and scientific choices developed using these same rows. A new random seed or repartition does not erase the historical exposure. A genuinely fresh claim for the complete adaptively developed campaign needs new independent labels, or rerunning the complete selection strategy inside outer splits without carrying label-informed choices across them. The original audit has already been exposed and is unavailable for this purpose. State the distinction rather than making an independence claim the data cannot support.

## Fully specified proposed three-arm mechanism test

This is a future protocol proposal, not executed training or new authorization. Its purpose is to separate schedule compression from additional updates. It deliberately uses fixed epoch endpoints rather than selecting each arm's best epoch, since unequal checkpoint searches would otherwise obscure the primary comparison.

| Arm | Immutable schedule horizon | Executed epochs | Scientific role |
|---|---:|---:|---|
| A | 4 | 4 | Short compressed-schedule control |
| B | 16 | 4 | Same short update budget on a longer schedule |
| C | 16 | 16 | Longer prefix of exactly the same trajectory as B |

Use the recorded `v2_realmlp_cat_raw_aux` feature/architecture/optimizer configuration, with eight members, seed 20261005, batch size 256, learning rate .053, weight decay .015 and zero label smoothing. Keep installed optimizer semantics unchanged, including the newly identified factor behavior. Use new immutable IDs and source/config hashes. Arm A is a fixed-four-epoch control; it is not the historical control that selected three epochs, and it cannot retroactively change the e500 run's failed gate.

For each of the same three frozen outer folds, run two stages:

1. **Monitoring stage:** reproduce the existing stratified 90%/10% inner-fit/probe split using seed + fold. Fit preprocessing only on inner-fit rows. Train A to four epochs and C to sixteen; save B from C's epoch-four state. Log minibatch-weighted training loss, fixed training-probe AUC, disjoint inner-probe AUC/log loss, schedule scalars, actual optimizer updates and gradient/parameter summaries. All endpoints are fixed in advance; the probe does not select checkpoints, change length, tune parameters or trigger quality-based early termination.
2. **Assessment stage:** refit preprocessing on all permitted outer-training rows. Start A and C afresh with matching initialization seeds and data order. Export B from the C trajectory at epoch four, then complete C through epoch sixteen. Score outer rows only for the three frozen endpoints after all three folds complete. Never score outer labels per epoch. Preserve native checkpoints and train-only transforms with keyed OOF predictions.

Within each stage, B and C share exact initialization, preprocessing, data order and first-four-update history. A versus B holds four epochs fixed and changes the normalized schedule horizon. C versus B holds the entire first-four-epoch path fixed and tests the additional twelve epochs. C versus A measures their combined practical difference. Across inner versus outer stages, equal epochs still correspond to different optimizer update counts because row counts differ; log this instead of claiming bitwise inner-to-outer trajectory identity.

This requires twelve actual fit calls and 120 epoch passes across three outer folds: A4 + C16 in both stages, with B obtained from the C checkpoint. Because inner fits contain 90% of outer-training rows, this is about 114 full-outer-equivalent epochs before preprocessing, validation and checkpoint costs. A provisional bound is 90 minutes of fitting and 120 minutes overall on one local GPU, with no cloud or second seed. Verify throughput in the authorized bounded smoke before launch; if the full estimate does not fit, reduce the scope before registration rather than truncating a quality-dependent subset of folds.

The only optional scientific question that should be incorporated **before this protocol is frozen** is fixed-weight marginal value. If included, fix alpha=.10 for every arm: M_arm=.90*incumbent+.10*p_arm. This is three predetermined mixtures, not another weight search. A strict advancement rule for a later confirmation study is that M_C improves pooled and mean-fold AUC by at least .00001 relative to the incumbent and both M_A and M_B, with no paired fold regression exceeding .00002 in any comparison. Report all outcomes and failed criteria. This fixed alpha can miss gains at another weight; failure does not prove that every blend weight is useless. No current release should change from this mechanism study alone.

The monitored trajectories explain whether training loss and held-inner discrimination diverge. They cannot prove a universal model ceiling. If C provides no stable endpoint or fixed-mixture benefit, stop duration escalation. If it does, freeze that candidate and evaluate a small weighting procedure with base fitting and meta selection contained in outer training partitions. Do not reopen the 49-point diagnostic grid or copy its maximizing coefficient into production.

### Additional independent mechanism reviews

Root's strict-rank geometry implementation correctly obtains opposite-class ranking changes by subtracting positive-only and negative-only Kendall discordances from all-row discordances, then uses the signed AUC win-count difference to divide changes into repairs and breaks. It rejects ties and checks integer rounding/parity. At the registered sample size its arithmetic tolerance is comfortably above float-rounding error and far below one pair; the brute-force tests cover 60 random small permutations. It supplies explanatory pair counts, not binomial confidence evidence.

The isolated CPU prototype uses the actual installed `TabNNModule` and library stopping callback. Its H12/stop4 model matches the H12 full run's captured epoch-four state, probabilities and schedule trace exactly on the generated toy data; H4/stop4 differs. This verifies a schedule-preserving cutoff mechanism on one CPU setup. It does not validate the production wrapper with eight members, CUDA/export behavior, changed training row counts, optimizer-state continuation, or competition quality. Those limitations must remain explicit before a future adapter is used.

## Executed synthetic nesting contract and second protocol review

The new generated-data [nesting proof](../artifacts/research_pass_v1/validation/nesting/proof.json) and [dataflow diagram/checklist](../artifacts/research_pass_v1/validation/nesting/dataflow.md) make the fixed-OOF limitation executable. All ten tests in `scripts/test_research_nested_blend_contract_v1.py` passed. The test uses 36 generated rows, a fitted label/feature-mean learner, and an additive meta correction. No competition files or AUC computation are used.

Ordinary OOF excludes each predicted row's own label, yet every one of the 24 proposed meta-training rows has a feature generator fitted using the 12 proposed outer-held-out rows. Perturbing only those held-out labels changes the old meta features and final held-out predictions, while leaving their original base predictions unchanged. Proper regeneration inside outer training makes the corresponding learned objects and predictions exactly invariant. Separate feature perturbation tests verify train-only preprocessing while allowing legitimate changes in predictions on changed queries. The toy changes establish a dependency path, not the magnitude or direction of real selection bias.

Independent review of `research/next_epoch_blend_experiment.md` agrees with its fixed H4/E4, H16/E4 and H16/E16 contrasts and 12-fit/120-epoch accounting. The following implementation conditions must be resolved before its proposed dispatch:

1. **Passive monitoring must not restore the best epoch.** Current `scripts/realmlp_categorical.py:310` sets `use_best_mean_epoch_for_cv=True`, and line 319 reads the library-selected stop epoch when validation is supplied. Installed `nn_creator.py:198` attaches `ModelCheckpointCallback` under default `use_best_epoch=True`; `lightning_callbacks.py:125` restores its saved best model at fit end. Thus `use_early_stopping=False` alone does not guarantee a fixed endpoint. A new adapter must explicitly disable best-epoch restoration or use passive monitoring outside that path, then prove with a synthetic nonmonotonic validation trace that epoch sixteen remains exported even if epoch four scores higher. The existing no-validation prefix prototype does not establish this case.
2. **Capturing B must not change live C.** The current terminal exporter moves the estimator to CPU at `scripts/realmlp_categorical.py:324`, wraps live modules at line 329, rewrites forward operations at line 345 and traces them at line 351. Invoking that path on the live C model mid-training is unsafe as an implementation strategy. Clone B's state/preprocessor into a separate export instance. Verify C with and without the capture/export operation produces the same subsequent synthetic update and endpoint under controlled seeds. Include live parameter/buffer state, optimizer state, RNG and schedule counters in that noninterference check. This is distinct from claiming arbitrary exact resume.
3. **Separate the two seeds explicitly.** Existing `scripts/common.py:221` passes model seed 20261005 unchanged across folds; `scripts/train.py:85` uses 20261005 + fold only for the inner partition. Record `model_seed=20261005` and `inner_split_seed=20261005+fold`; an ambiguous fold-seed field could accidentally add a third experimental factor.
4. **Record the loss being compared.** The mean minibatch training objective can include regularization and ensemble/member reductions, whereas monitor log loss scores averaged probabilities. Keep both definitions visible. A falling training objective and falling monitor AUC are observed trajectories, not alone a proof of one specific generalization mechanism.

The new adapter, eight-member export path, monitor-only validation and checkpoint noninterference remain implementation work for a future authorized training campaign. No changes to the current training wrapper or library were made by this agent.

## Contrast-specific uncertainty across the frozen 49-point grid

The registered `configs/research_grid_uncertainty_v1.json` extension reuses exactly the existing 49 weights. It does not refine weights, score new candidates, fit models or change the release. [analysis_grid_uncertainty_v1.py](../scripts/analysis_grid_uncertainty_v1.py) independently repeats the accepted-artifact/hash/key checks, reads the two accepted v3 development OOF blocks, and reproduces all 49 existing pooled AUCs. The sign is grid minus incumbent throughout. For the 49-column class-specific placement-difference matrices D10 and D01, the estimated joint covariance is `sample_cov(D10)/n_positive + sample_cov(D01)/n_negative`. Every column uses the same keyed rows and the same incumbent. This directly estimates each reweighting contrast's uncertainty; the larger incumbent-versus-v2 SE must not be substituted for it.

Eleven generated-data tests passed. They compare each covariance diagonal to the separately implemented paired-AUC variance; check brute pair accounting with ties, class-row and column permutations, sign swaps, identical and zero contrasts; verify seeded Gaussian repeatability and covariance-factor reconstruction; and reject substantive non-PSD/asymmetric matrices. Exactly identical covariance columns receive identical perturbations so eigensolver roundoff cannot decide a theoretical tie. Zero-SE columns are excluded from standardization and receive zero perturbation.

```powershell
.venv/Scripts/python.exe -m unittest discover -s scripts -p test_analysis_grid_uncertainty_v1.py -v
.venv/Scripts/python.exe scripts/analysis_grid_uncertainty_v1.py
```

The immutable [grid_uncertainty.json](../artifacts/research_pass_v1/validation/grid_uncertainty.json), SHA256 `c24f3793d9f859f481894f3674aec0c4776f836c4624a2a1647d6eac19e05894`, stores all 49 contrasts and bands, full covariance/eigenvalues, input/source hashes, versions and exact invocation. Planned extra arrays are 0.5893 GiB under the 2 GiB bound, with two numerical threads; peak RSS was not measured. Its covariance is positive definite numerically: smallest eigenvalue 1.37385e-15, versus the roundoff tolerance 1.20937e-21. **No eigenvalues were clipped and no diagonal jitter was added.** The v2 corner's SE is 1.25446204267e-5, matching the previous independent paired calculation.

Using 20,000 zero-mean Gaussian perturbations with seed 20261003, the 95th percentile of maximum absolute standardized perturbation is 2.494817829. The simultaneous interval uses that multiplier for each of the 49 SEs. This is an approximate conditional fixed-grid band under independent-row sampling. It does not absorb historical adaptation, training randomness, overlapping CV training sets or distribution shift, and it is not exact finite-sample coverage.

| Existing grid weights (v2/NN/XGB) | AUC delta vs incumbent | Paired SE | Pointwise conditional 95% interval | Simultaneous fixed-grid conditional 95% interval |
|---|---:|---:|---|---|
| 65/20/15, observed grid maximum | +0.0000015830 | 0.0000034999 | [-0.0000052767, +0.0000084427] | [-0.0000071487, +0.0000103146] |
| 60/20/20 | +0.0000008818 | 0.0000019848 | [-0.0000030083, +0.0000047719] | [-0.0000040699, +0.0000058335] |
| 70/15/15 | +0.0000004525 | 0.0000026443 | [-0.0000047302, +0.0000056352] | [-0.0000061445, +0.0000070495] |

No grid point has a positive lower endpoint even in its pointwise conditional interval. This does not prove every true gain is zero, but the observed maximum is not a precise new optimum supported by this diagnostic. Its simultaneous upper endpoint is slightly above 0.00001, so this analysis also does not rule out a gain of that operational size. Under the same Gaussian perturbations, the winner among the 49 registered points is 65/20/15 in 25.88% of draws, 60/20/20 in 18.84%, and 70/15/15 in 15.23%. **These are perturbation winner frequencies, not posterior probabilities of optimality.** The incumbent's 64/16/20 weights are not in the 49-point candidate grid and therefore are not included in that winner calculation. Ties within 1e-12 use the base protocol's closest-to-incumbent rule, then smaller NN and XGB weight; none occurred in this execution.

The appropriate consequence is to preserve the incumbent and treat nearby coefficients as a broad region for future independently contained selection. It is not to choose whichever point has the highest perturbation frequency, enlarge the grid, or use the new diagnostic as a release gate.

## Primary evidence refreshed October 2

- [Cawley and Talbot (2010), JMLR 11:2079–2107](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf), methods and evaluation discussion inspected, especially section 5.2 and conclusions. Optimizing a noisy selection criterion can overfit it; evaluating choices made before resampling can retain information from the evaluation rows. Their examples support the procedural warning, not a numerical estimate of bias in this project.
- [LeDell, Petersen and van der Laan (2015), DOI 10.1214/15-EJS1035](https://escholarship.org/content/qt2j87q257/qt2j87q257.pdf), sections 2–4 including Theorem 4.1 and practical construction inspected. The target is an average of fold-specific AUCs, with asymptotic influence-curve conditions. It is not automatically the pooled cross-fold AUC, full-data refit's AUC, or a post-selection campaign guarantee. The companion [cvAUC source](https://github.com/ledell/cvAUC) and `ci.cvAUC.R`, package 1.1.4 / Apache-2.0 metadata, were inspected for comparison; no package code was copied or installed.
- [Bengio and Grandvalet (2004), JMLR 5:1089–1105](https://jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf), abstract and statement of the variance-estimation issue inspected. No universal unbiased variance estimator exists for K-fold CV under arbitrary distributions. This does not imply uncertainty estimation is impossible; it rules out treating three fold scores as an assumption-free independent sample.
- [scikit-learn nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), official explanation/example inspected. Its implementation nests parameter selection within each outer training partition. No paper benchmark or external method was experimentally reproduced here.
