# Method evidence and overnight experiment priorities

Inspected 2026-10-01, America/Indianapolis. Research status: primary papers' methods/evaluation and selected source files inspected; no paper result reproduced by this research pass. Main-agent dataset audit reports 699,635 training rows, 299,844 test rows, 21 features, four string categorical columns, thirteen ordinal ratings, and ROC AUC on a Boolean target. Treat the competition contract and actual data audit as authoritative; this note does not establish external-data permission.

## Decision

Use CatBoost GPU, LightGBM CPU, and XGBoost GPU as the first portfolio. Test one small TabM candidate for complementary errors after the first valid tree artifacts exist. Choose configurations by shared local validation, not paper rankings or notebook titles. Defer RealMLP unless TabM is infeasible or measured remaining time is ample. A 700k-row dataset and approximately nine hours make full literature-style neural tuning inappropriate.

The main agent's planned contract is suitable: reserve 10% untouched audit data, use three shared stratified folds on the 90% development partition, choose stopping iterations with an inner split, then refit on the outer training portion for a fixed number of iterations. Final OOF predictions never use their own targets for fitting or stopping. All pipelines fit preprocessing inside the relevant training partition. Investigate duplicated/group-related observations before assuming rows are independent.

## Five primary papers reviewed

| Paper and inspected edition | Methods and evaluation evidence | Relevant limitation and decision |
|---|---|---|
| Prokhorenkova, Gusev, Vorobev, Dorogush, Gulin. **CatBoost: unbiased boosting with categorical features**, NeurIPS 2018. [Conference PDF, sections 3-6](https://proceedings.neurips.cc/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf). Related arXiv record [1706.09516v5, 2019-01-20](https://arxiv.org/abs/1706.09516v5). | Ordered target statistics exclude the current example's target; ordered boosting addresses a separate training prediction shift. The paper compares nine datasets using an 80/20 training-testing division, log loss and classification error, and reports gains against then-current boosting implementations. | This is not evidence that current CatBoost must win this competition. Plain and Ordered boosting are distinct modes. Use native categorical handling, train-fold-only data, and a practical GPU Plain baseline before spending time on Ordered. |
| Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu. **LightGBM: A Highly Efficient Gradient Boosting Decision Tree**, NeurIPS 2017. [Conference PDF, sections 3-5](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf). | GOSS preserves high-gradient rows and reweights sampled low-gradient rows; EFB bundles nearly exclusive sparse features. Five large benchmarks include classification AUC and ranking NDCG; timing uses a Linux server with 16 active CPU threads and 256 GB RAM. | Its large sparse benchmark speedups do not predict this small-width Windows workload. Use histogram GBDT CPU as a cheap complementary family. GOSS is a separate experiment, not an assumed improvement. |
| Holzmüller, Grinsztajn, Steinwart. **Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data**, NeurIPS 2024. [Conference PDF, sections 2-5](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf); [arXiv 2407.04491v3, 2025-01-15, correction inspected](https://arxiv.org/html/2407.04491v3). | RealMLP combines robust preprocessing, numerical embeddings, initialization and scheduled regularization. Defaults are tuned on 118 datasets, evaluated on 90 disjoint datasets and another benchmark. Training uses ten 60/20/20 splits; its default neural schedule runs all 256 epochs and restores the best epoch. Trying a portfolio of defaults competes well with 50-trial single-family searches. | Data sizes extend to 500k, below this full dataset. Numeric-missing rows were excluded in the benchmark. The corrected version documents an XGBoost categorical-encoding bug affecting reported comparisons. For AUC the authors advise disabling label smoothing. Defer this heavier dependency and training recipe tonight. |
| Gorishniy, Kotelnikov, Babenko. **TabM: Advancing tabular deep learning with parameter-efficient ensembling**, ICLR 2025. [Conference PDF, sections 3-4 and appendix D](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf). [arXiv 2410.24210v3, 2025-02-18](https://arxiv.org/abs/2410.24210v3). | MLP submodels share weight matrices with small per-member adapters and make multiple predictions per row. Evaluation covers 46 datasets, including nine domain-aware splits, up to 723k training rows; metrics include accuracy, ROC AUC and RMSE. Most configurations receive 50-100 tuning trials, followed by usually 15 training seeds, mainly on A100 hardware. | Strong reported neural results support a local test, not the full reported tuning cost. Use a smaller TabM-mini candidate, measure two epochs, and promote only on useful OOF blend gain. Default k=32 is not 32 independently trained models. |
| Cawley and Talbot. **On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation**, JMLR 11, 2079-2107, July 2010. [Paper, sections 3-5](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf). | Synthetic examples and thirteen benchmark datasets with repeated partitions show that optimizing finite-sample selection criteria can overfit, and common evaluation shortcuts can favor the selection procedure itself. Evaluation must include model selection inside the fitted procedure. | Demonstrations use kernel models and small datasets, not airline prediction, but the statistical concern applies to tuning ensemble weights. Keep a one-time untouched audit assessment and report development OOF as selection evidence, not an unbiased final estimate. |

These are author-reported benchmark results. Neither private Kaggle performance nor local generalization has been reproduced at the time of this note.

## Repositories, versions and licensing

Versions were queried directly from official GitHub releases/tag commits and PyPI JSON endpoints on the inspection date. This is discovery metadata, not proof of installation. Prefer the already smoke-tested environment over changing versions for novelty.

| Candidate | Inspected release/commit | License and requirements | Reuse verdict |
|---|---|---|---|
| [CatBoost](https://github.com/catboost/catboost/releases/tag/v1.2.10) | 1.2.10, `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`, 2026-02-19 release | [Apache-2.0](https://github.com/catboost/catboost/blob/v1.2.10/LICENSE). Windows 64-bit CPython wheels include CUDA support and statically linked CUDA libraries; actual RTX 5070 fit must still be tested. [Install requirements](https://catboost.ai/docs/en/concepts/python-installation). | Integrate native library; inspect serialization/inference in smoke test. |
| [LightGBM](https://github.com/lightgbm-org/LightGBM/releases/tag/v4.7.0) | 4.7.0, `8f7036f03627054d5a54a6f965b13f4b9ff2cb63`, 2026-07-18 | [MIT](https://github.com/lightgbm-org/LightGBM/blob/v4.7.0/LICENSE). PyPI requires Python >=3.10. [4.7 installation documentation](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html) says CUDA implementation is unsupported on Windows; OpenCL is a different implementation. | CPU, initially eight threads. Avoid a GPU source-build detour. |
| [XGBoost](https://github.com/dmlc/xgboost/releases/tag/v3.4.2) | GitHub 3.4.2, `fdf0888bedddbd444d72994d845c59b3ca182c5b`; PyPI observed 3.4.1 | [Apache-2.0](https://github.com/dmlc/xgboost/blob/v3.4.2/LICENSE). [PyPI metadata](https://pypi.org/pypi/xgboost/json) requires Python >=3.12. Current [GPU docs](https://xgboost.readthedocs.io/en/stable/gpu/) require CUDA >=12.9; Windows wheel includes GPU support. | Use installed 3.4.1, `tree_method='hist', device='cuda'`; prove a fit before unattended work. QuantileDMatrix reduces working memory. |
| [TabM](https://github.com/yandex-research/tabm/releases/tag/v0.0.3) | 0.0.3, `a507095893d784c5702059d737ddfbd1299c41dd`, 2025-08-14 | [Apache-2.0](https://github.com/yandex-research/tabm/blob/v0.0.3/LICENSE). [pyproject](https://github.com/yandex-research/tabm/blob/v0.0.3/pyproject.toml): Python >=3.9, torch >=1.12,<3; rtdl_num_embeddings >=0.0.12,<0.1; typing_extensions >=4.6,<5. | Best bounded neural candidate. Inspected `tabm.py`, manifest, README and example source. No test/workflow paths were present in the inspected release tree. No pretrained model or dataset is required. |
| [pytabkit/RealMLP](https://github.com/dholzmueller/pytabkit/releases/tag/v1.7.3) | 1.7.3, `c126ea51187c5080b91f28d352481dbd3b2194b0`, 2026-01-06 | [Apache-2.0](https://github.com/dholzmueller/pytabkit/blob/v1.7.3/LICENSE). [PyPI requirements](https://pypi.org/pypi/pytabkit/json) include Python >=3.9, torch>=2, Lightning>=2, torchmetrics>=1.2.1, numpy>=1.25, pandas>=2 and sklearn>=1.3. | Source interface permits CPU/CUDA/MPS; tests and testing workflow are present, not executed here. Requires numeric imputation. Larger integration surface than direct TabM. |

Only selected code paths were reviewed, not complete repositories or issue histories. No upstream test suites were run. All five licenses were read from their release-tag LICENSE contents through the official GitHub API. Code licenses do not establish permission to use separate external datasets or pretrained weights.

PyTorch compatibility is a runtime gate. [PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/) introduced Blackwell support and CUDA 12.8 wheels. [2.12 guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/) deprecates CUDA 12.8 wheels and recommends CUDA 13.0+ for newer GPUs, requiring Windows driver >=580.88. The operations note reports driver 616.56; nevertheless inspect the installed torch CUDA build and execute a forward/backward CUDA kernel. Do not install CPU-only torch accidentally. A reported CUDA device alone does not prove usable kernels.

## Important source-code finding

The [TabM v0.0.3 example notebook source](https://github.com/yandex-research/tabm/blob/v0.0.3/example.ipynb) places gradient clipping before `loss.backward()` in its displayed loop. With gradients cleared first, this is ineffective; AMP unscaling also occurs too early. Implement the correct order: zero gradients, forward/loss, backward, unscale if applicable, clip, optimizer step, scaler update. This is a source-derived finding, not an executed upstream reproduction.

The same example correctly illustrates per-member losses and probability-space averaging. The model emits `[batch, k, classes]`; flatten the first two axes and repeat each label k times for cross-entropy. At inference, apply softmax per member, then average across k. Do not average logits before softmax. With one binary logit, use BCEWithLogitsLoss per member and average sigmoids. The example optimizer is AdamW with learning rate 0.002 and weight decay 0.0003; these are reasonable initial hypotheses, not optimal parameters established here.

## Bounded experiments and stopping decisions

1. **Execution baseline:** one small tree fit, saved model reload, held-out prediction, test prediction and submission-format readback. Measure seconds, system RSS and VRAM before scaling.
2. **Two-family coverage:** CatBoost depth 7-8, learning rate around 0.05-0.08, 1,500-3,000 maximum iterations with inner early stopping; LightGBM 31-63 leaves, 40-100 minimum samples per leaf, learning rate around 0.04-0.06, 2,000-4,000 upper bound. These ranges are our budget hypotheses. Obtain complete compatible OOF/test predictions before broad search.
3. **Tree diversity:** one XGBoost histogram depth 6-8 candidate with moderate row subsampling. Keep ordinal ratings numeric initially; a later CatBoost rating-as-categorical variant is a separate representation experiment. Do not treat zero ratings as missing until the data description establishes that meaning.
4. **Neural diversity:** TabM-mini, width 128 or 256, k=16 or32, two or three blocks, batch1024, AdamW. Time two epochs before selecting the maximum epochs and allocation. Start without expensive numerical embeddings; a piecewise-linear version is one possible follow-up. A candidate gets full OOF only if fold timing fits the deadline and predictions can add stable blend value. Low early-epoch AUC alone is insufficient evidence to kill a neural run that is still improving.
5. **One targeted feature variant:** only after residual/slice evidence. Candidate domain summaries include count/mean of ratings and departure-arrival delay differences. Fit any learned statistics inside folds. Avoid broad interaction expansion, high-cardinality target encoding, pseudo-labeling and unknown-provenance public OOF tonight.
6. **Seeds versus methods:** prefer a complementary family before extra seeds of the same configuration. Once the best few configurations are established, one additional seed can reduce prediction variance if wall time remains.

Use one GPU process and respect the operations plan's RAM reserve. The machine has 16 GB installed RAM but substantially less free RAM, so concurrent independent full-data copies can be the real bottleneck. No architecture here has a verified local runtime until its smoke timings are recorded. Set per-job bounds from measured completion costs and stop launching exploration before the documented 06:15 EDT cutoff.

For neural preprocessing, the simplest low-cardinality contract is fold-local median-imputed and scaled numerical/rating features plus one-hot string categoricals with an explicit unknown policy. An alternative passes category indices/cardinalities separately to TabM; casting arbitrary category codes to scaled continuous values is not equivalent. Fit numerical bin boundaries, quantile transforms, imputers and category dictionaries exclusively on the training partition. Save the actual transformation with the checkpoint. Refitting the outer model requires refitting preprocessing on the outer training rows.

## Ensemble selection and release criteria

- Save OOF and test predictions with explicit row IDs, fold IDs, class mapping, configuration/data hashes and model provenance. Join by validated IDs with one-to-one cardinality; never infer alignment from matching array lengths alone.
- Report pooled OOF ROC AUC and each fold's AUC separately. Inspect paired improvements and residual/slice behavior, including travel type, class and missing-delay status. Raw prediction correlation is insufficient justification for a blend.
- Compare best single model and equal-weight probability averaging first. Fit a nonnegative sum-to-one blend with coarse weights, for example a 0.05 simplex grid for at most four candidates or a short constrained greedy search. Keep the search budget fixed and retain the simpler blend if gains are tiny or unstable. These OOF scores are selection scores.
- ROC AUC permits monotone transformations, but rank averaging changes cross-model scaling and is only a bounded alternative experiment. Use the exact same transform for OOF, audit and test. Calibration cannot improve a single model's AUC when strictly monotone; avoid spending the night on calibration-only changes.
- Freeze candidate set, transformations and weights before opening the 10% audit labels. Assess the frozen blend once and record paired bootstrap uncertainty if time allows. Conditional-on-model bootstrap captures row-sampling uncertainty, not training-seed uncertainty. If audit findings trigger further tuning or selection, disclose that the audit became development data; do not continue calling it untouched.
- Save the frozen blend manifest and component checkpoints. A later full-data refit may use fixed training lengths derived solely from development; report that its audit evidence comes from the earlier held-out recipe, not from full-data fit predictions.
- Require finite probabilities in [0,1], exact sample-submission columns/ID order/row count, explicit positive class, independent blend recomputation, saved-model reload inference, and final file hashes. Preserve the last validated fallback until all replacement checks pass.

## Evidence limits

The competition-specific community research is a separate agent deliverable. No Kaggle score is claimed here. The recommendations are budgeted hypotheses justified by primary papers and current implementation evidence; final promotion depends on local paired evaluation. There is no reason to claim a rank, private-score improvement, or paper reproduction from this research pass.

## Executed decision update during overnight setup

Measured tree runtimes were much shorter than the initial conservative budget: the three-fold LightGBM baseline completed in about54seconds and CatBoost baseline in about144seconds, including independent stopping and outer refits. This left sufficient budget for the competition-specific route hypotheses, two TabM representations and a short RealMLP candidate. The initial suggestion to defer RealMLP and high-cardinality target statistics was therefore revised using observed local runtime and inspected competition-source evidence.

Route target statistics are implemented through shuffled, label-independent five-fold cross-fitting inside each training partition, persisted separately from native boosters. Six regressions verify held-out label invariance, unknown fallback, key precision/order, and save/load. A real-data LightGBM integration smoke reproduced its saved encoded model exactly. Actual feature/teacher candidates are scored on the same outer folds, not ranked by the public notebook's claims.

RealMLP uses official PyTabKit1.7.3 and a local native TorchScript inference wrapper. Its learned median/robust/L2 preprocessing was instrumented with an extreme validation-only shift and fitted only the training300rows. Full eight-member512/256/128 CPU/CUDA smoke, batch sizes1/17/257, fixed refit, and serialization checks passed. Maximum measured CPU/CUDA prediction difference was7.45e-7, within predeclared float32 tolerance. Export debugging found a shared-storage ordering issue: calling the public estimator after moving the exported graph could move shared weights back to CPU. Caching reference predictions first fixed it. The earlier hypothesis about unregistered constant weights was rejected, and unnecessary parameter conversion was removed. No competition-quality claim follows from these synthetic smoke results.

## Implemented neural smoke verification

After the main agent installed PyTorch 2.14.1+cu130, this agent installed `tabm==0.0.3` and `rtdl-num-embeddings==0.0.12` into the project's existing `.venv`, without changing torch. Implemented only `scripts/neural.py`, using the parent-owned preprocessing contract.

Executed `.venv\Scripts\python.exe scripts/neural.py --smoke --device cuda` successfully on the RTX 5070. It exercised two-epoch binary training, same-device repeated-seed agreement, native weights save/load agreement at absolute tolerance 1e-6, probability sums, empty/single/partial inference batches, fixed-epoch refit with no validation, and rejection of NaN input. The 193-row synthetic training had validation AUC 0.5961 then 0.7291, identically on the repeated run. These numbers verify updates and seed handling only; they are not competition performance, convergence evidence, or paper reproduction. Default training is float32 without AMP. Optional bfloat16 AMP is implemented but was not exercised by this smoke test.

Inference snapshots contain JSON metadata and native state dictionaries with hashes and restricted `weights_only=True` loading. Each completed improving epoch publishes weights before atomically replacing its metadata pointer; fixed-epoch refits checkpoint every epoch. `progress.json` records every completed epoch. Checkpoints intentionally omit optimizer/RNG continuation state and do not imply exact training resumption.

## Optional piecewise-linear embeddings and preprocessing review

Implemented an opt-in branch in `scripts/neural.py`, leaving the baseline architecture and artifact metadata unchanged when `num_embeddings` is absent or false. Configuration: `num_embeddings=true`, `n_bins=32`, `d_embedding=16`. The official [rtdl_num_embeddings 0.0.12 source](https://github.com/yandex-research/rtdl-num-embeddings/blob/7381c5a4ed01253fbef9ea8cfed137f907265552/package/rtdl_num_embeddings.py) and installed source were inspected at release commit `7381c5a4ed01253fbef9ea8cfed137f907265552`. The implementation uses `compute_bins` with unsupervised quantiles from X_train only, and `PiecewiseLinearEmbeddings(version='B', activation=False)`, which the TabM API explicitly supports. Version B adds a linear feature embedding and initializes the additional piecewise component to zero.

All nonconstant columns are embedded, including existing one-hot columns. Binary columns have one interval and therefore a learned linear embedding, adding some unnecessary width but preserving the simple existing matrix contract. Train-constant columns are removed only in this branch because the official bin estimator rejects constants. Exact float32 edges and input column indices are recorded in JSON, with numerical buffers also included in the native state dictionary. Loading reconstructs the architecture from these saved edges; it never estimates bins on inference data. This approach is not claimed identical to the paper's numeric/categorical separation.

Executed `.venv\Scripts\python.exe scripts/neural.py --smoke --device cpu --num-embeddings` successfully. In addition to the standard trainer checks, it verifies constant-column exclusion, binary-column support, and exclusion of a validation-only extreme from fitted bin edges. Re-ran the baseline CPU smoke successfully. After the main agent confirmed the teacher had finished, also executed `.venv\Scripts\python.exe scripts/neural.py --smoke --device cuda --num-embeddings` successfully; it completed in about six seconds including interpreter/import overhead. This branch has not yet been timed on 700k rows, and the two-epoch synthetic smoke establishes execution, not useful predictive performance.

Reviewed `common.py` without editing it. Its TabM transformation learns medians, means, standard deviations and categorical vocabularies on the supplied training partition; this matches the split contract. It preserves ordinal ratings numerically, handles unseen categories as an all-zero block, and passes numerical features plus category one-hot blocks. Raw delays can be heavy-tailed, so their standardization may still leave large outliers; an existing `features=true` variant adds log1p delay features and is a smaller experiment than replacing the transform. If numerical missingness exists, explicit missing indicators are another hypothesis because median imputation discards that signal. Neither change has a verified gain here.

The proposed run's weight decay 0.01 is roughly 33 times the official example's 0.0003. This does not establish it is wrong, but a 0.0003 follow-up is a low-cost underfitting check. More than twelve epochs may be warranted if validation is still improving; choose using observed epoch duration and inner validation, without opening the audit. For a distinct PLE candidate, two blocks match the official embedding default and reduce compute, but that should be a separately recorded configuration rather than silently changing the baseline.

## Development-only review and four bounded follow-ups, 2026-10-02 UTC

This update reviews the first 18 completed candidates, their recorded configurations and neural inner-fit histories, then the newly completed native categorical TabM candidate. Sources are local `artifacts/runs/*/result.json`, neural inner `metadata.json`, the 18-candidate blend snapshot created at 02:59:58 UTC, and ID-aligned development `oof.parquet` files. All OOF IDs, labels and fold IDs were checked for identical alignment before paired calculations. No audit predictions, audit labels or test scores were evaluated. No training, GPU work or upstream benchmark reproduction was performed during this review. The primary-paper and repository evidence above is reused without a fresh research search.

The 18-candidate blend has development AUC **0.9610702667**; the strongest individual is `cat_route_teacher` at **0.9606576958**. `realmlp_te_teacher` scores **0.9606333241**, `cat_route_teacher_d8` **0.9606151824**, and `xgb_route_te_teacher` **0.9605210839**. These are selection scores, not independent generalization estimates.

### Prioritized configurations

| Priority | Additional experiment | Measurable purpose | Local time and memory expectation |
|---|---|---|---|
| 1 | `realmlp_te_teacher` with maximum epochs raised from 4 to **16**, keeping seed 20261005, TE, teacher, architecture, schedule, and `ls_eps=0` unchanged | Test whether the best neural complement benefits from a longer training schedule | Estimate **10-16 minutes**, based on the completed four-epoch pipeline's 241 seconds; peak model/activation memory should be similar because batch size and architecture do not change. This runtime is an extrapolation. |
| 2 | **CatBoost numerical categorical twins + teacher, depth 7**, retaining original numeric measurements and the existing tree parameters | Test native categorical statistics across ratings and exact numeric values, beyond the distance-only representation | Initially budget **15 minutes**, then revise from the first fold. The existing distance+teacher run took 126 seconds, but additional categorical-statistics combinations make a linear runtime or RAM extrapolation unreliable. |
| 3 | **Native categorical RealMLP + TE + teacher, four epochs**, matching the existing numerical RealMLP training recipe | Isolate the categorical representation while preserving the strongest existing feature set | Estimate **4-8 minutes**, subject to the first fold. High-cardinality learned embeddings avoid a full dense distance one-hot training matrix; additional preprocessing and model allocations still require measurement. |
| 4 | **One predetermined additional seed of the strongest contributing CatBoost recipe**, followed by an equal-weight probability average with its existing seed | Assess whether a small, fixed seed ensemble improves the selected recipe's development predictions | The existing distance+teacher recipe took about **2-3 minutes** for the full pipeline. A categorical-twin winner requires its newly measured runtime instead. Run seeds sequentially to avoid concurrent RAM/VRAM pressure. |

These are four bounded hypotheses, not claimed future gains. Under a strict four-configuration limit, prioritize the native categorical RealMLP **with TE** over a teacher-only categorical variant. Do not enlarge TabM or extend its epoch ceiling on the evidence below. Keep a valid blend available while additions run.

### RealMLP cap and schedule interpretation

Both numerical RealMLP variants selected epoch **4 of 4 on every fold**. This makes insufficient training a plausible hypothesis, but it does not establish underfitting: per-epoch RealMLP validation curves were not persisted, and each inner fit already performs approximately 5,900 optimizer updates at batch size 256. Raising `flat_anneal` to 16 epochs stretches the schedule, so the experiment tests a longer schedule, not continuation of the identical four-epoch optimization trajectory. The [RealMLP paper](https://arxiv.org/html/2407.04491v3) supports scheduled training and checkpoint selection; its benchmark's 256 epochs do not establish an appropriate competition budget.

The comparison `realmlp_teacher` -> `realmlp_te_teacher` changes both TE and label smoothing (`ls_eps=.01` -> `0`). Its observed AUC improvement therefore cannot be attributed solely to TE. The 16-epoch follow-up holds `ls_eps=0` fixed to remove that confound from the budget comparison.

### Paired blend contribution and categorical-statistics variance

The existing blend assigns `realmlp_te_teacher` weight 0.3360234375. Removing it and renormalizing the remaining weights, **without reoptimizing them**, loses **0.0001228599** pooled AUC. The losses on folds 0/1/2 are **0.0001247278 / 0.0001202549 / 0.0001262123**. This is unusually consistent conditional contribution within the current candidate set; it is not a statistical significance test or an unbiased evaluation of selected weights.

`cat_route_teacher` and `xgb_route_te_teacher` have probability correlation **0.9986948**, yet their fixed 50/50 average scores **0.9608388779**, improving over CatBoost by **0.0001811822**, with positive changes on all three folds. CatBoost plus `realmlp_te_teacher` at 50/50 scores **0.9609217322**, improving by **0.0002640364**, also positive on all folds. High raw correlation therefore does not eliminate useful complementarity.

Native CatBoost categorical statistics and manual five-fold cross-fitted TE both help, but cross-family results do not identify which encoding has lower training variance. The manual encoder has a train/inference count mismatch: a training row's statistics use four inner folds, while validation/inference statistics use all supplied training rows. This can alter shrinkage and variance even though own-target leakage is excluded. CatBoost uses a different statistics construction; its ordered target-statistics and ordered-boosting concepts are distinct, as described in the [CatBoost paper](https://proceedings.neurips.cc/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf). The local GPU recipe should not be assumed to reproduce the paper's Ordered boosting mode.

A label-free count audit using each OOF row's outer training partition found 289 development rows with unseen distance, 2,342 with 1-5 matches, 13,713 with 6-20, and 613,327 with more than 20. Thus only **2.60%** have at most 20 matches. CatBoost scored better than the two TE models in the 1-5 and 6-20 slices, but these selected slice results do not establish causality, and rare distances alone cannot explain the global differences. No within-slice score was used as an additional tuning target.

### Native TabM already shows early overfitting

`tabm_cat_teacher` completed in **139 seconds** with development AUC **0.9595284896**, selecting epochs **3 / 4 / 2**. Inner training continued to epochs 9/10/8 under patience 6. In all three folds, training loss kept falling while validation AUC worsened after the selected epoch. For example, fold 0 peaked at **0.9607505861** at epoch 3 and ended at **0.9592838193** at epoch 9. Additional epochs or wider networks are not justified by these curves.

Its errors are nevertheless complementary. Against the frozen 18-candidate development blend above, adding **5%** native TabM probability improves pooled AUC by **0.0000219500** and all three folds by **0.0000170319 / 0.0000440517 / 0.0000044971**. At 10%, the pooled gain is 0.0000296068 but the third fold is slightly negative; at 20%, two folds are negative. These three prespecified comparison weights are exploratory development evidence, not guaranteed private-score gains. The main controller may use its existing constrained blend procedure to decide membership.

### Fixed seed averaging contract

Choose the additional seed before running and average both seeds with equal weights; do not keep only the better seed or tune internal seed weights. Preserve the shared outer folds and manual TE split seed. The current `train.py` derives the inner stopping split from `run.seed + fold`, so changing the model seed also changes its stopping partition. This measures combined **training-and-stopping variability**, not isolated optimizer or boosting randomness. Existing depth-7/depth-8 comparisons change architecture and are not seed-variance estimates. Fixed seed values improve repeatability on the current setup but do not imply cross-platform or GPU bitwise determinism.

Judge the predetermined seed average and other additions using pooled development OOF AUC, paired fold changes, and constrained blend contribution. Keep the candidate/weight search bounded and freeze the final recipe before the one-time audit assessment. No guarantee of improvement follows from these recommendations.

## Bounded diagnosis of the native TabM PLE+TE pooled-AUC gap

Read-only CPU analysis of development OOF and saved inner/outer training records found substantial fold-dependent probability-scale drift in `tabm_cat_ple_te_teacher`. IDs, labels and fold assignments match the two comparison models exactly. No calibration was fitted, and no audit/test predictions or scores were read.

| Model | Pooled OOF AUC | Mean within-fold AUC | Mean-fold minus pooled | Pooled log loss | Pooled Brier |
|---|---:|---:|---:|---:|---:|
| Native TabM PLE+TE+teacher | 0.9593178620 | 0.9601000987 | 0.0007822367 | 0.22625922 | 0.06033931 |
| Native TabM+teacher | 0.9595284896 | 0.9595917394 | 0.0000632498 | 0.22644134 | 0.06036002 |
| RealMLP TE+teacher | 0.9606333241 | 0.9606441513 | 0.0000108272 | 0.22361544 | 0.05981047 |

Pooled AUC compares positives and negatives across different outer models as well as within one fold. Mean fold AUC omits those cross-model comparisons. Therefore equality is not expected in general, but the observed reversal is material: PLE+TE improves on plain native TabM within **every** fold by approximately 0.000585 / 0.000404 / 0.000536, while losing pooled AUC by 0.000211. The reversal is localized to cross-fold comparisons, not an aggregation or row-alignment error.

PLE+TE per-fold diagnostics:

| Statistic | Fold 0 | Fold 1 | Fold 2 |
|---|---:|---:|---:|
| Actual positive fraction | 0.44357309 | 0.44357044 | 0.44357521 |
| Mean predicted probability | 0.45502354 | 0.44586515 | 0.43860718 |
| Probability standard deviation | 0.44464480 | 0.43324063 | 0.42643881 |
| Mean logit | -0.21430 | -0.35231 | -0.50371 |
| Logit standard deviation | 3.35945 | 3.04480 | 2.90198 |
| Mean logit among positive rows | 3.02522 | 2.58762 | 2.29840 |
| Mean logit among negative rows | -2.79679 | -2.69594 | -2.73751 |
| Log loss | 0.23060551 | 0.22410147 | 0.22407066 |
| Brier score | 0.06129103 | 0.05986216 | 0.05986475 |

Logits were computed with probabilities clipped to [1e-7, 1-1e-7] for numerical safety. The positive-fraction difference is negligible, while mean predictions and confidence scales change substantially. By comparison, plain native TabM's fold mean predictions are 0.44462 / 0.43935 / 0.44483 and logit standard deviations 3.108 / 3.110 / 3.037; RealMLP's are 0.44302 / 0.44373 / 0.44482 and 3.063 / 3.051 / 3.058.

The PLE+TE cross-fold AUC using **fold-0 positives against fold-2 negatives is 0.96651940**, whereas the reverse pairing gives **0.94941378**. Their corresponding within-fold AUCs are 0.95965392 and 0.96046450. This directional asymmetry is direct evidence that differing output scales distort pooled rankings. It is consistent with calibration drift; these diagnostics alone do not identify its optimization or representation cause.

Crucially, PLE+TE selected **exactly two epochs in every fold**, and every outer refit ran exactly two epochs. Different selected epoch counts cannot explain this candidate's gap. All inner curves peaked at epoch 2 and then generally worsened through epoch 8 while training loss fell. Short training with the current constant learning rate, fold-specific fits/preprocessing, and the TE training/inference distribution mismatch are plausible contributors, not established causes. PLE, TE and seed differ from the plain categorical comparison, so that comparison does not isolate one contributor.

A single bounded calibration experiment is justified after the already prioritized runs if the candidate's blend contribution warrants it. A low-flexibility sigmoid calibration must be learned exclusively from predictions that are held out **inside each outer training partition**, then assessed on untouched outer OOF rows with exactly the same inference procedure. Either preserve an excluded calibration subset when fitting the final base model, or generate training-only cross-fitted predictions for the calibration fit. Do not fit fold-specific calibration on the existing outer OOF labels and report the adjusted pooled score as validation; do not overwrite raw artifacts. Inner-model calibration transferred to a full outer refit also changes the prediction-generating model and requires honest outer evaluation rather than assumed portability. Longer unmodified TabM training is not supported by the existing curves. No calibration improvement has been measured or promised.

## Macro-fold versus pooled-AUC blend selection, bounded comparison

Primary-source refresh on 2026-10-02 inspected sections 2-3 of [Airola et al., A comparison of AUC estimators in small-sample studies, PMLR 8](https://proceedings.mlr.press/v8/airola10a/airola10a.pdf) and the current [scikit-learn cross_val_predict documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html). Airola et al. explicitly distinguish pooled and averaged AUC, explaining that cross-fold positive-negative pairs can introduce bias because their scores come from different fitted functions. Their experiments use small, often imbalanced datasets, and also find higher variance for averaged estimates; they do not establish a universal winner or a bias magnitude for our large dataset. Scikit-learn warns that scoring concatenated cross-validation predictions need not measure generalization correctly, particularly when the metric does not decompose over individual samples. AUC is pairwise, so equal fold sizes alone do not resolve this issue. Attempts to inspect the complete LeDell et al. 2015 article encountered browser verification; this review does not rely on its methods beyond the accessible abstract/search excerpts.

### First-principles distinction

Let `g_k(x; w) = sum_m w_m f_{m,-k}(x)` be the candidate blend using each family's model trained without outer fold k. For a positive-negative pair, let H be 1 for correct ordering, 0.5 for a tie, and 0 otherwise. Define `A_kl(w)` using positive rows from fold k scored by g_k and negative rows from fold l scored by g_l. Then:

- **Macro-fold AUC** is `mean_k A_kk(w)`: both observations in every scored pair are unseen by the same fitted blend.
- **Pooled OOF AUC** is `sum_kl [n_k_positive * n_l_negative / (N_positive * N_negative)] * A_kl(w)`. With three essentially equal stratified folds, approximately **two thirds** of pair weight comes from k != l.

Thus pooled OOF combines genuine same-model rankings with rankings across different functions and probability scales. A fold-specific monotone transformation of the completed blend preserves each fold's AUC but can change pooled AUC. For an extreme illustration, a large fold-specific score offset can damage pooled ordering without changing any model's ranking quality. The local PLE+TE cross-fold asymmetry above demonstrates this mechanism in the saved predictions; no transformation is needed to recognize it.

For the narrower target of mean discrimination of a model-training procedure at the outer training size, **macro-fold AUC is more directly aligned with that target** under the observed score-scale drift. Equal weighting is reasonable here because folds contain 209,891 / 209,890 / 209,890 observations and virtually identical class proportions. This is a methodological recommendation, not a claim that macro optimization guarantees a better deployed fold ensemble. Arbitrarily replacing the metric because it raises a preferred model's score would be inappropriate; any protocol change should be recorded explicitly, with pooled AUC retained as a scale-stability diagnostic.

### Executed comparison with exactly 20 candidates

The comparison froze the queue prefix through `tabm_cat_ple_te_teacher`, excluding later experiments and incomplete second seeds. Candidate order was: `lgb_baseline`, `cat_baseline`, `xgb_baseline`, `lgb_route`, `cat_route`, `lgb_route_teacher`, `cat_route_teacher`, `lgb_route_teacher_63`, `xgb_teacher`, `xgb_route_te`, `lgb_route_teacher_127`, `xgb_route_te_teacher`, `cat_route_teacher_d8`, `xgb_features`, `realmlp_teacher`, `tabm_baseline`, `tabm_ple_teacher`, `realmlp_te_teacher`, `tabm_cat_teacher`, `tabm_cat_ple_te_teacher`.

Only development OOF files were loaded. All 629,671 IDs, labels and fold IDs were checked for identical alignment. The float64 prediction matrix occupied 100,747,360 bytes. OMP/OpenBLAS/MKL thread counts were set to one, and no GPU operations were used. Each objective used the existing greedy nonnegative convex search unchanged: start at its best single candidate; alphas `[.5, .25, .1, .05, .02]`; at most three accepted additions at each alpha; require an improvement greater than 0.000002. Both searches evaluated 260 objective values, taking about 31 seconds for macro and 29 seconds for pooled on the active machine.

**Both objectives returned exactly the same weights and the same sequence of additions.** The pooled reproduction matched the published `current.json` snapshot from **2026-10-02T03:10:46.066474+00:00** with maximum absolute weight difference 0.0. This establishes agreement for the executed bounded heuristic, not equality of the objectives everywhere or a globally optimal solution.

| Search objective | Mean fold AUC | Pooled AUC | Fold 0 AUC | Fold 1 AUC | Fold 2 AUC |
|---|---:|---:|---:|---:|---:|
| Pooled OOF | 0.9611099479 | 0.9611031706 | 0.9606818606 | 0.9610201351 | 0.9616278480 |
| Macro outer-fold | 0.9611099479 | 0.9611031706 | 0.9606818606 | 0.9610201351 | 0.9616278480 |

Nonzero weights for both searches:

| Candidate | Weight |
|---|---:|
| `lgb_route` | 0.0475 |
| `cat_route` | 0.1370671875 |
| `cat_route_teacher` | 0.1542005859375 |
| `lgb_route_teacher_127` | 0.081225 |
| `xgb_route_te_teacher` | 0.09025 |
| `cat_route_teacher_d8` | 0.05 |
| `realmlp_te_teacher` | 0.3369568359374999 |
| `tabm_cat_teacher` | 0.10280039062499999 |

All other 12 candidates, including native PLE+TE TabM, receive zero. Recognizing its pooled-AUC distortion therefore does **not** imply it needs a positive blend weight. No production blend code, current blend manifest, prediction artifacts or final model selection were changed by this comparison.

### Why the fold-averaged deployment still needs untouched assessment

OOF routing predicts a development row with only `g_k`, the blend trained without that row's fold. Deployment instead predicts every new row with `G(x; w) = mean_k g_k(x; w)`. Its AUC depends on `H(mean_k [g_k(x_positive) - g_k(x_negative)])`. That is neither the mean of the within-fold H values nor the pooled OOF construction that independently assigns the two rows to different fitted functions. Ensemble averaging may cancel offsets, reduce variance or change rankings; neither development objective directly measures those effects on a common untouched population.

Consequently, the macro comparison is a useful correction of the evaluation target, not a substitute for the audit. Freeze the candidate set, shared weights, transformations and the exact fold-averaging inference recipe **before** evaluating the held-out audit once. That audit scores the actual common fold-averaged function on rows unseen by every component. Searching candidate recipes or weights using either OOF objective still induces selection optimism, and the three folds are not independent replicates because their training sets overlap. Neither the small pooled/macro difference of the selected blend nor the number of pairwise comparisons supplies a valid confidence interval by itself. No audit/test predictions or labels were consulted in this comparison, and no final model was selected.
