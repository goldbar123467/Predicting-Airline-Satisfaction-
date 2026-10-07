# S6E10 competition evidence and local experiment priorities

Sources first inspected during the overnight setup on **October 1, 2026, America/Indianapolis (EDT)**, equivalent to October 2 UTC. Exact source slug: `playground-series-s6e10`. Public claims, visible remote outputs, and locally executed results are distinguished below. Dated sections record the evidence and experiment state at each checkpoint.

## Competition contract, authoritative host pages

| Item | Verified fact | Source and inspection depth |
|---|---|---|
| Task | Predicting Airline Satisfaction, Season 6 Episode 10; binary probability prediction | [Overview](https://www.kaggle.com/competitions/playground-series-s6e10), full rendered browser page |
| Metric | ROC AUC between target and predicted probability; higher is better | [Evaluation](https://www.kaggle.com/competitions/playground-series-s6e10/overview/evaluation), evaluation section of Overview read |
| Submission | CSV header `id,satisfaction`, one probability for every test ID; example starts 699635 | Same host Evaluation section |
| Data | `train.csv`, `test.csv`, `sample_submission.csv`; source distributions similar but unequal to original | [Data](https://www.kaggle.com/competitions/playground-series-s6e10/data), full rendered text |
| Original | [arseniyshutko/binary-aviation-satisfaction-129k](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k) explicitly linked by host | Data page, plus official dataset download/metadata |
| Data rights | Competition data CC BY 4.0; original linked dataset metadata says CC0-1.0 | Data and Rules pages; saved original metadata |
| Timeline | Starts October 1; entry/team merger/final deadline October 31, 2026 23:59 UTC | Overview Timeline; user overnight delivery is separate |
| Limits | Max 10 submissions/day, up to 2 final selections, team max 3 | [Rules](https://www.kaggle.com/competitions/playground-series-s6e10/rules), competition-specific sections 2.1–2.2 read |
| External resources | External data/models allowed if publicly accessible or meet host reasonableness criterion; open-source code must use OSI-approved license | Rules 2.6 and 3.6 read |
| Other restrictions | No hand labeling of test/validation records; no private sharing across teams; private leaderboard determines placement | Rules sections 3.4–3.7 read |
| Split/runtime | Public/private test fractions not published in inspected pages. Prediction-file competition; no special notebook runtime/internet constraint identified | Do not import quotas/split ratios from other episodes |

All four visible discussion topics were read, including all five substantive replies across the two threads containing comments. No host announcement or pinned clarification was visible in this four-topic index. This is an access-time observation, not a promise there will be no new clarification.

Web search/fetch could not open exact S6E10 pages and frequently returned other seasons. In-app browser succeeded. Kaggle CLI 2.2.4 initially lacked credentials; main agent then provided an existing credential location, used only through process-local environment. No rules were accepted, predictions submitted, comments posted, or remote notebooks executed by this agent.

## Discussion ledger

| Source / author / posted UTC | Inspected claim and evidence | Assessment / experiment |
|---|---|---|
| [744890: service ratings](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/744890), Mohan Krishna Thalla, 2026-10-01 09:06:34 | Whole post + 3 comments. Reports 292 arrival missing values; asks about service interactions. Reply suggests zero flags, digital-service and comfort means; author proposes strict OOF ablations | Plausible hypotheses, not demonstrated gains. Tree splits already can isolate zero; no need to assert continuous trees assume monotonically worse zero values |
| [744809: purported hidden signals](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/744809), giannhsstamelias, 2026-10-01 02:08:30 | Whole post + 2 replies. Author calls Cleanliness p=.0376 evidence of structural shift. paddykb notes 21 tests/multiple correction, invalid MCAR conclusion, missing true adversarial classifier, Isolation Forest bug, imputation before split. Author acknowledges revision v6; reports OOF .958604 / LB .95808 | Do not rely on original statistical claims. No exact train/test overlap does not prove no leakage. Flight-distance units in source metadata are miles; discussion's per-100-km wording is not established |
| [744830: early GPU baseline](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/744830), Rugved Bane, 2026-10-01 05:26:20 | Full post, 0 replies; 5-fold LGB/XGB starter, type/class/boarding/entertainment important | Broad baseline support, no verified score or settings in post |
| [744973: XGB tuning](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/744973), giannhsstamelias, 2026-10-01 17:41:14 | Full post, 0 replies; notebook link only | No additional independent result |

Official topic JSON saved for 744809/744890. Notebook index with author and `lastRunTime` saved in `downloaded_sources/notebook_index.json`.

## Notebook source and output ledger

Downloaded source is untrusted reference material, inspected statically, not executed. Downloaded `.ipynb` files contain **zero saved outputs**, so reported markdown tables are author claims unless separately marked rendered output below. Exact source hashes are in `downloaded_sources/source_hashes.json`. Latest-source pulls are not inherently version-pinned.

### Main high-signal source: route features and original-data teacher

[goodpjw2008, Route-ID FE + OG-Model Stack](https://www.kaggle.com/code/goodpjw2008/s6e10-route-id-fe-og-model-stack-lb-0-96108), last run from API 2026-10-01 04:27:31 UTC; source+metadata under `route_og`. Full training, features, inference, stack and claims inspected. No artifact reuse.

- Author reports raw XGB OOF .958763, route block .960362, plus original-only prediction .961078, 13-member stack .961459; LB .96108. The teacher step also changes learning rate, so .00072 is not a clean single-factor effect. Scores are author-reported and not local evidence.
- Fold recipe: `StratifiedKFold(5, shuffle=True, random_state=42)`. Six target-encoding keys: Flight Distance, Age, Distance×Age, Distance×Class, Distance×Travel Type, Age×Class×Travel Type×Customer Type. Training encodings inner `KFold(5, shuffle=True, random_state=0)`, smoothing 20. Validation encodings use outer training rows only.
- Route profile: means of each other feature by exact Flight Distance; count of rows per distance. Four category values are arbitrary integer-coded before profile means, a representation choice that warrants ablation. Source computes label-free counts/profiles on train+test; this is transductive preprocessing, not fold-fitted preprocessing. Our primary assessment should fit these maps on training partitions or explicitly declare any alternative protocol.
- Original-only XGB: 600 trees, lr .05, depth 8, subsample .8, colsample .5, seed 0, device CUDA when available. Public source uses `teejmahal20/airline-passenger-satisfaction` train+test, drops extra `Inflight service`, maps `satisfied` to 1. Official linked source already has the exact competition columns and bool target, so is simpler for our run.
- Main XGB: upper bound 20,000 trees, lr .02, depth8, subsample .8, colsample .5, min_child_weight5, lambda1, max_bin256, hist, native four categoricals, AUC early stopping400. LGB: 127 leaves, lr .02, min_child_samples50, subsample .8, colsample .5, lambda1. CatBoost baseline: depth8, lr .05, l2=3, border_count254, logloss early stopping400.
- Negative author findings: appending original rows .958763→.958629; rating aggregates .958763→.958520; excessive rating/route crosses harm; more seeds little gain. These justify experiment priority, not unconditional bans.

### Latest refinement, with browser execution evidence

[goodpjw2008, CatBoost CTR + GLM-Margin XGB](https://www.kaggle.com/code/goodpjw2008/s6e10-catboost-ctr-glm-margin-xgb-lb-0-96127), API latest run timestamp 2026-10-02 00:11:03 UTC; downloaded source under `cat_ctr`. Full code/markdown inspected. Browser **Version 1 completed in 3h38m51s on GPU T4×2**, output3 files. **Version 2 still Running** in inspected version history, with +115 lines. The latest downloaded source appears to include v2's added illustration; no claim it corresponds byte-for-byte to completed v1. Attempt to pull `/1` source returned HTTP403; recorded access limitation, not bypassed. Browser footer verifies **Apache 2.0** license.

Author's local tables claim CV .961612 / LB .96127. Separately, **rendered executed v1 output** showed ordinary XGB .961078 (not GLM-margin XGB), LGB .961072, CatBoost .960907, raw RealMLP .960777, engineered flat_anneal RealMLP .961194, CatBoost CTR depth6 .961143, cross-validated OOF-meta stack **.961597**, simple mean logits **.961550**. These are visible remote run outputs, not reproduced locally. Original-only XGB printed .95514 against synthetic train, LGB .95494. Distances had 3521 unique values over train+test. Competition shapes printed `(699635,23)` and `(299844,22)`.

Practical additions:

- Native CatBoost variant: numerical categorical twins + service ratings categorical, retain route profiles/original scores, omit manual TE. Depth6/l2=10/lr=.05/seed11 or depth7/l2=10/lr=.04/seed21. Completed v1's 5-fold depth6 variant took1797s, compared to117s for first XGB: prioritize after cheaper main feature tests.
- PyTabKit/RealMLP uses numeric categorical twins. Public recipe: n_ens8, batch256, hidden512/256/128, SiLU, lr .053, wd .015–.0236, dropout .05, embedding5, max_one_hot_cat_size18, label smoothing .01. Prefer **flat_anneal, 3–4 epochs**.
- Correction: 6-epoch lin_cos_log_15 model scores .961420 with checkpoint selected on scored fold, .961014 when checkpoint chosen on5% internal training. Flat_anneal model changes only about .00002–.00004 in author's analogous comparison. Keep our audit set outside checkpoint selection.
- Their “nested CV” actually refits logistic weights over already-created OOF predictions using another 5-fold split seed7. Base models are not regenerated inside those meta folds. This reduces direct meta-training overfit but is **not fully nested end-to-end assessment**. Other OOF rows may carry the scored fold's labels through their base models. Reserve untouched audit data for the final stack.
- Their statement that AUC depends on rank does not mathematically establish logit averaging as superior to probability averaging. Both are candidates; select with paired validation.
- Authors report original-source datasets identical after dropping extra column, but our pipeline uses official source directly rather than relying on this equivalence claim. Their teacher is not automatically leakage-free just because it sees no synthetic labels: exact source/synthetic duplication must be audited (done below).

### Independent/related recipes

| Source and inspection | Evidence and limitation | Next experiment |
|---|---|---|
| [yekenot / Vladimir Demidov, RealMLP PyTabKit](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit), API lastRun2026-10-02 00:42:43, full source `yekenot` | Latest code already uses flat_anneal3epochs, wd.015, n_ens8, 5fold seed42. Numeric categorical twins, counts, Class×Travel×Gender category, inner sklearn TargetEncoder. No saved outputs in pull; exact version not pinned. Category/count dictionaries fitted on all training rows before fold splitting | Reimplement minimum viable numeric-twin recipe with fold-local maps; benchmark complementarity to engineered trees |
| [arhancanli12, generator fingerprints](https://www.kaggle.com/code/arhancanli12/s6e10-generator-fingerprints-lgbm-cv-0-9610), code/claims inspected, `fingerprints` | Author claims LGB raw .958898→counts/original means/digits .960250→GPT2 token keys .960364→TE .960547→teacher .960769→127leaves/lr.02 .961029. Reports 17 original/train exact matches (our audit independently confirms17). Calling generator GPT2 is inference, not host confirmation | Separate secondary feature view only after main route+teacher variant; tokens' marginal effect small and uncertain |
| [megayak, Six Models One Honest Stack](https://www.kaggle.com/code/megayak/s6e10-six-models-one-honest-stack), full short source `megayak` | Trains no base models; reads public six-model OOF library. 5fold seed42, author meta-CV .961410. Ancestry credits yekenot/arhancanli; not an independent model family. No public predictions downloaded for blending | Do not substitute public artifact blending for user's own runs; reuse lessons only |
| [megayak, initial stack](https://www.kaggle.com/code/megayak/s6e10-honest-oof-stack), downloaded `megayak_initial` | Saved as reference; less deeply inspected than sources above | No execution/reproduction claim |
| [evgendvorkin, LightGBM 5fold](https://www.kaggle.com/code/evgendvorkin/lightgbm-cv-5-folds-0-96031), full source `lightgbm` | Seed123, up to2000trees/lr.02/depth8/lambda2=5; triple nested TE smoothing auto/10/100, original-value means, digits, frequency, numeric twins. Source loads extra visualization package; no reason to reuse that dependency. Title .96031 is an author claim | Stronger feature bundle but harder to attribute; defer until simpler controlled ablations |
| [thisray, noise measured](https://www.kaggle.com/code/thisray/s6e10-is-your-0-0001-gain-real-noise-measured), methods and claims inspected, `noise` | Paired rescoring of fixed OOF predictions; 20% public is an assumption, also examines50%. Retraining variance and fold-ensemble/test differences not measured. Small .0001 gaps can reverse; a constant CV-LB gap doesn't verify validation. Adversarial .499 is author report, not proof of no shift | Paired bootstrap/AUC deltas on our frozen rows; favor robust improvements over LB chasing |

Unverified licenses of secondary notebook snapshots are not treated as blanket code reuse authorization. No third-party notebook code has been run. Public source attribution is preserved here; fresh local implementation should retain relevant recipe attribution.

## Ranked overnight experiments

1. Preserve raw CatBoost/XGB/LGB baseline and frozen data/split IDs immediately. Compare pooled AUC on the same validation rows; keep a separate audit set.
2. Add exact Flight Distance representation: first category twin or fold-local count/profile, then nested target encoding. This is the largest repeated source-supported hypothesis.
3. Train original-only fixed XGB teacher on the cleaned source below; append its probability or clipped logit. It must never fit on synthetic labels. Compare against step2 with identical main hyperparameters to remove the public-source confound.
4. Own RealMLP numeric-twin member, flat_anneal3–4epochs, internal checkpoint selection. Size batch/ensemble to actual GPU smoke measurement.
5. CatBoost native categorical statistics variant and a second LGB feature view only if time remains; stop inferior hypotheses on one common development fold before spending5folds.
6. Restrained probability/logit blending from verified own OOF predictions; fit low-dimensional weights on development predictions and assess once on untouched audit rows. A plateau is acceptable. Keep a valid fallback and time for complete inference/schema/hash checks.

Do not spend the overnight budget on broad transformer/GPT2 training, 20fold CV, noisy-label deletion, excessive seeds, or dozens of hyperparameter trials before feature hypotheses are tested. Public negative reports support deprioritization, but they do not prove those ideas universally fail.

## Locally executed original-source audit and preparation

`scripts/prepare_original.py` uses `data/test.csv` header as canonical21feature contract, never reads synthetic target values, preserves all numeric values and missingness, maps original False/True to int8 0/1, and emits `data/original.parquet`.

- Source archive official Kaggle download:129,880rows/22columns; CC0-1.0 metadata stored alongside archive. Exact source CSV SHA256 `2ad76465f6115f93d16121891a6b1235acb8007a4ab065cae0dc46b60ad116e3`.
- No original duplicate feature profiles and no conflicting-label groups. Exact all-feature matching found17original rows in synthetic train and4in synthetic test; no profile overlapped both. Removed all21 using label-free anti-joins.
- Output129,859rows; 73,451negative /56,408positive; Arrival Delay393missing retained. No fitted statistics, imputation or teacher training.
- Output parquet SHA256 `633165a332541e41c7de41cfe2395128eb2e52bc39eb78a79c979b4d2de475b6`; full hashes, dtypes, counts and versions in `research/original_audit.json`.
- Verification executed: synthetic duplicate/conflict/NA/train/test-overlap regression; Python byte compilation; full script run; exact parquet readback equality; repeated anti-joins show zero surviving overlap; unique profiles and both classes verified. Python3.12.14/pandas3.0.6/numpy2.5.3.
- Limitation: exact exclusion does not eliminate approximate similarity or shared original ancestry. Source domain descriptions are not all reliable (dataset text describes delay columns as ratings despite actual values); model features should follow actual schema/ranges, not that prose.

Reproduction after archive/data download: `.venv\Scripts\python.exe scripts\prepare_original.py`.

## Original-only teacher executed locally at 22:21 EDT

`scripts/teacher.py` trained exactly600XGBoost trees on the129,859cleaned original rows, seed0, depth8, lr.05, subsample.8, colsample.5, min_child_weight5, reg_lambda5, CUDA hist,4CPUthreads. Source-only preprocessing vocabulary; no synthetic labels were read and no synthetic score or tuning was performed. Main controller explicitly authorized this run.

The overlap exclusion uses all synthetic training/test feature profiles, including held-out feature rows, solely to remove exact original-source matches before fitting. This is a declared label-free exclusion step, not a claim that external-data selection is independent of test features. All learned teacher preprocessing and model parameters use only the retained original rows. The evidence establishes no direct synthetic-target fitting and no surviving exact feature-profile overlap under the canonical schema; it does not establish independence from the synthetic generator, remove near-duplicate/shared ancestry, or prove absence of every possible leakage mechanism. No synthetic teacher-performance claim is made by the local teacher manifest.

Saved `artifacts/teacher/model.ubj`, `transform.json`, `model_metadata.json`, `manifest.json`, and `data/teacher_predictions.parquet` with999,479unique keyed IDs covering every synthetic train/test row. Output SHA256 `8dd064e96f500e900e4fef3a4bed41b1342d93ee7802646e320d42b1bf4dfbfb`; native model SHA256 `3f20a9b549d333c68a9eb0f9071842b8c8de245744c5b808d9e1eb2e083d696f`.

Input hashes and exact anti-overlap assertions were rerun. First attempt's native roundtrip exposed CPU-vs-CUDA inference differences of at most3e-7 on two probe rows; export gate correctly withheld publication. Explicit CPU inference on both instances isolated serialization from device arithmetic and passed the unchanged rtol1e-6/atol1e-7 criterion. Final successful run11.66seconds, bounds/finite/coverage/order checks,769-row native reload probe, and full parquet readback passed. CPU inference is recorded separately from CUDA training in the manifest. No further GPU work is active in this research agent.

## RealMLP implementation and categorical export audit, 2026-10-02 03:12 UTC

### Exact notebook representation and library behavior

Re-inspected the saved full source of [yekenot's RealMLP notebook](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit). It retains all original features and duplicates **all 17 original numeric columns** as categories: Age, Flight Distance, Inflight wifi service, Departure/Arrival time convenient, Ease of Online booking, Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service, Baggage handling, Checkin service, Cleanliness, Departure Delay in Minutes, and Arrival Delay in Minutes. The four original categoricals are Gender, Customer Type, Type of Travel, and Class. Raw rating zero remains a real category.

It also adds Class × Type of Travel × Gender as a categorical interaction, five count features for the four original categoricals plus Leg room service, and a fold-local target encoding of the interaction. That is 44 features before the target encoding, 45 afterward. Its category/count dictionaries are initially fitted outside the folds. Our implementation instead requires fold-local preprocessing and does not claim to reproduce that transductive detail or every engineered feature automatically.

Inspected official [PyTabKit source](https://github.com/dholzmueller/pytabkit) from the [1.7.3 wheel](https://pypi.org/project/pytabkit/1.7.3/), Apache-2.0. The sklearn estimator concatenates explicit training and inner-validation matrices before calling `ToDictDatasetConverter.fit_transform`. With `cat_indicator`, its OrdinalEncoder therefore learns vocabulary from both matrices; it shifts known codes by one and reserves native code zero for unseen values. Passing numerical category codes without a category indicator treats them as continuous. Separately, the learned median, robust scale, and L2 normalization are fitted through `NNCreator` on the training indices only. An instrumented execution with 300 training rows and 100 validation rows shifted by +10,000 confirmed every numerical-statistic fitting call saw only the 300 training rows.

### Implemented contracts

- `scripts/realmlp.py`: finite float32 numeric matrices, official estimator optimization, four-epoch flat_anneal recipe by default, eight ensemble members, hidden sizes 512/256/128, lr .053, wd .015, batch 256, label smoothing .01. This initial version lacks categorical twins.
- `scripts/realmlp_categorical.py`: the same fit/predict/save/load shape, plus a required `categorical_indices` list. Non-categorical matrix columns remain continuous. Category columns must contain unscaled integer values represented exactly in float32. The caller may supply raw numeric twins and training-fold codes for strings. The module fits its own sorted vocabulary from **X_train only**, excludes sentinel -1, derives cardinalities, maps known values to 1..K, and maps -1 or any unseen value to native code zero. No supplied validation/test category can expand a vocabulary. Native zero is an unknown category code; the embedding behavior follows the library, rather than assuming every unknown embedding vector is zero.
- A small estimator/interface adapter splits the public converter's all-numeric matrix into continuous tensors and frozen categorical tensors. Training, losses, optimization, model architecture, and internal normalization still run through the official library. The adapter and export internals are explicitly version-gated to PyTabKit 1.7.3. Category vocabulary fitting needs no labels.
- Both modules use only the supplied inner validation to select epochs. Refit with no validation uses `val_fraction=0`, one CV component, no hidden refit, and the requested fixed epoch count. Outer-fold/audit labels are not read by either module.

### Export findings and executed verification

The library's documented estimator persistence uses pickle; our artifacts instead contain a native TorchScript graph and JSON metadata with hashes, vocabulary, class order [0,1], configuration, versions, and code provenance. TorchScript is deprecated in installed PyTorch 2.14.1 but still supported and tested here. These are locally trusted executable model graphs, not optimizer-resume checkpoints or a guarantee of portability across future releases.

Two export failures were caught and repaired before promotion. First, the numeric wrapper's post-export reference prediction moved shared estimator/graph weights back to CPU; caching reference predictions before moving the returned wrapper resolved this. Second, the library's categorical inference creates one-hot tensors and embedding offsets with explicit device arguments, which tracing fixed to CPU. Binary one-hot also embedded a nonportable tensor constant. The categorical exporter uses equivalent inference-only device-inheriting allocations/index operations on the fitted instances. It caches predictions from the **unmodified** native model before replacing these operations and checks the exported graph against those references. No training or optimizer algorithm is replaced.

Executed checks:

- Numeric full-architecture CUDA smoke: eight members, 512/256/128, inner epoch selection, fixed refit, disk reload, and CPU/GPU batches of 1,17,257. Native/export maximum absolute difference 1.79e-7; returned/reloaded GPU difference 2.98e-7; CPU/GPU difference 7.45e-7. Tiny-fixture verification took 6.16 seconds and allocated 138.6 MiB peak CUDA memory.
- Categorical CPU smoke: validation-only category 555 excluded from vocabulary, sentinel/unseen equivalence, empty input, partial batches, fixed refit, and native reload. The five category columns exercise cardinalities 1,2,3,4,101, covering empty, constant, signed binary, ordinary one-hot, and embedding paths.
- Independent CPU comparison against unmodified public `RealMLP_TD_Classifier(...).fit(..., cat_indicator=...)`, with all validation categories already present in training: predictions matched within 1.19e-7. This checks that the adapter preserves native categorical semantics.
- `scripts/test_categorical_pipeline.py --family realmlp_cat`: passed on actual competition rows with categorical twins, teacher features, nested target encoding, unknown strings, and native reload. Its initial failure exposed the binary one-hot constant omitted by the first toy fixture; the final smoke now covers that case.
- Final categorical CUDA smoke, 03:12 UTC: eight members, full hidden dimensions, two selected epochs and a two-epoch fixed refit. Native-before-replacement/export difference 1.19e-7; GPU disk reload difference 3.58e-7; CPU/GPU difference 5.96e-7. All 1/17/257-row, unknown, class-order, probability, and vocabulary checks passed. Tiny-fixture verification took 10.61 seconds, peak CUDA allocation 120.25 MiB. The process exited successfully and released the GPU.

Float32 export checks use rtol1e-5/atol2e-6; cross-device checks use rtol2e-5/atol2e-6. Tiny smoke timings and synthetic-fixture metrics do **not** establish competition quality or full-data memory requirements. No public submission was made.

Verified implementation hashes at this checkpoint: `realmlp.py` SHA256 `8da0d62f1555c91ffe91befe7855f553f58fe6781db0e7249a7e7041130e03d8`; `realmlp_categorical.py` SHA256 `70c8b78f77c8f53df7f94e7fc4c92d3e47592973dd942fae20ed3beb2bbe4184`.

### Current local development evidence, before categorical full-data training

Read saved result files, rather than inferring quality from the smoke tests. Both numeric runs use split hash `4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c` and selected four epochs in each of three development folds.

| Run | Pooled development OOF AUC | Fold AUCs | Recorded run seconds |
|---|---:|---|---:|
| `realmlp_teacher` | .958571011970 | .957922941128 / .958689143376 / .959151649189 | 207.234 |
| `realmlp_te_teacher` | .960633324085 | .960179256160 / .960586745584 / .961166452219 | 241.344 |

The latter adds nested target/count encodings **and** changes label smoothing from .01 to zero. The improvement is positive on all three folds, but these results do not isolate the contribution of target encoding from label smoothing. Future categorical comparison should hold those settings constant. Categorical full-data OOF quality was not yet measured at this checkpoint. No audit labels or audit scores were inspected for this research step.

### Subsequent smoke-lineage correction

The root agent subsequently found that the initial shared categorical integration test sampled 1,024 competition rows before filtering the audit split. It included 102 audit rows, 88 in temporary fitting and 14 in temporary validation. This agent ran that shared test once to completion before the correction; it did not run the root's earlier 500-row common integration smoke. Those temporary smoke models were discarded and never entered the candidate or blend pipeline. Root recorded the affected IDs in `data/audit_smoke_exclusions.json`, corrected the test to sample development rows only, and reserved a secondary unexposed audit. All standalone RealMLP, native-estimator comparison, preprocessing instrumentation, and CPU/CUDA export tests run by this agent used generated inputs/labels. Original-teacher fitting used original-source labels only. The claim above about no audit scores for the saved-result comparison remains true; the initial integration smoke's label exposure must nevertheless be accounted for separately.

## GLM-margin source audit and ordinary-XGB comparison, 2026-10-02 03:41 UTC

Decision: defer a new GLM-margin family. Recommend one controlled ordinary-XGB route-profile ablation instead. No implementation, GPU run, competition-label smoke, or new metric evaluation was performed during this audit.

### Source depth and exact recipe

Re-inspected cells 7, 9, 11, 14, 16, 23-27 of the downloaded [goodpjw2008 notebook](https://www.kaggle.com/code/goodpjw2008/s6e10-catboost-ctr-glm-margin-xgb-lb-0-96127), snapshot SHA256 `94bf11d2946dc9f3d0589ce307dc143e6f2e6d2bd8929b73a3571df10ba1b7a6`. The downloaded cells contain no executed outputs for the new margin section; its scores are author-reported narrative, not locally reproduced measurements. In particular, **.961078 is ordinary XGB**, not GLM-margin XGB. The author reports GLM-only .958541, depth-4 margin XGB .960752, depth-6 .960808, and approximately .00001 additional stack AUC. These claims do not establish a gain in our split or blend.

The GLM has 122 inputs: two original-source model logits and 120 target-rate logits. Keys comprise 21 individual features, five distance/age/delay buckets, all 78 pairs among 13 chosen strong features, the travel/class/customer-type segment, its 13 rating crosses, and its age/distance-bucket crosses. Each outer training set receives five-way inner OOF encodings with smoothing 20; outer validation/test maps use all outer-training labels. Probabilities are clipped to [1e-4, 1-1e-4], transformed to logits, standardized on outer-training rows, and fit by L2 logistic regression with C=.05 and max_iter=400. The same fitted GLM produces the training, validation, and test initial margins. Residual XGB receives the separate ordinary engineered feature matrix, depth 4 or 6, learning_rate .03, subsample .8, colsample_bytree .5, min_child_weight 20, reg_lambda 5, hist, seed42, native categoricals, and max_cat_to_onehot1. The cap is 20,000 rounds with 300-round outer-fold early stopping. Test predictions average fold logits before sigmoid. A simple low-cardinality one-hot GLM on our existing inputs would therefore be a different experiment.

### Validation and leakage assessment

The target-encoding code excludes each row's own inner-fold labels and excludes outer-validation labels from GLM fitting. In-sample margins from a low-dimensional GLM used as the first fitted component are not themselves an outer-validation leak. However, outer validation selects XGB best iteration, so the reported fold AUC is not an independent test of the fully selected estimator. Global category factorization, counts, and route profiles use concatenated train/test feature rows; these are transductive, without target use. Original rows are deduplicated on full rows but not anti-joined against synthetic feature profiles; our teacher excludes those overlaps. The public second-stage cross-validation fits a combiner on the existing OOF library, without refitting base learners inside each meta split. It prevents direct combiner resubstitution but is not a fully nested evaluation of the complete ensemble pipeline. These distinctions matter when interpreting a claimed .00001 gain.

### Primary theory, native IO, and cost

[Chen and Guestrin, 2016, sections 2.1-2.3](https://arxiv.org/html/1603.02754v3) formulate additive tree updates around the current prediction using loss gradients and Hessians. A fitted linear initial predictor changes that starting point; the paper does not establish that it improves this dataset. Official [XGBoost intercept documentation](https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html) and the [boost-from-prediction example](https://xgboost.readthedocs.io/en/stable/python/examples/boost_from_prediction.html) confirm that binary margins must be untransformed logits and override base_score. Every inference call must supply the GLM margin. The [API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) requires a matching `base_margin_eval_set` for validation. [Prediction documentation](https://xgboost.readthedocs.io/en/stable/prediction.html) distinguishes sklearn's automatic best-iteration selection from native Booster inference; a wrapper must persist and apply the selected iteration range explicitly. [Native saving documentation](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html) supports UBJ for trees, but external imputation, scaling, category vocabularies, GLM coefficients/intercept, feature order, and class order need separate JSON/array artifacts. No estimator pickle is necessary. Online docs identify 3.4.2; the installed XGBoost is 3.4.1, so any future implementation still needs local API verification.

A faithful replica adds 120 encoders and a dense 122-column GLM matrix: at 419,000 training rows that matrix alone is about 204 MB in float32 or 409 MB in float64, before transformed copies and validation/test matrices. It also adds 600 inner group aggregations per outer fold and a GLM solve. Wall time was not measured; a simpler one-hot variant would be cheaper but has no corresponding positive source result. CPU reload tests would need to verify sigmoid(GLM margin + tree residual), validation-margin attachment, exact feature order, unknown categories, and fixed-round refitting. No additional dependency is required.

### One concrete existing-family ablation

Source ordinary XGB uses the same six target-encoding keys and smoothing20 as our `xgb_route_te_teacher`, plus one original-XGB teacher, counts, and per-Flight-Distance means of all other 20 raw features. Our implementation has target/count encodings and a teacher but **no route-profile means**. A focused local variant should keep our existing seed, split, .03 learning rate, depth8, teacher, target encodings, counts, and other parameters unchanged, adding only train-fitted Flight-Distance means for Age, the 13 rating columns, and both delay columns. That is 16 numerical group means; omitting the source's arbitrary ordinal category averages is a deliberate simplification. Unseen route values use that fitting split's global means; all-missing columns require a fixed fallback. Fit the maps separately inside the stopping split and during outer-fold refit, persist them as JSON, and never learn them from audit/test rows. Test only paired development-fold AUC and predeclared blend contribution.

The local baseline's saved result is .9605210838951841, 52.453 seconds, selected rounds326/315/495; its depth6 variant is .96053768398209, 52.719 seconds. These are existing logged measurements. The source/local gap is not causal evidence: our outer fits use about419k rows versus about559k in the author's five-fold/full-training procedure, and stopping, transductive statistics, native-vs-ordinal categorical handling, seed, learning rate (.02 vs .03), and original-teacher regularization differ. The route-profile proposal is justified by a specific omitted input and low implementation cost, not by assuming the entire gap is recoverable. No new family or dependency is recommended.

### Route-profile implementation verification

At root's subsequent request, implemented only `scripts/route_profiles.py` and its dedicated synthetic test module. `RouteProfiles` accepts an optional fixed key/column/prefix config; fit, fit_transform, transform, directory save/load are supported. Outputs `route_profile_mean_00` through `_15` are ordered float32 features, with original columns/index/order preserved. Route keys normalize to float32; profile feature NaNs are ignored. Missing/unseen routes use training global means; missing group measurements fall back to global means, and an entirely missing measurement uses zero. JSON metadata records schema, global fallbacks, training row count, and a native Parquet map's SHA256. No counts or target statistics are added.

Executed `.venv/Scripts/python.exe scripts/test_route_profiles.py`: all six tests passed in 0.203 seconds. Fixtures cover validation-only extreme values and unseen routes without changing fitted statistics, missing measurements/routes, raw CSV integer vs float32 and nullable keys, duplicate indices and row permutations, empty transforms/maps, native save/load equality, schema violations, and checksum corruption. All fixtures were constructed in code; no competition data or labels were loaded. No GPU was used. Implementation SHA256 `8f61d0ab05eb146a0b6532265d1e23f9f4f5b81c9aa5a719b511cf5b7512a0e1`; test SHA256 `6f25701b290c6ebbadf7a00be990720de48e138f4d19fab989a1d8db6dd880e4`. Competition-quality evidence awaits root's unchanged-baseline profile ablation. No `xgb_margin.py` was created.

## Matched profile result and final candidate boundary, 2026-10-02 03:57 UTC

Read the existing baseline/profile `result.json` files, configuration, and development-blend metadata only. No audit labels, prediction arrays, training, or new metric evaluation were used for this documentation closeout. The main controller completed `xgb_route_te_teacher_profiles`; the only recorded model-config differences from `xgb_route_te_teacher` are the run ID and `route_profiles: true`. Both use split SHA256 `4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c`, seed20261004, the same teacher and target/count encodings, depth8, learning_rate.03, and the same stopping protocol.

| Logged result | Pooled development OOF AUC | Fold 0 | Fold 1 | Fold 2 |
|---|---:|---:|---:|---:|
| Existing XGB teacher + target/count encodings | .960521083895 | .960212266732 | .960427817408 | .960944390700 |
| Same configuration + 16 route-profile means | .960674529171 | .960311865754 | .960667303350 | .961060778463 |
| Paired difference | **+.000153445276** | +.000099599022 | +.000239485942 | +.000116387763 |

All three fold differences are positive. The profile run recorded61.047seconds and selected315/346/558rounds, versus52.453seconds and326/315/495rounds for baseline. Its result contract hash is `0aecdca51ffdad7d657670538f70b20ce16d3d8f67cd53cc83fbd5131977733b`; OOF-file hash `cc41caf0539b31352795536f2e67c20b706a116a07bb732e631ae912890c5853`. The positive development result supports this local feature hypothesis; it does not estimate an independent test gain or validate the public-source score gap.

`artifacts/blend/current.json`, created03:54:35.899UTC, records development OOF AUC **.961221030185844**, best single-model OOF .9607325385508942, and profile-XGB weight .1542005859375. The recorded procedure is coarse nonnegative convex weight search on development OOF, excluding the final audit. This is the blend's selected development score, not an untouched evaluation and not a claim that the profile alone contributed the difference from an earlier blend.

The main controller predeclared a stop after **30 candidates**. The current configuration contains exactly30 runs, with `cat_route_te_teacher_profiles` as the final queued ablation. It matches the existing `cat_route_te_teacher` model/feature settings except for the added profiles and a reduced wall-clock timeout of **600seconds** (baseline6000seconds); model parameters and stopping settings remain identical. At this checkpoint its `result.json` did not exist, so no CatBoost-profile completion or score is claimed. The supervisor snapshot was stopped with no active child pending the controller's queue continuation. No GLM-margin implementation or additional model family is proposed.
