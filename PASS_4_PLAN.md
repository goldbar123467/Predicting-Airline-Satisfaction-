# Pass 4: plan to compete for first place in S6E10

Written 2026-10-07 for the agents running the next pass. Competition closes **2026-10-31 23:59 UTC** (about 24 days). This file is the directive for Pass 4. It needs the user's approval of the scope in section 9 before any training or submission. Everything here is either (a) measured in this repository, (b) measured by another competitor and published, or (c) a hypothesis. Each claim is tagged.

Tags: **[ours]** measured in this repo's receipts. **[field]** published by another competitor (GitHub README, Kaggle notebook title or search snippet), not reproduced by us. **[paper]** academic or vendor source. **[hyp]** untested hypothesis.

---

## 0. The bet, in five lines

1. We are behind on the leaderboard because of **ensemble breadth and base-model strength**, not because of training duration. Stop all epoch/schedule research.
2. Switch to the field's validation: **5-fold `StratifiedKFold(5, shuffle=True, random_state=42)` on all 699,635 rows**. It gives each model more data, makes our OOF comparable with public numbers, and makes public OOF libraries stackable.
3. Rebuild strong members with the levers other teams measured: LightGBM `extra_trees` on a rich view, CatBoost with all columns categorical, fine binning of `Flight Distance`, original-data per-value target statistics, RealMLP with `n_ens=32`, and a **TabPFN-3.5** member.
4. Combine everything with a **nested-CV L2 logistic-regression stacker on member logits**, the method used by every top public S6E10 stack.
5. Decide (user) whether to stack **public OOF libraries**. Public evidence says this is the largest single lever: one team went from LB 0.96047 with own models to 0.96177 (rank 4 of 885) by adding 184 public plus own members.

Honest expectation: own-models-only work should close part of the gap (estimate LB 0.9614 to 0.9616 [hyp]). Reaching the 0.9618+ band almost certainly requires public libraries or a merge. First place on the private LB within a band 1e-4 wide is partly luck (section 7).

---

## 1. Where we stand

### Our verified state [ours]

| Item | Value | Source |
|---|---:|---|
| Best public LB | **0.96125** (submission 56775181, Oct 2) | THIRD_PASS_BATCH04_REPORT.md, LONG_LOCAL_500_REPORT.md |
| Earlier submissions | 0.96093 (56771783), 0.96122 (56774588) | SECOND_PASS_REPORT.md |
| Dev OOF of submitted blend | 0.961407501 (3-fold, dev rows only) | THIRD_PASS_REPORT.md |
| Best single model | 0.960872 (`v3_realmlp_cat_raw_aux_probability`) | handoff inventory |
| Best XGB / CatBoost / LightGBM | 0.960774 / 0.960757 / 0.959704 | handoff inventory |
| Members in blend | 15 (RealMLP 47.6%, XGB 32.6%, Cat 17.9%, LGB 1.9%) | DEEP_RESEARCH_REPORT.md |
| Last LB improvement | Oct 2. Oct 2 to Oct 7 went to duration and continuation tests: 72 paired fits, 1,152 epochs, zero release gain | CONTINUATION_SIX_V1_REPORT.md |

### Field state (external, dated; Kaggle itself was blocked from the research container, re-verify on day 1)

| Date | Fact | Tag / source |
|---|---|---|
| Oct 3 | LB top 0.96155; rank 100 = 0.96100 | [field] jamesidriss/airline_S6E10 `research/current_meta.md` |
| Oct 4 | LB top 0.96165; rank 25 = 0.96152; rank 100 = 0.96123 | [field] jamesidriss STATUS.md |
| Oct 5 | 768 teams; top 100 at about 0.96150 | [field] aditbytes README |
| Oct 6 | ChrisLegge 0.96177, rank 4 of 885 (5-fold CV 0.96208) | [field] ChrisLegge README |
| current | Search index shows #1 isthereanycode 0.96183 | [field] search snippet of the LB page |
| Public LB share | 20% of test (about 60k rows); score truncated to 5 decimals | [field] jamesidriss, from Kaggle API |
| Calibration | public LB is about 5-fold OOF minus 0.00035; paired public noise about +/-0.0002 | [field] 4 independent authors per jamesidriss; ChrisLegge reports -0.0004 |

**Gap to #1: about 0.0006 on public.** Our 0.96125 was roughly rank 100 on Oct 4 and is now likely lower. Treat our current rank as unknown until checked.

---

## 2. Why we are behind: honest diagnosis

Ranked by expected cost to our score.

**D1. Ensemble too small, too similar, weighted with a coarse grid.** 15 members, convex weight search. Top public stacks use 60 to 184 members and an L2 logistic stacker on logits with nested CV [field: ChrisLegge, bansal1600 PRs]. ChrisLegge's own-model stack scored LB 0.96047; adding public members raised it to 0.96160 (48 members), then 0.96177 (184 members) [field]. Within one team's own models, equal-weight and nested-LR stacks were within 1e-5 of each other at 59 members [field: jamesidriss]. **Diversity of members matters; the choice of combiner barely does.**

**D2. Weak GBDTs.** Our LightGBM (0.9594 to 0.9597) gets 1.9% blend weight. Others report 0.96053 tuned LightGBM with `max_bin=4095` [field: thomasjpfan], 0.961175 to 0.961260 for LightGBM with `extra_trees=True` on a 285-feature view [field: jamesidriss, reproduced on a second fold seed]. Our configs use default `max_bin` 255 for LightGBM, 256 for XGB, `border_count` 128 for CatBoost, 31 to 127 leaves, lr .03 to .04, and LightGBM/XGBoost stopped after about 110 to 560 rounds [ours: configs/overnight.json, handoff inventory]. Coarse bins merge the 3,474 exact `Flight Distance` values, which behave like route IDs [field: thomasjpfan, ChrisLegge]. Our `cat_twins_teacher` used `max_ctr_complexity=1`, the setting another team found always worst [field: thomasjpfan; ours: 0.96036].

**D3. No foundation-model member.** A public TabPFN-3.5 member with `Flight Distance` as a category reached OOF 0.961053 on the seed-42 folds [field: bansal1600 PR search snippet, unverified], above our best single model. Its errors come from a different model class, which is what the stack needs.

**D4. Original data used only partly.** We use an original-only XGB teacher and 13 expected-rating features. Another team measured smoothed per-value `P(y | value)` lookups from the original data at **+0.001015** on raw LightGBM, conditional surfaces keyed on FD/Class/travel/boarding at +0.001333 alone, and only +0.000035 for a teacher on top of those [field: jamesidriss]. ChrisLegge reports FD group stats over train+test+original plus original satisfaction rate per FD as "about 10x every other idea" (single LightGBM 0.95874 to 0.96071) [field].

**D5. Validation design costs data and blocks comparison.** 10% audit holdout plus 3 folds on dev means OOF models fit on 60% of labels (54% inside the inner stopping split) [ours: scripts/prepare.py, scripts/train.py]. Field folds fit on 80%. Another team measured +6.3e-4 AUC per doubling of training rows on this data, with the law over-predicting by about 2x near the top of the range [field: jamesidriss learning curve and correction]. Expected OOF effect of 60% to 80%: +1e-4 to +2.6e-4 per member. **Caveat:** our submitted test models were full refits on 100% of labels, so this mostly understated our OOF and weakened weight estimation; it is not a direct LB loss. The bigger costs are that our numbers are not comparable with the field, and no public member can be stacked on our folds.

**D6. Process cost.** The six continuation experiments needed 14 startup receipts, exact-prefix gates and once-only evaluators per stage. That rigor answered a question that cannot move the LB. Pass 4 keeps the checks that prevent wrong submissions and drops the rest (section 4).

---

## 3. What to stop (evidence attached)

| Stop | Why |
|---|---|
| Epoch horizon, duration, dropout, LR cooling, EMA, label smoothing, frozen embeddings, head-only continuation | 0 of 6 continuation tests passed; 60/500-epoch runs lost 1e-4 [ours] |
| 3-fold dev split and audit holdout as the selection protocol | Audit already exposed twice [ours]; costs data (D5) |
| Coarse convex weight grids and the 1e-5 gate on reused OOF | Grid optimum is a step function and unstable across folds [ours: DEEP_RESEARCH_REPORT.md] |
| Appending original rows to training | -0.000585 at weight .3, -0.001103 at 1.0 [field: jamesidriss]; -0.00038 [field: discussion 745098] |
| TE on duplicate rating keys, TE of rating pairs, 118 pairwise original tables | -0.00005, -0.00002, -0.000136 [field: jamesidriss] |
| kNN target-encoding block, TabR | -1.5e-4 and negative blend weight [field: jamesidriss] |
| Region-local reliability gating | 0.000000 under nested test [field: jamesidriss] |
| `extra_trees` on raw or small feature views | -0.0056 on 21 features, -0.0016 on 75 [field: jamesidriss, shadow folds] |
| Seed averaging beyond 3 seeds, Optuna beyond about 40 trials | +3e-5 and flat [field: ChrisLegge] |
| Public LB hill climbing | Paired public noise +/-0.0002 [field]; in an S6E9 analysis cited by jamesidriss, a public-fitted submission went from 0.94945 public to 0.94313 private, and private change correlated at r = -0.97 with public change [field] |
| Rating aggregate features for GBDTs (means, zero counts, travel x class) | -0.0001 to -0.0002 [field: thomasjpfan, ChrisLegge]. Keep N/A masks only as RealMLP/TabPFN inputs |

---

## 4. Operating rules for Pass 4

1. **One validation scheme.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over `train.csv` in file order with `y = satisfaction` as int. Save `folds_v4.parquet` (id, fold) and its SHA256 once. Every member uses it. Before stacking any public member, confirm alignment: the public dataset's fold column, or its per-fold AUCs recomputed on our folds, must match.
2. **Member contract (the only required artifact).** `members/<name>/oof.parquet` (id, pred) for all 699,635 rows, `test.parquet` for all 299,844 rows, `meta.json` (config, feature view, fold AUCs, pooled AUC, seconds, code hash). Test predictions are the **average of the 5 fold models** (field standard, consistent with OOF). Full-data refits are optional and only for the final top members (section 6, P3).
3. **Early stopping.** GBDTs may stop on the outer fold with patience of 200 or more; the AUC curve is flat near the optimum, so the bias is small and equal to the field's. **Neural nets use a fixed epoch count, never selected on the scored fold**: a public author measured .961420 when choosing the checkpoint on the scored fold against .961014 on an internal split [field: goodpjw2008, in research/competition_evidence.md].
4. **Admission by stack ablation, not standalone AUC.** A member stays if the nested stack OOF with it beats the stack without it by at least +1e-5 pooled and on at least 3 of 5 folds. Report every ablation in `members/ledger.csv`.
5. **Selection uses nested OOF only.** Public LB is a sanity check against the calibration (LB is about OOF minus 0.00035). Investigate any deviation beyond +/-0.0003.
6. **Submissions:** at most 3 per day; each must carry its nested OOF in the description. Keep the existing validator checks: 299,844 rows, exact ID order, finite values in [0,1], header `id,satisfaction`, SHA256 dedup against earlier submissions.
7. **Leakage guards for features.** Label-free statistics may use train+test+original (field standard, legal, no labels). Statistics that use competition labels (TE) are fit inside each training fold with inner K-fold for the training rows. Statistics from original labels are fit on the cleaned original table only (129,859 rows, anti-joined; existing `data/original.parquet`).
8. **Compute:** local RTX 5070 (12 GB) plus free private Kaggle T4x2 (15 GB each, about 30 GPU-hours per week). One job per GPU. No paid compute unless the user adds it. CPU work (stacking, features) runs in parallel.
9. **Reporting:** one `PASS_4_LOG.md` appended daily: members added, nested stack OOF, submissions, LB. No per-stage authorization documents.

---

## 5. Work plan

### Phase 0 (day 1, at most 4 hours): re-baseline on the new folds

1. Re-verify on Kaggle: current LB top 10 and our rank, Rules section 2.6 (external data/models) and section 6, the list of public OOF datasets in section 6 P2 and their licenses.
2. Create `folds_v4.parquet`. Build `scripts/v4/` with three pieces: `features_v4.py` (feature store, section 5 Phase 1), `train_member.py` (one generic 5-fold runner for xgb/lgb/cat/realmlp/tabpfn that writes the member contract), `stack_v4.py` (nested LR stacker, equal-logit mean and Caruana hill climb).
3. Reuse, do not rewrite: `scripts/route_profiles.py`, `scripts/encoding.py` (fold TE), `scripts/original_aux.py`, `scripts/original_aux_probability.py`, `scripts/teacher.py`, `scripts/original_lgb_teacher.py`, `scripts/realmlp_categorical.py`. The original-only banks in `data/*_predictions.parquet` are keyed by id and need no refit.
4. Retrain the three strongest existing recipes unchanged on the new folds: `v3_xgb_route_te_teacher_profiles_aux_probability`, `v3_realmlp_cat_raw_aux_probability`, `cat_route_te_teacher`. Stack them. **Exit:** we have comparable 5-fold numbers. Submit the 3-member stack once to calibrate our LB minus OOF gap.

### Phase 1 (days 1 to 3): feature store v4

All blocks are columns joined by id. Build each once for train+test.

| Block | Content | Evidence |
|---|---|---|
| F1 `orig_te` | For each of 21 columns: smoothed `P(y=1 | value)` from the original table (m-estimate, m=20), plus count of that value in the original | +0.001015 on raw LightGBM [field: jamesidriss] |
| F2 `orig_surf` | Original satisfaction rate per FD, per (FD, Class), (FD, Type of Travel), (Online boarding, Class, Type of Travel); smoothed | +0.001333 alone, redundant with F1 [field: jamesidriss]; FD rate in ChrisLegge's best view [field] |
| F3 `fd_profile` | Label-free per-FD stats over train+test+original: row count, mean and std of all 14 ratings, Age and both delays; residual of each rating from its FD mean | ChrisLegge single LightGBM 0.95874 to 0.96071 [field]; our train-only profile means gave +0.000153 [ours] |
| F4 `counts` | Value counts (train+test) for all numeric columns; N/A masks for rating 0; arrival-delay missing flag | count encoding +0.0007 [field: thomasjpfan] |
| F5 `aux` | Existing teacher probability, 13 expected ratings, 16 aux probabilities and log score | already in the 0.96125 release [ours] |
| F6 `te` | Fold-safe TE (existing 6 keys, smoothing 20) | +0.0014 with counts [field: thomasjpfan]; partly redundant with F3 |
| F7 `twins` | Exact-value categorical copies of all numeric columns, plus Class x Travel x Gender | RealMLP and CatBoost input [ours, field] |
| F8 `tokens` | GPT-2 BPE token keys of numerics | +2.5e-5 only [field]; build last, if ever |

Views: `raw` (21), `rich` (F1 to F6, about 150 to 300 columns, for GBDTs), `nn` (raw + F5 + F7 + N/A masks, for RealMLP), `pfn` (raw + FD as category + F5 + F1, under about 60 columns, for TabPFN).

**Gate for the feature store:** LightGBM with the field's settings on `rich` must reach at least 0.9608 on the new folds. If it does not, find the bug before training more members.

### Phase 2 (days 2 to 10): strong, diverse own members

Run in this order. Each line gives the expected standalone OOF on 5 folds from field measurements; ours may differ.

| # | Member | Key settings | Expected [field] | Cost (est.) |
|---|---|---|---:|---|
| M1 | LightGBM extra-trees | `rich` view, `extra_trees=True`, 127 leaves, lr .02, `max_bin` 1023, colsample .8, subsample .8 freq 1, 3 seeds; then 10-fold copy of the best seed | 0.96118 to 0.96126 | CPU, about 20 to 40 min per seed |
| M2 | LightGBM tuned | `rich`, deterministic, `max_bin` 4095, 67 to 255 leaves, strong L1 and L2 (about 15), feature_fraction .5; Optuna at most 40 trials on fold 0, lr .05 for search, .02 for final | 0.9605 to 0.9610 | CPU, 2 to 4 h |
| M3 | CatBoost all-categorical | all 21 columns as categoricals plus numeric copies, `max_ctr_complexity` 4, `one_hot_max_size` 2, depth 7; second variant depthwise depth 10; `border_count` 254 | 0.96075 to 0.9611 | GPU, 30 to 90 min |
| M4 | XGBoost rich | lr .02, depth 8, `max_bin` 1024, colsample .5, min_child_weight 5, lambda 1, patience 400 | 0.9610 to 0.9612 | GPU, 10 to 20 min |
| M5 | XGBoost from GLM margin | fit an L2 logistic regression on standardized F1/F2/TE logits per fold; pass its logit as `base_margin` (train, valid and test) to XGB depth 4 to 6, lr .03 | GLM-margin depth 6 .960808, adds about 1e-5 to a stack [field: goodpjw2008]; S6E9 40th place used it [field] | GPU, 20 min |
| M6 | RealMLP | PyTabKit 1.7.3, `nn` view with numeric twins, flat_anneal, **fixed 4 epochs**, `n_ens` 32, hidden 512/256/128, lr .053, wd .015, batch 256; seeds 2 to 3; one variant raw + aux only (no TE) | n_ens 8 to 32: +0.0002 (0.960820 to 0.961017) [field: jamesidriss] | GPU, 30 to 45 min each |
| M7 | TabPFN-3.5 | `pfn` view; `fit_mode="fit_with_cache"`; context per fold subsampled (start 100k rows, then 250k), 2 to 4 context bags averaged; also try TabPFN-3.5-Fast | member OOF 0.961053 [field, unverified]; full context needs about 17.5 GB VRAM [field: ChrisLegge] | Kaggle T4, 2 to 6 GPU-h; measure on fold 0 first |
| M8 | TabM | k=32, PLR embeddings, `nn` view, fixed epochs | 0.960693, marginal +3e-6 [field] | only if a GPU sits idle |

Kill rules: stop a member family after fold 0 if its fold-0 AUC is more than 0.0008 below the best family and it has not shown stack value elsewhere. Stop TabPFN if peak VRAM at 100k context exceeds the device or fold-0 AUC is below 0.9595.

### Phase 3 (from day 3, rolling): stacking

1. **Level-2 inputs:** per member, clip probabilities to [1e-6, 1-1e-6], take logits, standardize using training-fold statistics.
2. **Combiners, all with nested CV on the same 5 folds:** (a) equal-weight logit mean; (b) L2 logistic regression with C chosen by inner 4-fold CV from {0.005, 0.01, 0.02, 0.03, 0.05, 0.1} (field picks 0.02 to 0.03 [field]); (c) Caruana greedy selection with replacement, bagged 20 times [paper: Caruana 2004, 2006]. Pick by nested OOF.
3. **Report** nested stack OOF, per-fold values, and the paired DeLong SE against the current best (existing `analysis_auc_uncertainty_v1.py`).
4. Do not add a second stacking level (GBDT meta-models) unless nested OOF gains more than 3e-5; field reports these overfit easily.

### Phase 4 (days 5 to 28): public members, only if decision D-A is approved

1. Candidate libraries named by top public stacks [field: ChrisLegge README, bansal1600 PRs, jamesidriss audit]: goodpjw2008 (route, aux-task, RealMLP and TabPFN-3.5 members, "TabPFN + Route Categories" LB 0.96160), megayak OOF library v3, dariushafshar "golem" OOF library, sachith7 stack OOF predictions, arhancanli12 OOF and test predictions, najiama/s6e10-oof, busyaprime PRIME (re-run and shared by kratosyan), sadamtorres artifacts library. Re-check that each is a public Kaggle dataset or notebook output and note its license.
2. **Ingestion checks per member:** 699,635 OOF and 299,844 test rows; ids match; fold alignment (rule 4.1); finite values; standalone OOF below 0.9625 (anything higher is a leakage suspect: quarantine); OOF and test prediction distributions similar (two-sample KS statistic below 0.01); prefix names (`pub_<author>_`).
3. Add public members to the stack through the same nested admission rule. Keep an own-only stack in parallel at all times.

### Phase 5 (days 26 to 31): finals

1. Freeze members on Oct 28. No new member after that unless it is already running.
2. Final pick 1: best nested-OOF stack overall.
3. Final pick 2 (hedge): the best nested-OOF stack that differs materially from pick 1, for example own members only, or stronger regularization (C one step lower). Never pick by public LB.
4. Optional P3 if compute is free: full-data refit of the top 3 own GBDT members using the median best iteration (no upward scaling: iteration count did not grow with data in another team's measurement [field: jamesidriss]). This cannot be scored by CV; use it in the hedge pick only, never in both.
5. Final verification: the existing raw-inference and schema checks, applied to the two final files only.

---

## 6. Priority queue with expected value

Expected gains are **transfers of other teams' measurements** to our pipeline. Treat them as priors.

| Priority | Item | Expected stack gain | Confidence | Cost |
|---|---|---:|---|---|
| P0 | New folds + member contract + nested stacker | enabler | high | 0.5 day |
| P0 | Feature store F1 to F4 | +0.0002 to +0.0005 on GBDT members | medium | 0.5 day |
| P0 | M1 LightGBM extra-trees on `rich` | +0.0001 to +0.0003 to the stack | medium-high (reproduced on 2 fold seeds by its author) | low |
| P0 | Decision D-A on public libraries | +0.0002 to +0.0005 LB over a strong own stack (ChrisLegge gained +0.0013 from a weaker own base of 0.96047) | medium; depends on library quality | low compute |
| P1 | M7 TabPFN-3.5 | +0.0001 to +0.0002 (new model class) | low-medium; VRAM risk | 2 to 6 GPU-h |
| P1 | M3 CatBoost all-categorical, M4 XGB rich | +0.00005 to +0.0001 each | medium | low |
| P1 | M6 RealMLP n_ens 32 | +0.00005 to +0.0001 | medium | medium |
| P2 | M5 GLM-margin XGB, M2 tuned LightGBM, 10-fold copies of top 3 members | +1e-5 to +1e-4 | low-medium | medium |
| P3 | Full-data refits, M8 TabM, F8 tokens | under +5e-5 | low | medium |

---

## 7. Statistics the agents must respect

- **Public LB noise.** About 60k public rows give paired SE of about 0.0002 for highly correlated submissions [field]. A public change under 0.0003 tells us nothing.
- **Private LB noise.** About 240k private rows; paired SE roughly 0.0001 at test-prediction correlation about 0.995 (scaled from the public estimate [hyp]). The top 45 public teams spanned 3e-4 on Oct 3 [field]. First place requires being about 1e-4 better in expectation **and** some luck.
- **OOF noise.** Our measured paired SE for a near-duplicate blend change was 1.25e-5 (conditional on fitted models) [ours]. GPU CatBoost reruns vary by about +/-0.0002 standalone [field: thomasjpfan]. Use the 3-of-5-fold rule plus the pooled gain, not one number.
- **Selection bias.** Choosing among many stacks on the same OOF inflates the winner [paper: Cawley and Talbot 2010; Varma and Simon 2006]. Nested CV for the stacker controls the combiner, not the member-selection history; keep the number of compared final candidates small (at most 5).

---

## 8. Academic and vendor grounding, tied to decisions

| Decision | Source |
|---|---|
| Linear stacker on logits, nested CV | Wolpert, "Stacked generalization", Neural Networks 1992; Ting and Witten, "Issues in stacked generalization", JAIR 1999 |
| Greedy selection with replacement as a check | Caruana et al., "Ensemble selection from libraries of models", ICML 2004; Caruana et al., "Getting the most out of ensemble selection", ICDM 2006 |
| Small, pre-registered final comparisons | Cawley and Talbot, "On over-fitting in model selection", JMLR 2010; Varma and Simon, "Bias in error estimation when using cross-validation for model selection", BMC Bioinformatics 2006 |
| Extra-trees to average out i.i.d. label noise | Geurts, Ernst and Wehenkel, "Extremely randomized trees", Machine Learning 2006 |
| CatBoost categorical combinations, ordered target statistics | Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features", NeurIPS 2018 |
| Target and count encoding | Micci-Barreca, "A preprocessing scheme for high-cardinality categorical attributes", SIGKDD Explorations 2001 |
| RealMLP defaults, internal ensembling | Holzmüller et al., "Better by default: strong pre-tuned MLPs and boosted trees on tabular data", NeurIPS 2024 |
| TabM, numeric embeddings | Gorishniy et al., "TabM", ICLR 2025; Gorishniy et al., "On embeddings for numerical features", NeurIPS 2022 |
| Foundation-model member up to 1M rows | Hollmann et al., Nature 2025 (TabPFN v2); Jäger et al., "TabPFN-3.5: Technical Report", arXiv 2609.17895 (2026). Package `tabpfn` 9.1.0 on PyPI (Oct 2, 2026); TabPFN-3.5 weights carry a non-commercial license and need a Prior Labs token (`TABPFN_TOKEN`) |
| Context selection when context must be subsampled | Thomas et al., "Retrieval and fine-tuning for in-context tabular models" (LoCalPFN), NeurIPS 2024 [use only if random-subsample context underperforms] |
| Cross-family ensembles beat single families | Erickson et al., "TabArena", NeurIPS 2025 Datasets and Benchmarks |
| Feature engineering still moves the frontier | "TabPrep: Closing the feature engineering gap in tabular benchmarks", arXiv 2606.02384 (2026) |
| More rows per model pays | Hestness et al., "Deep learning scaling is predictable, empirically", 2017; on this data: +6.3e-4 per doubling [field] |

---

## 9. Decisions the user must make before Pass 4 starts

| ID | Decision | Recommendation | Why it is the user's call |
|---|---|---|---|
| D-A | Stack public OOF/test libraries from other competitors | **Yes**, with the ingestion checks in Phase 4 and a parallel own-only stack | Rules 2.6 allow publicly available external data at no cost [field: rules quoted by jamesidriss]; earlier passes chose "own runs only" as an agent judgment, not a user rule. It changes what "our" solution is. Prize code-delivery (rule 2.8.b) may be harder; prizes are merchandise |
| D-B | Accept the TabPFN-3.5 non-commercial weight license with a Prior Labs account | Yes, for this non-commercial competition | Requires the user's account and token |
| D-C | Scope approval: local GPU training, free private Kaggle GPU jobs, up to 3 submissions/day through Oct 31, final 2 selections by the rules in Phase 5 | Approve | AGENTS.md requires explicit authorization for training and submissions |
| D-D | Team merge with another strong team before the merger deadline | Optional; consider after Phase 3 if we sit in the 0.9616 to 0.9618 band | Social decision; legal under rules 2.9 if the merged submission count fits |

Proposed authorization text for AGENTS.md once approved: "Pass 4 authorized per PASS_4_PLAN.md: 5-fold seed-42 validation, own members, [public libraries: yes/no], [TabPFN-3.5: yes/no], local GPU plus free private Kaggle GPU, at most 3 submissions per day, finals by nested OOF. Supersedes earlier no-training/no-submission wording for Pass 4 only."

---

## 10. Day plan

| Dates (UTC) | Goal | Exit check |
|---|---|---|
| Oct 7 to 8 | Phase 0, F1 to F4, M1 on fold 0 | 5-fold re-baseline stacked; LightGBM `rich` at 0.9608 or above; 1 calibration submission |
| Oct 8 to 11 | M1 (3 seeds), M3, M4, M6; nested stacker live | own nested stack OOF at least +0.0002 over the Phase 0 re-baseline. Reference: LB 0.96125 plus the 0.00035 calibration implies our current blend is worth about 0.9616 on field-style 5-fold OOF [hyp] |
| Oct 10 to 13 | M7 TabPFN on Kaggle; Phase 4 ingestion if D-A | TabPFN fold-0 decision; first public-member stack submission |
| Oct 13 to 24 | Fill gaps by stack ablation; 10-fold copies of top members; M5, M2 | each day: nested OOF up, or a written reason why not |
| Oct 25 to 28 | Freeze candidates; optional refits | members frozen Oct 28 |
| Oct 29 to 31 | Two finals, verification, selection on Kaggle | both finals selected by Oct 31 18:00 UTC (6 h buffer) |

---

## 11. Known unknowns and risks

- **Field numbers are not ours.** Every [field] gain came from another pipeline. Some came from search snippets of pages that were not opened (bansal1600 PRs now return 404). Re-measure before relying on them.
- **TabPFN-3.5 VRAM.** Full 560k context reportedly needs about 17.5 GB; T4 has 15 GB and the local card 12 GB. Subsampled context may lose most of the member's value.
- **Public library hazards:** fold misalignment, hidden test-label use in a dataset (would show as implausible OOF), or members selected on the public LB. The stacker fits on OOF only, which limits but does not remove that selection effect.
- **Transductive label-free features** (train+test counts and profiles) are legal and standard. They do not leak labels, but they mean CV and test share preprocessing; the field's LB calibration already reflects this.
- **Data location.** Raw data, folds, model files and the original-only caches live on the user's Windows machine (`C:\Users\thecl\Documents\Predicting Airline Satisfaction`), not in this git repository.
- **Win probability.** Being top 10 on private is a realistic target if D-A is approved. First place needs a genuinely better signal than the 0.9618 band shares; candidates are F1 to F3 tuned on our side, TabPFN context bagging, and extra-trees variants. None is guaranteed.

---

## 12. Sources used for this plan

Repository: README.md, REPORT.md, SECOND_PASS_REPORT.md, THIRD_PASS_*.md, LONG_LOCAL_500_REPORT.md, CHECKPOINT_BLEND_REPORT.md, DEEP_RESEARCH_REPORT.md, CONTINUATION_SIX_V1_REPORT.md, ALL_ATTEMPTS_AND_METHODS_AGENT_HANDOFF.md (model inventory), research/competition_evidence.md, research/second_pass_competition.md, configs/overnight.json, scripts/prepare.py, scripts/train.py, scripts/common.py.

External (accessed 2026-10-07 from a container where kaggle.com, arxiv.org and huggingface.co were blocked; facts come from GitHub raw files, PyPI and search-result text):

- ChrisLegge, [kaggleAirlineSatisfaction](https://github.com/ChrisLegge/kaggleAirlineSatisfaction) README: progression to LB 0.96177, public library list, what worked and what did not.
- jamesidriss, [airline_S6E10](https://github.com/jamesidriss/airline_S6E10) `research/current_meta.md` and `STATUS.md`: rules, LB calibration, original-data ablations, extra-trees, learning curve, rejected ideas.
- thomasjpfan, [kaggle-playground-series-s6e10](https://github.com/thomasjpfan/kaggle-playground-series-s6e10) `analysis.md`: max_bin, CatBoost all-categorical, segment weaknesses, TabICL/Kumo context results.
- ferreret, [kaggle-playground-s6e10](https://github.com/ferreret/kaggle-playground-s6e10) README: FD exact-value TE, CV and LB tracking.
- aditbytes, [kaggle-s6e10-airline-satisfaction](https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction): Oct 5 LB snapshot.
- bansal1600, kaggle-airline-prediction PRs 2 to 7 (search snippets only; pages returned 404): 62 to 97 member LR stacks, nested CV 0.961687 to 0.961890, LB 0.96135 to 0.96161, TabPFN member OOF 0.961053.
- IsThereAnyCode, [playground-s6e9-solution](https://github.com/IsThereAnyCode/playground-s6e9-solution): S6E9 40th, LR-margin XGB and nested stacking lessons.
- [tabpfn on PyPI](https://pypi.org/project/tabpfn/): version 9.1.0, TabPFN-3.5 default, 1M-row limit, license and token notes.
- Kaggle S6E10 [leaderboard](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard), [code](https://www.kaggle.com/competitions/playground-series-s6e10/code) and [discussion](https://www.kaggle.com/competitions/playground-series-s6e10/discussion) pages, through search-result text only.
