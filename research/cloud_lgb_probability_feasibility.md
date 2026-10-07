# One cloud CPU LightGBM contrast, 2026-10-02

Decision recorded during the 14:31–14:40 UTC review: **YES to exactly one preregistered matched pair, conditional on a complete, selection-ready package by 15:10 UTC. Otherwise do not launch.** Root set that final readiness deadline and will implement release integration in parallel with the cloud agent. This is not authorization for a diagnostic-only fallback, another architecture, or a parameter search. No fit, upload, dependency installation or source-code edit was performed during this review.

The earlier 20–25-minute engineering estimate was optimistic. The cloud implementation owner estimates 30–45 minutes for a new selection-eligible LightGBM path, because the existing runtime/importer/production helpers assume RealMLP and TorchScript. Root's parallel integration and firm readiness check are essential to feasibility; the existing RealMLP receipt hashes must not be reused for a different bundle.

## Evidence and hypothesis

The following values were read from existing development result JSON files, without loading labels or prediction matrices. Differences were calculated only from those saved scores.

| Matched change | Pooled AUC change | Mean-fold change | Fold changes |
|---|---:|---:|---|
| XGB route/TE/teacher/profiles/EV: add q and log score | +0.0000928375 | +0.0000931481 | +0.0000778278, -0.0000075320, +0.0002091483 |
| CatBoost route/TE/teacher/EV: add q and log score | -0.0000981781 | -0.0001028321 | -0.0000172292, -0.0001623886, -0.0001288786 |
| LightGBM route/teacher: add EV | +0.0001523417 | +0.0001534455 | +0.0001380219, +0.0001343078, +0.0001880069 |

The source control is `artifacts/runs/v2_lgb_route_teacher_aux_127/result.json`: pooled AUC 0.9596013322227334; folds [0.9588736060136374, 0.9597402338746208, 0.960229165459156]; selected rounds [187,278,267]; complete local three-fold nested runtime 45.609 seconds. The prior EV LightGBM's best anchor gain was only about 0.00000779 in the earlier second-pass search, below its gate against that earlier anchor. Thus the expected incremental value is modest, not established.

The hypothesis is a learner-specific transfer of original-source rating-distribution information. Expected values discard distribution shape; observed-category probabilities and their composite log score can provide different split variables. XGB's positive and CatBoost's negative results show that this does not transfer uniformly across tree learners. The remaining LightGBM contrast tests that uncertainty using its existing recipe, not a newly tuned tree.

The method motivation is consistent with LightGBM's existing leaf-wise tree construction and categorical split controls, but neither its paper nor documentation demonstrates a gain from these competition features. Reused the previously reviewed methods/evaluation of [Ke et al., LightGBM, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Paper.html); refreshed the official [4.7.0 parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) and [native Booster API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html) today. The q/log idea remains attributed to the Part4 notebook in `research/third_pass_validation.md`; independently implemented mathematics and existing source banks are reused, not unlicensed notebook code.

## Exact single contrast

Proposed immutable IDs:

- Control: `v3_cloud_lgb_route_teacher_aux_control`.
- Candidate: `v3_cloud_lgb_route_teacher_aux_probability`.

Clone the completed `v2_lgb_route_teacher_aux_127` run without changing its statistical recipe:

```json
{
  "family": "lightgbm",
  "route": true,
  "teacher": true,
  "original_aux": true,
  "seed": 20261003,
  "max_rounds": 2600,
  "patience": 150,
  "timeout_seconds": 600,
  "params": {
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 80,
    "colsample_bytree": 0.85,
    "subsample": 0.9,
    "subsample_freq": 1,
    "reg_lambda": 6.0,
    "cat_smooth": 30.0,
    "cat_l2": 10.0
  }
}
```

Both run on the same private CPU job/image, pinned LightGBM 4.7.0, four CPU threads, with a new cloud execution identity. The candidate adds only `original_aux_probability: true`. No TE, numeric route profiles, categorical twins, new teacher, calibration, altered sampling, or additional seed. Four threads are an explicit shared execution setting; the isolated adapter must replace the existing hardcoded eight-thread constructor rather than pass a duplicate `n_jobs` parameter. Record that source difference before fitting.

Control inputs: 21 raw predictors, existing native route category, unchanged original XGB satisfaction-teacher probability, and 13 fixed original-only expected ratings. Candidate appends 13 q values plus the fixed summed log score. Supported q values remain native and **unclipped**; missing/unseen rating q is 1e-6; only each log argument is clipped using `log(max(q,1e-6))`. Preserve the existing feature order, numeric dtypes, rating-class mappings, missingness and cache checksums. Neither bank is refit.

## Validation and unchanged admission gates

Use exactly the existing 629,671 development IDs and three outer folds. Exclude all 69,964 audit IDs before reading or packaging labels. Package no audit rows or audit predictions. Freeze explicit inner IDs from each outer training set using the existing stratified 90/10 split with seed `20261003 + outer_fold`; both cloud recipes use the identical inner IDs. Fit vocabularies and any learned transform on the applicable inner or outer training rows only. Stop using inner ROC AUC and patience150; refit each outer training partition for its own selected number of rounds, then evaluate its untouched outer fold. No early stopping on outer validation.

The old local LightGBM result is context, not the matched comparator: operating system, libraries, thread count and compiler can change training. Both members of this comparison must actually complete on the same cloud platform. The [official parameters documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#deterministic) explicitly cautions that different systems, versions and binaries can yield different results; do not claim bitwise cross-platform training determinism.

Control is excluded from blend selection. The candidate must first meet the unchanged cloud matched gate: pooled and mean-fold AUC changes each >=0, and no fold change below -0.00002. It then faces the unchanged frozen-v2 anchor grid alpha [.05,.10,.20,.30], pooled and mean-fold gains each >=0.00001, worst fold >=-0.00002, at most two new units in the entire release. Any later submission must independently pass the same incremental thresholds against the best previously submitted development blend. Public scores and exposed audit scores do not select recipes or weights. The existing single global confirmation allowance is not renewed.

Pooled/macro disagreement is already a known LightGBM risk. Report both and all three folds; do not introduce calibration, rank transforms, new stopping rules, or changed gates to rescue an unsuccessful pair. All scores remain adaptively reused development evidence, not a new unbiased audit.

## Runtime, environment and return proof

Observed CPU probe: four CPUs, 31.35 GiB RAM, Python3.12.13, NumPy2.0.2, sklearn1.6.1 and LightGBM4.6.0; tiny generated-data LightGBM fit/save/reload passed in 5.42 seconds with zero prediction difference. That was a synthetic execution check, not full-data throughput or cross-platform proof. Pin **only LightGBM4.7.0** using a no-dependencies install; preserve the remaining cloud environment and record it. [PyPI's official release files](https://pypi.org/project/lightgbm/4.7.0/) provide a 3.5 MB Python3 manylinux x86-64 wheel, SHA256 `d23e922acd891e77212e4d0fbcee9ba973c96dee479491341d05ba595357ebb7`. A small generated-data fit and native return test must execute on the actual prepared adapter before launch.

Planning estimate, not a measured cloud benchmark: about 4–10 minutes for the pair, allowing several times the local45.6-second runtime per recipe for fewer/slower CPU cores, additional features, IO and inference. Hard cap600 seconds per recipe and1200 seconds for the combined CV job. If either recipe does not complete its full three folds, the contrast is incomplete and cannot enter selection. No GPU quota is needed; use only the authorized free private CPU service. Queue latency remains unmeasured and consumes schedule slack.

Required artifacts are native LightGBM text models, transform JSON, exact model/config/feature metadata, explicit fit/stop/heldout ID receipts, complete OOF/test predictions, selected rounds, code and environment hashes, and a closed output-file hash manifest. The importer must reload each returned text model locally, apply its saved transform, reproduce every heldout row and test fold mean, and test unseen categories, one row, reordered rows and a partial batch. Enforce class-one probabilities and exact keyed joins/schema. Fix numerical comparison tolerances before looking at differences; tree float64 predictions should not borrow RealMLP's looser float32 tolerance without justification. Any mismatch blocks admission. Exact source readback or a narrowly proved UTF8 newline-only transport equivalence must bind the dispatched runtime to the manifest.

If selected after freeze, production is a fresh full-data CPU fit in the same pinned cloud image, using the frozen recipe, seed and median of the three selected rounds. Only that post-freeze production phase may consume full official training labels. Return its text model, transform, full-test predictions and freeze-bound receipt; require local native replay and the independent all-row raw-bank/final-blend verification. Do not silently refit the cloud-selected candidate locally. Do not apply LightGBM's leaf-output `Booster.refit` API in place of the project's fresh fixed-round training.

Root's final schedule is authoritative:

| Checkpoint | UTC deadline |
|---|---|
| Complete package, synthetic CPU contract tests, independent source review and release-proof integration ready | **15:10**, else no launch |
| Complete full CV pair | **15:30** |
| Return proof, gates and frozen selection | **15:40** |
| Same-image fixed production complete if selected | **16:00** |
| All-row raw release verification complete | **16:25** |
| User delivery deadline | **17:00** |

This is tighter than the overall15:45 no-new-experiment cutoff and preserves a production margin. A missed readiness milestone is a no-go, not permission to publish diagnostic outputs as release-ready. Current local refit/import work retains RAM priority; prepare the allowlisted package without launching local full-data fits. Existing verified submissions remain the fallback throughout.
