# Second-pass interim development review

Reviewed 2026-10-02. **Evidence snapshot: `artifacts/second_pass/blend/current.json`, created 12:01:28.334147 UTC, 11 of 17 primary new configurations completed.** The registered queue was still active. This review loaded saved result/configuration JSON and the compact experiment ledger, not OOF matrices, training labels, audit labels or test predictions. The handoff includes historical v1 audit/public metadata; those values were not used in any comparison or decision here. No training, queue change, model/configuration edit, submission or new evaluation was performed.

## Preliminary decision

`v2_realmlp_cat_raw_aux` is currently the strongest qualifying recipe for the single registered seed-20261021 confirmation. **Do not queue it yet:** the policy requires all 17 primary experiments to complete. Re-read the complete step-0 grid before choosing; a pending primary recipe may supersede this preliminary choice.

The confirmation policy requires fixed-anchor pooled and mean-fold AUC improvements of at least 0.00003, with all fold deltas nonnegative, at a registered alpha. Maximum one confirmation, latest start 15:15 UTC. The current raw+aux recipe qualifies at **all four** registered alphas:

| Alpha | Pooled gain | Mean-fold gain | Fold 0 / 1 / 2 gains |
|---|---:|---:|---|
| .05 | +0.000032853113 | +0.000032840800 | +0.000036109588 / +0.000032313200 / +0.000030099613 |
| .10 | +0.000058978749 | +0.000058958017 | +0.000065379755 / +0.000058235900 / +0.000053258396 |
| .20 | +0.000093452920 | +0.000093545702 | +0.000105881206 / +0.000094450261 / +0.000080305641 |
| .30 | +0.000106591990 | +0.000106904229 | +0.000124640938 / +0.000111458176 / +0.000084613573 |

Raw-only RealMLP also qualifies, but its best step-0 mixture, alpha .30, gains only +0.000070100642 pooled and +0.000072626048 mean-fold. XGB+aux passes the ordinary blend gate but fails the stronger confirmation threshold: its best anchor mixture gains +0.000022830915 pooled and +0.000022110327 mean-fold at alpha .20. No other completed recipe qualifies for confirmation.

If raw+aux remains the strongest qualifier after all 17 primary results, record the choice before fitting, copy its recorded run object, change only the new run ID and model seed to 20261021, and declare the fixed 50/50 seed group before fitting. Keep the four-epoch budget, eight internal members, architecture, categorical twins and original auxiliary feature bank unchanged. First-seed total runtime was 256.359 seconds, selected epochs [3, 3, 3]; the existing 1,200-second cap is conservative. The seed also changes the inner stopping split, so this measures training-plus-stopping variability. Evaluate the equal average through the same anchored gates; never choose the better individual seed. Incomplete groups remain withheld rather than reverting to the fortunate completed seed.

## What the auxiliary features add

The raw+aux versus raw-only comparison is clean at the recorded-configuration level: excluding the run ID and operational timeout, the sole difference is `original_aux=true`. Pooled standalone AUC improves **0.960776968604 -> 0.960852059590**, delta +0.000075090986; mean-fold improves +0.000069634197. Fold deltas are +0.000086538393 / -0.000002551647 / +0.000124915844. Thus the direct representation gain is positive on two folds, with a small decline on fold 1, rather than uniformly positive.

The blend evidence is stronger than that standalone comparison. At the identical alpha .30, raw+aux's anchor mixture exceeds raw-only's by **+0.000036491349 pooled AUC**. Both variants already complement the fixed v1 mixture, and the auxiliary version contributes more under the registered grid. This establishes useful development-set complementarity; it does not prove a generalization improvement or a particular residual-correlation mechanism.

The current two-addition blend is **63% fixed v1 anchor + 27% raw+aux RealMLP + 10% XGB profiles+aux**. Its pooled AUC is **0.961345626656648**, versus anchor **0.961221030185844**: +0.000124596471. Mean-fold improvement is +0.000124688233. Fold AUCs are 0.960879578083 / 0.961288078463 / 0.961881257666, with gains **+0.000141355567 / +0.000130902639 / +0.000101806494**. The second XGB addition adds +0.000018004480 pooled and +0.000017784004 mean-fold over the first accepted mixture, improving all three folds. This explains its inclusion despite a negligible standalone gain and a worse standalone fold 0.

## Matched hypothesis results

All deltas below are absolute AUC changes against the configuration's declared control. They come from saved development metrics, not new inference. “No blend gate” refers to the best fixed-anchor mixture in this snapshot, not a universal failure of the method.

| Changed factor | Pooled delta | Mean-fold delta | Fold 0 / 1 / 2 deltas | Interpretation |
|---|---:|---:|---|---|
| Native RealMLP TE+teacher: add profiles | +0.00003509 | +0.00004458 | +0.00019164 / +0.00001161 / -0.00006952 | Mixed standalone improvement; no blend gate. |
| LightGBM route+teacher: add TE | -0.00056030 | +0.00029674 | +0.00044426 / +0.00025856 / +0.00018741 | All within-fold ranks improve but pooled AUC declines; no blend gate. See scale caveat below. |
| LightGBM TE variant: add profiles | +0.00081558 | +0.00001405 | -0.00001502 / -0.00004180 / +0.00009896 | Most pooled recovery does not correspond to a similar mean-fold gain; no blend gate. |
| CatBoost depth 7 -> 6 | -0.00003038 | -0.00002697 | +0.00010589 / -0.00004256 / -0.00014423 | Weaker on two folds; no blend gate. |
| CatBoost L2 5 -> 20 | +0.00003367 | +0.00003757 | +0.00010337 / +0.00003323 / -0.00002390 | Modest mixed standalone improvement; no blend gate. |
| XGB profiles+teacher: add auxiliary features | +0.00000645 | +0.00000373 | -0.00015413 / +0.00014556 / +0.00001976 | Standalone nearly unchanged; accepted complementary blend addition. |
| Native RealMLP TE+teacher: add auxiliary features | +0.00009533 | +0.00009583 | +0.00011983 / +0.00012995 / +0.00003772 | All-fold standalone improvement; best anchor gain +0.00000904 is below the ordinary gate. |
| Native raw RealMLP: add auxiliary features | +0.00007509 | +0.00006963 | +0.00008654 / -0.00000255 / +0.00012492 | Strongest completed anchor complement and preliminary confirmation choice. |
| LightGBM route+teacher: add auxiliary features | +0.00015234 | +0.00015345 | +0.00013802 / +0.00013431 / +0.00018801 | All-fold standalone improvement; best anchor gain +0.00000779 is below gate. |
| CatBoost route+TE: add second original teacher | -0.00003168 | -0.00003250 | +0.00004074 / -0.00015972 / +0.00002148 | No support for this transfer in the completed matched run. |

Raw-only RealMLP is an explicitly unmatched representation baseline in the ledger. Comparing raw+aux with TE+teacher+aux removes **both** manual TE and the original XGB satisfaction teacher. That bundled comparison cannot identify whether TE or the teacher individually caused the difference. Do not describe it as an isolated “TE removal win.” The auxiliary rating models themselves use original feature data, exclude their own predicted rating from inputs, and do not use synthetic satisfaction labels, as documented by the source-bank implementation and its earlier checks.

## Validation cautions and remaining work

- LightGBM+TE's mean-fold AUC is 0.959757632979 versus pooled 0.958888692339, a **0.000868940640 gap**. Adding profiles reduces that gap to 0.000067407747 while mean-fold AUC changes only +0.000014047461. This is consistent with fold-dependent probability scales disturbing cross-fold ranking, as previously observed for TabM, but the cause is **not established** by these summary statistics. Selected rounds vary [186, 246, 108] without profiles versus [155, 250, 174] with profiles. Do not infer calibration causality, fit post hoc OOF calibration, or switch selection objectives to rescue the candidate. All registered anchor mixtures fail anyway.
- Every one of the 11 completed result objects matched its live exact run object and the frozen development split hash. Declared matched comparisons differed only in the stated factor, apart from run identity/timeout. This check did not rehash prediction files or independently recompute AUC; those remain the release controller's checks. The review used one timestamped saved grid consistently.
- Six primary recipes were pending in this snapshot: native RealMLP second teacher, weight decay .030, learning rate .0265, embedding size 10, XGB TE smoothing 100, and native RealMLP profiles+teacher without TE. Wait for all six before confirmation choice. No extra sweep is justified by this interim review.
- These are **adaptively reused development selection gains** after 30 first-pass and 11 completed second-pass configurations, with more comparisons pending. Positive fold deltas are a useful robustness condition, not independent replications or statistical significance. The seed check does not reset that adaptive exposure. There is no fresh independent v2 audit or public score in these observations.
- OOF routes each passenger to one held-out fold model; fallback test predictions average fold models, and the intended release uses fixed-length full-data refits after freeze. Neither current pooled nor mean-fold OOF directly evaluates that final fitted function. Preserve the frozen gates, equal seed grouping and final native/raw-inference verification rather than claiming an unbiased or public improvement.
