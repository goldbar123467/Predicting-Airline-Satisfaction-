# RealMLP epoch-selection semantics, audited October 2, 2026

**Finding:** the current eight-member RealMLP recipe selects the AUC of its averaged member probabilities. It does not select mean per-member AUC, average eight separately selected epochs, or combine eight member-specific best checkpoints. All eight members use one common selected epoch within each outer fold. This aggregation matches the native inference graph in mathematical form. It does not explain the deterioration of the longer-cap experiments.

This audit used the pinned installed PyTabKit 1.7.3 source, the current adapter, both archived adapter copies, and six inner metadata files. The control and 500-cap archived adapters are byte-identical to the current adapter, SHA256 `70c8b78f77c8f53df7f94e7fc4c92d3e47592973dd942fae20ed3beb2bbe4184`. A four-row mathematical example exercised the installed aggregation and metric functions on CPU without fitting anything. Its inputs, outputs, metadata and source hashes are saved in `artifacts/research_pass_v1/epochs/selection_semantics.json`. No competition predictions or labels were read or scored in this audit. No initial diagnostic, frozen plan, production code or model changed.

## Exact selection path

| Step | Source evidence | Meaning in this recipe |
| --- | --- | --- |
| Configure eight members | `scripts/realmlp_categorical.py:281`; `310–312` | `n_ens=8`, `n_cv=1`, `n_repeats=1`, `n_refit=0`, `use_early_stopping=False`, `use_best_mean_epoch_for_cv=True`, metric `1-auc_ovr`, `ens_av_before_softmax=False`. |
| Define split dimensions | `.venv/Lib/site-packages/pytabkit/models/training/nn_creator.py:71–80`; sklearn `sklearn_base.py:329–352,379–380` | One train/test split object and one train/validation sub-split. These dimensions are separate from eight members. |
| Repeat validation rows | `nn_creator.py:233–237` | The same validation indices are repeated eight times, one per vectorized member. |
| Evaluate ensemble | `training/lightning_modules.py:161–178,290–303` | Eval-mode logits have shape `[8, rows, 2]`. The code reshapes to `[1, 8, rows, 2]`, applies softmax per member, averages over the member dimension, then takes `log(probability + 1e-30)`. Labels are deduplicated with `[::8]`. |
| Compute AUC | `lightning_modules.py:184–197`; `training/metrics.py:386–388,505–543` | The one aggregated prediction matrix is converted back to probabilities, its positive-class column is passed to sklearn ROC AUC, and its AUC is cast to float32 before taking `1 - AUC`. |
| Select best epoch | `lightning_modules.py:204–244` | Errors have shape `[1,1]`. The subsequent mean is over train/validation sub-splits, not over the eight members. Here that mean contains one value. |
| Save a checkpoint | `training/lightning_callbacks.py:112–123,27–35` | At a best mean epoch, `ParamCheckpointer.save` clones the contiguous tensor slice containing all eight members. |
| Restore after fitting | `lightning_callbacks.py:125–140`; `nn_creator.py:198–205` | The callback restores all eight members from the common selected epoch. Disabling early termination does not disable this restoration. |
| Return epoch | `lightning_modules.py:260–265`; sklearn `sklearn_base.py:456–457`; adapter `realmlp_categorical.py:319–320` | The adapter reads one integer from `fit_params_["stop_epoch"]["1-auc_ovr"]`. No mean or rounding across member epochs occurs. |

For member logits `z_m(x)` and eight members, the monitored classifier is

`q(x) = (1/8) * sum_m softmax(z_m(x))`.

The selected epoch is the epoch with the largest computed `AUC(y, q[:,1])`, with the tie policy below. It is not the epoch maximizing `(1/8) * sum_m AUC(y, softmax(z_m)[:,1])`.

The names `best_val_epochs` and `best_indiv_stop_epochs` can be misleading if read without the dimensions. Their “individual” dimension indexes train/validation sub-splits, after the `n_ens` predictions have already been combined. The constructor's `use_best_mean_epoch_for_cv=True` chooses a common epoch across those sub-splits. With only one such sub-split, both mean and individual bookkeeping refer to the same one eight-member ensemble. Changing that flag alone would not create independently stopped members in this configuration.

## What `[3,3,3]` and `[4,4,4]` mean

The three numbers index the three outer development folds in the project training loop. Each number is the selected common epoch of that fold's inner eight-member ensemble. Metadata independently confirms `[3,3,3]` for `v2_realmlp_cat_raw_aux` and `[4,4,4]` for `v3_realmlp_cat_raw_aux_e500`; all six constructors record the selection flags above.

`scripts/train.py:85–90` fits the inner model, obtains this integer, discards the inner model, and trains a new outer model on all outer-training rows. `scripts/common.py:186,219–223` sets the new model's `epochs` to that integer. Therefore the control's outer ensembles each train for three epochs; the 500-cap run's outer ensembles each train for four epochs. Each outer ensemble again has eight members. There is no member-specific epoch mixture and no averaging of the three inner checkpoints.

The separately executed final-release procedure has another, clearly separate aggregation: `scripts/third_pass_release.py:429–438` and `scripts/second_pass_release.py:248–257` use `max(1, int(np.median(result['rounds'])))` to set the fixed full-data refit duration for a selected recipe. The median is over the three outer-fold epoch decisions. For three integer entries it is already an integer. It is neither rounding an average of eight member epochs nor selecting a fresh epoch on full-data labels. This rule does not imply that the rejected 500-cap recipe received a final release refit.

## Tie handling and numerical precision

`lightning_modules.py:207,223–226,235–243` defaults `use_last_best_epoch=True` and compares errors using `<=`. If two stored metric values are equal, the later epoch replaces the earlier checkpoint. No epsilon or patience threshold is applied to this comparison. The “latest best” behavior applies even though early termination is disabled.

There is a numerical qualification: `metrics.py:541–543` casts sklearn's AUC result to float32. Around AUC 0.96 the float32 spacing is `5.960464477539063e-8`. The synthetic values `0.96085201` and `0.96085202` both become the same stored error `0.03914797306060791`. An extremely small improvement or deterioration can therefore be a tie at the library's comparison precision. The six-decimal logging format at `lightning_modules.py:201` is a further display limitation, not the comparison precision.

No saved per-epoch RealMLP curves exist, so this audit cannot establish that ties occurred in the actual control or 500-cap searches, or that they affected the selected epoch. This is a small reproducibility/telemetry issue, not evidence for the observed loss of AUC and not a reason to alter historical results.

## Inference aggregation and the remaining objective distinction

`TabNNModule.predict_step` uses the same `_postprocess_ens_pred` function (`lightning_modules.py:285–303`). The sklearn wrapper then applies softmax and averages its remaining prediction-group dimension (`sklearn_base.py:502–506`). In this recipe that remaining dimension has size one. The saved production `_MixedGraph.forward` directly averages the eight member probabilities (`scripts/realmlp_categorical.py:84–88`). The adapter verifies exported/native prediction parity on several probe shapes and unknown categories before saving (`realmlp_categorical.py:342–365`).

In exact arithmetic, softmax of `log(q)` recovers `q`. Finite-precision log/softmax round trips can introduce tiny differences. The executed four-row example measured maximum probability differences of `1.49e-8` and `2.98e-8`; neither changed its AUC. This does not prove exact ranking equality for every possible nearly tied real prediction. It also does not constitute a different mean-member selection objective or evidence that precision caused the historical result. Preserve the established native-export tolerances and log the monitored probabilities' dtype.

The important remaining distinction is **standalone ensemble AUC versus marginal value in a larger blend**. Inner selection maximizes the eight-member RealMLP ensemble's own AUC. It does not maximize the AUC of `0.90*incumbent + 0.10*RealMLP`. Even a perfectly implemented standalone selection criterion need not select the epoch that adds most useful errors to another model. Current artifacts cannot reconstruct whether later checkpoints would have been more complementary, because their predictions were not retained. This supports the proposed future fixed-endpoint mixture comparison; it does not justify retrospectively selecting or promoting a failed historical candidate.

One API trap matters for future telemetry: `predict_proba_ensemble` in `sklearn_base.py:508–513` omits the last averaging across prediction groups, but its input has already passed through `_postprocess_ens_pred`. With this adapter's `n_ens=8`, `n_cv=1`, it therefore does **not** expose eight raw member probability matrices. To log individual member AUC or disagreement, capture the model output before `_postprocess_ens_pred`, apply softmax per member, and explicitly assert `[8, rows, 2]`. Treat those member metrics as diagnostics, not a new selection rule.

## Executed mathematical counterexample

Labels are `[0,0,1,1]`. These deliberately constructed probabilities illustrate the distinction without modeling any competition data:

| Synthetic epoch | Member predictions | Mean member AUC | AUC of mean probabilities | Installed selection error |
| --- | --- | ---: | ---: | ---: |
| A | All eight: `[.1,.8,.7,.9]` | 0.75 | 0.75 | 0.25 |
| B | Four: `[.1,.2,.8,.9]`; four: `[.6,.7,.4,.5]` | 0.50 | 1.00 | 0.00 |

An average-member-AUC criterion would choose A. The installed code chooses B, consistent with the actual ensemble prediction rule. This example directly called the installed `_postprocess_ens_pred` and `Metrics.apply` functions on generated CPU tensors; no estimator was fit. It proves the mechanisms differ in general and confirms which one this recipe uses. It is not an empirical explanation of the competition's duration behavior.

## Implications for the fixed A/B/C protocol

1. Preserve probability averaging across eight members and the same member initialization/seed policy. There is no discovered aggregation bug to fix before the experiment.
2. Keep A=H4/e4, B=H16/e4 and C=H16/e16 as literal fixed endpoints. Disable best-checkpoint restoration; `use_early_stopping=False` is insufficient. Inner curves remain monitor-only, so float32 tie selection cannot change those endpoints.
3. Log ensemble AUC separately from optional mean member AUC. Record raw sklearn AUC at full precision plus the library comparison value when reproducing historical selection mechanics. Do not silently substitute one for the other.
4. Compare the predeclared 90/10 mixtures at A/B/C to assess complementarity directly. Do not add per-member weights, member-specific stopping or a new epoch search to this experiment.
5. Continue prioritizing the already verified schedule-horizon/refit mismatch and missing telemetry. This audit rules out one plausible aggregation explanation; it does not identify a new causal reason that more epochs failed.
