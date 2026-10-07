# Duration diagnosis and a testable next experiment

Prepared October 2, 2026 UTC for the bounded deep research pass. This is an analysis of existing development predictions and training code, plus synthetic CPU mechanism tests. No real-data model was fitted, no GPU was allocated, no exposed audit was scored, no test prediction was selected, and no release or submission was changed. The completed saved-checkpoint blend search was not repeated.

The direct conclusion is that **another increase of `max_rounds` is not justified**. The 500-cap recipe completed its full inner searches, selected four epochs, and then deployed newly initialized four-epoch outer models. Changing the cap changes three normalized schedules. Refitting compresses those schedules again. This is evidence against the registered selection/refit recipe, not a clean causal test of taking the same neural trajectory from epoch 4 to epoch 500.

An independently reproduced upstream optimizer discrepancy also changes effective decay by parameter group. It is shared by successful and unsuccessful recipes, so it cannot by itself explain the observed score regression. Keep it separate from a duration experiment.

## Executed evidence and provenance

`scripts/analysis_epoch_diagnosis_v1.py` reads only split IDs/folds, development OOF, result JSON and model metadata. It first restricts split identities to `fold >= 0`, then requires exact sorted IDs/folds, unique IDs, aligned labels and valid probabilities. All eight OOF scores below were recomputed to an absolute tolerance of 1e-12 on 629,671 development rows. Recorded source/input hashes, per-fold scores, log loss, Brier score and metadata are in `artifacts/research_pass_v1/epochs/diagnosis.json`. These remain adaptively reused development estimates.

| Saved recipe | Inner horizon | Selected/outer epochs, folds 0/1/2 | Pooled OOF AUC | Mean fold AUC | Pipeline seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| Local raw + auxiliary control | 4 | 3 / 3 / 3 | 0.9608520596 | 0.9608586157 | 256.359 |
| Local raw + auxiliary | 60 | 3 / 4 / 4 | 0.9607842682 | 0.9607933886 | 1,852.125 |
| Local raw + auxiliary | 500 | 4 / 4 / 4 | 0.9607352453 | 0.9607474591 | 13,688.797 |
| Cloud raw + auxiliary control | 4 | 3 / 3 / 4 | 0.9608107753 | 0.9608211586 | 633.617 |
| Cloud raw + auxiliary | 12 | 4 / 3 / 4 | 0.9607459889 | 0.9607653150 | 970.262 |
| Cloud raw + auxiliary | 60 | 4 / 4 / 4 | 0.9606847043 | 0.9606917029 | 3,568.372 |
| Numerical TE + teacher control | 4 | 4 / 4 / 4 | 0.9606333241 | 0.9606441513 | 241.344 |
| Numerical TE + teacher | 16 | 5 / 6 / 5 | 0.9603704701 | 0.9603856441 | 539.203 |

The eight exact run IDs, in table order, are `v2_realmlp_cat_raw_aux`, `v3_realmlp_cat_raw_aux_e60`, `v3_realmlp_cat_raw_aux_e500`, `v3_cloud_realmlp_raw_aux_e4`, `v3_cloud_realmlp_raw_aux_e12`, `v3_cloud_realmlp_raw_aux_e60`, `realmlp_te_teacher`, and `realmlp_te_teacher16`.

The numerical 16-cap result is also negative. Its pooled delta is -0.0002628540, with fold deltas -0.0001818762 / -0.0004643479 / -0.0001292975. It is a different representation and cannot be combined with the raw-auxiliary comparisons as independent repetitions of one treatment.

For local raw-auxiliary 500 versus control, the pooled delta is -0.0001168143 and the fold deltas are -0.0001370308 / -0.0000468957 / -0.0001495433. Log loss also worsens, 0.2230935325 to 0.2236119660, and Brier score worsens, 0.0596294437 to 0.0597228642. This is not merely a pooled-AUC probability-scale reversal: mean within-fold AUC drops by 0.0001111566 and every fold drops.

The training stderr explicitly records `max_epochs=500` three times and `max_epochs=4` three times. These are at lines 10/103/196 and 59/152/245 of `logs/third_pass_batch07/supervisor/20261002T165140543028Z_v3_realmlp_cat_raw_aux_e500.stderr.log`. The long fit did run. Its 13,528.983 seconds of inner fitting consumed about 98.8% of total pipeline time; the three fresh outer fits took 131.890 seconds in total. The pipeline cost was about 53.4 times the short control for a worse observed development score.

Source comparison of the control and 500-cap stored snapshots found identical neural adapters, categorical transformer, auxiliary feature implementation and dependency lock. The `train.py`/`common.py` differences add feature branches that are inactive for both recipes. All six corresponding inner/outer `transform.json` files are byte-identical. Hashes and the exact two source diffs are in `source_comparison.json` beside the diagnosis. This supports a matched recipe comparison, while allowing the documented schedule and stopping differences.

## What the code actually does

The relevant source chain is:

1. `scripts/train.py:85` makes a stratified inner stopping split from each outer training partition. Outer validation labels do not select the epoch.
2. `scripts/train.py:87` fits the inner model; line 88 discards that model; line 90 fits a fresh outer model for `rounds=n`.
3. `scripts/common.py:186` sets `limit = rounds or max_rounds`; lines 219–222 pass `epochs=limit` to the categorical adapter.
4. `scripts/realmlp_categorical.py:283–289` configures `flat_anneal` LR, `cos_log_15` weight decay and `invsqrtp1e-3` dropout. The `epochs` alias at line 293 becomes `n_epochs`.
5. `scripts/realmlp_categorical.py:310–312` disables early termination while requesting the best inner AUC epoch. Lines 319–320 read the selected epoch after the full fit. Metadata is saved at lines 369–383.

Thus `use_early_stopping=False` means the inner run traverses the horizon, not that its last epoch is deployed. The installed library selects the best probability-averaged eight-member ensemble by `1-auc_ovr`, then restores its parameters. `lightning_modules.py:290–303` averages member probabilities before validation; `metrics.py:386–388` computes one minus sklearn AUC. This review found no metric-direction reversal or per-member-versus-ensemble selection error.

At the installed version, `NNCreator` defaults classification fitting to cross entropy (`nn_creator.py:88–102`). Better training cross entropy does not mathematically imply better validation AUC. AUC scores positive-negative ordering, while cross entropy also rewards confidence and calibration. For illustration, labels `[0,0,1,1]` with predictions `[.2,.6,.61,.7]` have perfect AUC; changing to `[.1,.51,.5,.9]` lowers log loss but introduces a ranking error. This illustration is not a claim about the unsaved RealMLP learning curves.

The published [RealMLP method](https://arxiv.org/html/2407.04491v3) uses a different multi-cycle LR recipe and a 256-epoch horizon across heterogeneous benchmark datasets. Its horizon is a benchmark setting, not evidence that this adapted competition recipe should use 256 or 500 epochs. Methods and best-epoch selection in sections 3 and B.3 were inspected; the local installed code is the authority for the executed behavior.

## The schedules differ from the first epochs

Installed `scheduling.py:15–16` defines progress as `epoch_float / max_epochs`; `TimeSchedule.update` at lines 78–79 uses that value. `flat_anneal` at lines 378–379 keeps LR constant until 60% progress, then anneals. The cosine-log weight-decay formula is at lines 407–408; dropout is at lines 460–461. Actual values were evaluated from the installed module, not approximated from a schedule name. The complete table is `schedule_boundaries.csv` beside the diagnosis JSON.

| Horizon / completed-epoch boundary | Base LR × schedule | Base WD × schedule | Dropout probability |
| --- | ---: | ---: | ---: |
| 4 / 1 | 0.0530000 | 0.00740234 | 0.00315597 |
| 500 / 1 | 0.0530000 | 0.00026762 | 0.02886751 |
| 4 / 3 | 0.0366413 | 0.01313487 | 0.00182453 |
| 500 / 3 | 0.0530000 | 0.00217436 | 0.01889822 |
| 4 / 4 | 0.00000053 | 0.00000000 | 0.00158035 |
| 500 / 4 | 0.0530000 | 0.00362147 | 0.01666667 |

These are mathematical boundary values. The last minibatch update occurs slightly before the final boundary. Per-parameter LR/WD factors still apply, as discussed below. Label smoothing is zero for these matched runs, so its nominal schedule contributes no change.

The 500-cap LR begins annealing around epoch 300. Its selected epoch 4 is at only 0.8% of the planned schedule. The outer four-epoch refit reaches the end of its newly compressed schedule. Its dropout, LR and decay history differ from the selected inner checkpoint. Even the short control selects three of four epochs, then compresses the outer schedule from four to three.

Repeated recorded outer recipes show nonzero variability. In local 60-versus-500 folds 1 and 2, both outer constructors are exactly equal and both run four epochs, yet AUC differs by -0.0000112738 and +0.0000108084. Mean absolute probability differences are 0.00256945 and 0.00250631. This is variability under the same recorded recipe, consistent with the explicit lack of GPU bitwise determinism. It is not a standalone estimate of seed variance or proof that all differences come from nondeterminism. The three outer training populations overlap and are not independent repetitions.

## An epoch is already many updates, and refit changes the clock

The library uses `drop_last=True` (`nn_creator.py:234`). At batch size 256, each 377,802-row inner fit performs 1,475 vectorized optimizer updates per epoch. One update trains all eight ensemble members in parallel. Counts should not be multiplied by eight to estimate serial wall time.

Each 500-epoch inner fit therefore performs 737,500 updates. Its selected four-epoch prefix comprises 5,900 updates, not four parameter changes. Fresh outer training has 419,780 or 419,781 rows, hence 1,639 updates per epoch and 6,556 updates for four epochs. The larger refit set gives 11.1186% more updates for the same epoch count.

Two defensible future contracts answer different questions:

- **Epoch clock:** retain the original schedule denominator and train the selected number of complete passes. The schedule fraction at epoch four remains 4/500, but the outer fit makes more updates. This preserves data-pass progress.
- **Update clock:** retain the original schedule's absolute update denominator and stop at the selected number of updates, 5,900 here. On the outer data this stops at 3.59975595 epochs. It preserves optimizer exposure but makes fewer passes over the larger dataset.

Neither preserves the original model when the fitting population and learned preprocessing change. Choose and record the clock before comparing results; neither is inherently the best statistical choice. A full-data refit creates a further population-size change that must follow the same declared policy.

## Synthetic proof of a viable fix to the experiment contract

New isolated files `scripts/research_epoch_control_v1.py` and `scripts/test_research_epoch_control_v1.py` exercise the installed `StopAtEpochsCallback`, which the library's own refit path uses while retaining `n_epochs`. They do not modify the production adapter.

On 256 generated CPU rows with a tiny network and eight updates per epoch:

- An uninterrupted horizon-12 fit ran 12 epochs and captured its state/probabilities after epoch 4.
- A horizon-12 fit stopped at epoch 4 using the installed refit callback, with `fit_params=[{"stop_epoch":{"1-auc_ovr":4}}]`.
- Both the model parameter/buffer state and probabilities matched the uninterrupted prefix exactly, maximum absolute difference 0.0. All 32 prefix schedule entries were identical.
- A fresh `n_epochs=4` fit made the same 32 updates but had a different schedule and state. Maximum parameter difference was 0.32241073 and probability difference 0.24951652 on this toy problem.

Four tests passed in 0.633 seconds after imports, 7.5 seconds for the process. All three new/analysis files compile. `artifacts/research_pass_v1/epochs/control_prototype/registry.json` records the synthetic plan before execution; `verification.json` contains results, complete schedule traces and source hashes.

This establishes a working mechanism for horizon-preserving refitting. It does not establish a competition-score improvement or GPU equivalence. The callback stores model parameters/buffers, not optimizer moments, sampler and random states. The saved production TorchScript graphs are inference artifacts; they cannot supply exact continuation or missing per-epoch predictions. Continuing from the end of the current four-epoch `flat_anneal` trajectory also requires a declared LR-tail/restart intervention because its LR has already annealed close to zero.

An additional isolated prototype tests exact **epoch-boundary resumption**, not just stopping. `research_epoch_resume_v1.py` uses the same toy and saves a Lightning checkpoint after epoch 4 of a horizon-12 run. Ordinary checkpoint restoration fails with `KeyError('__dict__')`: the installed optimizer wrapper's inherited `load_state_dict` calls its pickle-specific `__setstate__` with the wrong dictionary structure. Its ordinary optimizer state also contains zero entries, while the underlying Adam in `wrapper.opt` has 13 parameter states. No installed file was patched.

The research subclass explicitly saves underlying Adam moments, pytabkit progress, CPU/NumPy/Python RNG state and static preprocessing. At load it bypasses only the unusable empty wrapper-state entry in the in-memory checkpoint, then restores the real states. After one bounded repair to record the expected native error and bypass that wrapper path, the acceptance test passed: resumed next minibatch and next schedule matched, and next-update parameters, final epoch-12 parameters, probabilities and Adam tensors all differed from uninterrupted execution by exactly 0.0. The test ran in 0.538 seconds after imports, 6.3 seconds for the process. Self-generated checkpoints, hashes and measured failure/success are in `epochs/resume_prototype/verification.json`.

This is a tested research mechanism for deterministic CPU epoch boundaries. It does not cover a partial minibatch/epoch, an already-prefetched loader permutation, validation checkpoint-selection state, GPU determinism or cross-platform replay. The toy reconstructs the exact synthetic data/preprocessor contract before restoring state. A production implementation must restore the saved preprocessing before constructing transformed data loaders and must explicitly support every used callback/generator. Merely adding `trainer.save_checkpoint` to the current adapter would not establish resumability.

### Native export during a continuing trajectory

The existing categorical adapter's export is terminal. It moves the fitted estimator to CPU, wraps the fitted modules by reference and rewrites categorical inference methods. Calling that path on live C at epoch 4 could change device, module modes or methods before the next update. A file hash proving B stayed fixed would not detect such interference with C.

`research_epoch_capture_v1.py` now provides an isolated executable alternative. Two deterministic synthetic CPU trajectories run the same horizon-12 configuration. At epoch 4, one makes separately owned copies of the network, static preprocessing and schema, then uses the actual production `_fit_schema`, `_InputSplit`, `_MixedGraph` and `_prepare_portable_inference` helpers to trace, save and reload B. Raw synthetic categorical values round-trip through the actual vocabulary mapping to the original native codes. The other trajectory performs no export. Neither trajectory uses the production fitting adapter.

The first execution passed. Clone/live parameter and buffer storage was disjoint. All ten live-state fingerprints were unchanged across export: parameters/buffers, static preprocessing, underlying Adam, CPU/NumPy/Python RNG, progress, schedules, and network/static modes and forward methods. All continued minibatches, RNG and schedules matched the control. Final parameters, probabilities and optimizer tensors differed by exactly 0.0. Portable rewrites covered one one-hot layer, one embedding layer and three encoding layers; native versus rewritten/reloaded, single-row and reversed-batch prediction differences were all 0.0. B's artifact hash stayed identical after C continued, while C's parameters changed from B by maximum absolute 1.073218. The acceptance test took 0.945 seconds after imports, 6.5 seconds for the process. Evidence is `epochs/capture_prototype/verification.json`.

This establishes clone-based capture and actual export-helper mechanics for a one-member tiny CPU network. It does not establish the future adapter's eight-member GPU integration, monitored-validation path, full real-data preprocessing contract or resumability of the inference artifact. This toy's static `state_dict` is empty; checking its fingerprint does not test copying learned production static statistics. A future implementation must clone all fitted tensors and preprocessing before device/mode/portable-method changes, preserve every active RNG, verify disjoint ownership and before/after state, and compare continued training against a no-export control on a bounded synthetic full-path smoke. The separate `research_passive_monitor_v1.py` test covers a real synthetic validation loader with eight members: forced nonmonotonic metric values establish that `use_best_epoch=False` preserves the literal endpoint, and added eval-mode probes preserve its continued trajectory. These complementary tests must not be described as one already integrated production implementation.

## Verified decay semantics, with a strict causal limit

The primary [upstream issue 40](https://github.com/dholzmueller/pytabkit/issues/40) identifies a duplicated parameter-factor application. Local `optimizers.py:31–35` applies the factors in `get_hyper_values`; lines 55–62 apply them again in manual weight decay. With scheduled base LR eta, WD lambda, LR factor a and WD factor b, current decay is `theta *= 1 - eta * lambda * (a*b)^2`. A single-factor version would use `1 - eta * lambda * a*b`.

The diagnostic compiled the frozen constructor with inherited defaults on 512 generated CPU rows, retaining the 8-member architecture, real feature cardinalities and hidden sizes. It inspected 19 parameter groups and 2,967,280 parameters, then ran six independent zero-gradient scalar checks. Every observed one-step decay matched the installed double-factor formula within 1e-15. No update was applied to the compiled architecture and no trained model was saved. Full evidence is `synthetic_optimizer.json`.

| Active parameter group | LR factor | WD factor | Current decay / single-factor decay |
| --- | ---: | ---: | ---: |
| First-layer weight | 0.25 | 1 | 0.25 |
| PBLD numerical embedding, including its biases | 0.1151 | 1 | 0.1151 |
| Parameterized activation | 0.1 | 1 | 0.1 |
| Ordinary hidden/output weights and categorical embedding | 1 | 1 | 1 |
| Ordinary affine bias | 0.025 or 0.1 | 0 | Both zero |

A literal correction increases the affected decay coefficients by 4, about 8.69, and 10 times. PBLD has WD factor 1 in the actual inherited defaults, not zero. Front scaling is disabled, so a large scale-layer factor from the generic upstream example does not apply here.

For a four-epoch outer refit, the schedule-integrated **hypothetical zero-gradient** first-layer retention is 0.8690 under current semantics, versus 0.5703 with factors applied once. For its first four epochs under a 500-epoch inner schedule, current first-layer retention is 0.9752. These are mathematical shrinkage diagnostics, not measured weight norms, evidence of collapse, or estimates of accuracy gains. Data gradients and Adam state are omitted by design.

There is another metadata caveat: the adapter requests `plr_act_name='gelu'`, but inherited `num_emb_type='pbld'` makes `models/nn_models/models.py:200–204` override that activation to `linear`. Constructor kwargs alone are therefore not a complete description of the compiled architecture. Future records should include resolved factory settings and parameter groups. This shared behavior is not evidence for why one duration cap scored worse.

## Ranked hypotheses and bounded next tests

| Rank | Hypothesis | Current support | What would distinguish it |
| --- | --- | --- | --- |
| 1 | Inner selection chooses a useful schedule prefix but outer compression changes it materially | Direct source/metadata evidence; synthetic exact-prefix test passes | Paired short outer refits with original horizon retained versus compressed, same fixed epoch decisions and recipe |
| 2 | The current representation/regularization reaches its best ranking early | All raw-auxiliary inner selections are at 3–4 epochs; multiple duration recipes lose OOF | Persist train loss, inner AUC and log loss across a controlled trajectory; require a later checkpoint or fixed temporal average to improve outside selection |
| 3 | Extra updates and changed regularization dose matter more than nominal epochs | Exact schedule/update accounting; no causal data ablation | Freeze step or epoch clock, vary only stop budget; hold optimizer semantics and schedules fixed |
| 4 | The historical decay semantics are suboptimal for this recipe | Confirmed implementation discrepancy; no model-quality evidence | Separate versioned once-factor versus historical-factor ablation, after clock semantics are controlled |
| 5 | Temporal averaging would retain useful diversity beyond the eight members | General method rationale only; no saved checkpoint sequence | One prospective predefined averaging policy, with strict held-out marginal ensemble evaluation |

The synthetic refit-contract proof is already complete. The **minimal** next duration experiment has three predetermined observations per outer fold and needs six fits. The integrated recommendation adds monitor-only inner trajectories to recover the missing diagnostic curves, as specified in `research/next_epoch_blend_experiment.md`.

| Arm | Schedule horizon | Fixed observation | Purpose |
| --- | ---: | ---: | --- |
| A | 4 epochs | Epoch 4 | Same-duration short-schedule reference |
| B | 16 epochs | Epoch 4 | Longer planned schedule, same amount of training |
| C | 16 epochs | Epoch 16 | Continue the same physical B trajectory through its annealing stage |

Run one A fit and one C fit on each of the existing three outer training partitions. Save B from C's epoch-4 prefix. Fix preprocessing, architecture, data rows, model seed, minibatch size and historical optimizer semantics. No inner selection is needed because the observation epochs are fixed before training. Both observations are scored only after those choices are frozen; do not turn the outer fold into an adaptive checkpoint monitor.

`C - B` measures additional training along one common schedule, including its later annealing. `B - A` isolates the planned schedule horizon at equal duration. The current incumbent pipeline selected three epochs, so A4 is **not** its exact control. Retain the frozen incumbent as a separate deployment-quality benchmark, with its historical adaptive-use and run-variability limitations disclosed.

This is 3 × (16 + 4) = 60 outer epochs, six fits and nine checkpoint observations. Three early observations have no additional fitting cost. With the observed roughly 44 seconds per four-epoch outer fit, raw fitting is approximately 11 minutes on the same local setup, plus preprocessing, checkpoint inference and verification. Use a maximum 45-minute local GPU budget if the user authorizes a future run. Measure the first fold and abort only for resource/correctness problems; stopping on negative early scores would fail to test the predeclared late annealing. This estimate is not a runtime guarantee. No such real-data training ran in this research pass.

The integrated plan performs the same A/B/C structure on each existing inner training/monitor split before the outer fits. Those six additional inner fits produce monitored train-loss and held-out curves without choosing epochs or changing the outer recipe. Total cost is 12 fits and 120 epochs, about 114 outer-training epoch equivalents. Its conservative budget is 90 fitting minutes plus 30 for verification, superseding the 45-minute estimate above when that full plan is chosen. All endpoints stay fixed. Disable best-validation-weight restoration or capture literal endpoint states before restoration; `use_early_stopping=False` alone does not prevent the library from restoring a previous best model. Record an epoch-end fixed training probe in evaluation mode in addition to minibatch training loss, because changing dropout makes the training-mode objective a poor direct comparator to evaluation-mode monitor loss.

Keep a second **refit-policy** experiment separate. If the specific goal is diagnosing the historical 500-cap selection/refit transfer, freeze its existing inner choices `[4,4,4]` and compare outer stop-4 with horizon 4 versus outer stop-4 with horizon 500. Six short outer fits would cost approximately 4–5 minutes of raw fitting plus overhead. The original inner searches need not be rerun just to test this interpretation. Both arms still train only four epochs; the test cannot show that training 500 epochs helps. It is optional after the cleaner fixed-duration study, unless preserving the existing early-selection pipeline is the immediate engineering objective. A second predetermined seed can confirm a promising direction, using fixed averaging and fixed stopping populations rather than choosing the better seed.

For either experiment, register these acceptance conditions before fitting:

1. Correctness: synthetic CPU trace/state parity passes; native serialization and probability/ID checks pass; actual horizon, stop, update count and active parameter factors match the plan. Never silently switch epoch and update clocks.
2. Model quality: compare identical development OOF rows and report pooled AUC, macro-fold AUC, per-fold deltas, log loss and cost for all three arms. These are mechanistic observations. Historical runs that failed their standalone admission safeguards remain failed. The proposed study does not retroactively change those gates.
3. Ensemble value: the integrated future protocol freezes exactly `0.90*incumbent + 0.10*arm` for A/B/C. C advances to confirmation only if its mixture improves both pooled and macro-fold AUC by at least 0.00001 over the incumbent and over each A/B mixture, with no fold losing more than 0.00002 against any comparator. This is a new explicitly prospective gate, not standalone admission or submission authority. No alpha grid or internal incumbent reweighting occurs. Confidence intervals conditional on saved predictions are not uncertainty for the entire adaptive training process. Any later seed confirmation uses fixed averaging, not selection of the better seed.
4. Scope: one challenger family/recipe at a time; no 500-epoch escalation, optimizer correction, new representation and blending-method change bundled together. No original audit reuse. These are proposed future fits, not authorization created by this report.

Required telemetry is small compared with training: epoch/update/wall time, schedule denominator and current fraction, base and group-effective LR/WD/dropout, train loss, inner AUC/log loss, selected epoch, checkpoint hash, seed and data/split/source versions. Save optimizer, scheduler, random states and data-loader progress only if exact resumption is required. Save selected and predeclared checkpoint predictions if temporal ensembling is part of the experiment. Current RealMLP metadata has no scalar history; later conclusions must not reconstruct fictitious curves from selected epochs alone.

The separate TabM histories do show classical deterioration: native TabM's best-to-last inner AUC changes are -0.00146677 / -0.00189917 / -0.00172696 while training loss falls by 0.01358 / 0.01638 / 0.01530. These persisted curves justify an overfitting diagnosis for those TabM runs. They do not establish the missing RealMLP curves or imply that all neural families need the same remedy.

Reproduction commands, using the project environment:

```powershell
.venv/Scripts/python.exe scripts/analysis_epoch_diagnosis_v1.py
.venv/Scripts/python.exe scripts/analysis_epoch_diagnosis_v1.py --synthetic-optimizer-only
.venv/Scripts/python.exe scripts/test_research_epoch_control_v1.py
.venv/Scripts/python.exe scripts/test_research_epoch_resume_v1.py
.venv/Scripts/python.exe scripts/test_research_epoch_capture_v1.py
```

The first command performs read-only saved-prediction diagnostics. The second initializes only a synthetic CPU architecture and executes scalar decay checks. The last three perform only toy CPU fits and contract tests. Outputs are isolated under `artifacts/research_pass_v1/epochs`; existing training configurations, model files, release predictions and submission state remain unchanged.
