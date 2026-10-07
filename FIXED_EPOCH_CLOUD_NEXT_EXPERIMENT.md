# Proposed next experiment: sustain dropout during continuation

Status: design proposal only. No new training, cloud job, registration, blend change or submission is authorized by this document. The completed `fixed_epoch_cloud_v1` experiment and all its artifacts remain frozen.

## Question and evidence

Does maintaining the recipe's base dropout rate after epoch 4 reduce the observed overfitting during continuation to epoch 16?

The completed H16 trajectory improved fitting-probe log loss but worsened held-out AUC, log loss and Brier across all three folds. Its recorded dropout fell from approximately 0.00316 at epoch 4 to 0.00158 at epoch 16. These observations motivate a regularization hypothesis; they do not identify dropout as the cause or predict a gain. The proposed 0.05 value is the existing configured base rate, not an optimized value found by scanning these curves.

## One intervention, matched controls

| Arm | Epochs 1–4 | Epochs 5–16 | Duration |
|---|---|---|---:|
| Control | Reproduce the frozen H16 recipe | Original dropout schedule |16|
| Treatment | Identical H16 prefix | Hold the configured dropout value at 0.05 |16|

Keep learning-rate and weight-decay schedules, every decay multiplier, model architecture, optimizer, seed/sub-seed mapping, feature construction, data order, batch size and all other stochastic settings fixed. Preserve the legacy duplicate decay-factor behavior in both arms. A correction to that behavior is a different experiment.

Specify the exact dropout parameter/layer scope in a future registration and verify its per-update effective value on synthetic data. Preserve all unrelated layer multipliers. Apply the intervention only after the fourth epoch has completed; the change in subsequent dropout masks is part of the intervention.

The same prefix must include identical network weights, learned preprocessing, optimizer moments/step counters, RNG states and data-order state. The saved native inference graphs from this run cannot resume training with that full state. Recreate the prefix in a future job and explicitly verify full-state equivalence. Reject a failed match rather than interpreting it as a dropout effect. An in-process branch requires a separate tested full-state capture/restore path; do not silently substitute an inference-model reload.

## Bounded prospective protocol

- Use the same three frozen development folds and the same inner/outer policy. Audit/test data remain excluded. Inner monitoring stays passive.
- A straightforward replay design has 12 new fits: three folds × inner/outer × control/treatment, each executing exactly 16 epochs, for 192 epochs total. Capture only the registered epoch-4 and epoch-16 endpoints needed for the contrasts. Predeclare any additional endpoint before fitting.
- Primary comparisons are treatment E16 minus paired control E16 and treatment E16 minus their common E4 prefix. Duration and learning-rate schedule are identical across the E16 arms.
- Compute outer metrics only after every registered fit, source/partition/prefix check and full-fold native verification passes. No best-epoch restoration, post-hoc checkpoint selection or partial scoring.
- If testing usefulness to the incumbent, preregister only the same 10% candidate weight. Require pooled and mean-fold AUC gains of at least 0.00001 and no fold regression worse than -0.00002 against each of incumbent, control mixture and prefix mixture. Passing advances confirmation only.
- Run the synthetic intervention/full-state smoke before a separately authorized cloud fit. Use one private free GPU worker and a new immutable namespace. Keep the gaming PC free of training.

The last run's adapter work totaled 34.54 minutes for 120 epochs. Linear scaling gives roughly 55 minutes for 192 epochs before setup, imports, export and full-fold inference, but this is only a planning estimate. Re-estimate throughput and remaining quota before fixing a later budget; do not launch based on this estimate alone.

## Interpretation and alternatives

A positive paired result would support this specific sustained-dropout continuation policy. It would not establish a globally optimal dropout, epoch count or blend amount. Failure would reduce support for this intervention, not prove that regularization cannot help. Historically reused development data remain exploratory; an accepted candidate still needs a separate confirmation design.

Earlier learning-rate cooling is a separate later policy test. Changing learning rate also changes the effective decoupled weight-decay dose, so it is less direct for isolating stochastic regularization. Do not combine cooling, dropout, a decay-factor correction and capacity changes in one follow-up.

Evidence: `FIXED_EPOCH_CLOUD_V1_REPORT.md`, the saved evaluation and descriptive curves under `cloud/fixed_epoch_v1/assessment_workspace/artifacts/fixed_epoch_cloud_v1/`, and `research/fixed_epoch_cloud_result_review.md`.
