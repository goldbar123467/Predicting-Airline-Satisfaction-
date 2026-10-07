# Second campaign, October 2, 2026

User requested another full local run using agents until 13:00 Eastern. Target 13:00 America/Indianapolis (17:00 UTC), earlier than literal EST. The original release remains immutable. This request supersedes the earlier campaign's stop-exploration instruction for new run IDs only.

## Contract and selection

Use the existing raw hashes, 629,671 development rows, three frozen folds, nested inner stopping, keyed predictions and ROC AUC. The 69,964 original audit rows have already been evaluated. No new candidate or blend may be selected, stopped or calibrated using their labels. New OOF results are development selection estimates, not an independent confirmation. Historical v1 audit and its 69,396-row sensitivity result belong to v1 only. Full-data refits may include all labels after selection is frozen.

Preserve all original run IDs, configurations, models, predictions, REPORT.md and artifacts/final. New experiment IDs begin v2_. New campaign state, logs, blends, release and report live under second_pass paths. One GPU training worker at a time; no paid compute or dependency upgrades. Existing v1 predictions remain the fallback throughout.

Anchor selection to the previously frozen v1 probability ensemble. Consider only completed, verified new candidates; old models retain their internal relative weights. Predeclare alpha grid 0.05, 0.10, 0.20, 0.30, at most three convex additions. Each addition must improve pooled and mean-fold AUC by at least 0.00001, with no fold regression worse than 0.00002. Ties favor smaller weights, then earlier declared run order. Retain the anchor if no addition qualifies. Grouped seeds must be averaged as declared before viewing their outcomes. These stability checks reduce flexibility but do not remove adaptive selection bias.

## Execution stages

1. Immediately test the matched native categorical RealMLP recipe with fold-fitted numeric route profiles. Same seed, architecture, epochs, encoding and teacher as its baseline. First execute a bounded development-only integration smoke and native reload check.
2. Agents refresh exact competition evidence, recommend bounded follow-up hypotheses and review validation. Prioritize distinct feature representations, regularization and source priors over broad seed or epoch searches. Register each batch before fitting and retain negative results.
3. The detached supervisor runs configs/second_pass.json sequentially. The existing 15-minute agent heartbeat reviews second-pass state, evidence and hardware, may diagnose and repair owned failures, and must not select using audit or test labels.
4. Stop launching experiments at 15:45 UTC, stop unfinished exploration by 16:00, freeze the development-only selection, then refit selected new members on all training rows. Reuse checksum-verified v1 full-data member artifacts where the configuration and chosen rounds are unchanged.
5. Before 17:00 UTC, verify native reload, exact keyed schema, finite probabilities, independent all-row blend arithmetic, and all-row inference from the raw test CSV. Write SECOND_PASS_REPORT.md and content-addressed provenance. Pause the monitor after verified completion. If any new package fails, preserve and report the last verified release.

Research may continue between completed batches; do not spend remaining time on arbitrary experiments solely to fill the clock. Any early freeze must document diminishing returns or the remaining validation work. No new public submission is part of this local run plan.

## Registered batches before fitting

The initial ten matched tests are listed in research/second_pass_validation.md: RealMLP profiles; LGB target encoding and then profiles; CatBoost depth6 and L2=20; native RealMLP weight decay .030, learning rate .0265, embeddings10; XGB TE smoothing100; and the RealMLP profile recipe without TE. All retain their named control's unlisted settings.

Fresh exact-competition research at 11:10 UTC supports a second representation batch, registered at approximately11:36 UTC. Build one fixed original-only13-rating multiclass XGB bank, seed0,400rounds,depth6, and a separate fixed original-only1500-round LightGBM satisfaction teacher. Neither bank fits on competition labels. The rating being predicted is excluded from its own inputs; satisfaction is absent from every auxiliary input. Retain audited exact-profile exclusions and disclose remaining near-source ancestry.

Seven downstream tests: add original rating expected values to the existing XGB profile model, native RealMLP TE+teacher, and LGB127 teacher; compare raw native RealMLP with and without rating expected values; add the new LightGBM teacher beside the existing XGB teacher in matched CatBoost and native RealMLP runs. Source-feature tests precede remaining regularization tests. Total planned new configurations:17, plus at most one preregistered seed20261021 confirmation if the documented strong-gain gate is met. No additional sweep is implied.

The auxiliary cache and raw inference use CPU prediction for numerical consistency even though model fitting uses CUDA. Expected-rating reductions use float64 accumulation followed by float32 storage, avoiding batch-dependent float32 reduction changes that could cross downstream tree thresholds. Actual native/cache and downstream saved-model parity must pass before queued source-feature fits begin.
