# Active second campaign handoff

## Campaign and cloud controls complete

The local blended release and all five private Kaggle controls are verified. No project-owned training or cloud session remains active. CLOUD_REPORT.md records the84.489second full-fold RealMLP fit/export and successful returned-model local inference. GPU inference timing remains unsuitable for steady-state comparisons, as documented. All run queues are closed; the15-minute monitor can now be paused. No v2 submission has been made, and the frozen release must remain unchanged.

## Verified local completion,13:00:32UTC

The second-pass supervisor is COMPLETE. All18 registered new configurations finished; the final13-member blend is saved at artifacts/second_pass/final/submission.csv with299844 rows, SHA256571479870a1b57a7a521d639f57f4751f0bfc358031f29ad384696a651be5efc. Full refits of the three new members used699635 rows only after freeze: raw_aux3epochs, seedconfirmation4epochs, XGBaux557rounds. Original10member artifacts were reused with checksum validation.

artifacts/second_pass/verification.json confirms exact keyed schema/order/bounds, independent all-row blend recomputation and all299844-row native inference from raw CSV. Maximum raw-inference difference8.7618999944e-8; audit_evaluated=false. SECOND_PASS_REPORT.md and final/release_provenance.json are published. The original v1 submission and split hashes remain unchanged. No v2 submission was made.

Do not rerun release or reopen selection. Remaining work is solely the private cloud fold0 timing control in CLOUD_HANDOFF.md. Keep the monitor active until its output is retrieved/verified or its failure is explicitly resolved/reported, then pause the monitor. The local deliverable is already verified before the17UTC deadline.

## Current release transition,12:47:54UTC

All17 primary configurations and the single seed20261021 confirmation have completed successfully. The preregistered equal-average raw+aux RealMLP group has27% total blend weight,13.5% perseed; XGBprofiles+aux10%; frozenv1anchor63%. Current development AUC0.9613637193250886, gain0.0001426891392446; foldgains0.0001677687330677 /0.0001450540743823 /0.0001156659878575. No remaining third addition passes the fixed gates.

Root recorded state/second_pass/early_freeze_decision.json before changing only top-level stop_new_runs_utc to12:47:54UTC. All48 runobjects are unchanged. This early freeze follows the plan's diminishing-return exception after the entire bounded campaign finished, with extra time reserved for fixed-length refits and raw-inference verification. The supervisor owns final_blend/refit/verify; do not duplicate those jobs or reopen selection. The previous timing/queue snapshot below is historical. Finaldeadline remains17UTC.

Free private Kaggle CPU/GPU controls allpassed; PyTabKit1.7.3 installation and native RealMLP compatibility are verified. See CLOUD_HANDOFF.md and state/kaggle_cloud/run_state.json. An agent is preparing the preregistered existing-recipe full-fold timing control, outside selection, with1800second cap. Keep15-minute monitor active until local release is verified and any cloudcontrol has completed/cancelled. No newcompetition submission.

Updated approximately2026-10-02 11:46 UTC / 07:46 EDT. Finish by13:00 America/Indianapolis today,17:00 UTC. Read SECOND_PASS_PLAN.md and current state before acting; values below are a handoff snapshot, not a live status claim.

## Running and scheduled

- Detached supervisor launched11:39:39 UTC. Launcher16204, actual supervisor26376; always revalidate PID, creation time and command before any intervention. State: state/second_pass/run_state.json. Logs: logs/second_pass/supervisor. Launcher: scripts/start_second_pass.ps1. Do not launch a second coordinator or a direct GPU job while this one is active.
- Existing heartbeat airline-satisfaction-overnight-monitor is ACTIVE every15minutes in this chat. It is permitted to use bounded agents, stop/fix/restart owned failures, add the registered conditional confirmation and finish the release. Pause it only after verified completion or a clearly reported verified-fallback outcome.
- Queue configs/second_pass.json contains30 immutable original configurations plus17 registered new ones. Five new direct runs finished before detached launch; the supervisor correctly validated and skipped them. The first supervised source-feature XGB also completed. RealMLP with original auxiliary features was progressing at11:44 UTC.
- Current blend snapshot: development OOF0.961243861101308, consisting of80% fixed v1 anchor plus20% v2_xgb_route_te_teacher_profiles_aux. Improvement over the anchor0.000022830915464. Paired blend fold gains:0.0000218523 / 0.0000342992 / 0.0000101795. This is adaptively reused development evidence, not a public or independent audit gain.

## Verified preparation

- Original-only LightGBM teacher completed in29.594seconds; cache data/original_lgb_teacher_predictions.parquet covers999,479 unique keyed rows. Native model under artifacts/original_lgb_teacher. No synthetic satisfaction labels read.
- Original-only13-model auxiliary bank completed in162.828seconds. Fixed seed0,400rounds/model, training on129,859 audited original rows, no satisfaction inputs/targets. Own predicted rating excluded. All999,479 rows have13 expected-rating features in data/original_aux_predictions.parquet. CUDA fit, CPU inference, float64 EV reductions followed by float32 storage. Native bank under artifacts/original_aux.
- Cache/native features and CUDA/native RealMLP composition passed a1,024-row smoke selected only after filtering frozen development IDs. Logs: logs/second_pass/source_smoke.log.
- Eight auxiliary synthetic tests, six original-teacher synthetic tests, and42 second-pass release/features/fallback/supervisor regressions passed. The latest23 release tests also passed after ledger integration. Original supervisor recovery/self-test passed. Do not call these whole-model performance evidence.
- Operational agent verified one supervisor tree and one GPU training process. RAM recovered1.40→1.72GiB while RealMLP progressed. RAM is the active constraint; avoid simultaneous dataframe/model-heavy inspection. Do not stop a healthy job for a single transient sample. Owned PID/create-time/command checks are mandatory before stopping.

## Monitor decisions

1. Read current state, active stdout/stderr, result.json and current.json. Healthy unchanged state needs no announcement. Diagnose actual tracebacks, stalled progress or resource pressure before repair; never edit an existing run object or overwrite completed artifacts. Finished runs are validated and resumed/skipped automatically.
2. Let the registered17 experiments finish. Model-selection score, all fold metrics and search grid are saved by the development-only release script. Original audits are closed and must not be re-scored. Do not submit to Kaggle during this local campaign.
3. At most one extra seed confirmation is preregistered in configs/second_pass.json: seed20261021; only after the primary batch completes and before15:15UTC; require a fixed-anchor mixture to gain at least0.00003 pooled AND mean-fold AUC with no negative fold changes. Use step0 entries of the saved blend-search grid against baseline_scores to identify the strongest qualifying recipe, checking its original exact run object. Record the choice before fitting, clone to a new ID, change only seed, and declare its fixed50/50 blend_group with the first seed before appending/starting. Do not compare seeds and choose the better one. If no recipe qualifies, skip confirmation. No arbitrary replacement sweep.
4. Stop launching11:45EDT/15:45UTC, stop exploration by12:00EDT/16:00UTC. The supervisor freezes at the cutoff, refits, verifies, and reports. Empty queue waits for the cutoff. An early freeze is allowed only with a documented diminishing-return or verification-budget reason, using the same fixed selection policy.
5. Release script: scripts/second_pass_release.py --phase blend --freeze, then --phase refit, then --phase verify, all with --config configs/second_pass.json. The supervisor already dispatches these; do not duplicate active phases. Full-data refits reuse verified v1 members and fit selected new members in fresh sequential processes. They may use all699,635 labels only after freezing. No new unbiased audit score is claimed.
6. Verification independently recomputes all299,844 blended probabilities, checks exact IDs/schema/bounds and native hashes, then runs saved-model inference from every raw test row. It writes artifacts/second_pass/verification.json, final/release_provenance.json, an archived CSV package and SECOND_PASS_REPORT.md. Inspect success, path and hashes before reporting complete. On failure, scripts/second_pass_fallback.py revalidates immutable v1 and writes an explicit failed-v2/fallback report; it does not pretend native inference ran again. The verify phase rethrows after publishing a valid fallback, so supervisor remains needs_attention and the cause is visible.

## Preserve

V1 final submission SHA02443aa5011384df8343cde5b43e73003a39a65fca28fb99c602f50c7d3e4b10, Kaggle submission56771783, public score0.96093. Its historical OOF0.96122103, original audit0.96116318,69,396-row unexposed sensitivity0.96127692 are v1-only evidence. Preserve the original568-row exclusion manifest. No current leaderboard rank claim without a new authenticated read.

Never print or copy credentials. No paid remote compute. Do not modify original release artifacts, original frozen selection, raw data/splits, completed run configurations, or REPORT.md. New report is SECOND_PASS_REPORT.md.

Monitor airline-satisfaction-overnight-monitor was paused and read back as PAUSED at 2026-10-02T13:12:08.250847+00:00.
