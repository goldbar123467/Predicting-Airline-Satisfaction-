## Latest best-blend submission authorization, October2 17:36UTC

The user asked to mix saved local/cloud epoch checkpoints, score blends locally and submit the best. The bounded saved-outer-model screen is complete:3fixedpools,12mixturetests, no qualifying gain. See CHECKPOINT_BLEND_REPORT.md, configs/checkpoint_blend_v1.json and artifacts/checkpoint_blend_v1/screen.json. Do not repeat this search or relabel prior failed standalone candidates as passing. Per-epoch checkpoint sequences were not saved; current500fit remains unchanged.

This request renews permission for at most ONE qualifying verified release from current third_pass_batch07 through October3 12:47:41UTC. It supersedes old no-overnight-submission wording only for that exact scope. Separate authorization `state/kaggle_best_blend_authorization.json`, SHA256 `530eab7d62064a8e6ff988db790e18411f6ffaf86f8c996a3a9e63f622b64104`. Original immutable configs still declare no internal automatic submission; the monitor performs this externally authorized step only after primary/completed-release checks. Run scripts/submit_verified_release.py --campaign third_pass_batch07 after full final verification, with its unchanged gain gate, deduplication, current daily allowance and durable intent/reconciliation. Never submit unchanged or nonimproving blends. No new cloud jobs authorized. Pause monitor only after campaign deliverables and any new authorized submission attempt are actually resolved.

## Immediate start amendment, October2 16:51UTC

The user explicitly requested starting the500-epoch run now. The idle scheduled waiter was identity-verified, stopped and replaced by exactly one supervisor at16:51:19UTC. This supersedes the earlier17:05 waiting time. Original protocol/queue/training recipes remain byte-identical. Separate hash-bound operational amendment: `state/long_local_500/start_now_authorization.json`, SHA256 `ce198f65392c1b1a62c5b4e906d1e9c5e658cd979e7d21b7dffb5d3702dfddce`. Policy `operational_timing()` advances all operational bounds equally, preserving the20hour ceiling and4hour refit/1hour verification reserves.

Effective exploration cutoff: October3 07:37:41UTC; final delivery cutoff: October3 12:47:41UTC (08:47:41Eastern). Latest launch gate17:17:41UTC is now moot because launch occurred. Read `state/long_local_500/launch_receipt.json`, `manual_launch_checks.json`, and `state/third_pass_batch07/run_state.json` for exact ownership/times. Supervisor launcherPID8772 creation1790959879.003804; actual supervisorPID27144 creation1790959879.0268204. Old waiterPID31836 is retired, not active. Never start another waiter or supervisor. No new cloud or automatic overnight submission authority. Keep the same15minute monitor ACTIVE, preserve all prior leakage/quality/native verification gates, and report actual primary training status separately from a verified fallback release.

Executed3timing,6policy,10supervisor and6waiter checks passed; compilation and current registration/readiness/prior-result checks passed. The amendment only changes operational scheduling. Statistical fitting sources and frozen run configs are unchanged.

# Registered local 500-epoch campaign

The user requested a long local training run, approximately15â€“20hours, after the current pass. This registration adds one500-epoch horizon contrast with a20-hour overall ceiling. Actual completion may be much sooner. The extension authorizes local training and a verified local release; it does **not** extend cloud execution or submission authorization beyond2026-10-02 17:00UTC.

## Files, identities and scope

- Authoritative protocol: `configs/long_local_500.json`, SHA256 `42a965cd97b3a584b39cf74810f349baaf168ba0d81e6fee446ed66d21fac318`.
- Executable queue: `configs/third_pass_batch07.json`, SHA256 `9cbfcdf4df6fc79cc1c3338d3589fb5ff182c58b497b1e0af7e4b4f7f8f2b9ce`.
- New candidate: `v3_realmlp_cat_raw_aux_e500`. Exact matched control: `v2_realmlp_cat_raw_aux`.
- Preserve all54 previous batch05 run objects, including the currently running e60 recipe. Add exactly one candidate; no new seed, feature block, architecture, or hyperparameter search. Existing runs must already be complete before this queue can launch.
- Candidate copies the completed v2 control, changing only ID, `max_rounds: 4 -> 500`, and the operational timeout to53400seconds. Native categorical raw+EV inputs, seed20261005, ensemble8, hidden[512,256,128], batch256, eval batch2048, LR.053, WD.015, threads4, label smoothing0 are unchanged.
- Paths: coordinator `state/third_pass_batch07`, logs `logs/third_pass_batch07/supervisor`, models `artifacts/runs/v3_realmlp_cat_raw_aux_e500`, isolated release `artifacts/third_pass_batch07`, report `THIRD_PASS_BATCH07_REPORT.md`. Waiter state/logs are separately under `state/long_local_500` and `logs/long_local_500`.

Use the existing supervised trainer, process-tree recovery, fold checkpoints, third-pass selection and native release code. No second training framework or independent release implementation is introduced. Root added exact-protocol guards allowing a4-hour refit budget and1-hour verification reserve only for this registered campaign. Its500-epoch paired admission gate is distinct from the existing60-epoch contrast.

## Schedule and boundaries

All times below are UTC; Indianapolis remains EDT.

| Event | UTC | Indianapolis |
|---|---|---|
|Earliest launch|October2 17:05|October2 13:05|
|Latest allowed launch|October2 17:35|October2 13:35|
|Exploration cutoff|October3 07:55|October3 03:55|
|Final blend cutoff|October3 08:05|October3 04:05|
|Refit cutoff / reserved verification start|October3 12:05|October3 08:05|
|Supervisor exit bound|October3 12:55|October3 08:55|
|Overall delivery deadline|October3 13:05|October3 09:05|

The53400second primary cap allows14h50m of training from the earliest launch. Finalization starts immediately if the full queue finishes earlier. The10minutes between exploration and final-blend cutoffs avoids the earlier campaign's boundary collision. Four hours are reserved for possibly expensive full-data refitting; the final hour covers raw inference and delivery. The existing supervisor stops children15seconds before its phase cutoffs. Later launch does not extend the fixed deadline, so total campaign time is then shorter than20hours.

No arbitrary idle delay, extra candidate, repeated seed, or optimizer changes will be added merely to occupy15â€“20hours. Do not automatically submit the overnight CSV. Notify the user with completed training/release evidence and retain the latest submitted artifact separately.

## Expected runtime and statistical interpretation

Measured while preparing this plan: the new local60-epoch raw+EV run's first inner fit took560.937seconds and selected epoch3. Linear500/60 scaling estimates4674seconds, approximately1.30hours per inner fit; three inner fits would take about3.9hours. This is a throughput projection, not a500-epoch benchmark. Three outer refits, native exports, prediction, packaging, and resource contention add time. If selected epochs stay low, completion could be around4â€“5hours. If selected lengths approach500, outer and full-data refits can add several hours. The20-hour ceiling is therefore a safety bound, not a prediction of convergence or useful GPU work.

The four-epoch control took256.359seconds overall and selected[3,3,3]. Its development pooled AUC was0.960852059590479, with folds[0.9604011182291085,0.9608477276434066,0.9613270013748533]. Historical scores are adaptively reused development evidence. No public or exposed audit scores select this experiment, checkpoints, weights, or acceptance.

Keep the exact frozen three development folds and inner90/10 split (`seed+fold`). Preprocessing and vocabularies fit only the relevant training partition. Each inner model actually traverses up to500epochs because early termination is disabled, while its best mean-ensemble inner-AUC epoch is selected. Each outer model is freshly fitted for that selected length. As documented in `research/third_pass_validation.md`, normalized scheduling means a500-epoch horizon changes the earlier learning-rate/regularization trajectory. Outer refitting starts another normalized schedule; this is not replay of the inner schedule prefix. Existing pinned PyTabKit1.7.3 method/source research applies; no architecture or dependency changes are proposed.

Before blend admission, the e500 candidate must exactly match the four-epoch control except ID/horizon/timeout, improve or tie pooled and mean-fold AUC, and lose no more than0.00002 on any fold. Then apply the unchanged fixed-v2 anchored blend gates: alpha[.05,.1,.2,.3], at most two additions, pooled/mean improvements at least0.00001, worst-fold loss at most0.00002. All three outer folds and valid native artifacts are required. An incomplete run cannot enter the blend. Full training labels may be used only for fixed-length production fitting after selection freezes; no exposed audit evaluation is rerun.

## Progress and recovery limits

The existing library uses `verbosity=0`, a Lightning `DummyLogger`, disabled progress bars, and disabled Lightning checkpointing. Thus it does **not** currently publish epoch scalar curves or resumable optimizer checkpoints. Epoch validation messages exist at library verbosity2, but changing logging configuration is deliberately not mixed into this exact registered recipe.

Durable evidence remains stdout/stderr phase markers, supervisor10-second PID/create-time/command/resource state, fold `done.json` and prediction hashes, and native metadata including selected epoch, full horizon, scope, source/library versions, elapsed time and export parity. A healthy inner500-epoch fit may have quiet stdout for over an hour. Diagnose CPU/GPU/process progress and memory trend before calling it stalled.

Completed outer folds resume exactly through their validated saved predictions and native artifacts. Interrupted inner or outer fits restart that unfinished fold; there is no in-epoch optimizer/RNG resume. No automatic retry is registered. If the process fails, the monitor may diagnose a bounded repair, but must preserve completed folds, unchanged run identity/source contract, the overall deadline, and valid fallback. Do not silently change epochs or seeds to salvage a partial result.

## Waiter and launch handoff

The new waiter is `scripts/long_local_500_wait.py`; the hidden Windows launcher is `scripts/long_local_500_start.ps1`. Root launched the waiter at2026-10-02 15:42:31UTC (launcherPID29684; recorded waiterPID31836, create_time1790955751.6301045). Its observed state at15:43:01UTC was `waiting_for_not_before`;500-epoch training had not started. Do not launch a second waiter. The waiter never trains, stops other processes, calls cloud APIs, or submits. It checks immutable registration/source hashes and only dispatches the established supervisor once.

Before dispatch it requires all of:

1.17:05 <= current UTC time <17:35 on October2.
2.The prior-release readiness receipt described below, including completed local raw verification, resolution of today's submission attempts, and terminal evidence for all owned cloud jobs.
3.All54 prior queue result manifests exist and retain their exact registered run objects. The supervisor then revalidates their full prediction hashes.
4.No other project-owned Python worker is active, and at least3.5GiB RAM is available. It waits rather than stopping a healthy worker. Root must not start another local worker after declaring final readiness.
5.No waiter/campaign STOP file and no previous launch intent. An OS-held lifetime lock prevents duplicate waiters; intent is durable before spawn, followed by PID/create-time/command/log receipt. Uncertain launch outcomes require inspection, never blind retry.

If e60 fails or remains incomplete, the waiter blocks instead of retraining it. Root must register a new isolated batch08/protocol from the completed batch03 base and the same500-epoch contrast, then review the corresponding narrow supervisor/release policy. Do not mutate batch07 or reinterpret the existing registration silently.

Validation only:

```powershell
./scripts/long_local_500_start.ps1 -CheckOnly
.venv/Scripts/python.exe scripts/long_local_500_test.py
```

For reference only, the command root already used to start the waiter was:

```powershell
./scripts/long_local_500_start.ps1
```

The script can wait before17:05. Do not use `start_third_pass.ps1 -CampaignId third_pass_batch07` directly, because that bypasses the additional time/resource/prior-resolution gates.

## Required prior-campaign readiness receipt

Root or the existing monitor writes `state/long_local_500/prior_campaign_ready.json` only after the current local release and today's cloud/submission work are actually resolved. Required fields:

```json
{
  "status": "ready",
  "queue_sha256": "9cbfcdf4df6fc79cc1c3338d3589fb5ff182c58b497b1e0af7e4b4f7f8f2b9ce",
  "resolved_utc": "ACTUAL_TIME_WITH_UTC_OFFSET",
  "local_release_complete": true,
  "submission_attempt_resolved": true,
  "all_owned_cloud_jobs_terminal": true,
  "evidence": {
    "release": {"PROJECT_RELATIVE_FINAL_VERIFICATION_JSON": "ACTUAL_SHA256"},
    "submission": {"PROJECT_RELATIVE_SUBMISSION_OR_NO_SUBMIT_RESOLUTION_JSON": "ACTUAL_SHA256"},
    "cloud": {"PROJECT_RELATIVE_TERMINAL_STATUS_JSON_FOR_EACH_OWNED_JOB": "ACTUAL_SHA256"}
  }
}
```

This is a schema example, not an existing ready attestation. Every evidence path must be a real project-relative JSON file with its actual hash; each role must be nonempty. Release proof must declare299844rows, all-row raw inference and independent blend recomputation. Cloud status must be terminal (`COMPLETE`, `ERROR` or cancellation variants); include every relevant job, such as `cloud/third_pass_lgb/latest_status.json` and `cloud/third_pass_long/latest_status.json`, plus any later production job. Root's all-jobs attestation covers completeness of that list; the waiter makes no network requests. Submission evidence may document a processed upload or an explicit no-submit/non-gain decision. Do not infer resolution merely because time passed.

## Verification and monitoring

Executed preparation checks: six synthetic waiter tests passed; Python compilation passed; live registration dry-run passed. Tests cover recipe/source mutation, readiness hashes and unresolved cloud status, incomplete prior runs, earliest/latest launch boundaries, active-worker/low-RAM blocking, and intent-before-spawn/duplicate rejection. Child launch was mocked, so no production worker was started by these tests. Prior54run objects and candidate's three allowed differences were asserted before registration.

Root should extend the existing15-minute heartbeat to monitor this overnight local campaign after today's cloud closure. Keep cloud deadlines and submission authority separate. Monitor `state/long_local_500/waiter_state.json`, then `state/third_pass_batch07/run_state.json`; final completion requires verified release artifacts and report. Pause only after the requested overnight delivery is complete or an unresolved failure is reported. No duplicate automation is created by this implementation.

Read-only integration review also checked the root-owned exact protocol/queue/source/base/control hash policy, the17:05 supervisor start prohibition, the14400-second refit allowance restricted to this protocol, the3600-second verification reserve, and mandatory exact500-versus4 paired admission before blending. No integration blocker was found. A supervisor status of `complete` can describe a verified fallback even if a queued candidate failed; the monitor must separately confirm the e500 run result exists and report any incomplete/failed primary experiment.
