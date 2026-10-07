# Local long-horizon registration, 2026-10-02

The user explicitly requested continued local and private-cloud training plus a much longer duration test. This authorizes one new local schedule contrast despite the previously closed short-horizon branches. This note registers the local experiment before fitting. No training was launched while preparing it.

## Exact contrast and isolation

- Campaign: `third_pass_batch05`; config `configs/third_pass_batch05.json`, SHA256 `cc6ae97f9234e8319f5863a071c3d2fd547c2cae5ebd05527d0ad6877df9e00d`.
- Candidate: `v3_realmlp_cat_raw_aux_e60`; control: completed `v2_realmlp_cat_raw_aux`.
- Copy the control exactly, changing only the ID, `max_rounds: 4 -> 60`, and operational `timeout_seconds: 1200 -> 3000`. Seed20261005, raw+original-EV features, native categorical twins, ensemble8, hidden[512,256,128], batch256, inference batch2048, LR.053, WD.015, four threads, and label smoothing0 remain identical. No source teacher, q probabilities, new seed, architecture, or learning-rate change.
- Preserve all53 previous batch03 run objects and the existing selection policy, comparison mapping, exclusion list and blend groups. Add the new candidate once. There are no cloud runs in this queue; pending/imported cloud LGB experiments stay in their separate campaign.
- Parent config `configs/third_pass_batch03.json` SHA256 `70af288ee50cbbb3c8f25c3e6492ed70e72f59b1dab3ab8d7b65fb8bfafbedd2`; control result SHA256 `92624c94d88f5967f0277cfda7350337eebbcaf295f1d821031663d4b1d090f1`.

## Nested training and interpretation

Existing `common.fit_model` passes `max_rounds` as `epochs`; the categorical RealMLP adapter accepts any positive integer horizon and sets `use_early_stopping=False`. All60 inner epochs therefore execute unless an operational timeout/failure intervenes. The checkpoint is selected by inner mean-ensemble ROC AUC only. Each outer model is freshly fitted on its outer-training partition for exactly that selected epoch count. All three frozen development folds must complete before a result can enter selection.

The existing schedule is normalized by total epochs. Changing the horizon changes the early learning-rate/regularization trajectory as well as extending training. The outer refit starts a newly normalized schedule at the selected length; it does not replay the selected prefix of the60-epoch inner schedule. Keep this known limitation matched to the control. This is a duration/schedule contrast, not a claim that only identical extra updates were added. Dated method evidence and pinned PyTabKit/paper references are already recorded in `research/third_pass_validation.md`; no new architecture or dependency decision is made here.

Frozen development rows/folds, inner split seed+fold, preprocessing fit scopes, source-only EV cache, and native export checks remain unchanged. Original audit labels do not select checkpoints, weights, admission, or reports. The generic trainer still produces its legacy audit predictions, but they are not scored or inspected. Full699635-row production fitting is allowed only after development selection freezes. Development scores remain adaptively reused selection estimates.

## Measured cost and uncertainty

The exact local four-epoch control completed in256.359s with outer lengths[3,3,3], OOF0.960852059590479 and fold AUC[0.9604011182291085,0.9608477276434066,0.9613270013748533]. Its native inner fits took[41.984,41.594,42.188]s; fixed-three-epoch outer fits took[34.578,34.485,35.125]s. Other processing accounted for26.405s.

Conservative linear scaling of these native-fit times gives:

| Each selected outer length | Estimated three-fold e60 total |
|---|---:|
|3|2017s,33.6min|
|12|2330s,38.8min|
|30|2955s,49.2min|
|60|3997s,66.6min|

These are estimates, not benchmarks of this new run. Native fit/export overhead is included in scaled values; wall-clock contention can also increase cost. The3000s cap is plausible when inner selection remains comparatively early, but cannot guarantee completion if late checkpoints win. Preserve incomplete checkpoints and exclude the run if unfinished; do not force an earlier model or select based on partial outer scores. No automatic retry is registered (`max_retries=0`).

Additional observed context: older local `realmlp_te_teacher16` took539.203s and selected[5,6,5], but its representation differs and it is not a matched timing or quality control. Same-image cloud raw+EV e4 took633.617s and e12 took970.262s, with selections[3,3,4] and[4,3,4]. Their hardware/software differ from local; cloud timings must not be treated as local benchmarks. The cloud e12 failed its existing paired quality gate. The new e60 branch is specifically user-authorized, not justified by claiming that result was positive.

## Admission and release

Config contains `paired_duration_controls: {"v3_realmlp_cat_raw_aux_e60": "v2_realmlp_cat_raw_aux"}`. Root implemented mandatory enforcement in the shared release code, reviewed before launch: exact control4/candidate60 recipes may differ only by ID, horizon and timeout; candidate pooled and mean-fold AUC must be at least the control's, with worst paired fold loss no worse than0.00002. The preexisting local `comparison_controls` mapping only reports diagnostics and is insufficient by itself.

Passing that paired gate only makes the candidate eligible. The unchanged fixed-v2 blend gate still requires at least0.00001 pooled and mean-fold improvement, at most0.00002 worst-fold regression, alpha in[.05,.1,.2,.3] and at most two additions. Another submission also needs the existing improvement gate against the best previously submitted development blend. Public scores and test predictions cannot guide this decision. Retain the latest verified release if any gate or full native verification fails.

## Timing and ownership

Root-approved config: stop new jobs15:25UTC, finalize16:15UTC, deliver17:00UTC, run timeout3000s, blend timeout120s, production refit timeout1500s, early finalization after the complete queue. Actual launch should be no later than15:20UTC so the full training budget ends by16:10. This leaves time to freeze before the supervisor's computed final-blend cutoff16:15 (`verify_at16:40 - refit budget25min`). A trainer still running at the16:14:45 hard stop would leave no final-blend dispatch window; do not rely on that boundary for a new release.

The last batch's observed phases were20.105s freeze,91.394s refit and670.022s full verification. A selected longer model increases full-data refit cost; the1500s refit reserve covers a conservative full60-epoch projection plus existing selected candidates, but remains an operational bound rather than a guarantee. Verify must finish before the supervisor's16:49:45 limit, leaving the delivery/submission reserve before17:00.

Only one local GPU worker may run. Prior batch03 status was complete; read-only process inspection at approximately15:07UTC found no project training/refit worker and4.876GiB available RAM (inspection commands themselves were present). Root rechecks ownership and resources immediately before launching. Avoid concurrent local heavy imports while the trainer is active; remote cloud CPU/GPU jobs are separate resources. The existing15-minute monitor handles state/log/RAM and PID/create-time/command checks. It may stop only demonstrably failed/stalled owned work or enforce this registered deadline. Quiet inner training logs alone are not a stall.

## Verification performed before handoff

Programmatic assertions verified: all53 prior run objects unchanged,54 unique IDs, no existing candidate artifacts, exact three allowed run differences, unchanged existing comparison/selection policy, no Kaggle backend in the local queue, and control object equality with its completed result. Supervisor dry-run passed for the isolated campaign paths, deadlines and queue. Two executed synthetic end-to-end blend checks also passed: a completed e60 candidate that loses to its control is excluded before the fixed-anchor search; omitting its mandatory paired-duration mapping raises before admission. Static review confirmed the gate requires exact recipe equality except ID/horizon/timeout and checks pooled, macro and worst-fold differences. No fitting, native model loading, audit scoring, API calls, or process mutation was performed.

Root launch after shared paired-gate review and resource check:

```powershell
./scripts/start_third_pass.ps1 -CampaignId third_pass_batch05
```
