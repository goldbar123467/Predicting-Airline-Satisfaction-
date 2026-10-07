# Overnight competition campaign

Deadline: October 2, 2026, 07:00 America/Indianapolis (EDT), 11:00 UTC. This is one hour earlier than literal EST.

## Objective and contract

Produce native saved models, a reproducible nonnegative probability blend, a verified competition submission file, and a report. Optimize ROC AUC with True satisfaction as positive class. Test labels are unavailable and never used. No Kaggle submission has been requested or performed.

Data: 699,635 competition training rows, 299,844 test rows, 21 raw features. Freeze raw file hashes and splits in data/manifest.json and data/splits.parquet. Exclude identifier from model features. Four string categorical fields, 13 ordinal rating fields, age, flight distance, and two delay fields. Preserve missing arrival delay information; model preprocessing is fitted within each training partition.

## Validation

Reserve 69,964 stratified audit rows using seed 8675309. Three stratified folds on remaining 629,671 development rows use seed 20261001. Each outer fold uses an internal 10% stopping set from that fold's training portion, then refits on the entire outer training portion for the selected number of trees/epochs. Outer fold labels never choose early stopping. Save keyed OOF, audit, and test probabilities. No identical feature-row duplicates were found in competition training data. Original-source duplicates are separately removed, including all exact profiles matching competition train/test.

Blend weights are chosen using development OOF only, with a coarse, nonnegative convex search. Compare pooled OOF, per-fold AUC, and strongest single candidate. Audit labels are scored once only after freezing the final selection. The audit score evaluates the development-trained ensemble. Subsequent full-data refits include all competition labels and therefore have no additional unbiased score.

Validation incident first identified at03:16UTC: integration smokes had sampled before restricting to development. A retrospective review of all root tool commands identified first5000-row tree checks, first4000-row encoded LightGBM, first600-row numeric RealMLP, and the1024-row categorical sample. Conservatively excluding their union removes568 reserved audit rows. These smoke models were isolated from competition candidates and never entered blend selection. The test now filters to development. Frozen splits remain unchanged. `data/audit_smoke_exclusions.json` records the568 IDs and command evidence, and finalization will additionally report ROC AUC on the69,396 unexposed audit rows. This assessment policy is hashed before audit scoring and cannot change on frozen replay.

## Experiment order

1. LightGBM CPU baseline to establish a valid prediction artifact.
2. CatBoost GPU and XGBoost GPU baseline diversity.
3. Flight distance as an additional categorical route feature, tested with LightGBM and CatBoost.
4. Original-data-only XGBoost teacher as an additional feature. Source labels are permitted; exact synthetic matches are excluded. Teacher uses fixed settings without synthetic-label tuning.
5. Bounded TabM plain and piecewise-linear embedding variants, and a short RealMLP variant only after CPU/CUDA export parity verification.
6. Cross-fitted route target statistics and counts, with held-out label invariance regression tests. Test with and without the original teacher on identical folds/settings.
7. Controlled leaf/depth capacity ablations of successful route/teacher models. Further hypotheses require paired development evidence and time budget. Rating aggregates are an experiment, not an assumed benefit.
8. At23:12EDT, queue16-epoch RealMLP TE+teacher, native-categorical RealMLP TE+teacher, CatBoost numeric categorical twins+teacher, and predetermined CatBoost seed20261009. The two seeds of CatBoost route+teacher enter blending only through their fixed equal average once both complete. The seed also changes the inner stopping partition, so this measures pipeline variability rather than model RNG alone.
9. Following the native RealMLP contribution at0.96115273 blend OOF, queue four bounded checks: add cross-fitted interaction statistics to the existing CatBoost route+teacher recipe; reduce XGBoost TE+teacher depth8 to6; fixed-average native RealMLP with predetermined second seed20261011; raise categorical TabM weight decay0.0003 to0.003 after its observed early overfitting. Keep every other setting fixed within each comparison. These bring the planned campaign to28 configurations, not a broad random sweep. Prefer stopping further exploration if subsequent batches add less than0.00002 mean-fold AUC without another specific failure mode or useful representation hypothesis. The final audit cannot inform that decision.
10. At03:52UTC, the 28 completed candidates yield blend OOF0.96119736. Add one source-supported representation check: `xgb_route_te_teacher_profiles` matches `xgb_route_te_teacher` exactly except for 16 numeric means within flight distance, fitted only on each training partition. Held-out and unseen routes use frozen maps/global training means. No labels enter these profiles; training rows may contribute their own features. Six feature-stage tests and native LightGBM/XGBoost composition/reload checks on synthetic data passed. The public GLM-margin experiment is deferred because its source results do not establish an advantage over the ordinary XGBoost baseline.
11. The route-profile XGBoost improved all folds by0.00009960/0.00023949/0.00011639 and pooled OOF by0.00015345. The blend improved0.00002367 to0.96122103, with positive changes on every fold and15.42% profile-model weight. Predeclare one final transfer: `cat_route_te_teacher_profiles`, identical to `cat_route_te_teacher` except for the same profiles and a600-second safety timeout. This tests information beyond CatBoost CTR and cross-fitted target statistics. After this30th candidate, stop exploration and freeze using development OOF regardless of this last result. Do not add another seed/capacity/feature sweep or use audit results to alter selection. Finalization may run early once operational checks pass; the07:00 deadline is a latest completion time, not a reason to keep selecting models.

## Operations

One detached supervisor and one experiment child, eight CPU threads. The 15-minute Codex heartbeat can review results, repair failures, and change pending queue entries. Avoid changing existing run IDs/configs, frozen splits or completed artifacts. Monitor state/run_state.json and logs/supervisor. Do not terminate unrelated processes. AC sleep was already disabled.

The original `tabm`/`realmlp` families use low-cardinality categorical one-hot inputs only. Their transformer rejects high-cardinality route matrices. New `tabm_cat`/`realmlp_cat` families use compact fold-fitted category codes and retain numerical copies of the original seventeen measurements. TabM constructs one-hot tensors per GPU batch; RealMLP uses native small one-hot blocks and learned embeddings. Unknown categories map to a dedicated index. Both may add the tested cross-fitted numeric route encoder. Each representation's own OOF results decide inclusion. Require real-data and CUDA saved-model checks before queueing a new family.

Stop adding new fits at 05:30 EDT; current fits may finish until 06:00. Then freeze/evaluate the blend, refit selected members on all training rows, and verify before 06:50. Keep current/fallback predictions available during the campaign. Limit full refit to 06:40, leaving ten minutes for verification. If refit fails, verify and report the development-fold blend fallback.

Early completion decision at04:01UTC: the final30th controlled candidate was already running, and the hypothesis stopping rule was fixed. Advanced stop_new_runs_utc so that the supervisor froze and refit immediately after this run, while preserving the original07:00 deadline and late failure-recovery margins. No additional model selection is authorized by the monitor after this point.

Completed release: selection frozen at04:02:37UTC after30 configurations; OOF0.96122103, original audit0.96116318, preregistered unexposed69,396-row audit0.96127692. All ten selected configurations were refitted on699,635 training rows. Native reload and full-file recomputation passed at04:10:06UTC. Separate inference from all299,844 raw test rows passed at04:15:45UTC, maximum absolute difference8.7071e-8. Submission and native models are under artifacts/final; evaluated fold predictions remain artifacts/blend/submission_fallback.csv. Source snapshots and native checksums are recorded in artifacts/final/release_provenance.json. Supervisor is complete and the15-minute heartbeat is paused. No Kaggle submission was made.

Final refits now run one member per fresh subprocess to release numerical-library and allocator memory between models. The lightweight coordinator loads only row counts and test IDs, validates cached completions and worker outputs, and packages only when every selected member succeeds. Six synthetic tests include actual Windows subprocess execution, log forwarding, distinct worker identities, failure preservation and resumability. Successful verification archives the CSV and selection under `artifacts/verified/<sha256>` so subsequent exploration cannot overwrite a verified fallback.

## Reproduction

Environment: .venv/Scripts/python.exe; exact installed dependency versions in requirements.lock.txt. Prepare only once: `python scripts/prepare.py`. Launch: `powershell -File scripts/start_overnight.ps1`. One candidate: `python scripts/train.py --config configs/overnight.json --run-id NAME`. Current blend: `python scripts/blend.py --config configs/overnight.json`. Final frozen selection: add `--finalize`, then run scripts/refit.py and scripts/verify.py with the same config.

Saved-model inference on a raw CSV: `python scripts/predict.py --input data/test.csv --output artifacts/reproduced.csv`. It recomputes any original-data teacher features from the native saved teacher, so it also supports unseen passenger IDs with the required raw feature schema. `--fold-ensemble` selects the evaluated fold ensemble instead of a full-data refit package. All-row inference of the first four-model blend was executed and agreed with stored predictions within 1.5e-7.

Each run stores config, split hash, native models and preprocessing, fold predictions, OOF metrics and hashes. Model training is seed-controlled but GPU training is not promised bitwise deterministic. Verification independently recomputes the blend and reloads native saved models for inference on real test rows.

## Executed initial checks

CatBoost, XGBoost, TabM and TabM PLE passed actual CUDA forward/training and native reload checks. LightGBM passed CPU fit/reload. Five pipeline contract tests verify partition exclusion, fold-local preprocessing, keyed prediction alignment, convex blend arithmetic and completed artifact hashes. Supervisor tests include Windows redirector/child-tree ownership and termination, not just launcher PID liveness. The first two models completed before 22:24 EDT and produced a verified working blend; subsequent work improves that preserved fallback.

New experiment contracts include code/data/dependency/teacher hashes and copies of generating source files. Two initial baseline source versions were copied to artifacts/source/initial. If a partial run's source or data hash changes, fail clearly and use a new experiment ID instead of mixing incompatible folds. Audit selection is written to immutable selection.json before audit scores are computed; a crash may repeat arithmetic for the same fixed selection but cannot select again after audit exposure.
