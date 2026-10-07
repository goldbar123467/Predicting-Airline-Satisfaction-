# Overnight handoff

Campaign complete at00:15EDT on2026-10-02, ahead of the07:00 Indianapolis /11:00UTC deadline. Supervisor status is complete, all training has stopped, and the15-minute heartbeat `airline-satisfaction-overnight-monitor` is PAUSED after verified delivery. No further unattended work is pending.

Final submission: `artifacts/final/submission.csv`, SHA256 `02443aa5011384df8343cde5b43e73003a39a65fca28fb99c602f50c7d3e4b10`. All299,844 raw-CSV predictions reproduced with maximum absolute difference8.7071e-8. Native models, preprocessing, ten full-data refits, evaluated fold-ensemble fallback, source snapshots and checksums are retained. See REPORT.md, artifacts/verification.json, artifacts/final/raw_inference_verification.json and artifacts/final/release_provenance.json. No Kaggle upload occurred.

At04:02:37UTC, all30 configurations completed and the selection was frozen. Final development OOF AUC0.96122103, full audit0.96116318, unexposed sensitivity audit0.96127692. These are local validation scores, not leaderboard results. Ten positive-weight configurations across LightGBM, CatBoost, XGBoost and RealMLP remain. The final CatBoost profile transfer receives zero weight. Full-data refits and verification completed successfully. Do not perform further model/feature/weight selection on this exposed audit.

No Kaggle upload was performed. A valid model ensemble and all299,844 test predictions already exist under artifacts/blend. Full raw-CSV inference was verified across every test row for the first four-model blend. A later teacher/route-encoding blend passed raw-CSV inference on1,024 rows, maximum difference3.52e-8. Native model reload and independent recomputation also pass for the29-candidate blend, whose weights exactly match the final30-candidate frozen selection. The audit was scored only after writing immutable selection.json. A retrospective executed-command review found early isolated smoke tests touched568 audit rows; see RUN_PLAN.md and data/audit_smoke_exclusions.json. The preregistered69,396-row sensitivity score is now available. Never omit this recorded limitation or alter frozen splits.

## Monitor procedure

1. Read state/run_state.json and active child's stdout/stderr. Check recorded supervisor and descendant identities, true aggregate CPU/RSS, and GPU use. Low GPU utilization during export/preprocessing is normal. Do not kill a healthy run based only on an unchanged log.
2. Read completed artifacts/runs/*/result.json, artifacts/blend/current.json and artifacts/diagnostics.json. Refresh diagnostics with `.venv/Scripts/python.exe scripts/diagnostics.py`; it excludes audit labels and test predictions.
3. If a run fails, diagnose first. Existing IDs/configs and source/data hashes are immutable. A source-changing fix for an incomplete run generally requires a new ID, preserving old completed folds. Never silently rewrite a contract to mix code versions.
4. If supervisor has died, inspect state and recorded owned descendants before restarting through scripts/start_overnight.ps1. A state/STOP file is deliberate; inspect its reason before removal. Current supervisor has Windows redirector-tree recovery, not just launcher PID control.
5. Exploration is complete and audit-exposed. Frozen selection and assessment must remain unchanged. Only mechanical refit/inference/recovery fixes are allowed for this release; preserve its weights, configurations, and selected rounds. No more experiments.
6. Existing deadlines stop new fits at05:30, bound active fitting at06:00, reserve final refit until06:40 and verification until06:50. Keep a fallback. Do not extend beyond07:00 to finish a speculative experiment.
7. Finalize once: blend.py --finalize freezes selection.json before audit exposure. Then refit.py and verify.py. Supervisor performs these automatically. If full-data refit fails, verify and deliver the evaluated fold-ensemble fallback with a clear limitation. Never adapt the selection after seeing audit results.
8. On verified completion, report actual final metrics/artifact paths and any limits, pause the heartbeat, and mark the goal complete only when no required work remains.

## Current research follow-up

Native-categorical RealMLP passed real-data CPU integration and full-eight-member CUDA export/reload checks after repairing pinned-library categorical inference device constants; it is now queued. Native-categorical TabM completed both experiments. Both use scripts/categorical_transform.py, with shared dispatch and source snapshots. Original scripts/realmlp.py and scripts/neural.py remain unchanged. The fixed-seed group support in blend.py passed6 new tests and3 release regressions. Audit sensitivity policy passed3 synthetic tests and3 release regressions.

All earlier agents completed bounded research, source audits, original-data preparation/teacher, neural wrappers, target encoding, process supervision, diagnostics, and regressions. No remote compute or untrusted notebook execution was used. See research/competition_evidence.md and research/method_evidence.md.

## Important files and checks

- `.venv/Scripts/python.exe` is the interpreter; PATH Python is a Windows Store alias.
- `configs/overnight.json` is the executable queue; `RUN_PLAN.md` gives validation and reproduction details.
- `scripts/test_contracts.py`, `scripts/test_encoding.py`, and `scripts/test_release.py` pass. Supervisor process-tree self-test also passed.
- `scripts/predict.py --input RAW.csv --output OUTPUT.csv` runs saved models without fitting. `--selection PATH --fold-ensemble` fixes a current/historical blend manifest for reproducible inference.
- `artifacts/verification.json` and `REPORT.md` reflect the last verification, which may precede newer completed experiments. Final verification must refresh them.
- Verification now creates immutable CSV archives under `artifacts/verified/<sha256>`. The0.96115273 blend passed full-file recomputation and256-row CPU native reload at03:32UTC; subsequent candidates continue. Final refit uses one fresh subprocess per selected member, with six synthetic subprocess/recovery regressions passing.
- Queue expanded to30 configurations, ending with the CatBoost transfer of route profiles after the XGBoost experiment improved every fold and the blend. See RUN_PLAN.md. GLM-margin is deferred after its actual source results were checked. `route_profiles.py` passed six synthetic feature tests and a synthetic full-pipeline fit/native-reload check covering LightGBM and XGBoost. No competition labels were used in these new checks. Do not modify active common/train source contracts.
- At03:55UTC the29-candidate blend passed native reload and independent recomputation; immutable archive is artifacts/verified/d35646df32c14c6882567a4f95a72305475dd6272e8a346533799d2035697e69. One earlier verification attempt correctly rejected a concurrently changed CSV and was rerun after the blend was stable.
- The supervisor was gracefully stopped while idle, patched, then restarted. Four synthetic regressions cover unexpected coordinator exit and descendant cleanup, including failed-cleanup recovery with ownership evidence retained. This prevents a recorded refit worker from surviving into the next phase. Only project-owned processes were involved.
- Do not expose the Kaggle credential. Existing source data and metadata are already downloaded; new work does not need to rediscover credentials.
