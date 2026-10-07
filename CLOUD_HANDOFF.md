# Private Kaggle worker handoff

## All cloud work verified,13:10UTC

All five private jobs are COMPLETE with downloaded verified outputs. The full-fold control passed:419780rows,3fixedepochs, fit+export84.489seconds,total484.344seconds. Returned native models also passed localCPUreload and1024-row comparisons,maxdifference1.19209290e-7. GPU inference timing was shape-inconsistent and is not a steady-state throughput measurement; see CLOUD_REPORT.md for the limitation. No model-quality score was computed for the control. state/kaggle_cloud/run_state.json records zero active cloudjobs and approximately29.83GPUhours remaining. The local second-pass release is already verified. No further job is queued; pause the monitor. Earlier active-state sections below are historical snapshots.

Updated2026-10-02 approximately12:31UTC. New user authorization: continue tracking local runs and investigate/use existing Kaggle CPU/GPU allowance with an agent. Preserve the local second-pass deadline17:00UTC and original exposed-audit restrictions. This is added private free compute, with no new competition submission or public publication.

## Verified cloud state

- Account clarkkitchen. Authenticated quota at12:15:29UTC: GPU30hours remaining, TPU20hours remaining. CPU weekly allowance was not established. Recheck quota before substantial dispatch. Do not print credential contents or environment.
- CPU probe: clarkkitchen/s6e10-compute-probe-cpu-20261002, version1, kernel136780605, accepted12:26:21UTC, COMPLETE by12:27:18UTC.
- GPU probe: clarkkitchen/s6e10-compute-probe-gpu-20261002, version1, kernel136780618, accepted12:26:29UTC, COMPLETE by12:27:18UTC.
- Both probes used explicit private=true, internet=false,600second timeout. Remote metadata was pulled and privacy verified. No fits used competition labels. Each mounted raw CSV exactly matched data/manifest.json hashes.
- CPU:4logical cores,33,658,318,848bytes RAM, Python3.12.13, torch2.10.0+cpu. GPU:4cores,33,658,314,752bytesRAM, two TeslaT4 devices with15,360MiB each, driver580.178.04, torch2.10.0+cu128. GPU memory is separate per device.
- Both environments: NumPy2.0.2, pandas2.3.3, pyarrow24.0.0, sklearn1.6.1, XGBoost3.2.0, LightGBM4.6.0, CatBoost1.2.10. PyTabKit and TabM are absent. All generated-data XGB/Cat/LGB/Torch fit-save-reload checks passed. Probe runtimes CPU24.51s, GPU23.81s, excluding queue/startup and shutdown. These are operational checks, not model-quality or full-training throughput evidence.
- Reports, remote metadata, logs and receipts: artifacts/kaggle_cloud/probe_cpu and probe_gpu. Scripts: scripts/kaggle_compute_probe.py and scripts/kaggle_cloud_control.py.

## Executed RealMLP controls,12:40UTC update

Both additional controls are now COMPLETE and verified. All four project cloud kernels have ended. State/report hashes and current quota: state/kaggle_cloud/run_state.json.

1. clarkkitchen/s6e10-raw-realmlp-portability-20261002, version1, kernel136781864: raw1024-row CPU inference passed batches1/17/1024 against frozen v2_realmlp_cat_raw fold0 references; maximum absolute error1.78813934e-7. Runtime24.086seconds. Private dataset clarkkitchen/s6e10-portability-bundle-20261002, datasetID12325834, contains only reviewed9.54MB payload.zip and metadata; remote info.isPrivate=true confirmed. No training or scoring. Report cloud/portability/output/portability/portability_report.json.
2. clarkkitchen/s6e10-realmlp-compatibility-20261002, version1, kernel136781682: pinned PyTabKit1.7.3 installed and actual eight-member RealMLP synthetic training/export/reload passed. Installer changed PyTabKit and pytorch-lightning2.6.6 only; torchmetrics1.9.0 already matched. All protected Torch2.10.0+cu128/CUDA and numeric-library versions remained identical. Native export maxerror1.78813934e-7; CUDA reload and CPU/CUDA maxerror2.98023224e-7 across batches1/17/257. Unknown-category and training-only vocabulary checks passed. Runtime55.683seconds including install; not a full-fold benchmark. Report cloud/realmlp_probe/output/realmlp_probe/realmlp_probe_report.json. Remote private flag verified; internet=true only for package installation, no data attached.

Reusable packaging paths are scripts/kaggle_portability_prepare.py, cloud/portability, artifacts/kaggle_cloud/portability_bundle, scripts/kaggle_realmlp_probe_prepare.py and cloud/realmlp_probe. Every new training kernel must include the pinned install bootstrap because Kaggle sessions are ephemeral. The existing completed scripts/manifests remain immutable. Generic reviewed-job controller: scripts/kaggle_private_job.py; use Python -X utf8 for output downloads because the installed CLI otherwise fails writing Unicode training logs on Windows.

## Active full-fold control,12:59UTC update

The preregistered full-fold engineering control has now been dispatched as clarkkitchen/s6e10-realmlp-fold0-control-20261002, version1, kernel136783697. Inspect cloud/fold_control/push_receipt.json, latest_status.json and state/kaggle_cloud/run_state.json for exact times/live state. Remote private flag and exact runtime source were read back. Dataset clarkkitchen/s6e10-fold-control-bundle-20261002 was privately uploaded and read back as private; its reviewed14.018MB ZIP hash is f04ba483bd4a9715a15886335f24189d59512f82b4094f8b615c6f3a5ddae293. Only frozen outer-training features/labels and1024 label-free held-out development feature rows are included; no audit rows or labels. Fixed seed20261005,3epochs,8members, original raw recipe. Existing fold-fitted preprocessing is supplied as exact float32 matrices with ID/schema/hash validation.

One visible T4; requested provider timeout1800seconds plus1800second Linux alarm. Code hash eb8b8e08fe36c6303d2805169a3073d970a1b6d557cada2e552edb77e47edf4a. No AUC scoring or model selection. This control measures full-scale training/export time, memory and native CPU/GPU reload batches1/17/1024. It must not enter the frozen local blend or reopen selection. The local release is now frozen and in verification. Keep the monitor active until BOTH are complete/verified or explicitly failed.

Inspect with:

    .venv/Scripts/python.exe -X utf8 scripts/kaggle_private_job.py status cloud/fold_control
    .venv/Scripts/python.exe -X utf8 scripts/kaggle_private_job.py output cloud/fold_control

On completion require cloud/fold_control/output/fold_control/fold_control_report.json status=passed, fixed3epochs/419780rows/zero validation labels, unchanged protected dependencies, finite predictions, CPU/GPU native parity, and output/source hashes. Record actual fit+export timing separately from install/preparation/inference and avoid comparing it with the local three-fold end-to-end runtime. If it fails, inspect saved logs/output before a narrow repair; never blindly repush the same immutable slug. No subsequent search run is registered. One possible future hypothesis is discussed in research/second_pass_post_confirmation_review.md and requires a separate matched cloud protocol.

Initial cloud GPU budget2hours total, one GPU session plus one CPU session at a time. All cloud exploration must end before16:00UTC; use explicit session timeouts shorter than remaining time. No fleet or arbitrary seed sweep. Keep controls out of today's fixed local selection. Further candidate admission requires a concrete preregistered hypothesis and review of native portability, frozen split identity and release integration.

## Operations

Use .venv/Scripts/python.exe. Local control loads the pre-existing token only inside its process. Never copy it into a kernel or dataset. A push launches immediately with installed Kaggle2.2.4. push_intent.json is written before mutation; if it exists without a receipt, inspect remote state rather than blindly retrying.

The installed high-level status/log/output methods ignore version suffixes. A real SDK status request with version_label="1" returned404, while the same immutable slug's latest status succeeded. Use unique one-push version1 slugs; do not rely on unverified version-bound requests. KernelID is not SessionID. Cancellation requires a verified kernel_session_id; never guess from kernel_id. Bounded provider timeouts are the fallback. All four controls are COMPLETE. Quota serialization also drops whole days in timedelta.to_json output: calculate with timedelta.total_seconds(), which correctly shows approximately30hours rather than6hours.

For existing probes:

    .venv/Scripts/python.exe scripts/kaggle_cloud_control.py status cpu
    .venv/Scripts/python.exe scripts/kaggle_cloud_control.py status gpu
    .venv/Scripts/python.exe scripts/kaggle_cloud_control.py output cpu
    .venv/Scripts/python.exe scripts/kaggle_cloud_control.py output gpu

The15-minute heartbeat must check both local campaign and any later cloud receipts. Do not pause while an owned cloud job is running. Final delivery still requires local release.py verification, independent all-row blend arithmetic, native raw inference, hashes and SECOND_PASS_REPORT.md.

Monitor airline-satisfaction-overnight-monitor was paused and read back as PAUSED at 2026-10-02T13:12:08.250847+00:00.
