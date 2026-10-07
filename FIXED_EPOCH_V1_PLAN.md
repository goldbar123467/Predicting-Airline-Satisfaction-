# Fixed schedule/duration experiment, October 3, 2026

The user explicitly requested continuing work while retaining the current blend and making the next experiment separate schedule length from training duration. This authorizes implementing and running the bounded local experiment below. It supersedes the earlier research-only status for this new `fixed_epoch_v1` namespace. The completed research artifacts and earlier campaigns remain unchanged.

## Frozen scientific question

Use the existing raw-plus-original-auxiliary RealMLP recipe (`v2_realmlp_cat_raw_aux`), its model seed 20261005, eight members, architecture, preprocessing, learning rate, legacy optimizer semantics and zero label smoothing. Train A with horizon/execution 4/4; train C with horizon/execution 16/16 and export B from C at epoch 4. B is never a separate fit. B minus A changes the horizon at equal epochs; C minus B continues that one H16 trajectory including its later schedule. A is a new literal four-epoch reference, not the historical three-epoch refit.

For each frozen outer fold, run inner A/C with a fixed 90/10 monitor and outer A/C on all outer-training development rows: 12 actual fits, 120 epochs. Keep the original public estimator's model-seed mapping; only the inner partition seed is 20261005 plus fold. Neither inner curves nor partial outer results may change endpoints, parameters, weights or run order. Disable best-checkpoint restoration as well as early stopping. Log effective schedules, updates, train eval probes and real inner monitor metrics. Persist native A/B/C checkpoints and fitted preprocessing.

## Data and assessment

Read the frozen split IDs/folds first and restrict to fold >= 0 before loading training labels, features, sampling or fitting. Use only 629,671 development rows. Raw parquet reads filter to these IDs; audit/test rows are neither predicted nor scored. Validate the original-only auxiliary bank's immutable manifest/output hash. Learn all outer/inner transforms and category dictionaries on their respective training subsets.

No outer AUC is computed until all 12 trajectory receipts and all nine outer endpoint prediction files pass integrity and native reload verification. Evaluate A/B/C and exactly three fixed mixtures `0.90*incumbent + 0.10*arm`. The incumbent remains `0.64*v2 + 0.16*v3_RealMLP + 0.20*v3_XGB`.

C's mixture advances to a later confirmation only if pooled and macro AUC exceed each of incumbent, A mixture and B mixture by >= 1e-5 and every fold loses <= 2e-5 against each comparator. Report every arm and failure. No reweighting, release promotion, submission, full-data refit or new cloud job is part of this experiment. All scores remain adaptively reused development evidence.

## Execution and protection

- One local GPU worker at a time, one fresh subprocess per trajectory, exact owned-process identity/creation/command checks. Keep all previous model and release bytes intact.
- Before real fitting, pass reusable synthetic CPU tests and an exclusive full-path CUDA smoke covering the actual feature transformer, eight-member architecture, passive monitor, fixed horizon, prefix export and native reload. Record versions, seeds, hashes, timings and limitations.
- Freeze `artifacts/fixed_epoch_v1/registry.json`, then the hash-bound wrapper `configs/fixed_epoch_v1.json`, after implementation/tests and before first real fit. Registry includes exact source/input hashes, expected jobs, endpoints, data contracts and scientific gates.
- Give this new campaign its own absolute deadlines at registration: 90 minutes for the fitting/prediction phase and an additional 30-minute verification/report reserve. It does not inherit the expired research cutoff or extend a previous campaign. Stop on deadline, resource/integrity failure or `state/fixed_epoch_v1/STOP`; never stop because an early score looks poor.
- Native verification compares the first endpoint reload against a second model+transform reload with raw feature reconstruction and different chunking on all outer rows. Framework versus portable-export parity is separately verified by the adapter's fixed probes. Distinguish these scopes explicitly.
- The generated CUDA smoke exposed a stall on the second optimized TorchScript inference call. The new runner scopes `torch.jit.optimized_execution(False)` around native prediction and uses four CPU threads. The old production loader and all model/training operations remain unchanged. A fresh full-path smoke must verify CPU/GPU and different-chunk reload parity before dispatch. Preserve the failed smoke and bounded diagnostic evidence.
- The coordinator records durable receipts, full process ownership and logs. No automatic retry after a source/config change. Completed trajectories and endpoint files are not overwritten.

The detailed rationale is the preserved `research/next_epoch_blend_experiment.md`; new execution receipts and `FIXED_EPOCH_V1_REPORT.md` describe what actually runs. The research papers/source review from October 2 remains applicable to this unchanged method decision.
