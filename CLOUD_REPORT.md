# Kaggle compute verification

Completed October2,2026 at approximately13:10UTC. All five private cloud jobs completed successfully. No cloud result entered the frozen local blend, no audit labels were used, and no competition submission was made.

## Measured full-fold control

The [private RealMLP control](https://www.kaggle.com/code/clarkkitchen/s6e10-realmlp-fold0-control-20261002) trained419780 frozen outer-training development rows,39features,3fixedepochs and8internal ensemble members on one TeslaT4. Exact local preprocessing matrices, source hashes and seed20261005 were fixed before fitting. Only1024 label-free held-out feature rows accompanied the training data; no held-out AUC was computed.

- Fit plus native export:84.489seconds. Package installation:6.589seconds. Entire correctness/timing job:484.344seconds.
- Peak CUDA allocated:1,402,843,136bytes; reserved:2,378,170,368bytes. Total host RAM on the instance:33,658,318,848bytes, with4CPU cores. The instance provides two15GiB T4s; this control deliberately used only one.
- Native export, CPU/GPU reload and prediction comparisons for batches1/17/1024 passed, maximum error1.19209290e-7.
- Downloaded native hashes matched. The cloud-trained model then loaded successfully on this Windows machine with Torch2.14.1+cu130, using CPU inference. All1024 keyed rows passed at the same three batch sizes, maximum difference1.19209290e-7 versus cloud predictions.
- Protected cloud Torch2.10.0+cu128, CUDA and numerical-library versions were unchanged. The reusable setup installs PyTabKit1.7.3 and pytorch-lightning2.6.6; torchmetrics1.9.0 was already compatible.

GPU inference timing is not a steady-state benchmark. The warm-up used batch17 before timing different shapes: CUDA batch1 took250.975seconds, batch17 took0.217seconds and batch1024 took130.731seconds for1024rows. Shape-specific initialization or compilation may contribute, but the cause was not profiled. Do not interpret those values as ordinary sustained GPU throughput or claim a local-versus-cloud speedup. Cloud CPU batch1024 inference took0.068seconds, and the same returned model took0.047seconds locally, each on these1024rows. Correctness passed despite the timing anomaly. The84.489second figure measures fit plus export, not the entire job or a matched comparison to local three-fold CV.

## Other executed checks

Private CPU and T4 environment probes verified competition file hashes, actual hardware and generated-data XGBoost/CatBoost/LightGBM/Torch native save/reload. A separate eight-member RealMLP synthetic test verified train-only category vocabulary, unknown values and CPU/GPU native export. A frozen local RealMLP model also reproduced1024 raw test predictions on Kaggle CPU within1.78813934e-7. Both cloud-to-local and local-to-cloud inference directions have therefore been exercised with saved models.

Approximately29.84 GPU hours remained after the five jobs, with no reserved time. The exact account snapshot, all job references and report hashes are in state/kaggle_cloud/run_state.json. CPU weekly quota and enhanced accelerator eligibility were not established. No paid compute was used.

## Evidence and reuse

- Full-fold report: cloud/fold_control/output/fold_control/fold_control_report.json.
- Local return check: cloud/fold_control/local_return_verification.json.
- Immutable preparation and split/source checks: cloud/fold_control/preparation_manifest.json and static_validation.json.
- Package bootstrap: scripts/kaggle_realmlp_probe_prepare.py and cloud/realmlp_probe/requirements-realmlp.txt. Fresh Kaggle sessions must run the pinned setup again.
- Local verified13-member release: SECOND_PASS_REPORT.md, development OOF0.9613637193. It is an adaptive development selection estimate, not a new public leaderboard result.

The next defensible research hypothesis is an independently implemented observed-rating probability block from the existing original-only feature models, assessed against a matched cloud control. It remains unrun and outside this completed campaign. No new sweep or additional cloud job is queued.
