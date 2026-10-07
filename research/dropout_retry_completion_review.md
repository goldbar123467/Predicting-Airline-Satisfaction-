# Dropout deterministic retry completion: independent provenance review

Reviewed 2026-10-04, metadata only. **No execution-provenance blocker found.** The returned receipts substantiate completion of the registered experiment and its exact-prefix gates. This review does not assess candidate quality, advancement, or blend weights.

## Scope and identity

Read AGENTS.md, DROPOUT_CLOUD_HANDOFF.md, and the Python ML quality and Kaggle skills. Inspected the preserved retrieval/assembly records and 78 small metadata files, verifying returned metadata against the original output manifest. The already verified 463,720,566-byte archive was not rehashed or extracted again. No network call, model loading, inference, real-data fitting, prediction decoding, curve/quality-metric reading, or scoring was performed. The only new outputs are this note and `state/fixed_epoch_dropout_v1/independent_completion_review.json`.

The provider identity is private `clarkkitchen/s6e10-dropout-deterministic-retry-20261004`, kernel **137074361**, version **1**. The requested slug ending `-r2` remains preserved in intent and amendment records. Its different actual slug is explicitly reconciled; it is not evidence of a second launch. Reconciliation SHA256: `65820c3046e06176c6f6130e10fbe1140d33c33204c97401a5fca077c1dcbc75`.

Retrieval recorded provider COMPLETE, internal `training_complete`, verified archive membership/hashes, and 287 returned files. Assembly recorded `assembled_unscored`. The preserved remote run state has 12 completed fits, no active child, and `outer_metrics_computed=false`. These are remote completion facts; the separate local evaluator is outside this review's scope.

## Completion and execution checks

- **12 real-data trajectories:** exactly the Cartesian product of folds 0–2, inner/outer phases and A/C. Every adapter receipt reports completed horizon 16, executed epoch 16 and captures at epochs 4 and 16. Every fit uses eight members, hidden sizes `[512,256,128]`, `ls_eps=0`, no early stopping or best-checkpoint restoration, seed 20261005 and derived sub-split seed 1581459527. All report `cuda:0`; listing two available T4 devices does not establish use of two GPUs.
- **Nine logical endpoint receipts:** A=control16, B=treatment/shared-prefix4, C=treatment16, across three outer folds. Each reports full-fold raw native reload verification, class order `[0,1]`, reference/independent chunk sizes 32,768 and 8,191, and maximum chunking discrepancy **0**. Endpoint/prediction hashes were compared as metadata against the returned manifest, without reading prediction contents or model binaries. Native inference ran in the cloud; this local review checked its receipts.
- **Six exact-prefix gates:** every treatment receipt reports all **19** components equal, exact native probe probabilities, identical prefix update-trace bytes, and an unchanged live state after observation. Every gate was recorded at completed epoch 4 before epoch 5: 5,900 updates for inner fits and 6,556 for outer fits. Each 257-row capture probe passed its native reload tolerance, maximum recorded discrepancy across the six comparisons **1.7881393432617188e-7**. Control/treatment prefix-state JSON file hashes were independently checked without decoding the stored probe probabilities. Equal full-state components do not require the two entire JSON files to share a hash because their provenance fields differ.
- **Six pre-fit pairing checks:** all recorded schema, rows, IDs, transform bytes and partition checks passed before treatment fitting. Initial network, train-index and final consumed-order hashes agree within each A/C pair. Inner fits use 377,802 training rows and 41,978 or 41,979 monitor rows. Outer fits have 419,780 or 419,781 training rows and zero monitor rows.
- **Three full-fold native-prefix checks:** A4 versus C4 covers 209,891, 209,890 and 209,890 rows respectively, totaling **629,671 development rows**, with exact recorded maximum discrepancy **0** on every fold. Each receipt explicitly records no quality-metric computation.
- **14 deterministic startup receipts:** one smoke, one controller and all 12 distinct worker fold/phase/arm identities, with unique PIDs. Every receipt binds the same reviewed hook/amendment bytes and target source, Torch 2.10.0+cu128 and CUDA build 12.8. All report strict deterministic algorithms, warnings disabled, cuDNN benchmark off/determinism on, matmul TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and CUDA not yet initialized at startup. Each worker startup precedes its trajectory start. Controller PID **51** matches the separate launch and run-state records. Historical worker identity is primarily established by each startup receipt's parsed invocation; only the last worker also remains in terminal controller state.

The mandatory generated CUDA smoke completed before the real controller started: receipt time **18:43:20.980627 UTC**, controller start **18:43:30.464769 UTC**. It executed four 16-epoch fits, 64 generated epochs, the full production architecture and feature/schema path, eight endpoint reload checks, and both exact-prefix gates. Smoke duration was **56.206436 seconds**. Maximum native reload discrepancy was **3.2782554626464844e-7**; maximum CPU/GPU reload discrepancy was **4.172325134277344e-7**, within the frozen tolerances. Both generated full-prefix comparisons recorded exact zero discrepancy. This supplies the previously missing CUDA execution evidence; it does not identify which nondeterministic kernel caused the preserved first attempt's failure.

## Timing

The cloud wrapper ran from **18:41:52.209259** to **19:54:03.412442 UTC**, or **4,331.203183 seconds (72.186720 minutes)**. The sum of the 12 adapter-reported elapsed durations is **3,718.321861 seconds (61.972031 minutes)**. Individual adapter durations range from **288.251582** to **336.195360 seconds**.

| Fold | Inner A seconds | Inner C seconds | Outer A seconds | Outer C seconds |
| --- | ---: | ---: | ---: | ---: |
| 0 | 292.534340 | 302.397144 | 322.356264 | 336.195360 |
| 1 | 292.582576 | 288.251582 | 324.149007 | 322.235925 |
| 2 | 292.697652 | 292.339242 | 319.866972 | 332.715796 |

Each inner trajectory executes 23,600 optimizer updates (1,475 per epoch); each outer trajectory executes 26,224 (1,639 per epoch), across eight vectorized members. The 12 fits therefore total **298,944 optimizer updates** and **192 epochs**. These update counts count vectorized optimizer steps, not eight independent serial calls.

Wrapper completion preceded the 6,300-second setup/smoke/fit budget, 6,840-second delivery bound and 6,900-second provider cap. The immutable original registry still records 7,200 seconds; the separately bound retry amendment narrows its operational cap to 6,900. The approximately 612.881 seconds between wrapper wall time and summed adapter elapsed time include setup, smoke, process/data preparation and surrounding verification. This subtraction is not an exclusive profiling decomposition. Neither runtime measure establishes provider-billed GPU allowance; this review made no quota API call.

## Evidence bindings and limits

| Evidence | SHA256 |
| --- | --- |
| Preserved retrieval receipt | `43e953d0e0462a0823c2c0bb9ddd31c38e3c1fc6ff1c4de99da6bb3bd905cdb2` |
| Assembly receipt | `922deb5c7b6fccea14da7e86a41336d43a7baae77c6eedfa31855e16761fba46` |
| Preserved cloud status | `19a67f1acada87d93d2afac20dd0dd3dd71646f027c1d643f30afc6b84eb9741` |
| Original output manifest | `58db0aa4d082955e5d916920950d8caf7109beaeb4bbefee5e6e1ae1d85aaa13` |
| Completion receipt | `4e2af7ed666ba2fe314da5664e66a4f58fc64b206eca42916a11ee7d7d82548d` |
| Completed manifest | `a066820e5db0f5a312f39ae4fdcbe92836ad955fe1f8f2a580931d626850b029` |
| Full generated CUDA smoke receipt | `39baace36c6dc4511935e4e7c29ddecc12ea260ee9e17df58c09295f4101ce9a` |
| Runtime amendment registration | `3445cf3a924294384465befbfbc833d70d832a1c8f0d80ad1894c3e07d33d4bb` |
| Unchanged scientific registry | `6ec58209d08edc0b29bb330a33404bbd0c1717a1eb451e0aa533aeb77e937282` |
| Unchanged payload manifest | `16d88ba105ce5e4f8c61cc22d45d3f7242ae83f8796a47fed2f22189cd4c9e5b` |
| This review's machine-readable receipt | `b6ef5b82f31113b56967d9c76d42359c26a4ef884e6e4128e08c9dcaae3c1551` |

The machine-readable review contains the full metadata hash inventory and per-fit/process summaries. Exact numerical state comparisons and native inference are cloud assertions bound to the previously reviewed implementation; this reviewer did not replay them. Full per-update intervention trace validation belongs to the frozen evaluator, and was not duplicated here. Startup flags plus the successful same-environment paired gates do not prove cross-version or cross-hardware determinism. The review supports provenance and successful protocol execution only; all quality conclusions remain the responsibility of the separately run, once-only evaluator.
