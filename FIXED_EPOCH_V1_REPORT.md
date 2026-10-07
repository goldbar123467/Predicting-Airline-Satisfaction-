# Separating schedule horizon and training duration

Status: the local attempt terminated on its registered memory guard at 2026-10-03 01:51:16 UTC, before any epoch completed. No outer score was computed. The current blend is retained. The user subsequently requested moving the same experiment to free Kaggle cloud because the local PC is being used for Fortnite; that continuation uses the separate `fixed_epoch_cloud_v1` namespace.

## Fixed question and comparisons

| Arm | Schedule horizon | Executed epochs | Relationship |
| --- | ---: | ---: | --- |
| A | 4 | 4 | Literal four-epoch reference |
| B | 16 | 4 | Captured prefix of C, no separate fit |
| C | 16 | 16 | Continues the exact trajectory that produced B |

B minus A tests schedule horizon at equal training duration. C minus B measures continuation along one fixed 16-epoch schedule, including its later learning-rate and regularization values. It does not isolate extra updates from those later schedule values. C minus A compares the complete recipes. The historical short-run score used a selected three-epoch refit, so it is not substituted for A.

For each of three frozen development folds, inner A/C uses the existing seeded 90/10 monitor split; outer A/C fits all outer-training development rows. There are exactly 12 fits and 120 executed epochs. Monitors are passive: no early stopping, best-checkpoint restoration or result-dependent choice of endpoints. Public-estimator sub-seed mapping and outer training-row permutation are preserved. A/C must have identical initial network, training order, probe identities and fitted preprocessing within each fold/phase.

All 629,671 development rows belong to frozen folds >= 0. The runner filters IDs before reading train labels/features. Audit and test rows are excluded from this experiment. The unchanged original-only auxiliary feature bank is validated by its manifest and output hash.

## Blending and acceptance

The retained incumbent remains 64% frozen v2 blend, 16% v3 RealMLP and 20% v3 XGBoost. This experiment computes exactly three mixtures: 90% incumbent plus 10% A, B or C. It performs no weight search.

C's mixture advances to a separate confirmation only if it improves both pooled and mean-fold AUC by at least 0.00001 against each of the incumbent, A mixture and B mixture, with no fold loss greater than 0.00002 against any comparator. Every criterion and arm will be reported. Passing this rule does not promote a release, refit on all labels or submit to Kaggle. Historically reused development OOF is not an independent assessment.

No outer metric may be computed before all 12 fits and nine outer endpoint prediction files pass the registered integrity checks. The one-time evaluator reconstructs the unchanged incumbent from verified saved OOF files and validates keyed row identities and labels before scoring.

## Executed readiness evidence

- Four adapter CPU tests passed: literal prefixes, capture noninterference in weights/Adam/RNG/batch order, public seed/order mapping, native export and exclusion/overwrite/deadline guards.
- Six runner tests passed: filtered reads, ID contracts, exact partitions, parity tolerance, owned-child failure and owned process-tree deadline termination.
- Nineteen evaluator tests passed: complete inventory before metrics, row/label/split contracts, native metadata/graph/transform binding, exclusions and every fixed gate comparator.
- The full eight-member CUDA smoke passed in 41.453 seconds on generated inputs using actual 52-feature construction, 22 categorical columns and architecture [512, 256, 128]. Inner C16, outer A4 and outer C16 produced five verified endpoints. Maximum GPU chunk/reload difference was 2.980232238769531e-7; maximum CPU/GPU difference was 5.364418029785156e-7, under atol 2e-6 plus rtol 1e-5. This proves the generated path, not full-data throughput or model quality.
- The first CUDA smoke failed its 300-second budget after training completed. A bounded follow-up reproduced a stall on the second loaded TorchScript optimized prediction, even with four CPU threads and the same input shape. Disabling executor optimization completed four calls in 0.860 seconds. The new runner scopes this workaround to native inference. The exact C++ cause remains unknown. Failed evidence is preserved and old production loaders are unchanged.
- Independent review verified the CUDA source/artifact hashes and adapter/evaluator contracts. Before registration, 92 prior research files and 17 frozen input/source contract checks still matched their recorded hashes.

Full-fold native verification compares a first saved-endpoint reload with a second model/transform reload, raw feature reconstruction and different chunking. Framework-to-portable-export parity is separately tested on fixed capture probes. These scopes are not interchangeable.

## Execution and results

The registry was frozen at 01:50:38 UTC and bound 104 source files and 165 inputs. Its SHA256 is `3118735d18dc0bd77145bfcefd42344c3fcc896e8007e38026807cfd60c2fef0`; the wrapper SHA256 is `c0e659a3c48a07cabd93e640e0ddf5390e3a8bdfcd2c7039ba83bc23fe02b381`. The controller launched at 01:50:58 UTC and terminated its identity-verified first worker at 01:51:16 when available system RAM fell below 512 MiB. The first inner A partition had 377,802 training rows and 41,978 monitoring rows. Its receipt records zero completed epochs and no saved endpoints; the campaign completed zero fits. This is an operational failure, not evidence about AUC or useful training duration.

All local attempt files are preserved. There is no local retry and no final evaluation for this incomplete campaign. The subsequent explicit cloud authorization is recorded in `configs/fixed_epoch_cloud_policy_v1.json`: one private free T4 job, maximum two hours, all A/C arms on the same cloud environment. See the separate cloud report/receipts for continuation; its results must not be described as a completed local experiment.

Detailed contracts and evidence: `FIXED_EPOCH_V1_PLAN.md`, `research/fixed_epoch_v1_evaluation_contract.md`, `research/fixed_epoch_v1_preflight_review.md`, `research/fixed_epoch_v1_native_inference_diagnosis.md`, and `artifacts/fixed_epoch_v1/`.
