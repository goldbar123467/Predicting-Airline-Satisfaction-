# Dropout continuation experiment, authorized October 4, 2026

The user explicitly requested launching the prepared dropout-only test. Execute a separate private free Kaggle experiment under `fixed_epoch_dropout_v1`; preserve every completed campaign and the current blend. No local GPU training, paid compute, submission, public posting, automatic promotion or full-data refit is part of this run.

## Fixed scientific contract

Both trajectories run the existing eight-member RealMLP raw-category-plus-original-auxiliary recipe with schedule horizon 16 and exactly 16 executed epochs. Keep the model/partition seeds, feature construction, train-only fitted preprocessing, batch order, learning-rate schedule, weight-decay schedule and every legacy optimizer factor unchanged. Inner monitoring is passive; there is no early stop or best-checkpoint restoration.

- Trajectory A, control: original dropout schedule throughout.
- Trajectory C, treatment: original dropout through completed epoch 4, then hold the registered dropout base at 0.05 through epoch 16, preserving unrelated layer/scope multipliers.
- Capture native epoch 4 and 16 from both trajectories. Scored logical endpoint A is control epoch 16; B is treatment epoch 4, whose prefix must match control; C is treatment epoch 16. There are nine scored outer endpoints across three folds.
- Execute three folds × inner/outer × two trajectories, for 12 fits and 192 epochs. Recreate each prefix from its registered seed. No inference export is used as a resumable training state.

Before treatment starts epoch 5, require exact portable equality of the epoch-4 network, learned/static preprocessing, optimizer state/counters, RNG states, progress and batch-order identity against its control. Verify native fixed-probe agreement at epoch 4. Process addresses and wall-clock durations are not scientific state and must not enter cross-worker fingerprints. A failed match stops the experiment rather than weakening the criterion.

Record active dropout scopes and per-update schedule evidence. Non-dropout schedules and data-order traces must match across paired trajectories; treatment dropout must match the original schedule before the boundary and the registered constant policy afterwards. Source review and generated tests must check the actual forward/optimizer path, not just requested configuration.

## Data and evaluation

Reuse the byte-identical, previously verified development-only payload tables. The original frozen split must still hash to `4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c`. Only 629,671 rows with frozen fold >= 0 are in the cloud tables. Audit/test rows and incumbent OOF are not exported. All fitted transforms remain within each registered training partition.

The frozen incumbent selection remains SHA256 `bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311`. Assess the new outer predictions locally only after all 12 fits, nine endpoints, source/input/partition/prefix checks and native receipts pass. Freeze a separate local evaluation protocol before reading new quality results.

Report all A/B/C standalone pooled and mean-fold AUC, fold AUC, log loss and Brier. Assess only the predeclared mixtures 90% incumbent + 10% of each endpoint. Treatment mixture advances to a separate confirmation only if it improves pooled and mean-fold AUC by at least 0.00001 against each of incumbent, control mixture and shared-prefix mixture, with no fold regression below -0.00002 against any. No alpha grid, adaptive calibration, checkpoint selection or automatic release.

These historically reused development folds provide exploratory evidence, not a fresh independent confirmation. A positive result supports this particular continuation intervention, not globally optimal dropout, training duration or blending weights.

## Compute and operation

`configs/fixed_epoch_dropout_policy_v1.json` binds the user authorization, prior proposal, parent registry, copied data and fixed budget. At the preflight read, Kaggle reported 105,260.243 free GPU seconds available, zero reserved and paid scaling disabled. Recheck at dispatch.

One private T4 job has a 7,200-second provider cap and 6,300-second setup/smoke/fit budget, with the remaining time reserved for return artifacts. The prior 120-epoch campaign required 45.10 minutes overall and 34.54 minutes inside adapters. Scaling to 192 epochs suggests roughly 55 minutes inside adapters plus setup, state checks and inference; this is an estimate rather than a completion guarantee.

Use the previous verified pinned Kaggle image, minimal dependency bootstrap and cloud feature compatibility policy. Run a generated production-sized CUDA smoke on cloud before any real fit, with a 330-second smoke cap. Local preparation uses only bounded CPU tests and file hashing/copying. Preserve failed attempts and original outputs.

Freeze source/input hashes before the first real fit. Use exact manifest discovery for Kaggle's expanded dataset mount, an allowlisted private payload, durable upload/push intents, new immutable version-1 dataset/kernel namespaces and remote privacy/source readback. Root owns dispatch; agents must not launch duplicate work. Retrieve selected terminal outputs through the verified signed full-path streaming approach, not the incompatible basename size-list API.

After dispatch, reuse the existing 15-minute research heartbeat for this campaign only. Keep prior monitors paused. Complete `FIXED_EPOCH_DROPOUT_V1_REPORT.md` with actual evidence, preserve the incumbent, and pause the heartbeat after verified completion or preserved terminal failure.

## Reused evidence

The October 3 source and result review in `research/fixed_epoch_cloud_result_review.md` supports the unchanged decision to isolate sustained dropout. The proposed intervention and causal limits are in the unchanged `FIXED_EPOCH_CLOUD_NEXT_EXPERIMENT.md`. This is implementation of that reviewed hypothesis, not a new method or dependency search.
