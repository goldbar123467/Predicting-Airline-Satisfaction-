# Local 500-epoch experiment outcome

All three registered folds completed successfully on October 2, 2026. Primary training ran from 16:51:40 to 20:39:56 UTC, with measured fit time 13,688.797 seconds (3 hours 48 minutes). The 20-hour authorization was a ceiling, not a minimum runtime.

| Development metric | 500-epoch horizon | Four-epoch control | Difference |
|---|---:|---:|---:|
| Pooled AUC | 0.9607352453 | 0.9608520596 | -0.0001168143 |
| Mean fold AUC | 0.9607474591 | 0.9608586157 | -0.0001111566 |
| Fold 1 | 0.9602640874 | 0.9604011182 | -0.0001370308 |
| Fold 2 | 0.9608008319 | 0.9608477276 | -0.0000468957 |
| Fold 3 | 0.9611774580 | 0.9613270014 | -0.0001495433 |

Each inner fit traversed the registered 500-epoch horizon and selected epoch 4. The fresh outer models then fitted four fixed epochs. Inner fitting took 4,591.031 / 4,485.359 / 4,452.593 seconds. This compares normalized schedule horizons and checkpoint selection; it does not compare an identical update sequence stopped at two times. No per-epoch checkpoint sequence was saved.

The paired admission gate failed: pooled and mean-fold changes are negative, and all three folds regress beyond the permitted 0.00002. The candidate was excluded before blending. The frozen 15-model blend has exactly the same weights and development scores as the already submitted third_pass release: pooled AUC 0.9614075009. These are adaptively reused development scores, not an unbiased fresh assessment. Original exposed audit labels were not used for selection or scoring.

The isolated batch07 release finished verification at 20:52:43 UTC. Checks passed for exact schema, IDs/order, finite probabilities, native model reload, raw inference on all 299,844 test rows, independent blend recomputation, and archived source/native hashes. Maximum raw inference difference was 1.06025e-7. The release CSV SHA256 is `3bcc2627880c1bfdf02f6e9f4304795c7cd41648db08d12d51e5cae7e9afe344`. Refitting on GPU can change prediction bytes without providing a new development gain.

The authorized submission helper executed successfully and returned `not_submitted`, because incremental development gain is zero. No new upload was made. Retain submission **56775181**, freshly confirmed COMPLETE with public AUC **0.96125**. No new leaderboard gain or current rank is claimed. All seven project-owned cloud jobs were freshly confirmed COMPLETE at 20:54 UTC, and no project Python worker remained. The requested campaign is resolved; the existing monitor can now be paused.

Independent review checked exact registered recipes, split/source/control hashes, three fold records, six referenced native graph hashes, ledger exclusion, and exact equality of the selected blend to the submitted one. The reviewer did not fit models, load audit data or issue API/submission calls. Root separately inspected the completed all-row verification and revalidated both batch07 and retained-release provenance.

Evidence:

- [Primary outcome](state/long_local_500/primary_outcome.json)
- [Frozen release report](THIRD_PASS_BATCH07_REPORT.md)
- [Verification](artifacts/third_pass_batch07/verification.json)
- [Submission resolution](state/third_pass_batch07/submission_resolution.json)
- [Worker and submission closure](state/long_local_500/final_workers_and_submissions.json)
