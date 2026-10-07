# Third pass cloud duration result

The registered60-epoch cloud RealMLP run completed all three folds in 3568.372 seconds (59.47 minutes). Each inner fit traversed the registered60-epoch horizon, selecting epoch4; each outer model was freshly fitted for4epochs. The longer horizon also changes the normalized schedule, so this is a duration/schedule comparison.

Development OOF AUC 0.9606847043183 versus frozen same-image four-epoch control 0.9608107752681. Pooled change -0.0001260709498, mean-fold change -0.0001294556258; fold changes [-0.00019039523450448126, -0.00017095392054977676, -2.701772228141941e-05]. All three exceed the permitted fold regression. The candidate fails the registered paired gate and is excluded before blending. No production refit or submission.

Root executed native replay on all629,671 development held-out rows, maximum absolute error3.5762786865234375e-7, within unchanged rtol1e-5/atol2e-6. Source/image/runtime/config/data/native proof passed. Independent review confirmed42 returned-file hashes,33 frozen-file hashes, all role IDs and the fixed control. These are adaptively reused development selection scores; no exposed audit labels were scored.

All seven project-owned private cloud jobs were freshly confirmed COMPLETE at16:26UTC. All three known competition submissions are processed. Latest retained submission56775181 has public AUC0.96125; no new leaderboard claim. Latest completed local batch05 release remains fully verified on299,844 raw test rows with independent blend recomputation; batch06 CPU shrinkage likewise produced no incremental blend gain.

The separately authorized500-epoch local campaign remains queued for17:05UTC (13:05Eastern), with a20-hour ceiling through October3 13:05UTC (09:05Eastern), and no automatic overnight submission. The60-epoch results provide no evidence that additional epochs improve this recipe.

Recorded 2026-10-02T16:28:49.133007+00:00. Detailed evidence: `cloud/third_pass_long/admission_decision.json`, `state/long_local_500/prior_cloud_terminal`, and `state/long_local_500/prior_submissions_checked.json`.
