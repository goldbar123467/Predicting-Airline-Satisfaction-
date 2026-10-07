# Six continuation interventions

User-authorized October 4, 2026. Immutable scientific plan: `configs/continuation_six_v1.json`, SHA256 `ff35aa6dec492f37345f09ab75c2f83b7ccc0b725ab109da8a93af34b2699f69`. Authority SHA256 `dc8e2502db03ce4e18fe7b656e0098cfd757ad159157898757ad6c53700df3fe`. Hard wall deadline October 5 at21:02:37UTC, 17:02:37Eastern. The user authorized training and submissions, not an automatic claim that any experiment will improve.

| Stage | Sole treatment change after completed epoch4 |
|---|---|
|01|Each scoped learning-rate multiplier cools linearly from its original E4 value to zero at E16|
|02|Manual decoupled decay applies parameter LR/WD factors once rather than twice|
|03|EMA of trainable parameters, one-epoch half-life, initialized at E4 and exported at E16|
|04|Binary uniform label smoothing epsilon .05, scheduled multiplier held at1|
|05|Freeze numeric/categorical embeddings and learned preprocessing, including their manual decay|
|06|Train only the final affine weight and bias, stopping both gradients and decay elsewhere|

Every stage fits a new control and treatment for all three inner and three outer folds. Both trajectories run H16/E16. Both hold dropout at its registered base .05 after E4; all other controls retain original math. A=controlE16, B=treatment's sharedE4, C=treatmentE16. Twelve fits,192epochs,nine endpoints,six exact full-training-state prefix gates,three full-fold native-prefix checks and fourteen process-start deterministic receipts. The complete generated CUDA smoke precedes real training. No partial quality reads or inner survival screening. Inputs contain only the629,671 canonical development IDs and target-free auxiliary features.

Post-E4 learning-rate changes also change the magnitude of LR-coupled decoupled decay, despite holding its WD schedule fixed. This is an expected optimizer consequence, measured from actual scoped getters each update. The decay correction is a mathematical intervention, not proof that the installed library's original behavior is a bug. EMA evidence comes mainly from other optimizers/tasks; label smoothing has weak AUROC support; the two freeze boundaries are related. See `research/continuation_six_method_evidence.md` for primary-source evidence and limits.

Sources, inputs, image, policies, tests and manifest are frozen before each launch. Experiments use fresh immutable private datasets and kernels. No duplicate save after unknown outcomes: reconcile private numeric identity/version/source/image/data first. One GPU slot; each experiment provider cap7200sec including6300sec setup/smoke/fit and delivery reserve. A conditional paired full-data refit has2400sec cap. Six of each reserve at most57,600GPU seconds,16hours. Native quota at20:53UTC showed100,818.915seconds remaining. This is a budget bound, not a guaranteed runtime.

After terminal provider status, preserve original cloud status/manifest/archive/log. Verify every returned member, sources, native artifacts, partitions, row order and actual update schedules before opening predictions for quality. Run the separately frozen evaluator once. The fixed10% mixture must improve pooled and mean-fold AUC by at least1e-5, with no fold regression worse than2e-5, against incumbent and both registered alternate mixtures. The original incumbent has development AUC .9614075009196988. Historically reused development data makes these exploratory adaptive comparisons, not independent confirmation or statistical significance.

A qualifying C mixture must additionally pass the same gain requirements against the best prior qualifying release. Freeze selection before full-data fitting. Full-data fitting may use all699,635 training labels after this freeze; no audit quality score is computed. Verify candidate native inference for all299,844 test IDs; inherit the hash-bound unchanged incumbent's completed full native proof; independently recompute .9incumbent+.1candidate. Keep original release files intact. Check live `num_allowed_now`, submission bytes, source/native inventories, deadline, maximum six attempts and durable digest intent immediately before submission. Unknown upload outcomes must be reconciled read-only, never blindly retried. Public scores are recorded but do not tune policies, stopping or alpha.

The same15-minute heartbeat advances the sequence, remains quiet on unchanged states, and pauses on verified closure or terminal deadline. GPT-6.1Sol/Low was requested; the heartbeat tool rejected model fields, so this chat needs that UI setting and the state does not claim a verified switch.
