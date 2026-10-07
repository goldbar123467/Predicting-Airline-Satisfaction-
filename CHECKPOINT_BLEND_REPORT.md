# Saved local/cloud checkpoint blend screen

Submitted anchor development AUC: 0.9614075009197 (submission56775181).

The saved models are selected-epoch outer-fold exports. No per-epoch checkpoint sequence exists, and the active500-epoch run has not completed a fold. Inner-selected models and stale/unreferenced graph files were not mixed. The active trainer was not changed.

Registered after prior individual scores were known and before these new ensemble scores: three fixed pools, three pool comparisons and12 fixed anchor mixtures. Development OOF has been adaptively reused. Original exposed audit labels and public leaderboard scores were not used.

| Pool | Pool comparison passed | Best tested alpha | Development AUC | Gain over anchor | Eligible |
|---|---:|---:|---:|---:|---:|
| short_local_cloud | True | 0.05 | 0.9614066516670 | -0.0000008492527 | False |
| long_local_cloud | False | 0.05 | 0.9614090212299 | +0.0000015203102 | False |
| all_durations_equal | False | 0.05 | 0.9614079031414 | +0.0000004022217 | False |

No candidate qualifies. The best diagnostic gain is0.0000015203102, below the required0.00001 pooled/mean-fold gain, and its long-duration pool failed the separate ensemble-level comparison. Prior standalone duration failures remain unchanged. No new native production release or competition submission was performed. Retain the fully verified submitted blend, public AUC0.96125.

Data verification covered frozen development IDs/folds, target equality, complete result/config hashes, saved OOF/test hashes, reproduction of the frozen submitted anchor, and existing complete cloud native-import provenance. Because selection failed, no additional test-label evaluation, full-data refit or unnecessary all-row raw production replay was performed.

The user renewed permission to submit one best qualifying verified current500-epoch campaign release before its existing delivery cutoff. Permission is separately hash-bound in `state/kaggle_best_blend_authorization.json`; the original statistical configs remain unchanged. The uploader still requires the unchanged incremental gain, full native/raw verification, current daily allowance, CSV deduplication and durable intent checks. If500fails these gates, retain the existing submission.

Screen time: 2026-10-02T17:42:44.901571+00:00. Protocol SHA256: `5a9f69b4433362a6b025c1957c0305a9daace85a9897ee00257facbe9451940c`. Full numerical trace: `artifacts/checkpoint_blend_v1/screen.json`.
