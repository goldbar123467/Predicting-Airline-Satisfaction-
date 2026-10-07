# Third pass batch06 selection result

The registered cloud CPU LightGBM shrinkage candidate completed all three folds in 997.927 seconds. Its development OOF AUC is 0.9596597528067, compared with 0.9596271008554 for the fixed same-image control. Pooled gain +0.0000326519512; mean-fold gain +0.0000265233752; fold changes [-1.1011938957472545e-05, 4.541940187341442e-05, 4.5162662663078557e-05]. Selected tree counts [1065, 993, 917].

Native replay passed on all 629,671 development held-out rows and all 299,844 test rows for each fold. Maximum absolute difference was 2.220446049250313e-16 at unchanged rtol1e-10/atol1e-12. Exact data/split/config/source/image/native hashes, inner/outer role IDs, and reused-control evidence were verified. An independent agent inspected all33 returned-file hashes,30 frozen-file hashes and role metadata; root separately executed native replay.

The candidate passed the paired gate, but the unchanged registered blend search retained exactly the previously submitted weights and development AUC 0.9614075009197. No new production refit or submission was performed. Retain submission56775181, reported public AUC0.96125. There is no new public score or leaderboard claim.

These scores use adaptively reused development OOF for selection. No original audit labels were scored. This is a selection-only outcome, not a new full-data production release. The previously verified production release remains available. See `state/third_pass_batch06/selection_only_outcome.json` for hashes and `artifacts/third_pass_batch06/experiment_ledger.csv` for the full selection ledger.

Recorded 2026-10-02T16:18:01.263096+00:00.
