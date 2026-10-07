# Airline satisfaction second-pass result

Verified 2026-10-02T13:00:32.577733+00:00.

Development OOF ROC AUC: **0.96136372**, fixed v1 baseline **0.96122103**. 18 completed new configurations; 13 positive-weight configurations.

Development fold AUC: 0.96090599 / 0.96130223 / 0.96189512.

Per-run paired controls, all fold scores, fit lengths and runtimes: [experiment ledger](artifacts/second_pass/experiment_ledger.csv). Full blend-search grid, including rejected mixtures: [frozen selection](artifacts/second_pass/blend/frozen.json).

These are adaptively reused development selection scores, not a fresh unbiased assessment. The exposed original audit was excluded from all second-pass selection and scoring. Full-data refits include all labels after selection freeze.

Historical v1 only: original audit 0.96116318; unexposed sensitivity audit 0.96127692 on 69,396 rows, excluding 568 smoke-exposed rows. These scores do not evaluate v2. V1 public score was 0.96093; v2 has not been submitted.

Submission: `artifacts/second_pass/final/submission.csv`, 299,844 rows. SHA256 `571479870a1b57a7a521d639f57f4751f0bfc358031f29ad384696a651be5efc`.

Verified exact IDs/order/schema, finite probability bounds, all-row independent blend arithmetic, native-model reload and raw inference on all rows. Maximum raw inference difference: 8.76e-08. GPU training is not guaranteed bitwise reproducible.

Weights:

- lgb_route: 0.02992500
- cat_route: 0.05117175
- cat_route_teacher: 0.02428659
- cat_route_teacher_d8: 0.06476425
- realmlp_te_teacher16: 0.03150000
- realmlp_cat_te_teacher: 0.09589151
- cat_route_teacher_seed2: 0.02428659
- cat_route_te_teacher: 0.11513644
- realmlp_cat_te_teacher_seed2: 0.09589151
- xgb_route_te_teacher_profiles: 0.09714637
- v2_realmlp_cat_raw_aux: 0.13500000
- v2_realmlp_cat_raw_aux_seed20261021: 0.13500000
- v2_xgb_route_te_teacher_profiles_aux: 0.10000000

Reproduce:

```powershell
.\.venv\Scripts\python.exe scripts/second_pass_release.py --phase predict --config configs/second_pass.json --input data/test.csv --output artifacts/second_pass/reproduced.csv
```
