# Airline satisfaction third-pass result

Verified 2026-10-02T14:44:39.665720+00:00.

Development OOF ROC AUC: **0.96140750**, fixed v2 baseline **0.96136372**. 5 completed new configurations; 15 positive-weight configurations.

Development fold AUC: 0.96094059 / 0.96135672 / 0.96193830.

Per-run paired controls, fold scores, fit lengths and runtimes: [experiment ledger](artifacts/third_pass_batch03/experiment_ledger.csv). Full search grid: [frozen selection](artifacts/third_pass_batch03/blend/frozen.json).

These are adaptively reused development selection scores, not a fresh unbiased assessment. The exposed original audit was excluded from all third-pass selection and scoring. Full-data refits include all labels after selection freeze.

Historical v1 only: original audit 0.96116318; unexposed sensitivity audit 0.96127692 on 69,396 rows, excluding 568 smoke-exposed rows. These historical scores do not evaluate v2 or v3. V1 public score was0.96093; v2 scored0.96122. Submission receipts are separate from this frozen report.

Submission: `artifacts/third_pass_batch03/final/submission.csv`, 299,844 rows. SHA256 `ce9628e05c21d2f37bc16fdf8ca02ac7d033b9c31f83b0d6a53b982eef599654`.

Verified exact IDs/order/schema, finite probability bounds, all-row independent blend arithmetic, native-model reload and raw inference on all rows. Maximum raw inference difference: 1.12e-07. GPU training is not guaranteed bitwise reproducible.

Weights:

- lgb_route: 0.01915200
- cat_route: 0.03274992
- cat_route_teacher: 0.01554342
- cat_route_teacher_d8: 0.04144912
- realmlp_te_teacher16: 0.02016000
- realmlp_cat_te_teacher: 0.06137056
- cat_route_teacher_seed2: 0.01554342
- cat_route_te_teacher: 0.07368732
- realmlp_cat_te_teacher_seed2: 0.06137056
- xgb_route_te_teacher_profiles: 0.06217368
- v2_realmlp_cat_raw_aux: 0.08640000
- v2_realmlp_cat_raw_aux_seed20261021: 0.08640000
- v2_xgb_route_te_teacher_profiles_aux: 0.06400000
- v3_realmlp_cat_raw_aux_probability: 0.16000000
- v3_xgb_route_te_teacher_profiles_aux_probability: 0.20000000

Reproduce:

```powershell
.venv/Scripts/python.exe scripts/third_pass_release.py --phase predict --config configs/third_pass_batch03.json --input data/test.csv --output artifacts/third_pass_batch03/reproduced.csv
```
