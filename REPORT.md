# Airline satisfaction overnight result

Verified: 2026-10-02T04:10:06.733702+00:00

Submission: `C:\Users\thecl\Documents\Predicting Airline Satisfaction\artifacts\final\submission.csv` (299,844 rows).

Development OOF ROC AUC: 0.96122103.

Final audit ROC AUC: 0.9611631797287501. Full-data refit: True.

Completed candidate configurations: 30. Positive-weight configurations: 10. Best individual development OOF ROC AUC: 0.96073254.

Audit sensitivity ROC AUC excluding 568 smoke-test exposure rows: 0.9612769187921033 (69,396 rows). Development-only smoke filtering was missing in early integration checks. A retrospective review of this chat's executed commands identified the first5000 training rows (tree reload tests), first4000 (encoded LightGBM), first600 (numeric RealMLP), and a1024-row sample with seed42 (categorical networks). Conservatively excluding their union removes568 audit rows. Smoke models were separate from competition candidates and never entered model/blend selection. The current test filters to development. Frozen splits are unchanged; this secondary audit uses69396 unexposed rows.

Weights:

- lgb_route: 0.047500
- cat_route: 0.081225
- cat_route_teacher: 0.038550
- cat_route_teacher_d8: 0.102800
- realmlp_te_teacher16: 0.050000
- realmlp_cat_te_teacher: 0.152209
- cat_route_teacher_seed2: 0.038550
- cat_route_te_teacher: 0.182756
- realmlp_cat_te_teacher_seed2: 0.152209
- xgb_route_te_teacher_profiles: 0.154201

Verified exact schema, identifiers and order, finite probabilities, independently recomputed blend, and saved-model reload/inference on 256 rows. No Kaggle submission or public score claimed. Audit score applies to development-only fold ensemble. Full-data refits include audit labels only after the selection was frozen. GPU reproducibility is not guaranteed bitwise.

Development OOF is the model-selection score and is optimistic after repeated comparisons. The audit evaluates fixed models and weights; the unexposed sensitivity audit excludes the recorded smoke-test rows. Full-data refit predictions have no separately measured score.

Reproduce inference from raw passenger rows in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py --input data/test.csv --output artifacts/reproduced.csv
```

Add `--fold-ensemble` to use the evaluated development-fold models. See [RUN_PLAN.md](RUN_PLAN.md), [competition research](research/competition_evidence.md), and [method research](research/method_evidence.md) for the validation design, experiment evidence, sources, and reproduction details.

Additional full-file raw-CSV inference verification: all 299,844 predictions reproduced; maximum absolute probability difference 8.71e-08. This also recomputed the original-data teacher from its saved model. Evidence: `artifacts/final/raw_inference_verification.json`.

The exact generating source, configuration, dependency lock, data manifest and frozen selection are saved under `artifacts/final/reproduction_source`. `artifacts/final/release_provenance.json` records 35 source snapshots and 97 native-model/prediction checksums.
