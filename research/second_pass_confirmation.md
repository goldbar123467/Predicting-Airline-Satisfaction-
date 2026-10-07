# Second-pass seed confirmation decision

Decision recorded 2026-10-02T12:36:07.037381+00:00, before queue append and before any confirmation fit.

All 17 primary configurations completed and appear in the saved blend created 2026-10-02T12:32:30.252643+00:00. The supervisor was idle, the release was not frozen, no confirmation existed, and the decision was before 2026-10-02T15:15:00Z.

Select `v2_realmlp_cat_raw_aux`. Its fixed-anchor mixture at alpha 0.30 gains +0.000106591990 pooled and +0.000106904229 mean-fold AUC. Fold deltas: +0.000124640938 / +0.000111458176 / +0.000084613573.

Append only `v2_realmlp_cat_raw_aux_seed20261021`, copying the exact source configuration and changing ID plus seed to 20261021. Declare `v2_realmlp_cat_raw_aux_fixed_seed_average` with equal weights on `v2_realmlp_cat_raw_aux` and `v2_realmlp_cat_raw_aux_seed20261021` before fitting. Existing 47 run objects and all other configuration fields remain unchanged.

## All registered recipes, best fixed-anchor mixture

| Recipe | Alpha | Pooled gain | Mean-fold gain | Fold 0 / 1 / 2 deltas | Strong gate |
|---|---:|---:|---:|---|---|
| v2_realmlp_cat_te_teacher_profiles | 0.05 | +0.000000192461 | +0.000000186645 | -0.000001867788 / +0.000004571067 / -0.000002143344 | fail |
| v2_lgb_route_te_teacher_127 | 0.05 | -0.000011934235 | -0.000004112664 | -0.000007033382 / -0.000002282958 / -0.000003021651 | fail |
| v2_lgb_route_te_teacher_profiles_127 | 0.05 | -0.000003567154 | -0.000003387211 | -0.000006408272 / -0.000001702079 / -0.000002051283 | fail |
| v2_cat_route_te_teacher_d6 | 0.05 | -0.000002073484 | -0.000002035385 | +0.000003239038 / -0.000004920091 / -0.000004425102 | fail |
| v2_cat_route_te_teacher_l220 | 0.05 | +0.000000396295 | +0.000000532007 | +0.000001369963 / -0.000000735572 / +0.000000961631 | fail |
| v2_xgb_route_te_teacher_profiles_aux | 0.20 | +0.000022830915 | +0.000022110327 | +0.000021852265 / +0.000034299189 / +0.000010179527 | fail |
| v2_realmlp_cat_te_teacher_aux | 0.10 | +0.000009042625 | +0.000009198005 | +0.000003132631 / +0.000018224028 / +0.000006237356 | fail |
| v2_realmlp_cat_raw | 0.30 | +0.000070100642 | +0.000072626048 | +0.000074571821 / +0.000117615349 / +0.000025690974 | pass |
| v2_realmlp_cat_raw_aux | 0.30 | +0.000106591990 | +0.000106904229 | +0.000124640938 / +0.000111458176 / +0.000084613573 | pass |
| v2_lgb_route_teacher_aux_127 | 0.05 | +0.000007790353 | +0.000007621275 | +0.000003505287 / +0.000007973938 / +0.000011384601 | fail |
| v2_cat_route_te_dual_teacher | 0.05 | +0.000000304336 | +0.000000110878 | +0.000000765270 / -0.000005210255 / +0.000004777620 | fail |
| v2_realmlp_cat_te_dual_teacher | 0.05 | -0.000000795236 | -0.000000867499 | -0.000005155662 / +0.000002904304 / -0.000000351139 | fail |
| v2_realmlp_cat_te_teacher_wd030 | 0.05 | -0.000001299625 | -0.000000895735 | -0.000004234045 / +0.000004767514 / -0.000003220672 | fail |
| v2_realmlp_cat_te_teacher_lr0265 | 0.05 | +0.000000957960 | +0.000001013415 | -0.000000849421 / +0.000004759880 / -0.000000870213 | fail |
| v2_realmlp_cat_te_teacher_emb10 | 0.05 | +0.000000055049 | -0.000000049925 | -0.000003988765 / +0.000004385289 / -0.000000546298 | fail |
| v2_xgb_route_te_teacher_profiles_te100 | 0.10 | +0.000004603257 | +0.000004781917 | +0.000011894009 / +0.000011901218 / -0.000009449475 | fail |
| v2_realmlp_cat_profiles_teacher_no_te | 0.05 | -0.000001560470 | -0.000001529558 | -0.000003305531 / +0.000005819369 / -0.000007102511 | fail |

All 68 step-0 mixtures, scores, individual gate booleans, source/result hashes, exact clone and group declaration are in `state/second_pass/confirmation_decision.json`. Raw-only RealMLP is the other qualifying recipe; raw+aux has the stronger pooled gain. No other primary recipe qualifies.

Source runtime was 256.359 seconds, selected epochs [3, 3, 3]; the unchanged confirmation cap is 1200 seconds. The supervisor alone starts the job.

The group must pass the ordinary anchored blend gates before use. An incomplete registered group is withheld by the release script. Do not compare seeds and choose the better one. No extra method, hyperparameter, epoch, representation or seed search is authorized by this decision.

Development selection is adaptively reused. Seed changes model randomness and inner stopping split. Confirmation is assessed only through the preregistered equal average; never select the better seed. Original audit/test labels were not read.

Pre-append config SHA256: `d5a7c3d9cca03e3cf9e8adb34b5ff1e1cec24bd6c6986a6e7960d91117021a9c`. Blend snapshot SHA256: `63a0aa6f95666ff2a1fab61f348427778ea1495751bd31986ad05a52839660ba`.
