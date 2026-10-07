# Development OOF diagnostics

Generated: 2026-10-02T03:55:53.495492+00:00

development OOF only; no audit labels, audit scores, or test predictions read.

629,671 development rows. Best completed single: `realmlp_cat_te_teacher_seed2`.

| Model | Pooled AUC | Fold 0 | Fold 1 | Fold 2 | Delta vs best |
|---|---:|---:|---:|---:|---:|
| lgb_baseline | 0.95859646 | 0.95798595 | 0.95876688 | 0.95905444 | -0.00213607 |
| cat_baseline | 0.95814852 | 0.95766252 | 0.95816266 | 0.95864378 | -0.00258402 |
| xgb_baseline | 0.95852033 | 0.95809659 | 0.95863091 | 0.95885213 | -0.00221221 |
| lgb_route | 0.95904038 | 0.95850962 | 0.95918317 | 0.95943811 | -0.00169216 |
| cat_route | 0.96011225 | 0.95980430 | 0.95994067 | 0.96060016 | -0.00062029 |
| lgb_route_teacher | 0.95918918 | 0.95843844 | 0.95952877 | 0.95977525 | -0.00154336 |
| cat_route_teacher | 0.96065770 | 0.96008668 | 0.96061102 | 0.96130296 | -0.00007484 |
| lgb_route_teacher_63 | 0.95938902 | 0.95864153 | 0.95959029 | 0.95996951 | -0.00134352 |
| xgb_teacher | 0.95878256 | 0.95820903 | 0.95886153 | 0.95932948 | -0.00194998 |
| xgb_route_te | 0.96001132 | 0.95966922 | 0.95992203 | 0.96044824 | -0.00072122 |
| lgb_route_teacher_127 | 0.95944899 | 0.95873558 | 0.95960593 | 0.96004116 | -0.00128355 |
| xgb_route_te_teacher | 0.96052108 | 0.96021227 | 0.96042782 | 0.96094439 | -0.00021145 |
| cat_route_teacher_d8 | 0.96061518 | 0.96008625 | 0.96063594 | 0.96115897 | -0.00011736 |
| xgb_features | 0.95840961 | 0.95794566 | 0.95854124 | 0.95875951 | -0.00232293 |
| realmlp_teacher | 0.95857101 | 0.95792294 | 0.95868914 | 0.95915165 | -0.00216153 |
| tabm_baseline | 0.95633829 | 0.95633902 | 0.95685220 | 0.95721093 | -0.00439424 |
| tabm_ple_teacher | 0.95779075 | 0.95751922 | 0.95820029 | 0.95871100 | -0.00294179 |
| realmlp_te_teacher | 0.96063332 | 0.96017926 | 0.96058675 | 0.96116645 | -0.00009921 |
| tabm_cat_teacher | 0.95952849 | 0.95906928 | 0.95977757 | 0.95992837 | -0.00120405 |
| tabm_cat_ple_te_teacher | 0.95931786 | 0.95965392 | 0.96018188 | 0.96046450 | -0.00141468 |
| realmlp_te_teacher16 | 0.96037047 | 0.95999738 | 0.96012240 | 0.96103715 | -0.00036207 |
| realmlp_cat_te_teacher | 0.96062671 | 0.95999200 | 0.96069287 | 0.96121098 | -0.00010583 |
| cat_twins_teacher | 0.96036145 | 0.95994811 | 0.96025008 | 0.96089918 | -0.00037109 |
| cat_route_teacher_seed2 | 0.96066883 | 0.96012079 | 0.96063195 | 0.96130361 | -0.00006371 |
| cat_route_te_teacher | 0.96071452 | 0.96017363 | 0.96066949 | 0.96131162 | -0.00001802 |
| xgb_route_te_teacher_d6 | 0.96053768 | 0.96023891 | 0.96042436 | 0.96097218 | -0.00019485 |
| realmlp_cat_te_teacher_seed2 | 0.96073254 | 0.96012729 | 0.96080545 | 0.96127656 | +0.00000000 |
| tabm_cat_teacher_wd003 | 0.95964592 | 0.95941946 | 0.95978716 | 0.95992637 | -0.00108662 |
| xgb_route_te_teacher_profiles | 0.96067453 | 0.96031187 | 0.96066730 | 0.96106078 | -0.00005801 |

Current blend AUC: **0.96122103**, delta vs best: **+0.00048849**.
Fold deltas: +0.00061093, +0.00035172, +0.00050289.
Blend contains every completed candidate in this snapshot: True.

| Slice | Rows | Positive rate | Best single AUC | Blend delta |
|---|---:|---:|---:|---:|
| travel: Business travel | 447,829 | 0.584 | 0.957371 | +0.000644 |
| travel: Personal Travel | 181,842 | 0.097 | 0.831550 | +0.001816 |
| class: Business | 308,181 | 0.725 | 0.939965 | +0.001018 |
| class: Eco | 294,490 | 0.168 | 0.904770 | +0.001082 |
| class: Eco Plus | 27,000 | 0.242 | 0.930589 | +0.001419 |
| gender: Female | 313,095 | 0.438 | 0.961468 | +0.000513 |
| gender: Male | 316,576 | 0.449 | 0.959972 | +0.000466 |
| age_bucket: 0-20 | 53,155 | 0.163 | 0.904521 | +0.001331 |
| age_bucket: 21-35 | 198,465 | 0.337 | 0.953897 | +0.000555 |
| age_bucket: 36-50 | 232,278 | 0.554 | 0.958520 | +0.000516 |
| age_bucket: 51-65 | 133,871 | 0.552 | 0.961120 | +0.000622 |
| age_bucket: 66+ | 11,902 | 0.098 | 0.865831 | +0.001336 |
| arrival_delay_missing: missing | 266 | 0.380 | 0.985359 | -0.001200 |
| arrival_delay_missing: observed | 629,405 | 0.444 | 0.960721 | +0.000490 |

AUC deltas use identical rows and folds. Weight selection uses these OOF rows, so these are development diagnostics, not an independent generalization estimate.

The JSON artifact includes complete model and residual correlation matrices, per-model slice scores, input hashes, and the supervisor process-tree snapshot.
