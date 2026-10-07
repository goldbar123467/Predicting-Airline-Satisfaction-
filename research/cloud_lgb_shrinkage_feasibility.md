# One slower-shrinkage LightGBM contrast, 2026-10-02

Decision at 15:37 UTC: **a conditional yes to one private CPU contrast**, provided the complete CV/import/production contract and independent review are ready for dispatch by 15:50 UTC. No fit, upload, remote mutation, or new label analysis was performed for this proposal. The user explicitly requested continued cloud experiments; this is not a proposal to occupy quota without a testable question.

## Evidence and hypothesis

The verified cloud q/log LightGBM result `v3_cloud_lgb_route_teacher_aux_probability` took **230.337701714 seconds for all three nested folds**. Its inner-selected outer-refit tree counts were **[201,322,217]**, compared with a 2600-round ceiling. Pooled AUC was 0.9596271008554396, macro AUC 0.9596417817632824. Against the feature-matched EV-only cloud run, gains were +0.0000257686332061 pooled, +0.0000274466474777 macro, with worst fold -0.0000039838791191. These are saved development results, not a new independent assessment.

All 629671 heldout rows and every fold's 299844 test rows passed native CPU replay, maximum prediction difference 2.220446049250313e-16. The bounded verifier observed 701022208 bytes RSS at sampled boundaries; this is not a continuous peak claim. The third-pass batch04 blend remained exactly 0.9614075009196988, and neither cloud LightGBM entered it. Thus the standalone positive q result has **not** established ensemble usefulness.

The proposed hypothesis is that reducing the contribution of each 127-leaf tree changes the boosting trajectory enough to make the fixed q/log variables useful at finer increments, while keeping the individual tree complexity and all input information fixed. This is a shrinkage/trajectory experiment, not a remedy for demonstrated underfitting: early optimal round counts alone cannot diagnose underfitting or overfitting. A second weak result should close this branch, not trigger a rescue sweep.

The current official [LightGBM 4.7.0 tuning guide](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html), read 2026-10-02, recommends pairing smaller learning rates with more iterations and explains that leaf-wise trees can overfit without suitable complexity controls. The [4.7.0 parameters reference](https://lightgbm.readthedocs.io/en/stable/Parameters.html) defines learning rate as shrinkage. These support a bounded test, not a prediction of competition improvement. The previously reviewed LightGBM methods evidence remains in `research/cloud_lgb_probability_feasibility.md`; no new algorithm or dependency is needed.

## Exact proposed recipe

Reuse the now-verified q/log cloud run as the immutable comparator. Do **not** rerun it, change its receipt, or substitute its EV-only predecessor. Candidate suggested ID: `v3_cloud_lgb_route_teacher_aux_probability_lr01`.

Only statistical changes from that comparator:

- `params.learning_rate`: 0.03 to **0.01**.
- `max_rounds`: 2600 to **8000**.
- `patience`: 150 to **300**.

Use 8000 rather than 6000 because 7800 rounds at 0.01 is the rough shrinkage-horizon counterpart of 2600 at 0.03. It is only a ceiling, not a promise to run all 8000. Patience300 represents a smaller shrinkage-time wait than the previous150 at 0.03; call this a coupled learning-rate/stopping recipe, not a perfectly isolated learning-rate ablation. No new stopping metric, tolerance, or outer-label decision is allowed. Changing patience to450 for exact shrinkage-time equivalence would be a separate proposal and is not recommended after this preregistration.

Everything else is identical: seed20261003, original three outer folds, exact precomputed inner90/10 IDs, 127 leaves, min_child_samples80, colsample_bytree0.85, subsample0.9/freq1, reg_lambda6, cat_smooth30, cat_l2=10; raw21 + route category + teacher + EV13 + q13/log; unchanged canonical cache bytes and training-only vocabularies. Pin LightGBM4.7.0, CPU4threads, and the exact verified CPU image:

`gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`.

Comparator bindings to freeze before fitting:

- result SHA256: `03786f9699e97c97f3ad5d13b94e4e964133141036eb75aabea1e22524726f31`.
- native import receipt SHA256: `722ede7c82e787897f8418f3c273f4ede581b8c04eb3322eac7d3111bd37d83e`.
- OOF SHA256: `a12070790cad20b1f434cee941a1d2810aa1afa68c0b7780437d02f9dc699a3b`.
- test SHA256: `75ed98626be98d7001ac44f3b8fdb42c2b377207cd26da3a99d2057585e79668`.
- split SHA256: `4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c`.

Also verify every comparator source/native/replay-map hash. The comparator becomes control-only for this contrast and is excluded from new selection.

## Measured-cost estimate and deadline

Let the sum of existing best iterations be S=740. Approximating inner training as best+patience and outer fitting as best gives old work 2S+3*150=1930 tree-iterations. A threefold increase in selected iterations and patience300 gives 6S+3*300=5340, or **2.7668 times** that work. Applying this to the entire measured230.34-second run conservatively gives about **637 seconds**. The simpler 3x estimate is691 seconds. Both include multiplying fixed feature/IO costs unnecessarily; neither accounts for changes in stopping behavior or cloud speed. Expect roughly **11â€“15 minutes plus unpredictable platform queue/startup**, plausibly under20 minutes. Do not label this a measured .01 benchmark.

The remaining uncertainty is real: selected rounds need not scale by exactly three, and an 8000-round ceiling could exceed the deadline. Enforce **1800 seconds total including bootstrap**, plus an absolute CV deadline **16:20 UTC**, whichever comes first. An incomplete fold set is ineligible; never truncate the selected number of outer/refit rounds to fit the time budget. No AUC-based early cancellation using outer folds.

Recommended checkpoints:

| Checkpoint | UTC |
|---|---|
| Reviewed full pipeline ready and CPU dispatch | <=15:50 |
| CV terminal hard limit | <=16:20 |
| Bounded native replay, matched/anchor gates, positive freeze if any | <=16:25 |
| Same-image fresh fixed-round CPU production, 900-second cap | <=16:40 |
| Independent native test return and all-row raw ensemble verification | <=16:55 |
| User deadline | 17:00 |

A full-data median-tree refit around651 rounds is plausibly much cheaper than six nested fits. There is no measured full-data .01 production timing yet. Register a conservative pre-label release-time check, reserve900 seconds for that fit and the existing roughly11-minute raw-verification allowance, and retain the current release if the forecast or measured availability does not fit. The separate long GPU recipe shares the final verification budget; root must serialize promotion rather than assume two full releases can both consume the same reserve.

## Reuse and proof required before launch

The existing LGB feature builder, transform, fit/predict/native IO and bounded replay are directly reusable. A new private protocol/package must preserve those functions and source hashes; only scoped administrative code changes are needed. The current frozen engine hardcodes600 seconds per CV recipe, and its runtime/production helpers have old absolute deadlines, protocol IDs and artifact roots. Therefore this cannot safely be dispatched by merely editing one JSON value. Use a new folder and new candidate-only package; do not mutate the completed bundle, original importer, comparator, or receipts.

Before dispatch, finish and review: new cap/deadline guards, exact comparator binding, candidate-only execution, same-image post-freeze fresh fixed-round production support, and generated-data nested-CV/import/native tests. The source functions for fit, features and prediction should remain byte- or AST-equivalent. Reuse the existing private data bundle via byte copies where possible: same629671 development-only labels, same299844 test rows, exact inner IDs, no69964 audit rows or labels. Keep the existing allowlist/native-text/JSON-only transport and pinned dependency bootstrap. Avoid importing executable files from returned output.

After CV, full heldout and each fold's full-test native replay are mandatory at the unchanged rtol1e-10/atol1e-12; archive the bounded verifier and its source hash. Only then may root wire the new immutable pins into release admission. Preserve the existing no-local-refit guard. If selected, production must use fresh training in the verified cloud image at the frozen median of all three selected round counts, then independent local native and raw-bank verification. No `Booster.refit` leaf updates and no label access before freeze.

Unchanged matched gate versus the verified q comparator: pooled and macro gains >=0, worst fold >=-2e-5. Then the existing alpha[.05,.10,.20,.30] search must pass pooled/macro >=1e-5 and worstfold >=-2e-5 against both the frozen-v2 anchor and the best current development blend. Maximum two new units remains in force. No renewed global confirmation allowance, no audit/public-score selection, and no further learning-rate branch if this candidate fails. This remains adaptive development selection with a potentially low incremental return.

## Recommendation

Proceed with preparation only if root approves this exact preregistration and can fit the complete engineering/review work before15:50. Given the verified native CPU path, this is a smaller and better bounded risk than introducing a new family. Expected ensemble improvement is modest and unproven. If the complete pathway is not ready, or late cloud queueing removes the release margin, skip the fit and preserve the verified release.

Root approved the exact combined contrast before fitting. Final production deadline overrides the earlier proposal: latest start16:25, 900-second hard cap, absolute end16:40; preregistered forecast90+0.75*median_selected_rounds<=900. This allows median<=1080 and never truncates a larger selected count. New protocol configs/third_pass_lgb_shrinkage.json SHA256 ea5ce787253b52eabb8353bf5584aafbc3f736e139f88ecdc332e8e9da3d7399.
