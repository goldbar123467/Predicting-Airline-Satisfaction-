# Continued local and private cloud campaign

Authorized October 2, 2026, approximately09:21EDT: continuously train and submit locally and via free Kaggle cloud in parallel until13:00 America/Indianapolis (17:00UTC). This supersedes prior no-new-submission/closed-queue scope for NEW runs. Preserve all v1/v2 releases, completed run objects, original data, exposed-audit exclusion records and frozen splits.

## Execution and evaluation

Use Kaggle and Python ML quality skills. Root coordinates one local GPU training/release process at a time and at most one free private cloud GPU session plus one CPU session. Initial renewed cloud allowance is2GPU hours, less than the29.83hours last confirmed available. No paid compute or public notebook/dataset posting. Check actual status/quota before each dispatch. Cloud jobs have explicit <=3600second timeouts, all exploration ends16:00UTC. Start no new experiment after15:45UTC. Reserve the last hour for full-data refit, verification and final scored submission. All training ends by17:00UTC, except processing a submitted CSV may finish afterward.

Every local batch has immutable configs/<campaign_id>.json, state/<campaign_id>, logs/<campaign_id>, artifacts/<campaign_id>, report<CAMPAIGN_ID>_REPORT.md. Initial identity third_pass; subsequent identities third_pass_batch02, etc. The campaign field stays third_pass. Each finished bounded queue freezes automatically with wait_for_cutoff=false, then refits and verifies. Only a verified release with a stable incremental development gain is submitted once per unique CSV hash. The next batch starts after that GPU tree exits. Repeated submissions are user-authorized, subject to actual competition daily allowance; public scores are recorded, never used to select weights or experiments.

Frozen split remains three development folds,629671rows. Original exposed audit and sensitivity labels NEVER select, tune, early-stop, calibrate or assess these models. Fit learned preprocessing in inner and outer training partitions. Development selection is adaptive reuse, not an unbiased performance estimate. Only frozen production refits can use all699635labels with fixed median development lengths. Real-data smoke filters frozen fold>=0 before sampling/fitting.

## Initial preregistered local batch

1. v3_realmlp_cat_raw_aux_probability: exact v2_realmlp_cat_raw_aux recipe plus13 observed-rating conditional probabilities and sum log(max(q,1e-6)).
2. v3_xgb_route_te_teacher_profiles_aux_probability: exact v2_xgb_route_te_teacher_profiles_aux recipe plus the same block.

The13 native original-only rating models are reused unchanged. No synthetic training/test feature fitting, satisfaction labels, ID inputs or new source-bank training. Use each bank's explicit class array for observed-value lookup; missing/unseen ratings receive1e-6. Supported probabilities remain original native values. Sum logs is a composite conditional score, not a normalized joint likelihood. Compute in float64 then store float32; keyed full cache and native batch parity required before fit. The corresponding bank does not see its own predicted rating as an input.

## Fixed selection and release gates

Each batch selects against the unchanged verified v2 blend (OOF0.9613637193250886) as one fixed anchor. Coarse additions alpha in[.05,.10,.20,.30], at most2 additions, require >=.00001 pooled AND mean-fold AUC gain per addition and vs anchor, worst-fold regression <=.00002. No arbitrary seed or unrestricted hyperparameter sweep. Later new batches may consider prior eligible v3 candidates but preserve prior freeze files. To submit another batch after v3, it must additionally improve over the best previously submitted development blend by >=.00001 pooled and mean-fold, without any fold regressing >.00002. Test predictions never select weights.

Only register later hypotheses in a dated handoff with evidence, exact controls and a bounded candidate count before fitting. At most one fixed equal-average seed confirmation globally, for a genuinely new representation if fixed-anchor gain >=.00003 pooled and mean-fold, all fold gains nonnegative. Seed20261022 is fixed; do not choose the better seed. Stronger training-duration follow-up must compare an unchanged-recipe control on the same cloud platform. See research/third_pass_validation.md for conditional follow-ups and source evidence.

Use scripts/third_pass_release.py. Freeze, refit selected new members, reuse checksum-verified v1/v2 native members, exact sample IDs/schema/order/probability checks, independent all-row blend recomputation, native reload and raw inference on299844rows, hashes and report. If no stable improvement or verification fails, retain latest verified submission and diagnose. A failed new release must not overwrite previous deliverables or trigger upload.

## Monitor and loop

Existing airline-satisfaction-overnight-monitor is reused every15minutes. Read THIRD_PASS_HANDOFF.md for current local config and cloud receipts. Inspect owned PID, creation time, command tree, log progress, artifact hashes, RAM/GPU, cloud job state/quota and time. Stop only demonstrably failed/stalled project-owned work, preserve completed fold caches, repair narrowly and resume. Never launch overlapping local GPU jobs. Continue bounded evidence-based batches while time remains, and submit verified improvements. Pause only after the deadline campaign's final deliverables and submission status are checked and all owned cloud sessions have completed or been cancelled. Stay quiet for unchanged state; report measured gains, repairs, failures and completion.
