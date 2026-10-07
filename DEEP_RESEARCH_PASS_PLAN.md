# Epoch and blend research pass, October 2, 2026

User request: a rigorous 3–4 hour pass with agents to explain why longer epochs are not improving scores and determine how to choose blend amounts.

Start: 2026-10-02 21:01 UTC (17:01 America/Indianapolis). Target completion: 2026-10-03 00:31 UTC (20:31 October 2 local); hard research cutoff 01:01 UTC (21:01 local). This is a research/diagnostic budget, not authorization for another long training campaign. Existing training and cloud campaigns are resolved and their monitor stays paused.

## Scope and immutable boundary

- Retain submitted release 56775181 and all v1/v2/v3 configurations, split identities, native artifacts, and frozen selections. Public score is reporting evidence only.
- No new cloud jobs, submissions, production refits, real-data model training, or changes to training/release source in this pass. Synthetic regression tests, CPU analysis of existing development OOF, literature/code review, and new isolated research files are allowed.
- Never read or score audit predictions/labels. Load `data/splits.parquet`, filter `fold >= 0`, and then load only verified keyed OOF files. Original audit exposure is permanent; it cannot become a fresh holdout.
- The completed three-pool/twelve-mixture saved-checkpoint screen is not repeated. Failed duration candidates keep their failed gates. No new diagnostic will promote a release.
- Record input and script hashes, exact commands, seeds, versions, and limitations. Use `.venv/Scripts/python.exe`; CPU analysis uses at most two numerical threads per process and avoids concurrent large arrays.

## Agent deliverables

1. `epoch_diagnosis`: inspect actual schedule, epoch selection, outer refit, and telemetry; quantify what the 500-epoch experiment measured; propose controlled follow-ups. Own `research/deep_pass_epoch_diagnosis.md`, `scripts/analysis_epoch_diagnosis_v1.py`, and `artifacts/research_pass_v1/epochs/`.
2. `blend_validation`: independent review of OOF provenance, weighting, selection bias, uncertainty, and requirements for honest evaluation. Own `research/deep_pass_blend_validation.md` and `artifacts/research_pass_v1/validation/`.
3. `methods_research`: primary-source methods/repository/competition research with provenance and inspected evidence. Own `research/deep_pass_sources.md` and `artifacts/research_pass_v1/sources/`.
4. Root: register and execute bounded diagnostics of the **already accepted 15-member incumbent**, integrate findings, prepare a specific experiment protocol, and cross-review implementation and conclusions.

## Bounded diagnostic protocol, recorded before new blend scores

The anchor is `artifacts/third_pass/blend/frozen.json`, already reported pooled OOF AUC 0.9614075009196988. Validate frozen split/result/OOF hashes, exact development IDs, folds, targets, finite probability domain and positive-class convention; independently reproduce anchor scores first.

1. Inventory the 15 accepted models' exact weights, families, selected rounds, training cost, and feature contracts. No candidate mining across failed models.
2. Fixed leave-one-unit-out ablation without reoptimization. Preserve registered equal-weight seed pairs as inseparable units. Also fixed leave-one-family-out ablation. Report pooled and within-fold AUC changes, log loss/Brier as diagnostics, and error correlation.
3. A single coarse neural-versus-tree curve. Preserve internal weights within each group. Neural weight grid `0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1`, plus the incumbent weight. This measures sensitivity and identifies broad stable ranges, not a new deployable winner.
4. A single coarse two-dimensional curve of the frozen v2 anchor plus the two already accepted v3 additions: each new-model weight in `0, .05, .10, .15, .20, .25, .30`, leaving the residual weight on v2. Include the exact incumbent `[.64,.16,.20]` separately. Fixed 49 grid points, no fine-grid follow-up. Preserve all v2 internal weights.
5. Conditional weight stability: select among the already fixed 49 grid points using two saved outer folds and evaluate on the third. This is explicitly **not genuine nested validation** because base training and prior recipe selection already depend on these data. Do not call resulting scores unbiased. Report three folds separately; do not treat them as independent runs.
6. Paired AUC influence/stratified row bootstrap diagnostics for a small fixed set: incumbent versus v2, incumbent versus each leave-one-family-out result, and incumbent versus simple three-block equal weighting. Exact correlated AUC influence functions must be checked against brute-force pairs on synthetic data (ties included) and sklearn AUC. These intervals condition on already fitted/selected scores, assume independent rows, omit training/selection uncertainty, and cannot establish a new gain.
7. Before claiming useful precision, estimate paired uncertainty and compare the size of the current 1e-5 operational gate and adjacent weight-grid differences. Pair count is not sample size.

Additional diagnostics require a dated rationale before scores, remain bounded, and cannot silently change these into release criteria. Stop expanding when remaining uncertainty requires new nested fits or missing telemetry rather than more analysis of the same labels.

## Stages and review

- First stage: provenance, training-source audit, literature refresh, frozen diagnostics implementation and synthetic tests.
- Second stage: execute bounded OOF analyses, inspect plots and uncertainty, independent code/statistical review.
- Third stage: challenge interpretations, distinguish confounds from established failures, design the minimum informative next training experiment, and verify source-backed conclusions.
- Final deliverable: `DEEP_RESEARCH_REPORT.md`, reproducible diagnostic JSON/CSV/plots and scripts, ranked next steps with data/tensor contracts, baseline, compute caps, leakage boundaries and acceptance criteria. Record actual elapsed research time honestly.

The research heartbeat may continue this exact pass after interruption. It must stop itself at completion/cutoff and must not reactivate the completed training campaign. Report only meaningful findings, failures or completion; no unchanged status messages.
