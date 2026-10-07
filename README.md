# Predicting Airline Satisfaction

A reproducible research project for Kaggle's [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10), covering tabular models, feature engineering, ensemble selection, and controlled experiments on why longer neural training can reduce validation performance.

The repository contains implementation, registered experiment configurations, research notes, and measured results. It also includes a self-contained handoff for independent agents working outside the original project environment.

## Start here

- **[Complete external-agent handoff](ALL_ATTEMPTS_AND_METHODS_AGENT_HANDOFF.md):** embedded configurations, 61 individual model result records, attempt histories, failure evidence, research, and Python source. Approximately 5.8 MB; download it for full-text inspection if GitHub does not render the entire document.
- **[Deep research report](DEEP_RESEARCH_REPORT.md):** schedule and refit confounding, optimizer behavior, blend sensitivity, uncertainty, and validation limitations.
- **[Six controlled continuation experiments](CONTINUATION_SIX_V1_REPORT.md):** completed comparisons of LR cooling, decay correction, EMA, label smoothing, frozen embeddings, and head-only continuation.
- **[Original results](REPORT.md), [second pass](SECOND_PASS_REPORT.md), and [third pass](THIRD_PASS_REPORT.md):** historical model portfolios and verified release construction.

## Dataset and evaluation

| Property | Recorded value |
|---|---:|
| Training rows | 699,635 |
| Test rows | 299,844 |
| Development rows | 629,671 |
| Original audit rows | 69,964 |
| Input features | 21 |
| Metric | Binary ROC AUC, higher is better |
| Development validation | Three shared stratified outer folds, with inner stopping selection |

The original audit and its sensitivity subset have already been evaluated. They cannot be treated as fresh validation or used for subsequent tuning. Early integration smokes exposed 568 audit rows; the historical sensitivity assessment excludes them and contains 69,396 rows. Development OOF comparisons are adaptively reused selection evidence, not independent confirmation.

## Models and methods

- LightGBM, CatBoost, and XGBoost baselines and feature variants.
- TabM and numerical/categorical RealMLP, including seed ensembles and alternative encodings.
- Route/distance structure, profiles, fold-aware target encoding, original-data teachers, and auxiliary expected-value/probability features.
- Registered blend searches, family ablations, rank-pair geometry, paired uncertainty estimates, and weight stability diagnostics.
- Local and cloud duration comparisons, followed by experiments separating schedule horizon from execution length.
- Exact epoch-4 prefix controls, passive monitoring, native export verification, and deterministic execution receipts.

## Recorded findings

| Release / comparison | Development OOF AUC |
|---|---:|
| Original portfolio | 0.96122103 |
| Second-pass portfolio | 0.96136372 |
| Accepted third-pass reference blend | 0.96140750 |
| Best fixed 10% mixture in the six continuation tests, head-only | 0.96140212 |

These are local development scores, **not public leaderboard scores**. The handoff and historical receipts distinguish submissions, release artifacts, and candidate results.

The reference blend is 64% frozen second-pass blend, 16% third-pass RealMLP with auxiliary probability features, and 20% third-pass XGBoost with auxiliary probability features. The six later continuation tests completed 72 new paired fits and 1,152 epochs; none passed the unchanged advancement gate, so no new release or submission resulted from that sequence.

Longer epoch ceilings originally changed normalized learning-rate, decay, and dropout schedules from the beginning. The outer refit then changed the horizon again. A 500-epoch inner search selecting epoch 4 is therefore not a saved epoch-500 model or an isolated duration comparison. Controlled later tests exposed these distinctions. Head-only continuation improved the epoch-4 standalone model, but did not improve the incumbent blend sufficiently to qualify.

## Repository layout

```text
scripts/       Training, features, inference, research diagnostics, tests, cloud controllers
configs/       Versioned campaign recipes and registered experiment policies
research/      Evidence ledgers, primary-source references, reviews, and method analysis
*.md           Campaign reports, operational histories, plans, and agent handoff
requirements.lock.txt  Historical local dependency snapshot
```

Raw datasets, row-level predictions, model binaries, generated cloud bundles, credentials, and local runtime state are excluded. Selected historical result and receipt contents are embedded in the handoff, so reviewers can inspect the evidence without those directories.

## Setup and reproduction limits

The original local environment used Python 3.12 on Windows. The dependency lock records that environment; cloud experiments used separately pinned runtime stacks. It is not a guarantee that the complete historical lock installs unchanged on every platform or reproduces GPU training bitwise.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
```

Obtain competition files through Kaggle after accepting the applicable rules, and place `train.csv`, `test.csv`, and `sample_submission.csv` in `data/`. Optional teacher/auxiliary variants require the additional source-data artifacts documented in the research notes. The scripts contain historical path, hash, quota, deadline, and authority checks; this repository is an evidence-backed research snapshot rather than a one-command fresh campaign launcher.

Saved full-model inference requires model and preprocessing artifacts that are not distributed here. Recomputing reported AUC requires the corresponding row-level OOF predictions and canonical splits. The handoff provides hashes and contracts, not a substitute for those bytes. Do not overwrite frozen splits or rerun once-only evaluations to reproduce a historical claim.

## Working with external agents

Give an agent the [complete handoff](ALL_ATTEMPTS_AND_METHODS_AGENT_HANDOFF.md) and ask it to independently audit the methods, identify confounders or bugs, and propose controlled next experiments. It can reconstruct embedded source files in a cloud sandbox for source review and synthetic tests without cloning this repository. Real-data reproduction still requires the omitted data and model artifacts.

Historical plans, deadlines, and authorization instructions describe past operations. The six-stage sequence is complete and its monitor is paused. A review request does not itself authorize new training, paid compute, submissions, audit rescoring, or public-score tuning.

## Attribution

Primary papers, public notebook sources, and inspected implementations are cited in [method evidence](research/method_evidence.md), [competition evidence](research/competition_evidence.md), and the research ledgers. Model libraries retain their own licenses. No blanket redistribution license is assigned here to third-party source material; consult the original sources and their terms.
