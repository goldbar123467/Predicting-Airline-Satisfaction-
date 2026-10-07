# Private Kaggle RealMLP compatibility and timing control

## Root execution addendum,2026-10-02 12:40UTC

The prepared GPU job was subsequently launched as private version1, kernel136781682, and completed successfully. Downloaded report: cloud/realmlp_probe/output/realmlp_probe/realmlp_probe_report.json. PyTabKit1.7.3 and pytorch-lightning2.6.6 were installed; torchmetrics1.9.0 already matched. All protected numeric and Torch/CUDA versions were unchanged. Eight-member synthetic fit, native export, GPU reload and CPU inference passed. Maximum reload/CPU-CUDA error2.98023224e-7, native-export error1.78813934e-7. Unknown-category, validation-exclusion and batches1/17/257 checks passed. Total runtime55.683seconds including installation. Private metadata was read back and verified. Kaggle GPU quota remaining29.97257hours at12:41:59UTC. The full-fold timing control below remains unexecuted.

The separate private CPU portability job also passed all1024 raw test rows in batches1/17/1024 against frozen local predictions, maximum error1.78813934e-7. These checks establish compatibility, not generalization or a leaderboard improvement. Use the pinned bootstrap for each new ephemeral Kaggle training session.

Prepared 2026-10-02, approximately 12:35 UTC. Scope: source/package inspection, script preparation and static checks only. This agent installed no packages, uploaded nothing and executed no training locally or remotely. The root agent controls launch and output retrieval. Read the Python/ML, Kaggle and research-before-build skills and the existing cloud-expansion/local campaign plans.

## Prepared compatibility job

`cloud/realmlp_probe/probe.py` is self-contained. The preparer is `scripts/kaggle_realmlp_probe_prepare.py`; the directory includes private GPU metadata, requirements and a preparation manifest. Ref: `clarkkitchen/s6e10-realmlp-compatibility-20261002`. Root will align the display title with the slug before launch; the generator now derives the title from its slug. **Recommended timeout: 600 seconds.** Internet is required for pinned package installation; no competition or private dataset is attached because all training data are synthetic. One T4 is used; the second GPU is not treated as pooled memory or automatically used.

The initial environment probe actually executed on two Tesla T4 devices, each 15,360 MiB, with four CPUs, 31.35 GiB host RAM, Python 3.12.13 and torch 2.10.0+cu128. PyTabKit was missing. Those are observed environment facts; RealMLP training compatibility and throughput are not yet established by this preparation.

Dependencies requested on every fresh kernel execution:

```text
pytabkit==1.7.3
pytorch-lightning==2.6.6
torchmetrics==1.9.0
```

The [official PyTabKit 1.7.3 package metadata](https://pypi.org/pypi/pytabkit/1.7.3/json) and [tagged pyproject.toml](https://github.com/dholzmueller/pytabkit/blob/v1.7.3/pyproject.toml) were inspected. RealMLP needs the base dependencies; the `[models]` extras, TabM package and benchmark/HPO stacks are unnecessary. PyTabKit is Apache-2.0 and supports Python >=3.9. The inspected universal PyTabKit wheel SHA256 is `1589f281e99d4a6965a83f954d9f78b038fc633a4246d450a1a3599952d4d841`. The locally installed Lightning/TorchMetrics metadata was also inspected: their core Torch/NumPy minimum requirements admit the probed cloud versions. This is dependency inspection, not proof of successful imports or training on Kaggle.

The installer first runs pip's resolver with `--dry-run --report`, pinning existing torch/torchvision/torchaudio, NumPy/pandas/sklearn/scipy, NVIDIA/CUDA and Triton distributions to their actual installed versions. It rejects package changes outside the small base-dependency allowlist and non-wheel/non-PyPI URLs. It installs only the inspected resolver plan with `--no-deps`, using wheel SHA256 fragments; the PyTabKit wheel must match the pre-inspected hash. It then confirms protected versions did not change and checks active base requirements. The operation fails rather than replacing the cloud CUDA stack or adding arbitrary extras. Dependency failure remains a failure report; no uninstall/upgrade workaround is hidden in the probe.

The embedded, unchanged local implementation hashes are:

| Source | SHA256 |
|---|---|
| realmlp.py | 8da0d62f1555c91ffe91befe7855f553f58fe6781db0e7249a7e7041130e03d8 |
| realmlp_categorical.py | 70c8b78f77c8f53df7f94e7fc4c92d3e47592973dd942fae20ed3beb2bbe4184 |
| categorical_transform.py | dc6e354e22f86eca08315ded64938c9fc234d3bf005149925feb3aac6fdbc2b2 |

Prepared probe SHA256: `f9a3632f1186b0ec95dbef3203a2e664781f6826fca1f99718d493e2343cb936`. Changing metadata title does not change that code hash. Never alter this source after pushing its immutable version; prepare a separate job if a runtime repair is necessary.

The actual runtime checks will:

1. Verify categorical preprocessing save/reload, train-only numeric statistics and unknown strings, then validate compact input shapes and category indices.
2. Fit 600 synthetic rows with 300 inner-validation rows, eight RealMLP members, widths 512/256/128, batch 256, two-epoch maximum and label smoothing zero. Include binary, multi-category, high-cardinality, all-unknown and single-known categorical columns.
3. Assert the validation-only category never enters the training vocabulary; refit all 600 training rows at the selected fixed length.
4. Save/reload native TorchScript artifacts on CUDA and CPU; compare batches 1, 17 and 257, an empty batch, and novel versus explicit-unknown categories. The existing exporter independently compares its native estimator and traced graph. Probability averaging and class order remain unchanged.
5. Record actual versions, synchronized fit/export times, allocator peaks, RSS, selected/fixed epochs, errors and tracebacks. Files are under `/kaggle/working/realmlp_probe`, including `realmlp_probe_report.json`, dependency logs/plan and native synthetic models.

Static checks passed for generated-source syntax, all embedded hashes, package pins, real constraint-file newlines, and private GPU/no-data metadata. **No local smoke or cloud RealMLP success is claimed yet.** Root should require report status `passed`, retrieve the exact version's outputs, verify the read-back private flag, and retain failed-install logs before moving to real data.

## One fixed-recipe fold timing control after compatibility passes

Use **existing completed `v2_realmlp_cat_raw`, outer fold 0**, without adding a new candidate or seed. This avoids dependence on large source-feature banks while exercising the native categorical architecture. The local model's saved metadata and result were inspected:

- Seed 20261005; native categorical twins; teacher false; no TE, route profiles or auxiliary features.
- Eight members, hidden sizes [512,256,128], batch 256, eval batch 2048, learning rate .053, weight decay .015, threads 4, label smoothing zero.
- The existing recipe allowed four inner-selection epochs and selected **three** for this outer fold. For the engineering timing control, freeze **three epochs**, with `X_valid=None, y_valid=None`; do not run a new stopping search or use outer validation to alter length.
- Outer training rows 419,780; held-out development rows 209,891. Frozen split SHA256 `4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c`.

Use a fresh private cloud-only identifier, a single T4 and **1,800-second hard cap**. The data bundle must contain only development IDs/rows or enforce `fold >= 0` before joining labels; exclude all original audit labels. Require keyed identity/order hashes, raw feature dtypes, frozen run JSON, exact source hashes and only this fold's required input artifacts. Mount official competition files for raw feature provenance when needed; do not include credentials or unrestricted local directories. The parallel CPU-portability preparer owns dataset preparation, so this note does not assume an unverified bundle schema.

The cleanest numerical control is to export the already-fitted outer-fold `CategoricalTransform` plus its compact training matrix or reproduce that exact matrix from frozen raw rows, then verify matrix/schema/vocabulary hashes before fitting. Hashing only raw CSVs is insufficient to show preprocessing equivalence across NumPy/pandas versions. Use the saved fold transform for held-out inference, not a vocabulary refit on train plus held-out rows. Raw categorical feature construction must happen before converting all original numerics to their categorical twins; source caches are disabled for this recipe.

Record separately: input verification/read time, feature/transform time, synchronized three-epoch fit plus export time, inference on the predetermined held-out rows, CPU reload time, native serialization size, host RSS and peak reserved/allocated VRAM. The wrapper currently reports fit plus export together; do not label that measurement pure GPU training time. Preserve predictions by held-out ID, but **do not enter them into the local candidate ledger/blend, score the audit, select a seed, or choose an improved configuration from this control**. A difference from local predictions may reflect torch/NumPy/sklearn/backend differences despite identical seed and recipe. No bitwise cross-platform guarantee is made.

The 271.86-second local figure is for all three folds including inner selection, outer refitting, inference and serialization. It is not a directly comparable one-fold timing baseline. Compare only measured matching stages. The tiny synthetic probe establishes execution and serialization correctness, not full-fold speed or model quality. Reserve today's local freeze/refit/verification budget; cloud expansion does not modify the 17 primary recipes, single confirmation allowance or exposed-audit policy.
