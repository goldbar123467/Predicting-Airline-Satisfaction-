# Six-campaign continuation: cloud lifecycle and release source audit

2026-10-04. Scope: source and saved metadata only. No API call, upload, dispatch, training, inference, prediction reading or new metric computation. Applies the Python ML quality, Kaggle competitive-validation and ML system design skills. Latest explicit human authorization reported by the root agent permits this new isolated sequence; old closed-campaign files remain immutable.

## Recommended boundary

Use one small new sequence controller for six ordered registered ideas, with explicit inner-screen, outer-confirmation and optional release-refit operations. Each remote operation has its own immutable version-1 kernel, prepared directory, source/metadata/registry/bundle hashes and durable intent. One account GPU reservation and one controller-owned active operation at a time. Keep scientific fitting, scoring and submission outside the lifecycle controller: it consumes their frozen hash-bound decision or verification receipts.

Preregister the inner rejection rule, maximum work and exact endpoints before first fitting. Run all registered inner screening folds before evaluating that rule, then either preserve a terminal rejection receipt or permit the registered outer fits. Do not pick a new best endpoint or change a threshold after reading curves. An inner rejection is an adaptive resource screen, not outer evidence that the candidate is harmful. A passing screen permits assessment, not a release. Full-data labels become available only after the candidate and release mixture have been frozen. All original audit restrictions remain.

Avoid a long cloud session waiting for local decisions: separate inner and outer jobs let the controller end GPU allocation after each stage. The cost is another startup and mandatory smoke. If a single bounded session does the deterministic inner gate internally, its gate and all expected terminal dispositions must be in the pre-dispatch registry. Do not treat provider COMPLETE alone as training success or inner rejection as a infrastructure failure.

## Reusable lifecycle contracts and required repairs

- `scripts/kaggle_fixed_epoch_dropout_retry1.py:119` verifies dispatch review and private metadata. Its `main` around line153 durably writes an exclusive push intent before mutation, requires ready private dataset version1, remaining free GPU quota >= provider timeout, no GPU reservation and paid scaling disabled. Preserve these contracts in new parameterized code. The root independently measures current quota; no quota was queried in this audit.
- The old retry launcher expects the requested slug in its push response. The provider actually canonicalized it from the title. New code must preserve requested ref and accept only the successful returned owner/ref, positive numeric kernel ID and immutable version1, then reconcile that actual identity against source, privacy, image, accelerator and data sources. Never repush to correct a slug. `scripts/status_fixed_epoch_dropout_retry1.py:47` demonstrates source verification from the API UTF-8 string, avoiding Windows CRLF translation in `kernels_pull`.
- A missing response after a recorded intent means outcome unknown. Stop further sequence dispatch until read-only reconciliation has either identified the unique save or established a separately reviewed disposition. A local process lock alone does not resolve an unknown remote outcome. Check all previous unresolved operations and account GPU reservation before each push. Do not reuse a slug/version for a failed attempt.
- `scripts/retrieve_fixed_epoch_dropout_retry1.py:269` implements the proven signed full-path GET transport, byte bounds using the same response Content-Length, no retained signed URLs and bounded pagination. Its archive validation at line109 enforces exact manifest inventory, hashes, sizes, no traversal/case collision/symlinks and bounded expansion. Reuse that behavior with explicit new operation ID and terminal-disposition contracts; do not carry its hardcoded dropout 12/9/6/3 counts into stage-specific jobs.
- Retrieve preserved terminal failures as well as successes. Assemble an assessment workspace only after registered successful completion, all source/input/partition/native checks and required exact prefix gates. Preserve the payload wrapper and registry bytes. Evaluation creates its own once-only claim; retrieval must never evaluate automatically or rewrite the returned manifest.

## Current incumbent and release boundary

Metadata inspected, without reading the prediction files:

| File | SHA256 |
| --- | --- |
| `artifacts/third_pass/blend/frozen.json` | `bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311` |
| `artifacts/third_pass/final/manifest.json` | `90247f5b5cf338248294f46a1df57cd56aef79d3df598e0ad3c822586044ad88` |
| `artifacts/third_pass/final/release_provenance.json` | `77cec962b2c37ccd43954bb9cf4fbe10673f5970a6e082ced2520c38ef1f5d3a` |
| `artifacts/third_pass/verification.json` | `51f47bbab2445e73a8cf40398ca4cc59239bffb580096f2d3bcc11ec5663264d` |
| Shipped submission identified by final release chain | `177eca4563e41b91c30c12d494284ea24d68951b6ada332edf49dc10646ccbc9` |

The frozen incumbent has 15 full-data members and recorded development AUC0.9614075009196988. The prior submission receipt links the shipped bytes to provider ref56775181. These are historical metadata, not recomputed evidence. The selection JSON contains a different `submission_hash` (`2b9ccbe6...`); use the final manifest/provenance/verification chain to bind shipped bytes. Do not silently replace that old field.

`third_pass_release.py:459–546` verifies saved native members and same-image cloud-production provenance. Its refit path at566 refuses to refit a selected cloud model locally and requires a fixed-round cloud refit after selection freeze. These contracts are appropriate, but its campaign globals, legacy run/result schema and old checkpoint-selection logic are not the new intervention API. Do not route a continuation endpoint through it unmodified.

For a qualifying fixed-alpha candidate, freeze its exact policy, epoch endpoint, seed policy, full training-row contract and mixture against the current incumbent. Cloud-refit only the new member, preserving all incumbent models and cached test predictions after validating their provenance. The release must bind the new model, train-only learned transform, test IDs/classes, source/environment, full-data exclusion change after freeze, and full raw native inference receipts. Independently reconstruct the final mixture from verified component predictions. A 90/10 mixture, if retained by the scientific policy, needs no retraining of the old 15 members. A new full-data refit need not equal OOF checkpoint predictions, but must use the registered policy and verify its own portable inference path.

If later stages establish another incumbent, freeze a new selection/release, never overwrite the original. Compare a prospective release against the current frozen best verified development selection and every relevant previous submitted selection, not just the initial stage-zero incumbent. Public leaderboard response is a separate observation and must not retroactively choose alpha or confer an OOF gain.

## Submission implementation must be new

`submit_verified_release.py:17–58` hardcodes expired October2/3 authorization and a third-pass-only campaign regex; it cannot execute the new authorization as written. Leave it untouched. Reuse the rigorous final artifact/native/source hash checks, finite/schema/ID verification, full-row raw inference proof, independent blend recomputation, no audit assessment and pooled/macro/per-fold incremental-gain gate. A new sequence-specific authorization receipt must bind permitted stage IDs, allowed release count, time/resource scope and unchanged quality requirements.

`submit_verified_release.py:90–122` already demonstrates digest-based durable intent and uncertain-response reconciliation. Improve the new path by obtaining the current team allowance from installed `KaggleApi.competition_get_submission_limits` (`kaggle_api_extended.py:2116`), which returns `num_today`, `num_total`, `num_allowed_now` and `limited_by_total`. Require a positive current allowance immediately before writing the attempt. Do not depend on hardcoded daily10 or infer UTC daily availability from a truncated history page.

For duplicate detection/reconciliation, page the underlying `ApiListSubmissionsRequest` response until exhausted or the exact known ID/hash-tag is found; `competition_submissions` at2039 returns only the submissions list and discards its response pagination token. A persisted uncertain attempt blocks another upload of the same bytes. After upload, save provider submission ID and response, then read processed status and score; upload success is not scored completion. A failed/no-gain candidate must never consume a submission merely because its cloud run completed.

## Replay and budget evidence

The frozen adapter explicitly saves `exact_training_resume=false` (`fixed_epoch_dropout_adapter_v1.py:596`) and disables Lightning checkpointing at886. Its graph is a portable inference export, not optimizer/scheduler/RNG/sampler state. Hashes in prefix receipts prove state comparisons but cannot reconstruct those states. Prior synthetic CPU resume evidence (`artifacts/research_pass_v1/epochs/resume_prototype/verification.json`) found ordinary Lightning restore fails with PyTabKit optimizer `KeyError __dict__`; a separate augmented toy restored exact state only after adding explicit inner Adam, RNG and progress handling. This is not a production resume capability.

Therefore replay epochs1–4 from identical initialization for each treatment. Reusing a previous immutable control is possible only if its full recipe, cloud environment, data/transform/order and source contracts are registered as reference and the replay passes the exact state gate before epoch5. Copy inference/reference receipts into the allowed payload, not a fictional training-resume checkpoint. Full control refits cost more but offer contemporaneous paired execution.

Measured successful deterministic T4 campaign: 12 H16/E16 fits totaled3718.321861 adapter seconds; wrapper4331.203183 seconds. Individual outer full16 fits took319.866972–336.195360 seconds at approximately419,781 rows, with1639 updates/epoch. A699,635-row full-data16-epoch refit has2732 updates/epoch with the same batch256/drop-last policy, approximately1.667 times as many updates. A linear compute-only extrapolation is roughly533–560 seconds; this is not a benchmark or guarantee. Reserve at least1200 seconds for a candidate full-refit job including setup, native export, all-test inference and delivery, revising only from an initial measured bounded smoke. Reserve release capacity before spending all free quota on the six screens. Exact E4 replay adds nonzero compute even if old control references are reused.

Six complete12-fit campaigns at the last wrapper duration would cost approximately7.22 GPU-session hours before releases, which may exceed remaining free quota. A preregistered inner screen and control reuse can reduce work, but neither guarantees all six complete under the current allowance. The root must use its live quota measurement to freeze stage allocations and release reserve. Wait for free quota reset or record bounded exhaustion; never silently switch to paid or local GPU.

## Required small tests before dispatch

Mocked transport/controller tests should reject changed plan/source hashes, concurrent or unknown-outcome intents, already-owned slugs, nonprivate/canonical identity drift, insufficient/reserved/paid quota, out-of-order phase unlocks, wrong completion counts and corrupt archives. They should preserve a successful canonicalized ref without a second push and permit read-only reconciliation after timeout. Release tests should reject stale incumbent bindings, insufficient gain, mutated artifacts, exhausted live submission allowance and duplicate/uncertain upload intents. Generated cloud CUDA smoke remains mandatory before real fitting; a CPU test cannot establish CUDA prefix equality.

Implementation API is pending root confirmation; no controller source or scientific policy was changed in this audit.
