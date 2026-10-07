# Adversarial integration review of the continuing research pass

Reviewer: independent blend/validation agent. Review began around 21:56 UTC on October 2, 2026. This is a bounded integration review within the longer pass that began at 21:01 UTC; it does not claim that the requested 3–4 hours have elapsed. The documents remain working documents, so the hashes below identify the specific revisions inspected.

## Scope and evidence boundaries

Read the integrated `DEEP_RESEARCH_REPORT.md`, `research/next_epoch_blend_experiment.md`, `scripts/research_manifest_v1.py`, the new ensemble-selection audit, and the generated-data mixed-input capture prototype. Reused earlier read-only source reviews and already saved numerical diagnostics. No development OOF, raw training, audit or test table was read or scored in this review. No estimator fit was started by this reviewer; generated-data integration execution belongs to the methods agent and is distinguished below. Edits by this reviewer are confined to this review file.

The review checked selection boundaries, the meaning of paired and simultaneous uncertainty, causal wording, experimental advancement rules, source preservation and preprocessing ownership. It does not establish the correctness of every library branch, CUDA reproducibility, competition quality of an untrained future recipe, or the elapsed duration of the complete investigation.

## Findings resolved during integration

1. **Epoch-selection wording now matches the available record.** The earlier phrase about “later recorded checkpoints” could imply a retained checkpoint sequence. It was replaced by the actual finding: the completed 500-epoch inner searches selected epoch 4 under a float32 ensemble-AUC rule, while curves and checkpoint predictions were not retained. The new source audit shows that probabilities of eight members are averaged before AUC, all eight members restore from one common epoch, and the three reported epoch entries index outer folds. It is not mean member AUC or eight separately selected checkpoints. The synthetic four-row counterexample illustrates that difference without estimating its historical effect. Float32 ties and last-best selection are documented as numerical semantics, not an explanation for the AUC regression.
2. **Preservation now has an anchored reference.** The first manifest revision compared live production sources with mutable archived copies. The revision pins `release_provenance.json` by SHA256 and compares both source copies with its recorded source hash. Eight production files are covered. The manifest's prose now correctly says its three executed-source bindings concern two root scripts; the rest of the inventory does not automatically verify every recorded import or executed-source binding.
3. **Epoch experiment causal wording now remains specific.** B minus A changes the normalized schedule horizon while fixing the four-epoch endpoint. C minus B continues one H16 trajectory for twelve more epochs, including the later schedule. It does not isolate an abstract “number of updates” effect with every effective hyperparameter held constant. The detailed protocol and revised integrated summary now share this qualification.
4. **The source ledger's route-one wording was corrected.** Its initial table referred to retaining a current H4/E4 control and using inner selection. The historical four-cap control actually refitted three epochs, whereas proposed A is a new literal four-epoch model. The proposed inner stage is passive monitoring. The revised route-one and route-two rows now match the fixed-endpoint protocol and fixed 10% mixtures.

## Claims that are supported within stated limits

The blend recommendation is appropriately conservative. The retained 64/16/20 mixture is not declared uniquely optimal. Family omissions and fixed neural/tree proportions quantify conditional contribution within the existing mixture, not causal importance or a retrained-family contest. The 49-point best observed gain remains below the existing operational gate; prior duration failures and the closed saved-pool search remain unchanged.

The paired uncertainty report uses the correct comparison-specific SE. The best 65/20/15 grid point has delta +1.582992250e-6 and SE 3.499915287e-6. Its pointwise interval and approximate simultaneous fixed-grid band span zero. No claim of equivalence is justified: the simultaneous upper endpoint is about 1.03146e-5. The report correctly calls out independent-row assumptions, overlapping training dependence, historical adaptation and training variance exclusions. Perturbation winner frequencies refer to the 49 fixed grid points, exclude the incumbent and are not posterior probabilities. The positive-negative pair count is not treated as an independent sample size.

The nested-validation warning is correct. Re-splitting an existing OOF matrix excludes held-out labels from the meta objective but can retain those labels in the base models generating meta-training features. Properly regenerated inner OOF must remain entirely within the proposed outer-training boundary. The generated-data perturbation proof demonstrates that path and its removal; it cannot establish the magnitude or sign of bias on the competition. Even subsequent correct nesting remains conditional on choices developed using these historical rows.

The optimizer discrepancy is verified implementation evidence, with no demonstrated causal attribution for the duration loss. Changing factor application while preserving nominal rates changes the effective regularization dose. Keeping that intervention separate from schedule control is appropriate. No benefit from EMA, SWA, a new family, calibration or an AUC-specific training loss is established by the current pass.

## Proposed experiment gates and unresolved deployment work

The 12-fit/120-epoch proposal is internally consistent: A4 and C16 in both inner-monitor and outer-assessment stages for each of three folds, with B captured from C rather than separately fitted. Approximately 114 full-outer-training epoch equivalents follows the inner 90% row fraction, before overhead. Model seed and inner-partition seed are now distinguished. Epoch count is deliberately the schedule clock; the inner/outer update-count difference is disclosed.

One fixed 10% mixture for each arm and the explicit C-versus-incumbent/A/B advancement rules bound the new ensemble question. They are engineering gates for a later confirmation study, not significance tests or submission permission. The report must retain all failed comparisons and must not switch its success definition if A or B happens to look favorable. Outer labels remain unavailable for per-epoch stopping or deciding new arms.

Passive monitoring must disable both early termination and final best-checkpoint restoration. Native B export must own its model, preprocessing and schema copies and leave live C's optimizer, RNG, scheduling and forward operations unchanged. Existing CPU prototypes supply partial mechanism evidence, not production-wrapper or CUDA acceptance. The complete future adapter still needs integration with actual feature preparation, training, endpoint artifacts and native inference verification before a real-data experiment. A throughput estimate from small generated data does not establish full-data wall time.

## Manifest source review and executed read-only checks

The revised manifest uses exclusive timestamped output creation, preserves previous snapshots and deliberately excludes earlier manifests from its inventory. Its `final` label does not claim elapsed duration. The recorded working execution at `artifacts/research_pass_v1/manifests/20261002T215901879107Z_working.json` reports 84 files and 17 passing checks: six frozen inputs, eight live/archive production-source checks, and three source bindings for two root scripts.

This reviewer independently executed twelve read-only checks: the pinned release-provenance hash, all eight source/archive/hash triples, and the three executed-source snapshot bindings. All passed. The command loaded JSON and source bytes only, not split or prediction tables. It did not rerun the manifest, models, metrics or tests. This confirms those byte bindings at review time, not mathematical correctness or a guarantee against later changes.

| Reviewed item | SHA256 at inspection |
|---|---|
| `DEEP_RESEARCH_REPORT.md` working revision | `7b6a02e0bdbc9c6fbf80a12c6b0159fe25492f7b4542f826bc7f59ebd60d9fe0` |
| `research/next_epoch_blend_experiment.md` | `7af17e699043fdd98b70bcb903906190ccbb7a46b5ae52b9b32f79e81c400f91` |
| `scripts/research_manifest_v1.py` | `acfa3a7ca420898104cf68afb7ed39c4a96ed74cb00f98cfb74c525d70ab8102` |
| Working manifest `20261002T215901879107Z_working.json` | `33f7eaa9908660d9bab3de393feda2e25f0e6349d15c4c1c11b5ed6111ac0be4` |
| `research/deep_pass_selection_semantics.md` | `4382cd1e6d9fd0de70567993fed2d6fcd73b6fd2766105986408200b792e2369` |
| `epochs/selection_semantics.json` | `474008b01d68f5f094c2ac1b17967de5ea63264d24c30ec1d0dc7a1c0c3d86e6` |

## Mixed-input preprocessing and capture proof: completed independent review

The registry was saved at 21:54:14 UTC before execution. The generated-data contract is 384 rows, 288 fit and 96 monitor, six continuous plus two categorical inputs, eight small vectorized members, H16 with capture at epoch 4, one CPU numerical thread, and a 300-second bound per attempt. Native reference predictions use the installed prediction loader and prediction step, which is stronger than comparing only two forms of the custom export graph. Unknown categories, multiple batch layouts, owned tensor storage and continuing optimizer/RNG paths are explicitly tested.

The first attempt failed during compile/inventory before training because the inventory assumed learned median/scale tensors would be in `creator.static_model`. The active PBLD parallel branch keeps these tensors inside the vectorized network. The failed attempt and its source were retained; the repair inspects both owners and asserts the fitted nontrainable tensors remain unchanged during optimization. **The preliminary suspicion of repeated fitted static preprocessing is retracted for this realized PBLD recipe.** A generic source-loop suspicion was insufficient to establish it.

The second attempt passed at 22:00:11 UTC, with 3.421 seconds inside the fit/check function. The methods agent executed the one integration test; this reviewer did not rerun it. Six fitted nontrainable `/tfms` tensors were found in the vectorized network, all registered and with first dimension eight: three shapes `[8,1,6]`, two `[8,1,4]`, and one `[8,1,9]`. Nonidentity median and scale values are present. `creator.static_model` itself remains empty; an empty static state dictionary therefore does not imply absence of learned preprocessing.

The feature-only monitor perturbation leaves fitted schema and every inspected network/static tensor identical under the same seed. The installed source confirms the learned fitters receive `train_ds`; fixed category vocabularies are learned from the first 288 fit rows. This is evidence about this realized generated-data path, with one deliberate counterfactual. It is not proof about every possible factory branch or a new audit of `common.py`'s external feature preparation.

Native reference predictions use installed `get_predict_dataloader` and `predict_step`, followed by its final softmax. Cloned native, portable and saved/reloaded predictions match within **1.1920928955e-7**, below the registered dtype tolerance. They are not claimed bitwise equal to that reference. Unknown sentinel and unseen categories agree exactly; reversed layouts agree exactly and split batches differ by at most the same 1.1921e-7. Both one-hot and embedding export branches are exercised.

Capture versus no-capture runs match exactly on all sixteen saved epoch states, the 144 batch/RNG/schedule/progress traces, actual monitor predictions/metrics, final model, underlying Adam state and final probabilities. Thirteen live-state categories remain unchanged around every observation. The epoch-four export hash and predictions remain unchanged after continuation; final parameters differ from the captured prefix. The real monitor's last best epoch is fifteen, but literal epoch sixteen is retained, confirming that the fixed endpoint is not restored to the best monitored epoch in this run.

The output key `learned_static_preprocessing_unchanged` concerns the empty static owner. The stronger evidence for the six learned tensors inside the network is `run_arm`'s pre/post `preprocessing_inventory` equality plus complete network-state equality. No result field was rewritten after execution; this distinction is recorded to avoid overreading that key.

This reviewer independently checked twelve byte bindings: eight recorded source files, unchanged registry, saved epoch-four graph, schema JSON and archived failed-attempt source. All passed. Also inspected the revised inventory code, native prediction path, preprocessing source split, clone ownership check, RNG/mode preservation, literal endpoint assertions and saved numerical fields. No blocking issue remains for the stated generated-data mechanism claim.

| Mixed-capture evidence | SHA256 |
|---|---|
| `epochs/mixed_capture/verification.json` | `d1d2ae2795dd95f3154371a2c32299b2e2bc90e29132b231ff01c29bf482e0a6` |
| Unchanged `epochs/mixed_capture/registry.json` | `6e7c5632c5842377ca6d33074bed6822ee22cbcbb0e8889f5fa909db950f9aaa` |
| Executed `scripts/research_mixed_capture_v1.py` | `261f7625c41a4451bd107d6dee299bd9dad9617fa559ecafff5ddac5e8f3daa5` |
| `scripts/test_research_mixed_capture_v1.py` | `cdd6463b1a7a8a991b7306c905b33088e6672ea08eff36c8d91c40e670f02bfb` |
| Preserved failed `attempt_1_source.py` | `8d27d711ae7d5087354a13edc1a79ed84613265abfecaab05b4dd27dc5e0c72d` |
| `capture_attempt_2/epoch4_graph.pt` | `48a57963b8064c0057376e4b010b47e0031c0488a7c9cc7adf8491c22cc9de82` |

The acceptance remains limited to the pinned CPU environment, generated mixed input and small eight-member architecture. It does not validate the production fitter, actual competition feature bank, CUDA, arbitrary object-graph ownership, exact resume or a quality gain. Both arms include the same monitor and common read-only audit; added fixed probes and clone/export are the tested intervention. The result closes the earlier one-member/empty-preprocessing proof gap without claiming a preprocessing correction or a complete deployment.

## Final reviewed revisions and disposition

Re-read the integrated result/limits and proposed experiment after these changes, and checked the corrected source-ledger route rows. As of **22:05:45 UTC**, no blocking methodological or source-preservation issue remains in this bounded review. This timestamp is approximately 65 minutes after the full pass began, not three or four hours; a final closeout must state actual elapsed time if it ends now. Any later document edits need their own inventory snapshot and are not silently covered by an earlier hash.

| Latest reviewed revision | SHA256 |
|---|---|
| `DEEP_RESEARCH_REPORT.md` | `55c24677df1f57e7fa0f8a7ce41fdc19246f189dd7df639dd590d88bdbe5998e` |
| `research/next_epoch_blend_experiment.md` | `65e45f9f2f1ff7f0e73641cbe071901b0c14cf6e69d2e2e00564670dcec151e6` |
| `research/deep_pass_sources.md` corrected route rows | `c8f82691deef237f089efa740f42cea33a534db9a646e4f6d2effec6c531e26d` |
| `research/deep_pass_mixed_capture.md` | `2f0e1b95aaa8e64acc75b71337eb59c517b95abfbeec5b5a154e5aa2ac0f9ce6` |

The recommendation remains to preserve the current blend and implement the bounded fixed-endpoint experiment under a new training protocol if training is subsequently requested. Unresolved scientific uncertainty requires new instrumented fitting and evaluation within correct selection boundaries. It cannot be removed by another fine search over the same saved OOF predictions. This review does not authorize that training, modify old release gates, or declare the requested 3–4-hour duration satisfied.
