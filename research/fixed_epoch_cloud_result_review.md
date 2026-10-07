# Independent interpretation of the fixed cloud duration experiment

Reviewed October3,2026 after root completed the frozen local assessment once. This review reads saved JSON/JSONL metrics, curves and receipts only. It performs no prediction scoring, weight search, model inference, fitting or provider API calls. Source and transport review are recorded separately in `research/fixed_epoch_v1_preflight_review.md`.

## Decision supported by the registered experiment

Reject C for advancement. Continuing the registered H16 trajectory from epoch4 to epoch16 substantially reduced development discrimination, and its fixed10% addition made every fold of the incumbent worse. Preserve the incumbent's existing64/16/20 composition. This is a decision about the tested candidate/procedure; it does not identify an optimal coefficient over every possible blend or prove that every positive C weight is harmful.

| Saved endpoint | Pooled development AUC | Mean fold AUC | Log loss |
|---|---:|---:|---:|
| Incumbent |0.961407500920|0.961411871318|0.221727290638|
| A: H4/E4 |0.960689515057|0.960696756911|0.223701854128|
| B: H16/E4 |0.960544144769|0.960667852803|0.223774310200|
| C: H16/E16 |0.954466852066|0.954470138697|0.254469753337|

C minus B is −0.006077292703 pooled and −0.006197714106 in mean fold AUC. Fold differences are −0.006215870150, −0.005904739383 and −0.006472532783. Both log loss and Brier score also deteriorate. This is consistent across these three fixed folds, but the experiment supplies no independent significance test or new untouched holdout.

The fixed mixtures have pooled AUC0.961405362015 for A,0.961393720886 for B and0.961305778007 for C. C's mixture loses0.000101722912 pooled and0.000100876544 macro AUC against the incumbent. Its fold losses are0.000080330704,0.000079397801 and0.000142901126, each larger than the permitted0.000020 regression. Against both A's and B's fixed mixtures, C also fails the pooled, macro and fold gates. The saved `advance_to_confirmation=false` therefore follows the original rule. No alternate arm is retroactively promoted. A's mixture has slightly better log loss than the incumbent but slightly worse AUC; auxiliary metrics cannot replace the registered AUC decision.

B minus A is smaller and heterogeneous: pooled−0.000145370288, mean fold−0.000028904108, with fold differences−0.000020345637,−0.000169845596 and+0.000103478909. Do not describe a universal per-fold penalty from the longer declared horizon. The difference between pooled and macro effects also should not be attributed to a particular calibration mechanism without another analysis.

## The saved curves support overfitting of this trajectory

For C, the fixed evaluation-mode probe consists of fitting rows. It is distinct from the held-out inner monitor and from dropout-affected minibatch objective values. Their epoch4-to-16 divergence is the relevant evidence:

| Inner fold | Fitting-probe log loss,4→16 | Monitor log loss,4→16 | Monitor AUC,4→16 |
|---|---|---|---|
|0|0.206053→0.087994|0.222349→0.261398|0.961186→0.954153|
|1|0.221895→0.092693|0.220352→0.258663|0.961853→0.955143|
|2|0.212311→0.096694|0.225780→0.262065|0.959950→0.954016|

Fitting-probe AUC rises to approximately0.995–0.997 while held-out monitor AUC falls. The training-mode objective also falls, but its different stochastic mode/reduction prevents treating it as a numerically comparable held-out BCE. The evaluation-mode evidence already establishes the divergence without that comparison. The pattern strongly supports overfitting under this recipe rather than merely a mismatch between the optimized BCE and evaluated AUC: held-out BCE, AUC and Brier all worsen while fitting-row predictions improve.

Degradation is visible during the extended high-LR phase, before the final LR descent. All three C monitors are below their epoch4 AUC by epoch8 while base LR is still0.053. The late annealing window does not recover performance and accompanies further fitting/monitor divergence. This refutes a claim that this particular16-epoch run only needed to reach its final annealing phase before becoming useful. It does not rule out a different continuation policy or an eventual recovery under an untested longer trajectory.

## Causal limits that belong in the report

- B is the immutable epoch4 prefix of C. C minus B tests continuation along that H16 trajectory, including later LR, weight-decay and dropout schedules. It does not isolate a schedule-independent effect of update count.
- B minus A changes declared schedule horizon at a fixed four epochs. It includes all horizon-dependent schedules and their stochastic consequences, not only LR. A is a new literal four-epoch endpoint; the historical four-cap member selected three epochs and is not this control.
- No single cause among capacity, LR, dropout, weight-decay schedule or the known legacy optimizer factor duplication is identified. The duplication is shared by all arms. Neither its existence nor this negative result proves it caused the degradation, and correcting it changes effective regularization.
- Inner curves describe a separate smaller training partition. Outer refits use the same epoch clock but more updates per epoch. Their agreement in the direction of deterioration is useful evidence, not exact inner-to-outer trajectory equivalence.
- A strictly increasing recalibration of C's final score cannot restore its lost standalone ranking AUC. Component calibration can affect a blend, but that is a different intervention and was not searched here. AUC-loss optimization is not established as the bottleneck by these results.
- Development rows have historical adaptive reuse. No audit labels or new independent public/private leaderboard result support this experiment. Fixed endpoints reduce within-run selection freedom; they do not erase earlier adaptation.

## One proposed future experiment, not authorized or launched

Prefer a single dropout continuation intervention to earlier LR cooling for the next mechanism test. Earlier cooling also changes effective decoupled weight-decay dose because the shrinkage update depends on LR, and the observed late cooling did not reverse degradation. Holding LR and weight-decay paths fixed gives the next comparison a narrower interpretation.

Recreate the H16 prefix through epoch4 twice under the exact frozen recipe, data order and seed, and verify full model, preprocessing, optimizer, progress and RNG state equality before intervention. Continue one trajectory with the original schedules. In the other, set effective dropout to the single predeclared0.05 base rate after epoch4 through epoch16 while preserving LR and every weight-decay schedule/factor. The original effective dropout is approximately0.003156 at epoch4 and0.001580 at epoch16;0.05 is an intentionally stronger stochastic-regularization hypothesis, not an estimated optimum or a promised gain. Do not add an optimizer correction, LR change, capacity change or dropout grid to this comparison.

Keep fixed epoch4/16 endpoints and the same passive inner/outer structure. Compare the modified16 endpoint against its matched original16 endpoint and common epoch4 prefix, and use only a prospectively registered10% candidate mixture with the existing pooled/macro/fold thresholds against the incumbent and both controls. Report all endpoints even if the intervention fails. Evidence that fitting improves less while held-out metrics improve would support useful regularization under this policy; a negative result would reject that fixed policy, not all regularization.

The existing TorchScript/native exports are inference artifacts and cannot provide an exact optimizer/RNG resume. A simple design recreates both prefixes within two full trajectories per partition:12 fits,192 epochs across three folds and inner/outer phases. Scaling this run's approximately34.5 minutes of total adapter time gives about55 minutes of adapter time, plus setup, process startup, full-fold inference and return verification. This is an extrapolation to re-estimate before a later authorized campaign, not a new budget or dispatch permission. Full-state branching could save repeated prefix work but requires separate mechanism verification and does not already exist in the saved inference checkpoints.

## Evidence identities

All files below are under `cloud/fixed_epoch_v1/assessment_workspace/artifacts/fixed_epoch_cloud_v1/`:

- `evaluation.json`, completed03:28:26.621764UTC, SHA256 `fab36afadcb2fe23d630593e906bed0cc66eb531772aa0ebc1f5383831cb875b`.
- `report/v1/evidence.json`, SHA256 `bdfe34357b622bcf96fb97b8ce587d46db6533cf68ed2f17ae3317162536d484`.
- `completion_receipt.json`, SHA256 `540e2022ed5b20b7f86586c677dacba1b754a815c43db7b0771e87c90a5e4e9c`.

These identities were independently read/rehashed. Metric values above are copied from the completed assessment; curve values are copied from the completed reporter evidence, with the fitting-probe definition checked against the frozen adapter source. No new metric calculation or fitting was performed for this review.
