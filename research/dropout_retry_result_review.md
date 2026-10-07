# Independent review of the dropout continuation result

Reviewed October 4, 2026. The registered decision is **no advancement**. Sustaining dropout at the configured base of 0.05 after epoch 4 improves the sixteen-epoch model relative to its matched scheduled-dropout control, but it does not make continuation beneficial relative to the shared four-epoch checkpoint or produce a useful addition to the incumbent at the tested 10% weight. I found no arithmetic or gate contradiction in the saved assessment.

This review reads only the saved evaluation, frozen assessment protocol and registered plans. It checks arithmetic on recorded scalar summaries; it does not recompute AUC, decode predictions or labels, read audit data, fit models, make network calls, search weights, or modify frozen artifacts.

## Evidence identity

- `cloud/fixed_epoch_dropout_v1_retry1/assessment_workspace/artifacts/fixed_epoch_dropout_v1/evaluation.json`: SHA256 `91aede41316207b6662848efaf73f97f40c9a96b8f078c18b7faaab7e52c3d27`; completed `2026-10-04T20:07:47.164186+00:00`.
- `artifacts/fixed_epoch_dropout_v1/local_evaluation_protocol.json`: SHA256 `0e3c838028058b80c82d8bc63e1737aac7631405ca8c13124134c77924207a2c`; frozen `2026-10-04T18:46:58.310881+00:00` before this assessment.
- `FIXED_EPOCH_DROPOUT_V1_PLAN.md`: SHA256 `48066e211e44079caebd5e225563ca57ad1cd3bdc1228e612b5d2eb7516ea7a2`.
- The motivating proposal, `FIXED_EPOCH_CLOUD_NEXT_EXPERIMENT.md`: SHA256 `4a14587a98969a1e45394afc3123cebdbdb259b093d431e8fe57335801af56a0`.

The evaluation records 629,671 development rows, 12 completed trajectories, nine native outer endpoints, six exact prefix comparisons, three full-fold prefix native checks and fourteen verified deterministic startup processes. It binds evaluator SHA256 `8ff9c036ae4480b4e7bcadbecf0700adf79d8df4fc9dc36b9629ddaac535676d` and the above protocol. This is a readback of the completed provenance, not a second execution of the prerequisite validators.

## Recorded results and gate

Both trajectories used horizon 16 and ended at epoch 16. Logical A is scheduled-dropout control E16; B is the exact shared E4 prefix; C is E16 with the configured dropout base held at 0.05 after E4. A is not the historical short-horizon four-epoch model.

| Score | Standalone pooled AUC | Mean-fold AUC | Pooled AUC of 90% incumbent + 10% score |
|---|---:|---:|---:|
| Incumbent | 0.961407500920 | 0.961411871318 | not applicable |
| A: scheduled E16 | 0.954543418753 | 0.954542509354 | 0.961317256791 |
| B: shared E4 | 0.960546132226 | 0.960687312152 | 0.961391526902 |
| C: sustained-dropout E16 | 0.955469094940 | 0.955471336114 | 0.961360066916 |

C minus A standalone is **+0.000925676188** pooled and **+0.000928826761** mean-fold AUC. All three fold differences are positive: +0.001152126413, +0.000783991319 and +0.000850362550. C minus B is **-0.005077037286** pooled, **-0.005215976037** mean-fold and negative in every fold. Relative to the control's B-to-A pooled loss of 0.006002713473, the treatment recovers approximately 15.42%. That ratio is descriptive arithmetic for these endpoints, not an estimate of a universal attributable fraction.

The treatment mixture passes the registered comparison against the control mixture, improving pooled AUC by +0.000042810125 and mean-fold AUC by +0.000042738916, with every fold positive. It fails both other required comparisons:

| Treatment-mixture reference | Pooled AUC difference | Mean-fold difference | Fold differences | Gate |
|---|---:|---:|---|---|
| Incumbent | -0.000047434004 | -0.000047330853 | -0.000021365292, -0.000040194892, -0.000080432374 | fails pooled, macro and fold limits |
| Prefix mixture | -0.000031459986 | -0.000036537968 | -0.000011338611, -0.000023695568, -0.000074579725 | fails pooled, macro and fold limits |

The gate requires pooled and mean-fold gains of at least 0.00001 versus **each** incumbent/control-mixture/prefix-mixture reference and no fold regression worse than -0.00002. Improvement against one reference cannot compensate for failure against another. Against the incumbent all three treatment-mixture folds exceed the permitted regression; against the prefix mixture two do. `advance_to_confirmation: false` is correct. There is no release, submission or incumbent-weight change supported by this result.

## What the comparison establishes

Given the verified common prefix, matched duration, identical data order and non-dropout schedules, the A-to-C difference supports a beneficial effect of this particular post-E4 dropout intervention on the recorded development evaluation. The intervention includes changing subsequent dropout masks and the resulting optimization trajectory. It does not identify an optimal dropout value or establish that a decaying dropout schedule is the sole cause of long-run degradation.

The improvement also appears in standalone log loss and Brier: C improves on A by 0.004944854960 and 0.000756299002, respectively, with lower being better. However, continuing from B to C worsens log loss by 0.025280807345 and Brier by 0.003196949694. Thus the failure of continuation is present in ranking and both reported probability scores; a simple claim that only the AUC objective is mismatched does not explain these endpoints. This bounded review did not read fitting-loss curves, so it does not independently establish whether every remaining component of degradation is overfitting, optimization behavior or another mechanism.

The C mixture has a slightly lower Brier than the incumbent, by 0.000006809871, while its log loss is worse by 0.000041199424 and AUC is worse. That does not overturn the preregistered AUC decision. Lower aggregate Brier alone also does not prove improved calibration, because Brier reflects discrimination/resolution as well as calibration.

Only one candidate fraction, 10%, was tested. None of the three tested additions improves pooled AUC over the incumbent. This supports retaining the current blend under the registered policy. It neither proves that zero is globally optimal for these models over all possible weights nor identifies globally optimal 64/16/20 family weights. Searching a smaller alpha after seeing this failure would be a new adaptive experiment, not a reinterpretation of a passing result.

All three validation folds belong to historically reused development data, and their training sets overlap. Their consistent signs are useful descriptive evidence, not three independent replications. This is one registered model-seed policy and one dataset/environment. No confidence interval, significance claim or cross-seed generality is established. The deterministic repair improves repeatability of the paired comparison; it does not make development reuse independent or guarantee results on another platform.

## One proposed follow-up, design only

The next informative test is **earlier learning-rate cooling during the same sixteen-epoch continuation**, conditional on sustained dropout. Do not run it under this note.

Use two newly reproduced, exact shared H16/E4 prefixes. Both future arms keep dropout at 0.05 after E4, preserving the best tested E16 dropout policy as the matched control. The control retains the original learning-rate schedule. The treatment changes only the post-E4 learning-rate multiplier: linearly cool its actual value at the E4 boundary to zero by E16. For epoch-fraction clock t in [4,16], the proposed common LR multiplier is m(4) times (16-t)/12, with existing per-parameter LR factors retained. This avoids selecting a new floor or endpoint after results are visible. Keep all other schedules, architecture, seeds, partitions, preprocessing and legacy optimizer semantics fixed.

This asks whether reducing the amount of high-rate optimization after the common prefix improves the remaining continuation damage. Changing LR also changes effective decoupled weight-decay dose even when WD coefficients and schedules are unchanged, so the result would identify a continuation-policy effect, not pure learning rate independent of regularization. Do not combine this test with fixing the optimizer's legacy decay-factor behavior or changing model capacity.

A direct counterpart would again be 12 fits and 192 executed epochs, with passive inner monitoring, fixed E4/E16 endpoints, exact-prefix rejection and strict deterministic execution. Recreate full state from the registered seed; the native inference graphs are not resumable optimizer/RNG checkpoints. Predeclare comparisons to both matched E16 control and shared E4, and retain only the fixed 10% incumbent-mixture gate if ensemble usefulness is tested. A result that merely improves a poor E16 control while still losing to E4 and the incumbent would again fail advancement. Any positive result would still need a separately designed confirmation before promotion. A future run needs its own frozen execution policy and bounded cloud authorization; this review performs none.
