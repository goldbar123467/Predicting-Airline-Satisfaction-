# Fixed schedule and duration experiment: verified cloud result

Keep the current blend. The registered 16-epoch continuation failed the advancement gate: its standalone AUC dropped in all three outer folds, and its fixed 10% blend reduced AUC in all three folds. No release, weights or submission changed.

The private Kaggle job completed all 12 fits and 9 outer endpoints. The 185-file return archive, source/input identities, partitions and native receipts passed verification. The frozen evaluator ran once on 629,671 development rows. This remains exploratory evidence on historically reused development data, not independent confirmation.

## Results

A is a 4-epoch schedule executed for 4 epochs. B is the saved epoch-4 prefix of the 16-epoch schedule. C is that same trajectory at epoch 16. All arms used the same pinned cloud stack, recipe, paired seeds, preprocessing and row-order contracts. Inner monitors were passive.

| Candidate | Pooled OOF AUC | Mean fold AUC | Log loss | Brier |
|---|---:|---:|---:|---:|
| Current blend | 0.961407501 | 0.961411871 | 0.221727291 | 0.059288590 |
| A: H4 / E4 | 0.960689515 | 0.960696757 | 0.223701854 | 0.059729720 |
| B: H16 / E4 | 0.960544145 | 0.960667853 | 0.223774310 | 0.059804686 |
| C: H16 / E16 | 0.954466852 | 0.954470139 | 0.254469753 | 0.063872594 |
| 90% current + 10% A | 0.961405362 | 0.961409956 | 0.221706220 | 0.059276698 |
| 90% current + 10% B | 0.961393721 | 0.961401759 | 0.221747241 | 0.059289941 |
| 90% current + 10% C | 0.961305778 | 0.961310995 | 0.221823619 | 0.059289626 |

| Registered comparison | Pooled AUC change | Mean fold change | Fold 0 | Fold 1 | Fold 2 |
|---|---:|---:|---:|---:|---:|
| B minus A: horizon at epoch 4 | -0.000145370 | -0.000028904 | -0.000020346 | -0.000169846 | +0.000103479 |
| C minus B: continuation to epoch 16 | -0.006077293 | -0.006197714 | -0.006215870 | -0.005904739 | -0.006472533 |
| 10% C blend minus current | -0.000101723 | -0.000100877 | -0.000080331 | -0.000079398 | -0.000142901 |

B minus A has mixed fold signs. Its pooled decline is much larger than its mean-fold decline, so this experiment does not establish that a longer horizon at epoch 4 is uniformly harmful. C minus B is strongly negative in every fold. It measures continuation along the common H16 schedule, including its later learning-rate and regularization phases; it does not isolate update count independently of that schedule.

The prespecified C-mixture gate required pooled and mean-fold gains of at least 0.00001 against each of current, A mixture and B mixture, plus every fold delta at least -0.00002. All three comparisons failed their pooled, mean-fold and fold-regression checks. No candidate advances. The A mixture was 0.000002139 below the incumbent in pooled AUC; that small difference does not establish meaningful harm, but supplies no accepted gain.

## What the curves show

The evaluation-mode fitting probe improved sharply while passive held-out monitors deteriorated. These probe values describe a fixed subset of fitting rows, not independent validation.

| Inner fold | Train-probe log loss, E4 to E16 | Monitor log loss, E4 to E16 | Monitor AUC, E4 to E16 |
|---|---:|---:|---:|
| 0 | 0.206053 to 0.087994 | 0.222349 to 0.261398 | 0.961186 to 0.954153 |
| 1 | 0.221895 to 0.092693 | 0.220352 to 0.258663 | 0.961853 to 0.955143 |
| 2 | 0.212311 to 0.096694 | 0.225780 to 0.262065 | 0.959950 to 0.954016 |

This supports overfitting along the registered long trajectory, beyond a mismatch between training loss and ranking AUC: held-out log loss, Brier and AUC all worsen. Deterioration starts before the final learning-rate anneal, and that anneal does not recover the lost performance. Learning rate, dropout, weight decay, model capacity and the intentionally preserved legacy decay-factor behavior were not independently ablated, so none is identified as the sole cause.

[Learning curves](cloud/fixed_epoch_v1/assessment_workspace/artifacts/fixed_epoch_cloud_v1/report/v1/inner_monitor_and_lr.png) show passive inner AUC and recorded first/last-update learning rates. Lines are epoch samples, not a reconstruction of every update.

## Blend amount decision

Retain the current 64/16/20 component blend. Add zero weight from this experiment under the current decision. The only new mixture weight assessed was the frozen 10%, which failed; this does not prove that zero is a globally optimal coefficient or that every small positive weight is harmful. No adaptive alpha search, calibration, checkpoint search or repeat of the old 12-mixture screen was performed. A later weighting study needs a newly frozen small candidate set and selection-aware validation, not a finer search around this failed run.

## Execution and verification

- Private kernel `clarkkitchen/s6e10-fixed-epoch-20261003-r2`, immutable version 1, kernel 136851188. One visible T4 was used on the provider host. No local GPU training.
- Started 2026-10-03 02:21:45.307147 UTC; training completed 03:06:51.029051 UTC. Actual cloud setup/smoke/controller duration: 2,705.722 seconds, or 45.10 minutes, below the 6,300-second fit budget and 7,200-second provider cap.
- Twelve adapter trajectories totaled 2,072.629 seconds including their telemetry/export. That sum excludes worker imports and later full-fold native reloads and is not billed GPU time.
- Generated cloud smoke passed in 38.84 seconds with the production-sized eight-member architecture, actual feature/transform path and five native endpoint checks; CPU/GPU probe parity passed.
- All nine full outer-fold native reload receipts passed, with zero maximum absolute difference between first native inference and independent raw-feature/transform/native reload. These full-fold checks ran on cloud; local assessment verified their exact files/receipts and did not execute inference.
- A/C initialization, training partition/order, fitted transformer and captured-probe contracts passed. The three installed PyTabKit source files exactly matched the frozen hashes.
- Canonical development IDs/folds matched exactly. No audit/test rows were uploaded or scored. Local assessment did not decode raw cloud train/auxiliary data; it verified their hashes.
- Preserved ZIP: 285,643,858 bytes, 185 manifest files. ZIP SHA256 `998e8299def11923a3e17563cc0329b714656b7b4428cb680fe32acd270a9742`; return-manifest SHA256 `6a416df1551c423c5be7f736066181266cf978ab34dde2ae61fc4c6bf21ebb4d`.
- Frozen local protocol SHA256 `1fbf6c31da6d582555026c1078f8314c7e2d46b221de43e2838bc305adcc7652`; completed evaluation SHA256 `fab36afadcb2fe23d630593e906bed0cc66eb531772aa0ebc1f5383831cb875b`.
- Readiness checks included 25 evaluator tests, 10 original retrieval tests, 5 transport-amendment tests and 4 reporter tests, plus prior adapter/runner/package checks. Independent reviews covered source and final interpretation.

## Preserved failures and transport amendments

The earlier local attempt completed zero epochs before its RAM guard stopped it. The first cloud job stopped before setup or fitting because Kaggle automatically unpacked its ZIP. A new immutable kernel corrected only mounted-payload discovery/copying and reused the identical scientific payload. During retrieval, the file-size endpoint exposed basenames and unusable sizes; its failed retrieval claim is preserved in `retrieval_01`. A separately tested transport wrapper used exact signed full paths and identity-encoded GET Content-Length, preserving the original archive/assembly checks. Successful original downloads are in `cloud/fixed_epoch_v1_retry1/retrieval_02`. These changes did not modify fits, endpoints, weights or scoring gates.

## Reproducibility and limits

Executed commands, retained as provenance rather than instructions to rerun completed scoring:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/retrieve_fixed_epoch_cloud_transport_v2.py --download-dir cloud/fixed_epoch_v1_retry1/retrieval_02 --include-log --assemble
.venv/Scripts/python.exe -X utf8 scripts/evaluate_fixed_epoch_cloud_v1.py --workspace cloud/fixed_epoch_v1/assessment_workspace
.venv/Scripts/python.exe -X utf8 scripts/summarize_fixed_epoch_cloud_v1.py --workspace cloud/fixed_epoch_v1/assessment_workspace --outputdir artifacts/fixed_epoch_cloud_v1/report/v1
```

Evaluation claim and result are preserved under `cloud/fixed_epoch_v1/assessment_workspace/artifacts/fixed_epoch_cloud_v1/`. Sources, inputs, endpoint receipts and partitions are bound by the registry, local protocol, returned manifest and evaluation provenance. Descriptive curve evidence and the PNG are under `report/v1/`.

The same development folds have been used historically. No confidence interval, significance claim, nested generalization estimate or public-leaderboard improvement follows from these three fold contrasts. All new arms used the same cloud stack; comparisons to historical local training are not pure epoch effects. The endpoint-4/16 contrast cannot locate an optimal stopping epoch or identify a single mechanism.

## Next experiment

The next proposed test holds H16, duration, learning rate and weight decay fixed, and changes only dropout after an identical epoch-4 prefix: original decaying dropout versus the configured base value 0.05 maintained through epoch 16. Compare final endpoints against matched control and common prefix. Recreate and verify full training state; saved native inference graphs cannot resume it. The bounded design, assumptions and future acceptance gate are in `FIXED_EPOCH_CLOUD_NEXT_EXPERIMENT.md`. No additional job is launched.
