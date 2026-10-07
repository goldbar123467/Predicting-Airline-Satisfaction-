# Review of the fixed-grid uncertainty diagnostic

Methods review, October 2, 2026 UTC. This addendum reviews `DEEP_RESEARCH_REPORT.md`, `configs/research_grid_uncertainty_v1.json`, the completed implementation, synthetic tests and saved output. It does not rerun OOF scores, add weights, fit a model, inspect audit data, or change any release decision.

The diagnostic has enough decision value to retain as one bounded analysis. The best observed grid gain is about 1.58e-6, and its paired uncertainty cannot be inferred from the much larger incumbent-versus-v2 contrast. A shared covariance calculation can describe how indistinguishable nearby grid points are under a specified approximation. The existing fold variation and failure to cross the operational gain gate already support retaining the incumbent; the uncertainty analysis adds a qualified explanation, not a new selection opportunity.

## Statistical contract

For grid contrast j, let Dpos[j] and Dneg[j] be its positive- and negative-row AUC placements minus the incumbent placements. The covariance estimator is the sample covariance of positive-row differences divided by the number of positives, plus the corresponding negative-row covariance divided by the number of negatives. Every contrast must use exactly the same keyed rows, class orientation and tie convention. This exploits paired scores and shared errors rather than treating models as independent.

With the fitted score functions treated as fixed, the method simulates a zero-mean Gaussian with that estimated covariance. The empirical 95th percentile of `max_j(abs(Z_j)/SE_j)` supplies an approximate joint critical value. Applying that one value to the 49 observed differences produces **conditional, iid-row Gaussian reference bands for this fixed grid**. Simultaneous coverage is an asymptotic approximation for that family of contrasts, not a finite-sample guarantee and not full post-selection coverage for the historical model/feature/anchor search.

Required implementation checks:

- Preserve exact row alignment and the declared grid-minus-incumbent direction consistently throughout output.
- Exclude zero-standard-error contrasts from Gaussian standardization and handle them deterministically. A zero denominator is not evidence of infinite significance.
- Report the smallest covariance eigenvalue and any numerical positive-semidefinite correction. Tiny roundoff clipping is reasonable; substantial correction or added jitter changes the model and needs explanation.
- Label the argmax statistic as the frequency with which each **of the 49 grid points** wins after Gaussian perturbation of its observed score. If the incumbent is not one of the 49, those frequencies do not describe winning against the incumbent.
- Disclose exact argmax tie handling. Report sensible precision for 20,000 draws; the simulation does not identify exact probabilities.

The argmax calculation is a stability diagnostic centered at noisy observed estimates. It is neither a posterior over true optima nor a probability that a grid point generalizes best. It does not cover the continuous weight simplex, new weight refinements, fold-dependent training randomness, historical selection, training-set overlap, or dataset shift. Gaussian covariance can summarize local sampling uncertainty without making those omitted sources small.

## Report interpretation

The current report appropriately separates schedule/refit confounding from ordinary overfitting and does not attribute the 500-cap regression to the shared optimizer issue. It correctly treats BCE/AUC mismatch as a possible later mechanism, with no evidence that it is the present bottleneck: all three recorded final metrics worsen together and the missing trajectory cannot support a late-loss diagnosis. The global binary-score calibration invariance and constituent-calibration distinction are correctly qualified.

Three minor wording corrections were sent to the integration owner: call 13,688.797 seconds pipeline elapsed rather than exact fitting time; replace “first four updates” with “first four epochs”; and describe inner curves as disjoint from fit rows rather than independently fresh evaluation. These do not change numerical findings.

Do not reinterpret a positive conditional lower bound as release authorization, or an interval containing zero as proof of equivalence. A band around one comparison cannot be recycled as the uncertainty of another contrast. The operational 1e-5 gate remains a decision rule, and any future independent evaluation must still rebuild the complete fitting/selection procedure inside the appropriate exclusion boundary.

## Completed implementation and output audit

Read `scripts/analysis_grid_uncertainty_v1.py`, its eleven synthetic tests and `artifacts/research_pass_v1/validation/grid_uncertainty.json`. Tests were run by the owning agent; this reviewer inspected their coverage and performed a separate saved-JSON/source-hash/arithmetic audit, without rerunning tests or reading OOF rows. Tests cover brute-force tie-aware placements, covariance diagonals against the paired implementation, row/column permutation, sign changes, zero and duplicate contrasts, PSD reconstruction, invalid inputs, relative roundoff clipping, deterministic seeds and the declared tie rule.

The final artifact SHA-256 is `c24f3793d9f859f481894f3674aec0c4776f836c4624a2a1647d6eac19e05894`. All six recorded source/input hashes matched. All 49 saved weights and AUCs match the preceding fixed-grid artifact. Covariance-diagonal-to-SE, interval-endpoint and frequency arithmetic had zero discrepancy; winner counts sum to 20,000. The source keeps the output isolated and refuses replacement. The observed covariance needed no eigenvalue clipping or diagonal jitter; no real-output contrast had zero SE, duplicate covariance columns or an argmax tie. The code's duplicate-column noise coupling is mathematically appropriate when identical covariance columns imply zero variance of their difference.

| Quantity for the observed 65% / 20% / 15% grid winner | Result |
| --- | ---: |
| Observed AUC difference from incumbent | +0.000001582992 |
| Paired placement standard error | 0.000003499915 |
| Conditional pointwise 95% reference interval | [-0.000005276716, +0.000008442700] |
| Conditional fixed-grid simultaneous 95% reference band | [-0.000007148659, +0.000010314643] |
| Joint Gaussian critical value | 2.494818 |
| Perturbation winner count among the 49 grid points | 5,176 / 20,000, or 25.88% |

None of the 49 contrasts has a positive simultaneous lower endpoint. This supports presenting the exact winner as unstable under the declared reference approximation. It does not prove equivalence or that every reweighting is worse. In particular, the best grid point's joint upper endpoint slightly exceeds 1e-5, so this analysis cannot rule out a gate-sized benefit. Retaining the incumbent remains the operational decision from its point-estimate/fold requirements and the absence of convincing evidence for a precise replacement coefficient.

The largest perturbation frequencies are spread across nearby weights: 65/20/15 at 25.88%, 60/20/20 at 18.84%, and 70/15/15 at 15.23%. These are Monte Carlo frequencies under the plug-in perturbation model, not probabilities that those blends are the true optimum. Planned extra array memory is 0.589 GiB with two numerical threads; peak RSS was not measured. No extra grid points or model fits were executed, and no release changed. No blocking implementation or interpretation issue remained in this audit.
