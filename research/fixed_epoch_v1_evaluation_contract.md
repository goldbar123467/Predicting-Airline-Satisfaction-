# Fixed A/B/C campaign evaluation contract

Prospective contract for the user-authorized `fixed_epoch_v1` experiment. The evaluator performs no fit, submission, release selection or incumbent mutation. It must not be run on real artifacts until all twelve trajectories and nine verified outer endpoint predictions are complete. Root owns the training runner and full raw/native reload verification; this evaluator independently verifies its saved receipts, hashes, row identities and final metrics.

## Immutable campaign wrapper and registry

`configs/fixed_epoch_v1.json` has these required keys:

```json
{
  "id": "fixed_epoch_v1",
  "registry_path": "artifacts/fixed_epoch_v1/registry.json",
  "registry_sha256": "SHA256",
  "output_dir": "artifacts/fixed_epoch_v1",
  "split_path": "data/splits.parquet",
  "split_sha256": "4e262277b0a1494cd5d26ff45a30c827480ef334974f1331d730df0a7c80075c",
  "incumbent_selection_path": "artifacts/third_pass/blend/frozen.json",
  "incumbent_selection_sha256": "bc773bb7a65a3357ac82f1553ebd852e0fc2773cf6683650dce5fcf5166b3311",
  "completed_manifest_path": "artifacts/fixed_epoch_v1/completed_manifest.json",
  "completion_receipt_path": "artifacts/fixed_epoch_v1/completion_receipt.json"
}
```

Registry is created once before real fits. Alongside the full recipe, data/bank identities, seeds, bounds and source record, it includes:

```json
{
  "id": "fixed_epoch_v1",
  "expected_fit_count": 12,
  "source_hashes": {"scripts/evaluate_fixed_epoch_v1.py": "SHA256"},
  "evaluation": {
    "mixture_alpha": 0.1,
    "min_pooled_gain": 0.00001,
    "min_macro_gain": 0.00001,
    "max_fold_regression": 0.00002,
    "class_order": [0, 1],
    "fold_ids": [0, 1, 2]
  }
}
```

`source_hashes` includes the runner, adapter and any other versioned sources, not merely the evaluator shown in the example. All listed source bytes must match; the evaluator's own path/hash is required. The registry cannot include the wrapper's hash because the wrapper pins the registry; completion records bind both after writing the wrapper.

## Completion manifest and receipt

The completed manifest binds the campaign wrapper and registry, exactly twelve `(phase, fold, trajectory)` tuples from `{inner,outer} × {0,1,2} × {A,C}`, and exactly nine `(fold,arm)` tuples from `{0,1,2} × {A,B,C}`:

```json
{
  "id": "fixed_epoch_v1",
  "campaign_sha256": "WRAPPER_SHA256",
  "registry_sha256": "REGISTRY_SHA256",
  "trajectories": [{
    "phase": "outer", "fold": 0, "trajectory": "C",
    "receipt_path": "artifacts/fixed_epoch_v1/fold_0/outer/C/done.json",
    "receipt_sha256": "SHA256"
  }],
  "endpoints": [{
    "fold": 0, "arm": "B",
    "prediction_path": "artifacts/fixed_epoch_v1/fold_0/outer/predictions_B.parquet",
    "prediction_sha256": "SHA256",
    "native_receipt_path": "artifacts/fixed_epoch_v1/fold_0/outer/native_verify_B.json",
    "native_receipt_sha256": "SHA256"
  }]
}
```

Lists above show one element only; all twelve/nine unique entries are mandatory. Every trajectory receipt contains `id`, `status="completed"`, `phase`, `fold`, `trajectory`, `registry_sha256`, `schedule_horizon_epochs`, `executed_epochs`, `checkpoint_epochs`, `telemetry_path`, `telemetry_sha256`, `partition_path`, `partition_sha256`, `adapter_receipt_path` and `adapter_receipt_sha256`. A requires H4/E4 with `[4]`; C requires H16/E16 with `[4,16]`. B is C's captured epoch four, not an additional trajectory. Telemetry bytes are verified without scoring their contents.

The evaluator independently verifies each hashed `trajectory.json` adapter receipt: completed status, horizon/executed endpoint list, epoch-fraction clock, unchanged legacy optimizer semantics, phase/fold/trajectory/partition context, telemetry hash, all endpoint graph/metadata hashes and recorded source hashes. Each `partitions.npz` is loaded with pickle disabled after filtering the authoritative split to development. Training/monitor identities must be unique, mutually disjoint and exclude the exact held-out fold; their union must equal the permitted outer-training population. Outer fits have no monitor; inner monitor count is the registered 10% split with sklearn's ceiling convention. Membership/count checks do not independently reconstruct seeded row order; that behavior is enforced by the pinned runner/adapter and matched-control receipt.

The final completion receipt is:

```json
{
  "id": "fixed_epoch_v1", "status": "completed",
  "campaign_sha256": "WRAPPER_SHA256",
  "registry_sha256": "REGISTRY_SHA256",
  "completed_manifest_sha256": "SHA256",
  "completed_fit_count": 12, "completed_endpoint_count": 9,
  "evaluation_ready": true
}
```

It is written only after all fits, endpoint exports, full raw/native reload checks and integrity checks finish. An incomplete/failed/duplicate trajectory or endpoint prevents any new metric calculation. Paths are workspace-relative and all new artifact paths remain inside `artifacts/fixed_epoch_v1`.

## Keyed outer predictions and native receipts

Each endpoint parquet has exactly `id`, `fold`, `satisfaction`, `prediction`, with one row per authoritative ID in its outer fold, binary label equal to verified incumbent development OOF, and a finite positive-class probability in `[0,1]`. IDs and fold values are integral. Row order may differ; all scoring uses a validated keyed join. Missing, extra, duplicate, audit (`fold < 0`) or mislabeled rows fail before scoring. Both classes must occur in each scored fold. The evaluator filters the frozen split to development before any label/prediction joins and never reads raw train, audit predictions or test predictions.

The native verification receipt contains:

```json
{
  "id": "fixed_epoch_v1", "status": "passed", "fold": 0, "arm": "B",
  "registry_sha256": "REGISTRY_SHA256",
  "verification_scope": "full_outer_fold", "class_order": [0, 1],
  "row_count": 209890, "ids_sha256": "SORTED_INT64_IDS_SHA256",
  "prediction_path": "artifacts/fixed_epoch_v1/fold_0/outer/predictions_B.parquet",
  "prediction_sha256": "SHA256",
  "model_directory": "artifacts/fixed_epoch_v1/fold_0/outer/C/epoch_004",
  "artifact_hashes": {
    "artifacts/fixed_epoch_v1/fold_0/outer/C/epoch_004/graph.pt": "SHA256",
    "artifacts/fixed_epoch_v1/fold_0/outer/C/epoch_004/metadata.json": "SHA256",
    "artifacts/fixed_epoch_v1/fold_0/outer/C/transform.json": "SHA256"
  },
  "reference_kind": "first_native_endpoint_reload",
  "adapter_parity_scope": "fixed_probes_at_capture",
  "atol": 0.000002, "rtol": 0.00001,
  "max_absolute_error": 0.0, "max_scaled_error": 0.0,
  "parity_passed": true, "raw_reload_verified": true
}
```

The example row count is illustrative; actual count must exactly match the frozen fold. ID digest is `SHA256(np.sort(ids).astype('<i8').tobytes())`, making it independent of row order. A uses `outer/A/epoch_004`; B uses `outer/C/epoch_004`; C uses `outer/C/epoch_016`. `artifact_hashes` must cover endpoint `metadata.json`, the exact `graph_file` it names, and exact trajectory `transform.json`. Metadata embeds `input_schema`; there is no separately consumed schema file in the current adapter. Its graph hash, class order, feature count, literal executed epoch and schedule horizon are checked. An arbitrary `.pt` without the loader-consumed metadata/graph binding is insufficient.

Root's first endpoint reference is a native reload using the in-memory fitted `CategoricalTransform` and precomputed raw-derived features, in chunks of 32768. The independent pass discards and reloads model plus transformer, rebuilds `common.features` from only raw held-out feature columns without labels, and uses chunks of 8191. This is full-outer **raw/native reload consistency**, not full-outer live-framework prediction parity. The adapter separately verifies installed-framework versus portable native export on fixed probes at capture. These two scopes are recorded explicitly and must not be conflated.

Record `max_scaled_error = max(abs(reference-reload)/(atol+rtol*abs(reference)))`; it must be finite and at most one. Both probabilities must have explicit `[0,1]` class order and pass schema/domain checks.

The evaluator rehashes all listed native files and checks their receipt bindings. Its report must say **runner-produced full raw/native verification receipt independently checked**, not claim the evaluator itself re-executed native inference.

## Fixed calculation and advancement

After all completion, source, artifact and row checks pass, reconstruct the unchanged incumbent from its hash-verified selected development OOF files. Compute pooled ROC AUC, each original fold's ROC AUC and their unweighted mean, plus log loss/Brier as descriptive secondary metrics. No secondary metric changes the acceptance rule. Score A/B/C and exactly `0.9*incumbent + 0.1*arm` for each arm. Report B−A, C−B, C−A for standalone and mixture predictions; also all mixtures versus incumbent.

Advance C to a separate confirmation study only if C's fixed mixture exceeds **each** of incumbent, A mixture and B mixture by at least `1e-5` pooled AUC **and** `1e-5` mean-fold AUC, with no fold difference below `-2e-5` against any of these three comparators. Report every criterion and failure. These comparisons use literal float64 metric differences without an extra threshold relaxation. No arm or alpha is selected as a substitute success rule.

An exclusive `evaluation_claim.json` prevents concurrent or repeated scoring attempts; `evaluation.json` is also created exclusively. A failure after claiming requires explicit recovery investigation and a new registered output namespace, not silent rerunning. All input/source hashes and actual invocation are recorded. There is no release/submission action and no modification to frozen weights. Results remain development evidence on historically reused data, not genuinely nested or independent confirmation.

## Implementation verification before registry freeze

`scripts/evaluate_fixed_epoch_v1.py` implements this contract without importing old training/release entry points. Nineteen synthetic tests in `scripts/test_evaluate_fixed_epoch_v1.py` pass. They exercise reordered key joins; missing/duplicate/extra IDs; wrong folds, labels, classes and dtypes; nonfinite/out-of-domain probabilities; both-class requirements; audit exclusion; complete inventory; partial campaign failure before any prediction read/metric; partition exclusion; native scope, graph/metadata/transform/hash failures; all three gate comparators and directions; macro/fold requirements; literal threshold boundaries; and exactly the fixed seven scored vectors/nine contrasts. No competition files were read and no real evaluation was executed by these tests.

```powershell
.venv/Scripts/python.exe -m unittest discover -s scripts -p test_evaluate_fixed_epoch_v1.py -v
```

Source review of `scripts/run_fixed_epoch_v1.py` found its wrapper/manifest/completion/native receipt names and endpoint layout consistent with this contract. The adapter confirms endpoint metadata embeds the input schema and names the hashed graph. Full runner/adapter preflight remains separately owned and reviewed; this agreement is not an executed campaign result.
