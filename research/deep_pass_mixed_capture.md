# Eight-member mixed-input capture and passive-monitor proof

Completed October 2, 2026 UTC. This is an executed synthetic CPU integration check, not a competition training run, production-ready adapter or model-quality result. No production or installed-library file changed, and no competition data, audit predictions or release artifacts were read or scored.

The test closes a specific limitation of the earlier capture prototype: it combines eight vectorized members, the categorical adapter's explicit preprocessing sequence, learned preprocessing state, real validation monitoring and cloned mid-training export in one run. The registered baseline and instrumented arm end with exactly identical parameters, buffers, Adam state and probabilities. Their full epoch, minibatch, RNG, schedule and monitor-prediction traces agree.

## Registered contract and execution

Registry: `artifacts/research_pass_v1/epochs/mixed_capture/registry.json`, dated `2026-10-02T21:54:14Z`, SHA-256 `6e7c5632c5842377ca6d33074bed6822ee22cbcbb0e8889f5fa909db950f9aaa`. It predates execution, limits attempts to an initial run plus two repairs, and bounds each attempt to five minutes on one CPU numerical thread. The implementation uses deadline checks at initialization, batches, observation and export boundaries; the actual successful fit/check phase took 3.421 seconds, well inside that bound.

There are 384 generated float32 feature rows: 288 fit rows and 96 monitor rows. Six continuous columns have deliberately nonzero centers and nonunit scales. Two raw category columns have four and 24 training levels; the schema adds a reserved unknown code, yielding cardinalities five and 25. Monitor-only categories 999 and 777 are excluded from training vocabularies. Fourteen derived probe rows exercise the explicit -1 sentinel and an unseen category 12345. Binary int64 labels come only from the deterministic generator. Training seed is 23197; data seed is 81271.

Both arms use eight small `[16,8]` members, batch size 32, prediction batches of 37, fixed horizon 16, and the production transform list:

```text
one_hot, median_center, robust_scale, smooth_clip, embedding, l2_normalize
```

Both use a real monitor loader, `use_early_stopping=False` and the low-level `use_best_epoch=False` flag. The baseline keeps read-only audit capture. The intervention adds fixed fit-row eval probes and, at epoch four, deep copies the schema, fitted static processing and fitted vectorized network, applies the actual production portability rewrite, traces/saves/reloads the graph and continues training to epoch 16. Installed optimizer semantics remain unchanged.

This matches the categorical transform and eight-member architecture path, not every production recipe setting: the hidden layers are deliberately small, execution is CPU, and the prototype explicitly uses `ls_eps=0.0` while the categorical adapter's default is 0.01 with a schedule. Neither arm changes that setting during this test.

One integration test passed:

```powershell
.venv/Scripts/python.exe -m unittest discover -s scripts -p test_research_mixed_capture_v1.py -v
```

The successful test took 3.482 seconds after imports. An initial attempt failed before optimization after 0.204 seconds because the inspection code incorrectly looked for learned preprocessing only in `creator.static_model`. The one repair expanded the inventory to the actual fitted network as well. That failed source is preserved at `mixed_capture/attempt_1_source.py`; both attempt receipts are retained. The registry and model semantics were unchanged. No additional fits are needed for this accepted proof. The test intentionally refuses to replace completed verification; any later reproduction requires a separately registered isolated output namespace rather than rerunning over this evidence.

## Where fitted preprocessing actually lives

The active PBLD factory uses a parallel numerical/categorical structure. Its learned preprocessing is retained inside the vectorized network. Six nontrainable tensors were found under `/tfms`, all registered and all with the leading eight-member axis:

| Network path | Fitted operation | Shape |
| --- | --- | --- |
| `tfms.0.layers.0.tfms.1.tfms.1` | Continuous median bias | `[8,1,6]` |
| `tfms.0.layers.0.tfms.1.tfms.2` | Continuous robust scale | `[8,1,6]` |
| `tfms.0.layers.0.tfms.1.tfms.4` | Continuous L2 normalization scale | `[8,1,6]` |
| `tfms.0.layers.1.tfms.1.tfms.2` | One-hot branch median bias | `[8,1,4]` |
| `tfms.0.layers.1.tfms.1.tfms.3` | One-hot branch robust scale | `[8,1,4]` |
| `tfms.0.layers.1.tfms.1.tfms.6` | One-hot/embedding branch L2 scale | `[8,1,9]` |

The static portion remains stateless in this configuration. The continuous median and scaling tensors are nonidentity, so this is substantive learned-preprocessing coverage. Every fitted preprocessing tensor remains unchanged during optimization. A separate compile-only comparison changes all monitor continuous features and category values while keeping fit rows and seeds fixed. The training vocabularies, initial network tensors and preprocessing tensors remain exactly identical. This tests the fitting exclusion boundary for this generated case; it is not a universal leakage proof.

The completed JSON retains a narrowly named `learned_static_preprocessing_unchanged` field that checks the empty static owner. The six actual learned tensors are covered by the explicit before/after `preprocessing_inventory` equality assertion in each `run_arm`, the complete network-state comparisons, and the full tensor-attribute signatures around observation. The static field alone is not the evidence for learned-state preservation.

A preliminary source-review suspicion that the single-split member loop might repeat nonidentity static preprocessing is **retracted for this active PBLD configuration**. `NNCreator` does append the data-transform stage in that loop, but the realized learned transforms here sit in the individual parallel network. There is no executed evidence of repeated nonidentity static preprocessing in this proof. No such defect should be added to the competition diagnosis.

## Observed results

| Acceptance condition | Executed result |
| --- | --- |
| Both real monitor loaders execute | 48 validation batches per arm, across 16 epochs |
| Literal endpoint, independent of monitored best | Both monitor traces identify epoch 15 as best; both retain literal epoch 16 and complete 144 optimizer updates |
| All 16 epoch states and final model agree | Maximum absolute difference 0 |
| Final probabilities and underlying Adam state agree | Maximum absolute difference 0 for each |
| Added observation is passive | Every captured live tensor/mode/forward-method/RNG/Adam/progress/schedule check passes; full batch and real-monitor traces match |
| Clone ownership | Registered and direct/container tensor-attribute storage is disjoint from live training storage |
| Native/portable/reloaded parity | Maximum absolute difference 1.1920929e-7, below preregistered `atol=2e-6`, `rtol=1e-5` |
| Input layouts | Sizes 1,7,37,96,384 and reversed/split batches pass; reverse difference 0, split-batch difference 1.1920929e-7 |
| Unknown behavior | Sentinel and unseen categories map to zero and produce exactly equal predictions |
| Snapshot immutability | Epoch-four artifact bytes and reloaded predictions remain unchanged after continuation; reload difference 0 |
| Continuation is nontrivial | Epoch-16 versus epoch-four parameter maximum difference 2.4273553 |

The native prediction reference is the installed `get_predict_dataloader` and `predict_step` path followed by its final probability conversion. Export parity is therefore checked against that separate native path, not only against the exported graph itself. The portability helpers exercised one one-hot operation, one embedding operation and three encoding operations. Small probability differences reflect float32 operation/normalization order and remain within the declared tolerance. Actual synthetic monitor errors are preserved only as mechanism diagnostics; they are not reported as generalization gains.

Evidence: `artifacts/research_pass_v1/epochs/mixed_capture/verification.json`, SHA-256 `d1d2ae2795dd95f3154371a2c32299b2e2bc90e29132b231ff01c29bf482e0a6`. The saved graph SHA-256 is `48a57963b8064c0057376e4b010b47e0031c0488a7c9cc7adf8491c22cc9de82`. All eight recorded source hashes matched on readback. Installed versions were PyTabKit 1.7.3, Torch 2.14.1+cu130 running CPU, PyTorch Lightning 2.6.6 and NumPy 2.5.3.

## Remaining limits

This proves the combined mechanism for generated data, a small eight-member network and this installed CPU environment. It does not prove CUDA or cross-platform equivalence, production fitter integration, full-scale throughput, checkpoint resumption, additional external `common.py` feature transformations, or a gain from more epochs. The tensor-ownership inspection covers registered tensors and direct/container tensor attributes on modules; it does not recursively certify arbitrary Python object graphs. All learned preprocessing tensors in this realized case were registered, so a nontrivial unregistered-static-statistics case remains unexecuted.

The result makes a fixed-endpoint implementation more concrete. It does not authorize launching the proposed competition experiment, modify the historical failed duration gates, or change the current blend.
