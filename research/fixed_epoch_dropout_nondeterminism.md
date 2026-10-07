# Dropout smoke failure and deterministic execution amendment

The first cloud job failed its exact epoch-four prefix gate before any real-data fit. The saved evidence is `cloud/fixed_epoch_dropout_v1/download_01/fixed_epoch_return/results.zip`, SHA256 `756585c1a6db44619c5f9028f51572a8d70c87daafd87706470e288cdd9c6225`. Only generated inner A and inner C trajectories exist in its returned inventory.

## What the saved smoke establishes

Cloud versions were PyTorch `2.10.0+cu128`, PyTabKit `1.7.3`, Lightning `2.6.6`, and NumPy `2.0.2`. A completed 16 epochs; C stopped at epoch 4. Both initialized to exactly the same network fingerprint. Seventeen of nineteen portable prefix components match, including optimizer progress, all random generators, sampled row order, current schedules, input data, schema, static preprocessing, model modes, and parameter-group inventory. The network and Adam-state fingerprints differ. The first four completed-update JSONL records are byte-identical. This smoke executed one optimizer update per epoch under the installed drop-last loader; the configured batch size is 256 and the eligible fit partition contains 384 generated rows.

The two already-saved native probability arrays differ by at most `1.7881393432617188e-7` over the same 257-row training probe. This is an arithmetic comparison of saved generated-smoke evidence, not new model inference or a quality score. A small inference difference cannot satisfy an exact continuation-state requirement. The rejection is therefore retained.

## Source-supported diagnosis, with limits

The evidence points toward numerical nondeterminism during CUDA training, rather than differing initial weights, random streams, sampled examples, or an early dropout intervention. It does not identify the first divergent operation or establish a particular CUDA kernel as the cause.

An active candidate is categorical embedding backward. Installed `pytabkit/models/nn_models/categorical.py:302` uses `emb_flat.index_select(0, x_cat)`. The saved parameter inventory contains four trained embeddings with shapes `[8,68,5]`, `[8,345,5]`, `[8,385,5]`, and `[8,375,5]`. PyTorch 2.10 documents differentiation of CUDA `index_select` among operations whose normal implementation can be nondeterministic and whose deterministic alternative is selected by the strict flag. The same API raises when a required deterministic implementation is unavailable. [PyTorch 2.10 API](https://docs.pytorch.org/docs/2.10/generated/torch.use_deterministic_algorithms.html)

Dense batched matrix multiplication also occurs in installed `nn_models/base.py:514` and numerical-embedding paths in `nn_models/nn.py`. NVIDIA documents cuBLAS reproducibility conditions and the `:4096:8` workspace configuration for deterministic behavior when concurrent streams share a handle. Neither stream contention nor a cuBLAS-specific failure was observed directly here. [NVIDIA cuBLAS reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)

The NVIDIA skill finder catalog was checked; no additional skill was installed. The relevant exact-library documentation and installed sources are sufficient for this bounded amendment.

## Minimal equal-arms amendment

`scripts/fixed_epoch_deterministic_execution_v1.py` is copied unchanged to an external `sitecustomize.py`. It targets only the exact frozen smoke and controller/worker entry paths. Before importing torch or initializing CUDA, it sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`; it then enables `torch.use_deterministic_algorithms(True, warn_only=False)`, disables cuDNN benchmarking, enables cuDNN determinism, and retains disabled CUDA matmul TF32. Current Lightning's `accelerator_connector.py:571–592` leaves the global deterministic flag unchanged when the frozen Trainer supplies `deterministic=None`.

Every targeted process writes an exclusive receipt identifying its parsed smoke/controller/worker role, worker fold/phase/trajectory where applicable, source hashes, amendment hash, library versions, and exact flags. Any startup failure exits with code 86 before the target executes, because Python otherwise can suppress ordinary `sitecustomize` exceptions. The installer and evaluator separately bind the source, authority, fourteen required process identities, and returned amendment bytes.

The scientific payload, optimizer semantics, seeds, horizons, dropout intervention, fixed endpoints, and exact prefix gate are unchanged. Deterministic execution can select different backend implementations, so this is an explicit execution-policy amendment with potential runtime and numerical differences from the failed attempt. PyTorch limits reproducibility guarantees across releases, platforms, and devices and warns that deterministic execution can be slower. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.10/notes/randomness.html)

## Executed checks and remaining gate

Eight generated CPU tests passed in 36.180 seconds. They include actual Python `sitecustomize` startup, fatal rejection before target execution, inherited policy in a child process, Lightning flag retention, restricted launch parsing, and all six existing CPU adapter tests running under the amendment. Receipt: `artifacts/fixed_epoch_dropout_v1/determinism_smoke/verification.json`.

No local CUDA execution, real-data fitting, model inference, or quality scoring was performed for this diagnosis. CPU success does not prove the cloud kernels now reproduce exactly. The fresh immutable retry must pass the unchanged full cloud smoke, including exact state equality at epoch four, before real fitting. An unsupported deterministic operation or another mismatch must terminate that retry; neither warning-only execution nor relaxed prefix tolerances is an accepted fallback.
