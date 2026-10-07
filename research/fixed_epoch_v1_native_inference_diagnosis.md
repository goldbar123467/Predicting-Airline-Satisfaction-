# Fixed-epoch CUDA native-inference diagnosis

Reviewed 2026-10-03, after the coordinator's generated-data CUDA smoke. This reviewer performed source and saved-artifact inspection only: no new fitting, GPU work, or production-source edits.

The bounded evidence supports disabling TorchScript executor optimizations only around native prediction in the new fixed-epoch runner. The failure is associated with the optimized execution path after the first call. The particular C++ optimization pass, compiler, or kernel responsible remains unidentified.

## Executed evidence

Commands below were run by the coordinator in separately owned, bounded subprocesses, from the project root:

```powershell
.venv/Scripts/python.exe scripts/smoke_fixed_epoch_v1.py --device cuda --output artifacts/fixed_epoch_v1/smoke_cuda_01
.venv/Scripts/python.exe scripts/diagnose_fixed_epoch_native_v1.py
.venv/Scripts/python.exe scripts/diagnose_fixed_epoch_native_v1.py --unoptimized
.venv/Scripts/python.exe scripts/smoke_fixed_epoch_v1.py --device cuda --output artifacts/fixed_epoch_v1/smoke_cuda_02
```

The initial smoke completed the inner H16 trajectory in 17.594 seconds, including endpoint exports at epochs 4 and 16, then stalled during native reload/inference. Its 300-second registered budget was exceeded; the coordinator terminated the identity-verified process and preserved `smoke_cuda_01/failure.json` and both checkpoints. That initial receipt alone does not isolate loading from execution.

The subsequent diagnostic uses the unchanged saved epoch-4 graph, generated inputs, four CPU threads, and Python stack dumps every 20 seconds. The default run was bounded by the coordinator to 60 seconds. Times below are elapsed from the diagnostic's internal timer, after imports; they are not whole-process times.

| Configuration | Loaded | First 37 rows | Second identical 37 rows | 17 rows | 128 rows |
|---|---:|---:|---:|---:|---:|
| Default optimized executor | 0.422 s | 1.110 s | Did not complete before termination | Not reached | Not reached |
| `optimized_execution(False)` | 0.328 s | 0.781 s | 0.797 s | 0.828 s | 0.860 s |

The default stack dumps remain inside the loaded model call: `scripts/realmlp.py:119`, called by `scripts/realmlp_categorical.py:209`. The second call has the same batch shape and values as the first. The failure therefore does not require a new batch shape. Both runs explicitly use four threads; thread oversubscription alone cannot explain this reproduction. Matching first-call probability minima/maxima are only a coarse check, not array-level parity evidence.

Receipts: `artifacts/fixed_epoch_v1/native_diagnosis_01/{default.log,run_state.json}` and `native_diagnosis_02/{unoptimized.log,run_state.json}`. Their run states record the owned process trees and show no active child after the checks.

## Mechanism and isolated remedy

Installed PyTorch is `2.14.1+cu130`, source commit `5c4886908584029761b579af026dcfb627c84070` (`.venv/Lib/site-packages/torch/version.py:4`). Its `torch/jit/_fuser.py:9-16` implements `optimized_execution` by saving, setting, and restoring the executor optimization flag. The new runner scopes that context around prediction at `scripts/run_fixed_epoch_v1.py:142`, preserving the saved graph and model parameters.

PyTorch's [JIT overview, Fusers section](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/OVERVIEW.md#fusers), retrieved 2026-10-03 through its raw source, explains that runtime type/device/shape information enables fusion after the first invocation, and that compilation can generate combined kernels. This makes deferred optimization or compilation a plausible explanation for a successful first call followed by a stalled second call. This general upstream description is not a trace of our installed C++ runtime and does not establish fusion as the exact root cause.

The adapter's original export checks compare the live eager GPU module with CPU TorchScript export/reload (`scripts/fixed_epoch_adapter_v1.py:185,383`). Their success did not verify loaded GPU TorchScript execution. The corrected smoke now checks that missing path. No optimizer, schedule, precision mode, training recipe, saved graph bytes, or historical loader was changed to obtain the workaround.

## Corrected smoke and precision review

`artifacts/fixed_epoch_v1/smoke_cuda_02/verification.json` reports success at `2026-10-03T01:39:32.034970+00:00`, taking 41.453 seconds after setup. It uses 512 generated rows, 384 fitting rows, 128 monitor rows, 52 transformed features including 22 categorical features, eight ensemble members, and the full `[512,256,128]` architecture. It completes inner C H16, outer A H4, and outer C H16, with five exported endpoints in total.

All five endpoint tests independently rebuild raw features, reload the categorical transform and native model, and change prediction chunking from 37 to 17 rows. A separate CPU reload uses chunks of 23 rows. Maximum absolute errors across the five endpoints are:

- GPU reload and changed chunking: `2.980232238769531e-7`.
- CPU versus GPU reload: `5.364418029785156e-7`.

Every test passes the preregistered elementwise tolerance `abs(a-b) <= 2e-6 + 1e-5*abs(b)`. Maximum observed scaled CPU/GPU error is `0.09065009086353062`, below the failure threshold of 1. These are numerical tolerance results, not bitwise identity. Outer A/C initialization hashes, training-index hashes, and public-estimator seed mappings agree. The monitor-only category is excluded from the fitted external vocabulary.

The reviewer independently checked the verification receipt's registry hash, all five registered source hashes against current files, all three trajectory hashes, and all ten endpoint graph/metadata hashes. All matched. Verification SHA256: `72aee2ca247e572c9329952f4e3300d0ef65d2c9d75e5e52127f2b0b128e48b2`. Registry SHA256: `c5ed79c4766bf275596c4e4bcac771f110beeb1885e32043114cfdf72c268da5`.

This clears the observed generated-data native-inference blocker for the scoped executor setting. It establishes neither competition quality nor full-data training/inference throughput. Peak recorded process RSS reaches 2,947,309,568 bytes across this multi-trajectory smoke; the fresh-worker policy remains relevant. The preserved failed graph need not be rewritten or retrained. Future real endpoints still require their own hash-bound native reload and parity receipts.
