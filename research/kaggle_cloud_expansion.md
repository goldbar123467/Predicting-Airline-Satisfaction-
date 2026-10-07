# Kaggle cloud expansion decision, 2026-10-02

Inspected approximately12:12–12:22UTC on October2,2026. Read kaggle-competitions, its research-and-notebooks reference, research-before-build, SECOND_PASS_PLAN.md and SECOND_PASS_HANDOFF.md. Scope was read-only research and static source capture. This agent created no Kaggle dataset/kernel, launched no remote job, accepted no rules, and made no submission or public post. Root may prepare and launch the bounded private probes under the user's newer cloud authorization.

## Recommendation

Use free Kaggle compute as an additional, separately tracked worker. Start with one private CPU environment/native-inference probe and one private T4 GPU synthetic fit/export probe, each capped at600seconds. This establishes actual hardware, image compatibility, private-source access, output retrieval and cancellation/timeout behavior before a costly experiment. The best immediate benefit is extra RAM and independent execution capacity, not a promise that an old T4 trains faster than this machine.

Preserve today's local deadline17:00UTC/13:00EDT, its17 registered candidates and at most one gated confirmation, immutable v1 release, and existing second-pass selection policy. At12:18:44UTC the local supervisor was running normally,14 new candidates completed and `v2_realmlp_cat_te_teacher_emb10` active. This is a timestamped observation, not a continuous status claim. New cloud recipes belong in a separate preregistered ledger and do not automatically enter today's blend. The exposed audit remains closed everywhere, including cloud notebooks.

## Account quota versus platform limits

The authenticated **official `KaggleApi.quota_view()`** returned at12:15:29UTC:

| Resource | Used | Total allowance | Remaining calculated as total minus used |
|---|---:|---:|---:|
| GPU |0h|30h|30h|
| TPU |0h|20h|20h|

Snapshot: `research/cloud_sources/account_quota.json`; root independently confirmed the same amounts. Refresh was returned as the naive string `2026-10-03T00:00:00`. The installed SDK's `DateTimeSerializer` strips a trailing `Z`; do not pretend this displayed value carried an explicit offset. These are account facts at read time, not a reservation of machines. The SDK also exposes `time_reserved`; it was not included in this basic snapshot. Recheck immediately before dispatch if other jobs have since started.

The [official GPU guide](https://www.kaggle.com/docs/efficient-gpu-usage) says the weekly quota is30hours or sometimes higher depending on capacity/demand. That general statement was not used to infer the account's remaining allowance. No numerical weekly CPU-hours quota was established from current official sources. Do not describe CPU as guaranteed unlimited.

| Resource/limit | Primary evidence | Confidence and application |
|---|---|---|
| CPU session |4CPU cores,30GB RAM;12h maximum session |[Notebook technical specifications](https://www.kaggle.com/docs/notebooks), indexed official text read today. Actual instance probe remains authoritative.|
| T4×2 session |4CPU cores,29GB host RAM;12h maximum;20GB saved `/kaggle/working` output |Same official specifications. Extra scratch space is not persisted.|
| T4 device memory |16GB nominal per device |[NVIDIA March2019 T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf). Two devices do not automatically give one model a32GB allocation.|
| T4 quota charging |One hour of T4×2 session uses one hour of GPU quota |Kaggle staff [October19,2022 announcement and reply](https://www.kaggle.com/product-feedback/361104). No claim this historical rule applies to special accelerators.|
| CPU concurrency |5 batch plus5 interactive sessions |Kaggle staff replies in [March13,2024 thread](https://www.kaggle.com/discussions/product-feedback/483684). This is the latest concrete staff statement found, not a current per-account capability response.|
| GPU concurrency |1 interactive plus2 batch sessions |Kaggle staff [August23,2019 announcement](https://www.kaggle.com/general/105509). Old evidence; current account cap unverified. Start one GPU batch and one CPU batch, not a fleet based on this old number.|
| Private dataset storage |200GB per dataset and200GB total private-dataset limit;50 top-level files |[Current dataset specifications](https://www.kaggle.com/docs/datasets). Current account storage usage not read. Our minimal probe does not need a dataset upload.|

The general notebook page still lists P100, but newer primary evidence overrides that accelerator listing: official CLI [PR1192 merged September11,2026](https://github.com/Kaggle/kaggle-cli/pull/1192) marks `NvidiaTeslaP100` retired and replaced by the default `NvidiaTeslaT4`. The same change says client code cannot see per-user/admin/competition accelerator gates. Do not request a retired P100 or treat an accepted request as proof of the actual device.

The [September2026 CLI accelerator list](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) includes `NvidiaTeslaT4`, `NvidiaTeslaA100`, `NvidiaL4`, `NvidiaL4X1`, `NvidiaH100`, `NvidiaRtxPro6000`, `TpuV5E8`, `TpuV6E8`. Some are competition/admin-only. S6E10 access to those enhanced options is **not established**. For example, [ARC2026 rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/upgraded-accelerators) restrict its L4×4 pool to that competition, require internet off, and charge quota at twice the legacy rate. Do not attach an unrelated competition to obtain its hardware.

## Version and API findings

Read local Kaggle2.2.4 `--help`, `kaggle_api_extended.py` and installed `kagglesdk` source. The current online CLI docs advertise `--no-run`, but **this installed `kernels push` has no such flag and a push launches execution**. No dependency upgrade is needed for the proposed probes. Online metadata docs say internet defaults false; the installed push implementation defaults true. Set `is_private`, `enable_gpu`, `enable_tpu` and `enable_internet` explicitly.

Supported local high-level interface:

```python
api.quota_view()
api.kernels_push(folder="prepared-private-probe", timeout="600", acc="NvidiaTeslaT4")
```

`kernels_push` builds `ApiSaveKernelRequest`: `slug`, `new_title`, `text`, `language`, `kernel_type`, `is_private`, `enable_gpu`, `enable_tpu`, `enable_internet`, `competition_data_sources`, `dataset_data_sources`, `kernel_data_sources`, `model_data_sources`, `session_timeout_seconds`, `machine_shape`. Optional `docker_image` and `docker_image_pinning_type` exist; accepted pinning strings are `original` and `latest`. Record the resulting image/environment, rather than inventing a Docker image identifier.

The save response contains `ref`, `url`, `version_number`, `kernel_id`, `error` and `invalid_*_sources` lists. Require no error and empty invalid-source lists, then inspect the mounted competition files and their hashes inside the job. Kernel metadata's `is_private` must be read back after creation. These checks are important because transport success alone is weaker than a correctly attached private run.

SDK responses support `.to_dict(ignore_defaults=False)` and `.to_json(ignore_defaults=False)` for explicit-field serialization; JSON names are camelCase. A final installed-source audit found a material2.2.4 mismatch: `kernels_status`, `kernels_logs` and `kernels_output` parse the optional version suffix but do not assign it to the outgoing request. They therefore select the latest session. For version-bound reads, use the official SDK `ApiGetKernelSessionStatusRequest` or `ApiListKernelSessionOutputRequest`, set `user_name`, `kernel_slug` and `version_label=str(saved_version)`, and invoke the corresponding service method. The returned output list contains file URLs and the log; follow `next_page_token` with the same version field. Alternatively, use a fresh immutable slug for each probe and push only once, making latest unambiguous. The high-level `kernels_output` implementation returns `(outfiles, next_page_token)`, despite its stale status-string docstring.

The installed official kernels service provides `list_kernels`, per-kernel/version `get_kernel_session_status`, logs, output listing/download, and `cancel_kernel_session`. It does **not** expose an available-accelerator list, account-wide active-session list, or per-account concurrency cap. `kernels list --mine` enumerates notebooks, not every active interactive/commit session. No unsupported internal endpoint was probed.

Cancellation is available through official SDK `ApiCancelKernelSessionRequest.kernel_session_id` and `client.kernels.kernels_api_client.cancel_kernel_session(request)`. The installed CLI lacks a cancel subcommand. A kernel ID is not a session ID; never substitute it. Use a verified session ID if one is obtained, otherwise the UI Active Events control and the explicit600-second job timeout. The [open CLI cancellation issue1172](https://github.com/Kaggle/kaggle-cli/issues/1172) explains the command gap. No cancellation call was made in this research.

## Minimal private workflow

An environment probe is a self-contained script plus metadata. It needs no credential inside the notebook, no upload of competition CSVs, and no private data bundle:

```json
{
  "id": "<account>/s6e10-cloud-gpu-probe",
  "title": "S6E10 Cloud GPU Probe",
  "code_file": "probe.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "machine_shape": "NvidiaTeslaT4",
  "competition_sources": ["playground-series-s6e10"],
  "dataset_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

For CPU use another unique slug/title, GPU=false, TPU=false, and omit `machine_shape`. Metadata fields follow the [official schema](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md). Competition attachment requires the account to have accepted that competition's rules; existing local download/submission history supports access, but attachment must still be checked. [Notebook data-source documentation](https://www.kaggle.com/docs/notebooks) also explains that notebook and dataset sharing permissions are separate.

Commands below are a prepared workflow, **not commands executed by this agent**:

```text
kaggle kernels push -p prepared-private-cpu-probe --timeout 600
kaggle kernels push -p prepared-private-gpu-probe --accelerator NvidiaTeslaT4 --timeout 600
kaggle kernels status <account>/<slug>/<returned-version>
kaggle kernels logs <account>/<slug>/<returned-version>
kaggle kernels output <account>/<slug>/<returned-version> -p research/cloud_outputs/<unique-run>
```

Capture returned version IDs before polling or retrieving artifacts; use a unique cloud run ID, never overwrite a local run. The commands above are valid installed CLI syntax, but their version suffix is ignored by2.2.4 as described above; use them only for unique one-push slugs or use the version-bound SDK requests. Logs/output help and implementations were inspected, not remotely executed. Write manifest, timing, environment and probe results under `/kaggle/working`; the saved outputs can be retrieved when the job finishes.

For a later real pipeline test, stage an allowlisted immutable source snapshot, frozen split IDs/hashes, required source-only caches/native teachers, independent test-feature expected predictions, and licenses into a **private** dataset. `kaggle datasets create -p <bundle> --keep-tabular --dir-mode zip` is supported and private by default; do not pass `--public`. Later versions use `kaggle datasets version -p <bundle> -m <message> --keep-tabular`; avoid delete-old-versions. [Official dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md) document these operations. After upload, wait for dataset readiness, verify private status and mounted file hashes, then attach it via `dataset_sources`.

Use `dataset-metadata.json` with title, owner/slug and exactly one license entry. For a bundle with mixed original/data/code terms, preserve attribution/licenses and explain them rather than declaring all contents CC0; the [metadata schema](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md) supports `other` with a description. Competition data is CC BY4.0 and original source CC0 per today's existing competition ledger. Mount competition data directly. Never upload the Kaggle token, local credential file, `.venv`, unrestricted workspace archives, or downloaded third-party notebooks. Publishing/collaborator sharing is outside this private workflow.

## Probe contract and useful next experiments

**CPU probe,600s cap:** record Python/platform/CPU count/available RAM; versions of numpy,pandas,sklearn,xgboost,catboost,lightgbm,torch,pytabkit; list only expected competition filenames and hashes; verify headers without loading labels. Fit generated-data tiny native XGB/LGB/CatBoost models where installed, reload and compare probabilities. Save the environment result even if an optional library is absent. For a subsequent frozen-artifact probe, recompute one selected native model on1024 test-feature rows, then all299,844 test rows if parity succeeds. This tests operational portability without using a scored holdout or consuming new model-selection degrees of freedom.

**GPU probe,600s cap:** record `nvidia-smi`, each device's memory/capability, `torch.version.cuda`, `torch.cuda.get_arch_list()` and actual CUDA availability. Assert the expected T4 class, perform a small synchronized CUDA matrix operation, then generated-data XGB/CatBoost train+native CPU/GPU reload. If the requisite pytabkit dependencies are available, run our native categorical RealMLP synthetic smoke with n_ens8, actual512/256/128 architecture and tiny row count; verify saved predictions for batches1/17/257. No synthetic competition targets or audit scores are needed.

The current [Kaggle Dockerfile snapshot](https://github.com/Kaggle/docker-python/blob/e096cbc5b6ff50ab49130187f8677dedb9b5e9bd/Dockerfile.tmpl) uses a September17,2026 Colab base and references Python3.13 paths, while our local environment is Python3.12. It inherits critical ML libraries from that base; the repository is not proof of the image assigned to a job. Its requirement list names CatBoost and sklearn but does not list pytabkit. Record actual imports before planning installation. Do not blindly install our Windows lockfile into Linux or replace the cloud CUDA stack. If pytabkit1.7.3 is missing, prepare a minimal compatible wheel/dependency plan after the first probe; installing with `--no-deps` is acceptable only after validating required dependencies.

The official [PyTorch support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) supports TuringSM7.5 in CUDA13 builds, while Pascal needs the legacy CUDA12.6 route. A [July28,2026 Kaggle issue1151](https://github.com/Kaggle/kaggle-cli/issues/1151) demonstrated P100/default-image failure and working T4, before P100's September retirement. Native exports, especially TorchScript produced by a newer local Torch, require executed compatibility checks on the actual cloud version. The [September17 TPU issue1197](https://github.com/Kaggle/kaggle-cli/issues/1197) reports push accepting TPU metadata but provisioning the wrong image; this is a user report in the official tracker, not an independently reproduced result. TPU conversion offers little value to our current CUDA/tree pipeline and should be deferred.

After probes, rank work as follows:

1. **Operational check with no selection:** full raw-test inference on a frozen selected member or the frozen ensemble, using the private snapshot. More RAM may relieve local pressure. Require exact input/order/hash contracts and established probability tolerances; an environment mismatch is a diagnostic, not permission to alter the frozen release.
2. **One realistic timing pilot:** one already registered native categorical RealMLP fold, fixed4epochs/n_ens8 and identical preprocessing/splits, cloud-only ID, capped1800s. Measure preparation, fit, inference, serialization and peak memory separately. Do not infer full-fold time from a256-row smoke. If T4×2 is actually available, first use one card; later independent fold processes can each be assigned one card with2CPUthreads, provided host RAM measurements support it. This is our engineering proposal, not automatically enabled multi-GPU training.
3. **Registered seed confirmation only if its existing gate passes:** it is potentially worth offloading the complete three-fold fixed-seed run, but keep a cloud matched control because image/hardware changes confound a pure seed comparison. Root must decide whether it can be included under the current protocol before results; do not silently use a better cloud seed or create a second confirmation.
4. **Separate future robustness campaign:** fully regenerate base fits, transformations and blend fitting inside held-out development partitions to compare the frozen anchor against the best source-feature recipe. This is more informative than many arbitrary GPU trials. Given prior adaptive use of these same development labels, even a newly nested diagnostic is not a pristine independent test. Do not relabel exposed audit data as untouched.

CatBoost is a lower-friction first training portability test because native categorical processing and CBM serialization are already implemented; RealMLP is the more useful capacity/complementarity pilot after library checks. Do not add new TPU implementations, large model families, broad random sweeps or public-score selection merely because30GPUhours are available. Stop cloud work that threatens retrieval/verification before the local deadline; retain the verified local fallback.

## Evidence capture and unresolved facts

Static official docs are saved under `research/cloud_sources`; `official_sources.json` records URLs, SHA256, access times and source commits. Kaggle CLI docs commit `b139309f3d4599d81febbb7c594d60ec25a1f909`; Docker source commit `e096cbc5b6ff50ab49130187f8677dedb9b5e9bd`. Discussion metadata confirmed exact original post dates for CPU/GPU concurrency and T4 changes. Full forum pages/replies were inspected via web; API metadata was read only. A guessed `docs/quota.md` path returned404; quota claims instead rely on the working installed official API and source implementation.

Unresolved until the private probe: actual hardware/image/versions, effective session caps, enhanced accelerator eligibility, account private-storage headroom, exact startup/queue latency, native TorchScript portability, and cross-platform throughput. No remote benchmark or score is claimed by this research. Current-source inspection found enough to proceed with bounded private CPU/T4 probes; more general quota searching is unlikely to resolve these runtime facts.
