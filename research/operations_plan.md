# Overnight local operations plan

Inspection: 2026-10-01 22:09 EDT / 2026-10-02 02:09 UTC. This note records observed host facts and a proposed execution contract. It does not claim that monitoring or training has been launched.

## Objective and deadline

Have a validated blended predictor, out-of-fold evaluation, reproducible configuration, saved model artifacts, and a submission-format CSV ready before 2026-10-02 07:00 America/Indianapolis, equivalent to 11:00 UTC. The user wrote EST; using local Eastern daylight time delivers an hour before literal fixed-offset EST. Reserve the last 45 minutes for ensemble selection, validation, and report generation. Maintain an already valid fallback submission as soon as the first model completes.

## Observed resources

- Windows 11 Pro 10.0.26200 on a desktop motherboard, Gigabyte B650 AORUS ELITE AX.
- AMD Ryzen 5 7600X: 6 physical cores, 12 logical processors.
- 15.11 GiB usable system RAM; 6.46 GiB free during initial inspection. Multiple ordinary desktop processes consume the remainder. Do not kill unrelated processes.
- NVIDIA GeForce RTX 5070: 12,227 MiB total VRAM, 10,685 MiB free, driver 616.56. These values were read with `nvidia-smi`; framework compatibility and successful kernels still require a smoke run.
- C: has about 128.8 GiB free. No disk-space constraint expected for tabular models, but check available space before launching subsequent jobs.
- `uv` is installed at `C:\Users\thecl\AppData\Local\hermes\bin\uv.exe`. Managed Python 3.12.14, 3.13.15, and 3.14.7 are installed. PATH `python.exe` points to a Windows Store alias; always use the workspace interpreter's absolute path. Python 3.12 is the conservative compatibility choice until package installation confirms otherwise.
- Balanced power plan: AC sleep and hibernate timers are both zero (disabled); DC sleep is 600 seconds. No power configuration change is necessary for this desktop's current AC configuration. This does not protect against a reboot, outage, user-initiated sleep, or application shutdown.

## Process and state contract

Use one detached supervisor, one active training child, and a durable queue under this workspace. Start the explicit environment interpreter through PowerShell `Start-Process -WindowStyle Hidden`, with an explicit working directory and distinct stdout/stderr log paths. The supervisor should use argument lists for subprocesses and an absolute script path rather than a shell command string. The supervisor remains responsible for its child even when the initiating shell exits.

The supervisor's durable state should record schema version, UTC timestamps, run identifier, supervisor PID and creation time, exact child command, child PID and creation time, job/config identifiers, deadline, attempts, latest progress, completion artifacts, and exit status. Write state and completion files to a temporary sibling then atomically replace their targets. Keep append-only lifecycle events in JSONL and per-attempt stdout/stderr files. Use a real exclusive lock held for the lifetime of the supervisor, such as a one-byte Windows `msvcrt.locking` lock, rather than assuming a PID file excludes duplicates.

Only terminate processes created by this supervisor and whose recorded PID, creation time, and command match. Never stop processes by executable name, broad wildcard, or all Python/GPU processes. A second supervisor must fail closed when the lock is held. A monitor must not start a worker directly while the supervisor remains active.

Each model family/config/seed/fold gets a deterministic artifact directory. A fold is complete only after the model loads, predictions are finite and correctly shaped, and an atomic completion manifest has been written. Recovery skips completed folds and preserves failed attempts for diagnosis. Native model snapshots can improve restart granularity where supported, but a completed fold is the minimum reliable checkpoint. A manifest must describe whether recovery is an exact continuation or a fresh fold fit.

## Resource and scheduling policy

Begin with a single training child and 8 CPU threads. Avoid simultaneous GPU models and independent fits that each copy the full training dataframe. Prefer float32 numerical arrays and categorical dtypes; materialize only the current fold's working data where practical. Measure process RSS, available RAM, and fit durations during the bounded smoke test before increasing threads or candidate size. A proposed system RAM reserve is 1.5 GiB. If available RAM remains below 1.0 GiB for two successive samples and a worker is growing, request graceful stop and retry a smaller configuration; do not kill other applications. If psutil is not installed, a standard-library Windows memory query or PowerShell read-only probe can collect the same observation.

1. Immediately run the smallest viable end-to-end baseline, including saved predictions and submission validation.
2. Use initial complete fold timing to estimate the remaining cost of each candidate. Account for fit plus prediction and saving. Do not begin work whose conservative projected finish exceeds the exploration cutoff.
3. Prioritize two independent competitive model families on the identical split contract before broad tuning. Expand only when smoke timings and measured metric justify it.
4. At 09:30 UTC (05:30 EDT), stop launching exploration. Bound any active training at 10:00 UTC (06:00 EDT); leave incomplete candidates out of final blending.
5. Freeze the candidate set and score the audit after exploration stops. Refit selected members on all training rows by 10:40 UTC, retaining the evaluated fold ensemble as fallback.
6. By 10:50 UTC, validate saved inference and submission outputs and write a concise evidence report. Keep the remaining ten minutes for recovery.

## Fifteen-minute agent monitor

The requested monitor is a recurring Codex heartbeat configured by the main agent, not this note. Every 15 minutes it should read the operations plan, experiment manifest, supervisor state, recent log tail, per-fold completion records, CPU/GPU usage, RAM, and disk. It may inspect failures and implement narrow repairs, but it must verify exclusive ownership before stop/restart actions. It should remain quiet while progress is healthy and notify only on meaningful improvement/completion, failure requiring attention, or a deadline risk.

The supervisor's own lightweight process polling can be more frequent (for example every 30-60 seconds) so crashes and deadline enforcement do not depend entirely on the next agent wakeup. Agent heartbeat execution depends on the desktop host being available. A detached local supervisor and durable artifacts provide continuity if a chat turn ends; they cannot guarantee continuity through system shutdown.

Recovery rules:

- Exited nonzero: retain full logs; identify whether the cause is configuration, library incompatibility, allocation failure, invalid data, or software error. Retry once after a concrete repair. Bound attempts per job at two; avoid repeated identical restarts.
- Out-of-memory: reduce representation size, tree depth/bins, category expansion, or batch size as appropriate to the actual model. Record the change as a new configuration identity. CPU fallback for a GPU compatibility error also receives a new identity.
- Missing progress: a silent log alone does not establish a hang. Check process CPU time and GPU utilization across samples plus native training logs. Treat no CPU/GPU activity and no artifact/log movement for 30 minutes as suspicious; validate before stopping. Use model-specific maximum runtimes based on observed fold duration.
- Cooperative stop: create the supervisor's stop request or send the supported signal, allowing the runner to finish/save at a known safe boundary. After a bounded grace period, terminate only the owned process. Avoid promoting partial predictions.
- Deadline: finalize the best already complete candidate set. If no compatible multi-model set is complete, preserve the valid single-model fallback and state the limitation rather than writing an invalid blend.

## Acceptance checks

Before unattended launch, exercise the actual smoke command, artifact save/load/predict path, detached process liveness after parent shell exit, lock rejection for a second supervisor, one deliberate child failure, and recovery without rerunning a completed unit. Testing process ownership must use dedicated dummy children, never unrelated processes.

For final artifacts, verify identical OOF row identities and fold assignments across blend candidates; use the competition's actual metric; fit blend weights only with the documented development data; validate prediction range, finite values, row count/order, and sample-submission columns. Record dataset hashes, source hash/commit, Python/package versions, hyperparameters, seed, hardware, fold scores, fit times, and candidate weights. Test saved model loading and prediction on a small raw-input slice. Hash the final CSV and model manifests. Preserve the last known valid output until a replacement has passed these checks.

No automatic Kaggle submission is implied by this operations plan. Local training and creation of reviewable submission files are authorized; submission behavior is determined by the user's task and the main agent's competition-rule review.
