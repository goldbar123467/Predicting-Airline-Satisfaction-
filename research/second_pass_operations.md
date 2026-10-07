# Second-pass local operations

Prepared October 2, 2026. The user authorized this separate local campaign until **13:00 America/Indianapolis, 17:00 UTC**. `SECOND_PASS_PLAN.md` controls selection and the validation policy. The existing release, original supervisor state, `REPORT.md`, and original audit assessment remain unchanged.

## Execution and state boundaries

`configs/second_pass.json` must contain `"campaign": "second_pass"`. The supervisor selects fixed paths from that value:

| Resource | Second-pass path |
|---|---|
| Durable state | `state/second_pass/run_state.json` |
| Cooperative stop request | `state/second_pass/STOP` |
| OS-held exclusive campaign lock | `state/second_pass/lock` |
| Worker logs and lifecycle events | `logs/second_pass/supervisor/` |
| Release implementation | `scripts/second_pass_release.py` |

The original default paths and release commands remain supported. A completed original supervisor state or original `state/STOP` does not control this campaign. The second-pass state cannot silently change campaign on a queue reread. Lock files may remain after exit; the OS lock, not file existence, proves an active coordinator.

Only `runs` in the second-pass configuration are scheduled. The thirty original completed runs may appear as candidate baselines. Existing completions are checked against the exact queued configuration, split checksum and prediction checksums, then skipped. Newly queued experiments require new `v2_` IDs. An unrelated directory under `artifacts/runs` is not automatically scheduled. The empty queue waits until the cutoff and rereads configuration between polls, supporting the configured `wait_for_cutoff: true` behavior and later controlled batches.

One coordinator may hold each campaign lock, but these are deliberately separate locks. They do not serialize an independently launched trainer or a different campaign. **Launch only after the directly started first experiment has exited and the root agent has confirmed the GPU is free.** Do not start another training process while this supervisor is active. An already complete first experiment will be validated and skipped.

## Release dispatch and time budget

All second-pass release commands receive `--config configs/second_pass.json`:

1. After completed candidate membership changes: `second_pass_release.py --phase blend`.
2. After exploration ends: `second_pass_release.py --phase blend --freeze`.
3. Once freezing succeeds: `second_pass_release.py --phase refit`.
4. Always attempt final verification, including after a failed freeze/refit: `second_pass_release.py --phase verify`.

The supervisor never invokes the original `blend.py --finalize` for this campaign. The release script owns isolated artifacts under `artifacts/second_pass` and `SECOND_PASS_REPORT.md`. It must make `--freeze` idempotent, require an existing frozen selection before refit, preserve the original release on failure, and verify the usable second-pass fallback when full refit fails. It must not score or select on the previously evaluated audit. Full-data fitting is permitted only after development-only selection is frozen.

| Boundary | UTC | Local EDT | Behavior |
|---|---|---|---|
| Stop launching experiments | 15:45 | 11:45 | An active run may finish within its hard deadline |
| Stop exploration workers | 16:00 | 12:00 | Abort incomplete work and release only complete candidates |
| Freeze must finish by | 16:15 | 12:15 | Also limited to 600 seconds by default |
| Refit must finish by | 16:40 | 12:40 | Fresh workers; default total refit timeout 1,500 seconds |
| Verification must finish by | 16:50 | 12:50 | Retain ten minutes before the user deadline |
| User delivery deadline | 17:00 | 13:00 | Report verified deliverable or preserved fallback |

Every child begins termination 15 seconds before its applicable hard boundary to reserve cleanup time. The optional `refit_timeout_seconds` controls the second-pass refit limit, default 1,500 and maximum 2,400 seconds. Its reserved duration also moves the freeze hard boundary earlier than 16:40. Changing it requires checking that exploration and the bounded freeze still leave that time. With the current defaults, exploration ending at 16:00 plus a ten-minute freeze leaves thirty minutes until 16:40, covering the 25-minute refit budget and five-minute buffer. This is an operational cap, not a measured runtime guarantee. Root controls selected member reuse and release worker isolation.

After a successful frozen phase, a restarted supervisor resumes refit/verification instead of returning to experiments. A failed final phase remains visible as `needs_attention`; it is not falsely reported complete. Normal final verification runs even if a full refit fails. Once the run is verified, the root should pause the user-authorized heartbeat.

## Process recovery

The supervisor records PID, creation time and exact argument vector for the launcher and observed descendants. It captures aggregate RSS/CPU time and appends stdout/stderr log paths to durable state. Termination verifies those identities and processes descendants deepest first. An unexpected coordinator exit with a surviving recorded worker fails the phase and cleans that worker before state is cleared. If cleanup or identity inspection fails, ownership evidence stays in `active_child` and no next phase starts. Restart recovery must successfully clear those owned processes before scheduling more work.

Create only `state/second_pass/STOP` to request a second-pass stop. The supervisor allows the configured grace interval, then terminates verified owned processes. The root must inspect saved state and logs before removing a stop request and relaunching. Never kill by executable name or remove the campaign lock to bypass an active owner. Completed fold checkpoints remain resumable; incomplete work is never promoted.

Known limit: the periodic ownership snapshots cannot guarantee capture of a worker that is spawned and orphaned entirely between polls. Kernel-enforced process containment was not added in this bounded change. Identity checks and cleanup of captured workers are exercised on actual Windows redirector/worker subprocesses.

## Launch and inspection

The commands below are for the root agent after confirming the direct first run has exited. They were documented, not executed by this agent. `start_overnight.ps1` is not the launcher for this campaign.

```powershell
$campaignRoot = 'C:\Users\thecl\Documents\Predicting Airline Satisfaction'
& "$campaignRoot\.venv\Scripts\python.exe" "$campaignRoot\scripts\supervisor.py" --config "$campaignRoot\configs\second_pass.json" --dry-run
& "$campaignRoot\.venv\Scripts\python.exe" "$campaignRoot\scripts\supervisor.py" --config "$campaignRoot\configs\second_pass.json" --status
if (Test-Path -LiteralPath "$campaignRoot\state\second_pass\STOP") {
    throw 'Review and resolve the second-pass STOP request before launch.'
}
$campaignLogs = Join-Path $campaignRoot 'logs\second_pass\supervisor'
New-Item -ItemType Directory -Path $campaignLogs -Force | Out-Null
$launchLabel = Get-Date -Format 'yyyyMMddTHHmmss'
Start-Process -FilePath "$campaignRoot\.venv\Scripts\python.exe" `
    -ArgumentList @('-u', 'scripts/supervisor.py', '--config', 'configs/second_pass.json') `
    -WorkingDirectory $campaignRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $campaignLogs "$launchLabel.launcher.stdout.log") `
    -RedirectStandardError (Join-Path $campaignLogs "$launchLabel.launcher.stderr.log")
```

After launch inspect `--status`, the recorded supervisor PID/creation time, active worker tree and first progress timestamps. Root owns the existing 15-minute heartbeat update; no duplicate monitor was created here. Monitoring should inspect second-pass state/logs and preserve the original verified release throughout.

## Executed checks

- `python -m unittest discover -s scripts -p 'test_*supervisor*.py' -v`: **13 tests passed**, 3.277 seconds. Nine new synthetic tests cover namespace/state/STOP isolation, independent locks, original default dispatch, second-pass freeze/refit/verify dispatch and timing, queue rereads and waiting, failure behavior, and deadline launch rejection. Four existing recovery regressions include real Windows launcher/coordinator/worker subprocesses.
- `python scripts/supervisor.py --self-test`: passed exclusive locking, atomic state, PID identity rejection, owned descendant termination and resource aggregation.
- `python scripts/supervisor.py --config configs/second_pass.json --dry-run`: passed on the actual 31-entry queue; displayed isolated paths and the correct 17:00 UTC deadline without launching or writing state.

No training, model loading, label scoring, credential access, original release mutation or production process action was performed for these tests. Release-script behavior is an integration dependency owned and tested by the root agent; supervisor tests use synthetic phase results.
