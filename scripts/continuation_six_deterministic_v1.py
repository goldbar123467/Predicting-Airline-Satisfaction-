"""Strict process-start execution policy for the six registered continuation studies.

Copy these bytes to an external PYTHONPATH directory as sitecustomize.py. Only
the two exact registered payload entry scripts are affected. No model recipe,
adapter, prefix gate, seed or source file is changed. Ordinary imports are inert.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

TARGETS = {"run_continuation_six_v1.py", "smoke_continuation_six_v1.py"}
WORKSPACE_CONFIG = ":4096:8"
EXIT_CODE = 86


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _launch(argv: list[str], root: Path, target_name: str, stage: str) -> dict:
    """Parse only the frozen commands; never retain arbitrary command-line text."""
    if re.fullmatch(r"fixed_epoch_continuation_0[1-6]", stage) is None:
        raise ValueError("Unregistered startup stage")
    flags, values, index = set(), {}, 1
    pair_keys = {"--campaign", "--fold", "--phase", "--trajectory", "--device", "--output"}
    while index < len(argv):
        key = argv[index]
        if key in {"--execute", "--worker"}:
            if key in flags:
                raise ValueError("Repeated registered launch flag")
            flags.add(key)
            index += 1
        elif key in pair_keys and key not in values and index + 1 < len(argv):
            values[key] = argv[index + 1]
            index += 2
        else:
            raise ValueError("Unrecognized deterministic target command")
    if target_name == "smoke_continuation_six_v1.py":
        expected_output = root / "artifacts" / stage / "smoke_cuda"
        if (flags or set(values) != {"--device", "--output"} or values["--device"] != "cuda"
                or Path(values["--output"]).resolve() != expected_output):
            raise ValueError("Smoke command differs from the registered CUDA smoke")
        return {"role": "smoke", "device": "cuda", "output": expected_output.relative_to(root).as_posix()}
    campaign = root / "configs" / (stage + ".json")
    if "--campaign" not in values or Path(values["--campaign"]).resolve() != campaign:
        raise ValueError("Target campaign differs from the registered wrapper")
    if flags == {"--execute"} and set(values) == {"--campaign"}:
        return {"role": "controller", "campaign": campaign.relative_to(root).as_posix()}
    if (flags == {"--worker"} and set(values) == {"--campaign", "--fold", "--phase", "--trajectory"}
            and values["--fold"] in {"0", "1", "2"} and values["--phase"] in {"inner", "outer"}
            and values["--trajectory"] in {"A", "C"}):
        return {"role": "worker", "campaign": campaign.relative_to(root).as_posix(),
                "fold": int(values["--fold"]), "phase": values["--phase"], "trajectory": values["--trajectory"]}
    raise ValueError("Target role differs from the registered controller/worker commands")


def configure(*, argv=None, environ=None, torch_loader=None, source_path=None) -> dict | None:
    """Set strict deterministic execution before CUDA and save one startup receipt.

    Dependency injection supports CPU-only boundary tests. The deployed hook
    uses the real process argv/environment and imports the actual installed torch.
    """
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    if not argv or Path(argv[0]).name not in TARGETS:
        return None
    root = Path(environ["FIXED_EPOCH_DETERMINISTIC_ROOT"])
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("Deterministic workspace must be an absolute non-symlink directory")
    root = root.resolve(strict=True)
    if Path(argv[0]).is_symlink():
        raise ValueError("Target entry must not be a symlink")
    target = Path(argv[0]).resolve(strict=True)
    if target != root / "scripts" / target.name or not target.is_file():
        raise ValueError("Target does not match the registered payload entry path")
    stage = environ["CONTINUATION_STAGE_ID"]
    launch = _launch(argv, root, target.name, stage)
    receipts = Path(environ["FIXED_EPOCH_DETERMINISTIC_RECEIPTS"])
    stage = environ["CONTINUATION_STAGE_ID"]
    expected = root / "artifacts" / stage / "runtime_amendment/processes"
    if not receipts.is_absolute() or receipts.resolve() != expected.resolve():
        raise ValueError("Startup receipts must use the registered campaign path")
    for path in (target, receipts, *receipts.parents):
        if path.is_symlink():
            raise ValueError("Symlink in target or receipt path")
    amendment = environ["FIXED_EPOCH_DETERMINISTIC_AMENDMENT_SHA256"]
    expected_hook = environ["FIXED_EPOCH_DETERMINISTIC_HOOK_SHA256"]
    if not all(re.fullmatch(r"[a-f0-9]{64}", value) for value in (amendment, expected_hook)):
        raise ValueError("Missing deterministic execution identity")
    source = Path(__file__ if source_path is None else source_path).resolve(strict=True)
    if _digest(source) != expected_hook:
        raise ValueError("Startup hook bytes differ from the registered amendment")
    existing = environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (None, WORKSPACE_CONFIG):
        raise ValueError("Conflicting cuBLAS workspace configuration")
    # This assignment precedes the first torch import in a deployed child.
    environ["CUBLAS_WORKSPACE_CONFIG"] = WORKSPACE_CONFIG
    if torch_loader is None:
        import torch
    else:
        torch = torch_loader()
    if torch.cuda.is_initialized():
        raise RuntimeError("Deterministic setup happened after CUDA initialization")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    flags = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
             "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
             "cudnn_benchmark": torch.backends.cudnn.benchmark,
             "cudnn_deterministic": torch.backends.cudnn.deterministic,
             "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "cublas_workspace_config": environ.get("CUBLAS_WORKSPACE_CONFIG"),
             "startup_cuda_initialized": torch.cuda.is_initialized()}
    if flags != {"deterministic_algorithms": True, "deterministic_warn_only": False,
                 "cudnn_benchmark": False, "cudnn_deterministic": True,
                 "cuda_matmul_allow_tf32": False, "cublas_workspace_config": WORKSPACE_CONFIG,
                 "startup_cuda_initialized": False}:
        raise RuntimeError("Deterministic execution flags did not apply exactly")
    value = {"schema_version": 1, "status": "passed", "policy": "strict_deterministic_execution_v1",
             "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
             "launch": launch,
             "target": target.relative_to(root).as_posix(), "target_sha256": _digest(target),
             "hook_sha256": expected_hook, "amendment_sha256": amendment,
             "torch_version": str(torch.__version__), "cuda_build_version": torch.version.cuda,
             "flags": flags, "prefix_gate_unchanged": True, "model_recipe_unchanged": False, "paired_execution_policy": True}
    receipts.mkdir(parents=True, exist_ok=True)
    with (receipts / f"process_{os.getpid()}.json").open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def startup() -> None:
    try:
        configure()
    except BaseException:
        # Python normally suppresses sitecustomize exceptions and continues.
        # A strict execution amendment must instead stop before the target runs.
        try:
            sys.stderr.write("FATAL: fixed-epoch deterministic startup failed; target was not executed.\n")
            sys.stderr.flush()
        finally:
            os._exit(EXIT_CODE)


if __name__ == "sitecustomize":
    startup()
