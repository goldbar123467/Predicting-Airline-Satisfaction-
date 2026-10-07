"""Private Kaggle-only environment probe. Fits synthetic data, never competition labels."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

# Replaced only when packaging a CPU or GPU probe.
EXPECT_GPU = False
RAW_HASHES = {
    "sample_submission.csv": "6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71",
    "test.csv": "ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6",
    "train.csv": "2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36",
}


def main() -> None:
    import numpy as np
    import psutil

    out = Path("/kaggle/working")
    if not out.is_dir():
        raise RuntimeError("This probe must execute on Kaggle, not the local trainer")
    started = time.time()
    report = {
        "schema_version": 1, "expected_gpu": EXPECT_GPU,
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
        "versions": {}, "checks": {},
        "scope": "synthetic fits and raw file hashes only; no real-label fitting or scoring",
    }
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "torch", "xgboost", "lightgbm", "catboost", "pytabkit", "tabm"):
        try:
            report["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][name] = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        report["gpu_inventory"] = result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        report["gpu_inventory"] = []
        report["gpu_inventory_error"] = str(exc)

    def record(name, action):
        begin = time.time()
        try:
            detail = action()
            report["checks"][name] = {"passed": True, "seconds": time.time() - begin, "detail": detail}
        except Exception as exc:
            # Probe each independent library, preserving explicit failures in the report.
            report["checks"][name] = {"passed": False, "seconds": time.time() - begin,
                                      "error": repr(exc), "traceback": traceback.format_exc()}
        (out / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(name, report["checks"][name]["passed"], flush=True)

    def check_raw():
        matched = {}
        for name, expected in RAW_HASHES.items():
            paths = list(Path("/kaggle/input").rglob(name))
            valid = []
            for path in paths:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest == expected:
                    valid.append(str(path))
            if len(valid) != 1:
                raise ValueError(f"Expected one hash-matching {name}, found {len(valid)}")
            matched[name] = {"path": valid[0], "sha256": expected}
        return matched

    rng = np.random.default_rng(20261002)
    x = rng.normal(size=(1024, 12)).astype("float32")
    y = (x[:, 0] + x[:, 1] > 0).astype("int32")
    test = x[:17]

    def check_xgb():
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=4, max_depth=3, n_jobs=2,
                                  tree_method="hist", device="cuda" if EXPECT_GPU else "cpu")
        model.fit(x, y)
        config = json.loads(model.get_booster().save_config())
        device = config["learner"]["generic_param"]["device"]
        if EXPECT_GPU and not device.startswith("cuda"):
            raise RuntimeError(f"XGBoost fell back to {device}")
        model.save_model(out / "synthetic_xgb.json")
        loaded = xgb.XGBClassifier()
        loaded.load_model(out / "synthetic_xgb.json")
        a, b = model.predict_proba(test), loaded.predict_proba(test)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        return {"device": device, "max_reload_difference": float(np.max(np.abs(a-b)))}

    def check_cat():
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=4, depth=3, thread_count=2, verbose=False,
                                   task_type="GPU" if EXPECT_GPU else "CPU", devices="0")
        model.fit(x, y)
        model.save_model(str(out / "synthetic_cat.cbm"))
        loaded = CatBoostClassifier().load_model(str(out / "synthetic_cat.cbm"))
        a, b = model.predict_proba(test), loaded.predict_proba(test)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        return {"task_type": model.get_all_params()["task_type"], "max_reload_difference": float(np.max(np.abs(a-b)))}

    def check_lgb():
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=4, num_leaves=7, n_jobs=2, verbosity=-1)
        model.fit(x, y)
        model.booster_.save_model(str(out / "synthetic_lgb.txt"))
        loaded = lgb.Booster(model_file=str(out / "synthetic_lgb.txt"))
        a, b = model.predict_proba(test)[:, 1], loaded.predict(test)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        return {"device": "cpu", "max_reload_difference": float(np.max(np.abs(a-b)))}

    def check_torch():
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(20261002)
        device = torch.device("cuda:0" if EXPECT_GPU else "cpu")
        model = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        tx, ty = torch.tensor(x, device=device), torch.tensor(y, dtype=torch.float32, device=device)
        model.train()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx).squeeze(-1), ty)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(tx[:17]).cpu()
        torch.save(model.state_dict(), out / "synthetic_torch.pt")
        loaded = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        loaded.load_state_dict(torch.load(out / "synthetic_torch.pt", map_location="cpu", weights_only=True))
        loaded.eval()
        with torch.no_grad():
            actual = loaded(torch.from_numpy(test))
        torch.testing.assert_close(expected, actual, rtol=2e-5, atol=2e-6)
        return {"device": str(device), "cuda_build": torch.version.cuda, "gpu_count": torch.cuda.device_count(),
                "loss": float(loss.detach().cpu()), "max_reload_difference": float((expected-actual).abs().max())}

    for name, action in (("competition_hashes", check_raw), ("xgboost", check_xgb),
                         ("catboost", check_cat), ("lightgbm_cpu", check_lgb), ("torch", check_torch)):
        record(name, action)
    report["elapsed_seconds"] = time.time() - started
    report["all_checks_passed"] = all(c["passed"] for c in report["checks"].values())
    (out / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_checks_passed"]:
        raise RuntimeError("At least one capability probe failed; inspect probe_report.json")


if __name__ == "__main__":
    main()
