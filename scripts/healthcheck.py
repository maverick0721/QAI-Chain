from __future__ import annotations

import compileall
import importlib
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_compile_check() -> bool:
    print("[1/3] compile check...")
    return compileall.compile_dir(
        str(ROOT),
        rx=re.compile(r"(^|/)(\.git|\.venv|\.venv311|__pycache__)(/|$)"),
        quiet=1,
        force=False,
    )


def run_import_check() -> tuple[bool, list[str]]:
    print("[2/3] import smoke check...")
    errors: list[str] = []
    optional_modules = {
        "ai.rl.omnisafe_envs",
        "experiments.run_omnisafe_constrained_baselines",
        "experiments.run_standard_constrained_transfer",
    }

    for dirpath, _, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT)
        if rel_dir.startswith(".venv") or rel_dir.startswith(".git"):
            continue

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            file_path = Path(dirpath) / filename
            rel = file_path.relative_to(ROOT)
            module_name = ".".join(rel.with_suffix("").parts)

            if module_name.endswith(".__init__"):
                continue

            if module_name in optional_modules:
                # These entrypoints require optional OmniSafe/Gymnasium extras.
                continue

            if filename.startswith("test_") or filename.endswith("_test.py"):
                continue

            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover
                errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    return len(errors) == 0, errors


def run_runtime_smoke() -> tuple[bool, str]:
    print("[3/3] runtime smoke check...")
    try:
        import torch
        from ai.models.metrics_encoder import MetricsEncoder
        from quantum.models.qnn import QNN

        torch.manual_seed(123)
        encoder = MetricsEncoder()
        qnn = QNN()

        x = torch.randn(2, 5).float()
        encoded = encoder(x)
        out = qnn(x)

        if encoded.shape != (2, 128):
            return False, f"Unexpected encoder shape: {encoded.shape}"
        if out.shape != (2, 1):
            return False, f"Unexpected QNN output shape: {out.shape}"

        return True, "Runtime checks passed"
    except Exception as exc:  # pragma: no cover
        return False, f"Runtime smoke failed: {type(exc).__name__}: {exc}"


def main() -> int:
    compile_ok = run_compile_check()
    import_ok, import_errors = run_import_check()
    runtime_ok, runtime_msg = run_runtime_smoke()

    if not compile_ok:
        print("FAIL: compile check failed")

    if not import_ok:
        print("FAIL: import check failed")
        for err in import_errors:
            print(f" - {err}")

    if not runtime_ok:
        print(f"FAIL: {runtime_msg}")

    if compile_ok and import_ok and runtime_ok:
        print("HEALTHCHECK OK")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())