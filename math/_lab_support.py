"""Shared, dependency-free metadata for the independent mathematical labs."""
import json
import os
from pathlib import Path
import platform


MODE = os.environ.get("CPH_MODE", "book")
if MODE not in ("quick", "book"):
    raise ValueError("CPH_MODE must be quick or book")
BUDGETS = {}


def begin(identifier):
    print(f"[M{identifier}] mode={MODE}; independent synthetic demonstration; not case A", flush=True)
    if MODE == "quick":
        print("Reduced Monte Carlo budgets: inspect result.json; these are not book-reference estimates.", flush=True)
    return MODE


def budget(name, full, quick):
    value = quick if MODE == "quick" else full
    BUDGETS[name] = {"book": full, "quick": quick, "actual": value}
    return value


def write_result(identifier, checks, values=None):
    checks = {name: bool(passed) for name, passed in checks.items()}
    result = {
        "id": identifier,
        "mode": MODE,
        "scope": "independent mathematical demonstration; not the main clinical case",
        "python": platform.python_version(),
        "budgets": BUDGETS,
        "checks": checks,
        "values": values or {},
    }
    Path("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise AssertionError(f"M{identifier}: numerical acceptance check failed; inspect result.json")
