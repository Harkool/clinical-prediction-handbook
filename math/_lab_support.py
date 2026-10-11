"""Shared, dependency-free metadata for the independent mathematical labs."""
import json
import hashlib
import os
from pathlib import Path
import platform
import sys


MODE = os.environ.get("CPH_MODE", "book")
if MODE not in ("quick", "book"):
    raise ValueError("CPH_MODE must be quick or book")
BUDGETS = {}


def begin(identifier):
    BUDGETS.clear()
    Path("result.json").write_text(json.dumps({"id": identifier, "mode": MODE,
        "status": "RUNNING"}) + "\n", encoding="utf-8")
    original_hook = sys.excepthook
    def record_failure(error_type, error, traceback):
        Path("result.json").write_text(json.dumps({"id": identifier, "mode": MODE,
            "status": "FAIL", "error": str(error), "budgets": BUDGETS},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        original_hook(error_type, error, traceback)
    sys.excepthook = record_failure
    print(f"[M{identifier}] mode={MODE}; independent synthetic demonstration; not case A", flush=True)
    if MODE == "quick":
        print("Reduced Monte Carlo budgets: inspect result.json; these are not book-reference estimates.", flush=True)
    return MODE


def budget(name, full, quick):
    value = quick if MODE == "quick" else full
    BUDGETS[name] = {"book": full, "quick": quick, "actual": value}
    return value


def write_result(identifier, checks, values=None):
    if not isinstance(checks, dict) or not checks:
        raise ValueError("Numerical acceptance requires a nonempty mapping of Boolean checks")
    for name, passed in checks.items():
        if not (isinstance(passed, bool) or
                (type(passed).__module__.startswith("numpy") and
                 type(passed).__name__ in ("bool", "bool_"))):
            raise TypeError(f"Acceptance check {name!r} must be Boolean, not {type(passed).__name__}")
    checks = {name: bool(passed) for name, passed in checks.items()}
    result = {
        "id": identifier,
        "mode": MODE,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "independent mathematical demonstration; not the main clinical case",
        "python": platform.python_version(),
        "budgets": BUDGETS,
        "checks": checks,
        "values": values or {},
        "support_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise AssertionError(f"M{identifier}: numerical acceptance check failed; inspect result.json")


def check_logistic_solution(X, y, coefficients, offset=None, sample_weight=None,
                            ridge=0., tolerance=1e-7):
    """Reject invalid or unfinished teaching fits; this is not a separation test.

    Fractional outcomes in [0,1] are allowed for the population-risk experiments.
    The score is normalized by total weight so its scale does not grow with n.
    These simple IRLS demonstrations still assume an identifiable finite optimum.
    """
    import numpy as np
    from scipy.special import expit
    X, y, coefficients = map(lambda value: np.asarray(value, float),
                             (X, y, coefficients))
    if (X.ndim != 2 or y.ndim != 1 or coefficients.ndim != 1 or
            X.shape != (len(y), len(coefficients)) or not len(y) or
            not all(np.all(np.isfinite(value)) for value in (X, y, coefficients)) or
            np.any((y < 0) | (y > 1))):
        raise ValueError("Logistic design/outcomes/coefficients have invalid shape or values")
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
    if (off.shape != y.shape or weight.shape != y.shape or
            not np.all(np.isfinite(off)) or not np.all(np.isfinite(weight)) or
            np.any(weight < 0) or weight.sum() <= 0):
        raise ValueError("Logistic offset/weights must be finite, aligned and have positive total weight")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Ridge must be finite and nonnegative")
    p = expit(X @ coefficients + off)
    score = X.T @ (weight * (y-p)) - ridge * coefficients
    if (not np.all(np.isfinite(p)) or np.any((p <= 0) | (p >= 1)) or
            not np.all(np.isfinite(score)) or np.max(np.abs(score))/weight.sum() > tolerance):
        raise RuntimeError("Logistic fit did not produce finite interior predictions and an accepted score")
    return float(np.max(np.abs(score))/weight.sum())
