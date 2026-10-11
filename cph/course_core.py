"""Pure Python case A: frozen teaching inputs, fixed design and checked ML fit.

The main cohort is loaded from the historical synthetic CSV, rather than
imitating R's RNG in Python. Independent validation simulations use PCG64.
Prediction identities use SHA-256 over the complete Python rule and its source;
they are new identities, not the former R payload's MD5 byte encoding.
"""
import csv
import hashlib
import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


PREDICTORS = ("age", "bmi", "male", "asa", "emergency", "diabetes", "copd",
              "smoker", "chf", "sbp", "hr", "hb", "alb", "crea")
DESIGN_TERMS = ("(Intercept)", "age_linear", "age_rcs1", "age_rcs2", "bmi", "male",
                "asa_II", "asa_III", "asa_IV", "emergency", "diabetes", "copd",
                "smoker", "chf", "sbp", "hr", "hb", "alb", "log_crea")
BINARY = ("male", "emergency", "diabetes", "copd", "smoker", "chf")
SCHEMA = "cph-case-A-python-v1"
ROOT = Path(__file__).absolute().parents[1]


def case_recipe():
    return {"age_knots": [20., 45., 65., 85.], "predictors": list(PREDICTORS),
            "parameters": 18, "version": "A-Python-2026-10-11"}


def case_units():
    return dict(zip(PREDICTORS, ("year", "kg/m2", "0/1", "I-IV", "0/1", "0/1",
                                "0/1", "0/1", "0/1", "mmHg", "beat/min", "g/L", "g/L", "umol/L")))


def case_input_ranges():
    return dict(zip(PREDICTORS, ((18, 90), (16, 45), (0, 1), (1, 4), (0, 1),
                                (0, 1), (0, 1), (0, 1), (0, 1), (75, 200),
                                (40, 140), (70, 180), (20, 52), (30, 400))))


def as_columns(data):
    """Accept records or equal-length column arrays; bootstrap IDs may repeat."""
    if isinstance(data, Mapping):
        columns = {str(name): np.asarray(values) for name, values in data.items()}
        if not columns or any(values.ndim != 1 for values in columns.values()) or len({len(v) for v in columns.values()}) != 1 or not len(next(iter(columns.values()))):
            raise ValueError("Case data requires nonempty, equal-length one-dimensional columns")
        return columns
    records = list(data)
    if not records or any(not isinstance(row, Mapping) for row in records):
        raise ValueError("Case data requires nonempty records or column arrays")
    names = list(records[0])
    if any(set(row) != set(names) for row in records):
        raise ValueError("All case records must have the same fields")
    return {name: np.asarray([row[name] for row in records]) for name in names}


def subset_case(data, indices):
    columns = as_columns(data)
    index = np.asarray(indices)
    if index.ndim != 1 or index.dtype.kind not in "iu" or np.any(index < 0) or np.any(index >= len(next(iter(columns.values())))):
        raise ValueError("Subset indices must be valid zero-based row indices")
    return {name: values[index] for name, values in columns.items()}


def _number(columns, name, allow_missing=False):
    if name not in columns:
        raise ValueError("Missing input: " + name)
    try:
        values = np.asarray(columns[name], dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Nonnumeric input: " + name) from None
    if np.any(np.isinf(values)) or (not allow_missing and not np.all(np.isfinite(values))):
        raise ValueError("Missing/nonfinite input: " + name)
    return values


def _recipe(recipe):
    recipe = case_recipe() if recipe is None else recipe
    if not isinstance(recipe, Mapping) or recipe.get("predictors") != list(PREDICTORS) or recipe.get("parameters") != 18 or not isinstance(recipe.get("version"), str) or not recipe["version"]:
        raise ValueError("Recipe must declare the fixed 14 predictor whitelist and 18 parameters")
    knots = np.asarray(recipe.get("age_knots"), dtype=float)
    if knots.shape != (4,) or not np.all(np.isfinite(knots)) or np.any(np.diff(knots) <= 0):
        raise ValueError("Age RCS requires exactly four ordered finite knots")
    return dict(recipe), knots


def normalise_case_units(data, units):
    canonical = case_units()
    if not isinstance(units, Mapping) or set(units) != set(canonical) or any(not isinstance(value, str) for value in units.values()):
        raise ValueError("Declare exactly one explicit unit for each of the 14 inputs")
    columns = {name: values.copy() for name, values in as_columns(data).items()}
    conversions = {("hb", "g/dL"): 10., ("alb", "g/dL"): 10., ("crea", "mg/dL"): 88.4}
    for name in PREDICTORS:
        values = _number(columns, name)
        if units[name] != canonical[name]:
            if (name, units[name]) not in conversions:
                raise ValueError("Unsupported explicit unit for " + name)
            values = values*conversions[(name, units[name])]
        lower, upper = case_input_ranges()[name]
        # Unit roundtrips at a domain endpoint can differ by a few ulps.
        tolerance = 1e-12*max(abs(lower), abs(upper), 1)
        if np.any(values < lower-tolerance) or np.any(values > upper+tolerance):
            raise ValueError("Outside the synthetic teaching domain: " + name)
        columns[name] = values
    if any(not np.all(np.isin(columns[name], (0, 1))) for name in BINARY) or not np.all(np.isin(columns["asa"], (1, 2, 3, 4))):
        raise ValueError("Invalid binary/ASA level")
    return columns


def rcs_age(age, knots):
    age = np.asarray(age, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if knots.shape != (4,) or not np.all(np.isfinite(knots)) or np.any(np.diff(knots) <= 0) or not np.all(np.isfinite(age)):
        raise ValueError("Invalid age or four-knot RCS specification")
    a, b = knots[-2:]
    positive = lambda value: np.maximum(value, 0)**3
    nonlinear = [(positive(age-k)-positive(age-a)*(b-k)/(b-a)+positive(age-b)*(a-k)/(b-a))/((b-knots[0])**2*10) for k in knots[:-2]]
    return np.column_stack(((age-50)/10, *nonlinear))


def make_design(data, recipe=None):
    """Return the fixed 19 columns; y, IDs, grades and times cannot enter design."""
    _, knots = _recipe(recipe)
    d = normalise_case_units(data, case_units())
    X = np.column_stack((np.ones(len(d["age"])), rcs_age(d["age"], knots),
        (d["bmi"]-25)/5, d["male"], d["asa"] == 2, d["asa"] == 3, d["asa"] == 4,
        d["emergency"], d["diabetes"], d["copd"], d["smoker"], d["chf"],
        (d["sbp"]-120)/20, (d["hr"]-75)/15, (d["hb"]-130)/20,
        (d["alb"]-40)/5, np.log(d["crea"]/80)))
    if X.shape[1] != len(DESIGN_TERMS) or not np.all(np.isfinite(X)):
        raise ValueError("Invalid fixed design")
    return X


def checked_logistic_fit(X, y, offset=None, initial=None):
    """Unpenalized logistic Newton fit, with full-rank and numerical guards.

    These guards reject numerical separation/boundary failures; they are not a
    complete mathematical separation diagnostic. No failed draws are dropped.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1 or not np.all(np.isfinite(X)) or y.shape != (len(X),) or not np.all(np.isin(y, (0, 1))) or len(np.unique(y)) != 2:
        raise ValueError("Logistic fit needs a finite design and two observed binary classes")
    rank = int(np.linalg.matrix_rank(X))
    if rank != X.shape[1]:
        raise ValueError("Logistic design is not full rank")
    offset = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    if offset.shape != y.shape or not np.all(np.isfinite(offset)):
        raise ValueError("Offset needs one finite value per row")
    beta = np.zeros(X.shape[1]) if initial is None else np.asarray(initial, dtype=float).copy()
    if beta.shape != (X.shape[1],) or not np.all(np.isfinite(beta)):
        raise ValueError("Invalid initial coefficients")
    if initial is None and np.all(X[:, 0] == 1):
        beta[0] = np.log(np.mean(y)/(1-np.mean(y)))-np.mean(offset)
    objective = lambda b: float(np.sum(np.logaddexp(0, X@b+offset)-y*(X@b+offset)))
    for iteration in range(1, 101):
        p = expit(X@beta+offset)
        score = X.T@(p-y)
        gradient = float(np.max(np.abs(score))/len(y))
        if gradient <= 1e-12:
            if np.any(p <= 1e-12) or np.any(p >= 1-1e-12) or not np.all(np.isfinite(beta)):
                raise ValueError("Logistic fit reached a separation/probability boundary")
            return {"coefficients": beta, "iterations": iteration, "gradient": gradient,
                    "rank": rank, "converged": True}
        H = X.T@(X*(p*(1-p))[:, None])
        if not np.all(np.isfinite(H)) or np.linalg.matrix_rank(H) != X.shape[1]:
            raise ValueError("Logistic weighted design is singular or numerically separated")
        step = np.linalg.solve(H, score)
        old, rate = objective(beta), 1.
        # Near the optimum, log-likelihood summation can differ by a few ulps
        # while the score still has useful Newton precision. Allow only that
        # floating-point objective noise; retain the strict final score guard.
        rounding_allowance = 16*np.finfo(float).eps*max(1., abs(old))
        for _ in range(40):
            candidate = beta-rate*step
            value = objective(candidate)
            if np.isfinite(value) and value <= old-1e-4*rate*float(score@step)+rounding_allowance:
                beta = candidate
                break
            rate *= .5
        else:
            raise ValueError("Logistic line search failed")
    raise ValueError("Logistic fit did not converge within 100 iterations")


def validate_availability(data):
    columns = as_columns(data)
    collection, available, decision, origin = [_number(columns, name) for name in
        ("collection_time", "result_available_time", "decision_time", "outcome_origin")]
    if np.any(collection > available) or np.any(available > decision) or np.any(decision >= origin):
        raise ValueError("Input unavailable at decision time or invalid decision/origin chronology")
    return True


def _source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def model_identity(model):
    if not isinstance(model, Mapping) or model.get("schema") != SCHEMA or model.get("terms") != list(DESIGN_TERMS):
        raise ValueError("Unknown Python model schema or coefficient order")
    _recipe(model.get("recipe"))
    beta = np.asarray(model.get("beta"), dtype=float)
    if beta.shape != (19,) or not np.all(np.isfinite(beta)):
        raise ValueError("Invalid named model coefficients")
    payload = {"schema": SCHEMA, "terms": list(DESIGN_TERMS), "beta": beta.tolist(),
               "recipe": model["recipe"], "target": model["target"],
               "units": case_units(), "ranges": case_input_ranges(),
               "levels": {"binary": [0, 1], "asa": [1, 2, 3, 4], "asa_reference": 1},
               "conversions": {"hb:g/dL": 10., "alb:g/dL": 10., "crea:mg/dL": 88.4},
               "implementation_sha256": _source_hash()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:"+hashlib.sha256(encoded).hexdigest()


def assert_model_identity(model, expected):
    if not isinstance(expected, str) or len(expected) != 71 or not expected.startswith("sha256:") or any(c not in "0123456789abcdef" for c in expected[7:]):
        raise ValueError("Expected Python identity must declare SHA-256")
    if model_identity(model) != expected:
        raise ValueError("Complete Python prediction-rule identity mismatch")
    return True


def fit_case(data, recipe=None):
    recipe, _ = _recipe(recipe)
    columns = as_columns(data)
    time_fields = {"collection_time", "result_available_time", "decision_time", "outcome_origin"}
    if time_fields.intersection(columns):
        validate_availability(columns)
    X, y = make_design(columns, recipe), _number(columns, "y")
    fit = checked_logistic_fit(X, y)
    model = {"schema": SCHEMA, "recipe": recipe, "terms": list(DESIGN_TERMS),
             "beta": fit["coefficients"].tolist(),
             "target": {"outcome": "30-day CD >= III, including death (CD V)",
                        "decision_time": "pre-anaesthetic assessment complete, before induction",
                        "outcome_origin": "actual start of surgery", "window_days": 30},
             "metadata": {"version": recipe["version"], "n": len(y), "events": int(np.sum(y)),
                          "parameters": 18, "rank": fit["rank"], "gradient": fit["gradient"],
                          "iterations": fit["iterations"], "converged": True,
                          "implementation_sha256": _source_hash()}}
    model["identity"] = model_identity(model)
    return model


def predict_case(model, data, units=None, expected_identity=None):
    expected = model.get("identity") if expected_identity is None else expected_identity
    assert_model_identity(model, expected)
    normalized = normalise_case_units(data, case_units() if units is None else units)
    return expit(make_design(normalized, model["recipe"])@np.asarray(model["beta"], dtype=float))


def predict_case_with_units(model, data, units, expected_identity=None):
    return predict_case(model, data, units, expected_identity)


def make_state(data, data_cut, window_days=30, outcome="composite_CD"):
    """Label only observed events; incomplete follow-up has y=None."""
    if outcome not in ("composite_CD", "nonfatal_complication"):
        raise ValueError("Declare a supported outcome target")
    d = as_columns(data)
    origin, last = _number(d, "outcome_origin"), _number(d, "last_contact")
    event, death = _number(d, "event_time", True), _number(d, "death_time", True)
    if isinstance(window_days, bool) or np.ndim(window_days) != 0 or not np.isfinite(window_days) or window_days <= 0:
        raise ValueError("Window must be one finite positive number of days")
    cut = np.asarray(data_cut, dtype=float)
    if cut.ndim == 0:
        cut = np.repeat(float(cut), len(origin))
    if cut.shape != origin.shape or not np.all(np.isfinite(cut)):
        raise ValueError("Data cut must be finite scalar or exactly one value per record")
    end = origin+float(window_days)
    if not np.all(np.isfinite(end)):
        raise ValueError("Outcome end overflows timestamp scale")
    if "decision_time" in d and np.any(_number(d, "decision_time") >= origin):
        raise ValueError("Decision must precede outcome origin")
    if np.any(last < origin) or np.any(np.isfinite(event) & (event < origin)) or np.any(np.isfinite(death) & (death < origin)) or np.any(np.isfinite(event) & np.isfinite(death) & (event > death)):
        raise ValueError("Impossible event/death/follow-up chronology")
    if outcome == "nonfatal_complication" and np.any(np.isfinite(event) & np.isfinite(death) & (event == death)):
        raise ValueError("Tied event/death times require adjudication before competing-risk labels")
    observed = np.minimum(np.minimum(last, cut), end)
    rows = []
    for i in range(len(origin)):
        known_event = np.isfinite(event[i]) and event[i] <= observed[i]
        known_death = np.isfinite(death[i]) and death[i] <= observed[i]
        if known_event or outcome == "composite_CD" and known_death:
            state, y = "event", 1
        elif outcome == "nonfatal_complication" and known_death:
            state, y = "competing", 0
        elif observed[i] >= end[i]:
            state, y = "no_event_complete", 0
        elif cut[i] < end[i] and cut[i] <= last[i]:
            state, y = "incomplete_admin", None
        else:
            state, y = "incomplete_lost", None
        rows.append({"state": state, "y": y, "observed_end": float(observed[i])})
    return rows


def load_case(path=None):
    path = ROOT/"examples/reference/main_case.csv" if path is None else Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Case CSV contains missing/duplicate columns")
        rows = []
        for record in reader:
            row = {}
            for name, value in record.items():
                if name == "id":
                    row[name] = value
                elif value in ("", "NA", "NaN"):
                    row[name] = None
                else:
                    row[name] = float(value)
            rows.append(row)
    columns = as_columns(rows)
    make_design(columns)
    validate_availability(columns)
    if "id" not in columns or any(not str(value).strip() for value in columns["id"]) or len(set(columns["id"])) != len(rows):
        raise ValueError("Frozen source cohort must have distinct observed patient IDs")
    labels = make_state(columns, float(np.max(_number(columns, "last_contact")))+1)
    if not np.array_equal([row["y"] for row in labels], _number(columns, "y")):
        raise ValueError("Stored case labels disagree with observed composite outcome")
    return rows


def simulate_validation(n=1600, seed=20261021, event_rate=.18):
    """Independent PCG64 synthetic cohort; seed does not reproduce R's RNG."""
    if not isinstance(n, Integral) or isinstance(n, bool) or n < 2 or not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0 or not np.isfinite(event_rate) or not 0 < event_rate < 1:
        raise ValueError("Simulation needs n>=2, a nonnegative integer seed and event_rate in (0,1)")
    rng = np.random.default_rng(int(seed))
    clip_normal = lambda mean, sd, low, high: np.clip(rng.normal(mean, sd, n), low, high)
    d = {"id": np.array(["V%05d" % (i+1) for i in range(n)]),
         "center": rng.choice(np.arange(1, 7), n, p=[.25, .23, .20, .17, .12, .03]),
         "age": np.clip(np.rint(rng.normal(58, 15, n)), 18, 90),
         "bmi": clip_normal(26, 4.5, 16, 45), "male": rng.binomial(1, .55, n),
         "asa": rng.choice([1, 2, 3, 4], n, p=[.16, .47, .30, .07])}
    for name, probability in (("emergency", .16), ("diabetes", .21), ("copd", .11), ("smoker", .25), ("chf", .07)):
        d[name] = rng.binomial(1, probability, n)
    d.update(sbp=clip_normal(125, 18, 75, 200), hr=clip_normal(77, 13, 40, 140),
             hb=clip_normal(129, 18, 70, 180), alb=clip_normal(39, 5, 20, 52),
             crea=np.clip(np.exp(rng.normal(np.log(80), .4, n)), 30, 400))
    beta = np.array([0, .30, .30, -.50, .15, .12, .30, .70, 1.15, .65, .35, .35, .12, .60, -.15, .20, -.25, -.45, .40])
    linear = make_design(d)[:, 1:]@beta[1:]
    intercept = brentq(lambda b: np.mean(expit(b+linear))-event_rate, -30, 30)
    d["y"] = rng.binomial(1, expit(intercept+linear))
    grade = rng.choice([0, 1, 2], n, p=[.45, .35, .20])
    grade[d["y"] == 1] = rng.choice([3, 4, 5], np.sum(d["y"]), p=[.65, .23, .12])
    d["cd_grade"] = grade
    d["decision_time"] = 20000+rng.uniform(0, 365, n)
    d["outcome_origin"] = d["decision_time"]+rng.uniform(2/24, 1, n)
    d["collection_time"] = d["decision_time"]-rng.uniform(2/24, 3, n)
    d["result_available_time"] = d["decision_time"]-rng.uniform(.1/24, 1/24, n)
    d["event_time"] = np.where(np.isin(grade, [3, 4]), d["outcome_origin"]+rng.uniform(.5, 29.5, n), np.nan)
    d["death_time"] = np.where(grade == 5, d["outcome_origin"]+rng.uniform(.5, 29.5, n), np.nan)
    d["last_contact"] = d["outcome_origin"]+30
    return [{name: (None if isinstance(value[i], (float, np.floating)) and np.isnan(value[i]) else value[i].item() if isinstance(value[i], np.generic) else value[i]) for name, value in d.items()} for i in range(n)]


def write_table(rows, path):
    """Write records/column arrays with full float precision and explicit NA."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, Mapping):
        columns = as_columns(rows)
        rows = [{name: values[i] for name, values in columns.items()} for i in range(len(next(iter(columns.values()))))]
    else:
        rows = list(rows)
    if not rows:
        raise ValueError("CSV export requires at least one row")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "NA" if value is None or isinstance(value, (float, np.floating)) and np.isnan(value) else format(float(value), ".17g") if isinstance(value, (float, np.floating)) else value for name, value in row.items()})
    return path


def write_model(model, path):
    assert_model_identity(model, model["identity"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return path


def export_case(data, model, output_dir):
    output = Path(output_dir)
    columns = as_columns(data)
    write_table(columns, output/"data/main_case.csv")
    write_model(model, output/"results/model.json")
    write_table([{"term": name, "coefficient": value} for name, value in zip(model["terms"], model["beta"])], output/"results/model_coefficients.csv")
    p = predict_case(model, columns)
    write_table([{"id": str(columns["id"][i]), "y": int(columns["y"][i]), "prediction": p[i]} for i in range(len(p))], output/"results/predictions.csv")
    (output/"results/prediction_identity.txt").write_text(model["identity"]+"\n", encoding="utf-8")
    return {"n": len(p), "identity": model["identity"], "scope": "frozen synthetic teaching cohort; Python SHA256 prediction rule"}
