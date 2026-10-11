"""All-Python executable lessons 01--20, with explicit scope and run identity.

The published synthetic CSV freezes the original book's R-generated cohort.
Python fits it directly; it does not claim to reproduce R's random-number
stream. Bootstrap and new validation simulations have their own Python seeds.
"""
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform

import numpy as np

from .metrics import evaluate, roc_data, calibration_summary

ROOT = Path(os.path.abspath(__file__)).parents[1]
TITLES = ["使用说明与总原则", "把项目、环境与运行记录建起来", "准备数据与合成教学案例",
          "把六要素写成规格表与队列代码", "先把结局、竞争事件与未知标签分清", "变量审查表与防泄漏管线",
          "样本量、参数预算与缺失处理", "回归类模型", "生存与竞争风险：独立 B 分支",
          "机器学习类模型", "调参与嵌套验证", "内部验证", "评价指标与决策曲线", "外部验证与更新",
          "稳健性、亚组、漂移与公平性", "反例实验室", "模型展示、投入临床与监测", "报告与模板包",
          "投稿实操", "AI 辅助的实操规矩"]


def course_budgets(mode):
    if mode not in ("quick", "book"):
        raise ValueError("mode must be quick or book")
    return dict(development=200, conditional=500, curve=200, curve_conditional=500,
                outer=200, inner=50, subgroup=300) if mode == "book" else dict(
                development=20, conditional=40, curve=20, curve_conditional=40,
                outer=10, inner=10, subgroup=40)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value):
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n"


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(value), encoding="utf-8")


def _write_text(lines, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(line) for line in lines)+"\n", encoding="utf-8")


def _write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_plain(row) for row in rows]
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def source_signature(mode):
    """Fingerprint every Python course/tool implementation and frozen source CSV."""
    files = list(sorted((ROOT/"cph").glob("*.py")))+[ROOT/"run.py", ROOT/"examples/reference/main_case.csv"]
    versions = {"Python": platform.python_version()}
    for package in ("numpy", "scipy", "scikit-learn", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    record = dict(format="python-course-v2", mode=mode, budget=course_budgets(mode), versions=versions,
                  source_hashes={path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in files}, resampling_rng="NumPy PCG64; distinct from original R RNG")
    return dict(signature=_digest(record), **record)


def _mode_guard(output, mode):
    path = output/"mode.csv"
    if path.exists():
        rows = _read_csv(path)
        if len(rows) != 1 or rows[0].get("mode") != mode:
            raise ValueError("Output directory belongs to another/invalid mode; use separate directories")


def _status(chapter, output, mode, state, signature, reason=""):
    here = output/"chapters"/("ch%02d" % chapter)
    here.mkdir(parents=True, exist_ok=True)
    row = dict(chapter=chapter, title=TITLES[chapter-1], mode=mode, status=state,
               output=here.relative_to(output).as_posix(), artifacts=len(list(here.glob("*"))),
               source_signature=signature, scope=reason or "synthetic executable exercise; read chapter limitations")
    _write_csv([row], here/"status.csv")
    _write_json(row, here/"status.json")
    failure = here/"failure.txt"
    if state == "FAIL":
        _write_text([reason], failure)
    elif failure.exists():
        failure.unlink()
    manifest = _read_csv(output/"manifest.csv") if (output/"manifest.csv").exists() else []
    manifest = [item for item in manifest if int(item["chapter"]) != chapter]+[row]
    _write_csv(sorted(manifest, key=lambda item: int(item["chapter"])), output/"manifest.csv")
    return row


def _invalidate_prior(output, current):
    path = output/"manifest.csv"
    if not path.exists():
        return
    rows = _read_csv(path)
    for row in rows:
        if row.get("source_signature") != current and row.get("status") in ("PASS", "PASS_EXECUTED", "RUNNING"):
            row.update(status="STALE", scope="Source/data/dependency signature changed; rerun this chapter")
            here = output/"chapters"/("ch%02d" % int(row["chapter"]))
            _write_csv([row], here/"status.csv")
            _write_json(row, here/"status.json")
    _write_csv(rows, path)


def _context(output, mode, signature):
    from . import course_core as core
    rows = core.load_case()
    model = core.fit_case(rows)
    p = np.asarray(core.predict_case(model, rows), dtype=float)
    core.export_case(rows, model, output)
    _write_csv([dict(file=name, sha256=hashlib.sha256((ROOT/"cph"/name).read_bytes()).hexdigest())
                for name in ("course_core.py", "course_validation.py", "course.py", "metrics.py",
                             "preprocessing.py", "validation.py", "predictions.py", "baseline.py")],
               output/"results/context_sources.csv")
    y = np.asarray([int(row["y"]) for row in rows])
    metrics = _performance(rows, p)
    _write_csv([dict(metric=name, estimate=value) for name, value in metrics.items()], output/"results/apparent_metrics.csv")
    _write_csv([dict(mode=mode, synthetic=True, n=len(rows), source="frozen original synthetic cohort",
                     source_signature=signature["signature"], **course_budgets(mode))], output/"mode.csv")
    _write_json(signature, output/"source_signature.json")
    identity = core.model_identity(model)
    _write_text([identity], output/"results/prediction_identity.txt")
    return dict(output=output, mode=mode, signature=signature["signature"], rows=rows, model=model,
                p=p, y=y, core=core, identity=identity, budget=course_budgets(mode))


def _performance(rows, p):
    evaluation = evaluate([int(row["y"]) for row in rows], p, [str(row["id"]) for row in rows])
    cal = calibration_summary(evaluation)
    intercept, joint = cal["calibration_in_the_large"], cal["joint_intercept_and_slope"]
    if intercept["status"] != "PASS" or joint["status"] != "PASS" or cal["O_over_E"] is None:
        reasons = [part.get("reason", "unknown calibration failure") for part in (intercept, joint)
                   if part["status"] != "PASS"]
        if cal["O_over_E"] is None:
            reasons.append("O/E has no positive expected-event denominator")
        raise ValueError("Performance/calibration is not estimable; chapter must not pass: " + "; ".join(reasons))
    return dict(C=roc_data(evaluation)["auc"], Brier=cal["Brier"], slope=joint["slope"],
                CITL=intercept["intercept"], OE=cal["O_over_E"])


def _cached(ctx, key, produce):
    path = ctx["output"]/"cache"/(key+".json")
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") == ctx["signature"]:
            if old.get("payload_sha256") != _digest(old["value"]):
                raise ValueError("Cached numerical payload changed; remove/rebuild the affected cache")
            return old["value"]
    value = _plain(produce())
    _write_json(dict(signature=ctx["signature"], payload_sha256=_digest(value), value=value), path)
    return value


def _internal(ctx):
    from . import course_validation as validation
    budget = ctx["budget"]
    result = _cached(ctx, "internal", lambda: dict(
        optimism=validation.bootstrap_optimism(ctx["rows"], ctx["model"], B=budget["development"], seed=20261010),
        conditional=validation.fixed_model_intervals(ctx["y"], ctx["p"], B=budget["conditional"], seed=20261011),
        development=validation.development_intervals(ctx["rows"], B_outer=budget["outer"], B_inner=budget["inner"], seed=20261012)))
    out = ctx["output"]/"results"
    for key, filename in (("optimism", "bootstrap_optimism"), ("conditional", "fixed_model_intervals"),
                          ("development", "development_intervals")):
        value = result[key]
        _write_csv(value["table"], out/(filename+".csv"))
        _write_csv(value.get("replicates", []), out/(filename+"_replicates.csv"))
        _write_json(value, out/(filename+".json"))
        if value.get("status") in ("FAIL", "INVALID", "NOT_ESTIMABLE"):
            raise ValueError(filename+" failed; inspect its full ledger")
    _write_csv(result["optimism"]["dca"], out/"decision_curve.csv")
    return result


def _calibration(ctx):
    from . import course_validation as validation
    result = _cached(ctx, "calibration", lambda: validation.calibration_artifacts(
        ctx["rows"], ctx["model"], B=ctx["budget"]["curve"],
        conditional_B=ctx["budget"]["curve_conditional"], seed=20261013))
    out = ctx["output"]/"results"
    for name in ("curve", "groups", "recipe", "log"):
        value = result[name]
        if isinstance(value, dict):
            value = [dict(item=key, value=json.dumps(_plain(item), ensure_ascii=False, allow_nan=False))
                     for key, item in value.items()]
        _write_csv(value, out/("calibration_"+name+".csv"))
    _write_json(result, out/"calibration.json")
    if not result.get("success"):
        raise ValueError("Calibration resampling failed; inspect calibration_log.csv")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    rows = result["curve"]
    grid = [row["prediction"] for row in rows]
    axes[0].fill_between(grid, [row["apparent_lower"] for row in rows], [row["apparent_upper"] for row in rows],
                         alpha=.2, color="tab:blue", label="Apparent pointwise conditional range")
    axes[0].plot(grid, [row["apparent"] for row in rows], color="tab:blue", label="Apparent smooth curve")
    axes[0].plot(grid, [row["corrected"] for row in rows], color="tab:orange", label="Optimism-corrected point estimate")
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", label="Ideal")
    axes[0].set(ylabel="Observed risk", ylim=(0, 1), title="Calibration: separate uncertainty targets")
    axes[0].legend(fontsize=8)
    axes[1].hist(ctx["p"], bins=np.linspace(0, 1, 21), histtype="step", color="tab:blue")
    axes[1].set(xlim=(0, 1), xlabel="Predicted probability", ylabel="Patients")
    for suffix in ("png", "pdf"):
        fig.savefig(out/("calibration."+suffix), dpi=140)
    plt.close(fig)
    return result


def _negative(name, fun):
    try:
        fun()
    except (ValueError, ArithmeticError) as exc:
        return dict(check=name, status="PASS_EXPECTED_REJECTION", observed_error=str(exc))
    raise ValueError("Negative fixture was accepted: "+name)


def _availability(rows):
    for row in rows:
        values = [row.get(name) for name in ("collection_time", "result_available_time", "decision_time")]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise ValueError("Unknown result availability")
        if not float(values[0]) <= float(values[1]) <= float(values[2]):
            raise ValueError("Input unavailable at decision time")
    return True


def _nested(ctx, tune):
    def compute():
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.base import clone
        from .preprocessing import build_pipeline
        from .predictions import positive_class_probability
        from .validation import _fit
        core, rows, y = ctx["core"], ctx["rows"], ctx["y"]
        design = core.make_design(rows, ctx["model"]["recipe"])[:, 1:]
        outer = StratifiedKFold(3, shuffle=True, random_state=20261031 if not tune else 20261032)
        metrics, roles, tuning, predictions = [], [], [], []
        for fold, (train, test) in enumerate(outer.split(design, y), 1):
            train_rows, test_rows = [rows[index] for index in train], [rows[index] for index in test]
            plain = core.fit_case(train_rows)
            selected = 10.
            if tune:
                inner = list(StratifiedKFold(3, shuffle=True, random_state=20261040+fold).split(design[train], y[train]))
                candidate_rows = []
                for lam in (1., 10., 100.):
                    losses = []
                    for inner_train, inner_test in inner:
                        pipe = build_pipeline(LogisticRegression(C=1/lam, max_iter=3000, tol=1e-9), list(range(18)), [])
                        fitted = _fit(pipe, design[train[inner_train]], y[train[inner_train]])
                        p = positive_class_probability(fitted, design[train[inner_test]])
                        losses.append(float(np.mean((y[train[inner_test]]-p)**2)))
                    candidate_rows.append(dict(outer_fold=fold, lambda_value=lam, inner_mean_Brier=float(np.mean(losses)),
                                               training_n=len(train), test_rows_used_for_tuning=0))
                selected = min(candidate_rows, key=lambda item: item["inner_mean_Brier"])["lambda_value"]
                tuning.extend([dict(row, selected=row["lambda_value"] == selected) for row in candidate_rows])
            pipeline = build_pipeline(LogisticRegression(C=1/selected, max_iter=3000, tol=1e-9), list(range(18)), [])
            ridge = _fit(pipeline, design[train], y[train])
            candidates = {"unpenalized_Logistic": core.predict_case(plain, test_rows),
                          "fold_scaled_ridge_Logistic": positive_class_probability(ridge, design[test])}
            for name, p in candidates.items():
                values = _performance(test_rows, p)
                for metric, value in values.items():
                    metrics.append(dict(fold=fold, candidate=name, lambda_value=selected if "ridge" in name else 0,
                                        train_n=len(train), test_n=len(test), metric=metric, estimate=value,
                                        scope="outer evaluation after inner tuning" if tune else "fixed-candidate outer evaluation"))
                predictions.extend(dict(id=str(row["id"]), y=int(row["y"]), prediction=float(value), fold=fold,
                                        candidate=name) for row, value in zip(test_rows, p))
            selected_train = set(train.tolist())
            roles.extend(dict(id=str(row["id"]), outer_fold=fold, role="TRAIN" if index in selected_train else "TEST",
                              baseline_training_n=len(train)) for index, row in enumerate(rows))
        return dict(metrics=metrics, roles=roles, tuning=tuning, predictions=predictions,
                    ridge_penalty="sum logistic loss + lambda/2 * squared standardized-design slopes; intercept unpenalized",
                    scope="Frozen spline structure, baseline coefficients refit per fold; ridge imputation/scaling fitted inside each train fold")
    return _cached(ctx, "nested_tuning" if tune else "outer_comparison", compute)


def _external(ctx):
    def compute():
        core = ctx["core"]
        rows = core.simulate_validation(1600, seed=20261021, event_rate=.18)
        p = core.predict_case(ctx["model"], rows)
        lp = np.log(p)-np.log1p(-p)
        y = np.array([int(row["y"]) for row in rows])
        train = np.sort(np.random.default_rng(20261022).choice(1600, 800, replace=False))
        test = np.setdiff1d(np.arange(1600), train)
        first = core.checked_logistic_fit(np.ones((800, 1)), y[train], offset=lp[train])["coefficients"]
        second = core.checked_logistic_fit(np.column_stack([np.ones(800), lp[train]]), y[train])["coefficients"]
        logistic = lambda value: np.exp(-np.logaddexp(0., -value))
        rules = dict(frozen=p[test], intercept_update=logistic(lp[test]+first[0]),
                     intercept_slope_update=logistic(second[0]+second[1]*lp[test]))
        metrics = []
        tested = [rows[index] for index in test]
        for name, probability in rules.items():
            metrics.extend(dict(rule=name, evaluation_n=800, metric=metric, estimate=value,
                                scope="independent Python synthetic held-out evaluation; updates fitted on the other 800 patients")
                           for metric, value in _performance(tested, probability).items())
        train_set = set(train.tolist())
        return dict(data=rows, metrics=metrics, assignments=[dict(id=str(row["id"]), role="UPDATE" if index in train_set else "TEST")
                     for index, row in enumerate(rows)], updates=[dict(rule="intercept_update", intercept=float(first[0]), slope=1., training_n=800),
                      dict(rule="intercept_slope_update", intercept=float(second[0]), slope=float(second[1]), training_n=800)],
                     seed=20261021, generator="NumPy PCG64; new independent simulation, not R RNG reproduction")
    return _cached(ctx, "external", compute)


def _competing(ctx):
    """Aalen--Johansen and cause-specific Cox; explicitly not Fine--Gray."""
    from scipy.optimize import minimize
    rows = ctx["rows"]
    ctx["core"].make_state(rows, max(float(row["last_contact"]) for row in rows)+1,
                           outcome="nonfatal_complication")
    times, codes = [], []
    for row in rows:
        origin = float(row["outcome_origin"])
        end = min(30., float(row["last_contact"])-origin)
        candidates = [(end, 0)]
        for field, cause in (("event_time", 1), ("death_time", 2)):
            value = row.get(field)
            if value is not None and math.isfinite(float(value)):
                elapsed = float(value)-origin
                if 0 <= elapsed <= end:
                    candidates.append((elapsed, cause))
        time, code = min(candidates, key=lambda item: (item[0], -item[1]))
        times.append(time)
        codes.append(code)
    times, codes = np.asarray(times), np.asarray(codes)
    survival, cif1, cif2, aj = 1., 0., 0., []
    for time in np.unique(times[codes != 0]):
        risk = int(np.sum(times >= time))
        d1, d2 = int(np.sum((times == time) & (codes == 1))), int(np.sum((times == time) & (codes == 2)))
        cif1 += survival*d1/risk
        cif2 += survival*d2/risk
        survival *= 1-(d1+d2)/risk
        aj.append(dict(time=float(time), n_risk=risk, d1=d1, d2=d2, survival=survival,
                       CIF_nonfatal=cif1, CIF_death=cif2, probability_sum=survival+cif1+cif2))
    X = np.array([[(float(row["age"])-50)/10, (float(row["bmi"])-25)/5, int(row["male"])] for row in rows])
    order = np.argsort(times, kind="mergesort")
    ordered_time, ordered_X = times[order], X[order]
    coefficients, baselines = [], []
    for cause in (1, 2):
        event = codes[order] == cause
        if not np.any(event):
            raise ValueError("Cause-specific Cox requires observed events of each modeled cause")
        event_index = np.searchsorted(ordered_time, ordered_time[event], side="left")
        def objective(beta):
            eta = ordered_X@beta
            if np.max(np.abs(eta)) > 100:
                return 1e100, np.ones(3)*1e50
            weight = np.exp(eta)
            risk = np.cumsum(weight[::-1])[::-1]
            weighted_X = np.cumsum((ordered_X*weight[:, None])[::-1], axis=0)[::-1]
            loss = -float(np.sum(eta[event]-np.log(risk[event_index])))
            gradient = -np.sum(ordered_X[event]-weighted_X[event_index]/risk[event_index, None], axis=0)
            return loss, gradient
        fitted = minimize(objective, np.zeros(3), jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 200})
        gradient = float(np.max(np.abs(objective(fitted.x)[1]))/len(rows))
        if not np.all(np.isfinite(fitted.x)) or gradient > 1e-7:
            raise ValueError("Cause-specific Cox failed finite/gradient checks")
        coefficients.extend(dict(cause=cause, term=term, coefficient=float(beta), gradient=gradient,
                                 method="cause-specific Cox; NOT Fine-Gray")
                            for term, beta in zip(("age_10_year", "bmi_5", "male"), fitted.x))
        risk = np.cumsum(np.exp(ordered_X@fitted.x)[::-1])[::-1]
        for time in np.unique(ordered_time[event]):
            count = int(np.sum(event & (ordered_time == time)))
            baselines.append(dict(cause=cause, time=float(time), baseline_increment=float(count/risk[np.searchsorted(ordered_time, time)]),
                                  reference="age=50,bmi=25,male=0"))
    counts = [dict(code=cause, state=state, n=int(np.sum(codes == cause))) for cause, state in
              [(0, "event_free_or_censored"), (1, "nonfatal_event"), (2, "competing_death")]]
    return dict(counts=counts, coefficients=coefficients, AJ=aj, baseline=baselines,
                scope="actual Python AJ and cause-specific Cox teaching branch; Fine-Gray and branch validation not implemented")


def _chapter(ctx, chapter):
    core, rows, model, out = ctx["core"], ctx["rows"], ctx["model"], ctx["output"]
    here = out/"chapters"/("ch%02d" % chapter)
    here.mkdir(parents=True, exist_ok=True)
    csvout = lambda value, name: _write_csv(value, here/name)
    text = lambda lines, name: _write_text(lines, here/name)
    notes = ["所有资料均为合成教学数据；运行成功不证明临床有效性或人工签审完成。"]
    if chapter == 1:
        _availability(rows)
        csvout([dict(check=name, observed=value, status="PASS") for name, value in
                [("14_inputs", len(core.case_recipe()["predictors"])), ("18_parameters", len(model["beta"])-1),
                 ("two_classes", len(set(ctx["y"].tolist()))), ("chronology", True)]], "orientation_checks.csv")
        notes += ["写定两时点、目标、验证层级与结论范围；人工字段不自动签名。"]
    elif chapter == 2:
        text([platform.python_version(), _json(source_signature(ctx["mode"]))], "session_info.txt")
        csvout([dict(item="source_signature", value=ctx["signature"]), dict(item="model_identity", value=ctx["identity"]),
                dict(item="mode", value=ctx["mode"])], "run_record.csv")
        csvout([dict(package=name, version=value) for name, value in source_signature(ctx["mode"])["versions"].items()], "environment.csv")
    elif chapter == 3:
        csvout([dict(n=len(rows), events=int(sum(ctx["y"])), deaths=sum(int(row["cd_grade"]) == 5 for row in rows),
                     duplicate_ids=len(rows)-len(set(str(row["id"]) for row in rows)), missing_input_cells=0)], "data_summary.csv")
        csvout([dict(field=name, missing=sum(row.get(name) is None for row in rows), role="predictor" if name in core.case_recipe()["predictors"] else "ID/outcome/timing")
                for name in rows[0]], "field_profile.csv")
        notes += ["主线读取冻结的原书合成CSV；没有在Python中假装复现R生成器随机流。真实风险答案列不公开进入输入数据。"]
    elif chapter == 4:
        csvout([dict(element=name, specification=value) for name, value in [
            ("population", "synthetic one-row-per-patient surgical cohort"), ("decision_time", "before anesthesia induction"),
            ("predictors", ";".join(core.case_recipe()["predictors"])), ("outcome", "30-day CD>=III including death"),
            ("origin", "actual surgery start"), ("window", "30 days")]], "six_elements.csv")
        csvout([dict(id=str(row["id"]), eligible=float(row["decision_time"]) < float(row["outcome_origin"]),
                     reason="frozen synthetic cohort; chronology verified") for row in rows], "cohort_flow.csv")
    elif chapter == 5:
        cut = max(float(row["last_contact"]) for row in rows)+1
        A = core.make_state(rows, cut, outcome="composite_CD")
        B = core.make_state(rows, cut, outcome="nonfatal_complication")
        if [int(item["y"]) for item in A] != ctx["y"].tolist():
            raise ValueError("A state labels disagree with source composite outcomes")
        csvout([dict(id=str(row["id"]), A_state=a["state"], A_y=a["y"], B_state=b["state"], B_y=b["y"])
                for row, a, b in zip(rows, A, B)], "labels_A_B.csv")
        fixture = [dict(outcome_origin=0., event_time=event, death_time=death, last_contact=last)
                   for event, death, last in [(5., None, 30.), (None, 3., 30.), (None, None, 30.), (None, None, 12.)]]
        csvout([dict(row, **state) for row, state in zip(fixture, core.make_state(fixture, 30))], "label_fixtures.csv")
        csvout([_negative("event_after_death", lambda: core.make_state([dict(fixture[0], death_time=3.)], 30))], "negative_checks.csv")
    elif chapter == 6:
        _availability(rows)
        csvout([dict(field=name, available_at_decision=True, role="prespecified predictor") for name in core.case_recipe()["predictors"]], "variable_review.csv")
        csvout([_negative("delayed_result", lambda: _availability([dict(rows[0], result_available_time=float(rows[0]["decision_time"])+1)])),
                _negative("unknown_availability", lambda: _availability([dict(rows[0], result_available_time=None)]))], "negative_checks.csv")
    elif chapter == 7:
        csvout([dict(input_variables=14, estimated_parameters=18, events=int(sum(ctx["y"])),
                     descriptive_events_per_parameter=float(sum(ctx["y"])/18), caution="EPP descriptive only")], "parameter_budget.csv")
        rng = np.random.default_rng(20261023)
        train = np.sort(rng.choice(len(rows), 2240, replace=False)); test = np.setdiff1d(np.arange(len(rows)), train)
        hb = np.array([float(row["hb"]) for row in rows]); hb[rng.choice(len(rows), 480, replace=False)] = np.nan
        median = float(np.nanmedian(hb[train])); changed = hb.copy(); changed[test] = 170.
        if median != float(np.nanmedian(changed[train])):
            raise ValueError("Changing held-out inputs changed the training recipe")
        imputed = [dict(row, hb=median if np.isnan(value) else float(value)) for row, value in zip(rows, hb)]
        trained = core.fit_case([imputed[index] for index in train])
        tested = [imputed[index] for index in test]
        csvout([dict(field="hb", training_n=2240, training_median=median, training_missing=int(np.isnan(hb[train]).sum()),
                     test_missing=int(np.isnan(hb[test]).sum()), method="training-only median; NOT MI")], "missing_recipe.csv")
        csvout([dict(metric=metric, estimate=value) for metric, value in _performance(tested, core.predict_case(trained, tested)).items()], "held_out_median_demo.csv")
        train_set = set(train.tolist())
        csvout([dict(id=str(row["id"]), role="TRAIN" if index in train_set else "TEST", hb_missing=bool(np.isnan(hb[index])), hb_after=imputed[index]["hb"])
                for index, row in enumerate(rows)], "missing_assignments.csv")
        csvout([dict(check="change_test_inputs_only", training_median=median, training_median_after_test_change=float(np.nanmedian(changed[train])),
                     full_data_median_before=float(np.nanmedian(hb)), full_data_median_after=float(np.nanmedian(changed)), status="PASS_TRAINING_RECIPE_UNCHANGED")], "boundary_checks.csv")
        notes += ["这不是完整MICE链；样本量三准则另见第30章。"]
    elif chapter == 8:
        csvout([dict(term=term, coefficient=float(beta)) for term, beta in zip(model["terms"], model["beta"])], "coefficients.csv")
        units = dict(core.case_units()); units.update(hb="g/dL", alb="g/dL", crea="mg/dL")
        alt = [dict(row, hb=float(row["hb"])/10, alb=float(row["alb"])/10, crea=float(row["crea"])/88.4) for row in rows[:10]]
        p = core.predict_case_with_units(model, alt, units, ctx["identity"])
        if np.max(np.abs(p-ctx["p"][:10])) > 1e-12:
            raise ValueError("Explicit-unit roundtrip changed predictions")
        csvout([dict(id=str(row["id"]), canonical_prediction=float(a), converted_prediction=float(b), absolute_difference=float(abs(a-b)))
                for row, a, b in zip(alt, ctx["p"][:10], p)], "unit_roundtrip.csv")
        csvout([dict(model["metadata"], identity=ctx["identity"])], "fit_state.csv")
    elif chapter == 9:
        branch = _cached(ctx, "competing", lambda: _competing(ctx))
        csvout(branch["counts"], "branch_counts.csv"); csvout(branch["coefficients"], "branch_coefficients.csv")
        csvout(branch["AJ"], "aalen_johansen.csv"); csvout(branch["baseline"], "cause_specific_baseline.csv")
        _write_json(branch, out/"results/branch_B.json")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        for key, label in [("CIF_nonfatal", "Nonfatal complication"), ("CIF_death", "Competing death")]:
            ax.step([0.]+[row["time"] for row in branch["AJ"]],
                    [0.]+[row[key] for row in branch["AJ"]], where="post", label=label)
        ax.set(xlabel="Days since outcome origin", ylabel="Cumulative incidence",
               xlim=(0, 30), ylim=(0, 1), title="Aalen-Johansen: nonfatal outcome and competing death")
        ax.legend()
        fig.savefig(here/"aalen_johansen.png", dpi=140); fig.savefig(here/"aalen_johansen.pdf"); plt.close(fig)
        notes += ["本Python分支实际运行Aalen–Johansen与两个原因的Cox；不是Fine–Gray。Fine–Gray及B独立验证尚未实现，不能以此替代原书该算法结果。"]
    elif chapter in (10, 11):
        result = _nested(ctx, chapter == 11)
        csvout(result["metrics"], "nested_metrics.csv" if chapter == 11 else "outer_comparison.csv")
        csvout(result["roles"], "outer_roles.csv"); csvout(result["predictions"], "outer_predictions.csv")
        if chapter == 11:
            csvout(result["tuning"], "inner_tuning.csv")
        notes += [result["scope"], result["ridge_penalty"], "不是1SE、森林、boosting或TabPFN；共享最终对象仍是主线无惩罚Logistic。"]
    elif chapter == 12:
        result = _internal(ctx)
        csvout(result["optimism"]["table"], "optimism_corrected_metrics.csv")
        csvout(result["development"]["table"], "development_percentile_ranges.csv")
        csvout(result["conditional"]["table"], "fixed_apparent_ranges.csv")
        notes += ["所有重抽样使用Python PCG64，预算与原书相同不表示随机数末位相同；范围目标见完整JSON ledger。"]
    elif chapter == 13:
        result, cal = _internal(ctx), _calibration(ctx)
        csvout(cal["curve"], "calibration_curve.csv"); csvout(cal["groups"], "risk_groups.csv")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        curves = result["optimism"]["dca"]
        for key, label in [("corrected_NB", "Bootstrap corrected"), ("treat_all", "Treat all"), ("treat_none", "Treat none")]:
            ax.plot([row["threshold"] for row in curves], [row[key] for row in curves], label=label)
        ax.set(xlabel="Prespecified threshold probability", ylabel="Net benefit"); ax.legend()
        fig.savefig(here/"decision_curve.pdf"); fig.savefig(here/"decision_curve.png", dpi=140); plt.close(fig)
        notes += ["蓝色带仅为固定模型表观曲线逐点条件范围；橙色校正曲线只有点估计与MC误差。"]
    elif chapter == 14:
        result = _external(ctx)
        for key, name in [("data", "synthetic_validation_data.csv"), ("metrics", "held_out_update_evaluation.csv"),
                          ("assignments", "validation_roles.csv"), ("updates", "update_coefficients.csv")]:
            csvout(result[key], name)
        notes += [result["generator"], "800人拟合更新，另800人评价；没有真实外部队列，未实现L3/L4更新。"]
    elif chapter == 15:
        from . import course_validation as validation
        result = _cached(ctx, "subgroups", lambda: validation.subgroup_performance(rows, ctx["p"], B=ctx["budget"]["subgroup"], seed=20261024))
        csvout(result["table"], "descriptive_subgroups.csv"); _write_json(result, here/"subgroup_ledger.json")
        perturbed = [dict(row, alb=max(20., float(row["alb"])-2)) for row in rows]
        p = core.predict_case(model, perturbed); difference = p-ctx["p"]
        csvout([dict(id=str(row["id"]), original=float(a), albumin_minus_2=float(b), difference=float(b-a)) for row, a, b in zip(rows, ctx["p"], p)], "input_perturbation.csv")
        csvout([dict(perturbation="albumin minus 2 g/L; truncated at 20", mean_change=float(np.mean(difference)),
                     max_absolute_change=float(np.max(np.abs(difference))), scope="numerical sensitivity, not a causal treatment effect")], "sensitivity_summary.csv")
        notes += ["quick仅40次亚组重复，不足100次有效重复时区间为NA；未实施中心随机效应或真实漂移监测。"]
    elif chapter == 16:
        units = dict(core.case_units()); units["hb"] = "unknown"
        changed = json.loads(_json(model)); changed["recipe"]["age_knots"][1] = 35
        def future_unknown():
            state = core.make_state([dict(outcome_origin=0., event_time=20., death_time=None, last_contact=30.)], 10)[0]
            if state["y"] is None:
                raise ValueError("Unknown early-cut outcome cannot be used as event/non-event")
        csvout([_negative("unknown_unit", lambda: core.predict_case_with_units(model, rows[:1], units, ctx["identity"])),
                _negative("prediction_recipe_changed", lambda: core.predict_case(changed, rows[:1], expected_identity=ctx["identity"])),
                _negative("complete_separation", lambda: core.checked_logistic_fit(np.column_stack([np.ones(100), np.linspace(-2, 2, 100)]), (np.linspace(-2, 2, 100)>0).astype(int))),
                _negative("future_label_at_early_cut", future_unknown)], "negative_checks.csv")
        notes += ["仅这四个实际拒绝用例，不宣称原书十个历史反例全部实跑。"]
    elif chapter == 17:
        from .predictions import FrozenPredictions
        probability = core.predict_case_with_units(model, rows[:20], core.case_units(), ctx["identity"])
        csvout([dict(id=str(row["id"]), probability=float(p), prediction_identity=ctx["identity"]) for row, p in zip(rows[:20], probability)], "example_predictions.csv")
        csvout([dict(field=field, unit=unit) for field, unit in core.case_units().items()], "input_units.csv")
        FrozenPredictions([str(row["id"]) for row in rows], ctx["p"], source="frozen A model predictions", model_identity=ctx["identity"]).save(here/"known_case_predictions.csv")
        text(["# 教学模型登记", "完整规则身份："+ctx["identity"], "状态：SHADOW_TEACHING_ONLY；人工审核：PENDING。",
              "已知病例概率表不是新病例预测模型。身份摘要只检测非恶意一致性，不是可信签名。"], "model_registry.md")
    elif chapter == 18:
        internal, cal, external = _internal(ctx), _calibration(ctx), _external(ctx)
        report = ["# 合成教学案例运行报告", "", "模式："+ctx["mode"], "源码签名："+ctx["signature"],
                  "主线3200人；14输入、18非截距参数；全部信息为合成教学资料。", "", "|指标|表观|校正|乐观MCSE|", "|---|---:|---:|---:|"]
        for row in internal["optimism"]["table"]:
            report.append("|%s|%s|%s|%s|" % tuple(row.get(key) for key in ("metric", "apparent", "corrected", "optimism_MCSE")))
        report += ["", "![平滑校准与风险分布](../../results/calibration.png)", "", "蓝带只描述固定模型表观曲线逐点范围，校正曲线未交付抽样区间。",
                   "独立Python合成验证有800人更新与800人评价，不构成临床外部验证。",
                   "人工审核：PENDING；完整MICE、TabPFN与临床部署未完成。"]
        text(report, "report.md")
        csvout([dict(artifact=name, target=target, excludes=excluded) for name, target, excluded in [
            ("development_intervals.csv", "internally corrected development statistic", "final fixed-model external performance"),
            ("fixed_model_intervals.csv", "fixed-model apparent evaluation", "optimism-corrected evaluation"),
            ("calibration_curve.csv", "apparent smooth curve, pointwise conditional range", "corrected curve interval or simultaneous coverage")]], "interval_targets.csv")
        csvout(external["metrics"], "synthetic_update_evaluation.csv")
    elif chapter == 19:
        csvout([dict(item=item, evidence=evidence, status=status) for item, evidence, status in [
            ("target", "ch04/six_elements.csv", "READER_VERIFY"), ("cohort_flow", "ch04/cohort_flow.csv", "READER_VERIFY"),
            ("complete_rule", "results/model.json", "READER_VERIFY"), ("validation_scope", "ch12+ch14", "READER_VERIFY"),
            ("calibration", "results/calibration.png", "READER_VERIFY"), ("human_review", "reader supplies", "PENDING"),
            ("ethics_and_real_data", "synthetic teaching example", "SYNTHETIC_ONLY")]], "submission_checklist.csv")
        text(["# 投稿材料练习", "读者核对问题、队列、样本量、完整规则、验证、局限及伦理适用性。",
              "期刊政策、作者和机构不由示例推定；本命令只生成本地草稿，没有提交稿件。"], "submission_materials.md")
    elif chapter == 20:
        csvout([dict(action=action, status=status, actor=actor) for action, status, actor in [
            ("generated_synthetic_lesson", "RECORDED", "course CLI"), ("reviewed_real_data_or_clinical_claims", "NOT_PERFORMED", "reader"),
            ("human_signed_off", "PENDING", "human reviewer")]], "ai_audit.csv")
        csvout([dict(role=role, name=None, decision="PENDING", date=None) for role in
                ("clinical_reviewer", "statistical_reviewer", "implementation_reviewer")], "human_review.csv")
        text(["# 人工审核练习", "逐项核对输入、命令、产物、范围与失败原因；真实核对人完成后才能填写签审。",
              "工具运行不证明作者已阅读、采纳或批准；没有调用AI服务或上传资料。"], "audit_instructions.md")
    text(["# 第 %02d 章：%s" % (chapter, TITLES[chapter-1]), "", "运行模式："+ctx["mode"],
          "源码签名："+ctx["signature"], "", "## 操作与边界"]+notes, "lesson.md")
    if source_signature(ctx["mode"])["signature"] != ctx["signature"]:
        raise ValueError("Source/dependency changed during execution; rerun this chapter")
    return _status(chapter, out, ctx["mode"], "PASS_EXECUTED", ctx["signature"])


def run_course(chapter, output_dir, mode="quick"):
    """Run one chapter (1--20) or 'all'; every failure invalidates its old PASS."""
    budget = course_budgets(mode)
    if chapter == "all":
        chapters = list(range(1, 21))
    else:
        if isinstance(chapter, bool) or str(chapter) not in {str(index) for index in range(1, 21)}:
            raise ValueError("Python course chapter must be an integer from 1 to 20 or all")
        chapters = [int(chapter)]
    output = Path(os.path.abspath(output_dir))
    output.mkdir(parents=True, exist_ok=True)
    _mode_guard(output, mode)
    signature = source_signature(mode)
    _invalidate_prior(output, signature["signature"])
    for name in ("data", "results", "cache", "chapters", "protocol"):
        (output/name).mkdir(exist_ok=True)
    _status(chapters[0], output, mode, "RUNNING", signature["signature"])
    try:
        ctx = _context(output, mode, signature)
    except Exception as exc:
        _status(chapters[0], output, mode, "FAIL", signature["signature"], str(exc))
        raise
    results = []
    for number in chapters:
        _status(number, output, mode, "RUNNING", signature["signature"])
        try:
            result = _chapter(ctx, number)
        except Exception as exc:
            _status(number, output, mode, "FAIL", signature["signature"], str(exc))
            raise
        results.append(result)
        print("CHAPTER_%02d_PASS mode=%s output=%s" % (number, mode, output/("chapters/ch%02d" % number)), flush=True)
    return dict(status="PASS_EXECUTED", chapters=results, mode=mode, source_signature=signature["signature"], budget=budget)
