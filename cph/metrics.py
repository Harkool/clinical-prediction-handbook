"""Probability evaluation with explicit patient identity and target definitions.

Inputs are observed binary labels and probabilities for the same patients.
No function fits a prediction model or selects a threshold on evaluation data.
DeLong uses paired U-statistic placements, giving tied pairs half credit:
https://pubmed.ncbi.nlm.nih.gov/3203132/
"""
from dataclasses import dataclass
from itertools import combinations
import math
from statistics import NormalDist

import numpy as np


def _vector(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) == 0 or np.iscomplexobj(raw):
        raise ValueError(name + " must be a nonempty one-dimensional real vector")
    try:
        result = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError(name + " must be numeric") from None
    if not np.all(np.isfinite(result)):
        raise ValueError(name + " must contain only finite observed values")
    return result


@dataclass(frozen=True)
class Evaluation:
    """Observed outcomes, fixed predictions and normalized unique patient IDs."""
    y: tuple
    probability: tuple
    ids: tuple

    def __init__(self, y, probability, ids):
        labels = _vector(y, "y")
        probabilities = _vector(probability, "probability")
        raw_ids = np.asarray(ids, dtype=object)
        if raw_ids.ndim != 1 or len(labels) != len(probabilities) or len(labels) != len(raw_ids):
            raise ValueError("y, probability and patient IDs must have equal lengths")
        if not np.all(np.isin(labels, [0, 1])) or len(np.unique(labels)) != 2:
            raise ValueError("y must contain observed 0/1 labels and both classes")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("probability must be in [0,1], not an arbitrary decision score")
        normalized = []
        for value in raw_ids:
            if value is None or (isinstance(value, (float, np.floating)) and not math.isfinite(value)):
                raise ValueError("Patient IDs must be observed")
            name = str(value).strip()
            if not name or name.lower() in {"nan", "na", "none", "null"}:
                raise ValueError("Patient IDs must be observed and nonempty")
            normalized.append(name)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Patient IDs must be unique; do not treat repeated records as independent patients")
        object.__setattr__(self, "y", tuple(int(value) for value in labels))
        object.__setattr__(self, "probability", tuple(float(value) for value in probabilities))
        object.__setattr__(self, "ids", tuple(normalized))

    @property
    def n(self):
        return len(self.y)


def evaluate(y, probability, ids):
    """Create one immutable evaluation; ID matching is never inferred from rows."""
    return Evaluation(y, probability, ids)


def _evaluation(value):
    if not isinstance(value, Evaluation):
        raise ValueError("Supply an Evaluation created with evaluate(y, probability, ids)")
    return value


def align_evaluations(evaluations):
    """Validate same ordered IDs and labels, without silently reordering models."""
    if not isinstance(evaluations, dict) or not evaluations:
        raise ValueError("Supply a nonempty mapping of model names to Evaluation objects")
    first = None
    for name, evaluation in evaluations.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model names must be nonempty strings")
        _evaluation(evaluation)
        if first is not None:
            if evaluation.ids != first.ids:
                raise ValueError("Paired evaluations require identical patient IDs in the same order")
            if evaluation.y != first.y:
                raise ValueError("Paired evaluations require identical labels for each patient")
        first = evaluation
    return evaluations


def _confidence(value):
    if isinstance(value, (bool, complex, np.complexfloating)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("confidence must be a finite number strictly between zero and one")
    return float(value)


def _threshold(value, strict=False):
    if isinstance(value, (bool, complex, np.complexfloating)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError("Declare a finite numeric threshold in advance; evaluation-side threshold selection is unsupported")
    value = float(value)
    if not (0 < value < 1 if strict else 0 <= value <= 1):
        raise ValueError("Threshold must be in " + ("(0,1) for DCA" if strict else "[0,1]"))
    return value


def _wilson(count, total, confidence=.95):
    if total == 0:
        return None
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    probability = count / total
    divisor = 1 + z*z/total
    center = (probability + z*z/(2*total))/divisor
    radius = z*math.sqrt(probability*(1-probability)/total + z*z/(4*total*total))/divisor
    return [max(0., center-radius), min(1., center+radius)]


def threshold_metrics(evaluation, threshold, confidence=.95):
    """Confusion counts and Wilson intervals at an explicitly declared threshold.

    p >= threshold is positive. The interval conditions on a fixed rule and
    independently observed patients; it contains no development uncertainty.
    """
    evaluation = _evaluation(evaluation)
    threshold = _threshold(threshold)
    confidence = _confidence(confidence)
    y = np.asarray(evaluation.y, dtype=int)
    predicted = np.asarray(evaluation.probability) >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    tn = int(np.sum(~predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    result = {"n": evaluation.n, "threshold": threshold, "positive_rule": "p >= threshold",
              "TP": tp, "FP": fp, "TN": tn, "FN": fn, "confidence": confidence,
              "scope": "fixed-rule observed-sample threshold evaluation; no threshold selection"}
    fractions = {"sensitivity": (tp, tp+fn), "specificity": (tn, tn+fp),
                 "PPV": (tp, tp+fp), "NPV": (tn, tn+fn), "accuracy": (tp+tn, evaluation.n)}
    reasons = {}
    for metric, (count, total) in fractions.items():
        result[metric] = count/total if total else None
        result[metric+"_interval"] = _wilson(count, total, confidence)
        if not total:
            reasons[metric] = "zero denominator; not estimable"
    result["F1"] = 2*tp/(2*tp+fp+fn)
    result["not_estimable"] = reasons
    return result


def _rank_curve(evaluation):
    evaluation = _evaluation(evaluation)
    y = np.asarray(evaluation.y, dtype=int)
    p = np.asarray(evaluation.probability)
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(p) != 0), len(p)-1]
    tp = np.r_[0, np.cumsum(y)[ends]].astype(float)
    fp = np.r_[0, (1+ends)-np.cumsum(y)[ends]].astype(float)
    return tp, fp, [None] + p[ends].tolist()


def roc_data(evaluation):
    """ROC at complete tied-score blocks, including the no-positive origin."""
    evaluation = _evaluation(evaluation)
    tp, fp, thresholds = _rank_curve(evaluation)
    positive = sum(evaluation.y)
    tpr, fpr = tp/positive, fp/(evaluation.n-positive)
    auc = float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
    return {"n": evaluation.n, "events": positive, "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": thresholds, "auc": auc,
            "threshold_note": "null threshold denotes above the maximum score (no predicted positives)",
            "scope": "fixed predictions; tied pairs receive half credit"}


def pr_data(evaluation):
    """Precision-recall blocks with AP distinguished from trapezoidal PR area."""
    evaluation = _evaluation(evaluation)
    tp, fp, thresholds = _rank_curve(evaluation)
    recall = tp/sum(evaluation.y)
    precision = np.divide(tp, tp+fp, out=np.ones_like(tp), where=tp+fp != 0)
    average_precision = float(np.sum(np.diff(recall)*precision[1:]))
    area = float(np.sum(np.diff(recall)*(precision[1:]+precision[:-1])/2))
    return {"n": evaluation.n, "event_rate": sum(evaluation.y)/evaluation.n,
            "recall": recall.tolist(), "precision": precision.tolist(), "thresholds": thresholds,
            "average_precision": average_precision, "trapezoidal_pr_area": area,
            "scope": "AP sums recall increments times precision, distinct from trapezoidal PR area"}


def decision_curve(evaluation, thresholds):
    """Binary-outcome net benefit against treat-all and treat-none.

    Only observed complete binary outcomes are supported. Censoring,
    competing risks and causal treatment effects require separate methods.
    """
    evaluation = _evaluation(evaluation)
    values = _vector(thresholds, "thresholds")
    values = [_threshold(value, strict=True) for value in values]
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("DCA thresholds must be unique and strictly increasing")
    rows = []
    prevalence = sum(evaluation.y)/evaluation.n
    for threshold in values:
        table = threshold_metrics(evaluation, threshold)
        odds = threshold/(1-threshold)
        rows.append({"threshold": threshold, "n": evaluation.n, "TP": table["TP"], "FP": table["FP"],
                     "net_benefit": (table["TP"]-table["FP"]*odds)/evaluation.n,
                     "treat_all": prevalence-(1-prevalence)*odds, "treat_none": 0.,
                     "scope": "fixed predictions and prespecified thresholds; complete binary outcomes"})
    return rows


def _placements(evaluation):
    y, p = np.asarray(evaluation.y), np.asarray(evaluation.probability)
    positive, negative = p[y == 1], p[y == 0]
    positive_sorted, negative_sorted = np.sort(positive), np.sort(negative)
    v_positive = (np.searchsorted(negative_sorted, positive, side="left")+
                  np.searchsorted(negative_sorted, positive, side="right"))/(2*len(negative))
    v_negative = 1-(np.searchsorted(positive_sorted, negative, side="left")+
                   np.searchsorted(positive_sorted, negative, side="right"))/(2*len(positive))
    return v_positive, v_negative


def delong_compare(first, second, confidence=.95):
    """Paired, unweighted DeLong AUC difference, normal interval and two-sided p.

    AUC(first) - AUC(second) targets fixed predictions on the same patients.
    It excludes model development and selection uncertainty. Degenerate or
    insufficient class covariance produces an explicit NOT_ESTIMABLE result.
    """
    align_evaluations({"first": first, "second": second})
    confidence = _confidence(confidence)
    vp1, vn1 = _placements(first)
    vp2, vn2 = _placements(second)
    auc = [float(np.mean(vp1)), float(np.mean(vp2))]
    delta = auc[0]-auc[1]
    result = {"status": "NOT_ESTIMABLE", "n": first.n, "events": len(vp1),
              "non_events": len(vn1), "auc_first": auc[0], "auc_second": auc[1],
              "difference": delta, "confidence": confidence, "covariance": None,
              "difference_variance": None, "standard_error": None, "interval": None,
              "p_value": None, "reason": "", "method": "paired DeLong U-statistic placements; ties = 0.5",
              "scope": "fixed-model correlated AUC difference; no development/selection uncertainty"}
    if min(len(vp1), len(vn1)) < 2:
        result["reason"] = "At least two events and two non-events are required to estimate covariance"
        return result
    covariance = np.cov(np.vstack([vp1, vp2]), ddof=1)/len(vp1) + np.cov(np.vstack([vn1, vn2]), ddof=1)/len(vn1)
    variance = float(covariance[0, 0]+covariance[1, 1]-2*covariance[0, 1])
    result["covariance"] = covariance.tolist()
    result["difference_variance"] = max(0., variance) if variance > -1e-12 else variance
    if not math.isfinite(variance) or variance < -1e-12:
        result["reason"] = "Covariance yielded a non-finite or negative contrast variance"
        return result
    if variance <= 1e-15:
        result["reason"] = "Zero contrast variance; normal interval and p-value are not estimable"
        return result
    se = math.sqrt(variance)
    z = NormalDist().inv_cdf((1+confidence)/2)
    result.update(status="PASS", standard_error=se,
                  interval=[delta-z*se, delta+z*se], p_value=math.erfc(abs(delta/se)/math.sqrt(2)),
                  reason="Normal approximation conditional on fixed predictions and independent patients")
    return result


def _logistic_calibration(evaluation, slope=False):
    y, p = np.asarray(evaluation.y, dtype=float), np.asarray(evaluation.probability)
    if np.any((p == 0) | (p == 1)):
        return {"status": "NOT_ESTIMABLE", "intercept": None, "slope": None,
                "reason": "Logit calibration needs probabilities strictly within (0,1); endpoints are not silently clipped"}
    lp = np.log(p)-np.log1p(-p)
    X = np.column_stack([np.ones(evaluation.n), lp]) if slope else np.ones((evaluation.n, 1))
    offset = np.zeros(evaluation.n) if slope else lp
    if np.linalg.matrix_rank(X) != X.shape[1]:
        return {"status": "NOT_ESTIMABLE", "intercept": None, "slope": None,
                "reason": "Rank-deficient calibration design (constant logit predictions)"}
    if slope:
        positive, negative = lp[y == 1], lp[y == 0]
        if np.min(positive) >= np.max(negative) or np.max(positive) <= np.min(negative):
            return {"status": "NOT_ESTIMABLE", "intercept": None, "slope": None,
                    "reason": "Complete or quasi-complete separation in the one-predictor calibration model; finite MLE does not exist"}
    beta = np.array([0., 1.]) if slope else np.array([0.])
    objective = lambda b: float(np.sum(np.logaddexp(0., X@b+offset)-y*(X@b+offset)))
    converged = False
    for iteration in range(100):
        eta = X@beta+offset
        fitted = np.exp(-np.logaddexp(0., -eta))
        gradient = X.T@(fitted-y)
        hessian = X.T@(X*(fitted*(1-fitted))[:, None])
        if np.min(np.linalg.eigvalsh(hessian)) < 1e-10 or np.max(np.abs(beta)) > 50:
            break
        if np.max(np.abs(gradient))/evaluation.n < 1e-11:
            converged = True
            break
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        rate = 1.
        old = objective(beta)
        # Near the solution, the Newton decrease can be smaller than one ULP
        # of the summed loss. Accommodate only floating-point objective noise;
        # retain the independent final score and finite-information checks.
        rounding_allowance = 16*np.finfo(float).eps*max(1., abs(old))
        directional_decrease = float(gradient@step)
        while rate > 2**-20:
            value = objective(beta-rate*step)
            if np.isfinite(value) and value <= old-1e-4*rate*directional_decrease+rounding_allowance:
                break
            rate /= 2
        if rate <= 2**-20:
            break
        beta -= rate*step
    if not converged or not np.all(np.isfinite(beta)):
        return {"status": "NOT_ESTIMABLE", "intercept": None, "slope": None,
                "reason": "Calibration fit failed finite-information/convergence checks; possible separation"}
    return {"status": "PASS", "intercept": float(beta[0]), "slope": float(beta[1]) if slope else 1.,
            "iterations": iteration+1, "reason": "Unpenalized observed-sample logistic calibration"}


def calibration_summary(evaluation, bins=10, confidence=.95):
    """Fixed probability bins, Wilson event-rate ranges and apparent calibration.

    This is a grouped descriptive diagnostic, not a smooth corrected curve.
    Counts and empty bins are retained; nothing is learned for new predictions.
    """
    evaluation = _evaluation(evaluation)
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or not 1 <= bins <= 1000:
        raise ValueError("bins must be a declared integer from 1 to 1000")
    confidence = _confidence(confidence)
    y, p = np.asarray(evaluation.y), np.asarray(evaluation.probability)
    edges = np.linspace(0., 1., bins+1)
    groups = np.minimum(np.searchsorted(edges, p, side="right")-1, bins-1)
    rows = []
    for index in range(bins):
        selected = groups == index
        n, events = int(np.sum(selected)), int(np.sum(y[selected]))
        interval = _wilson(events, n, confidence)
        rows.append({"bin": index+1, "lower": float(edges[index]), "upper": float(edges[index+1]),
                     "upper_inclusive": index == bins-1, "n": n, "events": events,
                     "mean_prediction": float(np.mean(p[selected])) if n else None,
                     "event_rate": events/n if n else None, "event_rate_interval": interval,
                     "status": "OBSERVED" if n else "EMPTY_BIN"})
    expected = float(np.sum(p))
    return {"n": evaluation.n, "events": int(np.sum(y)), "bins": rows, "confidence": confidence,
            "Brier": float(np.mean((y-p)**2)), "mean_prediction": float(np.mean(p)),
            "event_rate": float(np.mean(y)), "O_over_E": float(np.sum(y)/expected) if expected else None,
            "O_over_E_reason": "Observed events divided by sum of predictions" if expected else "Zero expected events; not estimable",
            "calibration_in_the_large": _logistic_calibration(evaluation, slope=False),
            "joint_intercept_and_slope": _logistic_calibration(evaluation, slope=True),
            "scope": "fixed predictions; apparent grouped calibration; Wilson ranges describe bin event rates only"}


def compare_models(evaluations, threshold=.5, thresholds=None, confidence=.95, bins=10):
    """Evaluate aligned models without selecting a winner or optimizing a cutoff."""
    align_evaluations(evaluations)
    threshold = _threshold(threshold)
    confidence = _confidence(confidence)
    thresholds = np.linspace(.01, .5, 50).tolist() if thresholds is None else thresholds
    models = {}
    for name, evaluation in evaluations.items():
        models[name] = {"threshold": threshold_metrics(evaluation, threshold, confidence),
                        "ROC": roc_data(evaluation), "PR": pr_data(evaluation),
                        "DCA": decision_curve(evaluation, thresholds),
                        "calibration": calibration_summary(evaluation, bins, confidence)}
    paired = []
    for first, second in combinations(evaluations, 2):
        paired.append(dict(first=first, second=second,
                           **delong_compare(evaluations[first], evaluations[second], confidence)))
    return {"n": next(iter(evaluations.values())).n, "models": models, "paired_auc_comparisons": paired,
            "scope": "same ordered patients and outcomes; fixed observed-sample evaluation, not development validation",
            "comparison_note": "Pairwise p-values are unadjusted; prespecify comparisons and consider multiplicity"}
