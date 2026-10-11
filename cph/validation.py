"""Patient-aware OOF validation and numerical learning curves.

Replaces Codebase's KFoldValidation, PlotLearningCurve and Plot_Learning_Curve
roles. Models are cloned and fitted per fold. Training-curve scores are
apparent; test curves share a held-out fold. Fold SD is descriptive, not a CI.
"""
from numbers import Integral, Real
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .preprocessing import validate_pipeline_input


def _integer(value, name, minimum=0):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < minimum:
        raise ValueError(name + " must be an integer of at least " + str(minimum))
    return int(value)


def _identifiers(values, n, name):
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or len(raw) != n:
        raise ValueError(name + " must provide one value per row")
    result = []
    for value in raw:
        if value is None or isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError(name + " cannot contain missing/nonfinite values")
        text = str(value).strip()
        if not text or text.lower() in ("na", "nan"):
            raise ValueError(name + " cannot contain missing values")
        result.append(text)
    return np.asarray(result, dtype=object)


def _inputs(pipeline, X, y, ids, groups, n_splits, seed, threshold):
    X = validate_pipeline_input(pipeline, X)
    raw_y = np.asarray(y)
    if raw_y.ndim != 1 or len(raw_y) != len(X) or raw_y.dtype.kind not in "biuf" or not np.all(np.isfinite(raw_y)) or not np.all(np.isin(raw_y, (0, 1))) or len(np.unique(raw_y)) != 2:
        raise ValueError("Validation requires one observed binary label per row and both classes")
    y = raw_y.astype(int)
    ids = _identifiers(ids, len(y), "ids")
    n_splits = _integer(n_splits, "n_splits", 2)
    seed = _integer(seed, "seed", 0)
    if seed > 2**32-1:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    if not isinstance(threshold, Real) or isinstance(threshold, (bool, np.bool_)) or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between 0 and 1")
    if groups is None:
        if len(set(ids)) != len(ids):
            raise ValueError("Repeated patient IDs require groups; record-level CV would leak patients")
        if np.min(np.bincount(y)) < n_splits:
            raise ValueError("Each outcome class needs at least n_splits records")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))
    else:
        groups = _identifiers(groups, len(y), "groups")
        if len(set(groups)) < n_splits:
            raise ValueError("There are fewer independent groups than folds")
        by_patient = {}
        for patient, group in zip(ids, groups):
            if patient in by_patient and by_patient[patient] != group:
                raise ValueError("The same patient ID cannot occur in different groups")
            by_patient[patient] = group
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y, groups))
    seen = np.zeros(len(y), dtype=int)
    for train, test in splits:
        if not len(train) or not len(test) or set(ids[train]).intersection(ids[test]):
            raise ValueError("Training and evaluation must contain disjoint patient IDs")
        if groups is not None and set(groups[train]).intersection(groups[test]):
            raise ValueError("Training and evaluation groups overlap")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("Every training and held-out fold must contain both outcome classes")
        seen[test] += 1
    if not np.all(seen == 1):
        raise ValueError("OOF folds must evaluate each row exactly once")
    return X, y, ids, groups, splits, seed, float(threshold)


def _probability(model, X):
    values = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_)
    if values.shape != (len(X), 2) or not np.array_equal(np.sort(classes), (0, 1)) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)) or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-7):
        raise ValueError("Classifier must return finite normalized binary probabilities")
    return values[:, np.flatnonzero(classes == 1)[0]]


def _fit(pipeline, X, y):
    model = clone(pipeline)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X, y)
    return model


def _metrics(y, p, threshold):
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)) or len(np.unique(y)) != 2:
        raise ValueError("Metrics require valid probabilities and both observed classes")
    decision = p >= threshold
    tp = int(np.sum(decision & (y == 1)))
    fp = int(np.sum(decision & (y == 0)))
    tn = int(np.sum(~decision & (y == 0)))
    fn = int(np.sum(~decision & (y == 1)))
    result = {"n": len(y), "events": int(np.sum(y)), "auc": float(roc_auc_score(y, p)),
              "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
              "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
              "sensitivity": tp/(tp+fn), "specificity": tn/(tn+fp),
              "ppv": tp/(tp+fp) if tp+fp else None,
              "npv": tn/(tn+fn) if tn+fn else None,
              "accuracy": (tp+tn)/len(y)}
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise ValueError("A validation metric is nonfinite")
    return result


def cross_validate_model(pipeline, X, y, *, ids, groups=None, n_splits=5,
                         seed=20261011, threshold=.5, return_models=False):
    """Return OOF predictions, held-out metrics and actual row/ID assignments.

    Repeated IDs are rejected unless their complete records belong to a single
    declared group. No hyperparameter tuning is performed here. Pooled OOF
    performance describes the cross-validation procedure, not an independent
    external validation of a final full-cohort model.
    """
    X, y, ids, groups, splits, seed, threshold = _inputs(pipeline, X, y, ids, groups, n_splits, seed, threshold)
    oof = np.full(len(y), np.nan)
    fold_assignment = np.zeros(len(y), dtype=int)
    metrics, assignments, models = [], [], []
    for fold, (train, test) in enumerate(splits, 1):
        model = _fit(pipeline, X[train], y[train])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            p = _probability(model, X[test])
        oof[test] = p
        fold_assignment[test] = fold
        metrics.append({"fold": fold, "train_n": len(train), **_metrics(y[test], p, threshold)})
        assignments.append({"fold": fold, "train_indices": train.tolist(), "test_indices": test.tolist(),
                            "train_ids": ids[train].tolist(), "test_ids": ids[test].tolist()})
        if return_models:
            models.append(model)
    result = {"oof": [{"row": i, "id": str(ids[i]), "y": int(y[i]),
                       "probability": float(oof[i]), "fold": int(fold_assignment[i])} for i in range(len(y))],
              "fold_metrics": metrics, "pooled_metrics": _metrics(y, oof, threshold),
              "splits": assignments,
              "scope": "OOF procedure evaluation; preprocessing learned within each training fold; no external validation or tuning"}
    if return_models:
        result["models"] = models
    return result


def _sizes(train_sizes):
    values = list(train_sizes)
    if not values or any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) or not np.isfinite(v) for v in values):
        raise ValueError("train_sizes must be a nonempty sequence of finite fractions or integers")
    absolute = all(isinstance(v, Integral) for v in values)
    if absolute:
        if any(v < 2 for v in values):
            raise ValueError("Absolute training sizes must be at least 2")
    elif any(isinstance(v, Integral) or not 0 < v <= 1 for v in values):
        raise ValueError("Use only fractions in (0,1] or only integer training counts")
    if len(set(values)) != len(values) or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("train_sizes must be unique and increasing")
    return values, absolute


def _subset(train, y, groups, count, seed):
    rng = np.random.RandomState(seed)
    if groups is not None:
        # Whole groups preserve patient records; actual counts can exceed targets.
        ordered = np.asarray(sorted(set(groups[train])), dtype=object)
        rng.shuffle(ordered)
        selected = []
        for group in ordered:
            selected.extend(train[groups[train] == group].tolist())
            if len(selected) >= count and len(np.unique(y[selected])) == 2:
                return np.asarray(selected, dtype=int)
        raise ValueError("Training group subset does not contain both classes")
    negative, positive = train[y[train] == 0].copy(), train[y[train] == 1].copy()
    rng.shuffle(negative)
    rng.shuffle(positive)
    n_positive = min(len(positive), max(1, int(round(count*len(positive)/len(train)))))
    n_negative = count-n_positive
    if n_negative < 1:
        n_negative, n_positive = 1, count-1
    if n_negative > len(negative):
        n_negative, n_positive = len(negative), count-len(negative)
    if n_positive > len(positive) or n_positive < 1 or n_negative < 1:
        raise ValueError("Requested training subset cannot contain both classes")
    return np.concatenate((negative[:n_negative], positive[:n_positive]))


def learning_curve(pipeline, X, y, *, ids, groups=None, n_splits=3,
                   train_sizes=(.25, .5, .75, 1.), seed=20261011,
                   threshold=.5, return_models=False):
    """Return numerical curves on fixed outer test folds and nested train subsets.

    Each size/fold clones and fits the entire pipeline. Groups are never split;
    requested and actual sample counts are both recorded. Means and fold SD
    describe the run; SD is not a confidence interval. Training scores are
    apparent, and held-out folds reuse the same records across training sizes.
    """
    X, y, ids, groups, splits, seed, threshold = _inputs(pipeline, X, y, ids, groups, n_splits, seed, threshold)
    sizes, absolute = _sizes(train_sizes)
    rows, assignments, models = [], [], []
    for fold, (train, test) in enumerate(splits, 1):
        previous = set()
        for size in sizes:
            count = int(size) if absolute else int(np.floor(float(size)*len(train)))
            if count < 2 or count > len(train):
                raise ValueError("Training size must allow both classes and not exceed any outer training fold")
            selected = _subset(train, y, groups, count, seed+fold)
            if not previous.issubset(set(selected)):
                raise ValueError("Learning-curve training subsets must be nested")
            previous = set(selected)
            model = _fit(pipeline, X[selected], y[selected])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                for role, index in (("train", selected), ("test", test)):
                    metrics = _metrics(y[index], _probability(model, X[index]), threshold)
                    rows.append({"fold": fold, "train_size": float(size) if not absolute else int(size),
                                 "requested_train_n": count, "actual_train_n": len(selected),
                                 "split": role, **metrics})
            assignments.append({"fold": fold, "train_size": size,
                                "train_indices": selected.tolist(), "test_indices": test.tolist()})
            if return_models:
                models.append({"fold": fold, "train_size": size, "model": model})
    summary = []
    for size in sizes:
        for role in ("train", "test"):
            selected_rows = [row for row in rows if row["train_size"] == size and row["split"] == role]
            for name in ("auc", "brier", "log_loss", "sensitivity", "specificity", "ppv", "npv", "accuracy"):
                values = [row[name] for row in selected_rows if row[name] is not None]
                summary.append({"train_size": size, "split": role, "metric": name,
                                "mean": float(np.mean(values)) if values else None,
                                "sd": float(np.std(values, ddof=1)) if len(values)>1 else None,
                                "valid_folds": len(values), "total_folds": len(splits)})
    result = {"fold_metrics": rows, "summary": summary, "splits": assignments,
              "scope": "apparent training and fixed outer-test curves; fold SD is descriptive, not a confidence interval; no tuning or external validation"}
    if return_models:
        result["models"] = models
    return result
