"""Fold-fitted preprocessing for raw numeric/categorical predictors.

This replaces Codebase's Valuestandard role. Build the pipeline before CV;
never fit an imputer, scaler or category encoder on the full cohort first.
Column selectors refer to a raw two-dimensional array. Names are resolved
against feature_names so pandas is not required.
"""
from numbers import Integral

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def validate_columns(numeric_columns, categorical_columns, feature_names=None):
    """Return explicit, disjoint column indices covering every predictor."""
    numeric, categorical = list(numeric_columns), list(categorical_columns)
    if not numeric and not categorical:
        raise ValueError("At least one predictor column is required")
    names = None if feature_names is None else list(feature_names)
    if names is not None:
        if not names or any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("feature_names must be nonempty unique strings")

    def resolve(columns):
        result = []
        for column in columns:
            if isinstance(column, str):
                if names is None or column not in names:
                    raise ValueError("Named column is absent from feature_names: " + column)
                index = names.index(column)
            elif isinstance(column, Integral) and not isinstance(column, (bool, np.bool_)) and column >= 0:
                index = int(column)
            else:
                raise ValueError("Column selectors must be names or nonnegative integer indices")
            if names is not None and index >= len(names):
                raise ValueError("Column index exceeds feature_names")
            result.append(index)
        return result

    numeric, categorical = resolve(numeric), resolve(categorical)
    selected = numeric + categorical
    if len(set(selected)) != len(selected):
        raise ValueError("Numeric/categorical columns must be unique and disjoint")
    width = len(names) if names is not None else max(selected) + 1
    if set(selected) != set(range(width)):
        raise ValueError("Declare every raw predictor exactly once; remove IDs/outcomes before preprocessing")
    return {"numeric": numeric, "categorical": categorical, "n_features": width}


def _missing(value):
    return value is None or (isinstance(value, str) and value.strip().lower() in ("", "na", "nan")) or (
        isinstance(value, (float, np.floating)) and np.isnan(value))


def _numeric_values(values):
    raw = np.asarray(values, dtype=object)
    result = np.empty(raw.shape, dtype=float)
    for index, value in np.ndenumerate(raw):
        if _missing(value):
            result[index] = np.nan
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError("Numeric predictor contains a nonnumeric observed value") from None
            if not np.isfinite(number):
                raise ValueError("Numeric predictor contains an infinite observed value")
            result[index] = number
    return result


def _categorical_values(values):
    raw = np.asarray(values, dtype=object)
    result = np.empty(raw.shape, dtype=object)
    for index, value in np.ndenumerate(raw):
        if _missing(value):
            result[index] = np.nan
        else:
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError("Categorical predictor contains an infinite observed value")
            result[index] = str(value)
    return result


def build_preprocessor(numeric_columns, categorical_columns, feature_names=None):
    """Create an unfitted ColumnTransformer; missing values are None/NA/NaN/blank.

    An entirely missing training numeric feature is retained and imputed to 0,
    explicitly using SimpleImputer's keep_empty_features policy. Numeric
    imputation and scaling, and categorical imputation/category discovery,
    remain inside each cloned training pipeline.
    """
    columns = validate_columns(numeric_columns, categorical_columns, feature_names)
    transformers = []
    if columns["numeric"]:
        numeric = Pipeline([
            ("normalize", FunctionTransformer(_numeric_values, validate=False)),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric, columns["numeric"]))
    if columns["categorical"]:
        categorical = Pipeline([
            ("normalize", FunctionTransformer(_categorical_values, validate=False)),
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical, columns["categorical"]))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def build_pipeline(estimator, numeric_columns, categorical_columns, feature_names=None):
    """Return an unfitted preprocessing/classifier pipeline for raw predictors."""
    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict_proba"):
        raise ValueError("Classifier must implement fit and predict_proba")
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_columns, categorical_columns, feature_names)),
        ("model", clone(estimator)),
    ])


def validate_pipeline_input(pipeline, X):
    """Reject undeclared/missing raw columns before cross-validation starts."""
    if not isinstance(pipeline, Pipeline) or not isinstance(pipeline.named_steps.get("preprocess"), ColumnTransformer):
        raise ValueError("Use build_pipeline with a fold-fitted ColumnTransformer")
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("Pipeline classifier must implement predict_proba")
    selected = []
    for _, _, columns in pipeline.named_steps["preprocess"].transformers:
        if not isinstance(columns, (list, tuple)) or any(not isinstance(c, Integral) for c in columns):
            raise ValueError("Preprocessing requires explicit integer column selectors")
        selected.extend(int(c) for c in columns)
    raw = np.asarray(X, dtype=object)
    if raw.ndim != 2 or raw.shape[0] < 2 or not selected or len(set(selected)) != len(selected) or set(selected) != set(range(raw.shape[1])):
        raise ValueError("Raw X must have exactly the uniquely declared predictor columns")
    return raw
