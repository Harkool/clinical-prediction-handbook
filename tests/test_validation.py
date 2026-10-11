"""Actual fold fitting, patient isolation, preprocessing leakage and failures."""
import unittest

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from cph.preprocessing import build_pipeline, build_preprocessor, validate_columns
from cph.validation import cross_validate_model, learning_curve


class BadProbabilityClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.tile([float("inf"), -float("inf")], (len(X), 1))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(20261011)
        n = 120
        self.X = np.empty((n, 3), dtype=object)
        self.X[:, 0] = rng.normal(size=n)
        self.X[:, 1] = rng.normal(size=n)
        self.X[:, 2] = np.where(np.arange(n) % 3, "A", "B")
        self.X[::11, 0] = np.nan
        self.X[::13, 2] = None
        self.y = np.tile([0, 1], n//2)
        rng.shuffle(self.y)
        self.ids = ["P%03d" % i for i in range(n)]
        self.pipeline = build_pipeline(LogisticRegression(max_iter=500),
                                       ["age", "lab"], ["group"],
                                       ["age", "lab", "group"])

    def cv(self, **kwargs):
        values = dict(ids=self.ids, n_splits=3, seed=42)
        values.update(kwargs)
        return cross_validate_model(self.pipeline, self.X, self.y, **values)

    def test_real_oof_and_no_estimator_reuse(self):
        result = self.cv(return_models=True)
        self.assertEqual(len(result["oof"]), len(self.y))
        self.assertEqual([row["row"] for row in result["oof"]], list(range(len(self.y))))
        self.assertEqual(len(result["fold_metrics"]), 3)
        self.assertEqual({row["fold"] for row in result["oof"]}, {1, 2, 3})
        self.assertFalse(hasattr(self.pipeline.named_steps["preprocess"], "transformers_"))
        self.assertEqual(len({id(model) for model in result["models"]}), 3)
        for split in result["splits"]:
            self.assertFalse(set(split["train_ids"]) & set(split["test_ids"]))
        p = np.array([row["probability"] for row in result["oof"]])
        self.assertAlmostEqual(result["pooled_metrics"]["brier"], np.mean((p-self.y)**2))
        self.assertEqual(sum(result["pooled_metrics"][name] for name in ("tp", "fp", "tn", "fn")), len(self.y))

    def test_held_out_changes_cannot_train_imputer_scaler_or_model(self):
        first_train, first_test = next(StratifiedKFold(3, shuffle=True, random_state=42).split(self.X, self.y))
        changed = self.X.copy()
        changed[first_test, 1] = 1e6
        changed[first_test[0], 2] = "ONLY_IN_HELD_OUT"
        before = self.cv(return_models=True)
        after = cross_validate_model(self.pipeline, changed, self.y, ids=self.ids,
                                     n_splits=3, seed=42, return_models=True)
        one, two = before["models"][0], after["models"][0]
        numeric = one.named_steps["preprocess"].named_transformers_["numeric"]
        numeric_changed = two.named_steps["preprocess"].named_transformers_["numeric"]
        np.testing.assert_allclose(numeric.named_steps["impute"].statistics_,
                                   numeric_changed.named_steps["impute"].statistics_)
        np.testing.assert_allclose(numeric.named_steps["scale"].mean_,
                                   numeric_changed.named_steps["scale"].mean_)
        np.testing.assert_allclose(one.named_steps["model"].coef_, two.named_steps["model"].coef_)
        categories = two.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"].categories_[0]
        self.assertNotIn("ONLY_IN_HELD_OUT", categories)
        self.assertAlmostEqual(numeric_changed.named_steps["impute"].statistics_[1],
                               np.median(np.asarray(changed[first_train, 1], dtype=float)))
        self.assertNotAlmostEqual(numeric_changed.named_steps["impute"].statistics_[1],
                                  np.median(np.asarray(changed[:, 1], dtype=float)))

    def test_repeat_ids_rejected_without_groups(self):
        with self.assertRaisesRegex(ValueError, "Repeated patient"):
            self.cv(ids=["same"]*len(self.y))

    def test_grouped_cv_keeps_whole_patients(self):
        ids = np.array(["P%03d" % (i//2) for i in range(len(self.y))])
        y = np.tile([0, 1], len(self.y)//2)
        result = cross_validate_model(self.pipeline, self.X, y, ids=ids, groups=ids,
                                     n_splits=3, seed=42)
        for split in result["splits"]:
            self.assertFalse(set(split["train_ids"]) & set(split["test_ids"]))
        self.assertEqual(len(result["oof"]), len(y))

    def test_same_patient_in_different_groups_rejected(self):
        ids = self.ids.copy()
        ids[1] = ids[0]
        with self.assertRaisesRegex(ValueError, "different groups"):
            self.cv(ids=ids, groups=self.ids)

    def test_group_fold_single_class_rejected(self):
        groups = np.where(self.y, "positive", "negative")
        with self.assertRaisesRegex(ValueError, "both outcome classes"):
            self.cv(groups=groups, n_splits=2)

    def test_configuration_unknown_overlap_and_omission(self):
        for numeric, categorical, names in ((["bad"], [], ["a"]), ([0], [0], None),
                                            ([0, 2], [], None), ([0], [], ["a", "b"])):
            with self.subTest(numeric=numeric):
                with self.assertRaises(ValueError):
                    validate_columns(numeric, categorical, names)
        with self.assertRaisesRegex(ValueError, "declared predictor"):
            cross_validate_model(self.pipeline, self.X[:, :2], self.y, ids=self.ids)

    def test_missing_and_unknown_category_transform(self):
        preprocessor = build_preprocessor([0], [1])
        training = np.array([[1., "A"], [None, "B"], [3., None], [4., "A"]], dtype=object)
        preprocessor.fit(training)
        values = preprocessor.transform(np.array([[None, "NEW"], [2., None]], dtype=object))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values[0, 1:] == 0))

    def test_entire_missing_training_feature_retained(self):
        preprocessor = build_preprocessor([0], [1])
        values = preprocessor.fit_transform(np.array([[None, None], [np.nan, ""]], dtype=object))
        self.assertEqual(values.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_bad_labels_lengths_thresholds_and_features(self):
        for labels in (self.y[:-1], np.full(len(self.y), .4), np.zeros(len(self.y))):
            with self.subTest(labels=len(labels)):
                with self.assertRaises(ValueError):
                    cross_validate_model(self.pipeline, self.X, labels, ids=self.ids)
        for threshold in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                self.cv(threshold=threshold)
        bad = self.X.copy(); bad[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "infinite"):
            cross_validate_model(self.pipeline, bad, self.y, ids=self.ids)

    def test_convergence_warning_fails_run(self):
        pipeline = build_pipeline(LogisticRegression(max_iter=1), [0, 1], [2])
        with self.assertRaises(ConvergenceWarning):
            cross_validate_model(pipeline, self.X, self.y, ids=self.ids, n_splits=3)

    def test_invalid_classifier_probability_fails_run(self):
        pipeline = build_pipeline(BadProbabilityClassifier(), [0, 1], [2])
        with self.assertRaisesRegex(ValueError, "normalized binary probabilities"):
            cross_validate_model(pipeline, self.X, self.y, ids=self.ids, n_splits=3)

    def test_numerical_learning_curves_nested_and_held_out(self):
        result = learning_curve(self.pipeline, self.X, self.y, ids=self.ids,
                                n_splits=3, train_sizes=(.25, .5, 1.), seed=42)
        self.assertEqual(len(result["fold_metrics"]), 18)
        for fold in range(1, 4):
            splits = [split for split in result["splits"] if split["fold"] == fold]
            for one, two in zip(splits, splits[1:]):
                self.assertTrue(set(one["train_indices"]).issubset(two["train_indices"]))
                self.assertEqual(one["test_indices"], two["test_indices"])
            for split in splits:
                self.assertFalse(set(split["train_indices"]) & set(split["test_indices"]))
                self.assertEqual(len(np.unique(self.y[split["train_indices"]])), 2)
        brier = next(row for row in result["summary"] if row["train_size"] == .5 and row["split"] == "test" and row["metric"] == "brier")
        values = [row["brier"] for row in result["fold_metrics"] if row["train_size"] == .5 and row["split"] == "test"]
        self.assertAlmostEqual(brier["sd"], np.std(values, ddof=1))
        self.assertEqual(brier["valid_folds"], 3)
        self.assertIn("not a confidence interval", result["scope"])

    def test_group_learning_curve_never_splits_patient(self):
        ids = np.array(["P%03d" % (i//2) for i in range(len(self.y))])
        y = np.tile([0, 1], len(self.y)//2)
        result = learning_curve(self.pipeline, self.X, y, ids=ids, groups=ids,
                                n_splits=3, train_sizes=(.3, .6, 1.), seed=42)
        for split in result["splits"]:
            train = set(split["train_indices"])
            patients = set(ids[list(train)])
            self.assertEqual(train, {i for i, patient in enumerate(ids) if patient in patients})

    def test_small_or_mixed_learning_sizes_rejected(self):
        for sizes in ((.001, .5), (2, .5), (1000,), (.5, .25), (.5, .5)):
            with self.subTest(sizes=sizes):
                with self.assertRaises(ValueError):
                    learning_curve(self.pipeline, self.X, self.y, ids=self.ids,
                                   n_splits=3, train_sizes=sizes, seed=42)


if __name__ == "__main__":
    unittest.main()
