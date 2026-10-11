"""Hand calculations and failures protecting probability/identity contracts."""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    from cph.metrics import (evaluate, threshold_metrics, roc_data, pr_data,
                             decision_curve, delong_compare, calibration_summary,
                             compare_models)
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    np = None


@unittest.skipIf(np is None, "Optional toolkit requires NumPy")
class ProbabilityMetricTests(unittest.TestCase):
    def evaluation(self, y, p):
        return evaluate(y, p, ["P%d" % index for index in range(len(y))])

    def test_inputs_reject_nonfinite_nonbinary_scores_ids_and_dimensions(self):
        invalid = [([0, 1], [0.1, float("nan")], ["a", "b"]),
                   ([0, 1], [0.1, float("inf")], ["a", "b"]),
                   ([0, float("nan")], [.1, .9], ["a", "b"]),
                   ([0, 2], [.1, .9], ["a", "b"]),
                   ([0, 0], [.1, .9], ["a", "b"]),
                   ([0, 1], [-.1, .9], ["a", "b"]),
                   ([0, 1], [.1, 1.1], ["a", "b"]),
                   ([0, 1], [.1, .9], ["a", "a"]),
                   ([0, 1], [.1, .9], ["a", ""]),
                   ([0, 1], [.1, .9], ["a", None]),
                   ([0, 1], [.1], ["a", "b"]),
                   ([[0], [1]], [.1, .9], ["a", "b"]),
                   ([0, 1], [.1+1j, .9], ["a", "b"])]
        for y, p, ids in invalid:
            with self.subTest(y=y, p=p, ids=ids), self.assertRaises(ValueError):
                evaluate(y, p, ids)

    def test_evaluation_freezes_input_values(self):
        labels, predictions, ids = [0, 1], [.2, .8], ["a", "b"]
        result = evaluate(labels, predictions, ids)
        labels[0], predictions[0], ids[0] = 1, .9, "changed"
        self.assertEqual(result.y, (0, 1))
        self.assertEqual(result.probability, (.2, .8))
        self.assertEqual(result.ids, ("a", "b"))

    def test_threshold_counts_and_prespecified_cutoff(self):
        result = threshold_metrics(self.evaluation([0, 0, 1, 1], [.1, .8, .4, .9]), .5)
        self.assertEqual([result[name] for name in ["TN", "FP", "FN", "TP"]], [1, 1, 1, 1])
        for name in ["accuracy", "sensitivity", "specificity", "PPV", "NPV", "F1"]:
            self.assertEqual(result[name], .5)
        self.assertLess(result["PPV_interval"][0], .5)
        self.assertGreater(result["PPV_interval"][1], .5)
        boundary = threshold_metrics(self.evaluation([0, 1], [.4, .5]), .5)
        self.assertEqual(boundary["TP"], 1)
        with self.assertRaises(ValueError):
            threshold_metrics(self.evaluation([0, 1], [.2, .9]), "Youden")

    def test_empty_prediction_class_is_not_given_invented_npv(self):
        result = threshold_metrics(self.evaluation([0, 1], [.8, .9]), .5)
        self.assertIsNone(result["NPV"])
        self.assertIsNone(result["NPV_interval"])
        self.assertIn("zero denominator", result["not_estimable"]["NPV"])

    def test_tied_auc_and_ap_differ_from_trapezoidal_pr_area(self):
        evaluation = self.evaluation([1, 1, 0, 0], [.5, .5, .5, .1])
        self.assertAlmostEqual(roc_data(evaluation)["auc"], .75)
        pr = pr_data(evaluation)
        self.assertAlmostEqual(pr["average_precision"], 2/3)
        self.assertAlmostEqual(pr["trapezoidal_pr_area"], 5/6)
        self.assertNotEqual(pr["average_precision"], pr["trapezoidal_pr_area"])

    def test_auc_and_ap_agree_with_independent_library(self):
        if not importlib.util.find_spec("sklearn"):
            self.skipTest("scikit-learn is optional")
        from sklearn.metrics import roc_auc_score, average_precision_score
        rng = np.random.default_rng(123)
        y = [0]*17+[1]*11
        p = np.round(rng.uniform(size=len(y)), 1).tolist()
        evaluation = self.evaluation(y, p)
        self.assertAlmostEqual(roc_data(evaluation)["auc"], roc_auc_score(y, p), places=14)
        self.assertAlmostEqual(pr_data(evaluation)["average_precision"], average_precision_score(y, p), places=14)

    def test_dca_matches_hand_counts_and_rejects_singular_thresholds(self):
        evaluation = self.evaluation([1, 1, 0, 0], [.9, .4, .8, .1])
        rows = decision_curve(evaluation, [.25, .5])
        self.assertAlmostEqual(rows[0]["net_benefit"], 5/12)
        self.assertAlmostEqual(rows[0]["treat_all"], 1/3)
        self.assertEqual(rows[1]["net_benefit"], 0)
        self.assertEqual(rows[1]["treat_all"], 0)
        self.assertEqual(rows[1]["treat_none"], 0)
        for thresholds in [[0], [1], [-.1], [float("nan")], [.5, .25], [.2, .2]]:
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                decision_curve(evaluation, thresholds)

    def test_paired_delong_hand_variance_difference_and_p(self):
        first = self.evaluation([1, 1, 0, 0], [.8, .5, .6, .1])
        second = self.evaluation([1, 1, 0, 0], [.9, .8, .4, .2])
        result = delong_compare(first, second)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["auc_first"], .75)
        self.assertEqual(result["auc_second"], 1)
        self.assertEqual(result["difference"], -.25)
        self.assertAlmostEqual(result["difference_variance"], .125)
        self.assertAlmostEqual(result["standard_error"], math.sqrt(.125))
        self.assertAlmostEqual(result["p_value"], math.erfc(.25/math.sqrt(.125)/math.sqrt(2)))
        self.assertLess(result["interval"][0], -.25)
        self.assertGreater(result["interval"][1], -.25)

    def test_delong_ties_agree_with_direct_pair_placements(self):
        first = self.evaluation([1, 1, 1, 0, 0, 0], [.7, .5, .2, .5, .1, .4])
        second = self.evaluation([1, 1, 1, 0, 0, 0], [.8, .4, .3, .3, .2, .6])
        blocks = []
        for evaluation in [first, second]:
            pos = [p for y, p in zip(evaluation.y, evaluation.probability) if y]
            neg = [p for y, p in zip(evaluation.y, evaluation.probability) if not y]
            u = np.array([[float(a>b)+.5*float(a==b) for b in neg] for a in pos])
            blocks.append((u.mean(axis=1), u.mean(axis=0)))
        difference_pos = blocks[0][0]-blocks[1][0]
        difference_neg = blocks[0][1]-blocks[1][1]
        variance = np.var(difference_pos, ddof=1)/3+np.var(difference_neg, ddof=1)/3
        result = delong_compare(first, second)
        self.assertAlmostEqual(result["difference_variance"], variance, places=14)

    def test_pairing_rejects_patient_order_and_label_mismatches(self):
        first = evaluate([0, 1, 0, 1], [.1, .8, .2, .9], ["a", "b", "c", "d"])
        for second in [evaluate([0, 1, 0, 1], [.1, .8, .2, .9], ["c", "b", "a", "d"]),
                       evaluate([1, 0, 0, 1], [.8, .1, .2, .9], ["a", "b", "c", "d"])]:
            with self.assertRaises(ValueError):
                delong_compare(first, second)
            with self.assertRaises(ValueError):
                compare_models({"first": first, "second": second})

    def test_degenerate_delong_reports_reason_without_nan_or_fake_p(self):
        constant = self.evaluation([0, 0, 1, 1], [.5]*4)
        result = delong_compare(constant, constant)
        self.assertEqual(result["status"], "NOT_ESTIMABLE")
        self.assertIsNone(result["p_value"])
        self.assertIsNone(result["interval"])
        self.assertIn("Zero contrast variance", result["reason"])
        scarce = self.evaluation([1, 0, 0], [.9, .1, .2])
        self.assertIn("At least two events", delong_compare(scarce, scarce)["reason"])
        json.dumps(result, allow_nan=False)

    def test_calibration_groups_include_endpoints_counts_and_empty_bins(self):
        evaluation = self.evaluation([0, 0, 1, 1], [0, .2, .8, 1])
        result = calibration_summary(evaluation, bins=10)
        self.assertEqual(sum(row["n"] for row in result["bins"]), 4)
        self.assertEqual(sum(row["events"] for row in result["bins"]), 2)
        self.assertEqual(result["bins"][0]["n"], 1)
        self.assertEqual(result["bins"][-1]["n"], 1)
        self.assertEqual(result["bins"][4]["status"], "EMPTY_BIN")
        self.assertAlmostEqual(result["Brier"], .02)
        self.assertEqual(result["O_over_E"], 1)
        self.assertEqual(result["joint_intercept_and_slope"]["status"], "NOT_ESTIMABLE")
        self.assertIn("not silently clipped", result["joint_intercept_and_slope"]["reason"])

    def test_offset_intercept_and_joint_slope_are_distinct_tasks(self):
        evaluation = self.evaluation([1]*4+[0]*16, [.25]*20)
        result = calibration_summary(evaluation)
        offset = result["calibration_in_the_large"]
        self.assertEqual(offset["status"], "PASS")
        self.assertAlmostEqual(offset["intercept"], math.log(.2/.8)-math.log(.25/.75), places=8)
        self.assertEqual(result["joint_intercept_and_slope"]["status"], "NOT_ESTIMABLE")
        self.assertIn("Rank-deficient", result["joint_intercept_and_slope"]["reason"])

    def test_separated_calibration_is_not_reported_as_finite_good_fit(self):
        evaluation = self.evaluation([0]*10+[1]*10, [.1]*10+[.9]*10)
        result = calibration_summary(evaluation)["joint_intercept_and_slope"]
        self.assertEqual(result["status"], "NOT_ESTIMABLE")
        self.assertIsNone(result["slope"])

    def test_calibration_tolerates_ulp_loss_noise_but_still_solves_the_score(self):
        from cph import metrics
        evaluation = self.evaluation([1]*4+[0]*16, [.25]*20)
        original_sum = np.sum
        perturbations = 0

        def rounded_objective_sum(values, *args, **kwargs):
            nonlocal perturbations
            total = original_sum(values, *args, **kwargs)
            losses = np.asarray(values)
            # At the finite MLE q=.2, the first event loss is -log(.2).
            # A few ULPs of summed-loss noise can conceal the last Newton
            # decrease. The discontinuity models platform reduction rounding;
            # it does not change outcomes, probabilities, score or information.
            if losses.shape == (20,) and abs(float(losses[0])-math.log(5)) < 1e-9:
                perturbations += 1
                for _ in range(4):
                    total = np.nextafter(total, np.inf)
            return total

        # Only the private regression is called under this patch: its np.sum
        # calls are objective reductions, not grouped counts or score checks.
        with patch.object(metrics.np, "sum", side_effect=rounded_objective_sum):
            result = metrics._logistic_calibration(evaluation, slope=False)
        self.assertGreater(perturbations, 0)
        self.assertEqual(result["status"], "PASS")
        expected = math.log(.2/.8)-math.log(.25/.75)
        self.assertAlmostEqual(result["intercept"], expected, places=10)
        eta = math.log(.25/.75)+result["intercept"]
        fitted_probability = 1/(1+math.exp(-eta))
        self.assertLess(abs(fitted_probability-.2), 1e-11)

    def test_finite_joint_calibration_matches_closed_form_after_row_reordering(self):
        # Two logit values with event fractions .2 and .8 give intercept 0,
        # slope log(4). Unlike a separation fixture, both labels occur at each
        # predictor value. Repeat to match the 800-patient held-out lesson size.
        y = np.tile([0]*8+[1]*2+[0]*2+[1]*8, 40)
        lp = np.tile([-1.]*10+[1.]*10, 40)
        p = 1/(1+np.exp(-lp))
        orders = (np.arange(800), np.arange(799, -1, -1),
                  np.random.default_rng(20261021).permutation(800))
        for order in orders:
            with self.subTest(first_row=int(order[0])):
                result = calibration_summary(self.evaluation(y[order], p[order]))["joint_intercept_and_slope"]
                self.assertEqual(result["status"], "PASS")
                self.assertAlmostEqual(result["intercept"], 0., places=10)
                self.assertAlmostEqual(result["slope"], math.log(4), places=10)
                q = 1/(1+np.exp(-(result["intercept"]+result["slope"]*lp[order])))
                score = np.column_stack([np.ones(800), lp[order]]).T@(q-y[order])
                self.assertLess(float(np.max(np.abs(score)))/800, 1e-11)

    def test_nonfinite_calibration_objective_cannot_be_accepted_as_converged(self):
        from cph import metrics
        evaluation = self.evaluation([1]*4+[0]*16, [.25]*20)
        with patch.object(metrics.np, "sum", return_value=float("nan")):
            result = metrics._logistic_calibration(evaluation, slope=False)
        self.assertEqual(result["status"], "NOT_ESTIMABLE")
        self.assertIsNone(result["intercept"])

    def test_model_comparison_has_same_evaluation_population_and_json_safe_results(self):
        first = self.evaluation([0, 1, 0, 1], [.1, .8, .2, .6])
        second = self.evaluation([0, 1, 0, 1], [.3, .5, .1, .9])
        result = compare_models({"plain": first, "alternative": second}, threshold=.15, thresholds=[.1, .2])
        self.assertEqual(result["n"], 4)
        self.assertEqual(len(result["paired_auc_comparisons"]), 1)
        self.assertEqual(result["models"]["plain"]["threshold"]["threshold"], .15)
        json.dumps(result, allow_nan=False)

    def test_metrics_import_does_not_load_plotting_or_neural_frameworks(self):
        code = "import sys; import cph.metrics; assert 'matplotlib.pyplot' not in sys.modules; assert 'keras' not in sys.modules; assert 'tensorflow' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], cwd=Path(os.path.abspath(__file__)).parents[1], check=True)


@unittest.skipIf(np is None or not importlib.util.find_spec("matplotlib"), "Optional plots require NumPy and Matplotlib")
class ProbabilityPlotTests(unittest.TestCase):
    def test_all_five_figures_return_data_save_and_preserve_other_figures(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from cph.plots import plot_roc, plot_pr, plot_dca, plot_calibration, plot_confusion
        evaluation = evaluate([0, 1, 0, 1, 0, 1], [.1, .7, .4, .8, .2, .6], list("abcdef"))
        untouched = plt.figure()
        try:
            with tempfile.TemporaryDirectory() as directory:
                for name, plot in [("roc", plot_roc), ("pr", plot_pr), ("dca", plot_dca),
                                   ("calibration", plot_calibration),
                                   ("confusion", lambda data: plot_confusion(data, .5))]:
                    with self.subTest(name=name):
                        fig, axes, data = plot(evaluation)
                        self.assertTrue(data)
                        self.assertIn(untouched.number, plt.get_fignums())
                        path = Path(directory)/(name+".png")
                        fig.savefig(path, dpi=100)
                        self.assertGreater(path.stat().st_size, 1000)
                        image = plt.imread(path)
                        self.assertGreater(image.shape[0], 100)
                        self.assertGreater(image.shape[1], 100)
                        self.assertGreater(float(image[..., :3].std()), .03)
                        plt.close(fig)
        finally:
            plt.close(untouched)

    def test_plot_overlays_require_the_same_patient_order(self):
        from cph.plots import plot_roc
        first = evaluate([0, 1], [.1, .8], ["a", "b"])
        second = evaluate([0, 1], [.2, .7], ["c", "d"])
        with self.assertRaises(ValueError):
            plot_roc({"first": first, "second": second})


if __name__ == "__main__":
    unittest.main()
