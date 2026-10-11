"""Executable chapter integration and stale/failure/cache contracts."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "sklearn", "matplotlib"))
if AVAILABLE:
    from cph.course import (run_course, course_budgets, source_signature, _cached,
                            _invalidate_prior)

ROOT = Path(os.path.abspath(__file__)).parents[1]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@unittest.skipUnless(AVAILABLE, "Python course needs NumPy, SciPy, scikit-learn and Matplotlib")
class PythonCourseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/"outputs").mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="python-course-test-", dir=ROOT/"outputs")
        cls.output = Path(cls.temporary.name)
        cls.course_result = run_course("all", cls.output, "quick")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def chapter(self, number, artifact):
        return self.output/("chapters/ch%02d" % number)/artifact

    def test_every_chapter_has_actual_current_status_and_lesson(self):
        manifest = read_csv(self.output/"manifest.csv")
        self.assertEqual([int(row["chapter"]) for row in manifest], list(range(1, 21)))
        for row in manifest:
            number = int(row["chapter"])
            self.assertEqual(row["status"], "PASS_EXECUTED")
            self.assertEqual(row["source_signature"], self.course_result["source_signature"])
            self.assertTrue(self.chapter(number, "lesson.md").is_file())
            status = json.loads(self.chapter(number, "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["chapter"], number)
            self.assertEqual(status["status"], "PASS_EXECUTED")

    def test_frozen_source_cohort_and_json_model_are_exported(self):
        data = read_csv(self.output/"data/main_case.csv")
        self.assertEqual(len(data), 3200)
        self.assertEqual(sum(int(row["y"]) for row in data), 384)
        self.assertNotIn("true_risk", data[0])
        model = json.loads((self.output/"results/model.json").read_text(encoding="utf-8"))
        self.assertEqual(len(model["beta"]), 19)
        self.assertEqual(model["metadata"]["parameters"], 18)
        sources = read_csv(self.output/"results/context_sources.csv")
        self.assertEqual({row["file"] for row in sources}, {"course_core.py", "course_validation.py", "course.py", "metrics.py",
                         "preprocessing.py", "validation.py", "predictions.py", "baseline.py"})
        self.assertEqual(len(sources), 8)

    def test_unknown_outcome_is_preserved_and_competing_death_differs_from_a(self):
        fixtures = read_csv(self.chapter(5, "label_fixtures.csv"))
        self.assertEqual(fixtures[-1]["y"], "")
        self.assertIn(fixtures[-1]["state"], ("incomplete_admin", "incomplete_lost"))
        labels = read_csv(self.chapter(5, "labels_A_B.csv"))
        competing = [row for row in labels if row["B_state"] == "competing"]
        self.assertTrue(competing)
        self.assertTrue(all(row["A_y"] == "1" and row["B_y"] == "0" for row in competing))

    def test_missing_recipe_is_fit_without_heldout_values(self):
        result = read_csv(self.chapter(7, "boundary_checks.csv"))[0]
        self.assertEqual(result["training_median"], result["training_median_after_test_change"])
        self.assertNotEqual(result["full_data_median_before"], result["full_data_median_after"])
        assignments = read_csv(self.chapter(7, "missing_assignments.csv"))
        self.assertEqual(sum(row["role"] == "TRAIN" for row in assignments), 2240)
        self.assertEqual(sum(row["hb_missing"] == "True" for row in assignments), 480)

    def test_branch_runs_aj_and_cause_specific_cox_without_finegray_claim(self):
        curves = read_csv(self.chapter(9, "aalen_johansen.csv"))
        self.assertGreater(len(curves), 10)
        self.assertTrue(all(abs(float(row["probability_sum"])-1) < 1e-12 for row in curves))
        coefficients = read_csv(self.chapter(9, "branch_coefficients.csv"))
        self.assertEqual(len(coefficients), 6)
        self.assertEqual({row["cause"] for row in coefficients}, {"1", "2"})
        self.assertTrue(all("NOT Fine-Gray" in row["method"] for row in coefficients))
        self.assertTrue(self.chapter(9, "aalen_johansen.png").is_file())
        self.assertTrue(self.chapter(9, "aalen_johansen.pdf").is_file())

    def test_outer_baseline_refits_and_heldout_predictions_cover_each_patient(self):
        rows = read_csv(self.chapter(10, "outer_predictions.csv"))
        for candidate in {row["candidate"] for row in rows}:
            subset = [row for row in rows if row["candidate"] == candidate]
            self.assertEqual(len(subset), 3200)
            self.assertEqual(len({row["id"] for row in subset}), 3200)
        roles = read_csv(self.chapter(10, "outer_roles.csv"))
        for fold in ("1", "2", "3"):
            selected = [row for row in roles if row["outer_fold"] == fold]
            train = {row["id"] for row in selected if row["role"] == "TRAIN"}
            test = {row["id"] for row in selected if row["role"] == "TEST"}
            self.assertFalse(train & test)
            self.assertEqual(len(train)+len(test), 3200)
            self.assertTrue(all(int(row["baseline_training_n"]) == len(train) for row in selected))

    def test_nested_selection_uses_train_only_and_one_lambda_per_fold(self):
        rows = read_csv(self.chapter(11, "inner_tuning.csv"))
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(int(row["test_rows_used_for_tuning"]) == 0 for row in rows))
        for fold in ("1", "2", "3"):
            selected = [row for row in rows if row["outer_fold"] == fold]
            winner = [row for row in selected if row["selected"] == "True"]
            self.assertEqual(len(winner), 1)
            self.assertEqual(float(winner[0]["inner_mean_Brier"]), min(float(row["inner_mean_Brier"]) for row in selected))

    def test_intervals_keep_budgets_and_targets_separate(self):
        development = json.loads((self.output/"results/development_intervals.json").read_text(encoding="utf-8"))
        fixed = json.loads((self.output/"results/fixed_model_intervals.json").read_text(encoding="utf-8"))
        self.assertEqual((development["B_outer"], development["B_inner"]), (10, 10))
        self.assertEqual(fixed["B"], 40)
        self.assertEqual(development["status"], "PASS")
        self.assertEqual(fixed["status"], "PASS")
        self.assertIn("development-sample uncertainty", development["scope"])
        self.assertIn("fixed model", fixed["scope"])
        curve = read_csv(self.chapter(13, "calibration_curve.csv"))
        self.assertEqual(len(curve), 80)
        self.assertEqual({row["interval_target"] for row in curve}, {"fixed_model_apparent_curve_pointwise"})
        groups = read_csv(self.chapter(13, "risk_groups.csv"))
        self.assertEqual(sum(int(row["n"]) for row in groups), 3200)

    def test_update_and_evaluation_have_disjoint_patient_ids(self):
        roles = read_csv(self.chapter(14, "validation_roles.csv"))
        train = {row["id"] for row in roles if row["role"] == "UPDATE"}
        test = {row["id"] for row in roles if row["role"] == "TEST"}
        self.assertEqual((len(train), len(test)), (800, 800))
        self.assertFalse(train & test)
        self.assertEqual({row["rule"] for row in read_csv(self.chapter(14, "held_out_update_evaluation.csv"))},
                         {"frozen", "intercept_update", "intercept_slope_update"})

    def test_negative_fixtures_actually_reject_and_humans_remain_pending(self):
        cases = read_csv(self.chapter(16, "negative_checks.csv"))
        self.assertEqual(len(cases), 4)
        self.assertTrue(all(row["status"] == "PASS_EXPECTED_REJECTION" and row["observed_error"] for row in cases))
        humans = read_csv(self.chapter(20, "human_review.csv"))
        self.assertTrue(all(row["decision"] == "PENDING" and not row["name"] and not row["date"] for row in humans))

    def test_mode_conflict_leaves_previous_pass_manifest_unchanged(self):
        before = (self.output/"manifest.csv").read_bytes()
        with self.assertRaises(ValueError):
            run_course(8, self.output, "book")
        self.assertEqual((self.output/"manifest.csv").read_bytes(), before)

    def test_source_change_invalidates_prior_pass_and_cache_payload_tampering(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"outputs") as directory:
            out = Path(directory)
            run_course(3, out, "quick")
            _invalidate_prior(out, "different-source-hash")
            self.assertEqual(read_csv(out/"manifest.csv")[0]["status"], "STALE")
            ctx = dict(output=out, signature="source-a")
            self.assertEqual(_cached(ctx, "probe", lambda: {"value": 1}), {"value": 1})
            self.assertEqual(_cached(ctx, "probe", lambda: {"value": 2}), {"value": 1})
            ctx["signature"] = "source-b"
            self.assertEqual(_cached(ctx, "probe", lambda: {"value": 2}), {"value": 2})
            cache = out/"cache/probe.json"
            altered = json.loads(cache.read_text(encoding="utf-8")); altered["value"]["value"] = 999
            cache.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                _cached(ctx, "probe", lambda: {"value": 3})

    def test_failure_overwrites_old_pass_for_that_chapter(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"outputs") as directory:
            out = Path(directory)
            run_course(8, out, "quick")
            with patch("cph.course._chapter", side_effect=ValueError("injected execution failure")):
                with self.assertRaises(ValueError):
                    run_course(8, out, "quick")
            row = read_csv(out/"manifest.csv")[0]
            self.assertEqual(row["status"], "FAIL")
            self.assertIn("injected execution failure", row["scope"])

    def test_budget_modes_are_explicit(self):
        self.assertEqual(course_budgets("quick")["outer"], 10)
        self.assertEqual(course_budgets("book")["outer"], 200)
        self.assertEqual(course_budgets("book")["inner"], 50)
        with self.assertRaises(ValueError):
            course_budgets("unknown")
        with self.assertRaises(ValueError):
            run_course(21, self.output, "quick")


if __name__ == "__main__":
    unittest.main()
