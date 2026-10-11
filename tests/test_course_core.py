"""Frozen case migration, unpenalized fits, units and observed-label contracts."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cph.course_core import (DESIGN_TERMS, ROOT, as_columns, assert_model_identity,
    case_recipe, case_units, checked_logistic_fit, export_case, fit_case, load_case,
    make_design, make_state, model_identity, predict_case, simulate_validation,
    subset_case, validate_availability)


class CourseCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_case()
        cls.columns = as_columns(cls.data)
        cls.model = fit_case(cls.columns)

    def test_frozen_case_counts_and_original_coefficients(self):
        self.assertEqual(len(self.data), 3200)
        self.assertEqual(sum(row["y"] for row in self.data), 384)
        self.assertEqual(sum(row["cd_grade"] == 5 for row in self.data), 39)
        self.assertNotIn("true_risk", self.data[0])
        self.assertEqual(self.model["terms"], list(DESIGN_TERMS))
        self.assertEqual(make_design(self.data).shape, (3200, 19))
        with (ROOT/"examples/reference/model_coefficients.csv").open(encoding="utf-8-sig") as stream:
            reference = list(csv.DictReader(stream))
        np.testing.assert_allclose(self.model["beta"], [float(row["coefficient"]) for row in reference], rtol=0, atol=1e-8)

    def test_all_frozen_predictions_reproduce_original_reference(self):
        with (ROOT/"examples/reference/predictions.csv").open(encoding="utf-8-sig") as stream:
            reference = list(csv.DictReader(stream))
        self.assertEqual([row["id"] for row in reference], [row["id"] for row in self.data])
        np.testing.assert_allclose(predict_case(self.model, self.data),
                                   [float(row["prediction"]) for row in reference], rtol=0, atol=1e-8)

    def test_outcomes_times_and_extra_fields_cannot_enter_design(self):
        changed = {name: value.copy() for name, value in self.columns.items()}
        for name in ("y", "cd_grade", "event_time", "death_time", "decision_time"):
            changed[name] = np.full(3200, 999.)
        changed["true_risk"] = np.ones(3200)
        np.testing.assert_array_equal(make_design(changed), make_design(self.columns))
        bad = case_recipe(); bad["predictors"][-1] = "true_risk"
        with self.assertRaisesRegex(ValueError, "whitelist"):
            make_design(changed, bad)

    def test_unit_roundtrip_complete_rule_and_json_export(self):
        data = subset_case(self.columns, np.arange(20))
        changed = {name: value.copy() for name, value in data.items()}
        changed["hb"] /= 10; changed["alb"] /= 10; changed["crea"] /= 88.4
        units = case_units(); units.update(hb="g/dL", alb="g/dL", crea="mg/dL")
        np.testing.assert_allclose(predict_case(self.model, changed, units, self.model["identity"]),
                                   predict_case(self.model, data), rtol=0, atol=1e-12)
        self.assertEqual(model_identity(self.model), self.model["identity"])
        self.assertTrue(self.model["identity"].startswith("sha256:"))
        temporary_root = ROOT/"outputs/test-course-core"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            export_case(data, self.model, folder)
            restored = json.loads((Path(folder)/"results/model.json").read_text(encoding="utf-8"))
            np.testing.assert_array_equal(predict_case(restored, data), predict_case(self.model, data))
            with (Path(folder)/"results/model_coefficients.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            np.testing.assert_array_equal([float(row["coefficient"]) for row in rows], self.model["beta"])

    def test_tampered_coefficients_recipe_and_target_are_rejected(self):
        for section in ("beta", "recipe", "target"):
            changed = copy.deepcopy(self.model)
            if section == "beta":
                changed["beta"][1] += .01
            elif section == "recipe":
                changed["recipe"]["age_knots"][1] = 35.
            else:
                changed["target"]["window_days"] = 60
            with self.subTest(section=section):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    predict_case(changed, self.data[:1])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            assert_model_identity(self.model, "md5:"+"0"*32)

    def test_invalid_units_levels_missing_and_ranges_rejected(self):
        for name, value in (("hb", None), ("crea", float("inf")), ("asa", 2.5), ("male", .5), ("age", 100)):
            data = copy.deepcopy(self.data[:1]); data[0][name] = value
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    predict_case(self.model, data)
        units = case_units(); units["hb"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            predict_case(self.model, self.data[:1], units)

    def test_bootstrap_repeated_rows_allowed_and_warmstart_checked(self):
        rng = np.random.default_rng(42)
        boot = subset_case(self.columns, rng.integers(0, 3200, 3200))
        fitted = fit_case(boot)
        self.assertEqual(fitted["metadata"]["n"], 3200)
        self.assertLess(len(set(boot["id"])), 3200)
        warm = checked_logistic_fit(make_design(self.columns), self.columns["y"],
                                    initial=self.model["beta"])
        np.testing.assert_allclose(warm["coefficients"], self.model["beta"], rtol=0, atol=1e-8)
        self.assertLess(warm["gradient"], 1e-10)

    def test_logistic_separation_rank_labels_and_offset_rejected(self):
        x = np.linspace(-2, 2, 100)
        X = np.column_stack((np.ones(100), x))
        cases = ((X, (x>0).astype(int), None),
                 (np.column_stack((np.ones(100), np.ones(100))), np.tile([0, 1], 50), None),
                 (X, np.full(100, .3), None), (X, np.tile([0, 1], 50), np.array([0., 0.])))
        for design, y, offset in cases:
            with self.subTest(shape=design.shape):
                with self.assertRaises(ValueError):
                    checked_logistic_fit(design, y, offset)

    def test_labels_death_target_unknown_and_cutoff(self):
        rows = [dict(outcome_origin=0., event_time=5., death_time=None, last_contact=30.),
                dict(outcome_origin=0., event_time=None, death_time=3., last_contact=30.),
                dict(outcome_origin=0., event_time=None, death_time=None, last_contact=12.),
                dict(outcome_origin=0., event_time=20., death_time=None, last_contact=30.)]
        self.assertEqual([row["y"] for row in make_state(rows, 30)], [1, 1, None, 1])
        self.assertEqual(make_state(rows, 30, outcome="nonfatal_complication")[1]["state"], "competing")
        self.assertIsNone(make_state(rows, 10)[3]["y"])
        self.assertEqual(make_state(rows, 10)[3]["state"], "incomplete_admin")
        labels = make_state(self.columns, max(self.columns["last_contact"])+1)
        self.assertEqual(sum(row["y"] for row in labels), 384)

    def test_invalid_label_chronology_and_windows_rejected(self):
        row = dict(outcome_origin=0., event_time=5., death_time=3., last_contact=30.)
        with self.assertRaisesRegex(ValueError, "chronology"):
            make_state([row], 30)
        row.update(event_time=None, death_time=None)
        for cut, window in ((float("inf"), 30), (30, 0), (30, -1)):
            with self.assertRaises(ValueError):
                make_state([row], cut, window)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            make_state([row]*4, [10, 30])
        row["outcome_origin"] = float("inf")
        with self.assertRaises(ValueError):
            make_state([row], 30)

    def test_availability_future_and_unknown_rejected_in_fit(self):
        for value in (None, self.data[0]["decision_time"]+1):
            changed = copy.deepcopy(self.data)
            changed[0]["result_available_time"] = value
            with self.assertRaises(ValueError):
                validate_availability(changed)
            with self.assertRaises(ValueError):
                fit_case(changed)

    def test_pcg_validation_reproducible_distinct_and_labeled(self):
        first, second = simulate_validation(200, 42, .18), simulate_validation(200, 42, .18)
        self.assertEqual(first, second)
        self.assertTrue(all(row["id"].startswith("V") for row in first))
        self.assertFalse(set(row["id"] for row in first) & set(self.columns["id"]))
        self.assertTrue(validate_availability(first))
        labels = make_state(first, max(row["last_contact"] for row in first)+1)
        self.assertEqual([row["y"] for row in labels], [row["y"] for row in first])
        self.assertNotEqual(first, simulate_validation(200, 43, .18))


if __name__ == "__main__":
    unittest.main()
