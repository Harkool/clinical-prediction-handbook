"""Independent Table 1 examples and rejection of ambiguous input."""
import csv
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from cph.baseline import baseline_table, generate_baseline
from test_cli import test_tempdir


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.schema = {"id_column": "id", "group": "arm", "group_levels": ["A", "B"],
                       "variables": [{"name": "age", "type": "continuous", "summary": "median_iqr"},
                                     {"name": "stage", "type": "categorical", "levels": ["I", "II", "III"]}]}
        self.rows = [{"id": "1", "arm": "A", "age": "10", "stage": "I"},
                     {"id": "2", "arm": "A", "age": "20", "stage": ""},
                     {"id": "3", "arm": "B", "age": "30", "stage": "II"},
                     {"id": "4", "arm": "B", "age": "", "stage": "II"}]

    def test_missing_denominators_zero_levels_and_quantiles(self):
        result = baseline_table(self.rows, self.schema)
        age = next(r for r in result["rows"] if r["variable"] == "age" and r["group"] == "Overall")
        self.assertEqual((age["n_observed"], age["n_missing"], age["median"], age["q1"], age["q3"], age["sd"]), (3, 1, 20, 15, 25, 10))
        stage = next(r for r in result["rows"] if r["variable"] == "stage" and r["group"] == "arm=A" and r["level"] == "I")
        self.assertEqual((stage["count"], stage["n_observed"], stage["percent"]), (1, 1, 100))
        self.assertEqual(len([r for r in result["rows"] if r["variable"] == "stage" and r["level"] == "III" and r["count"] == 0]), 3)

    def test_rejects_nonfinite_unknown_category_missing_group_and_duplicate_id(self):
        for column, value in (("age", "Inf"), ("age", "bad"), ("stage", "IV"), ("arm", ""), ("id", "2")):
            with self.subTest(column=column, value=value):
                rows = [dict(r) for r in self.rows]
                rows[0][column] = value
                with self.assertRaises(ValueError):
                    baseline_table(rows, self.schema)

    def test_empty_and_single_observation_are_not_zero_sd(self):
        result = baseline_table(self.rows[:1], self.schema)
        age_rows = [r for r in result["rows"] if r["variable"] == "age"]
        self.assertTrue(all(r["sd"] is None for r in age_rows))
        self.assertIsNone(next(r for r in age_rows if r["group"] == "arm=B")["mean"])

    def test_failed_rerun_invalidates_pass_and_export_formats_escape(self):
        with test_tempdir() as folder:
            root = Path(folder)
            source, schema, output = root/"data.csv", root/"schema.json", root/"table"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            self.schema["variables"][0]["label"] = "Age <test> & _ %"
            schema.write_text(json.dumps(self.schema), encoding="utf-8")
            status = generate_baseline(source, schema, output)
            self.assertEqual(status["status"], "PASS")
            self.assertTrue(all((output/name).exists() for name in status["artifacts"]))
            self.assertIn("&lt;test&gt;", (output/"table1.html").read_text(encoding="utf-8"))
            self.assertIn(r"\& \_ \%", (output/"table1.tex").read_text(encoding="utf-8"))
            source.write_text("age\n1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                generate_baseline(source, schema, output)
            self.assertEqual(json.loads((output/"report.json").read_text(encoding="utf-8"))["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
