"""Reader-facing numerical contracts; run with python -m unittest discover -s tests."""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).absolute().parents[1]
CATALOG = json.loads((ROOT / "math" / "catalog.json").read_text(encoding="utf-8"))
HAS_SCIENTIFIC_PACKAGES = all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "sklearn"))


class CatalogContract(unittest.TestCase):
    def test_every_mathematical_chapter_has_an_entry(self):
        self.assertEqual({entry['chapter'] for entry in CATALOG}, set(range(21,32)))
        self.assertEqual(len({entry['id'] for entry in CATALOG}), 16)
        for entry in CATALOG:
            source = ROOT / entry['script']
            self.assertTrue(source.is_file(), entry['id'])
            self.assertIn('result.json',entry['expected_artifacts'])
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


@unittest.skipUnless(HAS_SCIENTIFIC_PACKAGES, "Install requirements.txt to test the mathematical labs")
class NumericalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        environment = dict(os.environ, CPH_MODE="quick", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        temporary_parent = ROOT / "outputs" / "test_math"
        temporary_parent.mkdir(parents=True,exist_ok=True)
        # TemporaryDirectory cleanup remains inside this project's ignored outputs.
        temporary_parent.relative_to(ROOT)
        with tempfile.TemporaryDirectory(prefix="cph-math-",dir=str(temporary_parent)) as directory:
            for identifier in ("21", "24b", "26", "31b"):
                destination = Path(directory) / identifier
                destination.mkdir()
                completed = subprocess.run([sys.executable,str(ROOT / "math" / f"ch{identifier}.py")],cwd=destination,env=environment,capture_output=True,encoding="utf-8",timeout=90)
                if completed.returncode:
                    raise AssertionError(f"M{identifier} failed:\n{completed.stderr}")
                result = json.loads((destination / "result.json").read_text(encoding="utf-8"))
                if result['mode'] != 'quick' or not all(result['checks'].values()):
                    raise AssertionError(f"M{identifier} failed its declared numerical contracts")
                cls.results[identifier] = result

    def test_log_odds_matches_the_two_by_two_table(self):
        expected = math.log((6/4)/(2/8))
        self.assertAlmostEqual(self.results['21']['values']['log_odds_ratio'], expected, places=7)
        self.assertAlmostEqual(self.results['21']['values']['flic_mean_risk'], 4/200, places=10)

    def test_competing_event_is_a_known_negative_for_cause_one(self):
        values = self.results['24b']['values']
        self.assertAlmostEqual(values['correct_brier'][0],779/3600,places=12)
        self.assertAlmostEqual(values['correct_brier'][1],79/300,places=12)
        self.assertLess(values['correct_brier'][0],values['correct_brier'][1])
        self.assertGreater(values['wrong_brier'][0],values['wrong_brier'][1])

    def test_backpropagation_and_model_size(self):
        values = self.results['26']['values']
        self.assertLess(values['max_gradient_error'],1e-8)
        self.assertEqual(values['parameters_20_64_64_1'],(20+1)*64+(64+1)*64+(64+1))

    def test_stabilized_weight_numerator_changes_the_estimand(self):
        values = self.results['31b']['values']['stabilization']
        self.assertAlmostEqual(values['target_brier'],.1875,places=12)
        self.assertAlmostEqual(values['ordinary_ipw_brier'],.1875,places=12)
        self.assertAlmostEqual(values['X_stabilized_normalized_brier'],.1275,places=12)
        self.assertAlmostEqual(values['constant_stabilized_normalized_brier'],.1875,places=12)


if __name__ == "__main__":
    unittest.main()
