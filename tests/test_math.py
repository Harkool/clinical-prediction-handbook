"""Reader-facing numerical contracts; run with python -m unittest discover -s tests."""
import ast
import contextlib
import io
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


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_functions(filename, names, namespace):
    """Exercise the actual teaching functions without running Monte Carlo sections."""
    source = ROOT / "math" / filename
    selected = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("Missing mathematical function under test")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"),namespace)
    return namespace


class CatalogContract(unittest.TestCase):
    def test_every_mathematical_chapter_has_an_entry(self):
        self.assertEqual({entry['chapter'] for entry in CATALOG}, set(range(21,32)))
        self.assertEqual(len({entry['id'] for entry in CATALOG}), 16)
        for entry in CATALOG:
            source = ROOT / entry['script']
            self.assertTrue(source.is_file(), entry['id'])
            self.assertIn('result.json',entry['expected_artifacts'])
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


class AuditFailureContracts(unittest.TestCase):
    def test_old_pass_cannot_survive_missing_wrong_or_failed_child_report(self):
        audit = load_module("cph_math_audit_test", ROOT / "math" / "_audit.py")
        parent = ROOT / "outputs" / "audit-math-v101" / "test_audit"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            working = Path(directory)
            cases = {
                "nonzero": "raise SystemExit(2)",
                "no_report": "pass",
                "wrong_mode": "import json;open('result.json','w').write(json.dumps({'id':'fixture','mode':'book','checks':{'ok':True}}))",
                "wrong_id": "import json;open('result.json','w').write(json.dumps({'id':'other','mode':'quick','checks':{'ok':True}}))",
                "not_boolean": "import json;open('result.json','w').write(json.dumps({'id':'fixture','mode':'quick','checks':{'ok':1}}))",
                "timeout": "import time;time.sleep(5)",
            }
            for name, text in cases.items():
                with self.subTest(name=name):
                    script = working / (name+".py")
                    script.write_text(text,encoding="utf-8")
                    out = working / name
                    target = out / "fixture"
                    target.mkdir(parents=True)
                    report = target / "result.json"
                    report.write_text(json.dumps(dict(id="fixture",mode="quick",checks={"old":True},status="PASS")),encoding="utf-8")
                    with contextlib.redirect_stdout(io.StringIO()):
                        status = audit.execute(dict(id="fixture",script=str(script)),"quick",out,timeout=.1 if name=="timeout" else 5)
                    result = json.loads(report.read_text(encoding="utf-8"))
                    self.assertEqual(status['status'],"FAIL")
                    self.assertEqual(result['status'],"FAIL")
                    self.assertIsNone(result['checks'])


@unittest.skipUnless(HAS_SCIENTIFIC_PACKAGES, "Install requirements.txt to test mathematical boundaries")
class MathematicalBoundaryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np=np
        cls.support=load_module("cph_math_support_test",ROOT / "math" / "_lab_support.py")

    def test_acceptance_cannot_coerce_nan_or_nonboolean_to_true(self):
        for checks in ({}, {"bad":float('nan')}, {"bad":1}, {"bad":"PASS"}):
            with self.subTest(checks=checks), self.assertRaises((ValueError,TypeError)):
                self.support.write_result("test",checks)

    def test_ipcw_rejects_invalid_time_cause_boundary_and_survival_steps(self):
        np=self.np
        functions=load_functions('ch24b.py',('G','ipcw_weights','competing_brier'),dict(np=np))
        function=functions['ipcw_weights']
        for time,cause,horizon,steps in (
                ([1,float('nan')],[1,2],3,[]),([],[],3,[]),([1,2],[1],3,[]),
                ([1,2],[1,3],3,[]),([1,2],[1,2],float('nan'),[]),
                ([1,2],[1,2],3,[(0,1.1)]),([1,2],[1,2],3,[(1,.5),(2,.8)]),
                ([1,2],[1,0],2,[(2,.5)])):
            with self.subTest(time=time,cause=cause,horizon=horizon),self.assertRaises(ValueError):
                function(time,cause,horizon,steps)
        self.assertAlmostEqual(function([1,2],[1,2],3,[]).sum(),2)
        with self.assertRaises(ValueError):
            functions['competing_brier']([1,2],[1,2],3,[.2,float('nan')],[])

    def test_irls_and_firth_iteration_limits_cannot_return_unfinished_fit(self):
        np=self.np
        y=np.r_[np.ones(2),np.zeros(8),np.ones(6),np.zeros(4)]
        X=np.c_[np.ones(20),np.r_[np.zeros(10),np.ones(10)]]
        functions=load_functions('ch22.py',('sigmoid','fit_logit'),dict(np=np,check_logistic_solution=self.support.check_logistic_solution))
        with self.assertRaises(RuntimeError):
            functions['fit_logit'](X,y,it=1)
        firth=load_functions('ch21.py',('sigmoid','firth'),dict(np=np))['firth']
        with self.assertRaises(RuntimeError):
            firth(X,y,maxit=1)
        with self.assertRaises(ValueError):
            firth(np.c_[X,X[:,1]],y)
        with self.assertRaises(ValueError):
            self.support.check_logistic_solution(X,y,np.zeros(2),sample_weight=np.full(20,-1.))

    def test_constant_neural_network_prediction_has_no_calibration_slope(self):
        np=self.np
        functions=load_functions('ch26b.py',('sigmoid','fit_logit','slope'),dict(np=np,check_logistic_solution=self.support.check_logistic_solution,CALIBRATION_LP_SD_RELATIVE_TOLERANCE=1e-6))
        self.assertIsNone(functions['slope'](np.full(20,.2),np.r_[np.ones(4),np.zeros(16)]))
        self.assertIsNone(functions['slope'](np.full(20,.2)+np.linspace(-1e-9,1e-9,20),np.r_[np.ones(4),np.zeros(16)]))
        prediction=np.r_[np.full(10,.2),np.full(10,.6)]
        outcome=np.r_[np.ones(2),np.zeros(8),np.ones(6),np.zeros(4)]
        self.assertAlmostEqual(functions['slope'](prediction,outcome),1.,places=7)


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
