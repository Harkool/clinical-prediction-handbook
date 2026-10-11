"""The sampling demonstration must compare the same fit before and after correction."""
import ast
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit


SOURCE = Path(__file__).absolute().parents[1] / "math" / "ch31.py"


def load_selected(names, namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class PriorCorrectionContracts(unittest.TestCase):
    def test_exact_shift_preserves_coefficients_scores_and_input(self):
        fn = load_selected({"prior_correct"}, {"np": np})["prior_correct"]
        original = np.array([0.4, 0.8, -0.3])
        frozen = original.copy()
        corrected = fn(original, 0.1, 0.5)
        self.assertAlmostEqual(corrected[0] - original[0], np.log(0.1 / 0.9))
        np.testing.assert_array_equal(original, frozen)
        np.testing.assert_array_equal(corrected[1:], original[1:])
        inputs = np.array([[1., 0., 0.], [1., 1., 0.], [1., -1., 2.]])
        np.testing.assert_array_equal(np.argsort(expit(inputs @ original)),
                                      np.argsort(expit(inputs @ corrected)))

    def test_correction_requires_existing_fit_and_cannot_draw_again(self):
        class NoDraw:
            def choice(self, *args, **kwargs):
                raise AssertionError("Correction attempted to resample")
        ns = load_selected({"prior_correct", "develop"}, {"np": np, "rng": NoDraw()})
        X = np.zeros((10, 2)); y = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        with self.assertRaises(ValueError):
            ns["develop"](X, y, "过采样 + 先验校正")
        b = np.array([0.4, 0.8, -0.3])
        corrected = ns["develop"](X, y, "过采样 + 先验校正", b)
        np.testing.assert_array_equal(corrected[1:], b[1:])

    def test_invalid_prevalence_cannot_create_infinite_intercept(self):
        fn = load_selected({"prior_correct"}, {"np": np})["prior_correct"]
        for prevalence in [0., 1., -0.1, float("nan")]:
            with self.subTest(prevalence=prevalence), self.assertRaises(ValueError):
                fn([0.], prevalence)


if __name__ == "__main__":
    unittest.main()
