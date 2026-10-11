"""Small numerical companion to the book's standalone linear algebra primer."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit


def run_foundations(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = output/"result.json"
    report.write_text('{"status":"RUNNING"}\n', encoding="utf-8")
    try:
        x = np.array([1., 2., -1.])
        beta = np.array([-.5, .3, .2])
        eta = float(x@beta)
        dependent = np.array([[1.,1.,2.],[1.,2.,4.],[1.,3.,6.]])
        X = np.column_stack([np.ones(4), np.arange(4)])
        y = np.array([1.,2.,2.,4.])
        coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y-X@coefficients
        gram = X.T@X
        eigenvalues = np.linalg.eigvalsh(gram)
        ridge = np.linalg.solve(gram+np.diag([0.,1.]), X.T@y)
        weights = np.array([.2,.21,.24,.25])
        information = X.T@(weights[:,None]*X)
        direction = np.array([.4,-.3])
        quadratic = float(direction@information@direction)
        direct = float(np.sum(weights*(X@direction)**2))
        checks = {"linear_predictor_by_hand":bool(np.isclose(eta,-.1)),
                  "dependent_matrix_rank_two":bool(np.linalg.matrix_rank(dependent)==2),
                  "OLS_residual_orthogonal":bool(np.allclose(X.T@residual,0.,atol=1e-12)),
                  "information_quadratic_identity":bool(np.isclose(quadratic,direct)),
                  "positive_information_eigenvalues":bool(np.all(np.linalg.eigvalsh(information)>0)),
                  "unpenalized_intercept_ridge_shrinks_slope":bool(abs(ridge[1])<abs(coefficients[1]))}
        if not all(checks.values()):
            raise ValueError("A primer identity did not reproduce")
        examples = [{"example":"linear predictor", "value":eta, "interpretation":"dot product is one scalar"},
                    {"example":"Logistic risk", "value":float(expit(eta)), "interpretation":"sigmoid of the scalar, not matrix inversion"},
                    {"example":"dependent matrix rank", "value":int(np.linalg.matrix_rank(dependent)), "interpretation":"third column equals twice the second"},
                    {"example":"OLS intercept", "value":float(coefficients[0]), "interpretation":"full column rank least-squares fit"},
                    {"example":"OLS slope", "value":float(coefficients[1]), "interpretation":"residual orthogonal to both design columns"},
                    {"example":"ridge slope", "value":float(ridge[1]), "interpretation":"lambda=1; intercept not penalized"}]
        with (output/"examples.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(examples[0]))
            writer.writeheader()
            writer.writerows(examples)
        result = {"status":"PASS", "scope":"linear algebra primer, independent of clinical case",
                  "checks":checks, "examples":examples,
                  "dimensions":{"X":[4,2],"beta":[2],"X_transpose_W_X":[2,2]},
                  "gram_eigenvalues":eigenvalues.tolist()}
        report.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return result
    except Exception as error:
        report.write_text(json.dumps({"status":"FAIL","error":str(error)})+"\n", encoding="utf-8")
        raise
