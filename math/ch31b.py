"""Experiment M31b: deterministic checks of conditioning and weight targets.

This supplements M31; it does not rerun or change the large synthetic case.
Run with NumPy/SciPy. The integrals use independent standard-normal L1,L2.
"""
from _lab_support import begin, budget, write_result
LAB_MODE = begin('31b')
import json

import numpy as np
from scipy.special import expit


def expected_logistic_fit(design, response_probability, mass):
    """Fit population expected log likelihood, with explicit score acceptance."""
    beta = np.zeros(design.shape[1])
    for iteration in range(100):
        probability = expit(design @ beta)
        score = design.T @ (mass * (response_probability - probability))
        information = design.T @ ((mass * probability * (1-probability))[:, None] * design)
        beta += np.linalg.solve(information, score)
        score_after = design.T @ (mass * (response_probability-expit(design @ beta)))
        if np.max(np.abs(score_after)) < 1e-12:
            break
    else:
        raise RuntimeError("Expected-log-likelihood fit did not pass score tolerance")
    if not np.all(np.isfinite(beta)):
        raise RuntimeError("Expected-log-likelihood fit returned nonfinite coefficients")
    return beta


def conditioning_check(order=80):
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    weights /= np.sqrt(2*np.pi)
    l1, l2 = np.meshgrid(nodes, nodes, indexing="ij")
    l1, l2 = l1.ravel(), l2.ravel()
    mass_l = np.outer(weights, weights).ravel()
    propensity = expit(.8*l1 + 1.2*l2)
    p0 = expit(-1 + .8*l1 + .8*l2)
    p1 = expit(-1 + .8*l1 + .8*l2 - 1.2)
    a = np.r_[np.zeros(len(l1)), np.ones(len(l1))]
    repeated_l1, repeated_l2 = np.tile(l1, 2), np.tile(l2, 2)
    mass = np.r_[mass_l*(1-propensity), mass_l*propensity]
    response = np.r_[p0, p1]
    beta_full = expected_logistic_fit(
        np.c_[np.ones(len(a)), repeated_l1, repeated_l2, a], response, mass)
    beta_xa = expected_logistic_fit(
        np.c_[np.ones(len(a)), repeated_l1, a], response, mass)
    assert np.max(np.abs(beta_full-[-1, .8, .8, -1.2])) < 1e-10
    # The decisive comparison is on the same risk scale and within the same X.
    # Coefficients conditioned on different variables need not be collapsible.
    e_at_x0 = expit(1.2*nodes)
    risks0, risks1 = expit(-1+.8*nodes), expit(-2.2+.8*nodes)
    intervention = np.array([np.dot(weights, risks0), np.dot(weights, risks1)])
    observed = np.array([
        np.dot(weights*(1-e_at_x0), risks0)/np.dot(weights, 1-e_at_x0),
        np.dot(weights*e_at_x0, risks1)/np.dot(weights, e_at_x0)])
    # Standardization uses the target L|X distribution; ordinary IPW recovers it.
    ipw_risks = np.array([
        np.dot(weights*(1-e_at_x0)/(1-e_at_x0), risks0),
        np.dot(weights*e_at_x0/e_at_x0, risks1)])
    assert np.max(np.abs(ipw_risks-intervention)) < 1e-12
    assert np.max(np.abs(intervention-observed)) > .04
    return dict(quadrature_order=order,
                full_L_logistic_coefficients=beta_full.tolist(),
                X_plus_A_logistic_coefficients=beta_xa.tolist(),
                intervention_risks_at_X0=intervention.tolist(),
                observed_risks_at_X0=observed.tolist(),
                intervention_risk_difference=float(intervention[1]-intervention[0]),
                observed_risk_difference=float(observed[1]-observed[0]),
                ipw_risks_at_X0=ipw_risks.tolist())


def stabilization_check():
    mass = np.array([.5, .5])
    e0 = np.array([.9, .1])
    risk0 = np.array([.1, .4])
    prediction = .25
    loss = risk0*(1-prediction)**2 + (1-risk0)*prediction**2
    untreated_mass = mass*e0
    target = np.dot(mass, loss)
    ordinary_ipw = np.dot(untreated_mass/e0, loss)
    # Here L=X, so e0(X)/e0(L)=1. Normalization retains treatment selection.
    x_stabilized = np.dot(untreated_mass, loss)/untreated_mass.sum()
    constant_stabilized_mass = untreated_mass*untreated_mass.sum()/e0
    constant_stabilized = np.dot(constant_stabilized_mass, loss)/constant_stabilized_mass.sum()
    assert np.isclose(target, .1875)
    assert np.isclose(ordinary_ipw, target)
    assert np.isclose(x_stabilized, .1275)
    assert np.isclose(constant_stabilized, target)
    return dict(stratum_mass=mass.tolist(), untreated_probability=e0.tolist(),
                risk_under_no_treatment=risk0.tolist(), prediction=prediction,
                target_brier=float(target), ordinary_ipw_brier=float(ordinary_ipw),
                X_stabilized_normalized_brier=float(x_stabilized),
                constant_stabilized_normalized_brier=float(constant_stabilized))


if __name__ == "__main__":
    conditioning = conditioning_check(80)
    coarser = conditioning_check(60)
    max_quad_change = max(
        np.max(np.abs(np.array(conditioning[key])-np.array(coarser[key])))
        for key in ("full_L_logistic_coefficients", "X_plus_A_logistic_coefficients",
                    "intervention_risks_at_X0", "observed_risks_at_X0"))
    assert max_quad_change < 1e-8
    result = dict(conditioning=conditioning, stabilization=stabilization_check(),
                  quadrature_60_vs_80_max_change=float(max_quad_change))
    print(json.dumps(result, ensure_ascii=False, indent=2))

# Repository acceptance checks use exact identities where possible.
write_result('31b', {'sufficient_adjustment_recovers_true_coefficients': np.max(np.abs(np.array(conditioning['full_L_logistic_coefficients'])-[-1,.8,.8,-1.2])) < 1e-10, 'quadrature_stable': max_quad_change < 1e-8, 'ordinary_ipw_targets_population': abs(result['stabilization']['ordinary_ipw_brier']-.1875) < 1e-12, 'x_stabilized_weights_change_target': abs(result['stabilization']['X_stabilized_normalized_brier']-.1275) < 1e-12}, result)
