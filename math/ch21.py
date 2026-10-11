from _lab_support import begin, budget, write_result
LAB_MODE = begin('21')
# Experiment M21; independent synthetic method demonstration, not main case A.
import numpy as np
from sklearn.linear_model import LogisticRegression
np.set_printoptions(precision=6, suppress=True)

def sigmoid(z): return 1/(1+np.exp(-z))

def loglik(beta, X, y):
    eta = X@beta
    return np.sum(y*eta - np.log1p(np.exp(eta)))

# ---------- Example A: 2x2 table, IRLS ----------
# x=0: 10 pts, 2 events ; x=1: 10 pts, 6 events
x = np.r_[np.zeros(10), np.ones(10)]
y = np.r_[np.r_[np.ones(2), np.zeros(8)], np.r_[np.ones(6), np.zeros(4)]]
X = np.c_[np.ones(20), x]
print("== A: IRLS on 2x2 table (x=0: 2/10 events; x=1: 6/10) ==")
beta = np.zeros(2)
for it in range(0, 7):
    eta = X@beta; p = sigmoid(eta); w = p*(1-p)
    U = X.T@(y-p)                      # score
    I = X.T@(w[:,None]*X)              # information
    ll = loglik(beta, X, y)
    print(f"iter {it}: beta=({beta[0]: .6f},{beta[1]: .6f})  loglik={ll: .6f}  score=({U[0]: .6f},{U[1]: .6f})")
    if it==0:
        print("   info at 0 =\n", I, "\n   inv(I)@U =", np.linalg.solve(I,U))
    z = eta + (y-p)/w                  # working response
    beta = np.linalg.solve(I, X.T@(w*z))   # IRLS step = Newton step
eta = X@beta; p = sigmoid(eta); w = p*(1-p)
I = X.T@(w[:,None]*X); cov = np.linalg.inv(I)
print("final beta", beta, " SE", np.sqrt(np.diag(cov)))
print("closed form: logOR =", np.log((6/4)/(2/8)), " Woolf SE =", np.sqrt(1/6+1/4+1/2+1/8), " intercept logit(0.2) =", np.log(0.25))
print("sum p =", p.sum(), " sum y =", y.sum())
lp = X@beta
# apparent calibration slope: regress y on lp
m = LogisticRegression(penalty=None, tol=1e-12, max_iter=5000).fit(lp.reshape(-1,1), y)
print("apparent recalibration: intercept", m.intercept_, "slope", m.coef_)
sk = LogisticRegression(penalty=None, tol=1e-12, max_iter=5000).fit(x.reshape(-1,1), y)
print("sklearn:", sk.intercept_, sk.coef_)

# ---------- Example B: quasi-complete separation ----------
print("\n== B: separation (x=0: 2/10 events; x=1: 10/10 events) ==")
yB = np.r_[np.r_[np.ones(2), np.zeros(8)], np.ones(10)]
beta = np.zeros(2)
for it in range(0, 13):
    eta = X@beta; p = sigmoid(eta); w = np.maximum(p*(1-p),1e-300)
    I = X.T@(w[:,None]*X)
    ll = loglik(beta, X, yB)
    try:
        se1 = np.sqrt(np.linalg.inv(I)[1,1])
    except Exception:
        se1 = np.inf
    print(f"iter {it:2d}: beta1={beta[1]: 9.4f}  loglik={ll: .6f}  SE(beta1)={se1: .3f}  Wald z={beta[1]/se1 if np.isfinite(se1) else 0: .3f}")
    z = eta + (yB-p)/w
    beta = np.linalg.solve(I, X.T@(w*z))
print("limit of loglik as beta1->inf:", 2*np.log(0.2)+8*np.log(0.8))

# ---------- Firth ----------
def firth(X, y, maxit=200, tol=1e-10):
    X, y = np.asarray(X, float), np.asarray(y, float)
    if (X.ndim != 2 or y.shape != (len(X),) or not len(X) or
            not np.all(np.isfinite(X)) or not np.all(np.isin(y, [0, 1])) or
            np.linalg.matrix_rank(X) != X.shape[1] or maxit < 1 or tol <= 0):
        raise ValueError("Firth demonstration requires a finite full-rank design and binary outcomes")
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = X@b; p = sigmoid(eta); w = p*(1-p)
        I = X.T@(w[:,None]*X); Iinv = np.linalg.inv(I)
        # hat diag: h_i = w_i x_i' Iinv x_i
        h = w*np.einsum('ij,jk,ik->i', X, Iinv, X)
        Ustar = X.T@(y - p + h*(0.5-p))
        step = Iinv@Ustar
        # Cap the Newton step; this is not an objective-based line search.
        mx = np.max(np.abs(step))
        if mx>5: step *= 5/mx
        b = b + step
        if np.max(np.abs(step))<tol:
            p = sigmoid(X@b); w = p*(1-p)
            final_Iinv = np.linalg.inv(X.T@(w[:,None]*X))
            h = w*np.einsum('ij,jk,ik->i', X, final_Iinv, X)
            if np.max(np.abs(X.T@(y-p+h*(.5-p)))) > 1e-7:
                raise RuntimeError("Firth adjusted score did not converge")
            return b, final_Iinv
    raise RuntimeError("Firth demonstration reached its iteration limit without convergence")

print("\n== Firth on example B ==")
bF, IinvF = firth(X, yB)
pF = sigmoid(X@bF)
print("Firth beta:", bF, " SE:", np.sqrt(np.diag(IinvF)))
# compare with adding 0.5 to each cell
b0 = np.log((2.5/8.5)); b1 = np.log((10.5/0.5)/(2.5/8.5))
print("add-0.5 per cell: intercept", b0, " beta1", b1)
print("Firth predicted risks: x=0:", pF[0], " x=1:", pF[-1], "  observed 0.2 and 1.0")

print("\n== Firth on example A (no separation) ==")
bFA,_ = firth(X, y)
print("Firth beta:", bFA, " ML beta:", beta if False else np.array([np.log(0.25), np.log(6)]))

# ---------- Rare-event example: average predicted risk too high ----------
print("\n== Rare events: x=0: 1/100 ; x=1: 3/100 ==")
xr = np.r_[np.zeros(100), np.ones(100)]
yr = np.r_[np.r_[np.ones(1), np.zeros(99)], np.r_[np.ones(3), np.zeros(97)]]
Xr = np.c_[np.ones(200), xr]
bFr,_ = firth(Xr, yr)
pFr = sigmoid(Xr@bFr)
print("Firth beta", bFr, "; mean predicted", pFr.mean(), " observed", yr.mean(), " sum pred", pFr.sum(), " events", yr.sum())
print("pred x=0:", pFr[0], " pred x=1:", pFr[-1], " (add-0.5: 1.5/101=%.5f, 3.5/101=%.5f)"%(1.5/101,3.5/101))
# FLIC: re-estimate intercept with beta fixed so sum p = events
from scipy.optimize import brentq
lpr = Xr[:,1:]*bFr[1]
a = brentq(lambda a: sigmoid(a+lpr).sum()-yr.sum(), -10, 5)
print("FLIC intercept", a, " vs Firth intercept", bFr[0], " mean pred after", sigmoid(a+lpr).mean())


# Repository acceptance checks use exact identities where possible.
write_result('21', {'irls_equals_2x2_log_or': abs(sk.coef_[0,0]-np.log(6)) < 1e-7, 'firth_equals_half_cell_correction': np.max(np.abs(bF-[b0,b1])) < 1e-7, 'flic_restores_observed_mean': abs(sigmoid(a+lpr).mean()-yr.mean()) < 1e-10}, {'log_odds_ratio': float(sk.coef_[0,0]), 'flic_mean_risk': float(sigmoid(a+lpr).mean())})
