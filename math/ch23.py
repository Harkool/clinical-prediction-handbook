from _lab_support import begin, budget, write_result, check_logistic_solution
LAB_MODE = begin('23')
# Experiment M23; independent synthetic method demonstration, not main case A.
import numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.stats import chi2
np.set_printoptions(precision=5, suppress=True, linewidth=150)
rng = np.random.default_rng(23)

# ---------- A: restricted cubic spline basis ----------
def pos3(u): return np.maximum(u,0)**3
def rcs_basis(x, t):
    """Harrell-form RCS: returns columns [x, h_1..h_{k-2}] (k-1 columns), unscaled."""
    t = np.asarray(t, float); k = len(t)
    cols = [x]
    for j in range(k-2):
        hj = (pos3(x-t[j])
              - pos3(x-t[k-2])*(t[k-1]-t[j])/(t[k-1]-t[k-2])
              + pos3(x-t[k-1])*(t[k-2]-t[j])/(t[k-1]-t[k-2]))
        cols.append(hj)
    return np.column_stack(cols)

t = np.array([2., 4., 6., 8.])          # k = 4
xs = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 12.])
B = rcs_basis(xs, t)
print("== A: RCS basis, k=4 knots at", t, "-> columns: x, h1, h2 (k-1 = 3)")
print("   x      h1        h2")
for xv, row in zip(xs, B): print(f"  {xv:4.0f} {row[1]:9.4f} {row[2]:9.4f}")
# tail checks: h_j = 0 for x<=t1 ; second difference ~0 for x>=t_k
xx = np.linspace(8,20,13); Bt = rcs_basis(xx,t)
print(" second differences of h1,h2 beyond t_k (should be 0):", np.abs(np.diff(Bt[:,1:],2,axis=0)).max())
xx = np.linspace(-5,2,8); print(" h1,h2 at x<=t1 (should be 0):", np.abs(rcs_basis(xx,t)[:,1:]).max())
# derivation check: truncated power basis with constraints -> k-1 free params
# unrestricted: [1,x,x^2,x^3,(x-t_j)+^3 j=1..k]  => k+4 columns ; constraints: left tail 2, right tail 2
k=len(t)
print(" unrestricted cubic spline columns (with intercept) =", 4+k, "; minus 4 constraints =", 4+k-4, "; minus intercept =", k-1)
# sum condition used in derivation: for h_j, coefficients of (x-t_j)^3, (x-t_{k-1})^3, (x-t_k)^3 sum to 0 and sum of coef*knots = 0
for j in range(k-2):
    c = np.array([1, -(t[k-1]-t[j])/(t[k-1]-t[k-2]), (t[k-2]-t[j])/(t[k-1]-t[k-2])]); kn = np.array([t[j], t[k-2], t[k-1]])
    print(f"  h_{j+1}: sum of cubic coefs = {c.sum():.12f} ; sum coef*knot = {np.dot(c,kn):.12f}")

# ---------- B: parameter counting illustration ----------
print("\n== B: parameter count: RCS with k knots has k-1 params; k=3,4,5 ->", [kk-1 for kk in (3,4,5)])

# ---------- C: penalized spline (P-spline) effective degrees of freedom ----------
print("\n== C: P-spline EDF (cubic B-spline basis K=13, 2nd-order difference penalty), n=80 ==")
from scipy.interpolate import BSpline
n = 80
x = np.sort(rng.uniform(0,1,n))
ytrue = np.sin(2*np.pi*x) + 0.5*x
y = ytrue + rng.normal(0,0.35,n)
deg=3; nint=10                      # 10 intervals, 9 interior knots, clamped ends -> K=13
kn = np.r_[[0]*deg, np.linspace(0,1,nint+1), [1]*deg]
K = len(kn)-deg-1
Bm = np.column_stack([BSpline(kn, np.eye(K)[i], deg, extrapolate=False)(np.clip(x,0,1-1e-12)) for i in range(K)])
D = np.diff(np.eye(K), 2, axis=0); P = D.T@D
BtB = Bm.T@Bm
print(" K =", K, " null space dim of penalty =", K-np.linalg.matrix_rank(P))
print("  lambda      EDF      GCV      RSS")
rows=[]
for lam in [0, 0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e6]:
    A = np.linalg.solve(BtB + lam*P + 1e-10*np.eye(K), BtB)    # F matrix
    edf = np.trace(A)
    fit = Bm@np.linalg.solve(BtB + lam*P + 1e-10*np.eye(K), Bm.T@y)
    rss = np.sum((y-fit)**2)
    gcv = n*rss/(n-edf)**2
    rows.append((lam,edf,gcv,rss))
    print(f" {lam:9.2g} {edf:8.3f} {gcv:8.4f} {rss:8.3f}")
best = min(rows, key=lambda r:r[2]); print(" GCV minimum at lambda =", best[0], " EDF =", round(best[1],3))
# finer search
lams = np.logspace(-3,6,400);
def gcv_of(l):
    A = np.linalg.solve(BtB + l*P + 1e-10*np.eye(K), BtB); edf=np.trace(A)
    fit = Bm@np.linalg.solve(BtB + l*P + 1e-10*np.eye(K), Bm.T@y); return n*np.sum((y-fit)**2)/(n-edf)**2, edf
g = [gcv_of(l) for l in lams]; i = int(np.argmin([a for a,_ in g]))
print(" fine search: lambda* = %.3g  EDF* = %.3f  GCV* = %.4f"%(lams[i], g[i][1], g[i][0]))
# RCS-4 fit for comparison
Xr = np.c_[np.ones(n), rcs_basis(x, np.quantile(x,[.05,.35,.65,.95]))]
br = np.linalg.lstsq(Xr,y,rcond=None)[0]; rss_r = np.sum((y-Xr@br)**2)
print(" RCS(4 knots) uses 4 params (incl. intercept): RSS = %.3f ; linear: RSS = %.3f"%(rss_r, np.sum((y-np.polyval(np.polyfit(x,y,1),x))**2)))

# Nullspace is linear in the COEFFICIENT INDEX, not necessarily in x.
# Clamped end knots here make the endpoint spacing nonuniform.
index = np.arange(K, dtype=float)
null_basis = Bm @ np.c_[np.ones(K), index]
null_fit = null_basis @ np.linalg.lstsq(null_basis, y, rcond=None)[0]
line_basis = np.c_[np.ones(n), x]
line_fit = line_basis @ np.linalg.lstsq(line_basis, y, rcond=None)[0]
greville = np.array([np.mean(kn[j+1:j+deg+1]) for j in range(K)])
index_curve = BSpline(kn, index, deg)
assert np.max(np.abs(D @ index)) == 0
assert abs(index_curve.derivative(2)(0.025) + 237.5) < 1e-10
assert np.max(np.abs(BSpline(kn, greville, deg)(x) - x)) < 1e-12
assert np.max(np.abs(D @ greville)) > .03
print(" nullspace-limit RSS = %.6f ; ordinary-line RSS = %.6f ; max fitted difference = %.6f" %
      (np.sum((y-null_fit)**2), np.sum((y-line_fit)**2), np.max(np.abs(null_fit-line_fit))))
print(" D@index =", D@index, "; index-curve f''(0.025) =", index_curve.derivative(2)(0.025))
print(" coefficients representing f(x)=x (Greville):", greville,
      "; their second differences:", D @ greville)
# An intrinsic prior is flat on ker(P). A normal with covariance pinv(P)
# instead fixes its nullspace components to zero, so it is not that prior.
constant_direction = np.ones(K) / np.sqrt(K)
degenerate_variance = float(constant_direction @ np.linalg.pinv(P) @ constant_direction)
constant_fit = Bm @ np.linalg.solve(BtB + P, Bm.T @ np.ones(n))
assert abs(degenerate_variance) < 1e-10
assert np.max(np.abs(constant_fit - 1)) < 1e-10
print(" pinv(P) normal variance in constant direction = %.3g ; penalized constant fit error = %.3g" %
      (degenerate_variance, np.max(np.abs(constant_fit-1))))

# ---------- D: LR test of nonlinearity in logistic regression ----------
NULLSPACE_RSS = float(np.sum((y-null_fit)**2))
ORDINARY_LINE_RSS = float(np.sum((y-line_fit)**2))
print("\n== D: logistic: linear vs RCS(4) likelihood-ratio test ==")
def sigmoid(z): return 1/(1+np.exp(-z))
def fit_logit(X, y, it=100):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = sigmoid(X@b); w = p*(1-p)
        step = np.linalg.solve(X.T@(w[:,None]*X)+1e-10*np.eye(X.shape[1]), X.T@(y-p)); b+=step
        if np.max(np.abs(step))<1e-10: break
    check_logistic_solution(X, y, b)
    return b
def ll(b,X,y):
    e=X@b; return np.sum(y*e-np.log1p(np.exp(e)))
for nn in (300, 1200):
    res=[]
    for rep in range(1):
        x = rng.normal(size=nn)*1.0
        lp_true = -2.0 + 0.9*x + 0.45*x**2 - 0.0*x**3     # U-ish shape
        y = (rng.random(nn)<sigmoid(lp_true)).astype(float)
        X1 = np.c_[np.ones(nn), x]
        X2 = np.c_[np.ones(nn), rcs_basis(x, np.quantile(x,[.05,.35,.65,.95]))]
        b1=fit_logit(X1,y); b2=fit_logit(X2,y)
        LR = 2*(ll(b2,X2,y)-ll(b1,X1,y))
        print(f" n={nn}: events={int(y.sum())}, LR chi2(2) = {LR:.2f}, p = {chi2.sf(LR,2):.4g}")
# power of the LR test for nonlinearity at different n (200 reps)
def power(nn, reps=budget('nonlinearity_repetitions',400,40), curv=0.30):
    rej=0
    for _ in range(reps):
        x = rng.normal(size=nn)
        y = (rng.random(nn)<sigmoid(-2.0+0.9*x+curv*x**2)).astype(float)
        X1 = np.c_[np.ones(nn), x]; X2 = np.c_[np.ones(nn), rcs_basis(x, np.quantile(x,[.05,.35,.65,.95]))]
        try:
            LR = 2*(ll(fit_logit(X2,y),X2,y)-ll(fit_logit(X1,y),X1,y))
        except Exception: continue
        rej += (chi2.sf(LR,2)<0.05)
    return rej/reps
print(" power of 2-df LR nonlinearity test (curvature 0.30*x^2 in logit):", {nn: power(nn) for nn in (200,500,1000,3000)})

# Repository acceptance checks use exact identities where possible.
write_result('23', {'rcs_tail_is_linear': np.abs(np.diff(Bt[:,1:],2,axis=0)).max() < 1e-10, 'penalty_nullspace_is_not_ordinary_x_line': np.max(np.abs(null_fit-line_fit)) > .04, 'intrinsic_prior_differs_from_pseudoinverse_normal': abs(degenerate_variance) < 1e-10 and np.max(np.abs(constant_fit-1)) < 1e-10}, {'nullspace_rss': NULLSPACE_RSS, 'ordinary_line_rss': ORDINARY_LINE_RSS})
