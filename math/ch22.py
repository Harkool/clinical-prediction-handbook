from _lab_support import begin, budget, write_result, check_logistic_solution
LAB_MODE = begin('22')
# Experiment M22; independent synthetic method demonstration, not main case A.
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge, Lasso, ElasticNet
np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(22)

def S(z, g): return np.sign(z)*np.maximum(np.abs(z)-g, 0)

# ---------- A: orthonormal design closed forms ----------
print("== A: orthonormal design, (1/n)X'X = I ==")
n, p = 40, 3
Q,_ = np.linalg.qr(rng.normal(size=(n,p)))
X = Q*np.sqrt(n)                    # columns orthogonal, (1/n) x_j'x_j = 1
z = np.array([3.0, 1.0, 0.4])       # OLS coefficients
y = X@z                              # y in column space => OLS = z exactly
print("check (1/n)X'y =", X.T@y/n)
lam, alpha = 0.5, 0.5
ridge = z/(1+lam)
lasso = S(z, lam)
enet = S(z, lam*alpha)/(1+lam*(1-alpha))
print("ridge  closed form:", ridge)
print("lasso  closed form:", lasso)
print("enet   closed form:", enet, "(alpha=0.5)")
# sklearn (fit_intercept False): Ridge alpha = n*lam ; Lasso alpha = lam ; ElasticNet alpha=lam, l1_ratio=alpha
print("sklearn Ridge  :", Ridge(alpha=n*lam, fit_intercept=False).fit(X,y).coef_)
print("sklearn Lasso  :", Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X,y).coef_)
print("sklearn ENet   :", ElasticNet(alpha=lam, l1_ratio=alpha, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X,y).coef_)

# ---------- B: ridge bias-variance (1-d orthonormal) ----------
print("\n== B: ridge MSE in 1-d (d=1, sigma^2=1) ==")
for beta in (1.0, 0.3):
    print(" beta =", beta, " optimum lambda = sigma^2/beta^2 =", 1/beta**2, " MSE* = sigma^2 beta^2/(beta^2+sigma^2) =", beta**2/(beta**2+1))
    for lam_ in (0, 0.5, 1, 2, 4, 1/beta**2):
        mse = (lam_**2*beta**2 + 1)/(1+lam_)**2
        bias = -lam_*beta/(1+lam_); var = 1/(1+lam_)**2
        print(f"   lambda={lam_:7.3f}  bias={bias: .4f}  var={var:.4f}  MSE={mse:.4f}")

# derivative at 0
print(" d MSE/d lambda at 0 (d=1) =", -2*1/1**4)

# (C and D of the first draft were superseded by ch22b.py)

# ---------- E: logistic: heuristic shrinkage vs actual slope in low-C / high-C scenarios ----------
print("\n== E: logistic, S_VH vs actual slope ==")
def sigmoid(z): return 1/(1+np.exp(-z))
def fit_logit(X, y, it=50, ridge=0.0):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = sigmoid(X@b); w = p*(1-p)
        H = X.T@(w[:,None]*X) + ridge*np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T@(y-p) - ridge*b)
        b = b + step
        if np.max(np.abs(step))<1e-10: break
    check_logistic_solution(X, y, b, ridge=ridge)
    return b
def loglik(b,X,y):
    eta=X@b; return np.sum(y*eta-np.log1p(np.exp(eta)))
def auc(lp, y):
    from scipy.stats import rankdata
    r = rankdata(lp); n1 = y.sum(); n0 = len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def scenario(n, p, bj, b0, reps=budget('logistic_repetitions',400,20), ntest=budget('logistic_test_n',60000,8000), seed=0):
    rg = np.random.default_rng(seed)
    beta = np.full(p, bj)
    Xt = rg.normal(size=(ntest,p)); yt = (rg.random(ntest) < sigmoid(b0+Xt@beta)).astype(float)
    Xtc = np.c_[np.ones(ntest), Xt]
    # population C of the true model
    Cpop = auc(Xt@beta, yt)
    Sopt=[]; Svh=[]; ev=[]
    for _ in range(reps):
        X = rg.normal(size=(n,p)); y = (rg.random(n) < sigmoid(b0+X@beta)).astype(float)
        if y.sum()<5 or y.sum()>n-5: continue
        Xc = np.c_[np.ones(n), X]
        b = fit_logit(Xc, y)
        b00 = fit_logit(np.ones((n,1)), y)
        LR = 2*(loglik(b,Xc,y)-loglik(b00,np.ones((n,1)),y))
        Svh.append(1-p/LR); ev.append(y.sum())
        lp_t = Xtc@b
        bs = fit_logit(np.c_[np.ones(ntest), lp_t], yt)
        Sopt.append(bs[1])
    return Cpop, np.mean(ev), np.mean(Sopt), np.mean(Svh), np.median(Sopt), np.median(Svh)
for name,(n,p,bj,b0) in {"low-C ":(300,10,0.10,-1.3), "mid-C ":(300,10,0.25,-1.3),"high-C":(300,10,0.55,-1.6)}.items():
    Cpop, ev, so, sv, mso, msv = scenario(n,p,bj,b0)
    print(f" {name}: n={n}, p={p}, C_pop(true lp)={Cpop:.3f}, mean events={ev:.1f}  mean S_opt(true slope)={so:.3f}  mean S_VH={sv:.3f}  bias(S_VH-S_opt)={sv-so: .3f}   [medians {mso:.3f} / {msv:.3f}]")

# Repository acceptance checks use exact identities where possible.
write_result('22', {'ridge_orthogonal_solution': np.max(np.abs(Ridge(alpha=X.shape[0]*.5,fit_intercept=False).fit(X,y).coef_-ridge)) < 1e-9, 'lasso_orthogonal_solution': np.max(np.abs(Lasso(alpha=.5,fit_intercept=False,tol=1e-12).fit(X,y).coef_-lasso)) < 1e-9}, {'ridge_coefficients': ridge.tolist(), 'lasso_coefficients': lasso.tolist()})
