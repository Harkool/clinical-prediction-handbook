from _lab_support import begin, budget, write_result
LAB_MODE = begin('22b')
# Experiment M22b; independent synthetic method demonstration, not main case A.
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Lasso
np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(220)
def S(z,g): return np.sign(z)*np.maximum(np.abs(z)-g,0)

print("== C': coordinate descent with moments only: R=[[1,.6],[.6,1]], r=(.9,.7), lambda=.1 ==")
R = np.array([[1,.6],[.6,1]]); r = np.array([.9,.7]); lam=.1
b = np.zeros(2)
for s in range(1,9):
    for j in range(2):
        zj = r[j] - sum(R[j,k]*b[k] for k in range(2) if k!=j)
        b[j] = S(zj, lam)
    print(f"  sweep {s}: beta = ({b[0]:.6f}, {b[1]:.6f})")
# exact (both active, positive): R b = r - lam
print("  exact solution R^{-1}(r - lam) =", np.linalg.solve(R, r-lam))
# sklearn check: build X with (1/n)X'X = R and y with (1/n)X'y = r
n=50
Q,_ = np.linalg.qr(rng.normal(size=(n,2)))
L = np.linalg.cholesky(R)               # R = L L'
X = Q*np.sqrt(n) @ L.T                  # (1/n)X'X = L (Q'Q) L' = R
print("  check (1/n)X'X =\n", X.T@X/n)
c = np.linalg.solve(R, r)               # y = X c  =>  (1/n)X'y = R c = r
y = X@c
print("  sklearn Lasso(alpha=0.1):", Lasso(alpha=lam, fit_intercept=False, tol=1e-14, max_iter=200000).fit(X,y).coef_)

print("\n== D': linear Gaussian, Wald/F-type plug-in vs LR-type plug-in ==")
def sim_lin(n,p,b,reps=budget('linear_repetitions',6000,300)):
    beta = np.full(p,b); V=np.sum(beta**2)
    g=[];sLR=[];sF=[]
    for _ in range(reps):
        X = rng.normal(size=(n,p)); y = X@beta + rng.normal(size=n)
        Xc = np.c_[np.ones(n),X]
        bh = np.linalg.lstsq(Xc,y,rcond=None)[0]
        rss1=np.sum((y-Xc@bh)**2); rss0=np.sum((y-y.mean())**2)
        LR = n*np.log(rss0/rss1)
        F = ((rss0-rss1)/p)/(rss1/(n-p-1))
        g.append(beta@bh[1:]/(bh[1:]@bh[1:]))
        sLR.append(1-p/LR); sF.append(1-1/F)
    lam_nc=n*V
    return lam_nc/(lam_nc+p), np.mean(g), np.mean(sLR), np.mean(sF)
for (n,p,b) in [(60,10,0.30),(100,10,0.30),(200,10,0.30)]:
    f, g, a, c = sim_lin(n,p,b)
    print(f" n={n}: first-order {f:.4f} | MC true slope {g:.4f} | plug-in with LR: {a:.4f} | plug-in with F (Wald-type, 1-1/F): {c:.4f}")


# Exact counterexamples used in M22: duplication and nonmonotone support.
print("\nM22: duplicate ridge columns, z=1, lambda=1, k=2")
print("single coefficient", 1/2, "each duplicated coefficient", 1/3)
gram = np.array([[1., .9], [.9, 1.]])
cross = np.array([10., 5.])
support_counts = []
kkt_residuals = []
for alpha, coef in [(0., np.linalg.solve(gram+np.eye(2),cross)),
                    (.3, np.array([9.7/1.7, 0.])),
                    (1., np.linalg.solve(gram,np.array([9.,6.])) )]:
    gradient = gram@coef-cross+(1-alpha)*coef
    active = np.abs(coef)>1e-10
    assert np.max(np.abs(gradient[active]+alpha*np.sign(coef[active])))<1e-10
    assert np.all(np.abs(gradient[~active])<=alpha+1e-10)
    support_counts.append(int(active.sum()))
    kkt_residuals.append(float(np.max(np.abs(gradient[active]+alpha*np.sign(coef[active])))))
    print("alpha",alpha,"beta",coef,"nonzero",active.sum(),"KKT verified")

# Repository acceptance checks use exact identities where possible.
write_result('22b', {'lasso_matches_moment_solution': np.max(np.abs(Lasso(alpha=.1,fit_intercept=False,tol=1e-12).fit(X,y).coef_-np.linalg.solve(R,r-.1))) < 1e-9, 'nonmonotone_support_counterexample': support_counts == [2,1,2] and max(kkt_residuals) < 1e-10}, {'moment_solution': np.linalg.solve(R,r-.1).tolist(), 'elastic_net_support_counts': support_counts, 'active_kkt_residuals': kkt_residuals})
