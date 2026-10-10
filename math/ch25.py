from _lab_support import begin, budget, write_result
LAB_MODE = begin('25')
# Experiment M25; independent synthetic method demonstration, not main case A.
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
np.set_printoptions(precision=6, suppress=True, linewidth=160)
rng = np.random.default_rng(25)
def sigmoid(z): return 1/(1+np.exp(-z))
def gini(y):
    if len(y)==0: return 0.0
    p=np.mean(y); return 2*p*(1-p)

x = np.arange(1,9, dtype=float)
for name,y in {"y1=[0,0,0,1,0,1,1,1]":np.array([0,0,0,1,0,1,1,1]), "y2=[0,0,0,0,1,0,1,1]":np.array([0,0,0,0,1,0,1,1])}.items():
    print("==", name, " parent Gini =", round(gini(y),6))
    for thr in np.arange(1.5,8,1.0):
        L=y[x<thr]; R=y[x>=thr]
        wg=(len(L)*gini(L)+len(R)*gini(R))/len(y)
        print(f"   split x<{thr}: nL={len(L)} nR={len(R)} giniL={gini(L):.4f} giniR={gini(R):.4f} weighted={wg:.4f} gain={gini(y)-wg:.4f}")

# ---------- one boosting step on y2 ----------
y = np.array([0,0,0,0,1,0,1,1], float)
p0 = y.mean(); F0 = np.log(p0/(1-p0))
g = p0 - y; h = np.full(8, p0*(1-p0))
print("\n== boosting step on y2: p0=%.4f F0=logit=%.6f  g_i=p-y: %s  h_i=%.6f"%(p0,F0,g,h[0]))
lam=1.0; gam=0.0
def score(G,H,l): return G**2/(H+l)
print(" split  G_L     H_L      G_R     H_R     gain(lam=1)   w_L(lam=1)  w_R(lam=1)  w_L(lam=0)  w_R(lam=0)")
best=None
for thr in np.arange(1.5,8,1.0):
    m=x<thr; GL,HL,GR,HR=g[m].sum(),h[m].sum(),g[~m].sum(),h[~m].sum()
    gain=0.5*(score(GL,HL,lam)+score(GR,HR,lam)-score(GL+GR,HL+HR,lam))-gam
    wl,wr=-GL/(HL+lam),-GR/(HR+lam); wl0,wr0=-GL/HL,-GR/HR
    print(f" x<{thr}: {GL: .4f} {HL:.4f} {GR: .4f} {HR:.4f}  {gain:.6f}   {wl: .5f}  {wr: .5f}  {wl0: .5f}  {wr0: .5f}")
    if best is None or gain>best[0]: best=(gain,thr,wl,wr,wl0,wr0)
print(" best split:", best)
gain,thr,wl,wr,wl0,wr0=best
eta=0.3
m=x<thr
F1=F0+eta*np.where(m,wl,wr); p1=sigmoid(F1)
def logloss(p,y): return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
print(" learning rate eta=0.3 -> F1 left/right:", F1[0], F1[-1], " p1:", p1[0], p1[-1])
print(" logloss before=%.6f after=%.6f"%(logloss(np.full(8,p0),y), logloss(p1,y)))
# objective-decrease check for the second-order approximation: exact loss after full step with leaf weights (lam=1, eta=1)
F1f=F0+np.where(m,wl,wr); print(" exact logloss*8 with eta=1 (lam=1): %.6f vs before %.6f  actual decrease=%.6f"%(8*logloss(sigmoid(F1f),y), 8*logloss(np.full(8,p0),y), 8*(logloss(np.full(8,p0),y)-logloss(sigmoid(F1f),y))))
print(" second-order predicted decrease (gain formula, lam=0 version) =", 0.5*(g[m].sum()**2/h[m].sum()+g[~m].sum()**2/h[~m].sum()))
# pure leaf: exact optimum is -infinity
print("\n pure left leaf (4 points, all y=0): leaf logloss contribution as weight w decreases from F0:")
for w_ in (0,-1,-1.6,-3,-6,-10):
    pl=sigmoid(F0+w_); print(f"   w={w_:6.1f}: p={pl:.5f}  4*(-log(1-p))={-4*np.log(1-pl):.5f}")
print("  Newton step with lam=0: -G/H = -1.6 ; with lam=1: %.5f ; exact optimum: w -> -inf"%wl)

# sklearn cross-check (lam=0 Newton leaf values; learning_rate=1)
gb = GradientBoostingClassifier(n_estimators=1, learning_rate=1.0, max_depth=1, random_state=0).fit(x.reshape(-1,1), y)
raw = gb.decision_function(x.reshape(-1,1))
print("\n sklearn GBM 1 stump, lr=1: raw scores:", raw, " split threshold:", gb.estimators_[0,0].tree_.threshold[0])
print(" my lam=0 leaf scores: left", F0+wl0, " right", F0+wr0)

# ---------- OOB fraction ----------
print("\n== OOB fraction (1-1/n)^n ==")
for n in (8,20,100,1000,100000): print(f"  n={n:6d}: {(1-1/n)**n:.6f}   distinct in bootstrap = {1-(1-1/n)**n:.6f}")
print("  limit exp(-1) =", np.exp(-1), " 1-exp(-1) =", 1-np.exp(-1))
sims=[];
for _ in range(budget('oob_repetitions',20000,2000)):
    idx=rng.integers(0,8,8); sims.append(1-len(np.unique(idx))/8)
print(f"  MC n=8 ({len(sims)} bootstrap samples): mean OOB fraction =", np.mean(sims))

# ---------- variance of average of correlated trees ----------
print("\n== Var of average of B equicorrelated predictions (sigma^2=1, rho=0.3) ==")
rho=0.3
for B in (1,10,100,1000):
    form = rho + (1-rho)/B
    # For jointly Gaussian equicorrelated predictions, their mean is exactly
    # N(0, rho+(1-rho)/B); sample that mean directly to avoid multi-GB matrices.
    # The mathematical distribution is unchanged; random-number consumption differs.
    averaged = rng.normal(scale=np.sqrt(form), size=budget('correlated_mean_repetitions',200000,20000))
    print(f"  B={B:5d}: formula rho + (1-rho)/B = {form:.4f};  MC = {averaged.var():.4f}")

# (the calibration simulation is in ch25b.py)

# Repository acceptance checks use exact identities where possible.
write_result('25', {'boosting_newton_stump_matches_sklearn': np.max(np.abs(raw-np.where(m,F0+wl0,F0+wr0))) < 1e-10, 'step_reduces_log_loss': logloss(p1,y) < logloss(np.full(8,p0),y), 'oob_limit': abs((1-1/100000)**100000-np.exp(-1)) < 1e-5}, {'stump_threshold': float(thr), 'loss_before': float(logloss(np.full(8,p0),y)), 'loss_after': float(logloss(p1,y)), 'variance_demo': 'exact Gaussian mean distribution; random stream differs from original matrix simulation'})
