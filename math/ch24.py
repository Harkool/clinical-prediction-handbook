from _lab_support import begin, budget, write_result
LAB_MODE = begin('24')
# Experiment M24; independent synthetic method demonstration, not main case A.
import numpy as np
from scipy.optimize import minimize
np.set_printoptions(precision=6, suppress=True, linewidth=160)

# =====================  A. Cox partial likelihood, Breslow, absolute risk  =====================
T = np.array([2,3,4,5,6,8,9,12], float)
D = np.array([1,1,0,1,1,0,1,1])
X = np.array([1,0,1,1,0,0,1,0], float)
ev = np.where(D==1)[0]
print("== A: Cox on n=8 (no ties) ==")
print("event times:", T[ev], " x at events:", X[ev])
def riskset(t): return np.where(T>=t)[0]
for i in ev:
    R = riskset(T[i]); print(f"  t={T[i]:>4.0f}: risk set ids(1-based)={list(R+1)}  size={len(R)}  x in R={X[R].astype(int).tolist()}")
def pl(beta):
    ll=0; U=0; I=0
    for i in ev:
        R = riskset(T[i]); w=np.exp(beta*X[R]); s0=w.sum(); s1=(w*X[R]).sum(); s2=(w*X[R]**2).sum()
        ll += beta*X[i] - np.log(s0); U += X[i]-s1/s0; I += s2/s0-(s1/s0)**2
    return ll,U,I
print("beta=0: logPL=%.6f  score=%.6f  info=%.6f  first Newton step=%.6f"%(pl(0)[0],pl(0)[1],pl(0)[2],pl(0)[1]/pl(0)[2]))
print("  check: -sum(log |R_i|) =", -sum(np.log(len(riskset(T[i]))) for i in ev))
b=0.0
print("Newton iterations:")
for it in range(8):
    ll,U,I = pl(b); print(f"  iter {it}: beta={b: .6f} logPL={ll: .6f} score={U: .6f} info={I: .6f}")
    b = b + U/I
ll,U,I = pl(b); bhat=b; se=1/np.sqrt(I)
print(f"beta_hat={bhat:.6f}  SE={se:.6f}  HR={np.exp(bhat):.4f}  95%CI=({np.exp(bhat-1.96*se):.3f},{np.exp(bhat+1.96*se):.3f})  logPL={ll:.6f}")
# independent: scipy
res = minimize(lambda v: -pl(v[0])[0], [0.0], method="BFGS", tol=1e-12)
print("scipy BFGS beta =", res.x[0])
# Breslow baseline cumulative hazard (reference x=0)
print("\nBreslow baseline (x=0 reference):")
H=0; Hs={}
for i in ev:
    R = riskset(T[i]); dh = 1.0/np.exp(bhat*X[R]).sum(); H += dh; Hs[T[i]]=H
    print(f"  t={T[i]:>4.0f}  jump={dh:.6f}  H0(t)={H:.6f}")
def H0_at(t):
    v=0
    for tt,h in Hs.items():
        if tt<=t: v=h
    return v
for t in (6,9,12):
    h0=H0_at(t)
    r0=1-np.exp(-h0); r1=1-np.exp(-h0*np.exp(bhat))
    print(f"  t={t}: H0={h0:.6f}  risk(x=0)={r0:.4f}  risk(x=1)={r1:.4f}")
# profile likelihood check: maximize full log-lik over (beta, h_k) numerically
Ts = T[ev]; K=len(ev)
def negfull(v):
    beta=v[0]; h=np.exp(v[1:])   # jumps
    ll=0
    for j in range(len(T)):
        eta=beta*X[j]
        cum = h[Ts<=T[j]].sum()          # cumulative baseline hazard to T_j
        if D[j]==1:
            k = np.where(Ts==T[j])[0][0]; ll += np.log(h[k]) + eta
        ll -= cum*np.exp(eta)
    return -ll
v0 = np.r_[0.0, np.log(np.full(K,0.1))]
res2 = minimize(negfull, v0, method="BFGS", options={"gtol":1e-10,"maxiter":2000})
print("\nfull-likelihood max: beta =", res2.x[0], " jumps =", np.exp(res2.x[1:]))
print("Breslow jumps (profile)   :", np.array([1.0/np.exp(bhat*X[riskset(T[i])]).sum() for i in ev]))
# baseline reference: centered vs x=0
xbar = X.mean()
print(f"\nreference-point pitfall: mean(x)={xbar}; H0 at x=xbar is H0(x=0)*exp(beta*xbar): at t=6 ->", H0_at(6)*np.exp(bhat*xbar))
h0c = H0_at(6)*np.exp(bhat*xbar)
print("  correct risk x=1 (uses H0 at x=0):", 1-np.exp(-H0_at(6)*np.exp(bhat)))
print("  WRONG risk x=1 if centered baseline is used as if x=0:", 1-np.exp(-h0c*np.exp(bhat)))
print("  correct for centered baseline: exp(beta*(x-xbar)):", 1-np.exp(-h0c*np.exp(bhat*(1-xbar))))
# Kaplan-Meier by group, for comparison at t=6
def km(t_, d_):
    s=1.0; out={}
    for t in np.unique(t_[d_==1]):
        n=(t_>=t).sum(); d=((t_==t)&(d_==1)).sum(); s*= (1-d/n); out[t]=s
    return out
for g in (0,1):
    m = X==g; kmg = km(T[m], D[m])
    s6 = [v for t,v in kmg.items() if t<=6]; s9=[v for t,v in kmg.items() if t<=9]
    print(f"  KM group x={g}: S(6)={s6[-1] if s6 else 1:.4f}, S(9)={s9[-1] if s9 else 1:.4f}  -> 1-S(6)={1-(s6[-1] if s6 else 1):.4f}")

# IPCW Brier at t=6 for the Cox model
print("\nIPCW Brier(t=6), Cox model risks:")
def km_cens(T_, D_):
    # KM of censoring distribution (censoring = 'event')
    ts = np.unique(T_[D_==0]); s=1.0; steps=[]
    for t in ts:
        n=(T_>=t).sum(); d=((T_==t)&(D_==0)).sum(); s*=(1-d/n); steps.append((t,s))
    return steps
def Gfun(steps, t, left=False):
    v=1.0
    for tt,s in steps:
        if (tt<t) if left else (tt<=t): v=s
    return v
Gst = km_cens(T,D)
print("  censoring KM steps:", Gst)
t0=6; H0_6=H0_at(t0)
pi = 1-np.exp(-H0_6*np.exp(bhat*X))
w=np.zeros(len(T))
for i in range(len(T)):
    if T[i]<=t0 and D[i]==1: w[i]=1/Gfun(Gst,T[i],left=True)
    elif T[i]>t0: w[i]=1/Gfun(Gst,t0)
Y = (T<=t0).astype(float)
print("  weights:", w, "\n  Y(t):", Y, "\n  pi:", pi)
bs = np.sum(w*(Y-pi)**2)/len(T)
print("  IPCW Brier(6) =", bs, "   (reference: null model pi=IPCW-risk)")
# IPCW-based risk = 1 - KM
km_all = km(T,D); s6=[v for t,v in km_all.items() if t<=6][-1]
print("  IPCW weighted event proportion:", np.sum(w*Y)/len(T), "  1-KM(6) =", 1-s6)

# =====================  B. competing risks, 10 patients  =====================
print("\n== B: competing risks (n=10) ==")
Tc = np.arange(1,11, dtype=float); C = np.array([1,2,1,0,2,1,0,2,1,0])
def aj(Tc,C,tmax):
    n=len(Tc); S=1.0; F1=0.0; F2=0.0; rows=[]
    for t in np.unique(Tc):
        nr=(Tc>=t).sum(); d1=((Tc==t)&(C==1)).sum(); d2=((Tc==t)&(C==2)).sum(); d=d1+d2
        F1+=S*d1/nr; F2+=S*d2/nr; S*=(1-d/nr)
        rows.append((t,nr,d1,d2,S,F1,F2))
    return rows
rows = aj(Tc,C,10)
print("   t  n_risk d1 d2   S(t)     CIF1     CIF2   sum")
for r in rows: print(f" {r[0]:3.0f} {r[1]:5d} {r[2]:3d}{r[3]:3d}  {r[4]:.6f} {r[5]:.6f} {r[6]:.6f} {r[4]+r[5]+r[6]:.6f}")
# naive 1-KM (cause 2 as censored)
Dn = (C==1).astype(int); kmn = km(Tc,Dn)
for t in (6,9):
    s=[v for tt,v in kmn.items() if tt<=t][-1]
    cif=[r[5] for r in rows if r[0]<=t][-1]
    print(f" t={t}: naive 1-KM(cause1)={1-s:.6f}   Aalen-Johansen CIF1={cif:.6f}   overestimate x{(1-s)/cif:.3f}")
# (IPCW and Fine-Gray weighted risk-set checks for these data: see ch24b.py)

# =====================  C. constant hazards: HR(cause1)=1.2 but CIF1 lower  =====================
print("\n== C: constant cause-specific hazards ==")
def F1(l1,l2,t): return l1/(l1+l2)*(1-np.exp(-(l1+l2)*t))
for t in (1,2,5,10):
    f0=F1(0.10,0.05,t); f1=F1(0.12,0.20,t)
    cumulative_ratio = np.log1p(-f1)/np.log1p(-f0)
    h0 = 0.10*np.exp(-0.15*t)/(1-f0)
    h1 = 0.12*np.exp(-0.32*t)/(1-f1)
    print(f" t={t:>2}: CIF1 unexposed={f0:.4f} exposed={f1:.4f} CIF ratio={f1/f0:.3f}; cumulative subhazard ratio={cumulative_ratio:.6f}; instantaneous subhazard ratio={h1/h0:.6f}")
print(" cause-specific HR cause1 = 0.12/0.10 =", 0.12/0.10, "; cause2 = 0.20/0.05 =", 0.20/0.05)
# sum of two CIFs at t=10 for both groups (should be <=1)
for nm,(l1,l2) in {"unexposed":(0.10,0.05),"exposed":(0.12,0.20)}.items():
    print(f" {nm}: CIF1+CIF2 at t=10 = {F1(l1,l2,10)+F1(l2,l1,10):.4f} (=1-exp(-(l1+l2)*10)={1-np.exp(-(l1+l2)*10):.4f})")

# Repository acceptance checks use exact identities where possible.
write_result('24', {'cox_score_at_solution': abs(pl(bhat)[1]) < 1e-10, 'full_likelihood_matches_profile': abs(res2.x[0]-bhat) < 1e-5, 'cif_mass_conservation': all(abs(row[4]+row[5]+row[6]-1) < 1e-12 for row in rows), 'cause_hr_does_not_order_cif': F1(.12,.20,10) < F1(.10,.05,10)}, {'cox_beta': float(bhat), 'ipcw_brier_6': float(bs)})
