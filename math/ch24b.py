from _lab_support import begin, budget, write_result
LAB_MODE = begin('24b')
# Experiment M24b; independent synthetic method demonstration, not main case A.
import numpy as np
np.set_printoptions(precision=6, suppress=True, linewidth=160)
Tc = np.arange(1,11, dtype=float); C = np.array([1,2,1,0,2,1,0,2,1,0])

def km_cens(T_, obs):
    """KM of the censoring distribution. obs=1 if event (any cause) observed, 0 if censored."""
    steps=[]; s=1.0
    for t in np.unique(T_[obs==0]):
        n=(T_>=t).sum(); d=((T_==t)&(obs==0)).sum(); s*=(1-d/n); steps.append((t,s))
    return steps
def G(steps,t,left=False):
    v=1.0
    for tt,s in steps:
        if (tt<t) if left else (tt<=t): v=s
    return v

def ipcw_weights(time, cause, horizon, censoring_steps):
    """Continuous-time/no event-censoring ties convention, HT denominator n.

    Events of either cause by horizon are known. Those still followed after
    horizon are known non-events for the cause-1 target. Earlier censoring is
    unknown and has weight zero. A censoring observation at exactly horizon
    must be adjudicated before this helper is used; it is rejected here.
    """
    time, cause = np.asarray(time, float), np.asarray(cause, float)
    if (time.ndim != 1 or cause.shape != time.shape or not len(time) or
            not np.all(np.isfinite(time)) or np.any(time < 0) or
            not np.all(np.isfinite(cause)) or not np.all(np.isin(cause, [0, 1, 2])) or
            not np.isfinite(horizon) or horizon < 0):
        raise ValueError("IPCW requires aligned nonempty finite times, causes 0/1/2 and a nonnegative horizon")
    steps = np.asarray(censoring_steps, float)
    if steps.size:
        if (steps.ndim != 2 or steps.shape[1] != 2 or not np.all(np.isfinite(steps)) or
                np.any(steps[:, 0] < 0) or np.any(np.diff(steps[:, 0]) <= 0) or
                np.any((steps[:, 1] < 0) | (steps[:, 1] > 1)) or
                np.any(np.diff(steps[:, 1]) > 0)):
            raise ValueError("Censoring survival steps must be ordered finite times and nonincreasing probabilities")
    if np.any((time == horizon) & (cause == 0)):
        raise ValueError("Censoring at the horizon requires a documented boundary convention")
    if np.intersect1d(time[cause == 0], time[cause != 0]).size:
        raise ValueError("Event/censoring ties require a documented ordering convention")
    weight = np.zeros(len(time))
    event = (time <= horizon) & (cause != 0)
    followed = time > horizon
    for i in np.flatnonzero(event):
        g = G(censoring_steps, time[i], left=True)
        if g <= 0:
            raise ValueError("Censoring positivity failed at an event time")
        weight[i] = 1 / g
    if np.any(followed):
        g = G(censoring_steps, horizon)
        if g <= 0:
            raise ValueError("Censoring positivity failed at the horizon")
        weight[followed] = 1 / g
    return weight

def competing_brier(time, cause, horizon, prediction, censoring_steps):
    """Cause-1 CIF prediction must be scored against the cause-1 label.

    A competing event has label zero; it is observed, not censored. Unknown
    outcomes remain NaN and are omitted BEFORE subtraction (0*NaN is NaN).
    """
    time, cause = np.asarray(time, float), np.asarray(cause, float)
    prediction = np.broadcast_to(np.asarray(prediction, float), time.shape)
    if not np.all(np.isfinite(prediction)) or np.any((prediction < 0) | (prediction > 1)):
        raise ValueError("Predictions must be finite probabilities")
    weight = ipcw_weights(time, cause, horizon, censoring_steps)
    known = weight > 0
    label = np.full(len(time), np.nan)
    label[known] = ((time[known] <= horizon) & (cause[known] == 1)).astype(float)
    return np.sum(weight[known] * (label[known] - prediction[known])**2) / len(time)
obs = (C!=0).astype(int)
Gc = km_cens(Tc, obs)
print("censoring KM G(t) steps:", [(float(a), round(b,6)) for a,b in Gc])

# Aalen-Johansen
def aj(T,C):
    S=1.0;F1=0.0;out={}
    for t in np.unique(T):
        nr=(T>=t).sum(); d1=((T==t)&(C==1)).sum(); d=((T==t)&(C!=0)).sum()
        F1+=S*d1/nr; S*=(1-d/nr); out[t]=F1
    return out
AJ = aj(Tc,C)
print("AJ CIF1(6)=%.6f  CIF1(9)=%.6f"%(AJ[6.0],AJ[9.0]))

# IPCW (Horvitz-Thompson)
for t0 in (6,9):
    w = ipcw_weights(Tc, C, t0, Gc)
    est=np.sum(w*((Tc<=t0)&(C==1)))/10
    print(f"t={t0}: weights={np.round(w,4)}  sum of weights/n={w.sum()/10:.4f}  IPCW CIF1={est:.6f}")

# Target-label counterexample: the wrong any-event label reverses the ranking.
t0 = 6.0
w = ipcw_weights(Tc, C, t0, Gc)
known = w > 0
y1 = ((Tc <= t0) & (C == 1)).astype(float)
y_any = ((Tc <= t0) & (C != 0)).astype(float)
p1, p_any = np.dot(w, y1)/len(Tc), np.dot(w, y_any)/len(Tc)
correct = [competing_brier(Tc, C, t0, p, Gc) for p in (p1, p_any)]
wrong = [np.sum(w[known] * (y_any[known]-p)**2)/len(Tc) for p in (p1, p_any)]
assert np.allclose([p1, p_any], [19/60, 8/15])
assert np.allclose(correct, [779/3600, 79/300])
assert np.allclose(wrong, [71/240, 56/225])
assert correct[0] < correct[1] and wrong[0] > wrong[1]
print("\nCause-1 Brier label check at t=6:")
for name, p, bs, bad in zip(("CIF1", "any-event"), (p1, p_any), correct, wrong):
    print(" %s prediction %.6f: correct cause-1 BS %.6f ; WRONG any-event-label BS %.6f" %
          (name, p, bs, bad))
try:
    ipcw_weights(Tc, C, 7.0, Gc)
except ValueError:
    print(" censoring exactly at horizon rejected: boundary convention must be adjudicated")
else:
    raise AssertionError("A day-7 censoring tie was silently accepted")

# Fine-Gray weighted risk sets, product-limit form (no covariates)
print("\nFine-Gray weighted risk set (no covariates):")
prod=1.0; Lam=0.0
for t in Tc[C==1]:
    atrisk=(Tc>=t).sum()
    comp=[j for j in range(10) if C[j]==2 and Tc[j]<t]
    ws=[G(Gc,t)/G(Gc,Tc[j]) for j in comp]
    size=atrisk+sum(ws); h=1/size; prod*=(1-h); Lam+=h
    print(f" t={t:.0f}: at-risk={atrisk}; weights of earlier competing events={[round(x,4) for x in ws]}; weighted size={size:.6f}; h={h:.6f}; 1-prod={1-prod:.6f}; 1-exp(-sum h)={1-np.exp(-Lam):.6f}; AJ={AJ[t]:.6f}")

# Same check on a bigger random dataset to make sure the identity is not a coincidence
rng=np.random.default_rng(5)
n=300
T1=rng.exponential(1/0.10,n); T2=rng.exponential(1/0.06,n); Cc=rng.exponential(1/0.08,n)
Tobs=np.minimum.reduce([T1,T2,Cc]); cause=np.where(Cc<np.minimum(T1,T2),0,np.where(T1<T2,1,2))
Gb=km_cens(Tobs,(cause!=0).astype(int))
ajb=aj(Tobs,cause)
prod=1.0
tt=np.sort(Tobs[cause==1])
vals=[]
for t in tt:
    atrisk=(Tobs>=t).sum()
    comp=np.where((cause==2)&(Tobs<t))[0]
    size=atrisk+sum(G(Gb,t)/G(Gb,Tobs[j]) for j in comp)
    prod*=(1-1/size); vals.append((t,1-prod))
maxdiff=max(abs(v-ajb[t]) for t,v in vals)
print("\nrandom data n=300: max |FG weighted product-limit - AJ| over cause-1 event times =", maxdiff)

# IPCW CIF identity AND actual competing-risk Brier on bigger data
t0=5.0
w = ipcw_weights(Tobs, cause, t0, Gb)
print("IPCW CIF1(5) =", np.sum(w*((Tobs<=t0)&(cause==1)))/n, " AJ =", [v for t,v in ajb.items() if t<=t0][-1])
p_null = [v for t,v in ajb.items() if t<=t0][-1]
bs_null = competing_brier(Tobs, cause, t0, p_null, Gb)
assert np.isclose(bs_null, p_null * (1-p_null))
print("Actual competing-risk Brier at t=5, constant AJ prediction =", bs_null)

# Repository acceptance checks use exact identities where possible.
write_result('24b', {'correct_target_ranking': correct[0] < correct[1], 'wrong_target_reverses_ranking': wrong[0] > wrong[1], 'fine_gray_product_limit_equals_aj': maxdiff < 1e-10, 'brier_constant_aj_identity': abs(bs_null-p_null*(1-p_null)) < 1e-10}, {'correct_brier': correct, 'wrong_brier': wrong, 'fg_aj_max_difference': float(maxdiff)})
