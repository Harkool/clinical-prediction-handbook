from _lab_support import begin, budget, write_result, check_logistic_solution
LAB_MODE = begin('25b')
# Experiment M25b; independent synthetic method demonstration, not main case A.
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
np.set_printoptions(precision=6, suppress=True, linewidth=160)
rng=np.random.default_rng(250)
def sigmoid(z): return 1/(1+np.exp(-z))

# regularized-objective check for the second-order gain
y=np.array([0,0,0,0,1,0,1,1.]); x=np.arange(1,9.)
p0=y.mean(); F0=np.log(p0/(1-p0)); g=p0-y; h=np.full(8,p0*(1-p0)); lam=1.0
m=x<4.5
GL,HL,GR,HR=g[m].sum(),h[m].sum(),g[~m].sum(),h[~m].sum()
gain=0.5*(GL**2/(HL+lam)+GR**2/(HR+lam)-(GL+GR)**2/(HL+HR+lam))
wl,wr=-GL/(HL+lam),-GR/(HR+lam)
def total_loss(F): p=sigmoid(F); return -np.sum(y*np.log(p)+(1-y)*np.log(1-p))
before=total_loss(np.full(8,F0))
after=total_loss(F0+np.where(m,wl,wr)) + 0.5*lam*(wl**2+wr**2)
print("gain (2nd-order predicted decrease of regularized objective) = %.6f"%gain)
print("actual decrease of regularized objective = %.6f  (before %.6f, after incl. penalty %.6f)"%(before-after,before,after))
print("relative error = %.2f%%"%(100*abs((before-after)-gain)/gain))
# Newton leaf with lam=0 = what sklearn gives (shown before). Also first-order (gradient) leaf: -mean(g)/1 ... shown for comparison
print("first-order leaf value with the same eta=1 would be -sum(g)/n_leaf:", -g[m].mean(), g[~m].mean()*-1)

# ---- calibration slope grid ----
def fit_logit(X,y,it=60):
    b=np.zeros(X.shape[1])
    for _ in range(it):
        p=sigmoid(X@b); w=np.maximum(p*(1-p),1e-9)
        step=np.linalg.solve(X.T@(w[:,None]*X)+1e-8*np.eye(X.shape[1]), X.T@(y-p)); b+=step
        if np.max(np.abs(step))<1e-9: break
    check_logistic_solution(X, y, b)
    return b
def slope(p,y):
    p=np.clip(p,1e-6,1-1e-6); lp=np.log(p/(1-p)); return fit_logit(np.c_[np.ones(len(lp)),lp],y)[1]
def auc(p,y):
    from scipy.stats import rankdata
    r=rankdata(p); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
pdim=8; beta=np.array([0.7,0.5,0.4,0.3,0.2,0,0,0]); b0=-2.2
Xt=rng.normal(size=(budget('test_n',40000,5000),pdim)); yt=(rng.random(budget('test_n',40000,5000))<sigmoid(b0+Xt@beta)).astype(float)
print("true-lp AUC on test: %.3f ; prevalence %.3f"%(auc(Xt@beta,yt), yt.mean()))
reps=budget('development_repetitions',20,3)
datasets=[]
for r in range(reps):
    X=rng.normal(size=(400,pdim)); yv=(rng.random(400)<sigmoid(b0+X@beta)).astype(float); datasets.append((X,yv))
print("mean events in dev:", np.mean([d[1].sum() for d in datasets]))
def run(make, name):
    s=[];a=[]
    for r,(X,yv) in enumerate(datasets):
        mdl=make(r).fit(X,yv); p=mdl.predict_proba(Xt)[:,1]; s.append(slope(p,yt)); a.append(auc(p,yt))
    print(f"  {name:44s} slope={np.mean(s):.3f} (sd {np.std(s):.3f})   AUC={np.mean(a):.3f}")
# LR reference
s=[];a=[]
for X,yv in datasets:
    b=fit_logit(np.c_[np.ones(400),X],yv); p=sigmoid(np.c_[np.ones(len(Xt)),Xt]@b); s.append(slope(p,yt)); a.append(auc(p,yt))
print(f"  {'Logistic (ML)':44s} slope={np.mean(s):.3f} (sd {np.std(s):.3f})   AUC={np.mean(a):.3f}")
print(f"Random forest, {budget('rf_max_trees',300,60)} trees, varying min_samples_leaf:")
for msl in (1,5,20,50):
    run(lambda r,msl=msl: RandomForestClassifier(n_estimators=budget('rf_max_trees',300,60),min_samples_leaf=msl,random_state=r,n_jobs=1), f"RF min_samples_leaf={msl}")
print("Random forest, min_samples_leaf=5, varying number of trees:")
for nt in ((10,30,60) if LAB_MODE == 'quick' else (10,50,300)):
    run(lambda r,nt=nt: RandomForestClassifier(n_estimators=nt,min_samples_leaf=5,random_state=r,n_jobs=1), f"RF trees={nt}")
print("Gradient boosting, depth 3:")
for (ne,lr) in ((25,0.1),(50,0.1),(100,0.1),(200,0.1),(200,0.02),(500,0.02)):
    run(lambda r,ne=ne,lr=lr: GradientBoostingClassifier(n_estimators=ne,learning_rate=lr,max_depth=3,random_state=r), f"GBM trees={ne}, lr={lr}")
print("Gradient boosting, 100 trees lr 0.1, varying depth / subsample:")
for (dp,ss) in ((1,1.0),(2,1.0),(3,1.0),(3,0.5)):
    run(lambda r,dp=dp,ss=ss: GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=dp,subsample=ss,random_state=r), f"GBM depth={dp}, subsample={ss}")

# Repository acceptance checks use exact identities where possible.
write_result('25b', {'regularized_objective_decreases': before > after, 'gain_positive': gain > 0, 'all_development_samples_have_both_classes': all(np.unique(yi).size==2 for _,yi in datasets)}, {'second_order_gain': float(gain), 'actual_regularized_decrease': float(before-after), 'development_repetitions': reps})
