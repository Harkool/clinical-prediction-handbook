from _lab_support import begin, budget, write_result
LAB_MODE = begin('26b')
# Experiment M26b; independent synthetic method demonstration, not main case A.
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPClassifier
rng=np.random.default_rng(260)
def sigmoid(z): return 1/(1+np.exp(-z))
def fit_logit(X,y,it=60):
    b=np.zeros(X.shape[1])
    for _ in range(it):
        p=sigmoid(X@b); w=np.maximum(p*(1-p),1e-9)
        step=np.linalg.solve(X.T@(w[:,None]*X)+1e-8*np.eye(X.shape[1]), X.T@(y-p)); b+=step
        if np.max(np.abs(step))<1e-9: break
    return b
def slope(p,y):
    p=np.clip(p,1e-6,1-1e-6); lp=np.log(p/(1-p)); return fit_logit(np.c_[np.ones(len(lp)),lp],y)[1]
def auc(p,y):
    from scipy.stats import rankdata
    r=rankdata(p); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
pdim=8; beta=np.array([0.7,0.5,0.4,0.3,0.2,0,0,0]); b0=-2.2
Xt=rng.normal(size=(budget('test_n',40000,5000),pdim)); yt=(rng.random(budget('test_n',40000,5000))<sigmoid(b0+Xt@beta)).astype(float)
reps=budget('development_repetitions',20,3); datasets=[]
for r in range(reps):
    X=rng.normal(size=(400,pdim)); yv=(rng.random(400)<sigmoid(b0+X@beta)).astype(float); datasets.append((X,yv))
s=[];a=[]
for X,yv in datasets:
    b=fit_logit(np.c_[np.ones(400),X],yv); p=sigmoid(np.c_[np.ones(len(Xt)),Xt]@b); s.append(slope(p,yt)); a.append(auc(p,yt))
print(f"Logistic (ML): slope={np.mean(s):.3f} (sd {np.std(s):.3f}) AUC={np.mean(a):.3f}")
for hs in ((16,),(64,64)):
    for alpha in (1e-4, 1, 10, 30, 100):
        s=[];a=[]
        for r,(X,yv) in enumerate(datasets):
            m=MLPClassifier(hidden_layer_sizes=hs, activation='relu', solver='lbfgs', alpha=alpha, max_iter=3000, random_state=r).fit(X,yv)
            p=m.predict_proba(Xt)[:,1]; s.append(slope(p,yt)); a.append(auc(p,yt))
        print(f"MLP {str(hs):9s} alpha={alpha:<7g}: slope={np.mean(s):.3f} (sd {np.std(s):.3f})  AUC={np.mean(a):.3f}")
print("true-lp AUC:", auc(Xt@beta,yt), " mean events:", np.mean([d[1].sum() for d in datasets]))

# Repository acceptance checks use exact identities where possible.
write_result('26b', {'all_development_samples_have_both_classes': all(np.unique(yi).size==2 for _,yi in datasets), 'last_grid_slope_finite': bool(np.all(np.isfinite(s)))}, {'development_repetitions': reps, 'last_grid_mean_slope': float(np.mean(s)), 'last_grid_mean_auc': float(np.mean(a))})
