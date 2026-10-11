from _lab_support import begin, budget, write_result, check_logistic_solution
LAB_MODE = begin('26b')
# Experiment M26b; independent synthetic method demonstration, not main case A.
import numpy as np, warnings
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
rng=np.random.default_rng(260)
CALIBRATION_LP_SD_RELATIVE_TOLERANCE = 1e-6
def sigmoid(z): return 1/(1+np.exp(-z))
def fit_logit(X,y,it=60):
    b=np.zeros(X.shape[1])
    for _ in range(it):
        p=sigmoid(X@b); w=np.maximum(p*(1-p),1e-9)
        step=np.linalg.solve(X.T@(w[:,None]*X)+1e-8*np.eye(X.shape[1]), X.T@(y-p)); b+=step
        if np.max(np.abs(step))<1e-9: break
    check_logistic_solution(X, y, b)
    return b
def slope(p,y):
    p=np.clip(p,1e-6,1-1e-6); lp=np.log(p/(1-p))
    sd=float(np.std(lp))
    # The slope is unidentified for a constant linear predictor. Near-constant
    # predictors also lack numerical precision at the original LP scale.
    if sd <= CALIBRATION_LP_SD_RELATIVE_TOLERANCE*max(1.,abs(float(lp.mean()))):
        return None
    z=(lp-lp.mean())/sd
    return float(fit_logit(np.c_[np.ones(len(z)),z],y)[1]/sd)
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
grid_results = []
grid_summary = []
initial_max_iter = budget('mlp_initial_max_iter',3000,3000)
retry_max_iter = budget('mlp_retry_max_iter',10000,10000)
retry_max_fun = budget('mlp_retry_max_fun',50000,50000)
for hs in ((16,),(64,64)):
    for alpha in (1e-4, 1, 10, 30, 100):
        s=[];a=[]
        for r,(X,yv) in enumerate(datasets):
            attempts = []
            for attempt in (1,2):
                options = dict(max_iter=initial_max_iter) if attempt == 1 else dict(max_iter=retry_max_iter,max_fun=retry_max_fun)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    m=MLPClassifier(hidden_layer_sizes=hs, activation='relu', solver='lbfgs', alpha=alpha, random_state=r, **options).fit(X,yv)
                convergence = [str(item.message) for item in caught if issubclass(item.category,ConvergenceWarning)]
                for item in caught:
                    if not issubclass(item.category,ConvergenceWarning):
                        warnings.warn(str(item.message),item.category)
                attempts.append(dict(attempt=attempt,iterations=int(m.n_iter_),options=options,
                                     converged=not convergence,warnings=convergence))
                if not convergence:
                    break
                print(f"retry required: MLP {hs} alpha={alpha} dataset={r}; {convergence[0]}")
            if convergence:
                raise RuntimeError(f"MLP {hs} alpha={alpha} dataset={r} did not converge after recorded retry")
            p=m.predict_proba(Xt)[:,1]
            if not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
                raise RuntimeError("MLP produced invalid probabilities")
            fitted_slope=slope(p,yt); s.append(np.nan if fitted_slope is None else fitted_slope); a.append(auc(p,yt))
            grid_results.append(dict(hidden_layers=list(hs),alpha=alpha,dataset=r,
                                     attempts=attempts,slope=fitted_slope,auc=float(a[-1]),
                                     prediction_sd=float(np.std(p)),
                                     lp_sd=float(np.std(np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))))),
                                     calibration_status='UNDEFINED_CONSTANT_PREDICTION' if fitted_slope is None else 'ESTIMATED'))
        defined=[value for value in s if np.isfinite(value)]
        mean_slope=float(np.mean(defined)) if len(defined)==reps else None
        sd_slope=float(np.std(defined)) if len(defined)==reps else None
        grid_summary.append(dict(hidden_layers=list(hs),alpha=alpha,cal_slope_mean=mean_slope,
                                 cal_slope_std=sd_slope,auc_mean=float(np.mean(a)),
                                 undefined_count=reps-len(defined),converged_fits=reps))
        print(f"MLP {str(hs):9s} alpha={alpha:<7g}: slope={mean_slope} (sd {sd_slope}); estimable {len(defined)}/{reps}; AUC={np.mean(a):.3f}")
print("true-lp AUC:", auc(Xt@beta,yt), " mean events:", np.mean([d[1].sum() for d in datasets]))

# Repository acceptance checks use exact identities where possible.
write_result('26b', {'all_development_samples_have_both_classes': all(np.unique(yi).size==2 for _,yi in datasets), 'all_grid_fits_converged': len(grid_results)==10*reps and all(item['attempts'][-1]['converged'] for item in grid_results), 'all_grid_metrics_valid_or_explicitly_undefined': all(np.isfinite(item['auc']) and ((item['slope'] is None and item['calibration_status']=='UNDEFINED_CONSTANT_PREDICTION') or (item['slope'] is not None and np.isfinite(item['slope']))) for item in grid_results)}, {'development_repetitions': reps, 'last_grid_mean_slope': None if len(defined)!=reps else float(np.mean(defined)), 'last_grid_estimable_slopes': len(defined), 'last_grid_mean_auc': float(np.mean(a)), 'grid_results': grid_results, 'grid_summary': grid_summary, 'calibration_lp_sd_relative_tolerance':CALIBRATION_LP_SD_RELATIVE_TOLERANCE, 'note': 'Constant/near-constant predictions have no estimable calibration slope at the declared LP SD tolerance; null denotes an undefined slope, not failed optimization. A mean is withheld if any dataset in that grid cell has undefined slope. AUC for those predictions is recorded only as raw numerical output and can depend on floating-point residual variation.'})
