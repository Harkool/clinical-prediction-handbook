"""Patient bootstrap validation for the frozen 19-column teaching case.

NumPy PCG64 resampling does not reproduce the former R random stream.
No failed replicate is deleted to create a success-only estimate. These
functions return JSON-serializable tables and retain a failed-fit ledger.
"""
import math
from numbers import Integral, Real

import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata

from .metrics import _wilson


METRICS = ("C","Brier","calibration_slope","calibration_intercept","OE")
THRESHOLDS = (.05,.10,.15,.20,.25,.30)
FAILED_FIT_POLICY = "Any failed replicate invalidates its entire aggregate; no deletion or replacement resampling"
RNG_METHOD = "NumPy default_rng PCG64; different from the historical R random stream"


class BootstrapFailure(RuntimeError):
    """A failed aggregate with a recoverable diagnostic report attached."""
    def __init__(self, report):
        self.report = report
        super().__init__(report.get("reason","Bootstrap aggregate not estimable"))


def _core():
    from . import course_core
    return course_core


def _integer(value,name,minimum=2):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<minimum:
        raise ValueError(name+" must be an integer >= "+str(minimum))
    return int(value)


def _settings(B,seed,failure_policy):
    B=_integer(B,"B")
    seed=_integer(seed,"seed",0)
    if failure_policy not in ("invalidate","raise"):
        raise ValueError("failure_policy must be invalidate or raise")
    return B,seed


def _thresholds(values):
    result=np.asarray(values,float)
    if (result.ndim!=1 or not len(result) or not np.all(np.isfinite(result)) or
            np.any((result<=0)|(result>=1)) or np.any(np.diff(result)<=0)):
        raise ValueError("Thresholds must be prespecified, finite, unique and increasing in (0,1)")
    return result


def _pairs(y,p,both=True,interior=True):
    y,p=np.asarray(y,float),np.asarray(p,float)
    if (y.ndim!=1 or p.shape!=y.shape or not len(y) or not np.all(np.isin(y,[0,1])) or
            not np.all(np.isfinite(p)) or np.any((p<0)|(p>1))):
        raise ValueError("Observed binary labels and finite probability vectors must be nonempty and aligned")
    if both and len(np.unique(y))!=2:
        raise ValueError("Both event and non-event outcomes are required")
    if interior and np.any((p<=0)|(p>=1)):
        raise ValueError("Calibration requires interior probabilities; endpoints are not silently clipped")
    return y,p


def _auc(y,p):
    n1=int(np.sum(y));n0=len(y)-n1
    if n1==0 or n0==0:
        return None
    return float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def _fit_beta(X,y,initial=None,offset=None):
    fit=_core().checked_logistic_fit(X,y,offset=offset,initial=initial)
    beta=np.asarray(fit["coefficients"],float)
    if beta.shape!=(X.shape[1],) or not np.all(np.isfinite(beta)) or not fit.get("converged",False):
        raise RuntimeError("Core logistic fit did not pass its finite convergence contract")
    return beta


def _performance(y,p):
    y,p=_pairs(y,p)
    lp=logit(p);sd=float(np.std(lp))
    if sd<=1e-6*max(1.,abs(float(lp.mean()))):
        raise ValueError("Calibration slope is not estimable for constant/near-constant predictions")
    z=(lp-lp.mean())/sd
    # Changing the column scale leaves the unpenalized calibration model unchanged.
    slope_fit=_fit_beta(np.column_stack((np.ones(len(y)),z)),y)
    citl_fit=_fit_beta(np.ones((len(y),1)),y,offset=lp)
    return dict(C=_auc(y,p),Brier=float(np.mean((p-y)**2)),
                calibration_slope=float(slope_fit[1]/sd),
                calibration_intercept=float(citl_fit[0]),OE=float(np.sum(y)/np.sum(p)))


def performance(y,p):
    """Five apparent metrics, with explicit non-estimability instead of NaN PASS."""
    try:
        estimates=_performance(y,p)
        return dict(status="PASS",estimates=estimates,scope="apparent fixed predictions; no clinical validity claim")
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
        return dict(status="NOT_ESTIMABLE",estimates={name:None for name in METRICS},reason=str(error))


def net_benefit(y,p,thresholds=THRESHOLDS):
    y,p=_pairs(y,p,interior=False)
    thresholds=_thresholds(thresholds)
    decision=p[:,None]>=thresholds[None,:]
    return np.mean(decision*y[:,None],axis=0)-thresholds/(1-thresholds)*np.mean(decision*(1-y[:,None]),axis=0)


def _initial_beta(model,X):
    beta=np.asarray(model["beta"],float)
    if beta.shape!=(X.shape[1],) or not np.all(np.isfinite(beta)):
        raise ValueError("Model coefficients do not match the frozen design")
    return beta


def _data(records,model=None):
    core=_core();columns=core.as_columns(records)
    y=np.asarray(columns["y"],float)
    X=core.make_design(columns,None if model is None else model.get("recipe"))
    if y.shape!=(len(X),) or not np.all(np.isin(y,[0,1])) or len(np.unique(y))!=2:
        raise ValueError("Development data require aligned observed outcomes of both classes")
    if model is not None:
        # Use the public predictor once to check the model recipe/identity contract.
        p=core.predict_case(model,columns)
        beta=_initial_beta(model,X)
    else:
        beta=_fit_beta(X,y)
        p=expit(X@beta)
    return columns,X,y,beta,p


def _record(stage,iteration,function):
    try:
        value=function()
        return value,dict(stage=stage,iteration=iteration,status="PASS",reason="",retry_count=0)
    except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
        entry=dict(stage=stage,iteration=iteration,status="FAILED",reason=str(error),retry_count=0)
        if isinstance(error,BootstrapFailure):
            entry["nested_failure_report"]=error.report
        return None,entry


def _finish(report,failure_policy):
    if report["status"]!="PASS" and failure_policy=="raise":
        raise BootstrapFailure(report)
    return report


def _optimism_design(X,y,beta,B,seed,thresholds,failure_policy):
    apparent=_performance(y,expit(X@beta));nb0=net_benefit(y,expit(X@beta),thresholds)
    rng=np.random.default_rng(seed);delta=[];nbdelta=[];ledger=[]
    for iteration in range(1,B+1):
        index=rng.integers(0,len(y),size=len(y))
        def replicate():
            boot_beta=_fit_beta(X[index],y[index],initial=beta)
            pb,po=expit(X[index]@boot_beta),expit(X@boot_beta)
            training,original=_performance(y[index],pb),_performance(y,po)
            return ([training[name]-original[name] for name in METRICS],
                    net_benefit(y[index],pb,thresholds)-net_benefit(y,po,thresholds))
        value,entry=_record("development",iteration,replicate);ledger.append(entry)
        delta.append(None if value is None else value[0]);nbdelta.append(None if value is None else value[1].tolist())
        if value is None and failure_policy=="raise":
            raise BootstrapFailure(dict(status="NOT_ESTIMABLE",ledger=ledger,
                                        reason="Development bootstrap failed: "+entry["reason"]))
    success=all(item["status"]=="PASS" for item in ledger)
    mean=np.mean(delta,axis=0) if success else None
    mcse=np.std(delta,axis=0,ddof=1)/np.sqrt(B) if success else None
    nbmean=np.mean(nbdelta,axis=0) if success else None
    nbse=np.std(nbdelta,axis=0,ddof=1)/np.sqrt(B) if success else None
    table=[dict(metric=name,apparent=apparent[name],mean_optimism=None if mean is None else float(mean[k]),
                corrected=None if mean is None else float(apparent[name]-mean[k]),
                optimism_MCSE=None if mcse is None else float(mcse[k]),B=B) for k,name in enumerate(METRICS)]
    dca=[dict(threshold=float(t),apparent_NB=float(nb0[k]),
              optimism=None if nbmean is None else float(nbmean[k]),
              corrected_NB=None if nbmean is None else float(nb0[k]-nbmean[k]),
              treat_all=float(np.mean(y)-(1-np.mean(y))*t/(1-t)),treat_none=0.,
              optimism_MCSE=None if nbse is None else float(nbse[k])) for k,t in enumerate(thresholds)]
    return _finish(dict(status="PASS" if success else "NOT_ESTIMABLE",table=table,
        replicates=[{name:None if draw is None else float(draw[k]) for k,name in enumerate(METRICS)} for draw in delta],
        dca=dca,ledger=ledger,B=B,seed=seed,valid_replicates=sum(item["status"]=="PASS" for item in ledger),
        failed_replicates=sum(item["status"]!="PASS" for item in ledger),failed_fit_policy=FAILED_FIT_POLICY,
        rng=RNG_METHOD,scope="whole frozen-design development fitting bootstrap; no imputation, selection or tuning in case A",
        reason="" if success else "At least one replicate failed; all corrected aggregates are invalid",
        OE_note="Internally corrected O/E is not an independent external calibration estimate"),failure_policy)


def bootstrap_optimism(records,model,B=200,seed=20261011,thresholds=THRESHOLDS,*,failure_policy="invalidate"):
    B,seed=_settings(B,seed,failure_policy);thresholds=_thresholds(thresholds)
    _,X,y,beta,_=_data(records,model)
    return _optimism_design(X,y,beta,B,seed,thresholds,failure_policy)


def fixed_model_intervals(y,p,B=500,seed=20261012,*,confidence=.95,failure_policy="invalidate"):
    """Pairs bootstrap of fixed predictions; excludes model-development uncertainty."""
    B,seed=_settings(B,seed,failure_policy);y,p=_pairs(y,p)
    if isinstance(confidence,bool) or not isinstance(confidence,Real) or not np.isfinite(confidence) or not 0<confidence<1:
        raise ValueError("confidence must be finite and strictly between zero and one")
    estimate=_performance(y,p);rng=np.random.default_rng(seed);draws=[];ledger=[]
    for iteration in range(1,B+1):
        index=rng.integers(0,len(y),size=len(y))
        draw,entry=_record("fixed_model",iteration,lambda:_performance(y[index],p[index]))
        draws.append(draw);ledger.append(entry)
        if draw is None and failure_policy=="raise":
            raise BootstrapFailure(dict(status="NOT_ESTIMABLE",ledger=ledger,reason=entry["reason"]))
    success=all(item["status"]=="PASS" for item in ledger);alpha=(1-confidence)/2
    table=[]
    for name in METRICS:
        values=np.array([draw[name] for draw in draws]) if success else None
        bounds=np.quantile(values,[alpha,1-alpha]) if success else [None,None]
        table.append(dict(metric=name,estimate=estimate[name],lower=None if not success else float(bounds[0]),
            upper=None if not success else float(bounds[1]),valid_replicates=sum(draw is not None for draw in draws),
            failed_replicates=sum(draw is None for draw in draws),B=B,
            bootstrap_mean_MCSE=None if not success else float(np.std(values,ddof=1)/np.sqrt(B)),
            endpoint_rank_MCSE=float(np.sqrt(B*alpha*(1-alpha))),
            method="conditional empirical percentile bootstrap; fixed model apparent evaluation"))
    return _finish(dict(status="PASS" if success else "NOT_ESTIMABLE",table=table,replicates=draws,ledger=ledger,
        B=B,seed=seed,confidence=float(confidence),scope="conditional on fixed model; no development or selection uncertainty",
        failed_fit_policy=FAILED_FIT_POLICY,rng=RNG_METHOD,
        mcse_note="bootstrap_mean_MCSE describes the bootstrap mean, not endpoint standard errors; endpoint_rank_MCSE is Monte Carlo order-statistic rank uncertainty",
        reason="" if success else "At least one replicate failed; no success-only percentile intervals"),failure_policy)


def development_intervals(records,B_outer=200,B_inner=50,seed=20261014,*,failure_policy="invalidate"):
    """Double bootstrap of an internally corrected estimate, not external validation."""
    B_outer,seed=_settings(B_outer,seed,failure_policy);B_inner=_integer(B_inner,"B_inner")
    _,X,y,beta,_=_data(records);rng=np.random.default_rng(seed);draws=[];inner_mcse=[];ledger=[]
    for iteration in range(1,B_outer+1):
        index=rng.integers(0,len(y),size=len(y))
        def replicate():
            outer_beta=_fit_beta(X[index],y[index],initial=beta)
            correction=_optimism_design(X[index],y[index],outer_beta,B_inner,seed+iteration,
                                        np.asarray(THRESHOLDS),"raise")
            return ({row["metric"]:row["corrected"] for row in correction["table"]},
                    {row["metric"]:row["optimism_MCSE"] for row in correction["table"]})
        value,entry=_record("double_bootstrap_outer",iteration,replicate);ledger.append(entry)
        draws.append(None if value is None else value[0]);inner_mcse.append(None if value is None else value[1])
        if value is None and failure_policy=="raise":
            raise BootstrapFailure(dict(status="NOT_ESTIMABLE",ledger=ledger,reason=entry["reason"]))
    success=all(draw is not None for draw in draws);table=[]
    for name in METRICS:
        values=np.array([draw[name] for draw in draws]) if success else None
        bounds=np.quantile(values,[.025,.975]) if success else [None,None]
        table.append(dict(metric=name,lower=None if not success else float(bounds[0]),
            upper=None if not success else float(bounds[1]),B_outer=B_outer,B_inner=B_inner,
            mean_inner_optimism_MCSE=None if not success else float(np.mean([item[name] for item in inner_mcse])),
            bootstrap_mean_MCSE=None if not success else float(np.std(values,ddof=1)/np.sqrt(B_outer)),
            method="approximate percentile double bootstrap of internally corrected development estimate"))
    return _finish(dict(status="PASS" if success else "NOT_ESTIMABLE",table=table,replicates=draws,
        ledger=ledger,B_outer=B_outer,B_inner=B_inner,seed=seed,rng=RNG_METHOD,failed_fit_policy=FAILED_FIT_POLICY,
        scope="development-sample uncertainty of the internally corrected estimate; finite inner/outer budgets cause Monte Carlo error",
        reason="" if success else "At least one outer/inner replicate failed; entire interval aggregate invalid"),failure_policy)


def _calibration_design(probability,knots):
    p=np.asarray(probability,float);knots=np.asarray(knots,float)
    if (p.ndim!=1 or not len(p) or not np.all(np.isfinite(p)) or np.any((p<=0)|(p>=1)) or
            knots.shape!=(3,) or not np.all(np.isfinite(knots)) or np.any(np.diff(knots)<=0)):
        raise ValueError("Smooth calibration requires interior probabilities and three ordered logit knots")
    x=logit(p);a,b,c=knots;span=c-a
    positive=lambda value:np.maximum(value,0.)**3
    basis=(positive(x-a)-positive(x-b)*(c-a)/(c-b)+positive(x-c)*(b-a)/(c-b))/span**3
    return np.column_stack((np.ones(len(p)),(x-b)/span,basis))


def calibration_fit(y,p,grid,knots):
    """Unpenalized natural cubic logit model: the same span as R ns with one inner knot."""
    y,p=_pairs(y,p)
    X=_calibration_design(p,knots);design_grid=_calibration_design(grid,knots)
    beta=_fit_beta(X,y)
    values=expit(design_grid@beta)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Smooth calibration produced nonfinite probabilities")
    return values


def calibration_groups(y,p,breaks=(0,.05,.10,.15,.20,.30,.50,1)):
    y,p=_pairs(y,p,both=False,interior=False);edges=np.asarray(breaks,float)
    if (edges.ndim!=1 or len(edges)<2 or not np.all(np.isfinite(edges)) or
            edges[0]!=0 or edges[-1]!=1 or np.any(np.diff(edges)<=0)):
        raise ValueError("Calibration breaks must be ordered, start at zero and end at one")
    assignments=np.minimum(np.searchsorted(edges,p,side="right")-1,len(edges)-2);table=[]
    for index in range(len(edges)-1):
        selected=assignments==index;n=int(np.sum(selected));events=int(np.sum(y[selected]))
        bounds=_wilson(events,n)
        table.append(dict(lower=float(edges[index]),upper=float(edges[index+1]),n=n,events=events,
            mean_prediction=None if not n else float(np.mean(p[selected])),event_rate=None if not n else events/n,
            wilson_lower=None if bounds is None else bounds[0],wilson_upper=None if bounds is None else bounds[1],
            scope="within-group apparent event rate; descriptive Wilson interval"))
    return table


def calibration_artifacts(records,model,B=200,conditional_B=500,seed=20261011,conditional_seed=20261012,*,failure_policy="invalidate"):
    B,seed=_settings(B,seed,failure_policy);conditional_B,conditional_seed=_settings(conditional_B,conditional_seed,failure_policy)
    _,X,y,beta,p=_data(records,model);knots=np.quantile(logit(p),[.05,.5,.95]);support=np.quantile(p,[.01,.99])
    grid=np.linspace(support[0],support[1],80);apparent=calibration_fit(y,p,grid,knots)
    delta=[];conditional=[];ledger=[];rng=np.random.default_rng(seed)
    for iteration in range(1,B+1):
        index=rng.integers(0,len(y),size=len(y))
        def replicate():
            boot_beta=_fit_beta(X[index],y[index],initial=beta)
            return (calibration_fit(y[index],expit(X[index]@boot_beta),grid,knots)-
                    calibration_fit(y,expit(X@boot_beta),grid,knots)).tolist()
        value,entry=_record("development_curve",iteration,replicate);delta.append(value);ledger.append(entry)
    rng=np.random.default_rng(conditional_seed)
    for iteration in range(1,conditional_B+1):
        index=rng.integers(0,len(y),size=len(y))
        value,entry=_record("fixed_apparent_curve",iteration,lambda:calibration_fit(y[index],p[index],grid,knots).tolist())
        conditional.append(value);ledger.append(entry)
    success_dev=all(value is not None for value in delta);success_cond=all(value is not None for value in conditional)
    corrected=apparent-np.mean(delta,axis=0) if success_dev else [None]*len(grid)
    mcse=np.std(delta,axis=0,ddof=1)/np.sqrt(B) if success_dev else [None]*len(grid)
    bounds=np.quantile(conditional,[.025,.975],axis=0) if success_cond else [[None]*len(grid),[None]*len(grid)]
    curve=[dict(prediction=float(grid[k]),apparent=float(apparent[k]),
        corrected=None if not success_dev else float(corrected[k]),
        apparent_lower=None if not success_cond else float(bounds[0][k]),apparent_upper=None if not success_cond else float(bounds[1][k]),
        bootstrap_success=sum(value is not None for value in delta),conditional_success=sum(value is not None for value in conditional),
        optimism_mcse=None if not success_dev else float(mcse[k]),interval_target="fixed_model_apparent_curve_pointwise",
        interval_method="empirical percentiles; fixed prediction pairs") for k in range(len(grid))]
    recipe=dict(logit_knots=knots.tolist(),support_probability=support.tolist(),grid_n=80,development_B=B,
        conditional_B=conditional_B,development_seed=seed,conditional_seed=conditional_seed,
        smooth_method="unpenalized natural cubic logistic calibration; intercept + linear + one restricted cubic basis; same span as historical R ns",
        corrected_curve_interval="none; corrected point estimate and optimism MCSE",failed_fit_policy=FAILED_FIT_POLICY,rng=RNG_METHOD)
    report=dict(status="PASS" if success_dev and success_cond else "NOT_ESTIMABLE",success=success_dev and success_cond,
        curve=curve,groups=calibration_groups(y,p),recipe=recipe,log=ledger,ledger=ledger,predictions=p.tolist(),
        scope="apparent pointwise conditional range; corrected curve has no interval and is not external validation",
        reason="" if success_dev and success_cond else "At least one fit failed; the affected aggregate is invalid")
    return _finish(report,failure_policy)


def subgroup_performance(records,p,B=300,seed=20261013,*,minimum_interval_replicates=100):
    """Descriptive within-development subgroups; small groups retain explicit NE."""
    B,seed=_settings(B,seed,"invalidate");minimum_interval_replicates=_integer(minimum_interval_replicates,"minimum_interval_replicates")
    columns=_core().as_columns(records);y,p=_pairs(columns["y"],p,both=False,interior=False)
    groups={"all":np.ones(len(y),bool)}
    if "male" in columns:
        male=np.asarray(columns["male"],float);groups.update({"male_"+str(k):male==k for k in (0,1)})
    if "center" in columns:
        center=np.asarray(columns["center"],float);groups.update({"center_"+str(k):center==k for k in range(1,7)})
    rng=np.random.default_rng(seed);table=[];ledger=[]
    for name,selected in groups.items():
        yg,pg=y[selected],p[selected];n=len(yg);draws=[]
        for iteration in range(1,B+1):
            index=rng.integers(0,n,size=n) if n else np.array([],int)
            value=_auc(yg[index],pg[index]) if n else None;draws.append(value)
            if value is None:
                ledger.append(dict(group=name,iteration=iteration,status="NOT_ESTIMABLE",reason="subgroup resample lacks both classes"))
        valid=sum(value is not None for value in draws);estimate=_auc(yg,pg)
        # A failed resample is not deleted to manufacture a successful interval.
        estimable=valid==B and B>=minimum_interval_replicates and estimate is not None
        bounds=np.quantile(draws,[.025,.975]) if estimable else [None,None]
        table.append(dict(group=name,n=n,events=int(np.sum(yg)),non_events=int(n-np.sum(yg)),C=estimate,
            lower=None if not estimable else float(bounds[0]),upper=None if not estimable else float(bounds[1]),
            valid_replicates=valid,failed_replicates=B-valid,B=B,
            interval_status="PASS" if estimable else "NOT_ESTIMABLE",
            conclusion="interval not estimable reliably in this resampling run" if not estimable else
                       "imprecise descriptive result; no subgroup superiority claim" if min(np.sum(yg),n-np.sum(yg))<30 else
                       "descriptive within-development-sample result"))
    return dict(status="DESCRIPTIVE",table=table,ledger=ledger,B=B,seed=seed,rng=RNG_METHOD,
        minimum_interval_replicates=minimum_interval_replicates,failed_fit_policy=FAILED_FIT_POLICY,
        scope="fixed-model empirical within-development subgroup ranges; excludes development uncertainty and subgroup superiority")
