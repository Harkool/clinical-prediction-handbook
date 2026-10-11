from _lab_support import begin, budget, write_result
LAB_MODE = begin('29')
# Experiment M29; independent synthetic method demonstration, not main case A.
# ch29.py —— 第29章 数值小例：缺失数据
# PyCharm 里右键运行；输入都在配置区
import time
import numpy as np
from scipy import special, stats

# ============================== 配置区 ==============================
SEED = 29
Q_EST = [1.02, 0.95, 1.10, 0.98, 1.05]        # A：5 份填补数据的估计
U_EST = [0.040, 0.050, 0.045, 0.040, 0.050]   # A：各份的方差（SE²）
N_BIG = budget('mechanism_n',400_000,30000)                               # B：机制比较的样本量
N_C, Q_MISS, M_IMP, R_C = budget('imputation_n',2000,1000), 0.4, budget('imputation_m',20,5), budget('imputation_repetitions',300,20)  # C：样本量、缺失比例、填补份数、重复数
N_PMM_OBS, N_PMM_MIS, D_PMM = 60, 40, 5       # D：PMM 的供体数、待填数、候选供体数
N_DEV, N_TEST, R_E = budget('deployment_train_n',5000,1000), budget('deployment_test_n',50000,5000), budget('deployment_repetitions',60,6)          # E：开发样本、测试样本、重复数
RHO = 0.6                                     # E：X1 与 X2 的相关
# ===================================================================
rng = np.random.default_rng(SEED)
expit = special.expit
t0 = time.time()

print("=" * 72)
print("A. Rubin 规则手算（m = 5）")
print("=" * 72)
Q = np.array(Q_EST); U = np.array(U_EST); m = len(Q)
Qbar = Q.mean(); Ubar = U.mean(); B = Q.var(ddof=1)
T = Ubar + (1 + 1 / m) * B
r_ = (1 + 1 / m) * B / Ubar
nu = (m - 1) * (1 + 1 / r_) ** 2
lam = (1 + 1 / m) * B / T
print(f"Q̄ = {Qbar:.4f}; Ū = {Ubar:.5f}; B = {B:.5f}; T = Ū+(1+1/m)B = {T:.5f}; SE = {np.sqrt(T):.4f}")
print(f"r = (1+1/m)B/Ū = {r_:.4f}; ν = (m-1)(1+1/r)² = {nu:.1f}; 缺失信息比例 λ = (1+1/m)B/T = {lam:.4f}")
print(f"只用 Ū 的 SE = {np.sqrt(Ubar):.4f}（低估 {1 - np.sqrt(Ubar / T):.1%}）")
print("m 大到无穷时：T∞ = Ū + B =", round(Ubar + B, 5), "；m=5 时多出的 B/m =", round(B / m, 5))

print()
print("=" * 72)
print("B. MCAR / MAR / MNAR：同一份数据，估计目标不同，结论不同")
print("=" * 72)
x = rng.normal(size=N_BIG); y = x + rng.normal(size=N_BIG)   # E[Y]=0，斜率=1
mech = {
    "MCAR（与谁都无关，P=0.5）": np.full(N_BIG, 0.5),
    "MAR（只与 X 有关，P=expit(1.5X)）": expit(1.5 * x),
    "MNAR（与 Y 本身有关，P=expit(1.5Y)）": expit(1.5 * y),
}
print(f"{'机制':<36}{'观测比例':>8}{'E[Y] 完整病例':>14}{'E[Y] 回归法':>12}{'斜率 完整病例':>14}")
for nm, pobs in mech.items():
    obs = rng.uniform(size=N_BIG) < pobs
    xo, yo = x[obs], y[obs]
    slope = np.cov(xo, yo)[0, 1] / xo.var(ddof=1)
    icpt = yo.mean() - slope * xo.mean()
    reg_mean = np.mean(icpt + slope * x)        # 用观测者拟合 Y~X，再对全体 X 平均
    print(f"{nm:<36}{obs.mean():>8.3f}{yo.mean():>14.3f}{reg_mean:>12.3f}{slope:>14.3f}")
print("真值：E[Y] = 0，斜率 = 1")

print()
print("=" * 72)
print(f"C. 结局要不要进填补模型（X1 缺 {Q_MISS:.0%}，MCAR；Y = X1 + ε；n={N_C}；m={M_IMP}；{R_C} 次重复）")
print("=" * 72)


def ols_se(Z, yv):
    b = np.linalg.solve(Z.T @ Z, Z.T @ yv)
    res = yv - Z @ b
    s2 = res @ res / (len(yv) - Z.shape[1])
    cov = s2 * np.linalg.inv(Z.T @ Z)
    return b, np.sqrt(np.diag(cov))


def impute_with_y(xv, yv, miss, covariates=None):
    """X1 | Y, observed covariates: parameter and residual draws (proper)."""
    obs = ~miss
    Zall = np.c_[np.ones(len(yv)), yv]
    if covariates is not None:
        Zall = np.column_stack([Zall, covariates])
    Zo = Zall[obs]
    b, _ = ols_se(Zo, xv[obs])
    res = xv[obs] - Zo @ b
    df = obs.sum() - Zo.shape[1]
    s2 = res @ res / stats.chi2.rvs(df, random_state=rng)
    L = np.linalg.cholesky(np.linalg.inv(Zo.T @ Zo))
    bs = b + np.sqrt(s2) * L @ rng.normal(size=Zo.shape[1])
    out = xv.copy()
    out[miss] = Zall[miss] @ bs + np.sqrt(s2) * rng.normal(size=miss.sum())
    return out


def impute_marginal(xv, miss):
    """不含结局：只用 X1 自己的分布"""
    obs = ~miss
    n_o = obs.sum(); mu = xv[obs].mean(); s2 = xv[obs].var(ddof=1)
    s2s = (n_o - 1) * s2 / stats.chi2.rvs(n_o - 1, random_state=rng)
    mus = mu + np.sqrt(s2s / n_o) * rng.normal()
    out = xv.copy(); out[miss] = mus + np.sqrt(s2s) * rng.normal(size=miss.sum())
    return out


def pooled(xv, yv, miss, imputer):
    Qs, Us = [], []
    for _ in range(M_IMP):
        xi = imputer(xv, yv, miss)
        b, se = ols_se(np.c_[np.ones(len(yv)), xi], yv)
        Qs.append(b[1]); Us.append(se[1] ** 2)
    Qs, Us = np.array(Qs), np.array(Us)
    Tt = Us.mean() + (1 + 1 / M_IMP) * Qs.var(ddof=1)
    return Qs.mean(), np.sqrt(Tt), np.sqrt(Us.mean())


out = {k: [] for k in ["完整病例", "单次填补（不含结局）", "MI 不含结局", "MI 含结局"]}
for _ in range(R_C):
    xv = rng.normal(size=N_C); yv = xv + rng.normal(size=N_C)
    miss = rng.uniform(size=N_C) < Q_MISS
    b, se = ols_se(np.c_[np.ones((~miss).sum()), xv[~miss]], yv[~miss])
    out["完整病例"].append((b[1], se[1], se[1]))
    xi = impute_marginal(xv, miss); b, se = ols_se(np.c_[np.ones(N_C), xi], yv)
    out["单次填补（不含结局）"].append((b[1], se[1], se[1]))
    out["MI 不含结局"].append(pooled(xv, yv, miss, lambda a, c, d: impute_marginal(a, d)))
    out["MI 含结局"].append(pooled(xv, yv, miss, impute_with_y))
print(f"{'方法':<22}{'斜率均值':>9}{'SD(重复间)':>12}{'报告的SE均值':>13}{'仅Ū的SE':>10}{'95%区间覆盖真值1':>16}")
for k, v in out.items():
    a = np.array(v)
    cover = np.mean(np.abs(a[:, 0] - 1.0) <= 1.96 * a[:, 1])
    print(f"{k:<22}{a[:, 0].mean():>9.3f}{a[:, 0].std(ddof=1):>12.4f}{a[:, 1].mean():>13.4f}{a[:, 2].mean():>10.4f}{cover:>16.3f}")
print(f"理论：不含结局的填补，斜率 ≈ (1-q)β = {1 - Q_MISS:.2f}")

print()
print("=" * 72)
print("D. 预测均值匹配（PMM）与未变换的正态线性回归填补：偏态、恒正的变量（像肌酐）")
print("=" * 72)
n_o, d, N_REC, REP_D = N_PMM_OBS, D_PMM, budget('pmm_recipient_n',4000,400), budget('pmm_repetitions',200,10)
def make_y(xv): return np.exp(0.5 + 0.8 * xv + 0.3 * rng.normal(size=len(xv)))
res_d = {}
for scen, sd_x in (("待填者的 x 与供体同分布 N(0,1)", 1.0), ("待填者的 x 更分散 N(0,1.5²)（需要外推）", 1.5)):
    pool = {"真值": [], "正态回归填补": [], "PMM": []}
    for _ in range(REP_D):
        xo = rng.normal(size=n_o); yo = make_y(xo)
        xm = sd_x * rng.normal(size=N_REC); ym = make_y(xm)
        Zo = np.c_[np.ones(n_o), xo]; bh, _ = ols_se(Zo, yo)
        s2h = np.sum((yo - Zo @ bh) ** 2) / (n_o - 2)
        s2s = s2h * (n_o - 2) / stats.chi2.rvs(n_o - 2, random_state=rng)
        L = np.linalg.cholesky(np.linalg.inv(Zo.T @ Zo)); bs = bh + np.sqrt(s2s) * L @ rng.normal(size=Zo.shape[1])
        Zm = np.c_[np.ones(N_REC), xm]
        y_norm = Zm @ bs + np.sqrt(s2s) * rng.normal(size=N_REC)
        yhat_o = Zo @ bh; yhat_m = Zm @ bs                       # type-1：供体用 β̂，待填者用抽到的 β*
        order = np.argsort(yhat_o); sorted_hat = yhat_o[order]
        pos = np.searchsorted(sorted_hat, yhat_m)
        y_pmm = np.empty(N_REC)
        for j in range(N_REC):                                   # 取预测均值最近的 d 个供体
            lo = max(0, pos[j] - d); hi = min(n_o, pos[j] + d)
            cand = order[lo:hi]
            near = cand[np.argsort(np.abs(yhat_o[cand] - yhat_m[j]))[:d]]
            y_pmm[j] = yo[rng.choice(near)]
        pool["真值"].append(ym); pool["正态回归填补"].append(y_norm); pool["PMM"].append(y_pmm)
        dmax = yo.max()
        pool.setdefault("_dmax", []).append(dmax)
    res_d[scen] = pool
    dm = np.array(pool["_dmax"])
    print(f"\n{scen}（供体 {n_o} 个，待填者 {N_REC} 个，{REP_D} 次重复）")
    print(f"{'':<14}{'5%分位':>9}{'中位':>8}{'95%分位':>9}{'99%分位':>9}{'均值':>8}{'≤0 的比例':>10}{'超过供体最大值的比例':>20}")
    for k in ("真值", "正态回归填补", "PMM"):
        v = np.concatenate(pool[k]); q = np.percentile(v, [5, 50, 95, 99])
        over = np.mean(np.concatenate([a > m_ for a, m_ in zip(pool[k], dm)]))
        print(f"{k:<14}{q[0]:>9.3f}{q[1]:>8.3f}{q[2]:>9.3f}{q[3]:>9.3f}{v.mean():>8.3f}{np.mean(v <= 0):>10.4f}{over:>20.4f}")
# 手算展示：一个待填者的 5 个最近供体
xo = rng.normal(size=n_o); yo = make_y(xo); Zo = np.c_[np.ones(n_o), xo]; bh, _ = ols_se(Zo, yo)
x_new = 0.2; yhat_new = bh[0] + bh[1] * x_new; yhat_o = Zo @ bh
near = np.argsort(np.abs(yhat_o - yhat_new))[:d]
print(f"\n手算展示：待填者 x = {x_new}，预测均值 = {yhat_new:.3f}；最近 {d} 个供体（预测均值，观测 y）：",
      [(round(float(yhat_o[i]), 3), round(float(yo[i]), 3)) for i in near])
print(f"随机取其中一个供体的观测 y 作填补值；候选值都在 {min(yo[near]):.3f}–{max(yo[near]):.3f} 之内（观测者 y 的范围 {yo.min():.3f}–{yo.max():.3f}）")

print()
print("=" * 72)
print(f"E. 开发—部署相容性（线性高斯；X1 在开发、部署两端都 MCAR 缺 {Q_MISS:.0%}；{R_E} 次重复）")
print("=" * 72)
sig = 1.0
def gen(n):
    x2 = rng.normal(size=n); x1 = RHO * x2 + np.sqrt(1 - RHO ** 2) * rng.normal(size=n)
    yv = x1 + x2 + sig * rng.normal(size=n)
    return x1, x2, yv
bayes_c = sig ** 2                       # X1、X2 都有：不可约误差
bayes_m = sig ** 2 + (1 - RHO ** 2)      # 只有 X2：Var(Y|X2) = σ² + Var(X1|X2)
# This fixed-parameter random imputation is a prediction-algorithm contrast.
# It does not draw the X1|X2 regression parameters and is NOT proper MI for Rubin inference.
strategies = ["完整病例", "回归填补（确定性，不含结局）", "固定参数随机填补（不含结局）", "MI 含结局"]
acc = {s: [] for s in strategies}
for rep in range(R_E):
    x1, x2, yv = gen(N_DEV); miss = rng.uniform(size=N_DEV) < Q_MISS
    # X1|X2 的回归（只用观测者），部署端补值用它
    Zo = np.c_[np.ones((~miss).sum()), x2[~miss]]; g, _ = ols_se(Zo, x1[~miss])
    coefs = {}
    b, _ = ols_se(np.c_[np.ones((~miss).sum()), x1[~miss], x2[~miss]], yv[~miss]); coefs["完整病例"] = b
    x1_ri = x1.copy(); x1_ri[miss] = g[0] + g[1] * x2[miss]
    b, _ = ols_se(np.c_[np.ones(N_DEV), x1_ri, x2], yv); coefs["回归填补（确定性，不含结局）"] = b
    bs_x, bs_y = [], []
    s2x = np.sum((x1[~miss] - Zo @ g) ** 2) / (len(Zo) - 2)
    for _ in range(budget('deployment_imputations',10,5)):
        xi = x1.copy(); xi[miss] = g[0] + g[1] * x2[miss] + np.sqrt(s2x) * rng.normal(size=miss.sum())
        bs_x.append(ols_se(np.c_[np.ones(N_DEV), xi, x2], yv)[0])
        xi2 = impute_with_y(x1, yv, miss, covariates=x2)
        bs_y.append(ols_se(np.c_[np.ones(N_DEV), xi2, x2], yv)[0])
    coefs["固定参数随机填补（不含结局）"] = np.mean(bs_x, 0); coefs["MI 含结局"] = np.mean(bs_y, 0)
    # 部署：测试集，X1 同样 MCAR 缺失，缺失者用 E[X1|X2] 补值（确定性）或抽一次
    # 超额 MSE = E[(预测 − 最优预测)²]：最优预测在模拟里已知（X1 有：X1+X2；X1 缺：(1+ρ)X2），可去掉结局的抽样噪声
    t1, t2, _ty = gen(N_TEST); tm = rng.uniform(size=N_TEST) < Q_MISS
    opt_c = t1 + t2; opt_m = (1 + RHO) * t2
    t1_mean = t1.copy(); t1_mean[tm] = g[0] + g[1] * t2[tm]
    t1_draw = t1.copy(); t1_draw[tm] = g[0] + g[1] * t2[tm] + np.sqrt(s2x) * rng.normal(size=tm.sum())
    for s in strategies:
        b = coefs[s]
        p_mean = b[0] + b[1] * t1_mean + b[2] * t2
        p_draw = b[0] + b[1] * t1_draw + b[2] * t2
        acc[s].append((np.mean((p_mean[~tm] - opt_c[~tm]) ** 2),
                       np.mean((p_mean[tm] - opt_m[tm]) ** 2),
                       np.mean((p_draw[tm] - opt_m[tm]) ** 2),
                       b[1], b[2]))
print(f"不可约误差：X1 有 = {bayes_c:.2f}；X1 缺（只有 X2）= {bayes_m:.2f}。表中是超额 MSE = E[(预测 − 最优预测)²]，越小越好，最优时为 0")
print(f"{'开发策略':<26}{'X1 有':>9}{'X1 缺·均值补':>13}{'X1 缺·抽一次补':>14}{'系数 X1':>9}{'系数 X2':>9}")
for s in strategies:
    a = np.array(acc[s]).mean(0)
    print(f"{s:<26}{a[0]:>9.4f}{a[1]:>13.4f}{a[2]:>14.4f}{a[3]:>9.3f}{a[4]:>9.3f}")
print("（真值：系数 X1 = 1，系数 X2 = 1）")

print()
print("=" * 72)
print("F. Logistic：部署时 X1 缺失，‘补均值再预测’与‘对填补抽样的概率取平均’")
print("=" * 72)
a0, a1, a2 = -1.0, 1.5, 1.0
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(80); gh_w = gh_w / gh_w.sum()
sd1 = np.sqrt(1 - RHO ** 2)
print(f"{'X2':>5}{'E[X1|X2]':>10}{'补均值再预测':>14}{'抽样概率取平均':>16}{'真值 P(Y=1|X2)':>16}")
for x2v in (-2, -1, 0, 1, 2):
    mu = RHO * x2v
    plug = expit(a0 + a1 * mu + a2 * x2v)
    truth = np.sum(gh_w * expit(a0 + a1 * (mu + sd1 * gh_x) + a2 * x2v))
    draws = mu + sd1 * rng.normal(size=budget('probability_draws',200000,20000))
    avg = expit(a0 + a1 * draws + a2 * x2v).mean()
    print(f"{x2v:>5}{mu:>10.2f}{plug:>14.4f}{avg:>16.4f}{truth:>16.4f}")
print(f"总用时 {time.time() - t0:.0f} 秒")

# Repository acceptance checks use exact identities where possible.
write_result('29', {'rubin_total_variance_includes_between_component': T > Ubar and abs(T-(Ubar+1.2*B)) < 1e-12, 'pmm_uses_observed_positive_donors': all(np.concatenate(pool['PMM']).min()>0 and all(np.all(a<=maximum) for a,maximum in zip(pool['PMM'],pool['_dmax'])) for pool in res_d.values())}, {'rubin_pooled_estimate': float(Qbar), 'rubin_total_variance': float(T), 'rubin_se': float(np.sqrt(T))})
