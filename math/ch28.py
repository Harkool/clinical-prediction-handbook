from _lab_support import begin, budget, write_result
LAB_MODE = begin('28')
# Experiment M28; independent synthetic method demonstration, not main case A.
# ch28.py —— 第28章 数值小例：重抽样
# PyCharm 里右键运行；输入都在配置区
import itertools
import time
import numpy as np
from scipy import special
from scipy.stats import rankdata

# ============================== 配置区 ==============================
SEED = 28
Y_SMALL = [1, 2, 3, 4, 5, 9]      # A：6 例的手算数据
N_OLS, P_OLS, SIGMA = 40, 5, 1.0  # B/C：线性回归的学习曲线与 K 折偏差
R_OLS = budget('ols_repetitions',2000,200)                      # B/C：重复数据集个数
R_BOOT_OLS, B_OLS = budget('ols_development_repetitions',600,30), budget('ols_bootstrap',200,50)      # D：bootstrap 乐观（数据集个数 × 每个数据集的 bootstrap 次数）
N_NOISE, P_NOISE, PREV = 100, 50, 0.3   # E：纯噪声设定（真实 C 恰为 0.5）
K_SCREEN = 5                      # E：筛选后保留的变量数
R_NOISE, B_NOISE, K_FOLD = budget('noise_repetitions',300,12), budget('noise_bootstrap',100,20), 10  # E：重复数、bootstrap 次数、折数
K_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 40, 50]  # F：嵌套 CV 要选的候选变量数
R_NEST = budget('nested_repetitions',150,8)                      # F：重复数
# ===================================================================
rng = np.random.default_rng(SEED)
expit = special.expit
t0 = time.time()


def auc_fast(y, s):
    r = rankdata(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


FIT_DIAGNOSTICS = dict(fits=0, max_gradient=0.0, max_iterations=0, failures=0)


def logit_fit(X, y, iters=200, ridge=1e-6):
    """Finite L2 logistic fit, with Armijo backtracking and gradient acceptance.

    The fixed ridge applies to all coefficients, including the intercept.
    It is part of this experiment's learning algorithm, not unpenalized ML.
    """
    b = np.zeros(X.shape[1])
    def objective(v):
        eta = X @ v
        return np.sum(np.logaddexp(0, eta) - y * eta) + .5 * ridge * (v @ v)
    for iteration in range(iters):
        eta = X @ b; p = expit(eta)
        grad = X.T @ (p - y) + ridge * b
        if np.max(np.abs(grad)) <= 1e-7:
            break
        w = p * (1 - p)
        hessian = X.T @ (w[:, None] * X) + ridge * np.eye(X.shape[1])
        direction = np.linalg.solve(hessian, grad)
        f0 = objective(b); descent = grad @ direction; step = 1.0
        while step > 2**-40:
            candidate = b - step * direction
            # Tiny allowance covers objective rounding near a stationary point.
            if objective(candidate) <= f0 - 1e-4 * step * descent + 1e-12:
                b = candidate
                break
            step *= .5
        else:
            break
    final_gradient = float(np.max(np.abs(X.T @ (expit(X @ b) - y) + ridge * b)))
    FIT_DIAGNOSTICS['fits'] += 1
    FIT_DIAGNOSTICS['max_gradient'] = max(FIT_DIAGNOSTICS['max_gradient'], final_gradient)
    FIT_DIAGNOSTICS['max_iterations'] = max(FIT_DIAGNOSTICS['max_iterations'], iteration + 1)
    if not np.isfinite(final_gradient) or final_gradient > 1e-5:
        FIT_DIAGNOSTICS['failures'] += 1
        raise RuntimeError(f"Logistic fit failed gradient check: {final_gradient:.3g}")
    return b


# -------------------------------------------------------------------
print("=" * 72)
print("A. 6 例手算：样本均值模型的 bootstrap 乐观（全部 6^6 个 bootstrap 样本枚举）")
print("=" * 72)
y = np.array(Y_SMALL, float); n = len(y)
ybar = y.mean(); s2hat = np.mean((y - ybar) ** 2)       # 插入式方差（除以 n）
idx = np.array(list(itertools.product(range(n), repeat=n)))  # 46656 × 6
ys = y[idx]
ybar_b = ys.mean(1)
mse_on_Dstar = ((ys - ybar_b[:, None]) ** 2).mean(1)        # f* 在自己的样本上
mse_on_D = ((y[None, :] - ybar_b[:, None]) ** 2).mean(1)    # f* 在原样本上
print(f"y = {Y_SMALL}, ȳ = {ybar:.4f}, σ̂²(÷n) = {s2hat:.4f}")
print(f"E*[MSE(f*; D*)] = {mse_on_Dstar.mean():.5f}   理论 σ̂²(1-1/n) = {s2hat * (1 - 1 / n):.5f}")
print(f"E*[MSE(f*; D )] = {mse_on_D.mean():.5f}   理论 σ̂²(1+1/n) = {s2hat * (1 + 1 / n):.5f}")
opt_boot = mse_on_D.mean() - mse_on_Dstar.mean()
print(f"bootstrap 乐观 = {opt_boot:.5f}   理论 2σ̂²/n = {2 * s2hat / n:.5f}")
print(f"表观 MSE = {s2hat:.4f}; 校正后 MSE = {s2hat + opt_boot:.4f}; "
      f"对照：σ² 的无偏估计 s²(÷n-1) = {y.var(ddof=1):.4f}，真误差期望 σ²(1+1/n) 的估计 = {y.var(ddof=1) * (1 + 1 / n):.4f}")
print(f"随机 bootstrap（B={budget('mean_random_bootstrap',200000,20000)}）核对：", end=" ")
ii = rng.integers(0, n, size=(budget('mean_random_bootstrap',200000,20000), n)); yy = y[ii]; mb = yy.mean(1)
print(round(((y[None, :] - mb[:, None]) ** 2).mean() - ((yy - mb[:, None]) ** 2).mean(), 5))
print(f"每个 bootstrap 样本中不同观测的平均个数 = {np.mean([len(set(r)) for r in idx]):.3f}（理论 n[1-(1-1/n)^n] = {n * (1 - (1 - 1 / n) ** n):.3f}，比例 {1 - (1 - 1 / n) ** n:.4f}）")

# -------------------------------------------------------------------
print()
print("=" * 72)
print(f"B/C. 线性回归：学习曲线、K 折偏差与方差（n={N_OLS}, p={P_OLS}, σ={SIGMA}，{R_OLS} 个数据集）")
print("=" * 72)
n, p = N_OLS, P_OLS
beta = np.full(p, 0.5)


def true_err(a_hat, b_hat):
    """给定拟合模型，在总体上的真实 MSE：X~N(0,I)，真实截距 0"""
    return SIGMA ** 2 + a_hat ** 2 + np.sum((b_hat - beta) ** 2)


def ols(X, y):
    Z = np.c_[np.ones(len(y)), X]
    b = np.linalg.solve(Z.T @ Z, Z.T @ y)
    return b


def exact_err(m):
    """IID Gaussian random design, fitted intercept, m > p+2."""
    return SIGMA ** 2 * (1 + 1 / m + (1 + 1 / m) * p / (m - p - 2))


folds = {2: n // 2, 5: n // 5, 10: n // 10}
app = np.zeros(R_OLS); terr = np.zeros(R_OLS)
cv = {K: np.zeros(R_OLS) for K in list(folds) + ["LOO"]}
cv_se = {K: np.zeros(R_OLS) for K in cv}
e_mat = {5: np.zeros((R_OLS, n))}
for r in range(R_OLS):
    X = rng.normal(size=(n, p)); yv = X @ beta + rng.normal(scale=SIGMA, size=n)
    b = ols(X, yv); Z = np.c_[np.ones(n), X]
    app[r] = np.mean((yv - Z @ b) ** 2); terr[r] = true_err(b[0], b[1:])
    # K 折（固定的连续折：数据是独立同分布的，折的划分不影响分布）
    for K in folds:
        m = n // K; e = np.zeros(n)
        for k in range(K):
            te = np.arange(k * m, (k + 1) * m); tr = np.setdiff1d(np.arange(n), te)
            bk = ols(X[tr], yv[tr])
            e[te] = (yv[te] - np.c_[np.ones(len(te)), X[te]] @ bk) ** 2
        cv[K][r] = e.mean(); cv_se[K][r] = e.std(ddof=1) / np.sqrt(n)
        if K == 5:
            e_mat[5][r] = e
    H = Z @ np.linalg.solve(Z.T @ Z, Z.T)
    e_loo = ((yv - Z @ b) / (1 - np.diag(H))) ** 2
    cv["LOO"][r] = e_loo.mean(); cv_se["LOO"][r] = e_loo.std(ddof=1) / np.sqrt(n)

print(f"表观 MSE 均值 = {app.mean():.4f}（理论 σ²(n-p-1)/n = {SIGMA ** 2 * (n - p - 1) / n:.4f}）")
print(f"真实误差均值   = {terr.mean():.4f}（理论 σ²[1+1/n+(1+1/n)p/(n-p-2)] = {exact_err(n):.4f}）")
print(f"真实乐观 = 真实误差 - 表观 = {terr.mean() - app.mean():.4f}（经典近似 2(p+1)σ²/n = {2 * (p + 1) * SIGMA ** 2 / n:.4f}）")
print("\n学习曲线：训练规模 m 的期望真实误差（理论式；括号内为 b/m 近似，b=(p+1)σ²）")
for m in (20, 32, 36, 39, 40):
    print(f"  m = {m:>2}: {exact_err(m):.4f}   ({SIGMA ** 2 + (p + 1) * SIGMA ** 2 / m:.4f})")
print(f"\n{'方法':<6}{'训练规模':>8}{'模拟偏差':>10}{'理论偏差':>10}{'b/(n(K-1))':>12}{'SD':>8}{'与真实误差相关':>14}{'朴素区间覆盖率':>14}")
for K in [2, 5, 10, "LOO"]:
    m = n // 2 if K == 2 else (n * 4 // 5 if K == 5 else (n * 9 // 10 if K == 10 else n - 1))
    kk = n if K == "LOO" else K
    bias = cv[K].mean() - terr.mean()
    th = exact_err(m) - exact_err(n)
    approx = (p + 1) * SIGMA ** 2 / (n * (kk - 1))
    cover = np.mean((terr >= cv[K] - 1.96 * cv_se[K]) & (terr <= cv[K] + 1.96 * cv_se[K]))
    print(f"{str(K):<6}{m:>8}{bias:>10.4f}{th:>10.4f}{approx:>12.4f}{cv[K].std(ddof=1):>8.4f}{np.corrcoef(cv[K], terr)[0, 1]:>14.3f}{cover:>14.3f}")
print(f"{'表观':<6}{'—':>8}{app.mean() - terr.mean():>10.4f}{'':>10}{'':>12}{app.std(ddof=1):>8.4f}{np.corrcoef(app, terr)[0, 1]:>14.3f}")
print("（偏差 = CV 均值 - 拟合模型的真实误差均值；覆盖率 = 朴素区间 CV±1.96·sd(e_i)/√n 覆盖‘所拟合模型的真实误差’的比例）")

# 方差的核算：Var(CV)=(1/n)[σ²+(m-1)ω+(n-m)γ]（K=5）
E = e_mat[5]; K = 5; m = n // K
C = np.cov(E.T)
fold_id = np.arange(n) // m
same = (fold_id[:, None] == fold_id[None, :]) & ~np.eye(n, dtype=bool)
diff = fold_id[:, None] != fold_id[None, :]
sig2 = np.mean(np.diag(C)); omega = C[same].mean(); gamma = C[diff].mean()
var_formula = (sig2 + (m - 1) * omega + (n - m) * gamma) / n
print(f"\nK=5 的协方差：σ² = {sig2:.4f}, ω(同折) = {omega:.5f}, γ(异折) = {gamma:.5f}")
print(f"Var(CV) 公式 = {var_formula:.5f}; 模拟 = {cv[5].var(ddof=1):.5f}; 朴素 σ²/n = {sig2 / n:.5f}; 比值（设计效应）= {var_formula / (sig2 / n):.2f}")

# -------------------------------------------------------------------
print()
print("=" * 72)
print(f"D. 线性回归的 bootstrap 乐观（{R_BOOT_OLS} 个数据集 × B={B_OLS}）")
print("=" * 72)
opt_est = np.zeros(R_BOOT_OLS); opt_true = np.zeros(R_BOOT_OLS)
for r in range(R_BOOT_OLS):
    X = rng.normal(size=(n, p)); yv = X @ beta + rng.normal(scale=SIGMA, size=n)
    Z = np.c_[np.ones(n), X]; b = np.linalg.solve(Z.T @ Z, Z.T @ yv)
    app_r = np.mean((yv - Z @ b) ** 2); opt_true[r] = true_err(b[0], b[1:]) - app_r
    ib = rng.integers(0, n, size=(B_OLS, n))
    Zb = Z[ib]; yb = yv[ib]
    A = np.einsum('bni,bnj->bij', Zb, Zb); rhs = np.einsum('bni,bn->bi', Zb, yb)
    bb = np.linalg.solve(A, rhs[..., None])[..., 0]
    mse_star = np.mean((yb - np.einsum('bni,bi->bn', Zb, bb)) ** 2, axis=1)
    mse_orig = np.mean((yv[None, :] - bb @ Z.T) ** 2, axis=1)
    opt_est[r] = np.mean(mse_orig - mse_star)
print(f"真实乐观均值 = {opt_true.mean():.4f}; bootstrap 乐观估计均值 = {opt_est.mean():.4f}（SD {opt_est.std(ddof=1):.4f}）；"
      f"相对偏差 = {opt_est.mean() / opt_true.mean() - 1:+.1%}")
print(f"数据集之间：corr(bootstrap 乐观, 真实乐观) = {np.corrcoef(opt_est, opt_true)[0, 1]:.3f}")

# -------------------------------------------------------------------
print()
print("=" * 72)
print(f"E. 纯噪声：{P_NOISE} 个噪声变量、n={N_NOISE}，筛选前 {K_SCREEN} 个再做 Logistic（真实 C = 0.5）")
print("=" * 72)


def pipeline(X, y, k):
    yc = y - y.mean(); Xc = X - X.mean(0)
    r_ = np.abs(Xc.T @ yc) / (np.sqrt((Xc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-12)
    cols = np.argsort(-r_)[:k]
    return cols, logit_fit(np.c_[np.ones(len(y)), X[:, cols]], y)


def predict_lp(model, X):
    cols, b = model
    return b[0] + X[:, cols] @ b[1:]


def strat_folds(y, K):
    f = np.zeros(len(y), int)
    for v in (0, 1):
        ii = np.where(y == v)[0]
        f[ii] = rng.permutation(np.arange(len(ii)) % K)
    return f


names = ["表观", "bootstrap（整条流程重做）", "bootstrap（只重拟合，做错）", "10 折 CV（折内筛选）", "10 折 CV（先筛选后 CV，做错）", "70/30 留出（训练部分筛选）",
         "bootstrap 袋外 C（流程重做）", ".632（移植到 C）", ".632+（移植到 C，γ=0.5）"]
res = {nm: [] for nm in names}
for r in range(R_NOISE):
    X = rng.normal(size=(N_NOISE, P_NOISE)); yv = rng.binomial(1, PREV, size=N_NOISE).astype(float)
    if yv.sum() < 5:
        continue
    m0 = pipeline(X, yv, K_SCREEN); app_c = auc_fast(yv, predict_lp(m0, X)); res[names[0]].append(app_c)
    cols0 = m0[0]
    o1 = []; o2 = []; oob = []
    for _ in range(B_NOISE):
        ib = rng.integers(0, N_NOISE, N_NOISE); yb = yv[ib]
        if yb.min() == yb.max():
            continue
        mb = pipeline(X[ib], yb, K_SCREEN)
        o1.append(auc_fast(yb, predict_lp(mb, X[ib])) - auc_fast(yv, predict_lp(mb, X)))
        out = np.setdiff1d(np.arange(N_NOISE), ib)
        if len(out) > 5 and yv[out].min() != yv[out].max():
            oob.append(auc_fast(yv[out], predict_lp(mb, X[out])))
        bb = logit_fit(np.c_[np.ones(N_NOISE), X[ib][:, cols0]], yb)
        mb2 = (cols0, bb)
        o2.append(auc_fast(yb, predict_lp(mb2, X[ib])) - auc_fast(yv, predict_lp(mb2, X)))
    res[names[1]].append(app_c - np.mean(o1)); res[names[2]].append(app_c - np.mean(o2))
    c_oob = np.mean(oob); res[names[6]].append(c_oob)
    res[names[7]].append(0.368 * app_c + 0.632 * c_oob)
    c_oob_p = max(c_oob, 0.5)                       # 无信息值 γ = 0.5
    R_ = (app_c - c_oob_p) / (app_c - 0.5) if app_c > 0.5 else 0.0
    w_ = 0.632 / (1 - 0.368 * R_)
    res[names[8]].append((1 - w_) * app_c + w_ * c_oob_p)
    fid = strat_folds(yv, K_FOLD); a_right = []; a_wrong = []
    for k in range(K_FOLD):
        te = fid == k; tr = ~te
        if yv[te].min() == yv[te].max():
            continue
        mk = pipeline(X[tr], yv[tr], K_SCREEN); a_right.append(auc_fast(yv[te], predict_lp(mk, X[te])))
        bk = logit_fit(np.c_[np.ones(tr.sum()), X[tr][:, cols0]], yv[tr])
        a_wrong.append(auc_fast(yv[te], predict_lp((cols0, bk), X[te])))
    res[names[3]].append(np.mean(a_right)); res[names[4]].append(np.mean(a_wrong))
    f2 = strat_folds(yv, 10); te = f2 < 3; tr = ~te
    mh = pipeline(X[tr], yv[tr], K_SCREEN); res[names[5]].append(auc_fast(yv[te], predict_lp(mh, X[te])))
print(f"{'估计':<28}{'均值':>8}{'偏差(−0.5)':>12}{'SD':>8}")
for nm in names:
    v = np.array(res[nm])
    print(f"{nm:<28}{v.mean():>8.3f}{v.mean() - 0.5:>12.3f}{v.std(ddof=1):>8.3f}")
print(f"（重复 {len(res[names[0]])} 次；用时 {time.time() - t0:.0f} 秒）")

# -------------------------------------------------------------------
print()
print("=" * 72)
print(f"F. 选 k 的嵌套交叉验证（同样的纯噪声数据；候选 k ∈ {K_GRID}）")
print("=" * 72)


def cv_auc_for_k(X, y, k, K=5):
    fid = strat_folds(y, K); a = []
    for j in range(K):
        te = fid == j; tr = ~te
        if y[te].min() == y[te].max():
            continue
        a.append(auc_fast(y[te], predict_lp(pipeline(X[tr], y[tr], k), X[te])))
    return np.mean(a)


inner_best, nested, k_chosen = [], [], []
sd_single = []
for r in range(R_NEST):
    X = rng.normal(size=(N_NOISE, P_NOISE)); yv = rng.binomial(1, PREV, size=N_NOISE).astype(float)
    sc = np.array([cv_auc_for_k(X, yv, k) for k in K_GRID])
    inner_best.append(sc.max()); sd_single.append(sc.std(ddof=1)); k_chosen.append(K_GRID[int(sc.argmax())])
    fo = strat_folds(yv, 5); a_out = []
    for j in range(5):
        te = fo == j; tr = ~te
        if yv[te].min() == yv[te].max():
            continue
        sc2 = np.array([cv_auc_for_k(X[tr], yv[tr], k) for k in K_GRID])
        kstar = K_GRID[int(sc2.argmax())]
        a_out.append(auc_fast(yv[te], predict_lp(pipeline(X[tr], yv[tr], kstar), X[te])))
    nested.append(np.mean(a_out))
ib_, ne_ = np.array(inner_best), np.array(nested)
print(f"内层最优 CV 的 C：均值 {ib_.mean():.3f}（SD {ib_.std(ddof=1):.3f}），偏差 {ib_.mean() - 0.5:+.3f}")
print(f"嵌套 CV 的 C    ：均值 {ne_.mean():.3f}（SD {ne_.std(ddof=1):.3f}），偏差 {ne_.mean() - 0.5:+.3f}")
print(f"候选之间 CV 的 C 的标准差（单次）均值 {np.mean(sd_single):.3f}；独立等方差正态候选的 E[max] 上界 0.5+σ√(2 ln M) → "
      f"{0.5 + np.mean(sd_single) * np.sqrt(2 * np.log(len(K_GRID))):.3f}（正态近似上界式；实际候选并不独立）")
print("选中的 k 的分布：", {k: k_chosen.count(k) for k in K_GRID})
print(f"总用时 {time.time() - t0:.0f} 秒")

print("求解验收（所有 E/F 拟合）:", FIT_DIAGNOSTICS)

# Repository acceptance checks use exact identities where possible.
write_result('28', {'enumerated_mean_optimism_equals_formula': abs(opt_boot-2*s2hat/6) < 1e-10, 'cv_covariance_formula_equals_empirical': abs(var_formula-cv[5].var(ddof=1)) < 1e-10, 'all_logistic_fits_pass_gradient': FIT_DIAGNOSTICS['failures']==0 and FIT_DIAGNOSTICS['max_gradient'] <= 1e-5}, {'enumerated_mean_optimism': float(opt_boot), 'fit_diagnostics': FIT_DIAGNOSTICS, 'inner_best_mean_auc': float(ib_.mean()), 'outer_mean_auc': float(ne_.mean())})
