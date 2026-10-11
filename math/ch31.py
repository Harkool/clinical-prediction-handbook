from _lab_support import begin, budget, write_result, check_logistic_solution
LAB_MODE = begin('31')
# Experiment M31; independent synthetic method demonstration, not main case A.
# ch31.py —— 第31章 数值小例：采样、处理与反事实预测
# PyCharm 里右键运行；输入都在配置区
import time
import numpy as np
from scipy import special, stats
from scipy.optimize import brentq

# ============================== 配置区 ==============================
SEED = 31
# A 病例对照抽样
N_POP, A_TRUE = budget('case_control_population_n',1_000_000,100000), (-4.0, 1.0, 0.5)   # 总体人数；真实 logit = -4.0 + 1.0*x1 + 0.5*x2
N_CASE_SAMPLE = budget('case_control_each_group',3000,1000)                            # 病例、对照各抽多少
# B 采样与校准
P_PRED, PREV_B, C_B = 8, 0.10, 0.75             # 预测因子数、事件率、真实 C
N_LARGE, N_SMALL, R_SMALL, N_TEST_B = budget('balance_large_train_n',40_000,4000), 400, budget('balance_repetitions',300,8), budget('balance_test_n',200_000,10000)
# C 处理与反事实预测
N_C = budget('causal_n',400_000,20000)
GAMMA_A = (0.8, 1.2)                            # P(A=1) = expit(0.8*L1 + 1.2*L2)
BETA_Y = (-1.0, 0.8, 0.8, -1.2)                 # P(Y^a=1) = expit(-1 + 0.8*L1 + 0.8*L2 - 1.2*a)
# ===================================================================
rng = np.random.default_rng(SEED)
expit, logit = special.expit, special.logit
t0 = time.time()


def irls(X, y, sw=None, offset=None, iters=40):
    sw = np.ones(len(y)) if sw is None else sw
    off = np.zeros(len(y)) if offset is None else offset
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b + off; p = expit(eta); v = np.clip(p * (1 - p), 1e-10, None)
        W = sw * v
        bn = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-10 * np.eye(X.shape[1]), X.T @ (W * (eta - off + (y - p) / v)))
        if np.max(np.abs(bn - b)) < 1e-9:
            b = bn; break
        b = bn
    check_logistic_solution(X, y, b, offset=off, sample_weight=sw)
    return b


def wauc(y, s, w):
    """Weighted case-control pairs; ties receive half the pair weight."""
    o = np.argsort(s); s, y, w = s[o], y[o], w[o]
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("AUC weights must be finite and nonnegative")
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    pos = np.add.reduceat(w * (y == 1), starts)
    neg = np.add.reduceat(w * (y == 0), starts)
    if pos.sum() <= 0 or neg.sum() <= 0:
        raise ValueError("AUC requires both outcome groups")
    below = np.cumsum(neg) - neg
    return np.sum(pos * (below + .5 * neg)) / (pos.sum() * neg.sum())


def metrics(y, p, sw=None):
    sw = np.ones(len(y)) if sw is None else sw
    lp = logit(np.clip(p, 1e-9, 1 - 1e-9))
    slope = irls(np.c_[np.ones(len(y)), lp], y, sw)[1]
    citl = irls(np.ones((len(y), 1)), y, sw, offset=lp)[0]
    return dict(mean_p=np.average(p, weights=sw), obs=np.average(y, weights=sw), citl=citl, slope=slope,
                C=wauc(y, p, sw), brier=np.average((y - p) ** 2, weights=sw))


print("=" * 72)
print("A. 病例对照抽样：斜率不变，截距平移 ln(s1/s0)")
print("=" * 72)
x = rng.normal(size=(N_POP, 2)); a0, b1, b2 = A_TRUE
pi = expit(a0 + b1 * x[:, 0] + b2 * x[:, 1]); y = rng.binomial(1, pi)
N1, N0 = int(y.sum()), int((1 - y).sum()); phi = N1 / N_POP
print(f"总体 {N_POP:,} 人，事件 {N1:,}，事件率 φ = {phi:.4f}；真值 (β0, β1, β2) = {A_TRUE}")
ic = rng.choice(np.where(y == 1)[0], N_CASE_SAMPLE, replace=False); io = rng.choice(np.where(y == 0)[0], N_CASE_SAMPLE, replace=False)
sel = np.r_[ic, io]; Xs = np.c_[np.ones(len(sel)), x[sel]]; ys = y[sel].astype(float)
b = irls(Xs, ys)
s1, s0 = N_CASE_SAMPLE / N1, N_CASE_SAMPLE / N0
shift = np.log(s1 / s0)
print(f"抽样比 s1 = {s1:.4f}，s0 = {s0:.5f}；ln(s1/s0) = {shift:.4f}；等价写法 ln{{(n1/n0)·(1-φ)/φ}} = {np.log(1.0 * (1 - phi) / phi):.4f}")
print(f"病例对照样本拟合：β0 = {b[0]:.4f}，β1 = {b[1]:.4f}，β2 = {b[2]:.4f}；β0 - 真值 = {b[0] - a0:.4f}（预测平移 {shift:.4f}）")
b_corr = b.copy(); b_corr[0] -= shift
pop_idx = rng.choice(N_POP, budget('case_control_evaluation_n',200_000,20000), replace=False)
Xp = np.c_[np.ones(len(pop_idx)), x[pop_idx]]; yp = y[pop_idx].astype(float)
for nm, bb in (("未校正", b), ("截距校正", b_corr)):
    m = metrics(yp, expit(Xp @ bb))
    print(f"{nm:<8}平均预测 {m['mean_p']:.4f}（实际事件率 {m['obs']:.4f}）；校准截距 {m['citl']:+.3f}；校准斜率 {m['slope']:.3f}；C {m['C']:.4f}；Brier {m['brier']:.4f}")
for k in (0.5, 1.5, 2.0):
    shift_k = np.log((1 - k * phi) / (k * phi))                  # 误把事件率当成 kφ 时用的平移（病例:对照 = 1:1）
    bb = b.copy(); bb[0] = b[0] - shift_k
    print(f"  若把事件率误当成 {k}×φ：校准截距 {metrics(yp, expit(Xp @ bb))['citl']:+.3f}（预测 -ln k = {-np.log(k):+.3f}）")

print()
print("=" * 72)
print("B. 过采样、欠采样、类权重：截距平移，小样本里斜率变差")
print("=" * 72)
Zg = np.linspace(-9, 9, 36001); WZ = stats.norm.pdf(Zg) * (Zg[1] - Zg[0])
def solve_dgp(C, phi_):
    def auc_for(s):
        a_ = brentq(lambda t: np.sum(expit(t + s * Zg) * WZ) - phi_, -30, 10)
        pi_ = expit(a_ + s * Zg); f1 = pi_ * WZ / phi_; f0 = (1 - pi_) * WZ / (1 - phi_)
        return np.sum(f1 * (np.cumsum(f0) - 0.5 * f0)), a_
    s = brentq(lambda t: auc_for(t)[0] - C, 0.05, 8); return s, auc_for(s)[1]
s_b, a_b = solve_dgp(C_B, PREV_B); beta_b = np.full(P_PRED, s_b / np.sqrt(P_PRED))
print(f"数据生成：{P_PRED} 个等系数正态预测因子，a = {a_b:.4f}，每个系数 {beta_b[0]:.4f}（线性预测值 SD = {s_b:.4f}），C = {C_B}，事件率 {PREV_B}")
Xt = rng.normal(size=(N_TEST_B, P_PRED)); pit = expit(a_b + Xt @ beta_b); yt = rng.binomial(1, pit).astype(float)


def develop(X, yv, method):
    n1, n0 = int(yv.sum()), int((1 - yv).sum())
    if method == "不处理":
        return irls(np.c_[np.ones(len(yv)), X], yv)
    if method == "过采样至1:1":
        i1 = np.where(yv == 1)[0]; i0 = np.where(yv == 0)[0]
        ii = np.r_[i0, rng.choice(i1, n0, replace=True)]
        return irls(np.c_[np.ones(len(ii)), X[ii]], yv[ii])
    if method == "欠采样至1:1":
        i1 = np.where(yv == 1)[0]; i0 = np.where(yv == 0)[0]
        ii = np.r_[i1, rng.choice(i0, n1, replace=False)]
        return irls(np.c_[np.ones(len(ii)), X[ii]], yv[ii])
    if method == "类权重（事件×n0/n1）":
        sw = np.where(yv == 1, n0 / n1, 1.0)
        return irls(np.c_[np.ones(len(yv)), X], yv, sw)
    if method == "过采样 + 先验校正":
        b_ = develop(X, yv, "过采样至1:1"); r = (n0 / n0) / (n1 / n0)    # 采样后比值 1，原比值 n1/n0
        b_ = b_.copy(); b_[0] -= np.log(r); return b_
    raise ValueError(method)


methods = ["不处理", "过采样至1:1", "欠采样至1:1", "类权重（事件×n0/n1）", "过采样 + 先验校正"]
print(f"\nB1. 大样本（n={N_LARGE:,}，事件约 {int(N_LARGE * PREV_B):,}）：一次开发，大测试集 {N_TEST_B:,}")
Xd = rng.normal(size=(N_LARGE, P_PRED)); yd = rng.binomial(1, expit(a_b + Xd @ beta_b)).astype(float)
n1, n0 = int(yd.sum()), int((1 - yd).sum())
print(f"预测的截距平移 = ln(n0/n1) = {np.log(n0 / n1):.4f}（平衡到 1:1 使比值乘以 n0/n1）")
print(f"{'方法':<22}{'平均预测':>9}{'校准截距':>10}{'校准斜率':>10}{'C':>8}{'Brier':>9}")
for me in methods:
    bb = develop(Xd, yd, me); m = metrics(yt, expit(bb[0] + Xt @ bb[1:]))
    print(f"{me:<22}{m['mean_p']:>9.4f}{m['citl']:>10.3f}{m['slope']:>10.3f}{m['C']:>8.4f}{m['brier']:>9.4f}")
print(f"（测试集事件率 {yt.mean():.4f}）")

print(f"\nB2. 小样本（n={N_SMALL}，事件约 {int(N_SMALL * PREV_B)}）：{R_SMALL} 个开发样本的平均")
acc = {me: [] for me in methods}
for _ in range(R_SMALL):
    Xd = rng.normal(size=(N_SMALL, P_PRED)); yd = rng.binomial(1, expit(a_b + Xd @ beta_b)).astype(float)
    if yd.sum() < 8:
        continue
    for me in methods:
        bb = develop(Xd, yd, me); lp_t = bb[0] + Xt @ bb[1:]; p_t = expit(lp_t)
        m = metrics(yt, p_t)
        acc[me].append((m['mean_p'], m['citl'], m['slope'], m['C'], m['brier'], np.mean(np.abs(p_t - pit))))
print(f"{'方法':<22}{'平均预测':>9}{'校准截距':>10}{'校准斜率':>10}{'斜率SD':>8}{'C':>8}{'Brier':>9}{'MAPE':>8}")
for me in methods:
    a = np.array(acc[me])
    print(f"{me:<22}{a[:, 0].mean():>9.4f}{a[:, 1].mean():>10.3f}{a[:, 2].mean():>10.3f}{a[:, 2].std(ddof=1):>8.3f}{a[:, 3].mean():>8.4f}{a[:, 4].mean():>9.4f}{a[:, 5].mean():>8.4f}")

print()
print("=" * 72)
print("C. 处理与反事实预测：混杂、逆概率加权、评价")
print("=" * 72)
n = N_C
L1, L2 = rng.normal(size=n), rng.normal(size=n)
A = rng.binomial(1, expit(GAMMA_A[0] * L1 + GAMMA_A[1] * L2))
bY = BETA_Y
p_y0 = expit(bY[0] + bY[1] * L1 + bY[2] * L2)                  # 若不治疗的个体风险
p_y1 = expit(bY[0] + bY[1] * L1 + bY[2] * L2 + bY[3])
Y0 = rng.binomial(1, p_y0); Y1 = rng.binomial(1, p_y1)
Y = np.where(A == 1, Y1, Y0).astype(float)
print(f"治疗比例 {A.mean():.3f}；观察到的事件率 {Y.mean():.4f}；全员不治疗的事件率 {Y0.mean():.4f}；全员治疗 {Y1.mean():.4f}")
print(f"未治疗者的事件率 {Y[A == 0].mean():.4f}（比全员不治疗 {Y0.mean():.4f} 低，因为他们更健康）")
X = L1.reshape(-1, 1)                                         # 部署时只有 X = L1
Xd1 = np.c_[np.ones(n), L1]
# 估计倾向评分：P(A=1 | L1, L2)
g = irls(np.c_[np.ones(n), L1, L2], A.astype(float)); e_hat = expit(np.c_[np.ones(n), L1, L2] @ g)
w0 = 1 / (1 - e_hat)
print(f"倾向评分模型系数 = {g.round(3)}（真值 (0, {GAMMA_A[0]}, {GAMMA_A[1]})）；未治疗者的权重：均值 {w0[A == 0].mean():.3f}，最大 {w0[A == 0].max():.1f}，99.9% 分位 {np.quantile(w0[A == 0], 0.999):.1f}")
w_un = w0[A == 0]
print(f"未治疗者 {int((A == 0).sum()):,} 人；权重的有效样本量 (Σw)²/Σw² = {w_un.sum() ** 2 / (w_un ** 2).sum():,.0f}")
fits = {}
fits["全体合并（忽略处理）"] = irls(Xd1, Y)
fits["只用未治疗者（朴素）"] = irls(Xd1[A == 0], Y[A == 0])
fits["未治疗者 + IPW"] = irls(Xd1[A == 0], Y[A == 0], sw=w0[A == 0])
fits["真值模型（对 Y⁰ 拟合，全体）"] = irls(Xd1, Y0.astype(float))
# 真实的边际风险 E[Y⁰ | L1] 的数值积分：L2~N(0,1)
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(60); gh_w = gh_w / gh_w.sum()
def truth0(l1): return np.sum(gh_w * expit(bY[0] + bY[1] * l1 + bY[2] * gh_x))
print(f"\n{'L1':>5}{'真值 E[Y⁰|L1]':>16}" + "".join(f"{k:>24}" for k in fits))
for l1 in (-2, -1, 0, 1, 2):
    print(f"{l1:>5}{truth0(l1):>16.4f}" + "".join(f"{expit(b_[0] + b_[1] * l1):>24.4f}" for b_ in fits.values()))
print("系数 (截距, 斜率)：" + "；".join(f"{k} {tuple(np.round(b_, 3))}" for k, b_ in fits.items()))

print("\n评价：同一批模型，三种评价口径")
target_p = {k: expit(Xd1 @ b_) for k, b_ in fits.items()}
print(f"{'模型':<26}{'口径':<30}{'平均预测':>9}{'实际':>8}{'校准截距':>9}{'校准斜率':>9}{'C':>8}{'Brier':>9}")
for k in ("全体合并（忽略处理）", "只用未治疗者（朴素）", "未治疗者 + IPW"):
    p = target_p[k]
    m1 = metrics(Y[A == 0], p[A == 0])                                   # (a) 朴素：只看未治疗者
    m2 = metrics(Y[A == 0], p[A == 0], sw=w0[A == 0])                    # (b) IPW 伪总体（全员假设不治疗）
    m3 = metrics(Y0.astype(float), p)                                    # (c) 真值：全体的 Y⁰
    for lab, m in (("(a) 只看未治疗者，不加权", m1), ("(b) 未治疗者按 1/P(A=0|L) 加权", m2), ("(c) 真值：全体的潜在结局 Y⁰", m3)):
        print(f"{k:<26}{lab:<30}{m['mean_p']:>9.4f}{m['obs']:>8.4f}{m['citl']:>9.3f}{m['slope']:>9.3f}{m['C']:>8.4f}{m['brier']:>9.4f}")

print()
print("=" * 72)
print("D. 自我实现的预言（本册构造的玩具例子）")
print("=" * 72)
pH, pL = 0.30, 0.70            # 高风险组、低风险组的人数比例（模型把人分成两组）
r_treat = {"H": 0.60, "L": 0.10}      # 部署前（都治疗）的风险
r_notreat = {"H": 0.80, "L": 0.10}    # 撤掉治疗后的风险
def stats_policy(risk):
    ev = pH * risk["H"] + pL * risk["L"]
    tpr = pH * risk["H"] / ev; fpr = pH * (1 - risk["H"]) / (1 - ev)
    return ev, tpr, fpr, 0.5 * (1 + tpr - fpr)
for nm, rk in (("部署前（都治疗）", r_treat), ("部署后（H 组撤掉治疗）", r_notreat)):
    ev, tpr, fpr, auc = stats_policy(rk)
    print(f"{nm:<22}事件率 {ev:.3f}；TPR {tpr:.4f}；FPR {fpr:.4f}；C = {auc:.4f}")
print("模型对 H 组的预测 0.60；部署后 H 组的实际 0.80（校准变差，但 C 变高）；部署后每 100 人多出", round(100 * pH * (0.8 - 0.6), 1), "个事件")
print(f"总用时 {time.time() - t0:.0f} 秒")

# Repository acceptance checks use exact identities where possible.
write_result('31', {'weighted_auc_ties_receive_half_credit': abs(wauc(np.array([1,0]),np.array([.5,.5]),np.array([2.,3.]))-.5) < 1e-12, 'ipw_effective_sample_size_le_n': w_un.sum()**2/(w_un**2).sum() <= len(w_un)+1e-9, 'policy_changes_risk': abs(stats_policy(r_notreat)[0]-stats_policy(r_treat)[0]-.06) < 1e-12}, {'prior_log_odds_shift': float(shift), 'ipw_effective_sample_size': float(w_un.sum()**2/(w_un**2).sum()), 'policy_extra_events_per_100': 6.0})
