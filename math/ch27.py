from _lab_support import begin, budget, write_result, check_logistic_solution
LAB_MODE = begin('27')
# Experiment M27; independent synthetic method demonstration, not main case A.
# ch27.py —— 第27章 数值小例：评价指标的数学
# PyCharm 里右键运行即可；要改的输入全放在配置区，不用命令行参数
import numpy as np
from scipy import stats, special
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

# ============================== 配置区 ==============================
SEED = 27                 # 随机种子
N_VAL = 1000              # 验证样本人数（与第13章同规模）
A0, B0 = -2.4, 1.0        # 真实 logit = A0 + B0*x，x~N(0,1)；总体事件率约11.4%，C约0.749
THRESH = 0.15             # 净获益的阈值（第13章主要看0.10-0.20）
BIN_LIST = [5, 10, 20, 50]  # Brier 分解的分箱数
N_BIG = budget('nri_n',2_000_000,50000)         # NRI 演示用的大样本
SHARPEN = [1.5, 0.6]      # NRI 演示：把 logit 绕事件率拉开(>1)或收拢(<1)的倍数
RECAL_TRUE = (-0.3, 0.7)  # 校准斜率演示：真实 logit E[y|预测] = a0 + b0*预测线性值
# ===================================================================

rng = np.random.default_rng(SEED)
expit, logit = special.expit, special.logit


def irls_logit(X, y, offset=None, iters=30):
    """第21章的 IRLS；X 已含截距列。返回系数。"""
    off = np.zeros(len(y)) if offset is None else offset
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b + off
        p = expit(eta)
        w = np.maximum(p * (1 - p), 1e-12)
        z = eta - off + (y - p) / w
        b_new = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * z))
        if np.abs(b_new - b).max() < 1e-10:
            b = b_new
            break
        b = b_new
    check_logistic_solution(X, y, b, offset=off)
    return b


def midrank(x):
    """中位秩（并列取平均秩）"""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and xs[j] == xs[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def struct_components(y, p):
    """DeLong 的结构分量 V10(病例)、V01(对照)，用中位秩，线性对数阶（Sun & Xu 2014 的思路）"""
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    r_all = midrank(np.r_[pos, neg])
    r_pos, r_neg = midrank(pos), midrank(neg)
    v10 = (r_all[:m] - r_pos) / n
    v01 = 1 - (r_all[m:] - r_neg) / m
    return v10, v01


def auc_delong(y, p):
    v10, v01 = struct_components(y, p)
    m, n = len(v10), len(v01)
    c = v10.mean()
    var = v10.var(ddof=1) / m + v01.var(ddof=1) / n
    return c, np.sqrt(var)


def delong_cov(y, pa, pb):
    """两个相关 AUC 的差：返回 (差, 标准误, z)"""
    va10, va01 = struct_components(y, pa)
    vb10, vb01 = struct_components(y, pb)
    m, n = len(va10), len(va01)
    S10 = np.cov(np.vstack([va10, vb10]))
    S01 = np.cov(np.vstack([va01, vb01]))
    S = S10 / m + S01 / n
    d = va10.mean() - vb10.mean()
    se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return d, se, d / se


def hanley_mcneil_se(c, n1, n0):
    q1, q2 = c / (2 - c), 2 * c ** 2 / (1 + c)
    return np.sqrt((c * (1 - c) + (n1 - 1) * (q1 - c ** 2) + (n0 - 1) * (q2 - c ** 2)) / (n1 * n0))


def brier(y, p):
    return np.mean((y - p) ** 2)


def murphy_bins(y, p, k):
    """等频分箱的 Murphy 分解（每箱用箱内平均预测）"""
    order = np.argsort(p)
    idx = np.array_split(order, k)
    n, obar = len(y), y.mean()
    rel = res = 0.0
    for ii in idx:
        nk, ok, pk = len(ii), y[ii].mean(), p[ii].mean()
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - obar) ** 2
    return rel / n, res / n, obar * (1 - obar)


def corp(y, p):
    """CORP：用 PAV 等渗回归做重校准，Brier = MCB - DSC + UNC"""
    iso = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip")
    pt = iso.fit_transform(p, y)
    bs, bs_rec, bs_const = brier(y, p), brier(y, pt), brier(y, np.full_like(p, y.mean()))
    return bs - bs_rec, bs_const - bs_rec, bs_const  # MCB, DSC, UNC


print("=" * 70)
print("A. 10 人手算：AUC 与 Mann–Whitney、DeLong 方差")
print("=" * 70)
pos = np.array([0.9, 0.7, 0.6, 0.4])
neg = np.array([0.8, 0.5, 0.35, 0.3, 0.2, 0.1])
y10 = np.r_[np.ones(4), np.zeros(6)].astype(int)
p10 = np.r_[pos, neg]
U = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
print("U =", U, " C = U/(n1 n0) =", round(U / 24, 6), " sklearn:", round(roc_auc_score(y10, p10), 6))
v10, v01 = struct_components(y10, p10)
print("V10 =", v10.round(4), " V01 =", v01.round(4))
print("s10^2 =", round(v10.var(ddof=1), 6), " s01^2 =", round(v01.var(ddof=1), 6))
c10, se10 = auc_delong(y10, p10)
print("DeLong: C =", round(c10, 6), " SE =", round(se10, 6),
      " 95%区间 =", (round(c10 - 1.96 * se10, 3), round(c10 + 1.96 * se10, 3)))
print("Hanley–McNeil SE =", round(hanley_mcneil_se(c10, 4, 6), 6))
# 带并列：把一个对照改成 0.6（与病例 0.6 并列）
p10t = p10.copy(); p10t[4] = 0.6
Ut = sum((a > b) + 0.5 * (a == b) for a in p10t[:4] for b in p10t[4:])
print("并列版：U =", Ut, " C =", round(Ut / 24, 6), " sklearn:", round(roc_auc_score(y10, p10t), 6),
      " 中位秩法:", round(auc_delong(y10, p10t)[0], 6))
# 暴力 DeLong（双重循环）核对快速法
psi = lambda a, b: (a > b) + 0.5 * (a == b)
v10b = np.array([np.mean([psi(a, b) for b in neg]) for a in pos])
v01b = np.array([np.mean([psi(a, b) for a in pos]) for b in neg])
print("暴力与中位秩法的最大差:", max(abs(v10b - v10).max(), abs(v01b - v01).max()))

print()
print("=" * 70)
print("B. 验证样本：AUC 不管校准（单调变换）")
print("=" * 70)
x = rng.normal(size=N_VAL)
eta_true = A0 + B0 * x
pi = expit(eta_true)
y = rng.binomial(1, pi)
n1, n0 = int(y.sum()), int((1 - y).sum())
print(f"n={N_VAL}, 事件 {n1}, 事件率 {y.mean():.4f}; 总体事件率 {expit(A0 + B0 * rng.normal(size=budget('population_risk_n',2_000_000,30000))).mean():.4f}")
variants = {
    "真实风险 π": eta_true,
    "过度自信 η×1.5": 1.5 * eta_true,
    "过于保守 η×0.6": 0.6 * eta_true,
    "整体偏高 η+0.8": eta_true + 0.8,
}
print(f"{'预测':<14}{'C':>8}{'Brier':>9}{'缩放Brier':>10}{'截距(斜率=1)':>14}{'斜率':>8}{'O/E':>8}")
ref = y.mean() * (1 - y.mean())
for name, eta in variants.items():
    p = expit(eta)
    c = roc_auc_score(y, p)
    X = np.c_[np.ones(N_VAL), eta]
    b = irls_logit(X, y)
    citl = irls_logit(np.ones((N_VAL, 1)), y, offset=eta)[0]
    print(f"{name:<14}{c:8.4f}{brier(y, p):9.4f}{1 - brier(y, p) / ref:10.4f}{citl:14.3f}{b[1]:8.3f}{y.mean() / p.mean():8.3f}")

c, se = auc_delong(y, expit(eta_true))
boots = []
for _ in range(budget('auc_bootstrap',1000,100)):
    i = rng.integers(0, N_VAL, N_VAL)
    if y[i].min() == y[i].max():
        continue
    boots.append(roc_auc_score(y[i], pi[i]))
print(f"AUC {c:.4f}; DeLong SE {se:.4f}; Hanley–McNeil SE {hanley_mcneil_se(c, n1, n0):.4f}; bootstrap SE {np.std(boots, ddof=1):.4f}")
print("DeLong 95%区间:", (round(c - 1.96 * se, 3), round(c + 1.96 * se, 3)),
      " bootstrap 百分位区间:", tuple(np.percentile(boots, [2.5, 97.5]).round(3)))
# 两个相关 AUC 之差：真实风险 vs 只用 x 的符号（0/1 预测，有大量并列）
d, sed, z = delong_cov(y, pi, (x > 0).astype(float))
print(f"ΔAUC(真实风险 - 二分化) = {d:.4f}, DeLong SE = {sed:.4f}, z = {z:.2f}")

print()
print("=" * 70)
print("C. Brier 分解")
print("=" * 70)
# C1 手算的离散例子：20 人
fc = np.r_[np.full(8, .1), np.full(6, .3), np.full(4, .7), np.full(2, .9)]
yy = np.r_[[1] + [0] * 7, [1] * 3 + [0] * 3, [1] * 3 + [0], [1] * 2]
ob = yy.mean()
rel = res = 0.0
for v in np.unique(fc):
    m_ = fc == v
    rel += m_.sum() * (v - yy[m_].mean()) ** 2
    res += m_.sum() * (yy[m_].mean() - ob) ** 2
rel, res, unc = rel / 20, res / 20, ob * (1 - ob)
print(f"20 人：BS = {brier(yy, fc):.5f}; REL = {rel:.5f}, RES = {res:.5f}, UNC = {unc:.5f}; REL-RES+UNC = {rel - res + unc:.5f}")
mcb, dsc, unc2 = corp(yy.astype(float), fc)
print(f"      CORP: MCB = {mcb:.5f}, DSC = {dsc:.5f}, UNC = {unc2:.5f}; MCB-DSC+UNC = {mcb - dsc + unc2:.5f}")
# C2 连续预测：分箱数改变，分解就变；CORP 不依赖分箱
nb = 4000
xb = rng.normal(size=nb)
etab = A0 + B0 * xb
yb = rng.binomial(1, expit(etab))
pb = expit(1.5 * etab)  # 过度自信
bsb = brier(yb, pb)
print(f"\n连续预测（n={nb}，过度自信 η×1.5）：BS = {bsb:.5f}")
print(f"{'分箱数':>6}{'REL':>10}{'RES':>10}{'UNC':>10}{'REL-RES+UNC':>14}{'与BS之差':>11}")
for k in BIN_LIST:
    r_, s_, u_ = murphy_bins(yb, pb, k)
    print(f"{k:>6}{r_:10.5f}{s_:10.5f}{u_:10.5f}{r_ - s_ + u_:14.5f}{bsb - (r_ - s_ + u_):11.5f}")
mcb, dsc, unc3 = corp(yb.astype(float), pb)
print(f"{'CORP':>6}{mcb:10.5f}{dsc:10.5f}{unc3:10.5f}{mcb - dsc + unc3:14.5f}{bsb - (mcb - dsc + unc3):11.5f}")
print("缩放 Brier = 1 - BS/UNC =", round(1 - bsb / unc3, 5), "; (DSC - MCB)/UNC =", round((dsc - mcb) / unc3, 5))
# 第13章式的核对：缩放 Brier
print("独立手算输入：BS=0.0926, 事件率 0.115 → 缩放 Brier =", round(1 - 0.0926 / (0.115 * 0.885), 4))

print()
print("=" * 70)
print("D. 适当评分规则与基本评分（决策阈值损失）")
print("=" * 70)
# D1 期望评分的最小点：Brier 与对数评分在 p = q 处最小；0/1 误分类在 p<.5 同一平台
q = 0.3
grid = np.linspace(0.01, 0.99, 99)
exp_brier = q * (1 - grid) ** 2 + (1 - q) * grid ** 2
exp_log = -(q * np.log(grid) + (1 - q) * np.log(1 - grid))
exp_acc = np.where(grid > 0.5, 1 - q, q)  # 以 0.5 为界的误分类损失的期望
print("q = 0.3：Brier 最小点", grid[exp_brier.argmin()], " 对数评分最小点", grid[exp_log.argmin()],
      " 误分类损失最小点个数", int((exp_acc == exp_acc.min()).sum()), "（适当、非严格适当）")
print("误分类损失：p<=0.5:", exp_acc[grid <= .5][0], "; p>0.5:", exp_acc[grid > .5][0])
assert np.allclose(exp_acc[grid <= .5], q) and np.allclose(exp_acc[grid > .5], 1-q)


def elementary(y_, p_, t):
    """基本评分：(1-t)*[y=1,p<=t] + t*[y=0,p>t]"""
    return (1 - t) * ((y_ == 1) & (p_ <= t)) + t * ((y_ == 0) & (p_ > t))


# D2 对 t 积分得 Brier/2；以 1/(t(1-t)) 加权得对数评分（单点验证）
ts = np.linspace(1e-6, 1 - 1e-6, 400001)
for yv, pv in [(1, 0.3), (0, 0.3), (1, 0.8), (0, 0.8)]:
    s = elementary(np.array([yv]), np.array([pv]), ts[:, None])[:, 0]
    i_b = np.trapezoid(s, ts)
    i_l = np.trapezoid(s / (ts * (1 - ts)), ts)
    print(f"y={yv}, p={pv}: ∫S_t dt = {i_b:.5f} (Brier/2 = {(yv - pv) ** 2 / 2:.5f});"
          f" ∫S_t/(t(1-t)) dt = {i_l:.4f} (对数损失 = {-np.log(pv if yv else 1 - pv):.4f})")

print()
print("=" * 70)
print("E. 净获益：手算例子与基本评分恒等式")
print("=" * 70)
# E1 100 人手算：20 例事件，阈值 0.2；模型判阳 30 人，其中 16 个真阳性
n_, E_, t_ = 100, 20, 0.2
TP, FP = 16, 14
FN = E_ - TP
w_ = t_ / (1 - t_)
nb_model = TP / n_ - FP / n_ * w_
nb_all = E_ / n_ - (n_ - E_) / n_ * w_
S_t = (1 - t_) * FN / n_ + t_ * FP / n_
print(f"NB(模型) = {nb_model:.4f}, NB(全部治疗) = {nb_all:.4f}, 每100人多得净真阳性 {100 * nb_model:.1f};"
      f" 基本评分 = {S_t:.4f}; (1-t)(φ-NB) = {(1 - t_) * (E_ / n_ - nb_model):.4f}")
# E2 验证样本上的核对
p_true = expit(eta_true)
pred_pos = p_true >= THRESH
TP2, FP2 = np.sum(pred_pos & (y == 1)), np.sum(pred_pos & (y == 0))
w2 = THRESH / (1 - THRESH)
nb2 = TP2 / N_VAL - FP2 / N_VAL * w2
nb_all2 = y.mean() - (1 - y.mean()) * w2
St2 = elementary(y, p_true, THRESH).mean()
print(f"阈值 {THRESH}: TP={TP2}, FP={FP2}; NB = {nb2:.5f}; 全部治疗 NB = {nb_all2:.5f}")
print(f"     E[S_t] = {St2:.6f};  (1-t)(φ̂-NB) = {(1 - THRESH) * (y.mean() - nb2):.6f}")
# NB 的方差：Z_i = d_i(y_i - w(1-y_i))，Var(NB) = (TP + w^2 FP - NB^2)/n
var_nb = (TP2 / N_VAL + w2 ** 2 * FP2 / N_VAL - nb2 ** 2) / N_VAL
print(f"NB 的标准误 = {np.sqrt(var_nb):.5f}；若要 95% 区间半宽 ≤0.01，需 n ≈ {1.96 ** 2 * (TP2 / N_VAL + w2 ** 2 * FP2 / N_VAL - nb2 ** 2) / 0.01 ** 2:.0f}")
# Brier = 2 ∫ E[S_t] dt（用细网格）
tg = np.linspace(0, 1, 2001)
Sbar = np.array([elementary(y, p_true, t).mean() for t in tg])
print(f"Brier 直接 = {brier(y, p_true):.5f};  2∫E[S_t]dt = {2 * np.trapezoid(Sbar, tg):.5f}")

print()
print("=" * 70)
print("F. 校准斜率：两个矩方程与恢复真值")
print("=" * 70)
a0, b0 = RECAL_TRUE
nr = budget('recalibration_n',1_000_000,20000)
xr = rng.normal(size=nr)
eta_r = A0 + B0 * xr
yr = rng.binomial(1, expit(eta_r))
eta_hat = (eta_r - a0) / b0  # 使 logit E[y | eta_hat] = a0 + b0*eta_hat
bhat = irls_logit(np.c_[np.ones(nr), eta_hat], yr)
res_ = yr - expit(bhat[0] + bhat[1] * eta_hat)
print(f"真值 (a0,b0) = ({a0}, {b0});  估计 (a,b) = ({bhat[0]:.4f}, {bhat[1]:.4f})")
print(f"矩方程残差（应≈0）：mean(res) = {res_.mean():.2e}, mean(res*η̂) = {(res_ * eta_hat).mean():.2e}")
p_h = expit(eta_hat)
p_rec = expit(bhat[0] + bhat[1] * eta_hat)
print(f"重校准前后：C = {roc_auc_score(yr, p_h):.5f} → {roc_auc_score(yr, p_rec):.5f}; Brier = {brier(yr, p_h):.5f} → {brier(yr, p_rec):.5f}")


# F2：估计噪声使斜率变小（第22章步5的一阶式：斜率≈V/(V+s²)，V为线性预测值的信号方差）
print(f"\nF2. 线性预测值带独立噪声 δ~N(0,s²) 时的斜率（n={nr:,}；V=B0²={B0 ** 2:.2f}）")
print(f"{'s²':>6}{'一阶式 V/(V+s²)':>18}{'模拟斜率':>10}{'模拟截距(斜率=1)':>18}")
for s2 in (0.0, 0.1, 0.25, 0.5, 1.0):
    eta_n = eta_r + rng.normal(scale=np.sqrt(s2), size=nr)
    bb = irls_logit(np.c_[np.ones(nr), eta_n], yr)
    cc = irls_logit(np.ones((nr, 1)), yr, offset=eta_n)[0]
    print(f"{s2:>6.2f}{B0 ** 2 / (B0 ** 2 + s2):>18.4f}{bb[1]:>10.4f}{cc:>18.4f}")

print()
print("=" * 70)
print("G. NRI 奖励过度自信（总体水平的大样本）")
print("=" * 70)
xg = rng.normal(size=N_BIG)
etag = A0 + B0 * xg
pig = expit(etag)
yg = rng.binomial(1, pig)
phi = yg.mean()
lphi = logit(phi)
print(f"n={N_BIG}, 事件率 {phi:.4f}; 旧模型 = 真实风险（已是最优）")
_tpr = (pig[yg == 1] > phi).mean(); _fpr = (pig[yg == 0] > phi).mean()
print(f"阈值取事件率 {phi:.4f}：TPR = {_tpr:.4f}, FPR = {_fpr:.4f}, 2(TPR-FPR) = {2 * (_tpr - _fpr):.4f}（应等于 k>1 时的 NRI）")
print(f"{'倍数k':>6}{'NRI(>0)':>10}{'ΔC':>9}{'ΔBrier':>10}{'Δ对数损失':>11}{'ΔNB@0.10':>10}{'ΔNB@0.15':>10}{'ΔNB@0.20':>10}")
for k in SHARPEN:
    p_new = expit(lphi + k * (etag - lphi))
    up, dn = p_new > pig, p_new < pig
    nri = (up[yg == 1].mean() - dn[yg == 1].mean()) - (up[yg == 0].mean() - dn[yg == 0].mean())
    dC = roc_auc_score(yg, p_new) - roc_auc_score(yg, pig)
    dB = brier(yg, p_new) - brier(yg, pig)
    ll = lambda p_: -np.mean(yg * np.log(p_) + (1 - yg) * np.log(1 - p_))
    dL = ll(p_new) - ll(pig)

    def nb(p_, t):
        pos_ = p_ >= t
        return np.mean(pos_ & (yg == 1)) - np.mean(pos_ & (yg == 0)) * t / (1 - t)
    d1, d2, d3 = (nb(p_new, t) - nb(pig, t) for t in (0.10, 0.15, 0.20))
    print(f"{k:>6}{nri:10.3f}{dC:9.5f}{dB:10.5f}{dL:11.5f}{d1:10.5f}{d2:10.5f}{d3:10.5f}")
print("解释：Δ>0 的 Brier / 对数损失表示新模型更差；NRI 却在 k>1 时为正。")

# Repository acceptance checks use exact identities where possible.
write_result('27', {'auc_matches_pairwise_probability': abs(c10-U/24) < 1e-12, 'delong_fast_matches_brute_force': max(abs(v10b-v10).max(),abs(v01b-v01).max()) < 1e-12, 'correction_score_equations': abs(res_.mean()) < 1e-8 and abs((res_*eta_hat).mean()) < 1e-8}, {'hand_auc': float(c10), 'hand_delong_se': float(se10), 'recalibration_coefficients': bhat.tolist()})
