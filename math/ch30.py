from _lab_support import begin, budget, write_result, check_logistic_solution
LAB_MODE = begin('30')
# Experiment M30; independent synthetic method demonstration, not main case A.
# ch30.py —— 第30章 数值小例：样本量公式
# PyCharm 里右键运行；输入都在配置区
import time
import numpy as np
from scipy import special, stats
from scipy.optimize import brentq

# ============================== 配置区 ==============================
SEED = 30
P_PAR = 18                 # Independent M30 parameter budget
PREV = 0.1333              # Independent M30 sensitivity scenario
C_EXPECT = 0.79            # 预期 C
S_TARGET = 0.90            # 目标全局收缩因子
DELTA_NAG = 0.05           # 准则2：表观与校正 Nagelkerke R² 之差的上界
DELTA_RISK = 0.05          # 准则3：总体事件率的误差界（绝对值）
LP_MEAN, LP_SD = -2.218, 1.012   # External-validation experiment M30-D: assumed LP distribution
W_OE, W_SLOPE, W_C = 0.2, 0.2, 0.1   # 外部验证：95% 区间的目标宽度（O/E 取对数尺度）
NB_T, NB_HALF = 0.15, 0.01 # 净获益：阈值与区间半宽
SIM_C = [0.70, 0.79, 0.90] # 模拟：三种模型强度
R_SIM, N_TEST = budget('development_repetitions',300,10), budget('test_n',100_000,5000)
R_CHECK = budget('se_repetitions',600,40)            # D2：公式核对的重复数
# ===================================================================
rng = np.random.default_rng(SEED)
expit = special.expit
t0 = time.time()

Z = np.linspace(-9, 9, 36001); DZ = Z[1] - Z[0]; WZ = stats.norm.pdf(Z) * DZ


def max_r2cs(phi):
    return 1 - np.exp(2 * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi)))


def model_logitnormal(C, phi):
    """logit π = a + s·z，z~N(0,1)：解出 s、a，使 AUC = C、事件率 = phi；返回 (R²_CS, s, a)"""
    def auc_for(s):
        a = brentq(lambda a_: np.sum(expit(a_ + s * Z) * WZ) - phi, -30, 10)
        pi = expit(a + s * Z)
        f1 = pi * WZ / phi; f0 = (1 - pi) * WZ / (1 - phi)
        F0 = np.cumsum(f0) - 0.5 * f0
        return np.sum(f1 * F0), a
    s = brentq(lambda s_: auc_for(s_)[0] - C, 0.05, 8)
    _, a = auc_for(s)
    pi = expit(a + s * Z)
    e_model = np.sum(WZ * (pi * np.log(pi) + (1 - pi) * np.log(1 - pi)))
    e_null = phi * np.log(phi) + (1 - phi) * np.log(1 - phi)
    r2 = 1 - np.exp(-2 * (e_model - e_null))
    return r2, s, a


def model_binormal(C, phi):
    """两个结局组里线性预测值都是方差相同的正态：C=Φ(Δ/√2)；此时 logit P(Y=1|lp) 对 lp 线性"""
    delta = np.sqrt(2) * stats.norm.ppf(C)           # (μ1-μ0)/σ
    lp = np.linspace(-12, 12, 48001); dl = lp[1] - lp[0]
    f0 = stats.norm.pdf(lp, loc=0, scale=1); f1 = stats.norm.pdf(lp, loc=delta, scale=1)
    dens = (1 - phi) * f0 + phi * f1
    pi = phi * f1 / dens
    e_model = np.sum(dens * (pi * np.log(np.clip(pi, 1e-300, 1)) + (1 - pi) * np.log(np.clip(1 - pi, 1e-300, 1)))) * dl
    e_null = phi * np.log(phi) + (1 - phi) * np.log(1 - phi)
    return 1 - np.exp(-2 * (e_model - e_null))


def sample_size(C, phi, p=P_PAR, S=S_TARGET, delta_nag=DELTA_NAG, delta_risk=DELTA_RISK, r2=None):
    r2 = model_logitnormal(C, phi)[0] if r2 is None else r2
    mx = max_r2cs(phi)
    n1 = p / ((S - 1) * np.log(1 - r2 / S))
    S2 = r2 / (r2 + delta_nag * mx)
    n2 = p / ((S2 - 1) * np.log(1 - r2 / S2))
    n3 = (1.96 / delta_risk) ** 2 * phi * (1 - phi)
    n = max(n1, n2, n3)
    return dict(r2=r2, max_r2=mx, S2=S2, n1=n1, n2=n2, n3=n3, n=int(np.ceil(n)), events=n * phi, epp=n * phi / p)


print("A0. 公开规划输入（p=18, 率=.12, C=.80, S=.9）；仅本脚本积分算法")
planning = sample_size(.80,.12)
print(planning)
print("A0 sensitivity C=.70, phi=.12:", sample_size(.70,.12))
print("理论册约1170为规划示例；此输出不是 pmsampsize 包实跑，也不是主线 A 的观察结果。")
print()
print("=" * 72)
print("A1. 独立敏感性输入 M30-A1（p=18, 事件率 0.1333, 预期 C=0.79, S=0.9）")
print("=" * 72)
r2_a, s_a, a_a = model_logitnormal(C_EXPECT, PREV)
r2_b = model_binormal(C_EXPECT, PREV)
print(f"max R²_CS = 1-exp(2[φlnφ+(1-φ)ln(1-φ)]) = {max_r2cs(PREV):.4f}")
print(f"R²_CS（logit-线性-正态设定）= {r2_a:.4f}  [s={s_a:.4f}, a={a_a:.4f}]；R²_CS（双正态等方差设定）= {r2_b:.4f}；独立手算输入 R²=0.1286")
r = sample_size(C_EXPECT, PREV)
rb = sample_size(C_EXPECT, PREV, r2=0.1286)
print(f"用本稿的 R²：准则1 n={r['n1']:.1f}；准则2 S={r['S2']:.4f}, n={r['n2']:.1f}；准则3 n={r['n3']:.1f}；n={r['n']}，事件 {r['events']:.1f}，每参数 {r['epp']:.2f}")
print(f"用独立指定的 R²=0.1286：准则1 n={rb['n1']:.1f}；准则2 S={rb['S2']:.4f}, n={rb['n2']:.1f}；准则3 n={rb['n3']:.1f}；n={rb['n']}；不作为主线 A 或当前 R 软件输出")

print()
print("=" * 72)
print("B. 独立 logit-normal 积分的敏感性网格（p=18, S=0.9）")
print("=" * 72)
prevs = [0.05, 0.10, 0.1333, 0.20, 0.30]
print("C / 事件率".ljust(10) + "".join(f"{pv:>10.4f}" for pv in prevs))
for C in (.70, .75, .78, .79, .80, .85):
    print(f"{C:<10}" + "".join(f"{sample_size(C,pv)['n']:>10d}" for pv in prevs))

print()
print("=" * 72)
print("C. ‘每变量 10 个事件’：同一套准则下，所需的每参数事件数（EPP = nφ/p）")
print("=" * 72)
print("C / 事件率".ljust(10) + "".join(f"{pv:>10.4f}" for pv in prevs))
for C in (0.70, 0.75, 0.79, 0.85, 0.90):
    print(f"{C:<10}" + "".join(f"{sample_size(C, pv)['epp']:>10.1f}" for pv in prevs))
grid = [(C, pv) for C in (0.70, 0.75, 0.79, 0.85, 0.90) for pv in prevs]
epps = np.array([sample_size(C, pv)['epp'] for C, pv in grid])
print(f"EPP 范围：{epps.min():.1f} – {epps.max():.1f}；所需 EPP ≤ 10 的格子占 {np.mean(epps <= 10):.0%}（10 够用甚至多余），"
      f"> 10 的占 {np.mean(epps > 10):.0%}（10 不够）")
print("p 的影响（C=0.79, 事件率 0.1333）：每参数事件数 EPP =",
      {p_: round(float(sample_size(0.79, 0.1333, p=p_)['epp']), 2) for p_ in (1, 2, 3, 5, 18, 60)},
      "；准则1 的 n =", {p_: round(float(sample_size(0.79, 0.1333, p=p_)['n1'])) for p_ in (1, 2, 3, 5, 18, 60)})

print()
print("=" * 72)
print("D. 外部验证样本量（线性预测值 ~ N(%.3f, %.3f²)；事件率 %.4f）" % (LP_MEAN, LP_SD, PREV))
print("=" * 72)
lp = np.linspace(LP_MEAN - 10 * LP_SD, LP_MEAN + 10 * LP_SD, 40001); dl = lp[1] - lp[0]
f_lp = stats.norm.pdf(lp, LP_MEAN, LP_SD) * dl
pi_lp = expit(lp); phi_ev = np.sum(pi_lp * f_lp)
wgt = pi_lp * (1 - pi_lp)
Ew, Ewl, Ewl2 = np.sum(wgt * f_lp), np.sum(wgt * lp * f_lp), np.sum(wgt * lp ** 2 * f_lp)
info_slope = Ewl2 - Ewl ** 2 / Ew            # 每人的校准斜率信息量（a、b 联合时）
print(f"该分布下的事件率 = {phi_ev:.4f}")
# O/E
se_t = W_OE / 3.92
n_oe = (1 - PREV) / (PREV * se_t ** 2)
print(f"O/E（ln 尺度，宽度 {W_OE}）：SE 目标 {se_t:.4f}；n = (1-φ)/(φ·SE²) = {n_oe:.0f}")
# 校准斜率
se_t = W_SLOPE / 3.92
n_sl = 1 / (se_t ** 2 * info_slope)
print(f"校准斜率（宽度 {W_SLOPE}）：每人信息量 = {info_slope:.5f}；n = 1/(SE²·信息量) = {n_sl:.0f}")
# C
def hm_width(C, n, phi):
    n1, n0 = n * phi, n * (1 - phi)
    q1, q2 = C / (2 - C), 2 * C ** 2 / (1 + C)
    var = (C * (1 - C) + (n1 - 1) * (q1 - C ** 2) + (n0 - 1) * (q2 - C ** 2)) / (n1 * n0)
    return 3.92 * np.sqrt(var)
def newcombe_width(C, n, phi):
    # CRAN pmvalsampsize_bin.R: both numerator factors use n/2 - 1.
    var = C*(1-C)/(n*n*phi*(1-phi)) * (
        1+(n/2-1)*(1-C)/(2-C)+(n/2-1)*C/(1+C))
    return 3.92*np.sqrt(var)

for C in (0.75, 0.79):
    for label, width_function in [("Hanley–McNeil",hm_width),
                                   ("Newcombe (CRAN formula)",newcombe_width)]:
        root_n = brentq(lambda n:width_function(C,n,PREV)-W_C,100,20000)
        n_c = int(np.ceil(root_n))
        print(f"C={C}, {label}: continuous n={root_n:.4f}; integer n={n_c}; events={n_c*PREV:.1f}")

# 净获益
d_pos = pi_lp >= NB_T
TPr = np.sum((pi_lp * f_lp)[d_pos]); FPr = np.sum(((1 - pi_lp) * f_lp)[d_pos])
w_t = NB_T / (1 - NB_T); NBv = TPr - FPr * w_t
varZ = TPr + w_t ** 2 * FPr - NBv ** 2
n_nb = 1.96 ** 2 * varZ / NB_HALF ** 2
print(f"净获益（阈值 {NB_T}）：NB = {NBv:.4f}（TP率 {TPr:.4f}，FP率 {FPr:.4f}）；半宽 {NB_HALF} 所需 n = {n_nb:.0f}，事件 {n_nb * phi_ev:.0f}")
print(f"四个目标取最大：{max(n_oe, n_sl, n_nb):.0f}（C 统计量在上面）")
# 区间宽度与人数：n ∝ 1/宽度²
print("校准斜率：宽度 0.2 → 0.3 时 n =", round(1 / ((0.3 / 3.92) ** 2 * info_slope)), "；宽度 0.4 →", round(1 / ((0.4 / 3.92) ** 2 * info_slope)))
print("‘至少 100 个事件’对应的总人数（事件率 %.4f）= %.0f；该人数下校准斜率的 95%% 区间宽度 = %.3f" %
      (PREV, 100 / PREV, 3.92 / np.sqrt(100 / PREV * info_slope)))

print()
print("D2. 模拟核对 D 里的标准误公式（模型完全校准：lp~N(%.3f, %.3f²)，y~Bernoulli(expit(lp))；每项 %d 次）" % (LP_MEAN, LP_SD, R_CHECK))
def irls_s(X, y, iters=25):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b; p = expit(eta); w = np.clip(p * (1 - p), 1e-10, None)
        bn = np.linalg.solve(X.T @ (w[:, None] * X) + 1e-10 * np.eye(X.shape[1]), X.T @ (w * (eta + (y - p) / w)))
        if np.max(np.abs(bn - b)) < 1e-9:
            b = bn; break
        b = bn
    check_logistic_solution(X, y, b)
    return b
def auc_s(y, sc):
    from scipy.stats import rankdata
    r = rankdata(sc); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
n_sl_i, n_oe_i, n_nb_i = int(round(n_sl)), int(round(n_oe)), int(round(n_nb))
sl_, oe_, oe_fix, nb_ = [], [], [], []
for _ in range(R_CHECK):
    lp_ = rng.normal(LP_MEAN, LP_SD, n_sl_i); yv = rng.binomial(1, expit(lp_)).astype(float)
    sl_.append(irls_s(np.c_[np.ones(n_sl_i), lp_], yv)[1])
for _ in range(R_CHECK):
    lp_ = rng.normal(LP_MEAN, LP_SD, n_oe_i); p_ = expit(lp_); yv = rng.binomial(1, p_)
    oe_.append(np.log(yv.sum() / p_.sum())); oe_fix.append(np.log(yv.sum() / (n_oe_i * phi_ev)))
for _ in range(R_CHECK):
    lp_ = rng.normal(LP_MEAN, LP_SD, n_nb_i); p_ = expit(lp_); yv = rng.binomial(1, p_); d_ = p_ >= NB_T
    nb_.append(np.mean(d_ * yv) - np.mean(d_ * (1 - yv)) * w_t)
print(f"校准斜率：n={n_sl_i}，公式 SE = {W_SLOPE / 3.92:.4f}；模拟 SD = {np.std(sl_, ddof=1):.4f}；均值 {np.mean(sl_):.3f}")
print(f"ln(O/E)：n={n_oe_i}，公式 SE = {W_OE / 3.92:.4f}；模拟 SD（E=Σp 随样本变）= {np.std(oe_, ddof=1):.4f}；模拟 SD（E 取固定值）= {np.std(oe_fix, ddof=1):.4f}")
print(f"净获益：n={n_nb_i}，公式 SE = {NB_HALF / 1.96:.4f}；模拟 SD = {np.std(nb_, ddof=1):.4f}；均值 {np.mean(nb_):.4f}（公式里的 NB = {NBv:.4f}）")
# C：先用大样本得到该 lp 分布隐含的总体 C，再比较 Hanley–McNeil 与模拟
lp_big = rng.normal(LP_MEAN, LP_SD, budget('auc_population_n',400_000,30000)); y_big = rng.binomial(1, expit(lp_big)).astype(float)
C_pop = auc_s(y_big, lp_big)
n_c_i = 900; c_ = []
for _ in range(R_CHECK):
    lp_ = rng.normal(LP_MEAN, LP_SD, n_c_i); yv = rng.binomial(1, expit(lp_)).astype(float)
    c_.append(auc_s(yv, lp_))
print(f"C：该 lp 分布隐含的总体 C = {C_pop:.4f}；n={n_c_i}，Hanley–McNeil 的 SE = {hm_width(C_pop, n_c_i, phi_ev) / 3.92:.4f}；模拟 SD = {np.std(c_, ddof=1):.4f}")
for n_try in (680, 785):
    c2 = []
    for _ in range(R_CHECK):
        lp_ = rng.normal(LP_MEAN, LP_SD, n_try); yv = rng.binomial(1, expit(lp_)).astype(float)
        c2.append(auc_s(yv, lp_))
    print(f"     n={n_try}：模拟 SD = {np.std(c2, ddof=1):.4f}，95% 区间宽度 ≈ {3.92 * np.std(c2, ddof=1):.3f}；Hanley–McNeil 宽度 = {hm_width(C_pop, n_try, phi_ev):.3f}")

print()
print("=" * 72)
print("E. 模拟：按准则1 算出的 n，新数据里的平均校准斜率是否 ≥ 0.9？（p=18，事件率 %.4f，每种强度 %d 个开发样本）" % (PREV, R_SIM))
print("=" * 72)
def irls(X, y, iters=30):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b; p = expit(eta); w = np.clip(p * (1 - p), 1e-10, None)
        bn = np.linalg.solve(X.T @ (w[:, None] * X) + 1e-9 * np.eye(X.shape[1]), X.T @ (w * (eta + (y - p) / w)))
        if np.max(np.abs(bn - b)) < 1e-9:
            b = bn; break
        b = bn
    check_logistic_solution(X, y, b)
    return b

print(f"{'C':>5}{'n取法':>10}{'n':>7}{'事件':>6}{'EPP':>6}{'平均斜率':>9}{'斜率SD':>8}{'<0.9 的比例':>11}{'MAPE':>8}")
for C in SIM_C:
    sz = sample_size(C, PREV); r2, s_, a = model_logitnormal(C, PREV)
    beta_j = np.full(P_PAR, s_ / np.sqrt(P_PAR))                 # 等系数，使线性预测值的 SD = s
    Xt = rng.normal(size=(N_TEST, P_PAR)); eta_t = a + Xt @ beta_j; pi_t = expit(eta_t)
    yt = rng.binomial(1, pi_t).astype(float)
    for label, n in (("仅准则1", int(np.ceil(sz['n1']))), ("三准则取大", sz['n'])):
        sl, mp = [], []
        for _ in range(R_SIM):
            X = rng.normal(size=(n, P_PAR)); yv = rng.binomial(1, expit(a + X @ beta_j)).astype(float)
            b = irls(np.c_[np.ones(n), X], yv)
            lp_t = b[0] + Xt @ b[1:]
            bs = irls(np.c_[np.ones(N_TEST), lp_t], pi_t, iters=12)   # 用真实风险作软标签：去掉测试集结局的抽样噪声，得到总体校准斜率
            sl.append(bs[1]); mp.append(np.mean(np.abs(expit(lp_t) - pi_t)))
        sl = np.array(sl)
        print(f"{C:>5.2f}{label:>10}{n:>7d}{n * PREV:>6.0f}{n * PREV / P_PAR:>6.1f}{sl.mean():>9.3f}{sl.std(ddof=1):>8.3f}{np.mean(sl < 0.9):>11.2f}{np.mean(mp):>8.4f}")
    print(f"      （C={C}：R²_CS = {r2:.4f}；n1 = {sz['n1']:.0f}，n2 = {sz['n2']:.0f}（S2 = {sz['S2']:.4f}），n3 = {sz['n3']:.0f}）")
print(f"总用时 {time.time() - t0:.0f} 秒")

# Repository acceptance checks use exact identities where possible.
write_result('30', {'planning_three_criteria_maximum': planning['n']==int(np.ceil(max(planning['n1'],planning['n2'],planning['n3']))), 'weaker_model_needs_more_development_data': sample_size(.70,.12)['n'] > planning['n'], 'slope_information_positive': info_slope > 0}, {'planning_n': planning['n'], 'planning_r2cs': float(planning['r2']), 'external_slope_n': float(n_sl)})
