# 逐章覆盖核查

本表对应 Python 独立学习版（2026-10-12）及配套项目 v2.0.1。原有 **31 个章号均有真实 Python 入口**，没有漏章或把一个实验冒充另一个章节。第 21—31 章共 16 个数学实验；不编号的线性代数基础另有小型入口。v2.0.1 修复 M31 先验校正的配对设计，原有部分实现与人工范围保持；本轮运行另见[验证记录](validation.md)。

覆盖分级评价该章已交付的可执行教学内容，不能用“章号齐全”推断所有算法、临床验证和人工审核已经完成。当前分级为 10 章主要示例已实现、19 章部分实现、2 章模板与人工工作；每一章均列出具体边界。

[逐章操作指南](chapter-guide.md) · [机器可读清单](chapter-coverage.json) · [分析工具箱](python-toolkit.md) · [验证记录](validation.md)

## 运行与证据

01—20 章由 `cph/course.py` 调度，核心预测、重抽样和工具模块均为 Python。单章准备所需共享输入，无需先运行所有前置章节。默认 quick 输出是 `outputs/quick/`；书籍预算改用 `--mode book`，输出在 `outputs/book/`。`PASS_EXECUTED` 仅表示该章命令实际成功；人工审核仍可为 `PENDING`。源码、输入、依赖或预算改变后旧章状态为 `STALE`，失败重跑写 `FAIL`。

主线读取冻结的原合成 CSV。Python 对同一输入重拟合；新增验证与重抽样使用 PCG64，不能承诺与旧版 R 随机流的区间逐位相等。共享对象为 `results/model.json`，模型身份绑定完整规则与核心实现 SHA-256。

## 重要边界

- 第 07、29 章没有完整主线 MICE；第 30 章自行计算样本量公式，不冒称包工具实跑。
- 第 09 章实际运行 AJ 与两个原因的病因别 Cox。没有 Fine–Gray 回归、个体 CIF 预测或 B 全套验证。
- 第 10—11 章实际运行普通 Logistic/ridge 外层比较及 3×3 嵌套调参；森林、boosting、TabPFN 未接入主线。
- 第 12—13 章区分开发流程、固定模型表观指标、平滑校准逐点范围。校正校准曲线没有抽样置信带。
- 第 14 章是独立合成验证；第 17—20 章不证明临床部署、人工签审或投稿已经完成。

## 不编号的线性代数基础

`python run.py foundations --mode quick` 运行 `cph/foundations.py`，保存 `outputs/quick/foundations/result.json` 与 `examples.csv`。此入口不占用或改变 01—31 章编号。

## 全部章号

| 章 | 书中标题 | 覆盖 | 实际执行范围 |
|---|---|---|---|
| 01 | 使用说明与总原则 | 部分实现 | 主线结构、两时点与可用时间检查，14 原始输入与 18 个非截距参数 |
| 02 | 把项目、环境与运行记录建起来 | 主要示例已实现 | Python 环境、种子、独立模式、SHA-256 规则身份与源码来源记录 |
| 03 | 准备数据与合成教学案例 | 主要示例已实现 | 读取并导出冻结的 3,200 人合成主线，核对字段、缺失与患者 ID |
| 04 | 把六要素写成规格表与队列规则 | 部分实现 | 六要素规格与合成单记录队列资格表 |
| 05 | 先把结局、竞争事件与未知标签分清 | 主要示例已实现 | 复合 CD 结局/非致死 B 状态、提前截止未知标签与错误时序拒绝 |
| 06 | 变量审查表与防泄漏管线 | 部分实现 | 白名单、晚回报和未知可得时间拒绝 |
| 07 | 样本量、参数预算与缺失处理 | 部分实现 | 18 参数计数、训练侧中位数填补及测试值扰动边界 |
| 08 | 回归类模型 | 部分实现 | 固定结构 Logistic 的秩/梯度/收敛检查、JSON 导出与显式单位往返 |
| 09 | 生存与竞争风险：独立 B 分支 | 部分实现 | 实际 Aalen–Johansen 表和阶梯图；两个原因分别拟合 3 参数病因别 Cox 与 Breslow 基线 |
| 10 | 机器学习类模型 | 部分实现 | 相同 3 个外层折内重拟合普通及预设 ridge Logistic，导出逐人 OOF 概率 |
| 11 | 调参与嵌套验证 | 部分实现 | 实际 3×3 嵌套折内按最小平均 Brier 选择 lambda，训练折内缩放/填补并外层评价 |
| 12 | 内部验证 | 部分实现 | 完整预设 A 拟合的乐观校正、固定表观范围与双重 bootstrap 开发流程范围 |
| 13 | 评价指标与决策曲线 | 部分实现 | C/Brier/校准斜率/offset CITL/OE、固定样条平滑校准与点位乐观校正、风险人数及净获益 |
| 14 | 外部验证与更新 | 部分实现 | Python PCG64 独立合成验证，800 人更新/800 人评价，冻结/L1/L2 规则比较 |
| 15 | 稳健性、亚组、漂移与公平性 | 部分实现 | 开发样本亚组描述与区间可估性台账、白蛋白输入扰动 |
| 16 | 反例实验室 | 部分实现 | 未知单位、recipe 改变、完全分离、提前截止未知标签 4 个真实拒绝用例 |
| 17 | 模型展示、投入临床与监测 | 部分实现 | Python JSON 预测对象、显式单位/范围检查、SHA-256 完整规则身份与按 ID 的冻结概率表 |
| 18 | 报告与模板包 | 部分实现 | 从同次实际 Python 结果生成报告草稿与区间目标表 |
| 19 | 投稿实操 | 模板与人工工作 | 生成本地投稿材料待核清单 |
| 20 | AI 辅助的实操规矩 | 模板与人工工作 | 教学运行留痕、未执行行为与人工待审记录 |
| 21 | 似然、Logistic 与迭代加权最小二乘 | 主要示例已实现 | 2×2 IRLS、分离、Firth 与 FLIC 数值例 |
| 22 | 正则化与收缩 | 主要示例已实现 | 惩罚闭式解、偏差方差、坐标下降、收缩及 KKT 反例 |
| 23 | 样条与广义可加模型 | 部分实现 | RCS 约束/尾部、P-spline EDF/GCV/零空间与非线性检验 |
| 24 | 生存分析 | 主要示例已实现 | Cox/Breslow、AJ、竞争风险/IPCW/Fine–Gray 风险集与标签反例 |
| 25 | 树与集成 | 主要示例已实现 | Gini/Newton 提升/OOB/相关树方差及森林、boosting 校准网格 |
| 26 | 神经网络与表格基础模型（选读） | 部分实现 | 前向/反向、参数量、早停与岭、MLP 权重衰减模拟 |
| 27 | 评价指标的数学 | 主要示例已实现 | AUC/DeLong、Brier 分解、校准、积分与 NRI 数学反例 |
| 28 | 重抽样 | 主要示例已实现 | bootstrap 枚举、CV 协方差、噪声筛选泄漏与嵌套选择 |
| 29 | 缺失数据 | 部分实现 | Rubin 规则、缺失机制、线性 MI/PMM 与部署概率平均例 |
| 30 | 样本量公式 | 部分实现 | 自行实现开发三准则/信号假设敏感性与外部验证精度例 |
| 31 | 采样、处理与反事实预测 | 主要示例已实现 | 病例对照/类平衡校正、IPW、条件化、反事实与政策反馈 |

## 第 01 章：使用说明与总原则

**入口：** `python run.py chapter 01 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)。

**已执行内容：** 主线结构、两时点与可用时间检查，14 原始输入与 18 个非截距参数。

**实际产物：**

- `outputs/quick/chapters/ch01/orientation_checks.csv`
- `outputs/quick/chapters/ch01/lesson.md`
- `outputs/quick/chapters/ch01/status.csv`
- `outputs/quick/chapters/ch01/status.json`

**覆盖边界：**

- 技术检查不包含全部临床红线签审和 G1—G16 门禁；PASS_EXECUTED 不是研究方案批准。

**人工工作：**

- 读者选择阅读路径，填写真实研究目标、决策时点与责任人。

## 第 02 章：把项目、环境与运行记录建起来

**入口：** `python run.py chapter 02 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)。

**已执行内容：** Python 环境、种子、独立模式、SHA-256 规则身份与源码来源记录。

**实际产物：**

- `outputs/quick/chapters/ch02/session_info.txt`
- `outputs/quick/chapters/ch02/run_record.csv`
- `outputs/quick/chapters/ch02/environment.csv`
- `outputs/quick/chapters/ch02/lesson.md`
- `outputs/quick/chapters/ch02/status.csv`
- `outputs/quick/chapters/ch02/status.json`

**覆盖边界：**

- 课程不自动改写书稿的 LaTeX 表图。
- Git 标签、源码摘要与时间记录不证明预先注册或可信签名。

**人工工作：**

- 真实项目审批、方案冻结与 Git 提交记录由研究者维护。

**对应书中步骤：** 步骤 1：取得指定版本，核对范围；步骤 2：先建立研究规格；步骤 3：检查环境与输入；步骤 4：选择任务与预算；步骤 5：核对结果与状态。

## 第 03 章：准备数据与合成教学案例

**入口：** `python run.py chapter 03 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)。

**已执行内容：** 读取并导出冻结的 3,200 人合成主线，核对字段、缺失与患者 ID。

**实际产物：**

- `outputs/quick/chapters/ch03/data_summary.csv`
- `outputs/quick/chapters/ch03/field_profile.csv`
- `outputs/quick/chapters/ch03/lesson.md`
- `outputs/quick/chapters/ch03/status.csv`
- `outputs/quick/chapters/ch03/status.json`

**覆盖边界：**

- 输入为冻结的历史合成 CSV；Python 不重新执行原 R 生成器或声称复现其随机流。
- 主线完整观察且输入无缺失；真实/公共数据映射及完整 B-LTFU、L-AVAIL、M-DEPLOY 主线扩展未交付。

**人工工作：**

- 真实数据权限、字典、时间来源与伦理适用性需人工核查。

**对应书中步骤：** 步骤 1：写生成说明与估计目标；步骤 2：运行主线并核对规模；步骤 3：检查时间与列白名单；步骤 4：另建方法实验；步骤 5：转向真实或公共数据。

## 第 04 章：把六要素写成规格表与队列规则

**入口：** `python run.py chapter 04 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)。

**已执行内容：** 六要素规格与合成单记录队列资格表。

**实际产物：**

- `outputs/quick/chapters/ch04/six_elements.csv`
- `outputs/quick/chapters/ch04/cohort_flow.csv`
- `outputs/quick/chapters/ch04/lesson.md`
- `outputs/quick/chapters/ch04/status.csv`
- `outputs/quick/chapters/ch04/status.json`

**覆盖边界：**

- 未实现真实多记录数据的首次合格记录选择、逐化验时间窗提取和完整逐步筛选汇总。
- 单位接口与标签构造另在第 05、08 章运行。

**人工工作：**

- 真实纳排与重复记录规则须在查看结局前写定。

**对应书中步骤：** 步骤 1：写定六要素；步骤 2：做六个配对检查；步骤 3：预设重复记录规则；步骤 4：按窗口取值并统一单位；步骤 5：生成逐步计数。

## 第 05 章：先把结局、竞争事件与未知标签分清

**入口：** `python run.py chapter 05 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)。

**已执行内容：** 复合 CD 结局/非致死 B 状态、提前截止未知标签与错误时序拒绝。

**实际产物：**

- `outputs/quick/chapters/ch05/labels_A_B.csv`
- `outputs/quick/chapters/ch05/label_fixtures.csv`
- `outputs/quick/chapters/ch05/negative_checks.csv`
- `outputs/quick/chapters/ch05/lesson.md`
- `outputs/quick/chapters/ch05/status.csv`
- `outputs/quick/chapters/ch05/status.json`

**覆盖边界：**

- 未实施真实文本结局抽取的敏感度/特异度验证；未知状态不能视为完整随访。

**人工工作：**

- 真实资料来源、判定者、盲法与同日事件裁定需人工确认。

**对应书中步骤：** 步骤 1：冻结结局卡；步骤 2：限定本次可观察范围；步骤 3：检查先后与标签；步骤 4：按目标选择分析；步骤 5：核查结局抽取。

## 第 06 章：变量审查表与防泄漏管线

**入口：** `python run.py chapter 06 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)。

**已执行内容：** 白名单、晚回报和未知可得时间拒绝。

**实际产物：**

- `outputs/quick/chapters/ch06/variable_review.csv`
- `outputs/quick/chapters/ch06/negative_checks.csv`
- `outputs/quick/chapters/ch06/lesson.md`
- `outputs/quick/chapters/ch06/status.csv`
- `outputs/quick/chapters/ch06/status.json`

**覆盖边界：**

- 当前合成数据共享时间字段；未建立真实逐变量来源、命名/单变量能力扫描或重复患者多记录分组管线。
- 训练/测试角色边界另在第 07、10、11 章检验。

**人工工作：**

- 变量定义是否含结局信息、真实测量是否可得须人工审阅。

**对应书中步骤：** 步骤 1：建审查记录；步骤 2：做时间和白名单检查；步骤 3：做辅助扫描；步骤 4：预设候选及选择策略；步骤 5：限制学习样本。

## 第 07 章：样本量、参数预算与缺失处理

**入口：** `python run.py chapter 07 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)、[`cph/preprocessing.py`](../cph/preprocessing.py)。

**已执行内容：** 18 参数计数、训练侧中位数填补及测试值扰动边界。

**实际产物：**

- `outputs/quick/chapters/ch07/parameter_budget.csv`
- `outputs/quick/chapters/ch07/missing_recipe.csv`
- `outputs/quick/chapters/ch07/missing_assignments.csv`
- `outputs/quick/chapters/ch07/held_out_median_demo.csv`
- `outputs/quick/chapters/ch07/boundary_checks.csv`
- `outputs/quick/chapters/ch07/lesson.md`
- `outputs/quick/chapters/ch07/status.csv`
- `outputs/quick/chapters/ch07/status.json`

**覆盖边界：**

- 不是多重填补；未实现主线完整 MICE、Rubin 合并和 MI 嵌入重抽样链。
- 本章未运行 pmsampsize 包；第 30 章另有 Python 自行积分的三准则与外部验证精度例。

**人工工作：**

- 预期信号、事件率、样本量敏感性与部署缺失策略须事先设定。

**对应书中步骤：** 步骤 1：先数参数；步骤 2：以事先输入计算三准则；步骤 3：另外规划外部验证；步骤 4：确定缺失策略和部署条件；步骤 5：合并不确定性。

## 第 08 章：回归类模型

**入口：** `python run.py chapter 08 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)。

**已执行内容：** 固定结构 Logistic 的秩/梯度/收敛检查、JSON 导出与显式单位往返。

**实际产物：**

- `outputs/quick/chapters/ch08/coefficients.csv`
- `outputs/quick/chapters/ch08/fit_state.csv`
- `outputs/quick/chapters/ch08/unit_roundtrip.csv`
- `outputs/quick/chapters/ch08/lesson.md`
- `outputs/quick/chapters/ch08/status.csv`
- `outputs/quick/chapters/ch08/status.json`

**覆盖边界：**

- 未交付本章全部关系/共线性/影响点诊断及逐人开发预测波动区间。
- 岭惩罚和训练内 lambda 选择由第 10—11 章演示；不能称本章已覆盖全部惩罚回归。

**人工工作：**

- 看结果后的结构修改与理由须记录，并重新验证改变后的流程。

**对应书中步骤：** 步骤 1：冻结模型结构；步骤 2：拟合并检查数值问题；步骤 3：做诊断；步骤 4：需要惩罚时验证选择过程；步骤 5：检查预测一致性与稳定性。

## 第 09 章：生存与竞争风险：独立 B 分支

**入口：** `python run.py chapter 09 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)。

**已执行内容：** 实际 Aalen–Johansen 表和阶梯图；两个原因分别拟合 3 参数病因别 Cox 与 Breslow 基线。

**实际产物：**

- `outputs/quick/chapters/ch09/branch_counts.csv`
- `outputs/quick/chapters/ch09/branch_coefficients.csv`
- `outputs/quick/chapters/ch09/aalen_johansen.csv`
- `outputs/quick/chapters/ch09/cause_specific_baseline.csv`
- `outputs/quick/chapters/ch09/aalen_johansen.png`
- `outputs/quick/chapters/ch09/aalen_johansen.pdf`
- `outputs/quick/chapters/ch09/lesson.md`
- `outputs/quick/chapters/ch09/status.csv`
- `outputs/quick/chapters/ch09/status.json`
- `outputs/quick/results/branch_B.json`

**覆盖边界：**

- 未实现 Fine–Gray 回归；病因别 Cox 与子分布风险模型不同，不能把本章 Cox 结果冒称 Fine–Gray。
- 未由个体两类风险率计算条件 CIF，未实现 B 的时间点校准/判别/DCA 及内部/外部验证。
- 第 24、24b 章的 Fine–Gray 风险集小例不是本章患者的完整回归与评价。

**人工工作：**

- 真实事件先后、删失与目标控制定义需人工确认。

**对应书中步骤：** 步骤 1：锁定目标和事件类型；步骤 2：先做非参数曲线；步骤 3：拟合预设病因别 Cox；步骤 4：拟合 Fine--Gray；步骤 5：评价同一目标。

## 第 10 章：机器学习类模型

**入口：** `python run.py chapter 10 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)、[`cph/preprocessing.py`](../cph/preprocessing.py)、[`cph/validation.py`](../cph/validation.py)、[`cph/predictions.py`](../cph/predictions.py)。

**已执行内容：** 相同 3 个外层折内重拟合普通及预设 ridge Logistic，导出逐人 OOF 概率。

**实际产物：**

- `outputs/quick/chapters/ch10/outer_comparison.csv`
- `outputs/quick/chapters/ch10/outer_roles.csv`
- `outputs/quick/chapters/ch10/outer_predictions.csv`
- `outputs/quick/chapters/ch10/lesson.md`
- `outputs/quick/chapters/ch10/status.csv`
- `outputs/quick/chapters/ch10/status.json`

**覆盖边界：**

- 主线 A 未接入森林、boosting、SHAP 或 TabPFN；第 25、25b 章为独立模拟。
- ridge 在训练折内学习设计列的缩放，惩罚作用于标准化后的系数；与旧版固定尺度 ridge 的数值不等价。

**人工工作：**

- 按研究规模和场景事先选择有限候选算法。

**对应书中步骤：** 步骤 1：冻结基线程序，限定训练范围；步骤 2：限定候选算法；步骤 3：明确默认值并调参；步骤 4：检查类别再平衡；步骤 5：评价概率与稳定性。

## 第 11 章：调参与嵌套验证

**入口：** `python run.py chapter 11 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)、[`cph/preprocessing.py`](../cph/preprocessing.py)、[`cph/validation.py`](../cph/validation.py)、[`cph/predictions.py`](../cph/predictions.py)。

**已执行内容：** 实际 3×3 嵌套折内按最小平均 Brier 选择 lambda，训练折内缩放/填补并外层评价。

**实际产物：**

- `outputs/quick/chapters/ch11/inner_tuning.csv`
- `outputs/quick/chapters/ch11/nested_metrics.csv`
- `outputs/quick/chapters/ch11/outer_roles.csv`
- `outputs/quick/chapters/ch11/outer_predictions.csv`
- `outputs/quick/chapters/ch11/lesson.md`
- `outputs/quick/chapters/ch11/status.csv`
- `outputs/quick/chapters/ch11/status.json`

**覆盖边界：**

- 不是 1SE 选择；没有主线 MICE、变量筛选、再采样/早停或概率再校准的完整嵌套链。
- 未冻结最终在全样本重新调参的 ridge 对象；共享最终模型仍是无惩罚 A Logistic。
- lambda 候选仅 1/10/100，有限网格不证明全局最优；惩罚作用于训练折内标准化设计。

**人工工作：**

- 真实分割方案、患者/中心/时间约束、搜索预算与失败规则须事先声明。

**对应书中步骤：** 步骤 1：先确定是否需要调参；步骤 2：冻结搜索方案；步骤 3：固定分割；步骤 4：在内层重做完整流程；步骤 5：外层只作预测与评价；步骤 6：全样本开发最终模型；步骤 7：如实报告。

## 第 12 章：内部验证

**入口：** `python run.py chapter 12 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)、[`cph/course_validation.py`](../cph/course_validation.py)。

**已执行内容：** 完整预设 A 拟合的乐观校正、固定表观范围与双重 bootstrap 开发流程范围。

**实际产物：**

- `outputs/quick/chapters/ch12/optimism_corrected_metrics.csv`
- `outputs/quick/chapters/ch12/development_percentile_ranges.csv`
- `outputs/quick/chapters/ch12/fixed_apparent_ranges.csv`
- `outputs/quick/chapters/ch12/lesson.md`
- `outputs/quick/chapters/ch12/status.csv`
- `outputs/quick/chapters/ch12/status.json`
- `outputs/quick/results/bootstrap_optimism.csv`
- `outputs/quick/results/bootstrap_optimism.json`
- `outputs/quick/results/fixed_model_intervals.csv`
- `outputs/quick/results/fixed_model_intervals.json`
- `outputs/quick/results/development_intervals.csv`
- `outputs/quick/results/development_intervals.json`

**覆盖边界：**

- 未执行建议的 10 折、留一中心或时间验证；第 10—11 章是另一套 3 折练习。
- 未执行 MI、统一收缩后模型交付或校正统计量名义覆盖率验证。
- 双重 bootstrap 范围与固定模型表观区间均不是最终固定模型真实外部性能区间。
- PCG64 与旧版随机流不同；任何失败重复均使对应汇总不可估计，不丢弃失败后补跑成功样本。

**人工工作：**

- 根据目标、计算误差与研究设计选择并报告验证层级。

**对应书中步骤：** 步骤 1 把“整条流程”写成一个函数，并让它记录自己做了哪些步骤。；步骤 2 重做开发过程，估计乐观并校正表现；步骤 3 选重复次数，并检查稳定性。；步骤 4 校准斜率要单独校正并报告；斜率明显小于 1 时用校正后的斜率做统一收缩。；步骤 5 用另一种办法复核：10 折交叉验证（折内重做全部流程；有多行的患者按患者分组）、留一中心、时间验证。；步骤 6 缺失分支先确定填补任务：研究 MI 与无结局部署预测分别验证。。

## 第 13 章：评价指标与决策曲线

**入口：** `python run.py chapter 13 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)、[`cph/course_validation.py`](../cph/course_validation.py)、[`cph/metrics.py`](../cph/metrics.py)。

**已执行内容：** C/Brier/校准斜率/offset CITL/OE、固定样条平滑校准与点位乐观校正、风险人数及净获益。

**实际产物：**

- `outputs/quick/chapters/ch13/calibration_curve.csv`
- `outputs/quick/chapters/ch13/risk_groups.csv`
- `outputs/quick/chapters/ch13/decision_curve.pdf`
- `outputs/quick/chapters/ch13/decision_curve.png`
- `outputs/quick/chapters/ch13/lesson.md`
- `outputs/quick/chapters/ch13/status.csv`
- `outputs/quick/chapters/ch13/status.json`
- `outputs/quick/results/calibration_curve.csv`
- `outputs/quick/results/calibration_groups.csv`
- `outputs/quick/results/calibration_recipe.csv`
- `outputs/quick/results/calibration_log.csv`
- `outputs/quick/results/calibration.json`
- `outputs/quick/results/calibration.png`
- `outputs/quick/results/calibration.pdf`
- `outputs/quick/results/decision_curve.csv`

**覆盖边界：**

- 主线未输出 ICI/E50/E90/Emax、scaled Brier、AUPRC、完整阈值表或配对模型差的 bootstrap 区间。
- 工具箱另有 ROC/PR、预设阈值混淆矩阵及 Wilson 区间、成对 DeLong、模型比较 API；不等于本章已自动生成全套主线表。
- 蓝色带只对应固定模型表观曲线的逐点条件经验范围；橙色校正曲线只有点估计与 Monte Carlo 标准误，没有抽样置信区间或同时置信带。
- 平滑曲线沿用固定 logit 结点的自然样条空间，不用 LOWESS 替代；未实现 B 的删失/竞争风险整套评价。

**人工工作：**

- 决策阈值及其临床来源须预设并由真实使用场景支持。

**对应书中步骤：** 步骤 1 先预设主表的字段与阈值范围，再算。；步骤 2 报告区分度及区间；步骤 3 校准：斜率与截距分开估；画平滑校准曲线并附风险分布；报告 O/E 与 ICI、E50、E90、Emax。；步骤 4 整体误差：Brier 与缩放 Brier，并给区间。；步骤 5 阈值指标：在预设的阈值上报混淆矩阵、灵敏度、特异度、PPV、NPV 及区间；说明 PPV/NPV 随患病率变。；步骤 6 在预设阈值上评价净获益；步骤 7 模型比较用配对 bootstrap：同一批 bootstrap 样本里同时算两个模型的差。；步骤 8 按删失与竞争风险目标评价。

## 第 14 章：外部验证与更新

**入口：** `python run.py chapter 14 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)、[`cph/metrics.py`](../cph/metrics.py)。

**已执行内容：** Python PCG64 独立合成验证，800 人更新/800 人评价，冻结/L1/L2 规则比较。

**实际产物：**

- `outputs/quick/chapters/ch14/synthetic_validation_data.csv`
- `outputs/quick/chapters/ch14/validation_roles.csv`
- `outputs/quick/chapters/ch14/update_coefficients.csv`
- `outputs/quick/chapters/ch14/held_out_update_evaluation.csv`
- `outputs/quick/chapters/ch14/lesson.md`
- `outputs/quick/chapters/ch14/status.csv`
- `outputs/quick/chapters/ch14/status.json`

**覆盖边界：**

- 没有真实外部队列；未输出完整病例组合分析、模型化预期 C、外部指标区间、L3 全系数或 L4 新项更新。
- 这是 Python PCG64 新模拟，不能与历史 R 验证队列逐人或逐位相等；模拟事件率变化不是临床跨医院证据。

**人工工作：**

- 真实原模型转写双人核对、外部字段映射与更新决策权限需人工完成。

**对应书中步骤：** 步骤 1 把论文里的模型写成函数，逐项核对：公式、变量编码、单位、样条结点、切点。；步骤 2 冻结完整预测对象并登记身份；核对变换、变量、单位与分类水平。；步骤 3 在自己的数据里按论文的定义重新构造预测因子（取值窗口、单位），不合格的变量按缺失处理。；步骤 4 先做病例组合比较，再按第 13 章的全套指标评价，每个数带区间。；步骤 5 更新方案事先写定：谁来决定、用哪一半数据更新、用另一半（或新数据）重新评价、什么情况下选哪个层级。；步骤 6 报告时写清验证层级，并让 G14 检查措辞。。

## 第 15 章：稳健性、亚组、漂移与公平性

**入口：** `python run.py chapter 15 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)、[`cph/course_validation.py`](../cph/course_validation.py)。

**已执行内容：** 开发样本亚组描述与区间可估性台账、白蛋白输入扰动。

**实际产物：**

- `outputs/quick/chapters/ch15/descriptive_subgroups.csv`
- `outputs/quick/chapters/ch15/input_perturbation.csv`
- `outputs/quick/chapters/ch15/sensitivity_summary.csv`
- `outputs/quick/chapters/ch15/subgroup_ledger.json`
- `outputs/quick/chapters/ch15/lesson.md`
- `outputs/quick/chapters/ch15/status.csv`
- `outputs/quick/chapters/ch15/status.json`

**覆盖边界：**

- 未实现缺失/结局/纳排/时间等完整敏感性场景、亚组联合校准检验、中心随机效应合并与新中心预测区间。
- 未实施流式漂移/输入分布监测或各组 TPR/FPR/PPV/净获益全表。
- quick 仅 40 次亚组重抽样，低于 100 次有效重复要求，区间保留 NA；book 为 300 次。

**人工工作：**

- 差异来源、精度与结论强度需审阅；输入扰动不是治疗效应或公平性证明。

**对应书中步骤：** 步骤 1 敏感性分析：预设场景清单，每个场景一个函数，输出同一套指标。；步骤 2 亚组：预设分组变量和精度要求；可估指标给合适区间，不可估指标说明原因，精度不足时不下稳定或优越结论。；步骤 3 中心效应：留一中心（第 12 章）或各中心的截距与斜率，用随机效应合并，给出新中心的预测区间。；步骤 4 预设漂移监测规则；步骤 5 报告各组表现与决策结果；步骤 6 写结论时只描述差异和证据强度，不下“公平”或“不公平”的总判断；差异的来源（模型、数据、结局登记、治疗政策）要逐一排查。。

## 第 16 章：反例实验室

**入口：** `python run.py chapter 16 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)。

**已执行内容：** 未知单位、recipe 改变、完全分离、提前截止未知标签 4 个真实拒绝用例。

**实际产物：**

- `outputs/quick/chapters/ch16/negative_checks.csv`
- `outputs/quick/chapters/ch16/lesson.md`
- `outputs/quick/chapters/ch16/status.csv`
- `outputs/quick/chapters/ch16/status.json`

**覆盖边界：**

- 本章不是原书 S16 十个历史反例的完整复跑。
- 晚回报与测试侧中位数边界另在第 06、07 章；筛选泄漏等另在数学第 28、31 章。

**人工工作：**

- 读者据真实管线补充反例并保留正确/错误版本的证据。

**对应书中步骤：** 步骤 1 选一个反例，按“做错的版本”跑一遍，记下被抬高或压低的数字。；步骤 2 按“正确的版本”再跑一遍，并与答案对照。；步骤 3 打开对应的检查点，记下输出。；步骤 4 红队测试：若某个反例没被拦住，说明检查点写错了，回改检查点而不是改结论。；步骤 5 把十个反例的结果存成回归测试：以后改检查点，十个反例必须仍然被拦住。。

## 第 17 章：模型展示、投入临床与监测

**入口：** `python run.py chapter 17 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)、[`cph/predictions.py`](../cph/predictions.py)。

**已执行内容：** Python JSON 预测对象、显式单位/范围检查、SHA-256 完整规则身份与按 ID 的冻结概率表。

**实际产物：**

- `outputs/quick/chapters/ch17/example_predictions.csv`
- `outputs/quick/chapters/ch17/input_units.csv`
- `outputs/quick/chapters/ch17/model_registry.md`
- `outputs/quick/chapters/ch17/known_case_predictions.csv`
- `outputs/quick/chapters/ch17/known_case_predictions.csv.json`
- `outputs/quick/chapters/ch17/lesson.md`
- `outputs/quick/chapters/ch17/status.csv`
- `outputs/quick/chapters/ch17/status.json`

**覆盖边界：**

- 没有多语言预测接口、评分表/列线图、完整浏览器计算器、追加式影子日志或生产版本停用系统。
- 没有临床静默运行与长期监测；SHA-256 摘要只核查规则/源码一致性，不是可信签名。
- FrozenPredictions 仅查找原有患者 ID，不能把已有概率表伪装成预测新病例的模型。

**人工工作：**

- 真实模型卡、可信身份登记、临床审核与部署责任由人完成。

**对应书中步骤：** 步骤 1 冻结并导出交付对象。；步骤 2 用独立实现复算同一批患者，逐人核对预测。；步骤 3 需要评分表或列线图时，先定刻度和每点代表的 logit，再把总点数映射回风险，然后重新评价。；步骤 4 写模型卡并用检查器过一遍。；步骤 5 计算器的最低要求：本地计算，输入不发出、不保存；超出取值范围的输入拒绝；缺任何一项不输出预测；页面显示版本号与适用人群。；步骤 6 静默运行：先记录预测，之后补结局。；步骤 7 监测：先在无变化的数据流里测触发规则的误报，再定阈值。；步骤 8 版本登记与停用。。

## 第 18 章：报告与模板包

**入口：** `python run.py chapter 18 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)、[`cph/course_validation.py`](../cph/course_validation.py)。

**已执行内容：** 从同次实际 Python 结果生成报告草稿与区间目标表。

**实际产物：**

- `outputs/quick/chapters/ch18/report.md`
- `outputs/quick/chapters/ch18/interval_targets.csv`
- `outputs/quick/chapters/ch18/synthetic_update_evaluation.csv`
- `outputs/quick/chapters/ch18/lesson.md`
- `outputs/quick/chapters/ch18/status.csv`
- `outputs/quick/chapters/ch18/status.json`

**覆盖边界：**

- 未迁移原书全部报告完整性/人工字段/锁定块检查器；生成草稿不能宣称报告已签审。

**人工工作：**

- 伦理、数据权限、研究定位、署名、清单映射、局限与结论需人工填写。

**对应书中步骤：** 步骤1 在运行前固定问题、两时点、结局与参数预算。；步骤2 全样本开发，按实际开发人数核查预算。；步骤3 从同次运行生成表、图和结果草稿，保留状态。；步骤4 说明区间对应哪一层不确定性。；步骤5 填人工项及关键局限，再核对完整性。；步骤6 定制稿保留核心证据，再考虑交付。。

## 第 19 章：投稿实操

**入口：** `python run.py chapter 19 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)、[`templates/report.md`](../templates/report.md)、[`templates/tripod_ai_mapping.csv`](../templates/tripod_ai_mapping.csv)。

**已执行内容：** 生成本地投稿材料待核清单。

**实际产物：**

- `outputs/quick/chapters/ch19/submission_checklist.csv`
- `outputs/quick/chapters/ch19/submission_materials.md`
- `outputs/quick/chapters/ch19/lesson.md`
- `outputs/quick/chapters/ch19/status.csv`
- `outputs/quick/chapters/ch19/status.json`

**覆盖边界：**

- 没有选刊政策实时采集、12 项完整投稿包、作者资格判定、锁定块审核或实际投稿/审稿回复。
- READER_VERIFY、PENDING、SYNTHETIC_ONLY 是当前状态，不代表材料已齐或已提交。

**人工工作：**

- 读者核查目标期刊官方政策、证据、伦理/署名与实际提交步骤。

**对应书中步骤：** 步骤 1 先定证据层级（仅内部验证、已做外部验证、已评价临床影响），再定文章类别，再看三类硬条件。；步骤 2 建期刊档案并写明更新日期与来源链接；未核对的字段写“待核对”。；步骤 3 按期刊定制时只改可以改的。；步骤 4 备齐材料包：12 个组成部分一个不缺，不留“【待填】”。；步骤 5 署名凭贡献。；步骤 6 投稿日志：同一稿件不能同时在多家期刊审理；使用同一队列、同一结局的两篇稿件要由负责人判断是否拆分发表。；步骤 7 回复审稿意见：先分表达、方法、主线三类；逐条回复；新增分析标为事后分析并写明位置；方向与原结论不一致的结果如实报告；改动主线结论要有负责人签字。；步骤 8 被拒后转投：重跑措辞检查和锁定块检查；预印本的取舍以目标期刊政策为准（待核对）。。

## 第 20 章：AI 辅助的实操规矩

**入口：** `python run.py chapter 20 --mode quick`。

**实现：** [`run.py`](../run.py)、[`examples/course_cli.py`](../examples/course_cli.py)、[`cph/course.py`](../cph/course.py)、[`cph/course_core.py`](../cph/course_core.py)、[`templates/human_review.csv`](../templates/human_review.csv)、[`templates/change_log.csv`](../templates/change_log.csv)。

**已执行内容：** 教学运行留痕、未执行行为与人工待审记录。

**实际产物：**

- `outputs/quick/chapters/ch20/ai_audit.csv`
- `outputs/quick/chapters/ch20/human_review.csv`
- `outputs/quick/chapters/ch20/audit_instructions.md`
- `outputs/quick/chapters/ch20/lesson.md`
- `outputs/quick/chapters/ch20/status.csv`
- `outputs/quick/chapters/ch20/status.json`

**覆盖边界：**

- 命令不调用 AI 服务或证明人已审核；未实现全部红线签名、文献签审、十错接受清单和每月审计。

**人工工作：**

- 真实工具使用、采纳/拒绝决定、核对人、引用原文核查与披露需人工记录。

**对应书中步骤：** 步骤 1 先分工：AI 只审，不定。；步骤 2 用提示词模板，输入只含方案文字、合成数据或汇总表。；步骤 3 拿到输出后，先过“十个错”验收清单（表\ref{tab:20-accept}），再决定采纳、修改还是拒绝。；步骤 4 AI 生成的代码，三步验收：断言、对照、最小复现。；步骤 5 参考文献逐条打开原文核对并签名，没打开的标“待核对”。；步骤 6 每次使用留一行，写核对人和核对办法。；步骤 7 披露：在方法或致谢里写工具、用途和核对办法；不列为作者，也不给 CRediT 角色。；步骤 8 每月检查一次留痕表和参考文献状态表。。

## 第 21 章：似然、Logistic 与迭代加权最小二乘

**入口：** `python run.py chapter 21 --mode quick`。

**实现：** [`run.py`](../run.py)、[`math/_lab_support.py`](../math/_lab_support.py)、[`math/ch21.py`](../math/ch21.py)。

**已执行内容：** 2×2 IRLS、分离、Firth 与 FLIC 数值例。

**实际产物：**

- `outputs/quick/math/21/stdout.txt`
- `outputs/quick/math/21/result.json`

**覆盖边界：**

- 独立教学小例；不是主线 A 已改用 Firth，也不证明所有数据条件下估计存在。

## 第 22 章：正则化与收缩

**入口：** `python run.py chapter 22 --mode quick`。

**实现：** [`run.py`](../run.py)、[`math/_lab_support.py`](../math/_lab_support.py)、[`math/ch22.py`](../math/ch22.py)、[`math/ch22b.py`](../math/ch22b.py)。

**已执行内容：** 惩罚闭式解、偏差方差、坐标下降、收缩及 KKT 反例。

**实际产物：**

- `outputs/quick/math/22/stdout.txt`
- `outputs/quick/math/22/result.json`
- `outputs/quick/math/22b/stdout.txt`
- `outputs/quick/math/22b/result.json`

**覆盖边界：**

- 独立线性/Logistic 设定；不替代真实数据完整惩罚路径与嵌套验证。

## 第 23 章：样条与广义可加模型

**入口：** `python run.py chapter 23 --mode quick`。

**实现：** [`run.py`](../run.py)、[`math/_lab_support.py`](../math/_lab_support.py)、[`math/ch23.py`](../math/ch23.py)。

**已执行内容：** RCS 约束/尾部、P-spline EDF/GCV/零空间与非线性检验。

**实际产物：**

- `outputs/quick/math/23/stdout.txt`
- `outputs/quick/math/23/result.json`

**覆盖边界：**

- 未实现主线多变量 GAM、REML 或张量积交互建模管线；当前零空间示例是夹持结点下的系数差分。

## 第 24 章：生存分析

**入口：** `python run.py chapter 24 --mode quick`。

**实现：** [`run.py`](../run.py)、[`math/_lab_support.py`](../math/_lab_support.py)、[`math/ch24.py`](../math/ch24.py)、[`math/ch24b.py`](../math/ch24b.py)。

**已执行内容：** Cox/Breslow、AJ、竞争风险/IPCW/Fine–Gray 风险集与标签反例。

**实际产物：**

- `outputs/quick/math/24/stdout.txt`
- `outputs/quick/math/24/result.json`
- `outputs/quick/math/24b/stdout.txt`
- `outputs/quick/math/24b/result.json`

**覆盖边界：**

- 玩具数据手工计算；不替代第 09 章 B 的完整预测与验证，也未提供真实比例风险假设检验。

## 第 25 章：树与集成

**入口：** `python run.py chapter 25 --mode quick`。

**实现：** [`run.py`](../run.py)、[`math/_lab_support.py`](../math/_lab_support.py)、[`math/ch25.py`](../math/ch25.py)、[`math/ch25b.py`](../math/ch25b.py)。

**已执行内容：** Gini/Newton 提升/OOB/相关树方差及森林、boosting 校准网格。

**实际产物：**

- `outputs/quick/math/25/stdout.txt`
- `outputs/quick/math/25/result.json`
- `outputs/quick/math/25b/stdout.txt`
- `outputs/quick/math/25b/result.json`

**覆盖边界：**

- 森林/提升只在独立数学模拟运行，未接入主线 A；有限网格不证明全局最优。

## 第 26 章：神经网络与表格基础模型（选读）

**入口：** `python run.py chapter 26 --mode quick`。

**实现：** [`run.py`](../run.py)、[`math/_lab_support.py`](../math/_lab_support.py)、[`math/ch26.py`](../math/ch26.py)、[`math/ch26b.py`](../math/ch26b.py)。

**已执行内容：** 前向/反向、参数量、早停与岭、MLP 权重衰减模拟。

**实际产物：**

- `outputs/quick/math/26/stdout.txt`
- `outputs/quick/math/26/result.json`
- `outputs/quick/math/26b/stdout.txt`
- `outputs/quick/math/26b/result.json`

**覆盖边界：**

- 没有加载、训练或评价 TabPFN 等表格基础模型；网格不是全部网络优化或超参数最优的证明。

## 第 27 章：评价指标的数学

**入口：** `python run.py chapter 27 --mode quick`。

**实现：** [`run.py`](../run.py)、[`math/_lab_support.py`](../math/_lab_support.py)、[`math/ch27.py`](../math/ch27.py)。

**已执行内容：** AUC/DeLong、Brier 分解、校准、积分与 NRI 数学反例。

**实际产物：**

- `outputs/quick/math/27/stdout.txt`
- `outputs/quick/math/27/result.json`

**覆盖边界：**

- 小例的 DeLong、阈值或 NRI 计算未自动生成主线 A 第 13 章的完整评价表。

## 第 28 章：重抽样

**入口：** `python run.py chapter 28 --mode quick`。

**实现：** [`run.py`](../run.py)、[`math/_lab_support.py`](../math/_lab_support.py)、[`math/ch28.py`](../math/ch28.py)。

**已执行内容：** bootstrap 枚举、CV 协方差、噪声筛选泄漏与嵌套选择。

**实际产物：**

- `outputs/quick/math/28/stdout.txt`
- `outputs/quick/math/28/result.json`

**覆盖边界：**

- 包含刻意错误的对照；不是缺失、分组/中心/时间数据的通用验证引擎。

## 第 29 章：缺失数据

**入口：** `python run.py chapter 29 --mode quick`。

**实现：** [`run.py`](../run.py)、[`math/_lab_support.py`](../math/_lab_support.py)、[`math/ch29.py`](../math/ch29.py)。

**已执行内容：** Rubin 规则、缺失机制、线性 MI/PMM 与部署概率平均例。

**实际产物：**

- `outputs/quick/math/29/stdout.txt`
- `outputs/quick/math/29/result.json`

**覆盖边界：**

- 手工 Python MI/PMM 数学例；没有主线 A 多变量 MICE、其嵌套验证或完整部署链。

## 第 30 章：样本量公式

**入口：** `python run.py chapter 30 --mode quick`。

**实现：** [`run.py`](../run.py)、[`math/_lab_support.py`](../math/_lab_support.py)、[`math/ch30.py`](../math/ch30.py)。

**已执行内容：** 自行实现开发三准则/信号假设敏感性与外部验证精度例。

**实际产物：**

- `outputs/quick/math/30/stdout.txt`
- `outputs/quick/math/30/result.json`

**覆盖边界：**

- 自行积分的信号分布假设与 pmsampsize/pmvalsampsize 包的输出不能直接等同；没有调用这些包。
- 模拟设定的精度例不证明任何真实队列已满足开发或验证样本量要求。

**人工工作：**

- 真实研究需事先选择预期信号、事件率及所需精度并说明依据。

## 第 31 章：采样、处理与反事实预测

**入口：** `python run.py chapter 31 --mode quick`。

**实现：** [`run.py`](../run.py)、[`math/_lab_support.py`](../math/_lab_support.py)、[`math/ch31.py`](../math/ch31.py)、[`math/ch31b.py`](../math/ch31b.py)。

**已执行内容：** 病例对照/类平衡校正、IPW、条件化、反事实与政策反馈。

**实际产物：**

- `outputs/quick/math/31/stdout.txt`
- `outputs/quick/math/31/result.json`
- `outputs/quick/math/31b/stdout.txt`
- `outputs/quick/math/31b/result.json`

**覆盖边界：**

- 潜在结局在模拟中可得；实际观察资料的反事实识别仍依赖不可由数值例证明的因果假设。
