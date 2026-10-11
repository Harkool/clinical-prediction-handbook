# 31 章操作指南

本指南将书中操作步骤对应到实际 Python 入口、产物与练习。先完成 [安装](installation.md)，再读 [覆盖边界](chapter-coverage.md)。以下命令在仓库根目录运行，代码均在项目中；本指南不需要 R。

## 运行约定

01—20 章单独运行时自动准备共享主线数据与拟合。每章有 lesson.md、status.csv 和 status.json；manifest.csv 只登记实际运行的章，PASS_EXECUTED 不代表人工已审阅。来源改变使旧章 STALE，失败重跑写 FAIL。

下文“本章产物”的文件名相对于 `outputs/quick/chapters/chNN/`；“共享结果”相对于 `outputs/quick/`。数学实验的编号目录相对于 `outputs/quick/math/`。

| 路径 | 内容 |
|---|---|
| outputs/quick/data/main_case.csv | 3,200 人冻结合成输入，384 个含死亡主线事件 |
| outputs/quick/results/model.json | 完整精度系数、变换、目标、元数据和 SHA-256 规则身份；单位/范围由绑定的核心实现提供 |
| outputs/quick/results/model_coefficients.csv | 含截距的 19 项系数 |
| outputs/quick/results/predictions.csv | 主线逐人标签及概率 |
| outputs/quick/results/context_sources.csv | 当前核心 8 个 Python 文件的 SHA-256 |
| outputs/quick/source_signature.json | 输入、完整源码、依赖、预算及随机数方案 |
| outputs/quick/chapters/chNN/ | 本章实际文件、状态与操作说明 |
| outputs/quick/math/编号/ | 数学实验日志与 result.json |

将 --mode quick 改为 --mode book 使用完整预算和 outputs/book/。两模式不能共用同一目录。

| 重复预算 | quick | book |
|---|---:|---:|
| 开发乐观校正 | 20 | 200 |
| 固定模型表观指标 | 40 | 500 |
| 校准开发 / 固定条件 | 20 / 40 | 200 / 500 |
| 双重外层 × 内层 | 10 × 10 | 200 × 50 |
| 描述性亚组 | 40 | 300 |

主线读取冻结 CSV，不重新运行历史生成器。新验证模拟与重抽样使用 PCG64；与旧版预算相同并不保证区间的随机末位相等。新增实验应保存不同输入与说明，避免改写参考文件后沿用主线名称。

## 不编号的线性代数基础

```sh
python run.py foundations --mode quick
```

结果为 outputs/quick/foundations/result.json 与 examples.csv。先核对向量内积，再看线性相关列、秩、最小二乘残差正交和岭收缩；此短章不改变 01—31 编号。

## Python 预测接口

完成第 08 章后，可用完整 JSON 模型直接预测。以下例子需要与生成模型相同的核心实现；源码改变后应重建模型。

```python
import json
from pathlib import Path
from cph.course_core import load_case, case_units, predict_case_with_units

out = Path("outputs/quick")
model = json.loads((out / "results/model.json").read_text(encoding="utf-8"))
expected = (out / "results/prediction_identity.txt").read_text(encoding="utf-8").strip()
patients = load_case()[:3]
print(predict_case_with_units(model, patients, case_units(), expected))

bad_units = dict(case_units())
bad_units["hb"] = "unknown"
predict_case_with_units(model, patients, bad_units, expected)  # 应触发 ValueError
```

预期错误也是练习证据。同次登记的身份用于一致性核查，不能替代真实部署的可信身份来源。需要 Table 1，可另运行 `python run.py baseline --mode quick`，详见 [基线表](baseline-table.md)。[工具箱](python-toolkit.md) 提供独立的阈值、曲线、成对比较、交叉验证和学习曲线 API。

## 第 01 章：使用说明与总原则

```sh
python run.py chapter 01 --mode quick
```

**本章产物：** `orientation_checks.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 14 原始输入与 18 非截距参数；决策时点与结局起点分开。技术状态不表示临床方案已批准。

**练习：** 复制 analysis_plan.csv 到自己的目录，填写目标人群、两时点、结局、验证层级与结论范围；未审核的人工作业保留待审。

**范围：** 技术检查不包含全部临床红线签审和 G1—G16 门禁；PASS_EXECUTED 不是研究方案批准。

## 第 02 章：把项目、环境与运行记录建起来

```sh
python run.py doctor
python run.py chapter 02 --mode quick
```

**本章产物：** `session_info.txt`、`run_record.csv`、`environment.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 环境来自本次 Python；模式、种子、规则身份与源码摘要必须对应同次结果。

**练习：** 保存 Git 提交、完整命令和依赖记录。更换 Python 环境时核对 doctor；使用另一输出模式观察独立目录。

**范围：** 课程不自动改写书稿的 LaTeX 表图。 Git 标签、源码摘要与时间记录不证明预先注册或可信签名。

## 第 03 章：准备数据与合成教学案例

```sh
python run.py chapter 03 --mode quick
```

**本章产物：** `data_summary.csv`、`field_profile.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 冻结主线 3,200 人、384 事件、ID 唯一；不含 true_risk 答案列。参考 CSV 是输入，不是 Python 重新执行历史随机生成器。

**练习：** 按 center 汇总人数和事件数；将 case_recipe() 的 predictors 与模型 terms 对照，确认 ID、结局和时间不进入设计。需要另做新数据实验时保存新输入，不覆盖参考 CSV。

**范围：** 输入为冻结的历史合成 CSV；Python 不重新执行原 R 生成器或声称复现其随机流。 主线完整观察且输入无缺失；真实/公共数据映射及完整 B-LTFU、L-AVAIL、M-DEPLOY 主线扩展未交付。

## 第 04 章：把六要素写成规格表与队列规则

```sh
python run.py chapter 04 --mode quick
```

**本章产物：** `six_elements.csv`、`cohort_flow.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 对象、预测时点、输入、结局、窗口和场景配对一致。手术起点的定义不使术中信息在诱导前可得。

**练习：** 抽查 5 人的 decision_time、outcome_origin、collection_time、result_available_time；在自己的规格副本中预设重复记录和取值窗口。

**范围：** 未实现真实多记录数据的首次合格记录选择、逐化验时间窗提取和完整逐步筛选汇总。 单位接口与标签构造另在第 05、08 章运行。

## 第 05 章：先把结局、竞争事件与未知标签分清

```sh
python run.py chapter 05 --mode quick
```

**本章产物：** `labels_A_B.csv`、`label_fixtures.csv`、`negative_checks.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 含死亡主线 A 与非致死 B 的死亡状态不同；提前截止且结局未知者的 y 保持空值。

**练习：** 手核窗口内先死病例的 A_y 与 B_state，另核提前截止的未知 y。构造事件发生在死亡后的记录，确认 make_state 拒绝。

**范围：** 未实施真实文本结局抽取的敏感度/特异度验证；未知状态不能视为完整随访。

## 第 06 章：变量审查表与防泄漏管线

```sh
python run.py chapter 06 --mode quick
```

**本章产物：** `variable_review.csv`、`negative_checks.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 白名单变量和回报时间须同时合格；采集早但回报晚，以及可得时间未知均拒绝。

**练习：** 在独立数据副本把 result_available_time 改到 decision_time 后；调用 validate_availability，保留具体拒绝原因。真实数据还需逐变量来源审查。

**范围：** 当前合成数据共享时间字段；未建立真实逐变量来源、命名/单变量能力扫描或重复患者多记录分组管线。 训练/测试角色边界另在第 07、10、11 章检验。

## 第 07 章：样本量、参数预算与缺失处理

```sh
python run.py chapter 07 --mode quick
```

**本章产物：** `parameter_budget.csv`、`missing_recipe.csv`、`missing_assignments.csv`、`held_out_median_demo.csv`、`boundary_checks.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 2240 人训练侧学习 Hb 中位数；改动 960 人测试侧 Hb 不改变填补规则，全样本中位数可变。480 格演示缺失不表示原主线有缺失。

**练习：** 比较训练中位数与全数据中位数；改变测试极端值后再次解释区别。完整 MICE 与样本量公式分别见覆盖清单和第 29—30 章，不能把本章单次填补当成 MI。

**范围：** 不是多重填补；未实现主线完整 MICE、Rubin 合并和 MI 嵌入重抽样链。 本章未运行 pmsampsize 包；第 30 章另有 Python 自行积分的三准则与外部验证精度例。

## 第 08 章：回归类模型

```sh
python run.py chapter 08 --mode quick
```

**本章产物：** `coefficients.csv`、`fit_state.csv`、`unit_roundtrip.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 主线 19 项系数包含截距；收敛、秩和梯度均检查。单位转换往返后逐人概率一致，未知单位拒绝。

**练习：** 用下方 Python 预测接口给前 3 人预测；把 hb 单位改成 unknown，核对拒绝；正确把 g/L 数值换为 g/dL 并声明单位，逐人比较。

**范围：** 未交付本章全部关系/共线性/影响点诊断及逐人开发预测波动区间。 岭惩罚和训练内 lambda 选择由第 10—11 章演示；不能称本章已覆盖全部惩罚回归。

## 第 09 章：生存与竞争风险：独立 B 分支

```sh
python run.py chapter 09 --mode quick
```

**本章产物：** `branch_counts.csv`、`branch_coefficients.csv`、`aalen_johansen.csv`、`cause_specific_baseline.csv`、`aalen_johansen.png`、`aalen_johansen.pdf`，以及 lesson.md、status.csv、status.json。

**核对：** AJ 每行的两类 CIF 加生存概率为 1。两个病因别 Cox 各用 age/10、bmi/5、male 三个参数，不能把其系数解释为 30 天概率或 Fine–Gray 系数。

**练习：** 手核一行 AJ 的风险人数与概率和；对照非致死事件和死亡曲线。第 24b 章可研究 Fine–Gray 风险集原理，本章没有其完整回归或 B 验证。

**范围：** 未实现 Fine–Gray 回归；病因别 Cox 与子分布风险模型不同，不能把本章 Cox 结果冒称 Fine–Gray。 未由个体两类风险率计算条件 CIF，未实现 B 的时间点校准/判别/DCA 及内部/外部验证。 第 24、24b 章的 Fine–Gray 风险集小例不是本章患者的完整回归与评价。

**共享结果：** `results/branch_B.json`。

## 第 10 章：机器学习类模型

```sh
python run.py chapter 10 --mode quick
```

**本章产物：** `outer_comparison.csv`、`outer_roles.csv`、`outer_predictions.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 三个外层折重拟合基线与预设 ridge；每候选有 3,200 个 OOF 预测，测试病例不参与该折拟合。ridge 的填补/缩放由训练折学习。

**练习：** 按 candidate/id 检查 outer_predictions.csv 每个候选覆盖全部病例且只出现一次；抽查某 fold 的 TRAIN/TEST 与训练人数。

**范围：** 主线 A 未接入森林、boosting、SHAP 或 TabPFN；第 25、25b 章为独立模拟。 ridge 在训练折内学习设计列的缩放，惩罚作用于标准化后的系数；与旧版固定尺度 ridge 的数值不等价。

## 第 11 章：调参与嵌套验证

```sh
python run.py chapter 11 --mode quick
```

**本章产物：** `inner_tuning.csv`、`nested_metrics.csv`、`outer_roles.csv`、`outer_predictions.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 三个外层折内各运行三个内层折、三个 lambda；按平均 Brier 最小选值，外层测试只预测。共享 model.json 仍是无惩罚主线模型。

**练习：** 手核每个 outer_fold 的三个候选中谁被选中；改变评价集后不能重新选阈值或 lambda。解释为什么有限网格的最优不等于全局最优。

**范围：** 不是 1SE 选择；没有主线 MICE、变量筛选、再采样/早停或概率再校准的完整嵌套链。 未冻结最终在全样本重新调参的 ridge 对象；共享最终模型仍是无惩罚 A Logistic。 lambda 候选仅 1/10/100，有限网格不证明全局最优；惩罚作用于训练折内标准化设计。

## 第 12 章：内部验证

```sh
python run.py chapter 12 --mode quick
```

**本章产物：** `optimism_corrected_metrics.csv`、`development_percentile_ranges.csv`、`fixed_apparent_ranges.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 乐观校正表、固定模型范围、双重 bootstrap 开发范围目标不同；JSON 留存全部失败台账，不丢弃失败后补成功重复。

**练习：** 比较三个范围表的估计目标、B 与成功数；说明何者包含重开发，何者条件于现有概率。使用 book 预算复核 Monte Carlo 标准误，不能把快跑区间当书籍预算。

**范围：** 未执行建议的 10 折、留一中心或时间验证；第 10—11 章是另一套 3 折练习。 未执行 MI、统一收缩后模型交付或校正统计量名义覆盖率验证。 双重 bootstrap 范围与固定模型表观区间均不是最终固定模型真实外部性能区间。 PCG64 与旧版随机流不同；任何失败重复均使对应汇总不可估计，不丢弃失败后补跑成功样本。

**共享结果：** `results/bootstrap_optimism.csv`、`results/bootstrap_optimism.json`、`results/fixed_model_intervals.csv`、`results/fixed_model_intervals.json`、`results/development_intervals.csv`、`results/development_intervals.json`。

## 第 13 章：评价指标与决策曲线

```sh
python run.py chapter 13 --mode quick
```

**本章产物：** `calibration_curve.csv`、`risk_groups.csv`、`decision_curve.pdf`、`decision_curve.png`，以及 lesson.md、status.csv、status.json。

**核对：** CITL 为斜率固定 1 的 offset 截距，与同时估计的校准截距不同。蓝色带是表观曲线的逐点条件范围；橙色是校正点估计，只有 MC 误差。DCA 采用预设阈值。

**练习：** 对照 calibration_recipe.csv 的固定 logit 结点、支持范围和重复数；解释图上两条曲线与蓝色范围。需要阈值矩阵、PR 或成对 DeLong 时另运行工具箱，勿在评价集选 Youden。

**范围：** 主线未输出 ICI/E50/E90/Emax、scaled Brier、AUPRC、完整阈值表或配对模型差的 bootstrap 区间。 工具箱另有 ROC/PR、预设阈值混淆矩阵及 Wilson 区间、成对 DeLong、模型比较 API；不等于本章已自动生成全套主线表。 蓝色带只对应固定模型表观曲线的逐点条件经验范围；橙色校正曲线只有点估计与 Monte Carlo 标准误，没有抽样置信区间或同时置信带。 平滑曲线沿用固定 logit 结点的自然样条空间，不用 LOWESS 替代；未实现 B 的删失/竞争风险整套评价。

**共享结果：** `results/calibration_curve.csv`、`results/calibration_groups.csv`、`results/calibration_recipe.csv`、`results/calibration_log.csv`、`results/calibration.json`、`results/calibration.png`、`results/calibration.pdf`、`results/decision_curve.csv`。

## 第 14 章：外部验证与更新

```sh
python run.py chapter 14 --mode quick
```

**本章产物：** `synthetic_validation_data.csv`、`validation_roles.csv`、`update_coefficients.csv`、`held_out_update_evaluation.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 新合成 1,600 人中 800 人拟合 L1/L2 更新，另 800 人评价，ID 集合互不相交。改变事件率不构成真实外部验证。

**练习：** 用 validation_roles.csv 证明 UPDATE/TEST 不相交；同时看更新系数与另一半上的指标。预先决定更新条件，不能据同一评价结果再挑最优规则后宣称无偏。

**范围：** 没有真实外部队列；未输出完整病例组合分析、模型化预期 C、外部指标区间、L3 全系数或 L4 新项更新。 这是 Python PCG64 新模拟，不能与历史 R 验证队列逐人或逐位相等；模拟事件率变化不是临床跨医院证据。

## 第 15 章：稳健性、亚组、漂移与公平性

```sh
python run.py chapter 15 --mode quick
```

**本章产物：** `descriptive_subgroups.csv`、`input_perturbation.csv`、`sensitivity_summary.csv`、`subgroup_ledger.json`，以及 lesson.md、status.csv、status.json。

**核对：** 描述性亚组记录实际有效重复数。quick 的 40 次低于最低 100 次，区间留空；book 用 300 次。白蛋白扰动是数值敏感性而非治疗效应。

**练习：** 对每组同时读样本、事件、失败数和 interval_status；将“区间不可估计”与“无差异”区分。解释白蛋白扰动为何不是可实施治疗的反事实。

**范围：** 未实现缺失/结局/纳排/时间等完整敏感性场景、亚组联合校准检验、中心随机效应合并与新中心预测区间。 未实施流式漂移/输入分布监测或各组 TPR/FPR/PPV/净获益全表。 quick 仅 40 次亚组重抽样，低于 100 次有效重复要求，区间保留 NA；book 为 300 次。

## 第 16 章：反例实验室

```sh
python run.py chapter 16 --mode quick
```

**本章产物：** `negative_checks.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 四个调用均实际触发错误；已知病例、规则变更或未知结局不能被静默修正后通过。

**练习：** 保留 expected rejection 与 observed_error；在自己的代码中为真实常见错误再建最小反例。晚回报和测试侧填补分别见第 06、07 章。

**范围：** 本章不是原书 S16 十个历史反例的完整复跑。 晚回报与测试侧中位数边界另在第 06、07 章；筛选泄漏等另在数学第 28、31 章。

## 第 17 章：模型展示、投入临床与监测

```sh
python run.py chapter 17 --mode quick
```

**本章产物：** `example_predictions.csv`、`input_units.csv`、`model_registry.md`、`known_case_predictions.csv`、`known_case_predictions.csv.json`，以及 lesson.md、status.csv、status.json。

**核对：** model.json 可以按完整输入预测新病例；known_case_predictions.csv 只查原患者 ID。摘要只检测规则/源码一致性，可信登记来源需真实部署另行设计。

**练习：** 加载 FrozenPredictions 查一个已有 ID，然后查一个不存在的 ID，核对拒绝；用完整 JSON 模型为符合输入要求的新病例预测，区分这两种对象。

**范围：** 没有多语言预测接口、评分表/列线图、完整浏览器计算器、追加式影子日志或生产版本停用系统。 没有临床静默运行与长期监测；SHA-256 摘要只核查规则/源码一致性，不是可信签名。 FrozenPredictions 仅查找原有患者 ID，不能把已有概率表伪装成预测新病例的模型。

## 第 18 章：报告与模板包

```sh
python run.py chapter 18 --mode quick
```

**本章产物：** `report.md`、`interval_targets.csv`、`synthetic_update_evaluation.csv`，以及 lesson.md、status.csv、status.json。

**核对：** 报告和区间目标表来自同次模式与源码；报告仍含人工待补字段，不把合成更新写成临床外部证据。

**练习：** 复制报告到自己的研究目录，填伦理、数据权限、定位、署名、局限和清单映射；没有真实证据的项目保留 PENDING。

**范围：** 未迁移原书全部报告完整性/人工字段/锁定块检查器；生成草稿不能宣称报告已签审。

## 第 19 章：投稿实操

```sh
python run.py chapter 19 --mode quick
```

**本章产物：** `submission_checklist.csv`、`submission_materials.md`，以及 lesson.md、status.csv、status.json。

**核对：** READER_VERIFY、PENDING、SYNTHETIC_ONLY 是待办状态；本命令没有实际提交稿件。

**练习：** 按目标期刊的当前官方要求建立档案并写来源与日期；收集真实署名/伦理/投稿文件；不要把教学清单的生成状态当提交状态。

**范围：** 没有选刊政策实时采集、12 项完整投稿包、作者资格判定、锁定块审核或实际投稿/审稿回复。 READER_VERIFY、PENDING、SYNTHETIC_ONLY 是当前状态，不代表材料已齐或已提交。

## 第 20 章：AI 辅助的实操规矩

```sh
python run.py chapter 20 --mode quick
```

**本章产物：** `ai_audit.csv`、`human_review.csv`、`audit_instructions.md`，以及 lesson.md、status.csv、status.json。

**核对：** 教学执行、未执行行为和人工审核分别记录；命令不调用 AI 服务，不替人签名。

**练习：** 在自己的留痕副本记录一次真实核查的工具版本、用途、决定、核对方法和核对人；独立重算一条数值，实际打开参考文献原文后再记录已核查。

**范围：** 命令不调用 AI 服务或证明人已审核；未实现全部红线签名、文献签审、十错接受清单和每月审计。

## 第 21—31 章的入口约定

以下数学实验与主线 A 分开。`chapter 22` 或 `math 22` 同时运行 22/22b；`math 22b` 只运行补充实验。各目录的 result.json 记录真实预算和范围，快速重复模拟不当作书籍预算。所有 16 个脚本可用 `python run.py math all --mode quick`，完整预算改用 book。

## 第 21 章：似然、Logistic 与迭代加权最小二乘

```text
python run.py math 21 --mode quick
```

**产物：** `21/stdout.txt`、`21/result.json`。

**核对：** 2×2 例的 IRLS 与闭式系数一致；分离例展示系数发散，Firth 与 FLIC 展示有限估计及平均风险变化。训练内平均预测等于事件率、再校准斜率为 1 的性质有其无惩罚 ML 条件。

**练习：** 手算第一步更新 `(-1.2, 1.6)`，再核对最终截距 `log(2/8)`、斜率 `log(6)`。解释分离例的高表观拟合为什么不能视为模型优秀。

## 第 22 章：正则化与收缩

```text
python run.py math 22 --mode quick
```

**产物：** `22/` 与 `22b/` 的日志、结构化结果及实际预算。

**核对：** 正交设计中的岭、套索、弹性网闭式解与库实现对应；注意各库目标函数的惩罚缩放。补充实验有坐标下降、线性高斯插件收缩、重复列及非单调活跃集反例。

**练习：** 对 `z=(3,1,0.4)`、λ=0.5 手算岭与套索结果；核对首轮坐标更新 `(0.8,0.12)`。需要比较书中重复模拟时使用 `python run.py math 22 --mode book`，并保留 Monte Carlo 波动的解释。

## 第 23 章：样条与广义可加模型

```text
python run.py math 23 --mode quick
```

**产物：** `23/stdout.txt`、`23/result.json`。

**核对：** RCS 的参数数、线性尾部与基函数约束；P-spline EDF 随惩罚变化。当前夹持端结点、普通系数差分的零空间对下标线性，不必对 x 线性。

**练习：** 用结点 2、4、6、8 手核 `h1(5)=27`、`h2(5)=1`；再比较 P-spline 零空间极限与普通直线拟合，解释为什么 EDF≈2 不能单独证明函数是一条直线。

## 第 24 章：生存分析

```text
python run.py math 24 --mode quick
```

**产物：** `24/` 与 `24b/` 的日志、结构化结果。

**核对：** Cox 风险集、Breslow 基线及其参照点；Aalen–Johansen、IPCW 与 Fine–Gray 加权风险集。竞争死亡在 cause-1 Brier 中是已知非目标事件，不当作未知删失或 any-event 阳性。

**练习：** 手核 10 人例在 t=6 的 CIF1 为 `19/60`；打开 24b 的标签反例，比较使用 cause-1 与 any-event 标签时两预测的 Brier 排名。补充脚本可单独运行 `python run.py math 24b`。

## 第 25 章：树与集成

```text
python run.py math 25 --mode quick
```

**产物：** `25/` 与 `25b/` 的日志、结构化结果和预算。

**核对：** 基尼分裂、一步 Newton 提升、袋外比例与相关预测平均的方差；25b 在独立小样本设定下运行树类算法的概率评价，与主线 A 分开。

**练习：** 对 x<4.5 手算加权基尼 0.1875 与增益 0.28125；在 25b 结果同时查看 AUC 和校准量，记录何种结论只适用于该模拟设定。随机流或预算变化可改变 Monte Carlo 末位。

## 第 26 章：神经网络与表格基础模型（选读）

```text
python run.py math 26 --mode quick
```

**产物：** `26/` 与 `26b/` 的日志、结构化结果和预算。

**核对：** 2-2-1 网络前向、反向、参数计数及早停与岭的收缩；26b 使用 scikit-learn MLP 的独立权重衰减实验。这里只运行这些数学及网络实验，未加载或评价 TabPFN 等表格基础模型。

**练习：** 手算 20-64-64-1 网络共有 5,569 参数，并对照小样本预算；比较 26b 不同 `alpha` 下校准斜率及失败状态。说明库的 L2 缩放和训练规模为何需要一起记录。

## 第 27 章：评价指标的数学

```text
python run.py math 27 --mode quick
```

**产物：** `27/stdout.txt`、`27/result.json`。

**核对：** AUC/DeLong、校准、Brier 分解与净获益分别回答不同问题。单调变换可维持排序而改变概率校准；分箱的 Brier 分解不能自动等同于原始连续预测分解。

**练习：** 对 10 人例逐对比较，手算 AUC=`20/24`；在 logit 缩放例中查看 AUC 不变而斜率变化。将这一结果写成对第 13 章主结果表的解释。

## 第 28 章：重抽样

```text
python run.py math 28 --mode quick
```

**产物：** `28/stdout.txt`、`28/result.json`。

**核对：** 小样本 bootstrap 枚举、学习曲线、交叉验证相关性、流程选择与嵌套评价。实际成功数、失败及预算写入输出；快速模拟不能当作书中长模拟数值。

**练习：** 对 6 人样本均值模型核对表观 MSE `40/6` 与乐观校正逻辑；在纯噪声实验比较“先筛再固定验证”与重做筛选的结果，说明为何必须重复完整开发流程。

## 第 29 章：缺失数据

```text
python run.py math 29 --mode quick
```

**产物：** `29/stdout.txt`、`29/result.json`。

**核对：** Rubin 规则、MCAR/MAR/MNAR、结局参与研究填补、PMM 供体支持及开发—部署相容性。数学实验里的自编机制与计算不是主线多变量 MICE 的完整部署链。

**练习：** 手算五份估计的 `T=Ubar+(1+1/m)B`，比较只用 Ubar 的标准误；在 Logistic 缺失例比较“补条件均值再预测”与“条件抽样后平均概率”，说明二者何时会不同。

## 第 30 章：样本量公式

```text
python run.py math 30 --mode quick
```

**产物：** `30/stdout.txt`、`30/result.json`。

**核对：** 开发三准则取最大；外部验证按各指标精度另算。公开规划输入 18 参数、事件率 0.12、预期 C=0.80 的本脚本积分结果与理论册近似、包工具输出分别报告；它不是 `pmsampsize` 或 `pmvalsampsize` 实跑。

**练习：** 比较 C=0.80 与 C=0.70 时的开发人数，解释主线采用 3,200 人的依据；再比较外部验证斜率区间宽度 0.2 与 0.4 的人数变化。核对 O/E 与净获益使用的尺度，不能把不同尺度的宽度当成同一输入。

## 第 31 章：采样、处理与反事实预测

```text
python run.py math 31 --mode quick
```

**产物：** `31/` 与 `31b/` 的日志、结构化结果和预算。

**核对：** 病例对照抽样与训练再平衡改变概率口径；先验校正需可靠总体事件率，且不能自动修复小样本斜率。反事实预测区分给定完整 L 的风险与仅有 X 的边际风险，识别和评价需要相应假设。

**练习：** 比较平衡抽样前、未校正后和先验校正后的平均风险及斜率；在 31b 比较 `m0(L)` 与 `E[Y0|X]`，说明标准化所积分的人群分布。记录 IPW 的支持与有效样本量，不能把观察性预测系数直接解释为治疗效果。

## 完成后核查

```sh
python run.py course --mode quick
python run.py math all --mode quick
python run.py verify --mode quick
python -m unittest discover -s tests -v
```

verify 会独立重算主线预测、表观 C/Brier，并检查已登记章节与源码；它不表示未运行章节或人工任务已经完成。最终逐章检查已生成的状态、范围和文件；将真实研究的责任人、证据来源和适用结论写入自己的记录。
