# 31 章操作指南

本指南把书中步骤对应到仓库的实际入口、产物和练习。先完成 [安装](installation.md)。以下命令在仓库根目录运行；Windows 或 macOS/Linux 的虚拟环境 Python 可替换命令中的 `python`。

## 读文件的约定

第 01—20 章默认使用快速模式。每章命令都会准备必要的共用合成数据和基线拟合；无需先手工运行前面的所有章节。每章的 `lesson.md` 给出本次操作说明，`status.csv` 记录执行状态。

| 路径 | 内容 |
| --- | --- |
| `outputs/quick/data/main_case.csv` | 3,200 人主线 A 合成教学数据 |
| `outputs/quick/results/model.rds` | 拟合、recipe、目标和元数据 |
| `outputs/quick/results/model_coefficients.csv` | 含截距的 19 项完整精度系数 |
| `outputs/quick/results/predictions.csv` | 主线逐人标签及预测 |
| `outputs/quick/chapters/chNN/` | 各章实际练习文件 |
| `outputs/quick/manifest.csv` | 本模式中实际执行的章节清单 |
| `outputs/quick/math/编号/` | 数学实验的 `stdout.txt` 和 `result.json` |

下文“产物”中，01—20 章的文件名相对于该章 `chapters/chNN/`；数学文件相对于该模式的 `math/`。`--mode book` 将结果保存到 `outputs/book/` 并使用书籍预算。快速与书籍预算见 [安装说明](installation.md)。

R 课程每次重新生成主线数据，不会把读者手改的 `main_case.csv` 静默当作新的主线。修改数据或方法的实验，应在独立的本机副本中调用相应函数，并记录新的实验说明；不要覆盖参考文件后沿用主线名称。

完成第 08 章后，可在仓库根目录打开 R/RStudio，直接练习预测接口：

```r
source("R/core.R")
source("R/interface.R")
d <- read.csv("outputs/quick/data/main_case.csv")
m <- readRDS("outputs/quick/results/model.rds")
expected <- readLines("outputs/quick/results/prediction_identity.txt")
predict_case_with_units(m, d[1:3, ], case_units(), expected)

bad_units <- case_units()
bad_units["hb"] <- "unknown"
predict_case_with_units(m, d[1:3, ], bad_units, expected) # 应拒绝未知单位
```

预期错误也是练习证据。这里使用同次运行登记的身份做一致性核查；真实部署的可信身份来源需要另行设计。

第 21—31 章是独立数学与方法实验 M21—M31。`math 22` 运行该章 22 和 22b 两个脚本；`math 22b` 只运行补充脚本。默认快速模式缩减重复模拟，实际预算和范围写入 `result.json`；书中预算用 `--mode book`。详细信息见 [数学实验指南](math-labs.md)。

## 第 01 章：使用说明与总原则

```text
python run.py chapter 01 --mode quick
```

**产物：** `orientation_checks.csv`、`lesson.md`、`status.csv`。

**核对：** 主线 A 有 14 个原始输入、18 个非截距参数；预测时点和结局计时原点分别定义。技术检查的 PASS 不代表研究问题、临床结论或签名得到批准。

**练习：** 复制 `templates/analysis_plan.csv` 到本机学习目录，写出目标人群、预测时点、结局、验证层级和结论范围。在未由真实负责人审核前保留人工待审状态。

## 第 02 章：把项目、环境与运行记录建起来

```text
python run.py doctor
python run.py chapter 02 --mode quick
```

**产物：** `session_info.txt`、`run_record.csv`、`environment.csv`。

**核对：** R 版本、种子、模式与模型身份来自本次运行。必要包的实际安装状态与数学依赖分别检查。

**练习：** 保存本次 Git 提交、完整命令和环境记录；在另一个终端运行同一入口，核对所用 Rscript 仍是同一版本。多个 R 并存时用 `--rscript` 明确指定。

## 第 03 章：准备数据与合成教学案例

```text
python run.py chapter 03 --mode quick
```

**产物：** `data_summary.csv`、`field_profile.csv`，以及共享 `data/main_case.csv`。

**核对：** 固定种子 20261010 生成 3,200 名合成成年人；14 个输入完整、患者 ID 唯一。观察事件数由生成过程产生，不能为凑计数筛种子。`true_risk` 是生成器答案信息，不能进入设计矩阵。

**练习：** 用 CSV 工具按 `center` 汇总人数及事件数，再将 `case_recipe()$predictors` 与实际模型设计核对，确认 ID、结局、时间和 `true_risk` 均不作为预测变量。

## 第 04 章：把六要素写成规格表与队列代码

```text
python run.py chapter 04 --mode quick
```

**产物：** `six_elements.csv`、`cohort_flow.csv`。

**核对：** 对象、预测时点、输入、结局、窗口和场景对应一致。主线在诱导前预测，从实际手术开始观察结局；实际手术信息不因此成为预测时已知输入。

**练习：** 从共享数据抽取 5 人，逐人比较 `decision_time`、`outcome_origin`、`collection_time` 和 `result_available_time`。在自己的规格副本中写明“预测前采集、预测后回报”的值如何处理，并在第 06 章查看相应拒绝用例。

## 第 05 章：先把结局、竞争事件与未知标签分清

```text
python run.py chapter 05 --mode quick
```

**产物：** `labels_A_B.csv`、`label_fixtures.csv`、`negative_checks.csv`。

**核对：** A 的 CD ≥ III 包括死亡；B 的非致死目标中，先发生死亡是竞争事件。提前结束有效观察且无已知事件时，标签为未知；不能补成 0 或 1。

**练习：** 在 `label_fixtures.csv` 找到第 12 天结束观察且无事件的病例，解释其未知状态；在 `labels_A_B.csv` 找到死亡病例，对照 A 标签与 B 状态。打开负向检查，确认死亡后的非致死事件被拒绝。

## 第 06 章：变量审查表与防泄漏管线

```text
python run.py chapter 06 --mode quick
```

**产物：** `variable_review.csv`、`negative_checks.csv`。

**核对：** 采集时间与结果可用时间均须合格，未知可得时间不能按“未发现问题”通过。预处理、筛选、调参、填补和基线系数拟合各自限制在相应训练侧。

**练习：** 读取 `delayed_result` 和 `unknown_availability` 两个用例的拒绝原因；为自己的变量审查表增加“采集字段”“回报字段”“取值规则”“决定与理由”列，分别填一项合格与不合格输入。

## 第 07 章：样本量、参数预算与缺失处理

```text
python run.py chapter 07 --mode quick
```

**产物：** `parameter_budget.csv`、`missing_recipe.csv`、`missing_assignments.csv`、`held_out_median_demo.csv`、`boundary_checks.csv`。

**核对：** 四结点年龄样条 3 参数，ASA 四等级 3 参数，其余 12 项各 1，合计 18。EPP 是描述量。这里的缺失练习仅在训练行估计 Hb 中位数，并用于测试行，**不是多重填补**；默认课程没有执行完整 MICE 链。

**练习：** 在 `boundary_checks.csv` 比较改变测试 Hb 前后的训练中位数和全数据中位数；前者应不变。读 `missing_assignments.csv`，按 TRAIN/TEST 统计缺失数，再用第 30 章比较开发样本量与外部验证精度规划。

## 第 08 章：回归类模型

```text
python run.py chapter 08 --mode quick
```

**产物：** `coefficients.csv`、`fit_state.csv`、`unit_roundtrip.csv`，以及共享 `results/model.rds`。

**核对：** 固定结构 Logistic 使用完整精度系数、固定年龄结点和预设肌酐对数变换。检查收敛、设计秩与梯度；数值问题不能被高表观 C 掩盖。

**练习：** 在 `unit_roundtrip.csv` 逐人比较标准单位与显式声明 g/dL、mg/dL 后的预测。Hb/Alb 乘 10、肌酐乘 88.4 转换后，预测应在数值容差内一致。

## 第 09 章：生存与竞争风险：独立 B 分支

```text
python run.py chapter 09 --mode quick
```

**产物：** `branch_counts.csv`、`branch_coefficients.csv`；共享结果中的 `branch_B_*` 文件和分支日志。

**核对：** B 的结局是先于竞争死亡发生的非致死 III/IV 并发症。实际分支采用 age、bmi、male 三参数 Fine–Gray 教学拟合，使用扩展数据权重和患者聚类。系数是子分布风险率关联，不能解释为 30 天概率或治疗效应。

**练习：** 将 B 的状态计数与第 05 章对照，核对死亡没有同时作为普通删失与竞争事件；再运行第 24、24b 数学实验，比较 CIF 与将竞争事件删失后的 `1-KM`。

**当前范围：** 本章运行事件构造与模型估计，尚未完成 B 的独立校准、判别、内部或外部验证链。

## 第 10 章：机器学习类模型

```text
python run.py chapter 10 --mode quick
```

**产物：** `outer_comparison.csv`、`outer_roles.csv`。

**核对：** 当前可运行课程比较普通 Logistic 与 base R 实现、预设 λ = 10 的岭 Logistic。两者在每个外层训练折重新拟合，并评价同一测试折。此处演示诚实比较的数据流；森林、提升和 TabPFN 的书中讨论属于扩展。

**练习：** 在 `outer_roles.csv` 筛选一个 `outer_fold`，核对每名患者只有 TRAIN 或 TEST 一个角色；对照该折两候选的 `train_n`、`test_n`、C、Brier 和校准量。不要把全样本基线系数直接当作折外预测。

## 第 11 章：调参与嵌套验证

```text
python run.py chapter 11 --mode quick
```

**产物：** `inner_tuning.csv`、`nested_metrics.csv`、`outer_roles.csv`。

**核对：** 外层 3 折评价；每个外层训练集内再用 3 折按 Brier 在 λ ∈ {1, 10, 100} 中选择，之后用该外层完整训练集重拟合。实际选择规则是最小平均 Brier，不写成 1SE。

**练习：** 为每个外层折记录所选 λ 与外层测试 Brier，确认 `test_rows_used_for_tuning` 为 0。只在同一外层划分内检查训练与测试不交叠，不能要求所有折训练 ID 的并集与测试 ID 的并集不交叠。

## 第 12 章：内部验证

```text
python run.py chapter 12 --mode quick
```

**产物：** `optimism_corrected_metrics.csv`、`development_percentile_ranges.csv`、`fixed_apparent_ranges.csv`；共享 `results/` 中保存重复结果。

**核对：** 每个开发 bootstrap 重构设计并重新拟合。双重 bootstrap 对重新开发后内部校正统计量给出经验分位数范围；固定预测重抽样描述原开发样本中的表观评价变异，两者目标不同。

**练习：** 对 C 和 Brier，核对“表观 − 平均乐观 = 校正”。分别写一句话解释两张范围表；再运行 `--mode book`，比较计算误差与预算变化，保留两种模式的结果。

**解释范围：** 有限双重 bootstrap 不提供最终固定模型真实外部性能的名义覆盖保证；训练内校准量还受自拟合约束。

## 第 13 章：评价指标与决策曲线

```text
python run.py chapter 13 --mode quick
```

**产物：** `calibration_curve.csv`、`risk_groups.csv`、`decision_curve.pdf`；共享 `results/calibration.png`、`calibration_recipe.csv` 和 `decision_curve.csv`。

**核对：** 平滑校准规则、概率网格、支持范围及有效次数可追踪。蓝色带属于固定模型**表观曲线的逐点条件经验范围**；橙色乐观校正曲线仅给点估计与 Monte Carlo 误差。风险分布人数要闭合。

**练习：** 对 `risk_groups.csv` 的人数求和，与 3,200 人核对；在决策曲线表中选一个事先列出的阈值，比较模型、全处理、不处理三条策略。解释为什么表观曲线的区间不能附到校正曲线上。

## 第 14 章：外部验证与更新

```text
python run.py chapter 14 --mode quick
```

**产物：** `synthetic_validation_data.csv`、`validation_roles.csv`、`update_coefficients.csv`、`held_out_update_evaluation.csv`。

**核对：** 独立种子生成 1,600 人、规划率 0.18 的合成新数据；800 人估计截距或截距加斜率更新，另 800 人评价冻结及更新规则。更新与评价角色分离。

**练习：** 按 `validation_roles.csv` 检查 UPDATE/TEST 各 800 人；比较三规则在 TEST 上的 C、Brier、校准截距与斜率。为结果写一句准确限定为合成迁移练习的结论。

**当前范围：** 该实验演示独立数据评价与更新结构，没有真实外部队列，也不提供实际跨医院迁移证据。

## 第 15 章：稳健性、亚组、漂移与公平性

```text
python run.py chapter 15 --mode quick
```

**产物：** `descriptive_subgroups.csv`、`input_perturbation.csv`、`sensitivity_summary.csv`。

**核对：** 亚组表现属于开发样本内描述。快速模式亚组重抽样仅 40 次，未达到函数预设的 100 次有效重复要求，区间会保留 NA；书籍模式使用 300 次。不可估或精度不足时说明原因。

**练习：** 找出事件数较少的亚组，记录其区间状态及原因；计算白蛋白减 2 g/L 后的平均预测变化，解释其模型数值敏感性范围。若需亚组区间，运行本章 `--mode book` 并检查实际成功次数。

**当前范围：** 输入扰动不代表治疗效果；当前课程不执行真实流式漂移监测，也不能凭这些亚组表断言公平性。

## 第 16 章：反例实验室

```text
python run.py chapter 16 --mode quick
```

**产物：** `negative_checks.csv`。

**核对：** 四个当前可执行反例分别针对未知单位、预测 recipe 改变、完全分离和提前截止下的未知标签。`PASS_EXPECTED_REJECTION` 表示注入错误已被指定检查拒绝。

**练习：** 逐条读 `observed_error`，给每个反例写出受保护的契约；再对照第 05、06、08、17 章检查形成自己的反例清单。书中其他历史 S16 反例不是这个四用例输出的实跑结果。

## 第 17 章：模型展示、投入临床与监测

```text
python run.py chapter 17 --mode quick
python run.py verify --mode quick
```

**产物：** `example_predictions.csv`、`input_units.csv`、`model_registry.md`，以及共享模型与预测文件。

**核对：** 预测接口使用显式单位和完整预测规则身份。身份覆盖 recipe、目标、单位、系数和相关实现；MD5 用于非恶意一致性核查，不是签名。`verify` 独立重算主线 3,200 人风险值。

**练习：** 对照第 08 章单位往返结果，检查前 20 人预测一致；阅读第 16 章 recipe 结点篡改用例，说明“系数相同”为何不能证明规则相同。

**当前范围：** 课程交付本地预测与登记练习，没有完成真实临床静默运行、生产服务或长期监测。

## 第 18 章：报告与模板包

```text
python run.py chapter 18 --mode quick
```

**产物：** `report.md`、`interval_targets.csv`、`synthetic_update_evaluation.csv`。

**核对：** 报告从本次结果生成，模型、数据、模式与区间目标对应。合成新数据的更新评价与 A 的内部验证分开。伦理、署名、清单适用性及临床解释仍需人工审核。

**练习：** 从报告任一数字追溯到第 12 或 13 章 CSV；用 `interval_targets.csv` 核查每种范围包含与排除的不确定性。在本机副本中补写数据、验证和计算预算的局限，保持证据范围一致。

## 第 19 章：投稿实操

```text
python run.py chapter 19 --mode quick
```

**产物：** `submission_checklist.csv`、`submission_materials.md`。

**核对：** 清单标记 `READER_VERIFY`、`PENDING` 或 `SYNTHETIC_ONLY`，不是已完成投稿。材料的实际存在、内容完整和人工确认分开核查；目标期刊政策由读者在使用时查官方来源。

**练习：** 选择一项 `READER_VERIFY`，执行对应章节，将真实证据位置写入自己的清单副本。对伦理与临床外部验证字段，按合成示例的实际范围说明适用性，不杜撰机构、审批编号或作者。

## 第 20 章：AI 辅助的实操规矩

```text
python run.py chapter 20 --mode quick
```

**产物：** `ai_audit.csv`、`human_review.csv`、`audit_instructions.md`。

**核对：** 生成教学产物、未执行行为和人工审核分别记录。工具运行不证明人已阅读、采纳或批准；人工项保持 PENDING，直到实际完成。

**练习：** 在自己的留痕副本中记录一次真实的方案或代码核查，包括工具与版本、用途、采纳/拒绝决定、核对方法和核对人；没有执行的行为仍标未执行。将一条数值结论用独立计算核对，不能以 AI 输出代替证据。

## 第 21 章：似然、Logistic 与迭代加权最小二乘

```text
python run.py math 21
```

**产物：** `21/stdout.txt`、`21/result.json`。

**核对：** 2×2 例的 IRLS 与闭式系数一致；分离例展示系数发散，Firth 与 FLIC 展示有限估计及平均风险变化。训练内平均预测等于事件率、再校准斜率为 1 的性质有其无惩罚 ML 条件。

**练习：** 手算第一步更新 `(-1.2, 1.6)`，再核对最终截距 `log(2/8)`、斜率 `log(6)`。解释分离例的高表观拟合为什么不能视为模型优秀。

## 第 22 章：正则化与收缩

```text
python run.py math 22
```

**产物：** `22/` 与 `22b/` 的日志、结构化结果及实际预算。

**核对：** 正交设计中的岭、套索、弹性网闭式解与库实现对应；注意各库目标函数的惩罚缩放。补充实验有坐标下降、线性高斯插件收缩、重复列及非单调活跃集反例。

**练习：** 对 `z=(3,1,0.4)`、λ=0.5 手算岭与套索结果；核对首轮坐标更新 `(0.8,0.12)`。需要比较书中重复模拟时使用 `python run.py math 22 --mode book`，并保留 Monte Carlo 波动的解释。

## 第 23 章：样条与广义可加模型

```text
python run.py math 23
```

**产物：** `23/stdout.txt`、`23/result.json`。

**核对：** RCS 的参数数、线性尾部与基函数约束；P-spline EDF 随惩罚变化。当前夹持端结点、普通系数差分的零空间对下标线性，不必对 x 线性。

**练习：** 用结点 2、4、6、8 手核 `h1(5)=27`、`h2(5)=1`；再比较 P-spline 零空间极限与普通直线拟合，解释为什么 EDF≈2 不能单独证明函数是一条直线。

## 第 24 章：生存分析

```text
python run.py math 24
```

**产物：** `24/` 与 `24b/` 的日志、结构化结果。

**核对：** Cox 风险集、Breslow 基线及其参照点；Aalen–Johansen、IPCW 与 Fine–Gray 加权风险集。竞争死亡在 cause-1 Brier 中是已知非目标事件，不当作未知删失或 any-event 阳性。

**练习：** 手核 10 人例在 t=6 的 CIF1 为 `19/60`；打开 24b 的标签反例，比较使用 cause-1 与 any-event 标签时两预测的 Brier 排名。补充脚本可单独运行 `python run.py math 24b`。

## 第 25 章：树与集成

```text
python run.py math 25
```

**产物：** `25/` 与 `25b/` 的日志、结构化结果和预算。

**核对：** 基尼分裂、一步 Newton 提升、袋外比例与相关预测平均的方差；25b 在独立小样本设定下运行树类算法的概率评价，与主线 A 分开。

**练习：** 对 x<4.5 手算加权基尼 0.1875 与增益 0.28125；在 25b 结果同时查看 AUC 和校准量，记录何种结论只适用于该模拟设定。随机流或预算变化可改变 Monte Carlo 末位。

## 第 26 章：神经网络与表格基础模型（选读）

```text
python run.py math 26
```

**产物：** `26/` 与 `26b/` 的日志、结构化结果和预算。

**核对：** 2-2-1 网络前向、反向、参数计数及早停与岭的收缩；26b 使用 scikit-learn MLP 的独立权重衰减实验。这里只运行这些数学及网络实验，未加载或评价 TabPFN 等表格基础模型。

**练习：** 手算 20-64-64-1 网络共有 5,569 参数，并对照小样本预算；比较 26b 不同 `alpha` 下校准斜率及失败状态。说明库的 L2 缩放和训练规模为何需要一起记录。

## 第 27 章：评价指标的数学

```text
python run.py math 27
```

**产物：** `27/stdout.txt`、`27/result.json`。

**核对：** AUC/DeLong、校准、Brier 分解与净获益分别回答不同问题。单调变换可维持排序而改变概率校准；分箱的 Brier 分解不能自动等同于原始连续预测分解。

**练习：** 对 10 人例逐对比较，手算 AUC=`20/24`；在 logit 缩放例中查看 AUC 不变而斜率变化。将这一结果写成对第 13 章主结果表的解释。

## 第 28 章：重抽样

```text
python run.py math 28
```

**产物：** `28/stdout.txt`、`28/result.json`。

**核对：** 小样本 bootstrap 枚举、学习曲线、交叉验证相关性、流程选择与嵌套评价。实际成功数、失败及预算写入输出；快速模拟不能当作书中长模拟数值。

**练习：** 对 6 人样本均值模型核对表观 MSE `40/6` 与乐观校正逻辑；在纯噪声实验比较“先筛再固定验证”与重做筛选的结果，说明为何必须重复完整开发流程。

## 第 29 章：缺失数据

```text
python run.py math 29
```

**产物：** `29/stdout.txt`、`29/result.json`。

**核对：** Rubin 规则、MCAR/MAR/MNAR、结局参与研究填补、PMM 供体支持及开发—部署相容性。数学实验里的自编机制与计算不是 R `mice` 包的完整部署链。

**练习：** 手算五份估计的 `T=Ubar+(1+1/m)B`，比较只用 Ubar 的标准误；在 Logistic 缺失例比较“补条件均值再预测”与“条件抽样后平均概率”，说明二者何时会不同。

## 第 30 章：样本量公式

```text
python run.py math 30
```

**产物：** `30/stdout.txt`、`30/result.json`。

**核对：** 开发三准则取最大；外部验证按各指标精度另算。公开规划输入 18 参数、事件率 0.12、预期 C=0.80 的本脚本积分结果与理论册近似、R 包工具输出分别报告；它不是 `pmsampsize` 或 `pmvalsampsize` 实跑。

**练习：** 比较 C=0.80 与 C=0.70 时的开发人数，解释主线采用 3,200 人的依据；再比较外部验证斜率区间宽度 0.2 与 0.4 的人数变化。核对 O/E 与净获益使用的尺度，不能把不同尺度的宽度当成同一输入。

## 第 31 章：采样、处理与反事实预测

```text
python run.py math 31
```

**产物：** `31/` 与 `31b/` 的日志、结构化结果和预算。

**核对：** 病例对照抽样与训练再平衡改变概率口径；先验校正需可靠总体事件率，且不能自动修复小样本斜率。反事实预测区分给定完整 L 的风险与仅有 X 的边际风险，识别和评价需要相应假设。

**练习：** 比较平衡抽样前、未校正后和先验校正后的平均风险及斜率；在 31b 比较 `m0(L)` 与 `E[Y0|X]`，说明标准化所积分的人群分布。记录 IPW 的支持与有效样本量，不能把观察性预测系数直接解释为治疗效果。

## 完成后核查

```text
python run.py course --mode quick
python run.py verify --mode quick
```

对照课程 `manifest.csv`、各章 `status.csv`、实际 CSV/图表与报告。学习记录保留章节、命令、证据文件和解释；未完成的人审、真实外部验证或临床评价保持实际状态。问题处理见 [故障排查](troubleshooting.md)，贡献方法见 [CONTRIBUTING](../CONTRIBUTING.md)。
