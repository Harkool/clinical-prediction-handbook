# 原书合成病例 A 的参考结果

这些文件来自 2026-10-11 第二次修订版实操册已执行的合成病例 A，不是读者本次运行的结果。3200 行、384 个复合结局事件，其中 39 例 CD V 死亡；14 个输入、18 个待估参数。生成种子为 20261010，R 4.6.1。

完整预算为开发 bootstrap 200 次、固定模型评价 500 次、校准开发 200 次 / 条件重抽样 500 次、双重 bootstrap 200 × 50、亚组 300 次。核心模型系数和逐人预测可用于代码移植的一致性检查；小预算重抽样不应逐项等于这些完整预算结果。

|文件|解释|
|---|---|
|`model_coefficients.csv`、`predictions.csv`|冻结病例 A 的 19 个系数（含截距）及 3200 行预测|
|`model_payload.txt`、`model_identity.txt`|完整预测规则的规范化载荷与 MD5 一致性标识；不是签名或安全校验机制|
|`apparent_metrics.csv`|原开发样本上的表观指标|
|`bootstrap_optimism.csv`|固定候选开发流程的内部乐观偏倚校正|
|`development_intervals.csv`|内部校正开发统计量的双重 bootstrap 经验分位数范围|
|`calibration_curve.csv`、`calibration.png`|校正平滑曲线没有区间；条件带属于固定模型的表观曲线|
|`calibration_groups.csv`、`calibration_recipe.csv`|事先固定的分组与平滑规格；分组区间为描述性 Wilson 区间|
|`dca.csv`|事先教学阈值的内部校正净获益，没有获批的临床行动|
|`subgroups.csv`|描述性亚组指标；不作亚组优势证明|
|`gate_status.csv`、`gate_scope_status.csv`|原书产物的限定自动检查状态，不能复制为读者的审核记录|

运行 `python run.py chapter 08 --mode book` 会重新生成同一病例，再用 `python run.py verify --mode book` 独立重算本次导出的预测与表观指标。各文件字节摘要见 `checksums.sha256`。CSV 换行、数字格式或软件版本变化可能改变文件摘要；数值比较需先核对同一数据、规则、随机种子和预算。

全部为作者生成的合成教学产物，按仓库 MIT 使用。书稿与 PDF 未包含在这些文件中。
