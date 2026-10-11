# 冻结病例 A 与历史数值参考

`main_case.csv` 是原书合成病例 A 的冻结输入：3200 行、25 列、384 个复合结局事件，其中 39 例 CD V 死亡。主线使用 14 个输入、18 个待估参数，含截距共 19 个系数。输入保留历史生成值，去除了生成器的理想风险答案列 `true_risk`；它不是临床数据。

v2 的纯 Python 课程直接读取这份输入。原书历史生成种子为 20261010；Python 新增验证数据与重抽样使用独立的 NumPy PCG64 随机流，无需仿制历史生成器。

|文件|用途|
|---|---|
|`main_case.csv`|读者课程的冻结主线输入|
|`model_coefficients.csv`、`predictions.csv`|历史模型的 19 个系数及 3200 行预测，供数值移植核对|
|`apparent_metrics.csv`|历史开发样本的表观指标|
|`bootstrap_optimism.csv`、`development_intervals.csv`|历史内部校正与双重 bootstrap 经验分位数范围|
|`calibration_curve.csv`、`calibration.png`|历史校正平滑曲线与固定模型表观曲线的条件逐点带|
|`calibration_groups.csv`、`calibration_recipe.csv`|历史分组与平滑规格；Wilson 分组区间为描述性区间|
|`dca.csv`、`subgroups.csv`|历史教学阈值净获益与描述性亚组指标|
|`gate_status.csv`、`gate_scope_status.csv`|历史产物的限定检查状态，不能复制为读者审核结果|

历史重抽样预算为开发 200、固定评价 500、校准 200/500、双重 bootstrap 200 × 50、亚组 300。它们保留作历史对照；当前 Python 随机流及较小 quick 预算不应逐项等于这些数值。

2026-10-11 的 Python 固定主线拟合与历史参考相比，系数最大绝对差为 `1.4707823847714963e-10`，预测最大绝对差为 `7.190983919436178e-12`。当前模型的完整规则和 SHA-256 身份写入读者运行目录的模型 JSON；历史文本身份文件已不再发布。

运行：

```sh
python run.py chapter 08 --mode book
python run.py verify --mode book
```

这会读取冻结输入，输出当前 Python 结果，再独立重算预测与表观 C/Brier。完整课程用 `python run.py course --mode book`。数值一致性与章节执行覆盖分别记录；参考目录不充当本次运行缓存。

本目录各文件的字节摘要见 [checksums.sha256](checksums.sha256)。摘要可以检查下载完整性，换行或数字格式变化也会改变摘要。比较数值前应核对输入、预测规则、评价目标、随机流和预算。

这些作者生成的合成教学数据与数值参考按仓库 MIT 使用；书稿和 PDF 未包含在本目录。当前验证范围见 [validation.md](../../docs/validation.md)。
