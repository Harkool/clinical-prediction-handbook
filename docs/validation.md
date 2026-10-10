# 实际验证与证据边界

本页记录项目的实跑检查。运行日期：2026-10-11；本地环境：Windows、R 4.6.1（survival 3.8.6）、Python 3.9.20、NumPy 2.0.2、SciPy 1.13.1、scikit-learn 1.6.1。数学安装锁定版本支持 Python 3.9—3.12；持续集成使用 Python 3.11。

## 数学实验

第 21—31 章的 16 个脚本分别以 quick 与 book 实际执行，每种模式的 47 项数值验收全部通过。记录包含实际预算、关键结果和环境，见 [reference_results.json](../math/reference_results.json)。确定性合同另由 5 项 Python 测试核对，包括二乘二表 log OR、原因 1 Brier、反向传播梯度及稳定权重目标。

第 25 章将相关正态树预测的大矩阵换成其均值的精确正态分布抽样，保留分布和理论方差；随机流消耗变化，所以不要求模拟末位相同。其他快速模拟预算与书籍预算的差异逐项列于 [数学指南](math-labs.md)。

## R 课程与独立核查

第 1—20 章已经以 quick 与 book 完成实际运行。完整预算包括开发 200 次、条件评价 500 次、校准 200 / 500 次，以及双重 bootstrap 200 × 50。输入保护及状态记录的最终修订已通过 quick 全 20 章及 20 项 R 语义检查；完整预算的最终重跑仍在进行，发行前更新此记录。

独立 Python 设计矩阵重算原书合成主线的全部 3200 个预测，最大差为 `2.9976021664879227e-15`，C = `0.7712383848248106`，Brier = `0.09069047612869838`，事件数 384。验证对 CSV 系数、输入、预测与 R 表观指标同时检查，并在报告中绑定这些文件的 SHA-256。

15 项 Python 测试全部通过，实际触发了错误预测、患者错序、非有限输入/指标、缺文件和失败重跑。成功记录每次先作废为 RUNNING，本次失败写 FAIL，避免旧 PASS 被误认为当前结果。R 课程的输入拒绝、标签、单位、折分工及人工待审项目由 `tests/test_course.R` 的 20 项语义检查验证。

## 自己复验

```sh
python run.py course --mode quick
python run.py verify --mode quick
Rscript --vanilla tests/test_course.R
python -m unittest discover -s tests -v
python run.py math all --mode quick
```

新机器上从同一提交重新运行，不搬运开发者缓存。`outputs/` 中的数据、模型、日志和图表只属于该次本地操作，没有纳入仓库。`examples/reference/` 的旧书门禁属于参考产物，不能复制成自己的审阅决定。

GitHub Actions 已配置 Windows、macOS、Linux 的快速流程；具体远端运行状态以 [Actions](https://github.com/Harkool/clinical-prediction-handbook/actions) 为准，本地实跑不能代替尚未完成的远端检查。完整预算的远端任务按需手动触发。

## 这些检查证明什么

检查验证代码按声明数据流执行、数值合同及导出一致性。合成外部队列只练习验证/更新操作；B 分支只完成估计；完整 MICE 和表格基础模型未接入主线。有限预算区间没有已证实的外部覆盖保证，条件曲线带没有同时覆盖保证。临床可用性、真实数据质量、伦理与人工签字仍须另行完成。
