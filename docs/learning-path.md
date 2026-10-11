# 学习路径

先按 [安装指南](installation.md) 准备同一个 Python 环境，逐章入口和范围见 [逐章指南](chapter-guide.md)。

## 先形成可核查结果

```text
python run.py doctor
python run.py course --mode quick
python run.py verify --mode quick
```

打开第 18 章报告、第 8 章模型与第 13 章图表，找到每个数字的来源。主线为冻结 3200 例合成数据、14 输入、18 个非截距参数；术前评估完成且诱导前预测，从实际手术开始观察 30 天，CD ≥ III 包括死亡。

固定设计与逐人预测用于数值复现；新验证队列及重抽样使用单独记录的 Python 随机流。

## 临床读者

按 1 → 4 → 5 → 7 → 18 阅读，与统计合作者运行 2、3、8、12、13。将六要素改写为自己的研究问题，区分完整无事件、窗口内事件、失访与死亡。

```text
python run.py baseline --input examples/reference/main_case.csv
```

核对观察分母、分类水平、单位和缺失数，比较显示表与数值 CSV。

## 统计读者

按 6 → 7 → 8 → 11 → 12 → 13 → 14 → 15 → 16 学习，再进入 21—24、27—30 的数学实验。追踪同一外层折的训练/评价 ID、内层调参边界和重采样中重新估计的步骤。

指出固定模型表观范围与内部校正开发统计量范围的区别；在校准图上辨认条件带和校正点估计；用反例验证无效输入确实被拒绝。

## 数据科学读者

先完成 4—7 的时间、标签和可得性审查，再学习 8、10—13、17。树与网络的方法实验见 25—26。工具箱示例：

```text
python -m pip install -r requirements-toolkit.txt
python run.py toolkit --mode quick
```

追踪每个训练折内的填补、尺度和类别；未知测试类别不能进入训练编码。学习曲线保留数值表，折间 SD 是描述性变异，不是置信区间。默认候选与扩展算法的覆盖范围分别见章节说明。

## 完整作业

为每章记录命令、产物、练习、问题与解释。先用 quick 调试，再运行完整预算：

```text
python run.py course --mode book
python run.py verify --mode book
python run.py math all --mode book
```

course 为第 1—20 章，math all 为 16 个数学脚本。报告包含方案、字典、模型/预测、图表、执行清单、环境、失败和未运行项、局限及人工状态。第 19—20 章生成材料不等于已投稿或已签审。

|状态或证据|含义|
|---|---|
|实际执行检查 PASS|通过该检查声明的范围|
|反例 PASS|指定错误被实际拒绝|
|STALE / RUNNING / FAIL|已过期、未完成或失败|
|partial / manual|已声明部分或读者进一步操作|
|人工 PENDING|尚未由真实人员确认|
|合成新队列评价|展示评价/更新步骤，不构成真实人群验证|

真实外部验证、临床影响评价与部署需要新的数据、方案和审核。
