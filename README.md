# 临床预测模型开发：Python 逐章实操项目

[English](README.en.md) · **中文**

《临床预测模型开发实操手册》配套项目，当前版本 **v2.0.1**。全部实现采用 Python，不需要 R。书稿可以独立阅读，操作所需概念、步骤理由、公式符号与结果判读均在书内说明；理论册在章末列为加深理解的选读。代码和具体命令集中在本项目；独立线性代数短章位于首次建模前。

初次学习建议先读实操册第 1—8、12—13、18 章，完成一次固定模型的开发、验证和报告，再补其他实操分支。第 21—31 章是按问题选读的原理实验。[学习路径](docs/learning-path.md) 将阅读、项目任务和完成标准连起来。

本书原有 **31 章均有真实运行入口**，另含 16 个原理实验。[逐章覆盖清单](docs/chapter-coverage.md) 列出实现、产物、缺项与人工工作；章号齐全不等于每章所有算法均已实现。

书中的独立线性代数基础短章有一个小型数值入口：`python run.py foundations --mode quick`，复算内积、矩阵秩、最小二乘残差正交和岭收缩，不改变原 31 章编号。

主线 A 是冻结的 3,200 例合成成年人，384 个事件、14 个原始预测变量、18 个非截距参数。结局是手术开始后 30 天 CD ≥ III，包含死亡；预测时点与结局计时原点分开。全部数据为合成资料。

## 开始运行

建议使用 Python 3.11 或 3.12；兼容环境最低为 Python 3.9。安装与系统环境说明见 [安装指南](docs/installation.md)。

```sh
git clone https://github.com/Harkool/clinical-prediction-handbook.git
cd clinical-prediction-handbook
python -m pip install -r requirements.txt
python run.py doctor
python run.py chapter 08 --mode quick
python run.py verify --mode quick
```

单章自动准备所需共享输入与模型。命令可以从任何工作目录调用，但需给出 `run.py` 的实际路径；相对 `--input`、`--schema`、`--output` 仍相对于当前工作目录。

```sh
python run.py chapter 05 --mode quick
python run.py chapter 24 --mode quick
python run.py course --mode quick       # 第 01—20 章
python run.py math all --mode quick     # 第 21—31 章的 16 个实验
python run.py verify --mode quick
```

`chapter 21`—`chapter 31` 会运行该章的所有补充实验。`math 24b` 可单独运行一个实验；`math 24` 会运行第 24 章的两个实验。

## 生成基线表 Table 1

```sh
python run.py chapter 03 --mode quick
python run.py baseline --mode quick
# 直接为已有 CSV 生成表；此模块无需额外 Python 包
python run.py baseline --input examples/reference/main_case.csv \
  --schema templates/table1_schema.json --output outputs/my-table1
```

[变量声明](templates/table1_schema.json) 明确变量、类型、单位、分类水平、分组及连续变量的汇总方式。默认 14 个原始变量，总体与结局分组分别报告均值（样本标准差）、中位数［四分位数］或人数/有效分母（百分比），缺失单独列出，零人数水平保留。

输出数值 CSV、展示 CSV、Markdown、HTML、LaTeX 片段与状态 JSON。分类百分比用非缺失分母，缺失比例用全组人数；不自动填补、不进行单因素变量筛选。单个观测的标准差标不可估计，不写成 0。[基线表说明](docs/baseline-table.md)

## 拆分后的分析工具箱

用户提供的 `Codebase.py` 的 12 项功能已按用途重写并拆分：预处理、交叉验证、学习曲线、混淆矩阵、ROC、PR、决策曲线、校准、成对 DeLong AUC 比较、模型比较和按病例 ID 存取概率。迁移映射及修正见 [工具箱指南](docs/python-toolkit.md)。旧类名与参数签名不作为新 API 保留。

```sh
python -m pip install -r requirements-toolkit.txt
python run.py toolkit --mode quick
```

示例保存基线表、固定留出预测和模型比较、6 张图、OOF 预测、折指标及学习曲线。预处理在训练折内学习，阈值预设为 0.15，评价病例 ID 必须一致。概率文件只能读取原病例，不能替代预测新病例的模型。学习曲线的折间标准差不冒称置信区间。

## 模式与来源

| 重抽样预算 | quick | book |
|---|---:|---:|
| 开发 bootstrap | 20 | 200 |
| 固定模型表观指标 | 40 | 500 |
| 校准曲线开发 / 条件重抽样 | 20 / 40 | 200 / 500 |
| 双重 bootstrap 外层 × 内层 | 10 × 10 | 200 × 50 |
| 描述性亚组 | 40 | 300 |

```sh
python run.py course --mode book
python run.py verify --mode book
python run.py math all --mode book
```

两种模式输出分别保存在 `outputs/quick/`、`outputs/book/`，缓存绑定实际输入、完整预测规则、源码、依赖、随机数方案和预算。来源变化后旧章状态为 `STALE`，失败重跑不会继续显示旧 `PASS`。`verify` 独立重算全部 3,200 个预测、表观 C 与 Brier，并检查已记录章节与源码来源；它不表示所有章节已经运行。

Python 主线使用与历史教材相同的冻结输入，重新拟合的系数/预测可与 [历史参考产物](examples/reference/README.md) 比较。新增模拟和重抽样使用 PCG64；相同预算不保证与旧版 R 随机流的区间逐位相同。开发流程范围、条件于固定模型的区间和校准曲线逐点带分别标注。[验证记录](docs/validation.md)

## 覆盖与检查

第 7 章有训练侧填补练习，完整 MICE 未实现。第 9 章为实际累计发生率和原因别 Cox 演示，**未实现 Fine–Gray 回归或其验证**。第 10—11 章实际比较 Logistic / ridge，并做内层调参；森林、boosting、TabPFN 未接入主线，数学篇另有部分模型模拟。第 14 章验证与更新基于独立合成资料。报告、投稿与人工审核仍保留 `PENDING`。

| 目录 | 用途 |
|---|---|
| `cph/` | Python 案例课程、验证、基线表及可复用工具 |
| `math/` | 第 21—31 章的 16 个原理实验 |
| `templates/` | 变量声明与人工材料模板 |
| `examples/reference/` | 冻结合成输入及有来源说明的历史数值参考 |
| `outputs/` | 当前结果、状态、日志和图，不进入 Git |
| `tests/` | 实际数值、输入反例、患者隔离和失败状态测试 |

```sh
python run.py course --mode quick
python run.py verify --mode quick
python -m unittest discover -s tests -v
```

GitHub Actions 检查 Windows、macOS、Linux。完整预算由维护者手动触发。[章节指南](docs/chapter-guide.md) · [学习路线](docs/learning-path.md) · [数学实验](docs/math-labs.md) · [故障排查](docs/troubleshooting.md) · [贡献](CONTRIBUTING.md)

代码、仓库文档、模板及合成示例采用 [MIT](LICENSE)。2026-10-12 修订的实操册正文与 PDF 采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)，见 [授权边界](BOOK_LICENSE.md)；理论册授权本次未变。引用软件时使用 [CITATION.cff](CITATION.cff) 并注明版本。
