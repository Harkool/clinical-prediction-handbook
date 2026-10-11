# 安装与第一次运行

v2 的课程、数学实验和工具箱全部使用 Python。项目地址：[Harkool/clinical-prediction-handbook](https://github.com/Harkool/clinical-prediction-handbook)。代码与项目文档采用 [MIT](../LICENSE)，书稿和 PDF 单独授权。

## 环境

推荐 Python 3.11 或 3.12；锁定依赖覆盖 Python 3.9—3.12。更高版本需重新解决依赖并验收数值。

|用途|安装内容|
|---|---|
|课程与数学实验|requirements.txt：NumPy 2.0.2、SciPy 1.13.1、scikit-learn 1.6.1、Matplotlib 3.9.4|
|基线表 baseline|仅 Python 标准库；可直接读取冻结 CSV|
|toolkit 扩展示例|requirements-toolkit.txt，另含 pandas 2.2.3；核心数组 API 不要求 pandas|

## 下载

```text
git clone https://github.com/Harkool/clinical-prediction-handbook.git
cd clinical-prediction-handbook
```

下载 ZIP 时保留完整目录结构，进入含 run.py 的目录。

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py doctor
.\.venv\Scripts\python.exe run.py chapter 08 --mode quick
```

直接调用虚拟环境解释器，无需修改 PowerShell 执行策略。

## macOS 与 Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py doctor
.venv/bin/python run.py chapter 08 --mode quick
```

若发行版未提供 venv，按其 Python 安装说明补齐。以下 python 均指同一个虚拟环境解释器。

## 完整课程与按章运行

```text
python run.py course --mode quick
python run.py verify --mode quick
python run.py chapter 24 --mode quick
python run.py math 24b --mode quick
python run.py math all --mode quick
```

chapter 接受 1—31；course 运行第 1—20 章，math all 运行第 21—31 章对应的 16 个脚本。共享数据和模型 JSON 在 outputs/quick/data 与 results，章内材料在 chapters/chNN，数学结果在 math。

主线读取 examples/reference/main_case.csv 的历史冻结合成输入：3200 例、384 事件，14 输入对应 18 个非截距参数。Python 重新拟合固定设计并核对逐人预测；新模拟与重抽样采用明确记录的 Python 随机流，不能据相同种子推定模拟数字与历史参考逐字相同。

verify 独立核对主线设计、模型身份与预测，重算 C/Brier并检查已有状态及来源。课程全覆盖另看章节清单，人工项目仍需读者完成。

## 基线表与工具箱

```text
python run.py chapter 03 --mode quick
python run.py baseline --mode quick
python run.py baseline --input examples/reference/main_case.csv
python -m pip install -r requirements-toolkit.txt
python run.py toolkit --mode quick
```

baseline 默认读取当前模式主线数据，也可直接读取冻结 CSV，独立生成时无需科学计算包。变量类型、水平、单位、分组和汇总方式由 templates/table1_schema.json 声明；用 --schema 和 --output 指定自己的配置与目录。默认结果在 outputs/quick/baseline，包含数值 CSV、显示表及 Markdown、HTML、LaTeX 片段。

## 快速与书籍预算

|计算|quick|book|
|---|---:|---:|
|开发 bootstrap|20|200|
|固定模型表观指标|40|500|
|校准开发 / 条件重抽样|20 / 40|200 / 500|
|双重 bootstrap 外层 × 内层|10 × 10|200 × 50|

```text
python run.py course --mode book
python run.py verify --mode book
python run.py math all --mode book
```

book 使用书中预算，输出到 outputs/book。快速重抽样结果不替代完整预算结果。

## 从其他目录运行

入口按自身位置寻找模块，默认输出仍在项目的 outputs。例如：

```powershell
& "D:\学习\clinical-prediction-handbook\.venv\Scripts\python.exe" "D:\学习\clinical-prediction-handbook\run.py" chapter 08 --mode quick
```

路径按本机修改。baseline 的 --input、--schema、--output 相对路径仍按当前终端目录解释，跨目录运行时写绝对路径。outputs 与虚拟环境不纳入 Git；保留命令、版本和失败日志。后续见 [逐章指南](chapter-guide.md) 与 [故障排查](troubleshooting.md)。
