# 安装与第一次运行

本项目配套《临床预测模型开发实操手册》。先用快速模式完成一次流程，再逐章读书、核对产物。书稿和 PDF 的授权独立于仓库；仓库代码及项目文档按 [MIT](../LICENSE) 使用。

## 准备什么

| 用途 | 所需环境 |
| --- | --- |
| 第 01—20 章课程 | Python 3.9 及以上；R 4.3 及以上，包含推荐包 `survival` 和 `splines` |
| 第 21—31 章数学实验 | Python 3.9—3.12，推荐 3.11 或 3.12；加 `requirements.txt` 中的 NumPy、SciPy、scikit-learn |
| 下载、更新代码 | Git；也可在 GitHub 的 Code 菜单下载 ZIP |

R 主线使用 base/recommended 包，不会自动联网安装包。原书主线实测 R 4.6.1，数学实验实测 Python 3.9.20、NumPy 2.0.2、SciPy 1.13.1、scikit-learn 1.6.1。新建数学环境建议使用 Python 3.11 或 3.12，并按仓库依赖文件安装。Python 3.13 及以后不在这组锁定数学依赖的支持范围内；若采用不同 Python 或包版本，应重新解决依赖、执行数值验收，并保留自己的版本记录。

从 [CRAN](https://cran.r-project.org/) 选择对应系统的 R 安装包。Python 安装与虚拟环境的依据见 [Python 安装指南](https://docs.python.org/3/installing/index.html) 和 [venv 文档](https://docs.python.org/3/library/venv.html)。虚拟环境属于本机环境，换机器时重新创建。

## 下载仓库

在你用于学习的目录打开终端：

```text
git clone https://github.com/Harkool/clinical-prediction-handbook.git
cd clinical-prediction-handbook
```

如果下载 ZIP，解压后进入含有 `run.py` 的目录。不要只下载一个章节文件，章节入口需要共用的模型、接口和课程模块。

## Windows：PowerShell

先确认 Python 能启动：

```powershell
python --version
python -m venv .venv
```

直接调用虚拟环境中的 Python，无需更改 PowerShell 执行策略：

```powershell
.\.venv\Scripts\python.exe run.py doctor
.\.venv\Scripts\python.exe run.py chapter 08 --mode quick
```

若提示找不到 Rscript，检查 R 的安装位置，并将实际路径传入。例如：

```powershell
.\.venv\Scripts\python.exe run.py chapter 08 --mode quick --rscript "C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
```

这个路径只是格式示例；以自己的安装目录为准。RStudio 可以帮助阅读 R 代码，但运行课程只需要 R 和 Rscript。

安装数学实验依赖后运行一个短实验：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py math 21
```

## macOS 与 Linux

安装适合系统的 Python 和 R 后，进入仓库根目录：

```bash
python3 --version
Rscript --version
python3 -m venv .venv
.venv/bin/python run.py doctor
.venv/bin/python run.py chapter 08 --mode quick
```

数学实验另外安装依赖：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py math 21
```

Linux 若 `venv` 不可用，按所用发行版的 Python 安装说明补齐虚拟环境组件；R 安装时保留推荐包。多套 R 并存时，可以用 `--rscript /实际路径/Rscript` 指定版本。

## 走完第一遍

下文中的 `python` 指你选定的同一个 Python；在 Windows 或 macOS/Linux，可继续用上面的虚拟环境完整命令替换：

```text
python run.py doctor
python run.py course --mode quick
python run.py verify --mode quick
```

`doctor` 报告环境和依赖。数学包缺失不影响只使用 base/recommended R 的章节；需要数学实验时先安装依赖。

课程将自动生成独立合成教学数据，随后保存模型、验证、图表及各章练习。共享产物位于 `outputs/quick/data/` 和 `outputs/quick/results/`，各章产物位于 `outputs/quick/chapters/chNN/`。`verify` 核对主线输入、系数和预测，独立重算 3,200 人风险及 C/Brier；其他章节检查和人工审核仍查看对应产物。

第一次运行结束后，按 [逐章指南](chapter-guide.md) 打开第 03、08、12、13、18 章产物。命令成功表示代码完成了其声明的检查；人工审核记录仍须由人填写。

## 两种运行模式

| 计算任务 | `quick` | `book` |
| --- | ---: | ---: |
| 开发流程 bootstrap 乐观校正 | 20 | 200 |
| 固定模型表观评价重抽样 | 40 | 500 |
| 平滑校准：开发 bootstrap | 20 | 200 |
| 平滑校准：固定预测 pairs 重抽样 | 40 | 500 |
| 重新开发并校正统计量的双重 bootstrap | 10 外层 × 10 内层 | 200 外层 × 50 内层 |

快速模式用于理解步骤、检查接口和调试，重复次数较少。书籍模式采用主线预算，输出到 `outputs/book/`；所需时间随设备、版本和拟合次数变化。

```text
python run.py course --mode book
python run.py verify --mode book
```

数学实验有各自的脚本预算。默认 `quick` 缩减重复模拟，确定性小例保持不变；每个实验的 `result.json` 记录实际预算。`--mode book` 保留书中实验预算，数学结果仍属于独立方法实验，不能混入主线 A 报告。

```text
python run.py math 24b
python run.py math 22 --mode book
python run.py math all --mode book
```

`all` 会运行全部数学实验；可先按章执行。数学日志与结果位于 `outputs/quick/math/` 或 `outputs/book/math/`。

## 保存自己的学习记录

`outputs/` 和 `.venv/` 不纳入 Git。若要比较两次运行，将完成的输出目录复制到本机另一学习目录，再进行下一次运行，并记录使用的 Git 提交、软件版本和命令。保留失败日志，修正原因后重跑。

公开 issue 或 PR 只附合成数据的最小复现。真实患者数据、访问凭据和含个人信息的日志保留在获准的本地环境中。
