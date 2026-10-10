# 临床预测模型开发：逐章实操项目

《临床预测模型开发实操手册》配套项目。读者可以生成合成数据、按章运行分析、检查结果，并修改练习。对应 2026-10-11 第二次修订版实操册，共 31 章。

主线 A：3200 例合成病例，14 个输入、18 个待估参数，预测手术开始后 30 天 CD ≥ III（包括 CD V 死亡）。第 9 章另设竞争风险 B 分支。第 21—31 章提供数值实验。数据全部由代码生成，没有真实患者资料。

## 从这里开始

推荐安装 Python 3.11 或 3.12 和 R ≥ 4.3（含推荐包 `survival`），然后：

```sh
git clone https://github.com/Harkool/clinical-prediction-handbook.git
cd clinical-prediction-handbook
python -m pip install -r requirements.txt
python run.py doctor
python run.py chapter 08 --mode quick
python run.py verify --mode quick
```

数学依赖只供第 21—31 章使用；前 20 章的入口只用 Python 标准库和 R 基础/推荐包。Windows 找不到 R 时，可在命令后加 `--rscript "本机 Rscript.exe 完整路径"`；安装步骤见 [安装指南](docs/installation.md)。

```sh
# 单章自动生成所需共享数据与模型，无须先运行前面章节
python run.py chapter 05
python run.py chapter 13

# 前 20 章全部运行；结果写 outputs/quick/
python run.py course --mode quick
python run.py verify --mode quick

# 指定章的数学实验；一个章节可能包含 a/b 两个脚本
python run.py math 24
python run.py math 24b
python run.py math all --mode quick

# 使用书中的完整重抽样预算，耗时明显增加
python run.py course --mode book
python run.py math all --mode book
```

## 按书学习

[31 章命令与练习对照](docs/chapter-guide.md) · [建议学习路线](docs/learning-path.md) · [数学实验说明](docs/math-labs.md) · [故障排查](docs/troubleshooting.md)

|目录|用途|
|---|---|
|`R/`|数据、模型、验证、校准、接口和 01—20 章课程|
|`math/`|21—31 章的 16 个数值实验及目录清单|
|`templates/`|分析计划、模型卡、报告、限制和人工审核模板|
|`examples/reference/`|原书完整预算的合成案例参考产物|
|`outputs/quick/`、`outputs/book/`|本次运行生成的数据、图表、日志；不进入 Git|
|`tests/`|实际课程运行、独立预测重算和反例检查|

每个 R 章节生成 `chapters/chNN/lesson.md`、章内产物和 `status.csv`；`manifest.csv` 记录已运行章节。模型系数和逐人预测保存于 `results/`。数学实验的日志位于 `outputs/<模式>/math/<实验编号>/stdout.txt`。

## 快速练习与完整计算

|重抽样预算|quick（默认）|book|
|---|---:|---:|
|开发 bootstrap|20|200|
|固定模型表观指标|40|500|
|校准曲线开发 / 条件重抽样|20 / 40|200 / 500|
|双重 bootstrap 外层 × 内层|10 × 10|200 × 50|
|描述性亚组重抽样|40|300|

两种模式分别保存，缓存包含算法、模式与预算。quick 用于练习与检查，有限预算结果不能冒充书中完整数值。模型与数据使用固定种子，平台和依赖版本仍可能影响模拟末位数字。

开发统计量的双重 bootstrap 范围、固定模型表观指标区间、表观平滑校准曲线的逐点条件带分别回答不同问题；它们都不等于真实外部性能区间。`verify` 用独立 Python 设计矩阵重算全部 3200 个预测、AUC 和 Brier，只验证导出一致性。

## 已交付的范围

第 7 章实际执行训练中位数填补演示，并检查缺失输入边界；完整 MICE 链需要另行实现与验证。第 10—11 章的主线比较采用 Logistic / ridge Logistic，并进行训练内调参；森林、boosting 和 TabPFN 未接入这条主线。数学第 25 章另有森林与 boosting 的可运行模拟。第 14 章独立合成数据用于验证和更新操作练习，没有真实临床迁移证据。第 9 章 B 分支提供 Fine–Gray 估计，没有完成其重抽样验证。

报告、AI 留痕与审核清单保留 `PENDING`；执行成功不表示已经获得临床审核、投稿或部署许可。请从产物和 [验证记录](docs/validation.md) 判断每一步实际做了什么。

## 检查与贡献

```sh
python run.py course --mode quick
python run.py verify --mode quick
Rscript --vanilla tests/test_course.R
python -m unittest discover -s tests -v
```

GitHub Actions 配置了 Windows、macOS、Linux 的快速课程检查，完整预算由维护者手动触发。参见 [贡献说明](CONTRIBUTING.md)。公开 issue 只附合成最小复现，勿上传真实患者记录、凭据或单位内部文件。

## 授权与引用

代码、仓库文档、模板及合成示例适用 [MIT](LICENSE)。书稿和 PDF 另行授权，见 [书籍授权边界](BOOK_LICENSE.md)。软件引用信息见 [CITATION.cff](CITATION.cff)。
