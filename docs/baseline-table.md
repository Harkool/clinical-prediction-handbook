# 基线表 Table 1

`cph/baseline.py` 使用 Python 标准库，独立于模型拟合，可为 CSV 生成总体及分组基线表。主线默认声明有 14 个原始预测变量，不将 18 个设计参数当成基线变量。

```sh
python run.py baseline --input examples/reference/main_case.csv \
  --schema templates/table1_schema.json --output outputs/my-table1
```

默认模式输出到 `outputs/quick/baseline/`。未指定输入时需先生成本模式数据，例如运行第 3 章。自定义 JSON 的 `variables` 必须逐一声明 `name`、`type`；连续变量可选 `summary: mean_sd` 或 `median_iqr`，分类变量必填 `levels`，可声明 `label`、`unit`、`level_labels`。

`group`、`group_levels` 与 `id_column` 可选。ID 必须非缺失且唯一；分组缺失和未知分类水平会停止，避免静默改变人群。默认缺失标记为 `""`、`"NA"`、`"NaN"`，可在 `missing_values` 显式修改。非缺失的连续变量必须是有限数值，`Inf` 或无法解析的文本不变成缺失。

| 输出 | 含义 |
|---|---|
| `baseline_summary.csv` | 长表数值，含每组总人数、有效人数、缺失人数、均值/SD/分位数或水平人数/百分比 |
| `baseline_table.csv` | 可直接查看的宽表，UTF-8 BOM 便于 Excel 打开 |
| `table1.md` | Markdown 表 |
| `table1.html` | 独立 HTML，可在浏览器查看 |
| `table1.tex` | LaTeX 片段，目标文档需 booktabs；中文标签需 CJK 引擎 |
| `report.json` | RUNNING/PASS/FAIL 与实际人群、文件和范围 |

标准差用 n−1，分位数按 (n−1)p 线性插值，即 R type 7。分类百分比的分母为该变量非缺失人数，缺失比例分母为该组总人数。全缺失或仅一个观测时相应不可估计量显示 `NE`，零人数预设分类水平照常显示。

模块不自动填补，不执行显著性检验、变量筛选或因果分析。默认按结局分组是描述性展示；基线表不能代替输入时序和标签规则检查。多组或标签很长时，LaTeX 片段需在目标文档中选择适当字号、分页或横向布局。
