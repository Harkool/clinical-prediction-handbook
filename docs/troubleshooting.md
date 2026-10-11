# 故障排查

先保存命令、模式、Git 提交和错误消息。日志在 outputs/<mode>/logs，章内状态与范围见对应产物目录。

## 解释器或依赖缺失

```text
python --version
python -m pip install -r requirements.txt
python run.py doctor
```

安装和运行必须使用同一个 Python。Windows 可调用 .venv\Scripts\python.exe，macOS/Linux 使用 .venv/bin/python。推荐 3.11/3.12；锁定依赖覆盖 3.9—3.12。toolkit 另需 requirements-toolkit.txt。

## 下载后缺少文件

仓库应保留 run.py、cph、math、examples、templates 和 docs，尤其是 examples/reference/main_case.csv。单独下载一个脚本不能代替完整项目。

## 基线表找不到输入

默认表格读取当前模式的数据；先执行第 3 章或明确提供冻结输入：

```text
python run.py chapter 03 --mode quick
python run.py baseline --mode quick
python run.py baseline --input examples/reference/main_case.csv
```

未知分类水平、缺列或重复 ID 需要修正数据与 schema 的对应关系。观察分母和缺失数保留在数值 CSV。

## 跨目录或中文路径

用 run.py 的绝对路径调用入口。自备 --input、--schema、--output 相对参数仍按当前目录解释，跨目录时写绝对路径；含空格时加引号。源码和 CSV 使用 UTF-8；Excel 通过“从文本/CSV 导入”并明确选择 UTF-8。

## 运行慢或数字不同

先使用 quick。book 双重 bootstrap 含 200 × 50 次内部计算，数学 all 也含重复模拟。修改预算时记录配置和实际次数，不能只改报告文字。

冻结 3200 例上的固定设计、系数和预测应在明确容差内复现；新模拟和重抽样使用独立 Python 随机流，可能与历史参考不同。核对输入、版本、预算、种子和估计目标，勿混用 reference 与 outputs 的表格。

## verify、状态或身份失败

保留失败项，修正原因后重跑受影响章节。需要全部课程时运行：

```text
python run.py course --mode quick
python run.py verify --mode quick
```

STALE 表示已过期，RUNNING 表示未完成，FAIL 表示失败，均不能当作完成。单章产物有效与课程全覆盖分别核对。Python 模型用 SHA-256 绑定规则；修改结点、目标或实现后重新拟合并生成模型 JSON，不拷贝旧身份或手改 PASS。

## 不收敛、分离、奇异或某折只有一类

检查事件数、水平、设计秩和折分配。重复患者记录应声明 groups，确保同一患者全部在一侧；每个训练与评价折需要两类结局。拟合边界和 ConvergenceWarning 使相应运行失败，不删除失败重复后只平均成功值。

## 反例 PASS 与人工 PENDING

反例 PASS 表示指定的错误被实际拒绝，读取具体原因。人工待审状态由真实负责人核对后填写；自动运行不代签。partial/manual 的范围以章节说明为准。

## 报告问题

在 [GitHub issues](https://github.com/Harkool/clinical-prediction-handbook/issues) 提供章节、系统、Python/包版本、Git 提交、命令、最短错误及预期行为，附合成数据最小复现。仅提供定位所需日志，排除患者资料、凭据和个人信息。
