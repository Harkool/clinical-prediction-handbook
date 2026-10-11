# Codebase.py 拆分与 Python 工具箱

原文件 SHA-256：`73205be1731cda645716e9f00603c3a98d469865b8c4bc2a12afbc06e89cea2e`。新版按功能重写，避免单文件导入时强制加载 Keras、绘图窗口和全部分析依赖。原文件没有作为副本公开；新模块不保留旧类的全部参数签名。

| 原模块 / 功能 | 新实现 | 对应章节 |
|---|---|---|
| `PlotDCA` | `cph.metrics.decision_curve`、`cph.plots.plot_dca` | 13、27 |
| `KFoldValidation` | `cph.validation.cross_validate_model` | 10—12、28 |
| `Valuestandard` | `cph.preprocessing.build_preprocessor/build_pipeline` | 6、7、10、11 |
| `ConfusionMatrix` | `Evaluation`、`threshold_metrics`、`plot_confusion` | 13、27 |
| `PlotROC` | `roc_data`、`plot_roc` | 13、27 |
| `PlotPRC` | `pr_data`、`plot_pr` | 13、27 |
| `DelongTest` | `delong_compare` | 13、27 |
| `PlotLearningCurve` | `learning_curve` + 示例的学习曲线图 | 10—12、26 |
| `CompareModel` | `compare_models` | 10、13 |
| `PlotCC` | `calibration_summary`、`plot_calibration` | 13、27 |
| `NNCalibrationModel` | `FrozenPredictions`、`positive_class_probability` | 17 |
| `Plot_Learning_Curve` | 同一 `learning_curve` API，保存各折数值和图 | 10—12、26 |
| 新增 Table 1 | `cph.baseline` | 3、6、18 |

## 运行整合示例

```sh
python -m pip install -r requirements-toolkit.txt
python run.py toolkit --mode quick
```

`examples/toolkit_demo.py` 从冻结输入建立一个独立线性特征示例。75% 训练、25% 留出，两个预设惩罚强度在同一留出病例上比较；训练侧学习填补、缩放和编码，阈值在评价前固定为 0.15。它与教材含年龄非线性项的 18 参数主模型分开，不用其结果替代主线表。

`outputs/quick/toolkit/` 保存基线表、留出预测、患者分配、模型比较 JSON、ROC/PR/DCA/校准/混淆矩阵图、学习曲线图、OOF 预测及逐折指标。报告只在全部任务成功后标 `PASS`；失败写 `FAIL`。

## 复用 API

```python
from cph.metrics import evaluate, threshold_metrics, delong_compare
from cph.plots import plot_calibration

# 每个 probability 对应同一 id 与已观察二分类结局
first = evaluate(y, probability, ids)
summary = threshold_metrics(first, threshold=0.15)
figure, axes, numerical_results = plot_calibration(first)
figure.savefig("calibration.png", bbox_inches="tight")
```

概率必须有限且在 [0,1]，不能把任意 decision score 当作概率。患者 ID 非空且唯一；成对比较要求相同顺序与相同结局。任一模型病例集合或标签不同即拒绝，不自行拼接患者。单类标签不能估计 AUC；没有预测阳性时 PPV 等不可估计量保留原因/空值。

ROC/PR 以完整并列预测块计数。AP 与梯形 PR 面积分别报告。DeLong 为未加权成对固定预测比较，每类至少两人；方差退化时明确不可估计，不在分母加微小数制造 p 值。差值区间条件于已拟合模型，不是开发流程或临床因果的不确定性；多次比较的 p 值未自动校正。

## 修复的原实现问题

- 测试侧分类变量曾重新 `fit_transform`，导致编码与训练侧不一致。新版完整预处理器在每个训练折拟合，未知测试类别不参与学习。
- `CompareModel` 默认在评价集用 Youden/F1 选择阈值。新版只评价预先给定的阈值；决策曲线拒绝 0、1 或越界阈值。
- 原 DeLong 构造过程覆盖标签，仅留下最后一个模型结局；新版检查患者与标签并行对齐。
- 原校准模块对分箱点做线性回归并称校准截距/斜率。新版分箱仅为描述，CITL 与联合截距/斜率使用逐人 logit 预测且分别标注；分离、常数预测或边界概率明确不可估计。
- 原“伪模型”忽略传入病例，直接返回旧概率。新版 `FrozenPredictions.predict(ids)` 只按已存 ID 重排；新 ID 拒绝，CSV 与元数据摘要不符则拒绝读取。
- 原绘图会关闭外部图。新版返回自己的 figure 与数值，不自动显示或关闭读者其他图。
- CV 克隆完整 pipeline，按患者隔离；学习曲线训练子集嵌套于外层训练折，全部阶段重学预处理。未收敛警告导致失败，折间 SD 只作描述。

## 模型与神经网络

任何能被 sklearn 克隆且实现 `fit`、`predict_proba` 的分类器都可进入训练 pipeline，包括 `MLPClassifier`。概率读取按 `classes_` 查找正类 1，不按模型名称猜测正类列。显式神经网络概率数组可直接进入 `Evaluation` 或 `FrozenPredictions`；本项目不提供旧 Keras 路径加载参数的兼容层。

保存概率不是校准训练。若要重新校准模型，须把校准学习放在训练侧并在独立评价病例测试；不能在评价集拟合后宣称获得独立校准表现。
