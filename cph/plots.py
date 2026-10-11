"""Optional plots returning figures plus the exact numerical data they display.

Matplotlib is loaded only on a plotting call. Figures are owned by the caller;
these functions do not show plots, save files, close other figures, or change
global styles. All model overlays require matching ordered patient identities.
"""
from .metrics import (Evaluation, align_evaluations, roc_data, pr_data,
                      decision_curve, calibration_summary, threshold_metrics)


def _models(value):
    models = {"Model": value} if isinstance(value, Evaluation) else value
    return align_evaluations(models)


def _axes(ax=None):
    if ax is not None:
        return ax.figure, ax
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(7, 5), layout="constrained")


def plot_roc(evaluations, ax=None):
    models = _models(evaluations)
    results = {name: roc_data(evaluation) for name, evaluation in models.items()}
    fig, ax = _axes(ax)
    for name, result in results.items():
        ax.plot(result["fpr"], result["tpr"], label="%s (AUC %.3f)" % (name, result["auc"]))
    ax.plot([0, 1], [0, 1], "--", color="0.5", label="Reference")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate",
           title="ROC: fixed predictions")
    ax.legend(loc="lower right")
    return fig, ax, results


def plot_pr(evaluations, ax=None):
    models = _models(evaluations)
    results = {name: pr_data(evaluation) for name, evaluation in models.items()}
    fig, ax = _axes(ax)
    for name, result in results.items():
        ax.step(result["recall"], result["precision"], where="pre",
                label="%s (AP %.3f)" % (name, result["average_precision"]))
    prevalence = next(iter(results.values()))["event_rate"]
    ax.axhline(prevalence, linestyle="--", color="0.5", label="Event rate %.3f" % prevalence)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
           title="Precision-recall: AP is not trapezoidal PR area")
    ax.legend()
    return fig, ax, results


def plot_dca(evaluations, thresholds=None, ax=None):
    models = _models(evaluations)
    thresholds = [index/100 for index in range(1, 51)] if thresholds is None else thresholds
    results = {name: decision_curve(evaluation, thresholds) for name, evaluation in models.items()}
    fig, ax = _axes(ax)
    for name, rows in results.items():
        ax.plot([row["threshold"] for row in rows], [row["net_benefit"] for row in rows], label=name)
    rows = next(iter(results.values()))
    x = [row["threshold"] for row in rows]
    ax.plot(x, [row["treat_all"] for row in rows], "--", color="0.4", label="Treat all")
    ax.plot(x, [row["treat_none"] for row in rows], ":", color="0.2", label="Treat none")
    ax.set(xlabel="Prespecified threshold probability", ylabel="Net benefit",
           title="Decision curve: complete binary outcomes, fixed predictions")
    ax.legend()
    return fig, ax, results


def plot_calibration(evaluations, bins=10, confidence=.95, axes=None):
    models = _models(evaluations)
    results = {name: calibration_summary(evaluation, bins, confidence) for name, evaluation in models.items()}
    if axes is None:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    else:
        if len(axes) != 2 or axes[0].figure is not axes[1].figure:
            raise ValueError("Calibration requires two axes belonging to the same figure")
        fig = axes[0].figure
    curve, distribution = axes
    for name, result in results.items():
        rows = [row for row in result["bins"] if row["n"]]
        mean = [row["mean_prediction"] for row in rows]
        rate = [row["event_rate"] for row in rows]
        errors = [[max(0., row["event_rate"]-row["event_rate_interval"][0]) for row in rows],
                  [max(0., row["event_rate_interval"][1]-row["event_rate"]) for row in rows]]
        line = curve.errorbar(mean, rate, yerr=errors, marker="o", capsize=3,
                              label="%s (Brier %.3f)" % (name, result["Brier"]))
        edges = [row["lower"] for row in result["bins"]] + [1.]
        distribution.hist(models[name].probability, bins=edges, histtype="step",
                          color=line.lines[0].get_color(), label=name)
    curve.plot([0, 1], [0, 1], "--", color="0.5", label="Ideal")
    curve.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed event rate",
              title="Grouped apparent calibration with Wilson event-rate ranges")
    curve.legend()
    distribution.set(xlabel="Predicted probability", ylabel="Patients")
    return fig, axes, results


def plot_confusion(evaluation, threshold, ax=None):
    result = threshold_metrics(evaluation, threshold)
    values = [[result["TN"], result["FP"]], [result["FN"], result["TP"]]]
    fig, ax = _axes(ax)
    ax.imshow(values, cmap="Blues")
    maximum = max(max(row) for row in values)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(values[row][column]), ha="center", va="center",
                    color="white" if values[row][column] > maximum/2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"],
           xlabel="Predicted label", ylabel="Observed label",
           title="Confusion matrix: prespecified p >= %.3f" % threshold)
    return fig, ax, result
