"""One reproducible workflow exercising every split Codebase capability.

This intentionally simple linear-feature comparison is separate from the
handbook's frozen 18-parameter nonlinear model. Fixed holdout predictions are
paired for DeLong; OOF and learning-curve scores remain separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "book"), default="quick")
    args = parser.parse_args()
    output = ROOT/"outputs"/args.mode/"toolkit"
    output.mkdir(parents=True, exist_ok=True)
    report = output/"report.json"
    report.write_text('{"status":"RUNNING"}\n', encoding="utf-8")
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from cph.baseline import generate_baseline
        from cph.course_core import load_case, PREDICTORS, write_table
        from cph.metrics import evaluate, compare_models
        from cph.plots import plot_roc, plot_pr, plot_dca, plot_calibration, plot_confusion
        from cph.predictions import FrozenPredictions, positive_class_probability
        from cph.preprocessing import build_pipeline
        from cph.validation import cross_validate_model, learning_curve

        rows = load_case()
        names = list(PREDICTORS)
        categorical = ["male", "asa", "emergency", "diabetes", "copd", "smoker", "chf"]
        numeric = [name for name in names if name not in categorical]
        X = np.asarray([[row[name] for name in names] for row in rows], dtype=object)
        y = np.asarray([row["y"] for row in rows], dtype=int)
        ids = np.asarray([row["id"] for row in rows], dtype=object)
        train, test = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=20261011)
        if set(ids[train]).intersection(ids[test]):
            raise ValueError("Patient overlap in the fixed holdout")
        models, fitted = {}, {}
        for name, strength in (("Logistic_C100", 100.), ("Ridge_C0.1", .1)):
            model = build_pipeline(LogisticRegression(C=strength, max_iter=1500, random_state=20261011), numeric, categorical, names)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(X[train], y[train])
            p = positive_class_probability(model, X[test])
            models[name] = evaluate(y[test], p, ids[test])
            fitted[name] = model
        comparisons = compare_models(models, threshold=.15)
        (output/"comparison.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        write_table([{"id": ids[index], "y": int(y[index]), **{name: model.probability[i] for name, model in models.items()}} for i, index in enumerate(test)], output/"holdout_predictions.csv")
        write_table([{"id": ids[index], "split": "train" if index in set(train) else "test"} for index in range(len(y))], output/"split_assignments.csv")

        for name, function in (("roc", plot_roc), ("pr", plot_pr), ("dca", plot_dca), ("calibration", plot_calibration)):
            figure, axes, data = function(models)
            figure.savefig(output/(name+".png"), dpi=160, bbox_inches="tight")
            plt.close(figure)
        figure, axes, data = plot_confusion(models["Logistic_C100"], .15)
        figure.savefig(output/"confusion.png", dpi=160, bbox_inches="tight")
        plt.close(figure)

        pipeline = build_pipeline(LogisticRegression(C=1., max_iter=1500, random_state=20261011), numeric, categorical, names)
        cv = cross_validate_model(pipeline, X, y, ids=ids, n_splits=3 if args.mode=="quick" else 5, threshold=.15)
        write_table(cv["oof"], output/"oof_predictions.csv")
        write_table(cv["fold_metrics"], output/"cv_fold_metrics.csv")
        (output/"cv.json").write_text(json.dumps(cv, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        curve = learning_curve(pipeline, X, y, ids=ids, n_splits=3,
                               train_sizes=(.25,.5,1.) if args.mode=="quick" else (.25,.5,.75,1.), threshold=.15)
        write_table(curve["fold_metrics"], output/"learning_curve_folds.csv")
        write_table(curve["summary"], output/"learning_curve_summary.csv")
        figure, axis = plt.subplots()
        for role in ("train", "test"):
            selected = [row for row in curve["summary"] if row["metric"]=="brier" and row["split"]==role]
            axis.errorbar([row["train_size"] for row in selected], [row["mean"] for row in selected],
                          yerr=[row["sd"] for row in selected], marker="o", label=role)
        axis.set(xlabel="Fraction of outer training fold", ylabel="Brier score", title="Learning curve: mean and fold SD")
        axis.legend()
        figure.savefig(output/"learning_curve.png", dpi=160, bbox_inches="tight")
        plt.close(figure)

        saved = FrozenPredictions(ids[test], models["Logistic_C100"].probability, source="fixed synthetic holdout; independently fitted preprocessing and classifier")
        saved.save(output/"frozen_predictions.csv")
        restored = FrozenPredictions.load(output/"frozen_predictions.csv")
        if not np.array_equal(restored.predict(ids[test][::-1]), np.asarray(models["Logistic_C100"].probability)[::-1]):
            raise ValueError("Frozen probability patient mapping changed")
        generate_baseline(ROOT/"examples/reference/main_case.csv", ROOT/"templates/table1_schema.json", output/"baseline")
        status = {"status":"PASS", "mode":args.mode, "n":len(rows), "holdout_n":len(test),
                  "scope":"all split toolkit capabilities; fixed linear-feature model holdout comparison, separate OOF and learning curves; not frozen nonlinear book-model performance",
                  "threshold":.15, "seed":20261011, "split_method":"stratified 75% train / 25% test",
                  "learning_curve_band":"fold SD, not a confidence interval",
                  "input_sha256":hashlib.sha256((ROOT/"examples/reference/main_case.csv").read_bytes()).hexdigest(),
                  "artifacts":[path.relative_to(output).as_posix() for path in sorted(output.rglob("*")) if path.is_file() and path!=report]}
        report.write_text(json.dumps(status, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(status, indent=2))
    except Exception as error:
        report.write_text(json.dumps({"status":"FAIL", "error":str(error)}, indent=2)+"\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
