#!/usr/bin/env python3
"""Portable, standard-library entry point for the handbook's R/Python labs."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(os.path.abspath(__file__)).parent
VERSION = "2.0.1"


def output_dir(mode):
    path = ROOT / "outputs" / mode
    path.mkdir(parents=True, exist_ok=True)
    return path


def execute(command, cwd, log, mode):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", CPH_MODE=mode, CPH_WORKERS="1")
    started = time.monotonic()
    print("运行:", " ".join(str(p) for p in command), flush=True)
    with log.open("w", encoding="utf-8", newline="\n") as stream:
        with subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace") as proc:
            for line in proc.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = proc.wait()
    if code:
        raise RuntimeError("运行失败 (exit %s)，日志: %s" % (code, log))
    print("完成 (%.1fs)，日志: %s" % (time.monotonic() - started, log), flush=True)


def doctor(args):
    print("Clinical Prediction Handbook", VERSION)
    print("Python:", sys.version.split()[0])
    ok = sys.version_info >= (3, 9)
    if not ok:
        print("需要 Python >= 3.9")
    for name in ("numpy", "scipy", "sklearn", "matplotlib"):
        installed = importlib.util.find_spec(name) is not None
        print("Python 依赖 %s: %s" % (name, "已安装" if installed else "缺少；python -m pip install -r requirements.txt"))
        ok = ok and installed
    print("输出:", ROOT / "outputs")
    return 0 if ok else 1


def run_course_cli(chapter, args):
    out = output_dir(args.mode)
    report = out / "course_execution.json"
    record = {"chapter": str(chapter), "mode": args.mode, "status": "RUNNING"}
    report.write_text(json.dumps(record)+"\n", encoding="utf-8")
    (out / "verification.json").write_text('{"status":"STALE","reason":"Python course started; rerun verify after completion"}\n', encoding="utf-8")
    try:
        execute([sys.executable, str(ROOT / "examples" / "course_cli.py"),
                 "--chapter", str(chapter), "--output", str(out), "--mode", args.mode],
                ROOT, out / "logs" / ("course.log" if chapter == "all"
                                        else "chapter_%02d.log" % int(chapter)), args.mode)
        record["status"] = "PASS_EXECUTED"
    except Exception as exc:
        record.update(status="FAIL", error=str(exc))
        raise
    finally:
        report.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def run_math(lab, args):
    catalog = json.loads((ROOT / "math" / "catalog.json").read_text(encoding="utf-8"))
    selected = catalog if lab == "all" else [entry for entry in catalog
               if entry["id"] == lab or (lab.isdigit() and entry["chapter"] == int(lab))]
    if not selected:
        raise ValueError("数学实验不存在: " + lab)
    for entry in selected:
        missing = [name for name in entry["dependencies"] if not importlib.util.find_spec(name)]
        if missing:
            raise ValueError("缺少数学依赖 %s；运行 python -m pip install -r requirements.txt" %
                             ", ".join(missing))
        out = output_dir(args.mode) / "math" / entry["id"]
        out.mkdir(parents=True, exist_ok=True)
        report = out / "result.json"
        report.write_text(json.dumps({"id": entry["id"], "mode": args.mode,
                                     "status": "RUNNING"})+"\n", encoding="utf-8")
        try:
            execute([sys.executable, str(ROOT / entry["script"])], out,
                    out / "stdout.txt", args.mode)
            result = json.loads(report.read_text(encoding="utf-8"))
            checks = result.get("checks")
            if (result.get("id") != entry["id"] or result.get("mode") != args.mode or
                    not isinstance(checks, dict) or not checks or
                    not all(value is True for value in checks.values())):
                raise ValueError("数学实验未产生本次模式的有效验收结果")
            result["status"] = "PASS"
            result["script_sha256"] = hashlib.sha256((ROOT / entry["script"]).read_bytes()).hexdigest()
            report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        except Exception as exc:
            try:
                failed = json.loads(report.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                failed = {"id": entry["id"], "mode": args.mode}
            failed.update(status="FAIL", error=str(exc))
            report.write_text(json.dumps(failed, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            raise


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def design(row):
    """Independent Python reconstruction of case A's frozen 19-column design."""
    inputs = {name: float(row[name]) for name in ("age", "bmi", "male", "asa",
              "emergency", "diabetes", "copd", "smoker", "chf", "sbp", "hr", "hb", "alb", "crea")}
    if not all(math.isfinite(value) for value in inputs.values()):
        raise ValueError("输入包含非有限值")
    if any(inputs[name] not in (0, 1) for name in ("male", "emergency", "diabetes", "copd", "smoker", "chf")):
        raise ValueError("二分类输入必须为 0/1")
    if inputs["asa"] not in (1, 2, 3, 4) or inputs["crea"] <= 0:
        raise ValueError("ASA 等级或肌酐输入无效")
    n = inputs.__getitem__
    age = n("age")
    cube = lambda z: max(z, 0.) ** 3
    basis = [(cube(age-k) - cube(age-65) * (85-k)/20 +
              cube(age-85) * (65-k)/20) / (65**2 * 10) for k in (20, 45)]
    asa = int(n("asa"))
    return [1., (age-50)/10] + basis + [(n("bmi")-25)/5, n("male"),
        float(asa == 2), float(asa == 3), float(asa == 4), n("emergency"),
        n("diabetes"), n("copd"), n("smoker"), n("chf"), (n("sbp")-120)/20,
        (n("hr")-75)/15, (n("hb")-130)/20, (n("alb")-40)/5, math.log(n("crea")/80)]


TERMS = ["(Intercept)", "age_linear", "age_rcs1", "age_rcs2", "bmi", "male",
         "asa_II", "asa_III", "asa_IV", "emergency", "diabetes", "copd", "smoker",
         "chf", "sbp", "hr", "hb", "alb", "log_crea"]


def verify_output(out):
    # Verify freshness separately from numerical agreement. Single-chapter
    # readers need not run all 20 lessons, but older evidence must not look new.
    execution = out / "course_execution.json"
    if execution.exists() and json.loads(execution.read_text(encoding="utf-8")).get("status") != "PASS_EXECUTED":
        raise ValueError("最近 Python 课程运行未成功完成；修复并重跑后再验证")
    manifest_path = out / "manifest.csv"
    source_path = out / "results/context_sources.csv"
    completed_chapters = []
    if manifest_path.exists():
        records = read_csv(manifest_path)
        completed_chapters = [int(row["chapter"]) for row in records]
        if (not records or len(set(completed_chapters)) != len(records) or
                any(chapter not in range(1, 21) for chapter in completed_chapters) or
                any(row["status"] != "PASS_EXECUTED" for row in records)):
            raise ValueError("章节记录含 STALE/FAIL/RUNNING 或非法编号；须重跑对应章节")
        if not source_path.exists():
            raise ValueError("旧产物缺少代码来源记录；请重跑已执行章节")
    if source_path.exists():
        sources = read_csv(source_path)
        expected = {"course_core.py", "course_validation.py", "course.py", "metrics.py", "preprocessing.py", "validation.py", "predictions.py", "baseline.py"}
        if len(sources) != len(expected) or {row["file"] for row in sources} != expected:
            raise ValueError("代码来源记录缺项或重复")
        if any(hashlib.sha256((ROOT/"cph"/row["file"]).read_bytes()).hexdigest() != row["sha256"] for row in sources):
            raise ValueError("Python 源码已改变，旧产物须重跑后再验证")
    required = [out / "data/main_case.csv", out / "results/model_coefficients.csv",
                out / "results/predictions.csv", out / "results/apparent_metrics.csv"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError("先运行 chapter 08 或 course；缺少: " + ", ".join(missing))
    data, coef, predictions = [read_csv(p) for p in required[:3]]
    if [r["term"] for r in coef] != TERMS:
        raise ValueError("模型系数的名称/顺序与冻结设计不符")
    beta = [float(r["coefficient"]) for r in coef]
    if not all(math.isfinite(b) for b in beta):
        raise ValueError("系数包含非有限值")
    if len(data) != 3200 or len(predictions) != len(data):
        raise ValueError("教学队列必须有 3200 行且预测行数相同")
    ids = [r["id"] for r in data]
    if any(not value.strip() for value in ids) or len(set(ids)) != len(ids) or ids != [r["id"] for r in predictions]:
        raise ValueError("患者 ID 重复或预测与输入的行顺序不一致")
    errors, y, p = [], [], []
    for row, saved in zip(data, predictions):
        eta = sum(x*b for x, b in zip(design(row), beta))
        value = 1 / (1 + math.exp(-eta))
        observed = int(row["y"])
        probability = float(saved["prediction"])
        if observed not in (0, 1) or observed != int(saved["y"]) or not 0 < probability < 1:
            raise ValueError("结局/预测不符合接口")
        errors.append(abs(value-probability))
        y.append(observed)
        p.append(probability)
    if max(errors) > 1e-10:
        raise ValueError("独立重算预测不一致: %.3g" % max(errors))
    brier = sum((a-b)**2 for a, b in zip(y, p)) / len(y)
    ordered = sorted(zip(p, y))
    negatives = concordance = 0.
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        positives = sum(v for _, v in ordered[i:j])
        zeros = j-i-positives
        concordance += positives * (negatives + zeros/2)
        negatives += zeros
        i = j
    events = sum(y)
    if not 0 < events < len(y):
        raise ValueError("评价需要同时有事件与非事件")
    auc = concordance / (events * (len(y)-events))
    metrics_file = out / "results/apparent_metrics.csv"
    if metrics_file.exists():
        metric_rows = read_csv(metrics_file)
        metrics = {r["metric"]: float(r["estimate"]) for r in metric_rows}
        if len(metrics) != len(metric_rows) or not {"C", "Brier"}.issubset(metrics):
            raise ValueError("指标名称重复或缺少 C/Brier")
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("导出指标包含非有限值")
        if abs(metrics["C"]-auc) > 1e-10 or abs(metrics["Brier"]-brier) > 1e-10:
            raise ValueError("导出的性能指标与独立重算不一致")
    return {"status": "PASS", "scope": "frozen case A predictions and independently recomputed apparent C/Brier",
            "completed_case_chapters": sorted(completed_chapters),
            "n": len(data), "events": events, "max_prediction_error": max(errors),
            "C": auc, "Brier": brier,
            "input_sha256": {path.relative_to(out).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in required},
            "note": "一致性验证不代表临床有效性；区间与门禁另见各章产物。"}


def parser():
    p = argparse.ArgumentParser(description="书籍逐章实操入口")
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="检查纯 Python 环境")
    baseline = subs.add_parser("baseline", help="生成可配置的基线表 Table 1（无需额外 Python 包）")
    baseline.add_argument("--mode", choices=("quick", "book"), default="quick")
    baseline.add_argument("--input", help="CSV 路径；默认使用本模式 main_case.csv")
    baseline.add_argument("--schema", help="JSON 变量与分组声明；默认 templates/table1_schema.json")
    baseline.add_argument("--output", help="输出目录；默认 outputs/<mode>/baseline")
    toolkit = subs.add_parser("toolkit", help="运行拆分后的 Python 分析工具箱示例")
    toolkit.add_argument("--mode", choices=("quick", "book"), default="quick")
    foundations = subs.add_parser("foundations", help="运行独立线性代数基础小例")
    foundations.add_argument("--mode", choices=("quick", "book"), default="quick")
    for name in ("chapter", "course", "math", "verify"):
        sub = subs.add_parser(name)
        if name == "chapter":
            sub.add_argument("chapter", type=int, choices=range(1, 32))
        if name == "math":
            sub.add_argument("lab", help="21—31、24b 等，或 all")
        sub.add_argument("--mode", choices=("quick", "book"), default="quick")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor(args)
        if args.command == "baseline":
            from cph.baseline import generate_baseline
            out = output_dir(args.mode)
            result = generate_baseline(Path(args.input) if args.input else out/"data/main_case.csv",
                         Path(args.schema) if args.schema else ROOT/"templates/table1_schema.json",
                         Path(args.output) if args.output else out/"baseline")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "foundations":
            from cph.foundations import run_foundations
            print(json.dumps(run_foundations(output_dir(args.mode)/"foundations"), indent=2))
        elif args.command == "toolkit":
            missing = [name for name in ("numpy", "scipy", "sklearn", "pandas", "matplotlib") if not importlib.util.find_spec(name)]
            if missing:
                raise ValueError("缺少工具箱依赖 %s；运行 python -m pip install -r requirements-toolkit.txt" % ", ".join(missing))
            execute([sys.executable, str(ROOT/"examples/toolkit_demo.py"), "--mode", args.mode],
                    ROOT, output_dir(args.mode)/"logs/toolkit.log", args.mode)
        elif args.command == "course":
            run_course_cli("all", args)
        elif args.command == "chapter":
            if args.chapter <= 20:
                run_course_cli(args.chapter, args)
            else:
                run_math(str(args.chapter), args)
        elif args.command == "math":
            run_math(args.lab, args)
        elif args.command == "verify":
            out = output_dir(args.mode)
            report = out / "verification.json"
            report.write_text('{"status":"RUNNING"}\n', encoding="utf-8")
            try:
                result = verify_output(out)
            except Exception as exc:
                report.write_text(json.dumps({"status": "FAIL", "mode": args.mode,
                                  "error": str(exc)}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
                raise
            result["mode"] = args.mode
            text = json.dumps(result, ensure_ascii=False, indent=2)
            report.write_text(text + "\n", encoding="utf-8")
            print(text)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, OverflowError, csv.Error) as exc:
        print("错误:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
