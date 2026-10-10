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
VERSION = "1.0.0"


def rscript_path(explicit=None):
    """Honor an explicit choice, then PATH, then standard Windows install dirs."""
    choice = explicit or os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if choice:
        resolved = shutil.which(choice) or choice
        if Path(resolved).is_file():
            return os.path.abspath(resolved)
        raise ValueError("Rscript 路径不存在: " + choice)
    if os.name == "nt":
        candidates = []
        for base in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "R",
                     Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "R"):
            candidates.extend(base.glob("R-*/bin/Rscript.exe"))
        if candidates:
            return str(sorted(candidates, key=lambda p: tuple(
                int(x) for x in p.parts[-3][2:].split(".") if x.isdigit()))[-1])
    raise ValueError("未找到 Rscript。安装 R 后加入 PATH，或传 --rscript 完整路径。")


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
    try:
        r = rscript_path(args.rscript)
        check = subprocess.run([r, "--vanilla", "-e",
            'cat(R.version.string,"\\n"); stopifnot(getRversion()>="4.3.0"); '
            'stopifnot(requireNamespace("survival",quietly=TRUE),'
            'requireNamespace("splines",quietly=TRUE)); cat("R baseline: PASS\\n")'],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        print("Rscript:", r)
        print(check.stdout.strip())
        if check.returncode:
            print(check.stderr.strip())
            ok = False
    except ValueError as exc:
        print(exc)
        ok = False
    for name in ("numpy", "scipy", "sklearn"):
        print("数学依赖 %s: %s" % (name, "已安装" if importlib.util.find_spec(name)
              else "缺少；python -m pip install -r requirements.txt"))
    print("输出:", ROOT / "outputs")
    return 0 if ok else 1


def run_r(chapter, args):
    out = output_dir(args.mode)
    r = rscript_path(args.rscript)
    execute([r, "--vanilla", str(ROOT / "R" / "lesson_cli.R"),
             "--chapter", str(chapter), "--output", str(out), "--mode", args.mode],
            ROOT, out / "logs" / ("course.log" if chapter == "all"
                                    else "chapter_%02d.log" % int(chapter)), args.mode)


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
    if len(set(ids)) != len(ids) or ids != [r["id"] for r in predictions]:
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
    return {"status": "PASS", "scope": "frozen case A coefficients/predictions/apparent metrics",
            "n": len(data), "events": events, "max_prediction_error": max(errors),
            "C": auc, "Brier": brier,
            "input_sha256": {path.relative_to(out).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in required},
            "note": "一致性验证不代表临床有效性；区间与门禁另见各章产物。"}


def parser():
    p = argparse.ArgumentParser(description="书籍逐章实操入口")
    subs = p.add_subparsers(dest="command", required=True)
    d = subs.add_parser("doctor", help="检查 R/Python 环境")
    d.add_argument("--rscript")
    for name in ("chapter", "course", "math", "verify"):
        sub = subs.add_parser(name)
        if name == "chapter":
            sub.add_argument("chapter", type=int, choices=range(1, 32))
        if name == "math":
            sub.add_argument("lab", help="21—31、24b 等，或 all")
        sub.add_argument("--mode", choices=("quick", "book"), default="quick")
        if name in ("chapter", "course"):
            sub.add_argument("--rscript", help="Rscript 的完整路径，也可设 RSCRIPT")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor(args)
        if args.command == "course":
            run_r("all", args)
        elif args.command == "chapter":
            if args.chapter <= 20:
                run_r(args.chapter, args)
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
