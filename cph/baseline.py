"""Schema-driven Table 1, with explicit observed denominators and missingness.

This module uses only the standard library. It neither imputes values nor tests
predictors against an outcome to choose a model. Every categorical level comes
from the declared schema, including levels with zero observations.
"""
import csv
import html
import json
import math
import statistics
from pathlib import Path


def quantile(values, probability):
    """Linear interpolation at (n-1)*p, matching R's default type 7."""
    ordered = sorted(values)
    index = (len(ordered)-1)*probability
    lower = int(index)
    upper = min(lower+1, len(ordered)-1)
    return ordered[lower]+(index-lower)*(ordered[upper]-ordered[lower])


def _schema(schema):
    if not isinstance(schema, dict) or not isinstance(schema.get("variables"), list) or not schema["variables"]:
        raise ValueError("Schema requires a nonempty variables list")
    names = []
    for variable in schema["variables"]:
        name = variable.get("name")
        if not isinstance(name, str) or not name or variable.get("type") not in ("continuous", "categorical"):
            raise ValueError("Each variable requires a name and continuous/categorical type")
        names.append(name)
        if variable["type"] == "categorical":
            levels = variable.get("levels")
            if not isinstance(levels, list) or not levels or len({str(x) for x in levels}) != len(levels):
                raise ValueError("Categorical levels must be nonempty and unique: " + name)
        elif variable.get("summary", "mean_sd") not in ("mean_sd", "median_iqr"):
            raise ValueError("Unknown continuous summary: " + name)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate variables in baseline schema")
    reserved = {schema.get("id_column"), schema.get("group")}-{None}
    if reserved.intersection(names):
        raise ValueError("ID/group columns must not also be summarized as predictors")
    return schema


def baseline_table(rows, schema):
    """Return numeric long rows and display wide rows without mutating inputs."""
    _schema(schema)
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Baseline input must contain rows of column-value mappings")
    variables = schema["variables"]
    group_name = schema.get("group")
    id_name = schema.get("id_column")
    required = [v["name"] for v in variables] + ([group_name] if group_name else []) + ([id_name] if id_name else [])
    if any(any(name not in row for name in required) for row in rows):
        raise ValueError("Missing required baseline columns")
    missing = {str(x) for x in schema.get("missing_values", ["", "NA", "NaN"])}

    def text(value):
        return "" if value is None else str(value).strip()

    if id_name:
        ids = [text(row[id_name]) for row in rows]
        if any(value in missing for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Patient IDs must be observed and unique")
    groups = []
    if group_name:
        observed_groups = [text(row[group_name]) for row in rows]
        if any(value in missing for value in observed_groups):
            raise ValueError("Group values are missing; declare/resolve the analysis population explicitly")
        groups = [str(value) for value in schema.get("group_levels", sorted(set(observed_groups)))]
        if not groups or len(set(groups)) != len(groups) or not set(observed_groups).issubset(groups):
            raise ValueError("Invalid declared group levels or unknown group values")
    subsets = [("Overall", rows)]+[("%s=%s" % (group_name, level),
               [row for row in rows if text(row[group_name]) == level]) for level in groups]
    long_rows, display_rows = [], []
    for variable in variables:
        name = variable["name"]
        label = variable.get("label", name)
        if variable.get("unit"):
            label += " (%s)" % variable["unit"]
        display = {"variable": label, "level": ""}
        cells = {}
        for group, subset in subsets:
            values = [text(row[name]) for row in subset]
            observed = [value for value in values if value not in missing]
            base = {"variable": name, "label": label, "group": group,
                    "n_total": len(subset), "n_observed": len(observed),
                    "n_missing": len(subset)-len(observed)}
            if variable["type"] == "continuous":
                try:
                    numbers = [float(value) for value in observed]
                except (TypeError, ValueError):
                    raise ValueError("Non-numeric continuous value: " + name) from None
                if not all(math.isfinite(value) for value in numbers):
                    raise ValueError("Non-finite continuous value: " + name)
                numeric = {"mean": statistics.mean(numbers) if numbers else None,
                           "sd": statistics.stdev(numbers) if len(numbers)>1 else None,
                           "median": quantile(numbers, .5) if numbers else None,
                           "q1": quantile(numbers, .25) if numbers else None,
                           "q3": quantile(numbers, .75) if numbers else None}
                long_rows.append(dict(base, type="continuous", level="", count="", percent="", **numeric))
                if not numbers:
                    cell = "NE"
                elif variable.get("summary", "mean_sd") == "median_iqr":
                    cell = "%.2f [%.2f, %.2f]" % (numeric["median"], numeric["q1"], numeric["q3"])
                else:
                    cell = "%.2f (%s)" % (numeric["mean"], "%.2f" % numeric["sd"] if numeric["sd"] is not None else "NE")
                cells[group] = cell+"; n=%d" % len(observed)
            else:
                levels = [str(level) for level in variable["levels"]]
                if not set(observed).issubset(levels):
                    raise ValueError("Unknown category for %s: %s" % (name, sorted(set(observed)-set(levels))))
                for level in levels:
                    count = observed.count(level)
                    percent = 100*count/len(observed) if observed else None
                    long_rows.append(dict(base, type="categorical", level=level, count=count,
                                          percent=percent, mean="", sd="", median="", q1="", q3=""))
        if variable["type"] == "continuous":
            display_rows.append(dict(display, **cells))
        else:
            for level in [str(value) for value in variable["levels"]]:
                level_label = variable.get("level_labels", {}).get(level, level)
                cells = {}
                for group, _ in subsets:
                    result = next(row for row in long_rows if row["variable"] == name and row["group"] == group and row["level"] == level)
                    cells[group] = "%d/%d (%s)" % (result["count"], result["n_observed"],
                                                 "%.1f%%" % result["percent"] if result["percent"] is not None else "NE")
                display_rows.append(dict(display, level=level_label, **cells))
        cells = {}
        for group, subset in subsets:
            count = sum(text(row[name]) in missing for row in subset)
            cells[group] = "%d/%d (%s)" % (count, len(subset), "%.1f%%" % (100*count/len(subset)) if subset else "NE")
        display_rows.append({"variable": label, "level": "Missing", **cells})
    return {"rows": long_rows, "display_rows": display_rows,
            "columns": ["variable", "level"]+[group for group, _ in subsets],
            "groups": {group: len(subset) for group, subset in subsets},
            "note": "Categorical percentages use observed denominators; missing percentages use total denominators. SD uses n-1; quantiles use linear interpolation (R type 7). NE means not estimable. Descriptive table only; no imputation, significance testing or predictor selection."}


def _latex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(character, character) for character in str(value))


def export_baseline(result, output):
    """Write numeric CSV, display CSV, Markdown, HTML and a LaTeX fragment."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    columns = result["columns"]
    for filename, rows, fields in (("baseline_summary.csv", result["rows"], list(result["rows"][0])),
                                    ("baseline_table.csv", result["display_rows"], columns)):
        with (output/filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    md = lambda value: str(value).replace("|", r"\|").replace("\n", " ")
    lines = ["# Baseline characteristics", "", " | ".join(md(c) for c in columns), " | ".join("---" for _ in columns)]
    lines += [" | ".join(md(row[c]) for c in columns) for row in result["display_rows"]]
    lines += ["", result["note"], "", "Group sizes: "+", ".join("%s=%s" % item for item in result["groups"].items())]
    (output/"table1.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    head = "".join("<th>%s</th>" % html.escape(c) for c in columns)
    body = "".join("<tr>"+"".join("<td>%s</td>" % html.escape(str(row[c])) for c in columns)+"</tr>" for row in result["display_rows"])
    (output/"table1.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Table 1</title><style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:.5rem;text-align:left}th{background:#eee}</style><h1>Baseline characteristics</h1><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table><p>'+html.escape(result["note"])+'</p></html>', encoding="utf-8")
    tex = [r"% Fragment: requires booktabs; Chinese labels require a CJK-capable engine.",
           r"% Wide group tables may require landscape or a smaller font in the target document.",
           r"\begin{tabular}{"+"l"*len(columns)+"}", r"\toprule",
           " & ".join(_latex(c) for c in columns)+r" \\", r"\midrule"]
    tex += [" & ".join(_latex(row[c]) for c in columns)+r" \\" for row in result["display_rows"]]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (output/"table1.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    return ["baseline_summary.csv", "baseline_table.csv", "table1.md", "table1.html", "table1.tex"]


def generate_baseline(input_path, schema_path, output):
    """Run with a status report that is invalidated before reading inputs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = output/"report.json"
    report.write_text('{"status":"RUNNING"}\n', encoding="utf-8")
    try:
        with Path(input_path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("Input has missing or duplicate column names")
            rows = list(reader)
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8-sig"))
        result = baseline_table(rows, schema)
        artifacts = export_baseline(result, output)
        status = {"status": "PASS", "n": len(rows), "groups": result["groups"],
                  "artifacts": artifacts, "note": result["note"]}
        report.write_text(json.dumps(status, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        return status
    except Exception as exc:
        report.write_text(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        raise
