"""Portable reports and machine-readable output, with explicit provenance."""

import csv
import html
import json
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _format(value, percent=False):
    if value is None:
        return "undefined"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def export_report(directory, summary):
    directory = Path(directory)
    reliability = summary["reliability"]
    columns = ["condition", "accuracy", "wrong_consensus", "retention", "agent_abstention"]
    rows = []
    for condition in "ABCD":
        row = {"condition": condition}
        for metric in columns[1:]:
            row[metric] = reliability["metrics"][f"{condition}.{metric}"]["estimate"]
        rows.append(row)
    with (directory / "reliability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "contextual.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["branch", "valid_tasks", "D", "H_plus", "positive_proportion", "conditional_positive_mean"])
        writer.writeheader()
        writer.writerows({"branch": name, **values} for name, values in summary["contextual"].items())

    label = "SIMULATED WORKFLOW DEMONSTRATION" if summary["backend"] == "simulation" else "MODEL EVALUATION"
    intro = ("All responses and hidden features in this run are simulated. These outputs exercise the software; "
             "they are not the paper's experimental results." if summary["backend"] == "simulation" else
             "Results describe this run's model, tasks, prompts, and decoding settings. They are not a reproduction claim.")
    def table(headers, contents):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in contents) + "</tbody></table>"
    body = f"<p class='eyebrow'>{label}</p><h1>MAI evaluation report</h1><p class='notice'>{intro}</p>"
    body += f"<p>Backend: <strong>{html.escape(summary['model'])}</strong> · Seed {summary['seed']}</p>"
    body += "<h2>1. Task construction and calibration</h2>" + table(
        ["Stage", "Count"], [[key, value] for key, value in summary["construction"].items()])
    body += f"<p>Prompt strength selected on calibration tasks: <strong>{summary['prompt_calibration']['selected_strength']}</strong>. "
    body += "All contextual evaluation tasks are held out from calibration.</p>"
    body += table(["Candidate strength", "Round conditional positive means", "Selection score"],
                  [[row["strength"], ", ".join(_format(v) for v in row["round_conditional_positive_means"]), _format(row["score"])]
                   for row in summary["prompt_calibration"]["grid"]])
    body += "<h2>2. Distributional drift on evaluation tasks</h2>" + table(
        ["Branch", "Valid tasks", "Signed drift D", "Positive burden H+", "Positive proportion"],
        [[key, row["valid_tasks"], _format(row["D"]), _format(row["H_plus"]), _format(row["positive_proportion"], True)]
         for key, row in summary["contextual"].items()])
    body += "<p>D and H+ average repetitions within each task first. Post-processing shrinks drift by construction.</p>"
    body += table(["Baseline", "Intervention", "H+ gain", "95% paired scenario interval"],
                  [[row["baseline"], row["intervention"], _format(row["H_plus_gain"]),
                    " to ".join(_format(v) for v in row["ci95"])]
                   for row in summary["contextual_inference"].get("comparisons", {}).values()])
    body += "<p>Positive gain means lower H+. The prompt-zero comparison measures the incremental effect of selected strength.</p>"
    body += "<h2>3. Four-condition reliability</h2>" + table(
        ["Condition", "System accuracy", "Wrong consensus", "Correct retention", "Abstention"],
        [[row["condition"]] + [_format(row[key], True) for key in columns[1:]] for row in rows])
    effects = []
    for key in ("accuracy_loss", "direct_recovery", "general_verification", "specific_protection"):
        cell = reliability["metrics"][key]
        adjusted = reliability["tests"].get(key, {}).get("holm_two_primary")
        effects.append([key.replace("_", " "), _format(cell["estimate"], True).replace("%", " pp"),
                        " to ".join(_format(value, True).replace("%", " pp") for value in cell["ci95"]),
                        _format(adjusted) if adjusted is not None else "not a primary test"])
    body += "<h2>4. Paired effects and uncertainty</h2>" + table(["Effect", "Estimate (pp)", "95% task bootstrap interval", "Holm p (2 tests)"], effects)
    body += (f"<p>{reliability['tasks']} tasks; {reliability['bootstrap_replicates']} stratified bootstrap replicates. "
             "All conditions and repetitions travel together during resampling. Retention is recalculated as a ratio of counts. "
             "Two-sided sign-flip tests assume sign exchangeability under the null.</p>")
    body += "<h2>5. Hidden intervention audit</h2>" + table(["Measure", "Value"], list(summary["hidden_audit"].items()))
    body += "<h2>Artifacts</h2><p>" + " · ".join(
        f"<a href='{filename}'>{filename}</a>" for filename in
        ("summary.json", "construction.json", "contextual_records.jsonl", "reliability_records.jsonl", "samples.jsonl", "manifest.json")) + "</p>"
    style = """html{background:#eef2f6;color:#183046;font:16px/1.65 system-ui,sans-serif}body{max-width:1100px;margin:40px auto;padding:44px;background:white;border-radius:16px}h1{font-size:38px;line-height:1.2}h2{margin-top:38px;font-size:22px}.eyebrow{font-weight:700;letter-spacing:.12em;color:#477184;font-size:12px}.notice{border-left:4px solid #cc9331;background:#fff6e5;padding:16px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dde5eb}th{background:#eaf1f6}a{color:#125e9c}@media(max-width:800px){body{padding:20px;margin:10px}table{display:block;overflow:auto}h1{font-size:28px}}
"""
    (directory / "report.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>MAI evaluation report</title><style>" + style + "</style><body>" + body + "</body></html>", encoding="utf-8")
    markdown = f"# MAI evaluation report\n\n**{label}**\n\n{intro}\n\n"
    markdown += "| Condition | Accuracy | Wrong consensus | Retention | Abstention |\n|:--|--:|--:|--:|--:|\n"
    for row in rows:
        markdown += "| " + " | ".join([row["condition"]] + [_format(row[key], True) for key in columns[1:]]) + " |\n"
    markdown += "\nSee `report.html` and `summary.json` for paired effects, confidence intervals, calibration, and provenance.\n"
    (directory / "report.md").write_text(markdown, encoding="utf-8")
