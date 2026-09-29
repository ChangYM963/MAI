"""One command from source material to recorded experiments and a report."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random

from .backends import AuditBackend, make_backend
from .construction import build_tasks
from .contextual import analyze_two_stage
from .interventions import fit_hidden_controller
from .protocols import (calibration_hidden_pairs, context_baseline, hidden_branch,
                        prompt_branch, summarize_drift)
from .reporting import export_report, write_json, write_jsonl
from .runner import run_task
from .statistics import drift_intervals, paired_analysis
from .tasks import make_tasks


def validate_config(config):
    required = {"backend", "materials", "seed", "agents", "samples", "repetitions", "prior_beta",
                "calibration_tasks", "calibration_rounds", "prompt_strengths", "postprocess_lambda",
                "hidden", "reliability_variants", "reliability_samples", "bootstrap_replicates", "sign_flip_replicates"}
    if required - set(config):
        raise ValueError("Missing configuration fields: " + ", ".join(sorted(required - set(config))))
    if set(config) - required - {"send_seed"}:
        raise ValueError("Unknown configuration fields; credentials must come from environment variables")
    for key in ("agents", "samples", "repetitions", "calibration_tasks", "calibration_rounds",
                "reliability_variants", "reliability_samples", "bootstrap_replicates", "sign_flip_replicates"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("prior_beta", "postprocess_lambda"):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    strengths = config["prompt_strengths"]
    if not strengths or len(set(strengths)) != len(strengths) or any(not 0 <= x <= 1 for x in strengths):
        raise ValueError("Prompt strengths must be unique numbers in [0,1]")
    if config["hidden"].get("enabled") and config["backend"] == "chat":
        raise ValueError("Hidden-state access requires a local model or the explicit simulator")


def calibrate(tasks, backend, config):
    records, hidden_pairs = [], []
    strengths = config["prompt_strengths"]
    for repetition in range(1, config["calibration_rounds"] + 1):
        for task in tasks:
            record = context_baseline(task, backend, config, repetition)
            record["strength_runs"] = {}
            if record["valid"]:
                for strength in strengths:
                    record["strength_runs"][str(strength)] = prompt_branch(task, record, backend, config, strength)
                if config["hidden"].get("enabled"):
                    hidden_pairs.extend(calibration_hidden_pairs(record, backend, config["prior_beta"]))
            records.append(record)
    common_ids = {task["id"] for task in tasks}
    for record in records:
        if (not record["valid"] or any(record["strength_runs"].get(str(c), {}).get("analysis", {}).get("mean_minority_drift") is None for c in strengths)):
            common_ids.discard(record["task_id"])
    grid = []
    for strength in strengths:
        rounds = []
        for repetition in range(1, config["calibration_rounds"] + 1):
            values = [row["strength_runs"][str(strength)]["analysis"]["mean_minority_drift"]
                      for row in records if row["task_id"] in common_ids and row["repetition"] == repetition]
            positive = [value for value in values if value > 0]
            rounds.append(sum(positive) / len(positive) if positive else None)
        score = sum(rounds) / len(rounds) if all(value is not None for value in rounds) else None
        grid.append({"strength": strength, "round_conditional_positive_means": rounds, "score": score})
    candidates = [row for row in grid if row["score"] is not None]
    selected = min(candidates, key=lambda row: (row["score"], row["strength"]))["strength"] if candidates else None
    selection = {"selected_strength": selected, "grid": grid, "common_valid_tasks": sorted(common_ids),
                 "criterion": "mean of round-wise conditional mean positive drift; epsilon=0; lower strength breaks ties",
                 "status": "selected" if selected is not None else "no defined candidate; selected-prompt evaluation skipped"}
    controller = fit_hidden_controller(hidden_pairs) if config["hidden"].get("enabled") else {"available": False, "reason": "Disabled for this backend/configuration"}
    return records, selection, controller, hidden_pairs


def run_pipeline(config_path, output, model=None, base_url=None, progress=print):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    materials_path = (config_path.parent / config["materials"]).resolve()
    materials = json.loads(materials_path.read_text(encoding="utf-8"))
    output = Path(output)
    # A completed or interrupted run is never silently overwritten.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"status": "running"})
    try:
        progress("[1/6] Construct and validate contextual tasks")
        construction = build_tasks(materials)
        write_json(output / "construction.json", construction)
        accepted = list(construction["accepted"])
        random.Random(config["seed"]).shuffle(accepted)
        split = config["calibration_tasks"]
        if len(accepted) <= split:
            raise ValueError("Need accepted tasks for both calibration and held-out evaluation")
        calibration_tasks, evaluation_tasks = accepted[:split], accepted[split:]
        write_json(output / "contextual_tasks.json", {"calibration": calibration_tasks, "evaluation": evaluation_tasks})
        raw_backend = make_backend(config, model, base_url)
        with (output / "samples.jsonl").open("w", encoding="utf-8") as stream:
            backend = AuditBackend(raw_backend, stream)
            progress("[2/6] Calibrate prompt strengths and the hidden risk head")
            calibration_records, selection, controller, hidden_pairs = calibrate(calibration_tasks, backend, config)
            write_jsonl(output / "contextual_calibration.jsonl", calibration_records)
            write_json(output / "prompt_calibration.json", selection)
            write_json(output / "hidden_controller.json", controller)
            write_jsonl(output / "hidden_calibration.jsonl", hidden_pairs)
            progress("[3/6] Evaluate baseline, prompt, post-processing, and hidden controls")
            contextual_records = []
            for repetition in range(1, config["repetitions"] + 1):
                for task in evaluation_tasks:
                    record = context_baseline(task, backend, config, repetition)
                    if record["valid"]:
                        zero = prompt_branch(task, record, backend, config, 0.0)
                        record["prompt_zero_raw"] = zero["raw"]
                        record["branches"]["prompt_zero"] = zero["analysis"]
                        strength = selection["selected_strength"]
                        if strength is not None:
                            selected = zero if strength == 0 else prompt_branch(task, record, backend, config, strength)
                            record["prompt_selected_raw"] = selected["raw"]
                            record["branches"]["prompt_selected"] = selected["analysis"]
                        record["branches"]["postprocess"] = analyze_two_stage(
                            record["initial"], record["social"], record["priors"], config["prior_beta"], config["postprocess_lambda"])
                        if controller.get("available"):
                            hidden = hidden_branch(record, backend, config, controller)
                            record["hidden_raw"] = hidden["raw"]
                            record["hidden_audit"] = hidden["audit"]
                            record["branches"]["hidden"] = hidden["analysis"]
                    contextual_records.append(record)
            write_jsonl(output / "contextual_records.jsonl", contextual_records)
            progress("[4/6] Run shared-initial four-condition reliability tasks")
            reliability_tasks = make_tasks(config["reliability_variants"], seed=config["seed"])
            write_json(output / "reliability_tasks.json", reliability_tasks)
            random.Random(config["seed"] + 100).shuffle(reliability_tasks)
            reliability_records = []
            with (output / "reliability_records.jsonl").open("w", encoding="utf-8") as records_stream:
                for index, task in enumerate(reliability_tasks):
                    for record in run_task(task, backend, config["agents"], config["reliability_samples"],
                                           config["repetitions"], config["seed"] + 1000 + index):
                        reliability_records.append(record)
                        records_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records_stream.flush()
        progress("[5/6] Compute paired bootstrap intervals, sign-flip tests, and Holm correction")
        reliability = paired_analysis(reliability_records, config["bootstrap_replicates"], config["sign_flip_replicates"],
                                      config["seed"], config["repetitions"])
        audits = [audit for row in contextual_records for audit in row.get("hidden_audit", {}).values()]
        summary = {
            "schema": "mai_run_v2", "backend": config["backend"], "model": raw_backend.model, "seed": config["seed"],
            "provenance": "simulated responses and hidden features" if config["backend"] == "simulation" else "model-generated responses on reference tasks",
            "sample_calls": backend.count,
            "construction": {"sources": len(materials), **{k: len(construction[k]) for k in ("accepted", "flagged", "rejected")},
                             "calibration_tasks": len(calibration_tasks), "evaluation_tasks": len(evaluation_tasks)},
            "prompt_calibration": selection,
            "contextual": summarize_drift(contextual_records),
            "contextual_inference": drift_intervals(contextual_records, config["bootstrap_replicates"], config["seed"]),
            "hidden_audit": {"available": controller.get("available", False), "minority_observations": len(audits),
                             "triggered": sum(row["triggered"] for row in audits),
                             "changed_after_trigger": sum(row["triggered"] and row["distribution_changed"] for row in audits)},
            "reliability": reliability,
        }
        write_json(output / "summary.json", summary)
        progress("[6/6] Export HTML, Markdown, CSV, and an artifact manifest")
        export_report(output, summary)
        write_json(output / "status.json", {"status": "complete"})
        manifest = {"schema": "mai_artifacts_v1", "software_version": "0.2.0", "python": platform.python_version(),
                    "created_utc": datetime.now(timezone.utc).isoformat(), "backend": config["backend"],
                    "materials_sha256": hashlib.sha256(materials_path.read_bytes()).hexdigest(),
                    "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}}
        write_json(output / "manifest.json", manifest)
        return summary
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)})
        raise
