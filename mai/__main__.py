"""CLI for generating tasks, running a model, and analyzing saved responses."""

import argparse
import json
import os
from pathlib import Path

from .metrics import summarize
from .runner import ChatBackend, run_task
from .tasks import make_tasks
from .construction import build_tasks
from .pipeline import run_pipeline, validate_config
from .reporting import write_json
from .statistics import paired_analysis


def main():
    parser = argparse.ArgumentParser(description="MAI reference protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    pipeline = sub.add_parser("pipeline", help="run source construction through evaluation and HTML reporting")
    pipeline.add_argument("--config", default="configs/demo.json")
    pipeline.add_argument("--output", default="runs/demo")
    pipeline.add_argument("--model")
    pipeline.add_argument("--base-url")
    pipeline.add_argument("--dry-run", action="store_true", help="show task counts and maximum generation calls without running models")
    construct = sub.add_parser("construct", help="build and audit contextual tasks from JSON source materials")
    construct.add_argument("source")
    construct.add_argument("--output", default="construction.json")
    construct.add_argument("--semantic", action="store_true", help="use model-based screening, normalization, rewriting, rendering and checking")
    construct.add_argument("--base-url", default=os.environ.get("MAI_BASE_URL"))
    construct.add_argument("--model", default=os.environ.get("MAI_MODEL"))
    generate = sub.add_parser("generate", help="write exactly solvable teaching tasks")
    generate.add_argument("--output", default="tasks.json")
    generate.add_argument("--variants", type=int, default=1)
    run = sub.add_parser("run", help="evaluate tasks using a chat-completions endpoint")
    run.add_argument("--tasks", default="tasks.json")
    run.add_argument("--output", default="records.jsonl")
    run.add_argument("--base-url", default=os.environ.get("MAI_BASE_URL"))
    run.add_argument("--model", default=os.environ.get("MAI_MODEL"))
    run.add_argument("--agents", type=int, default=5)
    run.add_argument("--samples", type=int, default=3)
    run.add_argument("--repetitions", type=int, default=2)
    run.add_argument("--seed", type=int, default=42)
    analyze = sub.add_parser("analyze", help="summarize saved raw responses")
    analyze.add_argument("records")
    analyze.add_argument("--inference", action="store_true")
    analyze.add_argument("--bootstrap", type=int, default=1000)
    analyze.add_argument("--output")
    args = parser.parse_args()
    if args.command == "pipeline":
        if args.dry_run:
            config_path = Path(args.config).resolve()
            config = json.loads(config_path.read_text(encoding="utf-8"))
            validate_config(config)
            material_path = config_path.parent / config["materials"]
            accepted = len(build_tasks(json.loads(material_path.read_text(encoding="utf-8")))["accepted"])
            c, n, r, t = config["calibration_tasks"], config["agents"], config["samples"], config["repetitions"]
            contextual = c * config["calibration_rounds"] * n * r * (2 + len(config["prompt_strengths"]))
            contextual += max(0, accepted - c) * t * n * r * (4 + int(config["hidden"].get("enabled", False)))
            reliability = 15 * config["reliability_variants"] * t * n * config["reliability_samples"] * 5
            print(json.dumps({"backend": config["backend"], "accepted_contextual_tasks": accepted,
                              "max_generation_calls": contextual + reliability,
                              "note": "Hidden feature extraction adds forward passes; no model is called by dry-run."}, indent=2))
        else:
            result = run_pipeline(args.config, args.output, args.model, args.base_url)
            print(f"Completed {result['sample_calls']} generations. Open {Path(args.output) / 'report.html'}")
    elif args.command == "construct":
        backend = None
        if args.semantic:
            if not args.base_url or not args.model:
                parser.error("Semantic construction needs --base-url and --model or their MAI environment variables")
            backend = ChatBackend(args.base_url, args.model, os.environ.get("MAI_API_KEY", ""), max_tokens=1200)
        result = build_tasks(json.loads(Path(args.source).read_text(encoding="utf-8")), backend)
        write_json(args.output, result)
        print(json.dumps({key: len(result[key]) for key in ("accepted", "flagged", "rejected")}))
    elif args.command == "generate":
        tasks = make_tasks(args.variants)
        Path(args.output).write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {len(tasks)} illustrative tasks to {args.output}")
    elif args.command == "run":
        if not args.base_url or not args.model:
            parser.error("run requires --base-url and --model (or MAI_BASE_URL and MAI_MODEL)")
        tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
        backend = ChatBackend(args.base_url, args.model, os.environ.get("MAI_API_KEY", ""))
        with Path(args.output).open("w", encoding="utf-8") as stream:
            for index, task in enumerate(tasks):
                for record in run_task(task, backend, args.agents, args.samples,
                                       args.repetitions, args.seed + index):
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
        print(f"Wrote {len(tasks) * args.repetitions} task repetitions to {args.output}")
    else:
        records = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()]
        result = paired_analysis(records, bootstrap=args.bootstrap) if args.inference else summarize(records)
        if args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
