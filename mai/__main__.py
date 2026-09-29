"""CLI for generating tasks, running a model, and analyzing saved responses."""

import argparse
import json
import os
from pathlib import Path

from .metrics import summarize
from .runner import ChatBackend, run_task
from .tasks import make_tasks


def main():
    parser = argparse.ArgumentParser(description="MAI reference protocol")
    sub = parser.add_subparsers(dest="command", required=True)
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
    args = parser.parse_args()
    if args.command == "generate":
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
        print(json.dumps(summarize(records), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
