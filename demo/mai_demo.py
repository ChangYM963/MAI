"""Run a clearly synthetic trace through the same analysis as saved model runs."""

import argparse
import json
from pathlib import Path

from mai.contextual import analyze_two_stage
from mai.metrics import empirical_distribution, mai, summarize
from mai.tasks import make_tasks


def example_record():
    task = next(task for task in make_tasks() if task["type"] == "quadratic_utility" and task["answer"] == "C")
    initial = {
        "agent-1": ["C", "C", "C"],
        "agent-2": ["C", "C", "D"],
        "agent-3": ["C", "C", "C"],
        "agent-4": ["D", "D", "D"],
        "agent-5": ["D", "D", "E"],
    }
    conditions = {
        "A": initial,
        "B": {"agent-1": ["D"] * 3, "agent-2": ["C", "D", "D"],
              "agent-3": ["C", "C", "D"], "agent-4": ["D"] * 3, "agent-5": ["D"] * 3},
        "C": {"agent-1": ["C"] * 3, "agent-2": ["C"] * 3,
              "agent-3": ["C"] * 3, "agent-4": ["D"] * 3, "agent-5": ["D"] * 3},
        "D": {"agent-1": ["C", "C", "D"], "agent-2": ["C", "C", "D"],
              "agent-3": ["C", "C", "D"], "agent-4": ["D"] * 3, "agent-5": ["D"] * 3},
    }
    return {"task_id": task["id"], "task_type": task["type"], "repetition": 1,
            "answer": task["answer"], "incorrect_majority": "D",
            "initial": initial, "conditions": conditions}


def main():
    parser = argparse.ArgumentParser(description="Inspect a synthetic MAI trace")
    parser.add_argument("--save-record", help="write the hand-written A-D trace as JSONL")
    args = parser.parse_args()
    contextual = analyze_two_stage(
        initial={"agent-1": ["C"] * 3, "agent-2": ["C"] * 3,
                 "agent-3": ["D"] * 3, "agent-4": ["D"] * 3, "agent-5": ["D"] * 3},
        subsequent={"agent-1": ["C", "C", "D"], "agent-2": ["C", "C", "D"],
                    "agent-3": ["D"] * 3, "agent-4": ["D"] * 3, "agent-5": ["D"] * 3},
    )
    print("Contextual two-stage example (synthetic)")
    print(f"Initial majority: {contextual['initial_majority']}; "
          f"mean initial-minority drift: {contextual['mean_minority_drift']:.2f}")
    print()
    record = example_record()
    if args.save_record:
        Path(args.save_record).write_text(json.dumps(record) + "\n", encoding="utf-8")
        print(f"Saved synthetic record to {args.save_record}")
    initial = empirical_distribution(record["initial"]["agent-1"])
    subsequent = empirical_distribution(record["conditions"]["D"]["agent-1"])
    reference = record["incorrect_majority"]
    print("Synthetic example (not study data)")
    print(f"Agent 1 keeps choice C; MAI relative to {reference}: "
          f"{mai(initial, reference):.2f} -> {mai(subsequent, reference):.2f}")
    print(json.dumps(summarize([record]), indent=2))


if __name__ == "__main__":
    main()
