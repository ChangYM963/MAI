"""Task-level paired inference; repetitions and all branches stay together."""

from collections import defaultdict
import math
import random

from .metrics import evaluate_record


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, previous = [None] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def sign_flip(values, seed=42, draws=10000):
    values = [v for v in values if abs(v) > 1e-12]
    if not values:
        return {"p": 1.0, "method": "all_zero", "nonzero_tasks": 0}
    observed = abs(sum(values))
    if len(values) <= 16:
        extreme = sum(abs(sum(v if mask & (1 << i) else -v for i, v in enumerate(values))) >= observed - 1e-12
                      for mask in range(2**len(values)))
        return {"p": extreme / 2**len(values), "method": "exact_two_sided", "nonzero_tasks": len(values)}
    rng = random.Random(seed)
    extreme = sum(abs(sum(v * rng.choice((-1, 1)) for v in values)) >= observed - 1e-12 for _ in range(draws))
    return {"p": (extreme + 1) / (draws + 1), "method": "monte_carlo_two_sided",
            "nonzero_tasks": len(values), "draws": draws}


def aggregate_tasks(records, expected_repetitions=None):
    grouped = defaultdict(list)
    seen = set()
    for record in records:
        key = (record["task_id"], record["repetition"])
        if key in seen:
            raise ValueError("Duplicate task/repetition record")
        seen.add(key)
        grouped[record["task_id"]].append(evaluate_record(record))
    if not grouped:
        raise ValueError("No completed records")
    expected = expected_repetitions or max(len(rows) for rows in grouped.values())
    if any(len(rows) != expected or {r["repetition"] for r in rows} != set(range(1, expected + 1))
           for rows in grouped.values()):
        raise ValueError("Incomplete task repetitions; do not silently drop failed branches")
    tasks = []
    for identifier, rows in sorted(grouped.items()):
        if len({row["task_type"] for row in rows}) != 1:
            raise ValueError("Task family changed between repetitions")
        task = {"id": identifier, "type": rows[0]["task_type"], "conditions": {},
                "initial_correct": sum(row["initial_correct"] for row in rows)}
        common_valid = all(row["conditions"][c]["minority_drift"] is not None for row in rows for c in "ABCD")
        task["common_valid_mai"] = common_valid
        for condition in "ABCD":
            cells = [row["conditions"][condition] for row in rows]
            task["conditions"][condition] = {
                "accuracy": sum(cell["correct"] for cell in cells) / len(cells),
                "wrong_consensus": sum(cell["wrong_consensus"] for cell in cells) / len(cells),
                "retained": sum(cell["retained_correct"] for cell in cells),
                "abstentions": sum(cell["agent_abstentions"] for cell in cells),
                "agent_count": sum(cell["agent_count"] for cell in cells),
                "drift": sum(cell["minority_drift"] for cell in cells) / len(cells) if common_valid else None,
            }
        tasks.append(task)
    return tasks


def _estimates(tasks):
    metrics = {}
    denominator = sum(task["initial_correct"] for task in tasks)
    for condition in "ABCD":
        cells = [task["conditions"][condition] for task in tasks]
        metrics[f"{condition}.accuracy"] = sum(cell["accuracy"] for cell in cells) / len(cells)
        metrics[f"{condition}.wrong_consensus"] = sum(cell["wrong_consensus"] for cell in cells) / len(cells)
        metrics[f"{condition}.retention"] = sum(cell["retained"] for cell in cells) / denominator if denominator else None
        metrics[f"{condition}.agent_abstention"] = sum(cell["abstentions"] for cell in cells) / sum(cell["agent_count"] for cell in cells)
    a, b, c, d = [metrics[f"{condition}.accuracy"] for condition in "ABCD"]
    metrics.update(accuracy_loss=a - b, direct_recovery=d - b, general_verification=c - a,
                   specific_protection=(d - b) - (c - a))
    return metrics


def paired_analysis(records, bootstrap=10000, flips=10000, seed=42, expected_repetitions=None):
    if bootstrap < 1 or flips < 1:
        raise ValueError("Resampling counts must be positive")
    tasks = aggregate_tasks(records, expected_repetitions)
    point = _estimates(tasks)
    strata = defaultdict(list)
    for task in tasks:
        strata[task["type"]].append(task)
    rng = random.Random(seed)
    samples = {key: [] for key in point}
    for _ in range(bootstrap):
        selected = [rng.choice(group) for group in strata.values() for _ in group]
        for key, value in _estimates(selected).items():
            if value is not None:
                samples[key].append(value)
    intervals = {key: {"estimate": value, "ci95": [percentile(samples[key], 0.025), percentile(samples[key], 0.975)],
                       "valid_bootstrap_replicates": len(samples[key])} for key, value in point.items()}
    primary = ("accuracy_loss", "specific_protection")
    tests = {key: sign_flip([_estimates([task])[key] for task in tasks], seed, flips) for key in primary}
    for key, adjusted in zip(primary, holm([tests[key]["p"] for key in primary])):
        tests[key]["holm_two_primary"] = adjusted
    valid_tasks = [task for task in tasks if task["common_valid_mai"]]
    drift = {condition: sum(task["conditions"][condition]["drift"] for task in valid_tasks) / len(valid_tasks)
             if valid_tasks else None for condition in "ABCD"}
    return {"tasks": len(tasks), "strata": {k: len(v) for k, v in strata.items()},
            "bootstrap_replicates": bootstrap, "unit": "task; paired branches and repetitions retained",
            "metrics": intervals, "tests": tests, "common_valid_mai_tasks": len(valid_tasks),
            "common_valid_mai": drift, "task_rows": tasks}


def drift_intervals(records, bootstrap=1000, seed=42):
    """Scenario-paired H+ intervals, retaining every repetition of eligible tasks."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["task_id"]].append(record)
    grouped = {task: rows for task, rows in grouped.items() if all(row["valid"] for row in rows)}
    if not grouped:
        return {"paired_tasks": 0, "comparisons": {}}
    branches = sorted(set.intersection(*(set(row["branches"]) for rows in grouped.values() for row in rows)))
    tasks = []
    for rows in grouped.values():
        if all(row["branches"][b].get("mean_minority_drift") is not None for row in rows for b in branches):
            tasks.append({b: sum(row["branches"][b]["mean_minority_drift"] for row in rows) / len(rows) for b in branches})
    if not tasks:
        return {"paired_tasks": 0, "comparisons": {}}
    rng = random.Random(seed)
    comparisons = {}
    for branch in branches:
        if branch == "social":
            continue
        values = [max(0, row["social"]) - max(0, row[branch]) for row in tasks]
        draws = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(bootstrap)]
        comparisons[branch] = {"H_plus_gain": sum(values) / len(values),
                               "baseline": "social", "intervention": branch,
                               "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)]}
    if "prompt_zero" in branches and "prompt_selected" in branches:
        values = [max(0, row["prompt_zero"]) - max(0, row["prompt_selected"]) for row in tasks]
        draws = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(bootstrap)]
        comparisons["prompt_incremental"] = {"baseline": "prompt_zero", "intervention": "prompt_selected",
                                             "H_plus_gain": sum(values) / len(values),
                                             "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)]}
    return {"paired_tasks": len(tasks), "comparisons": comparisons,
            "note": "Exploratory scenario-paired intervals, not the paper's full template sensitivity analysis"}
