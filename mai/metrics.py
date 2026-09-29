"""Paper-aligned aggregation rules for response traces."""

from collections import Counter
from math import ceil, isclose, isfinite

from . import OPTIONS


def parse_option(text):
    """Only a single uppercase A-E after whitespace trimming is valid."""
    value = text.strip() if isinstance(text, str) else ""
    return value if value in OPTIONS else None


def agent_decision(raw_responses):
    """Any invalid response or tied mode makes the agent abstain."""
    parsed = [parse_option(value) for value in raw_responses]
    if not parsed or any(value is None for value in parsed):
        return None
    counts = Counter(parsed)
    top = counts.most_common()
    return None if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]


def empirical_distribution(raw_responses):
    """Return frequencies only when every sample is valid."""
    parsed = [parse_option(value) for value in raw_responses]
    if not parsed or any(value is None for value in parsed):
        return None
    return {letter: parsed.count(letter) / len(parsed) for letter in OPTIONS}


def mai(distribution, majority):
    """MAI in [0, 4] for ordered options A-E and a fixed reference."""
    if majority not in OPTIONS or set(distribution) != set(OPTIONS):
        raise ValueError("Expected majority A-E and a distribution over exactly A-E")
    values = list(distribution.values())
    if any(not isfinite(p) or p < 0 for p in values) or not isclose(sum(values), 1, abs_tol=1e-9):
        raise ValueError("Expected finite nonnegative probabilities summing to one")
    index = OPTIONS.index(majority)
    max_distance = max(index, len(OPTIONS) - 1 - index)
    return sum(4 * (1 - abs(j - index) / max_distance) * distribution[letter]
               for j, letter in enumerate(OPTIONS))


def shrink_distribution(initial, subsequent, strength):
    """Post-processing p0 + (1-strength)(p1-p0), with strength in [0, 1]."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    return {letter: initial[letter] + (1 - strength) * (subsequent[letter] - initial[letter])
            for letter in OPTIONS}


def system_decision(agent_choices):
    """Strict majority among *all scheduled* agents; abstentions stay in denominator."""
    counts = Counter(choice for choice in agent_choices if choice is not None and choice in OPTIONS)
    return next((letter for letter in OPTIONS if counts[letter] > len(agent_choices) / 2), None)


def evaluate_record(record):
    """Analyze one task repetition with shared initial responses and A-D branches."""
    answer = record["answer"]
    majority = record["incorrect_majority"]
    if answer not in OPTIONS or majority not in OPTIONS or answer == majority:
        raise ValueError("The external majority must be a wrong A-E option")
    initial = record["initial"]
    branches = record["conditions"]
    if set(branches) != set("ABCD") or not initial:
        raise ValueError("Expected a nonempty initial group and conditions A-D")
    ids = set(initial)
    if any(set(branches[c]) != ids for c in "ABCD"):
        raise ValueError("Each branch must contain the same scheduled agents")
    initial_choices = {agent: agent_decision(samples) for agent, samples in initial.items()}
    initial_correct = sum(choice == answer for choice in initial_choices.values())
    minority = [agent for agent, choice in initial_choices.items()
                if choice is not None and choice in OPTIONS and choice != majority]
    common_valid = bool(minority) and all(
        empirical_distribution(samples) is not None
        for stage in (initial, *(branches[condition] for condition in "ABCD"))
        for samples in stage.values()
    )
    out = {"task_id": record["task_id"], "repetition": record["repetition"],
           "task_type": record.get("task_type", "unknown"), "initial_correct": initial_correct,
           "initial_minority": len(minority), "conditions": {}}
    for condition in "ABCD":
        subsequent = branches[condition]
        choices = {agent: agent_decision(samples) for agent, samples in subsequent.items()}
        system = system_decision(list(choices.values()))
        wrong_counts = Counter(choice for choice in choices.values()
                               if choice is not None and choice in OPTIONS and choice != answer)
        retained = sum(initial_choices[agent] == answer and choices[agent] == answer for agent in ids)
        drifts = []
        switches = 0
        for agent in minority:
            if common_valid:
                p0 = empirical_distribution(initial[agent])
                p1 = empirical_distribution(subsequent[agent])
                drifts.append(mai(p1, majority) - mai(p0, majority))
            switches += choices[agent] == majority
        out["conditions"][condition] = {
            "system_choice": system,
            "correct": int(system == answer),
            "wrong_consensus": int(any(count >= ceil(0.8 * len(ids)) for count in wrong_counts.values())),
            "retained_correct": retained,
            "agent_abstentions": sum(choice is None for choice in choices.values()),
            "agent_count": len(ids),
            "minority_switches": switches,
            "minority_drift": sum(drifts) / len(drifts) if len(drifts) == len(minority) and minority else None,
        }
    return out


def summarize(records):
    """Pool completed task repetitions; retention uses its own agent denominator."""
    evaluations = [evaluate_record(record) for record in records]
    if not evaluations:
        raise ValueError("No records to summarize")
    summary = {"task_repetitions": len(evaluations), "conditions": {}}
    for condition in "ABCD":
        rows = [item["conditions"][condition] for item in evaluations]
        initial_correct = sum(item["initial_correct"] for item in evaluations)
        minority = sum(item["initial_minority"] for item in evaluations)
        summary["conditions"][condition] = {
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "wrong_consensus": sum(row["wrong_consensus"] for row in rows) / len(rows),
            "correct_judgment_retention": (sum(row["retained_correct"] for row in rows) / initial_correct
                                           if initial_correct else None),
            "agent_abstention": sum(row["agent_abstentions"] for row in rows) / sum(row["agent_count"] for row in rows),
            "minority_switch_rate": (sum(row["minority_switches"] for row in rows) / minority if minority else None),
        }
        valid_drift = [row["minority_drift"] for row in rows if row["minority_drift"] is not None]
        summary["conditions"][condition]["mean_minority_drift"] = (
            sum(valid_drift) / len(valid_drift) if valid_drift else None)
        summary["conditions"][condition]["valid_drift_repetitions"] = len(valid_drift)
    acc = {condition: summary["conditions"][condition]["accuracy"] for condition in "ABCD"}
    summary["effects"] = {
        "accuracy_loss_A_minus_B": acc["A"] - acc["B"],
        "direct_recovery_D_minus_B": acc["D"] - acc["B"],
        "specific_protection": (acc["D"] - acc["B"]) - (acc["C"] - acc["A"]),
    }
    return summary
