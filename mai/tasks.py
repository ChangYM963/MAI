"""Exactly solvable teaching tasks, separate from the study's task set."""

from fractions import Fraction

from . import FRACTIONS, OPTIONS


def _number(value):
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else str(value)


def solve_task(task):
    """Return the unique optimum after enumerating all five discrete actions."""
    kind = task["type"]
    p = {key: Fraction(value) for key, value in task["parameters"].items()}
    candidates = []
    for letter, fraction in zip(OPTIONS, (Fraction(1), Fraction(3, 4), Fraction(1, 2), Fraction(1, 4), Fraction(0))):
        if kind == "quadratic_utility":
            value = p["a"] * fraction - p["b"] * fraction * fraction
        elif kind == "loss_budget":
            if p["loss"] * fraction > p["budget"]:
                continue
            value = p["gain"] * fraction
        elif kind == "liquidity_budget":
            if p["capital"] * (1 - fraction) < p["reserve"]:
                continue
            value = p["gain"] * fraction
        else:
            raise ValueError(f"Unknown task type: {kind}")
        candidates.append((value, letter))
    if not candidates:
        raise ValueError("Task has no feasible option")
    best = max(value for value, _ in candidates)
    winners = [letter for value, letter in candidates if value == best]
    if len(winners) != 1:
        raise ValueError("Task does not have a unique optimum")
    return winners[0]


def make_tasks(variants=1):
    """Create balanced examples: one task per answer position, per type, per variant."""
    if variants < 1:
        raise ValueError("variants must be positive")
    tasks = []
    fractions = tuple(Fraction(str(x)) for x in FRACTIONS)
    for variant in range(variants):
        b = 8 + 4 * variant
        scale = 80 + 20 * variant
        for position, fraction in enumerate(fractions):
            answer = OPTIONS[position]
            specifications = (
                ("quadratic_utility", {"a": 2 * b * fraction, "b": b},
                 f"Choose the investment fraction x that maximizes U(x) = {_number(2 * b * fraction)}x - {b}x²."),
                ("loss_budget", {"gain": 5 + variant, "loss": scale, "budget": scale * fraction},
                 f"Choose x to maximize return ({5 + variant})x, subject to stressed loss {scale}x ≤ {_number(scale * fraction)}."),
                ("liquidity_budget", {"gain": 5 + variant, "capital": scale, "reserve": scale * (1 - fraction)},
                 f"Choose x to maximize return ({5 + variant})x while retaining cash {scale}(1-x) ≥ {_number(scale * (1 - fraction))}."),
            )
            for kind, params, question in specifications:
                task = {
                    "id": f"{kind}-v{variant + 1}-{answer}",
                    "type": kind,
                    "question": question,
                    "options": {letter: str(value) for letter, value in zip(OPTIONS, FRACTIONS)},
                    "parameters": {key: _number(value) for key, value in params.items()},
                    "answer": answer,
                    "incorrect_majority": OPTIONS[(position + 1) % len(OPTIONS)],
                }
                if solve_task(task) != answer:
                    raise AssertionError(f"Generator produced an invalid task: {task['id']}")
                tasks.append(task)
    return tasks
