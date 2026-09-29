"""Exactly solvable teaching tasks, separate from the study's task set."""

from fractions import Fraction
import random

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


def integer_oracle(task):
    """Independent integer check using x=k/4, with a common factor of 16."""
    p = {key: int(value) for key, value in task["parameters"].items()}
    scores = {}
    for letter, k in zip(OPTIONS, (4, 3, 2, 1, 0)):
        if task["type"] == "quadratic_utility":
            scores[letter] = 4 * p["a"] * k - p["b"] * k * k
        elif task["type"] == "loss_budget":
            if p["loss"] * k <= 4 * p["budget"]:
                scores[letter] = p["gain"] * k
        elif task["type"] == "liquidity_budget":
            if p["capital"] * (4 - k) >= 4 * p["reserve"]:
                scores[letter] = p["gain"] * k
        else:
            raise ValueError("Unknown task type")
    best = max(scores.values())
    winners = [letter for letter, score in scores.items() if score == best]
    if len(winners) != 1:
        raise ValueError("Integer oracle found a tie")
    return winners[0]


def make_tasks(variants=1, seed=42):
    """Balanced random cases; discard ties and duplicate type/parameter pairs.

    Every variant adds five tasks of each family. This implements the paper's
    task families, but does not claim to recreate its original 60 task records.
    """
    if variants < 1:
        raise ValueError("variants must be positive")
    rng, tasks, used = random.Random(seed), [], set()
    for kind in ("quadratic_utility", "loss_budget", "liquidity_budget"):
        for index in range(5 * variants):
            answer = OPTIONS[index % 5]
            for _ in range(20000):
                if kind == "quadratic_utility":
                    params = {"a": rng.randint(-25, 600), "b": rng.randint(20, 400)}
                    question = f"Choose x to maximize stipulated utility U(x) = {params['a']}x - {params['b']}x²."
                elif kind == "loss_budget":
                    params = {"gain": rng.randint(5, 90), "loss": rng.randint(30, 400), "budget": rng.randint(0, 420)}
                    question = f"Choose x to maximize payoff {params['gain']}x, subject to stress loss {params['loss']}x ≤ {params['budget']}."
                else:
                    capital = rng.randint(40, 400)
                    params = {"gain": rng.randint(5, 90), "capital": capital, "reserve": rng.randint(0, capital)}
                    question = f"Choose x to maximize payoff {params['gain']}x while retaining cash {capital}(1-x) ≥ {params['reserve']}."
                fingerprint = (kind, tuple(sorted(params.items())))
                task = {"type": kind, "parameters": params}
                try:
                    optimum = solve_task(task)
                except ValueError:
                    continue
                if optimum != answer or fingerprint in used:
                    continue
                if integer_oracle(task) != answer:
                    raise AssertionError("Independent oracles disagree")
                used.add(fingerprint)
                task.update(id=f"{kind}-{index + 1:03d}", question=question,
                            options={letter: str(value) for letter, value in zip(OPTIONS, FRACTIONS)},
                            answer=answer, incorrect_majority=rng.choice([letter for letter in OPTIONS if letter != answer]),
                            provenance="generated reference task", generator_seed=seed)
                tasks.append(task)
                break
            else:
                raise ValueError("Could not generate a balanced unique task")
    return tasks
