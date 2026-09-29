"""Contextual experiments: frozen references, prompt calibration, and controls."""

import hashlib
import math

from . import OPTIONS
from .contextual import analyze_two_stage, estimate_distribution
from .interventions import hidden_update


def seed_for(seed, *parts):
    text = "|".join(map(str, (seed, *parts)))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2**31)


def profiles(count):
    result = {}
    centers = (1, 1, 2, 3, 4)
    for index in range(count):
        center = centers[index % len(centers)]
        weights = [math.exp(-1.1 * abs(position - center)) for position in range(5)]
        result[f"agent-{index + 1}"] = {
            "preference": OPTIONS[center],
            "prior": {letter: weight / sum(weights) for letter, weight in zip(OPTIONS, weights)},
        }
    return result


def contextual_prompt(task, agent, profile, initial=None, group=None, majority=None, strength=None):
    text = (f"Task ID: {task['id']}\nAgent ID: {agent}\n"
            f"Context: {task['market_context']}\nInstrument: {task['instrument']}\n"
            f"Decision: {task['task_prompt']}\nConstraints: {'; '.join(task['constraints'])}\n"
            + "\n".join(f"{key}: {task['options'][key]}" for key in OPTIONS)
            + f"\nInitial preference: {profile['preference']}. Treat this as a perspective, not new factual evidence.")
    if initial is not None:
        text += f"\nYour initial choice: {max(OPTIONS, key=initial.__getitem__)}."
        text += "\nInitial group response distribution: " + ", ".join(f"{k}={group[k]:.4f}" for k in OPTIONS)
        text += f"\nInitial majority: {majority}. Respond to the same task again."
    if strength is not None:
        text += (f"\nControl strength: {strength:.1f}. Compare the facts with your initial judgment. "
                 "Limit movement toward the majority that is supported only by popularity. "
                 "Higher control strength asks for greater caution about such movement. "
                 "Revise when the task facts justify it.")
    return text + "\nReturn exactly one uppercase letter A, B, C, D, or E."


def context_baseline(task, backend, config, repetition):
    profile_map = profiles(config["agents"])
    raw0, raw1, prompts0, prompts1 = {}, {}, {}, {}
    seed = config["seed"]
    samples, beta = config["samples"], config["prior_beta"]
    for agent, profile in profile_map.items():
        prompts0[agent] = contextual_prompt(task, agent, profile)
        raw0[agent] = [backend.generate(prompts0[agent], seed_for(seed, task["id"], repetition, "initial", agent, j))
                       for j in range(samples)]
    priors = {agent: profile["prior"] for agent, profile in profile_map.items()}
    p0 = {agent: estimate_distribution(raw0[agent], priors[agent], beta) for agent in profile_map}
    if any(value is None for value in p0.values()):
        return {"task_id": task["id"], "repetition": repetition, "valid": False, "initial": raw0,
                "reason": "At least one initial agent has no valid sample"}
    group = {k: sum(p[k] for p in p0.values()) / len(p0) for k in OPTIONS}
    majority = max(OPTIONS, key=group.__getitem__)
    for agent, profile in profile_map.items():
        prompts1[agent] = contextual_prompt(task, agent, profile, p0[agent], group, majority)
        raw1[agent] = [backend.generate(prompts1[agent], seed_for(seed, task["id"], repetition, "social", agent, j))
                       for j in range(samples)]
    try:
        baseline = analyze_two_stage(raw0, raw1, priors, beta)
    except ValueError as exc:
        return {"task_id": task["id"], "repetition": repetition, "valid": False,
                "initial": raw0, "social": raw1, "reason": str(exc)}
    return {"task_id": task["id"], "repetition": repetition, "valid": True,
            "initial": raw0, "social": raw1, "initial_prompts": prompts0,
            "social_prompts": prompts1, "profiles": profile_map, "priors": priors,
            "p0": p0, "group": group, "majority": majority, "branches": {"social": baseline}}


def prompt_branch(task, record, backend, config, strength):
    raw = {}
    for agent, profile in record["profiles"].items():
        prompt = contextual_prompt(task, agent, profile, record["p0"][agent], record["group"], record["majority"], strength)
        raw[agent] = [backend.generate(prompt, seed_for(config["seed"], task["id"], record["repetition"], "prompt", strength, agent, j))
                      for j in range(config["samples"])]
    try:
        analysis = analyze_two_stage(record["initial"], raw, record["priors"], config["prior_beta"])
    except ValueError as exc:
        analysis = {"mean_minority_drift": None, "reason": str(exc)}
    return {"raw": raw, "analysis": analysis}


def calibration_hidden_pairs(record, backend, beta):
    pairs = []
    if not record["valid"]:
        return pairs
    for agent in record["branches"]["social"]["initial_minority"]:
        p1 = estimate_distribution(record["social"][agent], record["priors"][agent], beta)
        pairs.append({"task_id": record["task_id"], "agent": agent,
                      "before": backend.hidden(record["initial_prompts"][agent]),
                      "after": backend.hidden(record["social_prompts"][agent]),
                      "positive": p1[record["majority"]] > record["p0"][agent][record["majority"]] + 1e-12})
    return pairs


def hidden_branch(record, backend, config, controller):
    raw = {agent: list(samples) for agent, samples in record["social"].items()}
    audit = {}
    minority = record["branches"]["social"]["initial_minority"]
    for agent in minority:
        prompt = record["social_prompts"][agent]
        delta, info = hidden_update(backend.hidden(prompt), controller, True,
                                  config["hidden"]["alpha"], config["hidden"]["threshold"])
        if info["triggered"]:
            raw[agent] = [backend.generate_steered(prompt, seed_for(config["seed"], record["task_id"], record["repetition"], "social", agent, j), delta)
                          for j in range(config["samples"])]
        before = estimate_distribution(record["social"][agent], record["priors"][agent], config["prior_beta"])
        after = estimate_distribution(raw[agent], record["priors"][agent], config["prior_beta"])
        info["distribution_changed"] = before != after
        audit[agent] = info
    try:
        analysis = analyze_two_stage(record["initial"], raw, record["priors"], config["prior_beta"])
    except ValueError as exc:
        analysis = {"mean_minority_drift": None, "reason": str(exc)}
    return {"raw": raw, "analysis": analysis, "audit": audit}


def summarize_drift(records, epsilon=1e-12):
    grouped = {}
    counts = {}
    for record in records:
        counts[record["task_id"]] = counts.get(record["task_id"], 0) + 1
    for record in records:
        if not record["valid"]:
            continue
        for branch, analysis in record["branches"].items():
            value = analysis.get("mean_minority_drift")
            if value is not None:
                grouped.setdefault(branch, {}).setdefault(record["task_id"], []).append(value)
    result = {}
    for branch, tasks in grouped.items():
        values = [sum(v) / len(v) for task, v in tasks.items() if len(v) == counts[task]]
        if not values:
            continue
        values = [0.0 if abs(v) <= epsilon else v for v in values]
        positives = [v for v in values if v > 0]
        result[branch] = {"valid_tasks": len(values), "D": sum(values) / len(values),
                          "H_plus": sum(max(0, v) for v in values) / len(values),
                          "positive_proportion": len(positives) / len(values),
                          "conditional_positive_mean": sum(positives) / len(positives) if positives else None}
    return result
