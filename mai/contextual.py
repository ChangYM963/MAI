"""Two-stage contextual MAI analysis with an endogenous initial reference."""

from . import OPTIONS
from .metrics import mai, parse_option, shrink_distribution


def estimate_distribution(raw_responses, prior=None, beta=0.35):
    """Normalize valid counts, then mix with a fixed agent prior if supplied."""
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    valid = [parse_option(text) for text in raw_responses]
    valid = [letter for letter in valid if letter is not None]
    if not valid:
        return None
    empirical = {letter: valid.count(letter) / len(valid) for letter in OPTIONS}
    if prior is None:
        return empirical
    # mai validates keys, finiteness, nonnegativity, and normalization.
    mai(prior, "A")
    return {letter: (1 - beta) * empirical[letter] + beta * prior[letter] for letter in OPTIONS}


def analyze_two_stage(initial, subsequent, priors=None, beta=0.35, shrink=0.0):
    """Keep the initial majority and initial minority group fixed across stages."""
    if not initial or set(initial) != set(subsequent):
        raise ValueError("The same nonempty agent set is required in both stages")
    if priors is not None and set(priors) != set(initial):
        raise ValueError("A fixed prior is required for every agent")
    distributions = {}
    for agent in initial:
        prior = priors[agent] if priors is not None else None
        p0 = estimate_distribution(initial[agent], prior, beta)
        p1 = estimate_distribution(subsequent[agent], prior, beta)
        if p0 is None or p1 is None:
            raise ValueError(f"Agent {agent} has no valid samples in one stage")
        distributions[agent] = (p0, p1)
    mean_initial = {letter: sum(pair[0][letter] for pair in distributions.values()) / len(distributions)
                    for letter in OPTIONS}
    majority = max(OPTIONS, key=mean_initial.__getitem__)
    minority = []
    agents = {}
    for agent, (p0, uncorrected) in distributions.items():
        p1 = shrink_distribution(p0, uncorrected, shrink)
        choice0 = max(OPTIONS, key=p0.__getitem__)
        choice1 = max(OPTIONS, key=p1.__getitem__)
        if choice0 != majority:
            minority.append(agent)
        agents[agent] = {"initial_choice": choice0, "subsequent_choice": choice1,
                         "initial_mai": mai(p0, majority), "subsequent_mai": mai(p1, majority),
                         "delta_mai": mai(p1, majority) - mai(p0, majority)}
    drift = [agents[agent]["delta_mai"] for agent in minority]
    switches = sum(agents[agent]["subsequent_choice"] == majority for agent in minority)
    return {"initial_majority": majority, "initial_group_distribution": mean_initial,
            "initial_minority": minority,
            "mean_minority_drift": sum(drift) / len(drift) if drift else None,
            "minority_switch_rate": switches / len(minority) if minority else None,
            "agents": agents}
