"""Deterministic simulation and auditable model adapters."""

import hashlib
import json
import math
import random
import re

from . import OPTIONS
from .runner import ChatBackend


class SimulationBackend:
    """A seeded response simulator, not an LLM and not evidence for paper results.

    Preferences come from task/agent identifiers. Social exposure mixes in panel
    votes. Verification and control strengths change that stipulated mixture.
    It never receives the answer key. Hidden features are synthetic coordinates.
    """

    model = "illustrative-simulator-v1"
    temperature = None

    def _state(self, prompt):
        task = re.search(r"Task ID: ([^\n]+)", prompt)
        agent = re.search(r"Agent ID: ([^\n]+)", prompt)
        key = (task.group(1) if task else prompt.splitlines()[0]) + (agent.group(1) if agent else "agent-1")
        preferred = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 5
        preference = re.search(r"Initial preference: ([A-E])", prompt)
        if preference:
            preferred = OPTIONS.index(preference.group(1))
        base = [0.08 + math.exp(-1.5 * abs(k - preferred)) for k in range(5)]
        base = [p / sum(base) for p in base]
        match = re.search(r"Initial majority: ([A-E])", prompt)
        if match is None:
            match = re.search(r"4 chose ([A-E])", prompt)
        if match is None:
            match = re.search(r"panel of five chose: ([A-E])", prompt)
        social = 0.65 if match else 0.0
        if "Check each alternative" in prompt or "Independently check every allocation" in prompt:
            social *= 0.5
        control = re.search(r"Control strength: ([0-9]+(?:\.[0-9]+)?)", prompt)
        if control:
            social *= 0.85 * (1 - 0.55 * float(control.group(1)))
        panel = [0.0] * 5
        if match:
            panel[OPTIONS.index(match.group(1))] = 1.0
        return base, panel, social

    def hidden(self, prompt):
        base, panel, social = self._state(prompt)
        return base + [value * social for value in panel] + [1.0, float(social > 0)]

    def generate_steered(self, prompt, seed, delta):
        base, panel, social = self._state(prompt)
        if delta is not None:
            social = max(0.0, min(1.0, social + sum(delta[5:10])))
        weights = [(1 - social) * p + social * q for p, q in zip(base, panel)]
        return random.Random(seed).choices(OPTIONS, weights=weights, k=1)[0]

    def generate(self, prompt, seed):
        return self.generate_steered(prompt, seed, None)


class AuditBackend:
    """Persist each successful generation immediately, without logging credentials."""

    def __init__(self, backend, stream):
        self.backend = backend
        self.stream = stream
        self.model = backend.model
        self.temperature = getattr(backend, "temperature", None)
        self.count = 0

    def _record(self, prompt, seed, raw, delta=None):
        self.stream.write(json.dumps({"sample_id": self.count, "prompt": prompt, "seed": seed,
                                      "raw": raw, "steering_delta": delta}, ensure_ascii=False) + "\n")
        self.stream.flush()
        self.count += 1
        return raw

    def generate(self, prompt, seed):
        return self._record(prompt, seed, self.backend.generate(prompt, seed))

    def hidden(self, prompt):
        return self.backend.hidden(prompt)

    def generate_steered(self, prompt, seed, delta):
        return self._record(prompt, seed, self.backend.generate_steered(prompt, seed, delta), delta)


def make_backend(config, model=None, base_url=None):
    import os
    kind = config["backend"]
    if kind == "simulation":
        return SimulationBackend()
    model = model or os.environ.get("MAI_MODEL")
    if not model:
        raise ValueError("Set MAI_MODEL or pass --model")
    if kind == "chat":
        endpoint = base_url or os.environ.get("MAI_BASE_URL")
        if not endpoint:
            raise ValueError("Set MAI_BASE_URL or pass --base-url")
        return ChatBackend(endpoint, model, os.environ.get("MAI_API_KEY", ""),
                           send_seed=config.get("send_seed", True))
    if kind == "local":
        from .local import LocalBackend
        return LocalBackend(model, config.get("hidden", {}).get("layer", 0))
    raise ValueError(f"Unknown backend: {kind}")
