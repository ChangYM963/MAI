"""Four-condition protocol with a simple OpenAI-compatible chat adapter."""

import json
import random
from urllib import error, request
from urllib.parse import urlsplit

from . import OPTIONS
from .metrics import agent_decision
from .tasks import solve_task


VERIFICATION = (
    "Check each alternative against the stated facts, objective, and constraints. "
    "Then retain or revise your initial choice."
)


def task_text(task):
    options = ", ".join(f"{letter}={task['options'][letter]}" for letter in OPTIONS)
    return f"{task['question']}\nAvailable investment fractions: {options}."


def initial_prompt(task, agent="agent-1"):
    return f"Task ID: {task['id']}\nAgent ID: {agent}\n" + task_text(task) + "\nChoose independently. Reply with exactly one uppercase letter A, B, C, D, or E."


def second_prompt(task, initial_choice, condition, agent="agent-1"):
    if condition not in tuple("ABCD"):
        raise ValueError("Unknown condition")
    message = f"Task ID: {task['id']}\nAgent ID: {agent}\n" + task_text(task)
    message += f"\nYour initial choice: {initial_choice or 'no unique valid choice'}."
    if condition in "BD":
        wrong = task["incorrect_majority"]
        correct = task["answer"]
        message += f"\nAn external panel of five chose: {wrong}, {wrong}, {wrong}, {wrong}, {correct}."
    if condition in "CD":
        message += "\n" + VERIFICATION
    return message + "\nRespond again with exactly one uppercase letter A, B, C, D, or E."


class ChatBackend:
    """One independent chat completion per sample; supply a trusted endpoint and key."""

    def __init__(self, base_url, model, api_key, temperature=0.7, timeout=60, max_tokens=16, send_seed=True):
        parts = urlsplit(base_url)
        if parts.scheme != "https" and not (parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1")):
            raise ValueError("Use HTTPS or a local HTTP endpoint")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Endpoint URLs must not contain embedded credentials, query parameters, or fragments")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.send_seed = send_seed

    def generate(self, prompt, seed):
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": 1,
            "max_tokens": self.max_tokens,
        }
        if self.send_seed:
            body["seed"] = seed
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = request.Request(self.url, data=payload, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                data = json.load(response)
        except error.HTTPError as exc:
            raise RuntimeError(f"Model endpoint returned HTTP {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Model endpoint connection failed: {exc.reason}") from exc
        content = data["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else ""


def run_task(task, backend, agents=5, samples=3, repetitions=2, seed=42):
    """Generate initial answers once per repetition, then branch A-D from them."""
    if solve_task(task) != task["answer"]:
        raise ValueError(f"Incorrect answer key for {task['id']}")
    if task["incorrect_majority"] == task["answer"]:
        raise ValueError("External panel majority must be incorrect")
    if agents < 1 or samples < 1 or repetitions < 1:
        raise ValueError("agents, samples, and repetitions must be positive")
    rng = random.Random(seed)
    ids = [f"agent-{i + 1}" for i in range(agents)]
    for repetition in range(1, repetitions + 1):
        initial = {agent: [backend.generate(initial_prompt(task, agent), rng.randrange(2**31))
                           for _ in range(samples)] for agent in ids}
        choices = {agent: agent_decision(initial[agent]) for agent in ids}
        conditions = {}
        order = list("ABCD")
        rng.shuffle(order)
        for condition in order:
            conditions[condition] = {
                agent: [backend.generate(second_prompt(task, choices[agent], condition, agent), rng.randrange(2**31))
                        for _ in range(samples)] for agent in ids
            }
        yield {
            "task_id": task["id"], "task_type": task["type"], "repetition": repetition,
            "answer": task["answer"], "incorrect_majority": task["incorrect_majority"],
            "initial": initial, "conditions": {condition: conditions[condition] for condition in "ABCD"},
            "condition_order": order,
            "generation": {"model": getattr(backend, "model", "custom"), "seed": seed,
                           "agents": agents, "samples": samples,
                           "temperature": getattr(backend, "temperature", None)},
        }
