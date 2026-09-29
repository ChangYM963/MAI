"""Tests of substantive protocol rules, without calling a model endpoint."""

import unittest

from demo.mai_demo import example_record
from mai.contextual import analyze_two_stage
from mai.metrics import (agent_decision, evaluate_record, mai, shrink_distribution,
                         summarize, system_decision)
from mai.runner import run_task, second_prompt
from mai.tasks import make_tasks, solve_task


class TestReference(unittest.TestCase):
    def test_generated_tasks_are_balanced_and_exact(self):
        tasks = make_tasks(2)
        self.assertEqual(len(tasks), 30)
        for kind in {task["type"] for task in tasks}:
            self.assertEqual(sorted(task["answer"] for task in tasks if task["type"] == kind), list("AABBCCDDEE"))
        self.assertTrue(all(solve_task(task) == task["answer"] for task in tasks))
        self.assertTrue(all(task["incorrect_majority"] != task["answer"] for task in tasks))

    def test_strict_parsing_and_system_majority(self):
        self.assertEqual(agent_decision(["C", "C", "D"]), "C")
        self.assertIsNone(agent_decision(["C", "D"]))
        self.assertIsNone(agent_decision(["C", "c", "C"]))
        self.assertIsNone(system_decision(["A", "A", "B", None, None]))
        self.assertEqual(system_decision(["A", "A", "A", None, None]), "A")

    def test_fixed_reference_and_shrink_identity(self):
        initial = {"A": 0, "B": 0, "C": 1, "D": 0, "E": 0}
        subsequent = {"A": 0, "B": 0, "C": 2 / 3, "D": 1 / 3, "E": 0}
        raw = mai(subsequent, "D") - mai(initial, "D")
        corrected = shrink_distribution(initial, subsequent, 0.6)
        self.assertAlmostEqual(mai(corrected, "D") - mai(initial, "D"), 0.4 * raw)
        result = analyze_two_stage(
            {"one": ["C"] * 3, "two": ["D"] * 3, "three": ["D"] * 3},
            {"one": ["C", "C", "D"], "two": ["D"] * 3, "three": ["D"] * 3},
        )
        self.assertEqual(result["initial_majority"], "D")
        self.assertEqual(result["initial_minority"], ["one"])
        self.assertGreater(result["mean_minority_drift"], 0)
        self.assertEqual(result["minority_switch_rate"], 0)

    def test_four_conditions_and_common_valid_filter(self):
        record = example_record()
        evaluated = evaluate_record(record)
        self.assertEqual([evaluated["conditions"][c]["correct"] for c in "ABCD"], [1, 0, 1, 1])
        self.assertEqual(evaluated["conditions"]["B"]["wrong_consensus"], 1)
        self.assertEqual(evaluated["conditions"]["B"]["retained_correct"], 1)
        self.assertEqual(summarize([record])["effects"]["accuracy_loss_A_minus_B"], 1)
        record["conditions"]["C"]["agent-1"] = ["C", "invalid", "C"]
        self.assertTrue(all(evaluate_record(record)["conditions"][c]["minority_drift"] is None for c in "ABCD"))

    def test_runner_reuses_initial_choices_without_leaking_answer(self):
        class FakeBackend:
            model = "fake"
            temperature = 0.7

            def generate(self, prompt, seed):
                return "C"

        task = next(task for task in make_tasks() if task["answer"] == "C")
        record = next(run_task(task, FakeBackend(), agents=5, samples=3, repetitions=1))
        self.assertEqual(set(record["conditions"]), set("ABCD"))
        self.assertTrue(all(values == ["C"] * 3 for values in record["initial"].values()))
        self.assertNotIn("external panel", second_prompt(task, "C", "A"))
        self.assertIn("external panel", second_prompt(task, "C", "B"))
        self.assertIn("Check each alternative", second_prompt(task, "C", "C"))
        self.assertIn("Check each alternative", second_prompt(task, "C", "D"))


if __name__ == "__main__":
    unittest.main()
