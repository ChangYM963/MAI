import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demo.mai_demo import example_record
from mai.backends import SimulationBackend
from mai.construction import build_tasks
from mai.interventions import fit_hidden_controller, hidden_update
from mai.metrics import parse_option
from mai.pipeline import run_pipeline
from mai.protocols import summarize_drift
from mai.runner import ChatBackend
from mai.statistics import aggregate_tasks, drift_intervals, holm, paired_analysis, sign_flip
from mai.tasks import integer_oracle, make_tasks, solve_task

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_parse_exactly_one_letter(self):
        for invalid in ("", " ", "AB", "ABCDE", "a", "Answer: A", None):
            self.assertIsNone(parse_option(invalid))
        self.assertEqual(parse_option(" A\n"), "A")

    def test_construction_audits_routes_and_assumptions(self):
        materials = json.loads((ROOT / "examples/materials.json").read_text())
        result = build_tasks(materials)
        self.assertEqual([len(result[key]) for key in ("accepted", "flagged", "rejected")], [12, 1, 1])
        self.assertEqual([stage["stage"] for stage in result["audit"][0]["stages"]],
                         ["screen", "normalize", "rewrite", "render", "quality"])
        self.assertTrue(all("answer" not in task for task in result["accepted"]))
        self.assertEqual(result["accepted"][0]["constraints"], materials[0]["constraints"])
        self.assertTrue(result["flagged"][0]["assumptions"])

    def test_two_independent_oracles_and_unique_cases(self):
        tasks = make_tasks(4, seed=2026)
        self.assertEqual(len(tasks), 60)
        self.assertEqual(len({(task["type"], tuple(sorted(task["parameters"].items()))) for task in tasks}), 60)
        for task in tasks:
            self.assertEqual(solve_task(task), integer_oracle(task))
            self.assertNotEqual(task["answer"], task["incorrect_majority"])

    def test_hidden_projection_gate_and_training(self):
        examples = [{"before": [0., 0.], "after": [1., 0.], "positive": True},
                    {"before": [0., 1.], "after": [0., 1.], "positive": False}]
        controller = fit_hidden_controller(examples)
        self.assertTrue(controller["available"])
        delta, info = hidden_update([2., 0.], controller, True, alpha=0.5, threshold=0.)
        self.assertAlmostEqual(delta[0], -0.5 * info["risk"] * 2.)
        self.assertEqual(delta[1], 0.)
        self.assertEqual(hidden_update([2., 0.], controller, False, threshold=0.)[0], [0., 0.])
        self.assertFalse(hidden_update([-2., 0.], controller, True, threshold=0.)[1]["triggered"])

    def test_paired_inference_and_common_task_exclusion(self):
        record = example_record()
        other = copy.deepcopy(record)
        other["task_id"] = "second-task"
        result = paired_analysis([record, other], bootstrap=100, flips=100)
        self.assertEqual(result["metrics"]["accuracy_loss"]["estimate"], 1.)
        self.assertEqual(result["metrics"]["accuracy_loss"]["ci95"], [1., 1.])
        self.assertEqual(result["metrics"]["B.retention"]["estimate"], 1 / 3)
        second = copy.deepcopy(record)
        second["repetition"] = 2
        second["conditions"]["D"]["agent-1"] = ["C", "invalid", "C"]
        task = aggregate_tasks([record, second], expected_repetitions=2)[0]
        self.assertFalse(task["common_valid_mai"])
        self.assertTrue(all(task["conditions"][c]["drift"] is None for c in "ABCD"))
        with self.assertRaises(ValueError):
            aggregate_tasks([record, record])
        with self.assertRaises(ValueError):
            aggregate_tasks([record], expected_repetitions=2)
        self.assertEqual(sign_flip([1, 1])["p"], 0.5)
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_chat_request_contract(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def read(self):
                return b'{"choices":[{"message":{"content":"C"}}]}'
        with patch("mai.runner.request.urlopen", return_value=Response()) as urlopen:
            backend = ChatBackend("http://127.0.0.1:8000/v1", "test-model", "test-only-value")
            self.assertEqual(backend.generate("choose", 7), "C")
            req = urlopen.call_args[0][0]
            payload = json.loads(req.data)
            self.assertEqual(payload["seed"], 7)
            self.assertEqual(payload["messages"], [{"role": "user", "content": "choose"}])
        with self.assertRaises(ValueError):
            ChatBackend("http://localhost.example.com/v1", "m", "")

    def test_contextual_excludes_incomplete_tasks_and_uses_prompt_zero(self):
        def record(task, repetition):
            return {"task_id": task, "repetition": repetition, "valid": True,
                    "branches": {branch: {"mean_minority_drift": value} for branch, value in
                                 (("social", 1.), ("prompt_zero", 0.4), ("prompt_selected", 0.1))}}
        rows = [record("complete", 1), record("complete", 2), record("partial", 1),
                {"task_id": "partial", "repetition": 2, "valid": False}]
        self.assertEqual(summarize_drift(rows)["social"]["valid_tasks"], 1)
        inference = drift_intervals(rows, bootstrap=10)
        self.assertEqual(inference["paired_tasks"], 1)
        self.assertAlmostEqual(inference["comparisons"]["prompt_incremental"]["H_plus_gain"], 0.3)

    def test_complete_offline_run_and_manifest(self):
        config = json.loads((ROOT / "configs/demo.json").read_text())
        config["materials"] = str(ROOT / "examples/materials.json")
        config["bootstrap_replicates"] = 100
        config["sign_flip_replicates"] = 100
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_file = root / "config.json"
            config_file.write_text(json.dumps(config))
            output = root / "run"
            summary = run_pipeline(config_file, output, progress=lambda _: None)
            self.assertEqual(summary["construction"]["accepted"], 12)
            self.assertEqual(summary["reliability"]["tasks"], 15)
            self.assertTrue(summary["hidden_audit"]["available"])
            self.assertEqual(set(summary["contextual"]), {"social", "prompt_zero", "prompt_selected", "postprocess", "hidden"})
            split = json.loads((output / "contextual_tasks.json").read_text())
            self.assertFalse({x["id"] for x in split["calibration"]} & {x["id"] for x in split["evaluation"]})
            records = [json.loads(line) for line in (output / "contextual_records.jsonl").read_text().splitlines()]
            for record in records:
                baseline = record["branches"]["social"]
                corrected = record["branches"]["postprocess"]
                self.assertEqual(baseline["initial_majority"], corrected["initial_majority"])
                self.assertEqual(baseline["initial_minority"], corrected["initial_minority"])
                self.assertAlmostEqual(corrected["mean_minority_drift"], 0.4 * baseline["mean_minority_drift"])
            manifest = json.loads((output / "manifest.json").read_text())
            for name, expected in manifest["files"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
            self.assertIn("SIMULATED WORKFLOW DEMONSTRATION", (output / "report.html").read_text())
            with self.assertRaises(FileExistsError):
                run_pipeline(config_file, output, progress=lambda _: None)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency is not installed")
    def test_real_tensor_hook_changes_only_final_position(self):
        import torch
        from mai.local import steering_hook
        tensor = torch.ones(1, 4, 3)
        hook = steering_hook(torch, [0., -0.5, 0.])
        result = hook(None, None, (tensor, "cache"))
        self.assertTrue(torch.equal(result[0][:, :-1, :], tensor[:, :-1, :]))
        self.assertAlmostEqual(result[0][0, -1, 1].item(), 0.5)
        self.assertEqual(result[1], "cache")
        self.assertEqual(tensor[0, -1, 1].item(), 1.)


if __name__ == "__main__":
    unittest.main()
