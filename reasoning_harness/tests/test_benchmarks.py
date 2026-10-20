import json
from pathlib import Path
import tempfile
import unittest

import torch

from sys2.benchmarks import audit_verifier, benchmark, paired_difference, sha256, wilson
from sys2.engine import load, train
from sys2.model import ModelConfig

class BenchmarkTests(unittest.TestCase):
    def test_learning_and_invalid_inference_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/"learning"
            train(run, config=ModelConfig(32,2,4,256), steps=100, batch_size=8,
                  train_examples=8, size=3, lr=.003, device="cpu")
            losses = [json.loads(line)["loss"] for line in (run/"metrics.jsonl").read_text().splitlines()]
            self.assertLess(sum(losses[-10:]), sum(losses[:10])*.5)
            model = load(run, "cpu")
            for temperature in [float("nan"), float("inf"), -1]:
                with self.assertRaises(ValueError):
                    model.generate("path:", temperature=temperature)
            with self.assertRaises(ValueError):
                model.generate("X"*256)

    def test_statistics_and_independent_audit(self):
        low, high = wilson(0, 100)
        self.assertAlmostEqual(low, 0)
        self.assertTrue(0 < high < .04)
        effect = paired_difference([True]*20, [False]*20)
        self.assertEqual(effect["difference"], 1)
        self.assertEqual(effect["bootstrap_95_ci"], [1, 1])
        self.assertEqual(paired_difference([True, False], [True, False])["bootstrap_95_ci"], [0, 0])
        audit = audit_verifier()
        self.assertEqual(audit["layouts"], 48)
        self.assertEqual(audit["path_checks"], 16512)

    def test_saved_benchmark_evidence_and_failure_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            run, output = Path(directory)/"run", Path(directory)/"benchmark"
            train(run, config=ModelConfig(16, 1, 2, 256), steps=2, batch_size=2,
                  train_examples=8, size=3, device="cpu")
            model = load(run, "cpu")
            checkpoint = torch.load(run/"last.pt", map_location="cpu", weights_only=True)
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, checkpoint["model"][key], rtol=0, atol=0)
            report = benchmark(run, output, count=2, sizes=(3,4), seeds=(100,200,300), attempts=2, temperature=0, device="cpu")
            self.assertEqual(report["count"], 12)
            self.assertFalse(report["ready"])
            self.assertFalse(report["gates"]["weight_size_30_to_100_MB"])
            self.assertTrue(report["gates"]["no_training_overlap"])
            self.assertTrue(report["gates"]["bfs_all_cases_valid_and_optimal"])
            self.assertEqual(report["weights_sha256"], sha256(run/"weights.pt"))
            raw = [json.loads(line) for line in (output/"records.jsonl").read_text().splitlines()]
            cases = json.loads((output/"cases.json").read_text())
            self.assertEqual(len(raw), len(cases))
            self.assertEqual(len({tuple(c["rows"]) for c in cases}), 12)
            for record in raw:
                self.assertEqual(record["feedback"]["history"][0], record["no_feedback"]["history"][0])
                self.assertTrue(record["bfs"]["history"][-1]["optimal"])
                self.assertGreaterEqual(record["feedback"]["seconds"], 0)
            with self.assertRaises(ValueError):
                benchmark(run, output, count=2)
            with self.assertRaises(ValueError):
                benchmark(run, Path(directory)/"invalid", sizes=(4,4))

if __name__ == "__main__":
    unittest.main()
