import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import urllib.error

import torch

from sys2.engine import batch, evaluate, load, solve, train
from sys2.model import BOS, EOS, ModelConfig, Planner, encode
from sys2.tasks import Maze, example, generate, verify
from sys2.ui import make_server

torch.set_num_threads(2)

class TaskTests(unittest.TestCase):
    def test_verifier(self):
        maze = Maze(("S.X", "...", "..G"))
        self.assertTrue(verify(maze, "DDRR")["optimal"])
        self.assertFalse(verify(maze, "DDRU" )["valid"])
        self.assertEqual(verify(maze, "RR")["reason"], "blocked")
        self.assertEqual(verify(maze, "U")["reason"], "blocked")
        self.assertEqual(verify(maze, "?")["reason"], "invalid_symbol")
        self.assertTrue(verify(maze, "DRUDDR")["valid"])
        self.assertFalse(verify(maze, "DRUDDR")["optimal"])
        with self.assertRaises(ValueError):
            Maze(("SS", "G."))

    def test_synthetic_data(self):
        for size in [2, 4, 8]:
            for seed in range(15):
                maze = generate(seed, size)
                self.assertEqual(maze, generate(seed, size))
                self.assertTrue(verify(maze, maze.shortest())["optimal"])
                prompt, answer = example(seed, size)
                self.assertTrue(prompt.endswith("path:"))
                self.assertTrue(verify(maze, answer)["valid"])

    def test_feedback_reaches_model(self):
        class Stub:
            def __init__(self): self.prompts = []
            def generate(self, prompt, **kwargs):
                self.prompts.append(prompt)
                return "U" if len(self.prompts) == 1 else "DR"
        stub = Stub()
        result = solve(stub, Maze(("S.", ".G")), attempts=3)
        self.assertTrue(result["solved"])
        self.assertEqual(result["attempts_used"], 2)
        self.assertIn("attempt:U\nfeedback:blocked", stub.prompts[1])

class ModelTests(unittest.TestCase):
    def test_causal_attention_and_masking(self):
        torch.manual_seed(1)
        model = Planner(ModelConfig(32, 2, 4, 256)).eval()
        a = torch.tensor([[BOS, 10, 20, 30]])
        b = torch.tensor([[BOS, 10, 50, 60]])
        torch.testing.assert_close(model(a)[:, :2], model(b)[:, :2])
        x, y = batch([("path:", "DR"), ("maze:\nSG\n..\npath:", "R")], "cpu", 256)
        self.assertEqual(int((y[0] != -100).sum()), 3)
        self.assertEqual(y[0, len(encode("path:"))].item(), ord("D"))
        self.assertEqual(y[0, len(encode("path:"))+2].item(), EOS)

    def test_production_artifact_size(self):
        model = Planner()
        self.assertTrue(30 <= model.info()["fp32_weight_MB"] <= 100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"weights.pt"
            torch.save(model.state_dict(), path)
            self.assertTrue(30e6 <= path.stat().st_size <= 100e6)

    def test_train_resume_eval_and_ui(self):
        cfg = ModelConfig(32, 2, 4, 256)
        with tempfile.TemporaryDirectory() as tmp:
            run, other = Path(tmp)/"run", Path(tmp)/"other"
            opts = dict(config=cfg, batch_size=4, train_examples=12, size=3, seed=42, save_every=2, device="cpu")
            train(run, steps=2, **opts)
            self.assertTrue((run/"last.pt").exists())
            train(run, steps=4, resume=True, device="cpu")
            train(other, steps=4, **opts)
            model, direct = load(run), load(other)
            for a, b in zip(model.parameters(), direct.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            result = evaluate(model, run, count=3, size=3, attempts=2, temperature=0)
            self.assertEqual(result["count"], 3)
            forbidden = set(json.loads((run/"manifest.json").read_text())["train_mazes"])
            self.assertTrue(all("\n".join(r["maze"]) not in forbidden for r in result["records"]))
            self.assertGreaterEqual(result["verified_retry_success"], result["single_attempt_success"])
            for record in result["records"]:
                self.assertEqual(record["history"][0], record["no_feedback"]["history"][0])
            server = make_server(model, port=0, attempts=1, temperature=0)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(url) as response:
                    self.assertIn(b"Tiny System 2", response.read())
                request = urllib.request.Request(url+"/api/solve", data=json.dumps({"rows":["S.", ".G"]}).encode(), headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(request) as response:
                    answer = json.load(response)
                    self.assertIn("history", answer)
                    self.assertEqual(answer["attempts_used"], 1)
                for rows in [["SS", ".G"], ["S?", ".G"], ["S..", ".G"]]:
                    request = urllib.request.Request(url+"/api/solve", data=json.dumps({"rows":rows}).encode(), headers={"Content-Type":"application/json"})
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(request)
                    self.assertEqual(error.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                worker.join()

if __name__ == "__main__":
    unittest.main()
