import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request

import torch

from sys2.engine import load
from sys2.policy import MovePolicy,PolicyConfig,features,task_key,train_policy
from sys2.tasks import Maze,generate,verify
from sys2.ui import make_server
from sys2.benchmarks import validation_loss

class PolicyTests(unittest.TestCase):
    def test_search_completeness_and_budget(self):
        model=MovePolicy(PolicyConfig(16,2)).eval()
        for size in (3,4,5,6,8):
            for seed in range(10):
                maze=generate(seed,size)
                result=model.search(maze,64)
                self.assertTrue(result["solved"])
                self.assertLessEqual(result["expanded_states"],size*size)
                self.assertTrue(verify(maze,result["history"][-1]["path"])["valid"])
        impossible=Maze(("SX","XG"))
        self.assertFalse(model.search(impossible,64)["solved"])
        self.assertFalse(model.search(Maze(("S.",".G")),1)["solved"])

    def test_export_resume_and_http_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            run,other=Path(directory)/"split",Path(directory)/"direct"
            options=dict(mazes=8,batch_size=4,seed=5,device="cpu",width=16,layers=2)
            train_policy(run,steps=2,**options)
            train_policy(run,steps=4,resume=True,**options)
            train_policy(other,steps=4,**options)
            m,n=load(run,"cpu"),load(other,"cpu")
            for a,b in zip(m.parameters(),n.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(m.training_steps,4)
            manifest=json.loads((run/"manifest.json").read_text())
            manifest["train_mazes"].append("\n".join(generate(4000000,4).rows))
            (run/"manifest.json").write_text(json.dumps(manifest))
            validation=validation_loss(m,run,count=3)
            self.assertEqual(validation["count"],3)
            self.assertGreaterEqual(validation["overlaps_skipped"],1)
            self.assertNotIn(4000000,validation["maze_seeds"])
            server=make_server(m,port=0,temperature=0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                url=f"http://127.0.0.1:{server.server_port}"
                with urllib.request.urlopen(url+"/api/model") as response:
                    metadata=json.load(response)
                self.assertTrue(metadata["supports_search"])
                request=urllib.request.Request(url+"/api/solve",data=json.dumps({"rows":["S.",".G"],"mode":"guided"}).encode(),headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(request) as response:
                    result=json.load(response)
                self.assertTrue(result["solved"])
                self.assertEqual(result["mode"],"policy_guided_search")
            finally:
                server.shutdown();server.server_close();thread.join()

    def test_policy_size_and_input_dimensions(self):
        self.assertEqual(task_key(("S.G","...","...")),task_key(("..G","S..","...")))
        self.assertNotEqual(task_key(("S.G","...","...")),task_key(("S.G",".X.","...")))
        model=MovePolicy()
        self.assertTrue(30 <= model.info()["fp32_weight_MB"] <= 100)
        x,legal=features(Maze(("SG","..")),(0,0))
        self.assertEqual(x.shape,(207,))
        self.assertEqual(legal.tolist(),[False,True,False,True])

if __name__=="__main__":
    unittest.main()
