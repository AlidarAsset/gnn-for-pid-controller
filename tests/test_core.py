import json
import tempfile
import unittest
from pathlib import Path
import torch
import yaml
from src.experiment import load_config, build_model, build_system, build_controller
from src.evaluation.metrics import compute_metrics
from src.training.rollout import rollout
from src.training.losses import compute_loss
from src.training.train import train_gnn
from src.gnn.models import SimpleMPNNLayer


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.c = load_config('configs/default.yaml')
        self.c['experiment']['T'] = .12
        self.c['training']['T_train'] = .12
        self.c['reference']['t_step'] = .03
        self.system = build_system(self.c,'cpu')

    def test_bidirectional_and_zero_equilibrium(self):
        net = build_model(self.c,'cpu')
        ctl = build_controller(self.c,self.system,net)
        self.assertEqual(ctl.deg.tolist(), [1,2,2,2,1])
        z = torch.zeros(5)
        u, _, correction, _, _ = ctl.compute(z,z,z,.01)
        self.assertTrue(torch.equal(u,z))
        self.assertTrue(torch.equal(correction,z))

    def test_saturation_both_controllers(self):
        self.c['pid']['u_sat'] = .1
        for net in (None,build_model(self.c,'cpu')):
            tr = rollout(self.system,build_controller(self.c,self.system,net),self.c,gnn=net is not None)
            self.assertLessEqual(float(tr['u'].detach().abs().max()), .100001)
            self.assertTrue(torch.isfinite(compute_loss(tr,self.c,self.system.edge_index)))

    def test_backward_and_checkpoint_loading(self):
        legacy = load_config('configs/legacy.yaml')
        # Verify serialization without requiring a distributed checkpoint.
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d)/'weights.pth'
            net = build_model(legacy,'cpu')
            torch.save(net.state_dict(), checkpoint)
            restored = build_model(legacy,'cpu')
            restored.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
            for expected, actual in zip(net.parameters(), restored.parameters()):
                torch.testing.assert_close(expected, actual)
        net = build_model(self.c,'cpu')
        tr = rollout(self.system,build_controller(self.c,self.system,net),self.c,train_mode=True,gnn=True)
        compute_loss(tr,self.c,self.system.edge_index).backward()
        grads = [p.grad for p in net.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum()>0 for g in grads))

    def test_message_passing(self):
        layer = SimpleMPNNLayer(2,3)
        x = torch.randn(3,2)
        edges = torch.tensor([[0,2,1],[1,1,2]])
        expected = torch.zeros(3,3)
        for s,d in edges.t():
            expected[d] += layer.message(x[d],x[s],None)
        torch.testing.assert_close(layer(x,edges),layer.update(expected,x))

    def test_settling_after_last_violation(self):
        t = torch.arange(7,dtype=torch.float32)
        r = torch.tensor([[0.],[1.],[1.],[1.],[1.],[1.],[1.]])
        x = torch.tensor([[0.],[0.],[1.],[.8],[1.],[1.],[1.]])
        tr={'t':t,'r':r,'x':x,'e':r-x,'u':torch.zeros_like(x)}
        edges=torch.empty(2,0,dtype=torch.long)
        m=compute_metrics(tr,1,edges,t_step=1)
        self.assertEqual(m['settling_time'],3)
        x[-1]=.5; tr['e']=r-x
        self.assertIsNone(compute_metrics(tr,1,edges,t_step=1)['settling_time'])

    def test_short_training(self):
        self.c['training'].update(episodes=3,batch_size=2,epochs=1,validation_episodes=1)
        with tempfile.TemporaryDirectory() as d:
            config=Path(d)/'config.yaml'; config.write_text(yaml.safe_dump(self.c))
            out=Path(d)/'run'
            train_gnn(config,out,'cpu')
            h=json.loads((out/'history.json').read_text())
            self.assertEqual(h[0]['episodes'],3)
            self.assertTrue((out/'gnn_model_best.pth').exists())
            with self.assertRaises(FileExistsError): train_gnn(config,out,'cpu')


if __name__ == '__main__': unittest.main()
