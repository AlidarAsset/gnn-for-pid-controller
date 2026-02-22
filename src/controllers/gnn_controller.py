import torch
from src.gnn.models import GNNAugmentor

class HybridController:
    def __init__(self, pid, gnn, edge_index, device='cpu'):
        self.pid = pid
        self.gnn = gnn
        self.edge_index = edge_index
        self.device = device

    def compute(self, x, v, r, dt):
        u_pid, e, I = self.pid.compute(x, r, dt)
        deg = torch.zeros(self.pid.Kp.shape[0], device=self.device)
        deg.index_add_(0, self.edge_index[0], torch.ones(self.edge_index.shape[1], device=self.device))
        node_feat = torch.stack([e, v, I, u_pid, deg], dim=1)
        u_gnn = self.gnn(node_feat, self.edge_index)
        u = u_pid + u_gnn
        if hasattr(self.pid, 'u_sat'):
            u = torch.clamp(u, -self.pid.u_sat, self.pid.u_sat)
        return u, u_pid, u_gnn, e, I