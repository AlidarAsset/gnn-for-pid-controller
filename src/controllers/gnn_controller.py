import torch


class HybridController:
    def __init__(self, pid, gnn, edge_index, device='cpu', *, graph_mode='legacy_directed',
                 normalize_degree=False, zero_equilibrium=False):
        self.pid = pid
        self.gnn = gnn
        if graph_mode == 'bidirectional':
            edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
        elif graph_mode != 'legacy_directed':
            raise ValueError(f'Unknown graph_mode: {graph_mode}')
        self.edge_index = edge_index
        self.normalize_degree = normalize_degree
        self.zero_equilibrium = zero_equilibrium
        self.deg = torch.zeros_like(pid.Kp)
        self.deg.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=device))

    def reset(self):
        self.pid.reset()

    def compute(self, x, v, r, dt):
        u_pid, e, I = self.pid.compute(x, v, r, dt)
        features = torch.stack([e, v, I, u_pid, self.deg], dim=1)
        u_gnn = self.gnn(features, self.edge_index,
                         deg=self.deg if self.normalize_degree else None)
        if self.zero_equilibrium:
            # Preserve zero correction at the zero-state/zero-reference equilibrium.
            # Smooth, node-local gate; not a formal stability guarantee.
            activity = e.abs() + v.abs() + I.abs() + u_pid.abs() + x.abs() + r.abs()
            u_gnn = u_gnn * torch.tanh(activity)
        u = self.pid.limit(u_pid + u_gnn)
        return u, u_pid, u_gnn, e, I
