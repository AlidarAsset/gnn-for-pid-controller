import torch

class MSDNetwork:
    def __init__(self, edge_index, k_c, m, d, k, device='cpu'):
        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()  # [2, E]
        self.k_c = torch.tensor(k_c, dtype=torch.float, device=device) if isinstance(k_c, list) else torch.full((self.edge_index.shape[1],), k_c, device=device)  # [E]
        self.m = torch.tensor(m, dtype=torch.float, device=device)  # [N]
        self.d = torch.tensor(d, dtype=torch.float, device=device)  # [N]
        self.k = torch.tensor(k, dtype=torch.float, device=device)  # [N]
        self.N = len(m)
        self.device = device
        self.params = {"m": self.m, "d": self.d, "k": self.k}

    def coupling_force(self, x):
        i, j = self.edge_index[0], self.edge_index[1]
        diff = x[i] - x[j]  # [E]
        f = -self.k_c * diff  # [E]
        f_c = torch.zeros(self.N, device=self.device)
        f_c.index_add_(0, i, f)
        f_c.index_add_(0, j, -f)  # opposite on j
        return f_c

    def f(self, state, u):
        x, v = state
        f_c = self.coupling_force(x)
        x_dot = v
        v_dot = (-self.d * v - self.k * x + f_c + u) / self.m
        return x_dot, v_dot