import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class SimpleMPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_feat_dim=0):
        super().__init__(aggr='add')
        self.mlp_msg = nn.Linear(2 * in_channels + edge_feat_dim, out_channels)
        self.mlp_update = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr] if edge_attr is not None else [x_i, x_j], dim=-1)
        return self.mlp_msg(tmp)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_update(tmp)

class GNNAugmentor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, u_max, norm_scales, dropout=0.0, activation='relu', device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SimpleMPNNLayer(input_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(SimpleMPNNLayer(hidden_dim, hidden_dim))
        self.readout = nn.Linear(hidden_dim, 1)
        self.u_max = u_max
        self.norm_scales = norm_scales
        self.dropout = dropout
        self.activation = F.relu if activation == 'relu' else F.tanh
        self.to(device)

    def forward(self, node_feat, edge_index, edge_attr=None, deg=None):
        e_n = node_feat[:, 0] / self.norm_scales['e']
        v_n = node_feat[:, 1] / self.norm_scales['v']
        I_n = node_feat[:, 2] / self.norm_scales['I']
        u_pid_n = node_feat[:, 3] / self.norm_scales['u_pid']
        deg_n = deg / self.norm_scales['deg'] if deg is not None else node_feat[:, 4]
        x = torch.stack([e_n, v_n, I_n, u_pid_n, deg_n], dim=1)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        raw = self.readout(x).squeeze(-1)
        u_gnn = self.u_max * torch.tanh(raw)
        return u_gnn