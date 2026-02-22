import torch

def compute_loss(trajectories, config, edge_index):
    dt = config['experiment']['dt']
    weights = config['training']['loss_weights']

    e_loss = weights['e'] * (trajectories['e'] ** 2).sum(dim=1).mean() * dt
    v_loss = weights['v'] * (trajectories['v'] ** 2).sum(dim=1).mean() * dt
    u_loss = weights['u'] * (trajectories['u'] ** 2).sum(dim=1).mean() * dt
    du_loss = weights['du'] * (trajectories['u_gnn'] ** 2).sum(dim=1).mean() * dt if 'u_gnn' in trajectories else 0

    i, j = edge_index[0], edge_index[1]
    sync_diff = (trajectories['x'][:, i] - trajectories['x'][:, j]) ** 2
    sync_loss = weights['sync'] * sync_diff.sum(dim=1).mean() * dt

    total_loss = e_loss + v_loss + u_loss + du_loss + sync_loss
    return total_loss