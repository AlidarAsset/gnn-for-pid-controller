import torch

def compute_metrics(trajectories, dt, edge_index):
    metrics = {}
    metrics['ISE'] = (trajectories['e'] ** 2).sum() * dt
    metrics['IAE'] = torch.abs(trajectories['e']).sum() * dt
    metrics['U2'] = (trajectories['u'] ** 2).sum() * dt

    i, j = edge_index[0], edge_index[1]
    sync_diff = (trajectories['x'][:, i] - trajectories['x'][:, j]) ** 2
    metrics['Sync'] = sync_diff.sum() * dt

    r_final = trajectories['r'][-1]
    e_norm = torch.abs(trajectories['e']) / (r_final + 1e-6)
    settled = (e_norm < 0.02).all(dim=1).float()
    settled_steps = (settled.cumsum(0) == torch.arange(1, len(settled)+1, device=settled.device)).int().argmax() + 1
    metrics['settling_time'] = settled_steps * dt if settled_steps < len(settled) else float('inf')

    overshoot = (trajectories['x'].max(0)[0] - r_final).clamp(min=0) / (r_final + 1e-6)
    metrics['overshoot'] = overshoot.mean().item()

    return metrics