import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
from src.training.rollout import rollout
from src.dynamics.msd_network import MSDNetwork
from src.controllers.pid import DistributedPID
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_trajectories
from seeds import set_global_seed

if __name__ == "__main__":
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_global_seed(config['experiment']['seed'])
    device = torch.device('cpu')

    edge_index = config['graph']['edges']
    m = config['node_params']['m']
    d = config['node_params']['d']
    k = config['node_params']['k']
    k_c = config['graph']['k_c']

    system = MSDNetwork(edge_index, k_c, m, d, k, device)
    pid = DistributedPID(system.N, config['pid']['Kp'], config['pid']['Ki'], config['pid']['Kd'],
                         config['pid']['I_max'], config['pid']['alpha'], device)
    traj = rollout(system, pid, config, gnn=False)

    metrics = compute_metrics(traj, config['experiment']['dt'], system.edge_index)
    print("Baseline Metrics:", metrics)

    plot_trajectories(traj, "Baseline PID", "outputs/baseline_plots.png")