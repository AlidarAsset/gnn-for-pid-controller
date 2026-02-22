import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
from src.training.rollout import rollout
from src.dynamics.msd_network import MSDNetwork
from src.controllers.pid import DistributedPID
from src.gnn.models import GNNAugmentor
from src.controllers.gnn_controller import HybridController
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_trajectories
from seeds import set_global_seed

if __name__ == "__main__":
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_global_seed(config['experiment']['seed'])
    device = torch.device('cpu')

    input_dim = 5
    gnn = GNNAugmentor(input_dim, config['gnn']['hidden'], config['gnn']['layers'], config['gnn']['u_max'],
                       config['gnn']['norm_scales'], config['gnn']['dropout'], config['gnn']['activation'], device)

    
    model_path = 'models/gnn_model_best.pth'
    if not os.path.exists(model_path):
        model_path = 'models/gnn_model.pth'  
        print(f"Лучшая модель не найдена, использую последнюю: {model_path}")
    else:
        print(f"Загружаю лучшую модель: {model_path}")
    
    gnn.load_state_dict(torch.load(model_path, map_location=device))
    gnn.eval()

    edge_index = config['graph']['edges']
    m = config['node_params']['m']
    d = config['node_params']['d']
    k = config['node_params']['k']
    k_c = config['graph']['k_c']

    system = MSDNetwork(edge_index, k_c, m, d, k, device)
    pid = DistributedPID(system.N, config['pid']['Kp'], config['pid']['Ki'], config['pid']['Kd'],
                         config['pid']['I_max'], config['pid']['alpha'], device)
    controller = HybridController(pid, gnn, system.edge_index, device)

    traj = rollout(system, controller, config, gnn=True)

    metrics = compute_metrics(traj, config['experiment']['dt'], system.edge_index)
    print("GNN-Augmented Metrics:", metrics)

    plot_trajectories(traj, "PID + GNN", "outputs/gnn_plots.png")