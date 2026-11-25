"""Shared construction for training and evaluation."""
import torch
import yaml
from src.controllers.pid import DistributedPID
from src.controllers.gnn_controller import HybridController
from src.dynamics.msd_network import MSDNetwork
from src.gnn.models import GNNAugmentor


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(c, device):
    g = c['gnn']
    return GNNAugmentor(5, g['hidden'], g['layers'], g['u_max'], g['norm_scales'],
                        g['dropout'], g['activation'], device)


def build_system(c, device, params=None):
    p = c['node_params']
    m, d, k, kc = params if params is not None else (p['m'], p['d'], p['k'], c['graph']['k_c'])
    return MSDNetwork(c['graph']['edges'], kc, m, d, k, device)


def build_controller(c, system, model=None):
    p = c['pid']
    pid = DistributedPID(system.N, p['Kp'], p['Ki'], p['Kd'], p['I_max'], p['alpha'],
                         system.device, u_sat=p.get('u_sat'))
    if model is None:
        return pid
    g = c['gnn']
    return HybridController(pid, model, system.edge_index, system.device,
                            graph_mode=g.get('graph_mode','legacy_directed'),
                            normalize_degree=g.get('normalize_degree',False),
                            zero_equilibrium=g.get('zero_equilibrium',False))


def sample_params(c, rng):
    if not c['training']['randomize_params']:
        return None
    ranges = c['training']['param_ranges']
    n = c['experiment']['N']
    return (*[torch.empty(n).uniform_(*ranges[k], generator=rng) for k in ('m','d','k')],
            float(torch.empty(1).uniform_(*ranges['k_c'], generator=rng)))
