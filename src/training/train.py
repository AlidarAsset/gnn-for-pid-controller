import torch
import yaml
import random
import numpy as np
from torch.optim import Adam
from src.training.rollout import rollout
from src.training.losses import compute_loss
from src.dynamics.msd_network import MSDNetwork
from src.controllers.pid import DistributedPID
from src.controllers.gnn_controller import HybridController
from src.gnn.models import GNNAugmentor
import os
import time

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def randomize_params(config, N, device):
    ranges = config['training']['param_ranges']
    m = torch.FloatTensor(N).uniform_(ranges['m'][0], ranges['m'][1]).to(device)
    d = torch.FloatTensor(N).uniform_(ranges['d'][0], ranges['d'][1]).to(device)
    k = torch.FloatTensor(N).uniform_(ranges['k'][0], ranges['k'][1]).to(device)
    k_c = config['graph']['k_c'] * torch.FloatTensor(1).uniform_(
        ranges['k_c'][0] / config['graph']['k_c'],
        ranges['k_c'][1] / config['graph']['k_c']
    ).item()
    return m, d, k, k_c

def train_gnn(config_path='configs/default.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['experiment']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    input_dim = 5  # e, v, I, u_pid, deg
    gnn = GNNAugmentor(
        input_dim,
        config['gnn']['hidden'],
        config['gnn']['layers'],
        config['gnn']['u_max'],
        config['gnn']['norm_scales'],
        config['gnn']['dropout'],
        config['gnn']['activation'],
        device
    )
    gnn.train()  
    print(f"GNN создана и в режиме train(): {config['gnn']['layers']} слоёв, hidden={config['gnn']['hidden']}")

    optimizer = Adam(
        gnn.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )

    episodes = config['training']['episodes']
    batch_size = config['training']['batch_size']
    num_batches = (episodes + batch_size - 1) // batch_size

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, config['training']['epochs'] + 1):
        epoch_start_time = time.time()
        print(f"\n=== Эпоха {epoch}/{config['training']['epochs']} начата ===")
        
        epoch_loss = 0.0
        
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            print(f"  Батч {batch_idx + 1}/{num_batches} начат...")
            
            batch_loss = torch.tensor(0.0, requires_grad=True, device=device)
            optimizer.zero_grad()

            for rollout_idx in range(batch_size):
                rollout_start = time.time()
                print(f"    Rollout {rollout_idx + 1}/{batch_size} начат...")

                N = config['experiment']['N']
                edge_index = config['graph']['edges']

                if config['training']['randomize_params']:
                    m, d, k, k_c = randomize_params(config, N, device)
                else:
                    m = torch.tensor(config['node_params']['m'], device=device)
                    d = torch.tensor(config['node_params']['d'], device=device)
                    k = torch.tensor(config['node_params']['k'], device=device)
                    k_c = config['graph']['k_c']

                system = MSDNetwork(edge_index, k_c, m, d, k, device)
                pid = DistributedPID(
                    N,
                    config['pid']['Kp'],
                    config['pid']['Ki'],
                    config['pid']['Kd'],
                    config['pid']['I_max'],
                    config['pid']['alpha'],
                    device
                )
                controller = HybridController(pid, gnn, system.edge_index, device)

                traj = rollout(system, controller, config, train_mode=True, gnn=True)
                loss = compute_loss(traj, config, system.edge_index)

                batch_loss = batch_loss + loss
                print(f"      Rollout {rollout_idx + 1} завершён, loss = {loss.item():.4f}, время: {time.time() - rollout_start:.2f} сек")

            batch_loss = batch_loss / batch_size
            print(f"  Средний batch loss перед backward: {batch_loss.item():.4f}")

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), config['training']['grad_clip'])
            optimizer.step()

            epoch_loss += batch_loss.item()

            batch_time = time.time() - batch_start_time
            print(f"  Батч {batch_idx + 1} завершён | Время: {batch_time:.2f} сек")

        avg_epoch_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        print(f"Эпоха {epoch} завершена | Средний Loss: {avg_epoch_loss:.4f} | Время эпохи: {epoch_time:.2f} сек")

        # Сохраняем лучшую модель
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch
            torch.save(gnn.state_dict(), 'models/gnn_model_best.pth')
            print(f"Новая лучшая модель сохранена! Loss: {best_loss:.4f} (эпоха {best_epoch})")

    # Сохраняем последнюю модель
    torch.save(gnn.state_dict(), 'models/gnn_model_last.pth')
    
    print(f"\nОбучение завершено!")
    print(f"Лучшая модель: models/gnn_model_best.pth (Loss: {best_loss:.4f} на эпохе {best_epoch})")
    print(f"Последняя модель: models/gnn_model_last.pth")
    return gnn