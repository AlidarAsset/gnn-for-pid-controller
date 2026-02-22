import matplotlib.pyplot as plt
import torch
import os

def plot_trajectories(trajectories, title, save_path):
    t = trajectories['t'].detach().cpu().numpy()
    x = trajectories['x'].detach().cpu().numpy()
    e = trajectories['e'].detach().cpu().numpy()
    u = trajectories['u'].detach().cpu().numpy()

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    # Positions
    for i in range(x.shape[1]):
        axs[0].plot(t, x[:, i], label=f'Node {i}')
    axs[0].set_title('Positions')
    axs[0].legend()

    # Errors
    for i in range(e.shape[1]):
        axs[1].plot(t, e[:, i], label=f'Node {i}')
    axs[1].set_title('Errors')
    axs[1].legend()

    # Controls
    for i in range(u.shape[1]):
        axs[2].plot(t, u[:, i], label=f'Node {i}')
    axs[2].set_title('Controls')
    axs[2].legend()

    
    if 'u_gnn' in trajectories and trajectories['u_gnn'] is not None:
        u_gnn = trajectories['u_gnn'].detach().cpu().numpy()
        for i in range(u_gnn.shape[1]):
            axs[3].plot(t, u_gnn[:, i], label=f'Node {i}')
        axs[3].set_title('GNN Contributions')
        axs[3].legend()
    else:
        axs[3].set_title('GNN Contributions (not available)')
        axs[3].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()