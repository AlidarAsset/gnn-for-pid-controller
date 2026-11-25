import torch
import yaml
from src.dynamics.msd_network import MSDNetwork
from src.dynamics.integrators import rk4_step, euler_step
from src.controllers.pid import DistributedPID
from src.controllers.gnn_controller import HybridController

def generate_reference(config, t, N, device):
    ref_type = config['reference']['type']
    if ref_type == 'step':
        r_values = torch.tensor(config['reference']['values'], dtype=torch.float, device=device)
        t_step = config['reference']['t_step']
        
        # Правильный способ: сравниваем скаляр t с t_step → получаем bool → преобразуем в float
        step_active = 1.0 if t >= t_step else 0.0
        return step_active * r_values
    
    if ref_type == "zero":
        return torch.zeros(N, device=device)
    raise ValueError(f"Unsupported reference: {ref_type}")

def rollout(system, controller, config, train_mode=False, gnn=False):
    dt = config['experiment']['dt']
    T = config['training']['T_train'] if train_mode else config['experiment']['T']
    if dt <= 0 or T <= 0:
        raise ValueError("dt and T must be positive")
    steps = round(T / dt)
    if steps < 1 or abs(steps * dt - T) > 1e-6:
        raise ValueError("T must be an integer multiple of dt")
    N = system.N
    device = system.device

    x = torch.zeros(N, device=device)
    v = torch.zeros(N, device=device)
    state = (x, v)

    integrator = {'rk4': rk4_step, 'euler': euler_step}[config['experiment']['integrator']]

    trajectories = {
        'x': torch.zeros(steps, N, device=device),
        'v': torch.zeros(steps, N, device=device),
        'u': torch.zeros(steps, N, device=device),
        'u_pid': torch.zeros(steps, N, device=device),
        'u_gnn': torch.zeros(steps, N, device=device) if gnn else None,
        'e': torch.zeros(steps, N, device=device),
        'I': torch.zeros(steps, N, device=device),
        'r': torch.zeros(steps, N, device=device),
        't': torch.arange(steps, device=device) * dt,
    }

    controller.reset()

    for step in range(steps):
        t = step * dt
        r = generate_reference(config, t, N, device)
        if gnn:
            u, u_pid, u_gnn, e, I = controller.compute(x, v, r, dt)
            trajectories['u_gnn'][step] = u_gnn
        else:
            u_pid, e, I = controller.compute(x, v, r, dt)
            u = controller.limit(u_pid)
        trajectories['x'][step] = x
        trajectories['v'][step] = v
        trajectories['u'][step] = u
        trajectories['u_pid'][step] = u_pid
        trajectories['e'][step] = e
        trajectories['I'][step] = I
        trajectories['r'][step] = r

        state = integrator(state, u, dt, system.f)
        x, v = state

    return trajectories