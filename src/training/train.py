"""Differentiable rollouts; select best checkpoint on fixed validation simulations."""
import json
from pathlib import Path
import torch
import yaml
from seeds import set_global_seed
from src.experiment import load_config, build_model, build_system, build_controller, sample_params
from src.training.rollout import rollout
from src.training.losses import compute_loss


def train_gnn(config_path='configs/default.yaml', output_dir='models/v2', device=None):
    c = load_config(config_path)
    seed = c['experiment']['seed']
    set_global_seed(seed)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    tr = c['training']
    episodes, batch, epochs = (tr[k] for k in ('episodes','batch_size','epochs'))
    n_val = tr.get('validation_episodes', 4)
    if min(episodes,batch,epochs,n_val) < 1:
        raise ValueError('Training counts must be positive')
    out = Path(output_dir)
    if any((out/name).exists() for name in ('gnn_model_best.pth','gnn_model_last.pth','config.yaml','history.json')):
        raise FileExistsError(f'Choose a fresh output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.yaml').write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8')
    model = build_model(c, device)
    opt = torch.optim.Adam(model.parameters(), **tr['optimizer'])
    train_rng = torch.Generator().manual_seed(seed)
    val_rng = torch.Generator().manual_seed(seed+10000)
    val_params = [sample_params(c,val_rng) for _ in range(n_val)]
    history, best = [], float('inf')

    def episode(params):
        system = build_system(c, device, params)
        controller = build_controller(c, system, model)
        traj = rollout(system,controller,c,train_mode=True,gnn=True)
        return compute_loss(traj,c,system.edge_index)

    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for start in range(0, episodes, batch):
            actual = min(batch, episodes-start)
            opt.zero_grad(set_to_none=True)
            for _ in range(actual):
                loss = episode(sample_params(c,train_rng))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                (loss/actual).backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr['grad_clip'], error_if_nonfinite=True)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = sum(float(episode(p)) for p in val_params)/n_val
        if not torch.isfinite(torch.tensor(val)):
            raise FloatingPointError('Non-finite validation loss')
        history.append({'epoch':epoch,'train_loss':total/episodes,'validation_loss':val,'episodes':episodes})
        if val < best:
            best = val
            torch.save(model.state_dict(),out/'gnn_model_best.pth')
        torch.save(model.state_dict(),out/'gnn_model_last.pth')
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(f'Epoch {epoch}/{epochs}: train={total/episodes:.6f}, validation={val:.6f}',flush=True)
    return model
