"""Paired evaluation on nominal or randomized physical parameters."""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import load_config, build_model, build_system, build_controller, sample_params
from src.training.rollout import rollout
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_trajectories
from seeds import set_global_seed


def main(mode='both'):
    p = argparse.ArgumentParser(description='Evaluate controllers on matched scenarios')
    p.add_argument('--config', default='models/v2/config.yaml')
    p.add_argument('--model', default='models/v2/gnn_model_best.pth')
    p.add_argument('--output-dir', default='outputs/v2')
    p.add_argument('--random-scenarios', type=int, default=0)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    if a.random_scenarios < 0: p.error('random-scenarios must be nonnegative')
    if mode != 'baseline' and not Path(a.model).is_file():
        p.error('Checkpoint not found. Train first with python scripts/train_gnn.py --output-dir models/v2, or specify --model and --config.')
    if mode == 'baseline' and a.config == 'models/v2/config.yaml' and not Path(a.config).exists():
        a.config = 'configs/default.yaml'
    c = load_config(a.config)
    set_global_seed(a.seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    model = None
    if mode != 'baseline':
        model = build_model(c,device)
        model.load_state_dict(torch.load(a.model,map_location=device,weights_only=True))
        model.eval()
        if Path(a.model).resolve().parent == Path('models').resolve() and c['gnn'].get('graph_mode') != 'legacy_directed':
            p.error('A legacy checkpoint in models/ requires configs/legacy.yaml.')
    rng = torch.Generator().manual_seed(a.seed)
    cases = [('nominal', None)] + [(f'random_{i+1}',sample_params(c,rng)) for i in range(a.random_scenarios)]
    results = []
    with torch.no_grad():
        for name,params in cases:
            system = build_system(c,device,params)
            case = {'scenario':name,'parameters':{'m':system.m.tolist(),'d':system.d.tolist(),'k':system.k.tolist(),'k_c':system.k_c.tolist()}}
            for label,net in [('baseline',None),('gnn',model)]:
                if (mode == 'baseline' and label == 'gnn') or (mode == 'gnn' and label == 'baseline'): continue
                traj = rollout(system,build_controller(c,system,net),c,gnn=(net is not None))
                case[label] = compute_metrics(traj,c['experiment']['dt'],system.edge_index,
                                              t_step=c['reference']['t_step'] if c['reference']['type']=='step' else None)
                if name == 'nominal':
                    plot_trajectories(traj,'PID' if net is None else 'PID + GNN',str(out/f'{label}_plots.png'))
            results.append(case)
    report={'config':a.config,'model':a.model if model is not None else None,'seed':a.seed,'results':results}
    if mode == 'both':
        report['mean_metrics']={label:{key:sum(vals)/len(vals) if (vals:=[r[label][key] for r in results if r[label][key] is not None]) else None for key in results[0][label]} for label in ('baseline','gnn')}
        report['settled_cases']={label:sum(r[label]['settling_time'] is not None for r in results) for label in ('baseline','gnn')}
    path=out/('comparison.json' if mode=='both' else f'{mode}_metrics.json')
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report.get('mean_metrics',results[0]),indent=2)); print(f'Saved {path}')

if __name__ == '__main__': main()
