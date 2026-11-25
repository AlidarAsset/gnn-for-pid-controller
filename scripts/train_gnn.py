import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import train_gnn

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train a new GNN without overwriting legacy weights')
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--output-dir', default='models/v2')
    p.add_argument('--device', choices=['cpu','cuda'])
    a = p.parse_args()
    train_gnn(a.config,a.output_dir,a.device)
