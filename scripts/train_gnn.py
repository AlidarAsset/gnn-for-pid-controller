import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.train import train_gnn

if __name__ == "__main__":
    train_gnn()