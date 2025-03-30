import os
import sys
import torch
import pandas as pd
import datetime

from functions.n_body_simulation import n_body_simulation
from functions.node_data_list import node_data_list 
from functions.GNN_MLP import GNN_MLP
from functions.train_model import train_model
from functions.pipeline import pipeline
from functions.datasets import GraphDataset
#repo paths for local habrok files

try:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    repo_dir = os.getcwd()

repo_root = os.path.abspath(os.path.join(repo_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if __name__ == "__main__":
    # results folder
    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(repo_root, f"results/training_habrok/job{job_id}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    dim = 2
    Ni = 3
    training = True
    testing = False

    model, train_df, test_dfs, history_loss = pipeline(
        train_iterations=250,
        test_iterations=20,
        N_train=Ni,
        N_test_list=[3, 4, 5],
        T=1000,
        dt=0.0002,
        dim=dim,
        hidden_channels=128,
        m_dim=2,
        out_channels=2,
        epochs=40,
        batch_size=4096,
        lr=0.0004,
        save_output=True,
        results_dir=results_dir,
        training=True,
        testing=False
    )