import os
import sys
import torch
import pandas as pd
from functions.n_body_simulation import n_body_simulation, generate_random_positions, generate_random_velocities, generate_unique_masses
from functions.node_data_list import node_data_list 
from functions.GNN_MLP import GNN_MLP
from functions.train_model import train_model
from functions.pipeline import pipeline
import datetime

repo_root = os.getwcd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


if __name__ == "__main__":
    # Creating folders for results on habrok
    repo_dir = os.getwcd()
    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(repo_dir, f"results/job{job_id}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    dim = 2
    N = 3
    # Running full model for training
    model, train_df, test_dfs, history_loss = pipeline(
        train_iterations=10,
        test_iterations=2,
        N_train=N,
        N_test_list=[3, 4, 5],
        T=100,
        dt=0.001,
        dim=dim,
        hidden_channels=128,
        m_dim=2,
        out_channels=2,
        epochs=10,
        lr=0.001,
        save=False,  # we’ll do the saving in this file manually
        training=True,
        testing=True
    )
    
    
    # Save results to /dlp/results
    train_df.to_csv(f"{results_dir}/train_messages.csv")

    for N, df in test_dfs.items():
        df.to_csv(f"{results_dir}/test_messages_N{N}.csv")

    history_loss_df = pd.DataFrame(history_loss)
    history_loss_df.to_csv(f"{results_dir}/history_loss.csv")

    torch.save(model.state_dict(), f"{results_dir}/model-dim:{dim}-job:{job_id}-N:{N}-.pt")
