import pandas as pd
import torch
import os
import sys

from functions.n_body_simulation import n_body_simulation, generate_random_positions, generate_random_velocities, generate_unique_masses
from functions.node_data_list import node_data_list 
from functions.GNN_MLP import GNN_MLP
from functions.train_model import train_model, RelativeL1Loss
from functions.datasets import GraphDataset

from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tmpdir = os.environ.get("TMPDIR", "/tmp")  # fallback to /tmp if TMPDIR not set
checkpoint_dir = os.path.join(tmpdir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

test_dir = os.path.join(checkpoint_dir, "test")
os.makedirs(test_dir, exist_ok=True)

test_dataset = GraphDataset(test_dir)
criterion = RelativeL1Loss()

def parity_flip_trajectory(traj):
    flipped_positions = [-p for p in traj["positions"]]
    flipped_velocities = [-v for v in traj["velocities"]]
    
    return {"time": traj["time"], "positions": flipped_positions, "velocities": flipped_velocities, "masses": traj["masses"].clone()}

def pipeline(train_iterations=100, test_iterations=20,
                 N_train=2, N_test_list=[2, 3, 4, 5, 6], T=500, dt=0.01, dim=2, hidden_channels=128, batch_size=128,
                 m_dim=2, out_channels=2, epochs=100, lr=0.001, save_output=True, results_dir=None, model=None, training=True, testing=True, G=1.0, single_node=False):
    
    train_messages=None
    test_messages_all=None
    
    if training:
     
        train_trajectories = [n_body_simulation(N=N_train, T=T, dt=dt, dim=dim, box_size=10, min_dist=2.5, G=1.0) for _ in range(train_iterations)]
    
        for i, traj in enumerate(train_trajectories):
            graphs = node_data_list(traj, self_loop=False, complete_graph=True)
            path = os.path.join(checkpoint_dir, f"train_graphs_{i}.pt")
            torch.save(graphs, path)
    
        if model is None:
            n_f = graphs[0].x.shape[1]  
            model = GNN_MLP(n_f=n_f, m_dim=m_dim, hidden_channels=hidden_channels,
                            out_channels=out_channels, single_node=False)
    
        model = model.to(device)
        
        train_dataset = GraphDataset(checkpoint_dir)
        
        model, loss_history = train_model(model, train_dataset, epochs=epochs, batch_size=batch_size, lr=lr)
        
        model.eval()
        model.message_storage = []

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for data in train_loader:
                data = data.to(device)
                _ = model(data.x, data.edge_index, save_messages=True)

        train_messages = pd.DataFrame(model.message_storage)

    if testing:
        if model is None:
            print("No model defined, can't test")

        else:
    # 7) Run and store test messages for each N in N_test_list
            test_messages_all = {}
            for N_test in N_test_list:
                criterion = RelativeL1Loss()
                total_loss = 0
                test_trajectories = [n_body_simulation(N=N_test, T=T, dt=dt, dim=dim, box_size=30, min_dist=7) for _ in range(test_iterations)]
                flipped_trajectories = [parity_flip_trajectory(traj) for traj in test_trajectories]
                test_trajectories.extend(flipped_trajectories) 
                test_graph_data = []
                for i, traj in test_trajectories:
                    graphs = node_data_list(traj, self_loop=False, complete_graph=True)
                    path = os.path.join(checkpoint_dir, f"train_graphs_{i}.pt")
                    torch.save(graphs, path)

                test_dataset = GraphDataset(test_dir)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                
                model = model.to(device)
                model.eval()
                model.message_storage = []
                with torch.no_grad():
                    for data in test_loader:
                        data = data.to(device)
                        out = model(data.x, data.edge_index, save_messages=True)
                        loss = criterion(out, data.y)
                        total_loss += loss.item()
                        total_samples += 1

                avg_loss = total_loss / total_samples
                print(f"average L1R loss per message for N={N_test}: {avg_loss:.6f}")

                test_messages = pd.DataFrame(model.message_storage)
                test_messages_all[N_test] = test_messages

    if save_output and results_dir is not None:
        print(f"[pipeline] Saving final outputs to {results_dir}")

        os.makedirs(results_dir, exist_ok=True)

        torch.save(model.state_dict(), f"{results_dir}/model-dim:{dim}.pt")

        if train_messages is not None:
            train_messages.to_csv(os.path.join(results_dir, "train_messages.csv"), index=False)

        if test_messages_all is not None:
            for N, df in test_messages_all.items():
                df.to_csv(os.path.join(results_dir, f"test_messages_N{N}.csv"), index=False)

        if loss_history is not None:
            pd.DataFrame(loss_history).to_csv(os.path.join(results_dir, "history_loss.csv"), index=False)

        print("[pipeline] Save complete.")

    return model, train_messages, test_messages_all, loss_history
