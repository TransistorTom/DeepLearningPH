import pandas as pd
import torch
from functions.n_body_simulation import n_body_simulation, generate_random_positions, generate_random_velocities, generate_unique_masses
from functions.node_data_list import node_data_list 
from functions.GNN_MLP import GNN_MLP
from functions.train_model import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parity_flip_trajectory(traj):
    flipped_positions = [-p for p in traj["positions"]]
    flipped_velocities = [-v for v in traj["velocities"]]
    
    return {"time": traj["time"], "positions": flipped_positions, "velocities": flipped_velocities, "masses": traj["masses"].clone()}

def pipeline(train_iterations=100, test_iterations=20,
                 N_train=2, N_test_list=[2, 3, 4, 5, 6], T=500, dt=0.01, dim=2, hidden_channels=128,
                 m_dim=2, out_channels=2, epochs=100, lr=0.001, save=False, model=None, training=True, testing=True, G=1.0, single_node=False):
    
    train_messages=None
    test_messages_all=None
    
    if training:
        # 1) Run training simulations with N_train
        train_trajectories = [n_body_simulation(N=N_train, T=T, dt=dt, dim=dim, box_size=10, min_dist=2.5, G=1.0) for _ in range(train_iterations)]
    
        train_graph_data = []
        for traj in train_trajectories:
            graphs = node_data_list(traj, self_loop=False, complete_graph=True)
            train_graph_data.extend(graphs)
        
        # train_data = [all_train_graph_data[i] for i in train_indices]
        train_data = [train_graph_data[i] for i in range(len(train_graph_data))]

        # 4) Initialize model
        input_dim = train_graph_data[0].x.shape[1]
        if model is None:
            model = GNN_MLP(n_f=input_dim, m_dim=m_dim, hidden_channels=hidden_channels,
                        out_channels=out_channels, single_node=False)
        
        model = model.to(device)
        # 5) Train model
        model, loss_history = train_model(model, train_data, epochs=epochs, lr=lr)
        
        if save:
            torch.save(model.state_dict(), "trained_gnn_model.pt")
        
        model.message_storage = []
        for data in train_graph_data:
            _ = model(data.x, data.edge_index, save_messages=True)

        # 6) Extract training messages
        train_messages = pd.DataFrame(model.message_storage)

    if testing:

        if model is None:
            print("No model defined, can't test")

        else:
    # 7) Run and store test messages for each N in N_test_list
            test_messages_all = {}
            for N_test in N_test_list:
                criterion = torch.nn.MSELoss()
                total_loss = 0
                test_trajectories = [n_body_simulation(N=N_test, T=T, dt=dt, dim=dim, box_size=30, min_dist=7) for _ in range(test_iterations)]
                flipped_trajectories = [parity_flip_trajectory(traj) for traj in test_trajectories]
                test_trajectories.extend(flipped_trajectories) 
                test_graph_data = []
                for traj in test_trajectories:
                    graphs = node_data_list(traj, self_loop=False, complete_graph=True)
                    test_graph_data.extend(graphs)

                model.message_storage = []
                model = model.to(device)
                for data in test_graph_data:
                    out = model(data.x, data.edge_index, save_messages=True)
                    loss = criterion(out, data.y)
                    total_loss +=loss.item()

                print(len(test_graph_data))    

                avg_loss = total_loss/len(test_graph_data)    

                print(f"average loss per/over timestep N={N_test}:   {avg_loss}")

                test_messages = pd.DataFrame(model.message_storage)

                test_messages_all[N_test] = test_messages

    return model, train_messages, test_messages_all, loss_history
