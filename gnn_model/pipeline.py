import pandas as pd
import torch
from simulations.n_body_trajectory import n_body_simulation, generate_random_positions, generate_random_velocities, generate_unique_masses
from gnn_model.node_data_list import node_data_list 
from gnn_model.GNN_MLP import GNN_MLP
from gnn_model.train_model import train_model

def pipeline(train_iterations=100, test_iterations=20,
                 N_train=2, N_test_list=[2, 3, 4, 5, 6], T=500, dt=0.01, dim=2, hidden_channels=128,
                 m_dim=2, out_channels=2, epochs=100, lr=0.001, save=False, model=None, training=True, testing=True, G=0.5, single_node=False):
    
    train_messages=None
    test_messages_all=None
    
    if training:
        # 1) Run training simulations with N_train
        train_trajectories = [n_body_simulation(N=N_train, T=T, dt=dt, dim=dim, box_size=10, min_dist=2.5, G=0.5) for _ in range(train_iterations)]
    
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

        # 5) Train model
        model = train_model(model, train_data, epochs=epochs, lr=lr)
        
        if save:
            torch.save(model.state_dict(), "trained_gnn_model.pt")
        
        model.message_storage = []
        for data in train_graph_data:
            _ = model(data.x, data.edge_index, save_messages=True)

        # 6) Extract training messages
        train_messages = pd.DataFrame(model.message_storage)
        train_messages[['pos_i_x', 'pos_i_y']] = pd.DataFrame(train_messages['pos_i'].tolist())
        train_messages[['pos_j_x', 'pos_j_y']] = pd.DataFrame(train_messages['pos_j'].tolist())
        train_messages[['message_x', 'message_y']] = pd.DataFrame(train_messages['message'].tolist())
        train_messages = train_messages.drop(columns=['pos_i', 'pos_j', 'message', 'edge'])


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
                test_graph_data = []
                for traj in test_trajectories:
                    graphs = node_data_list(traj, self_loop=False, complete_graph=True)
                    test_graph_data.extend(graphs)

                model.message_storage = []
                for data in test_graph_data:
                    out = model(data.x, data.edge_index, save_messages=True)
                    loss = criterion(out, data.y)
                    total_loss +=loss.item()
                print(len(test_graph_data))    

                avg_loss = total_loss/len(test_graph_data)    

                print(f"average loss per/over timestep N={N_test}:   {avg_loss}")

                test_messages = pd.DataFrame(model.message_storage)
                test_messages[['pos_i_x', 'pos_i_y']] = pd.DataFrame(test_messages['pos_i'].tolist())
                test_messages[['pos_j_x', 'pos_j_y']] = pd.DataFrame(test_messages['pos_j'].tolist())
                test_messages[['message_x', 'message_y']] = pd.DataFrame(test_messages['message'].tolist())
                test_messages = test_messages.drop(columns=['pos_i', 'pos_j', 'message', 'edge'])

                test_messages_all[N_test] = test_messages

    return model, train_messages, test_messages_all
