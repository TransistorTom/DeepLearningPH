import torch
from torch_geometric.data import Data

def node_data_list(trajectory_dict, self_loop=True, complete_graph=True):
    data_list = []

    N = trajectory_dict["masses"].numel()
    # Ensure masses has the correct shape (N, 1)
    mass_data = trajectory_dict["masses"]

    # If masses is scalar (0-dimensional), repeat it for N particles
    if mass_data.dim() == 0:  # scalar case
        mass_data = mass_data.repeat(N, 1)  # shape (N, 1)

    # If masses is already a 1D tensor of shape (N,)
    elif mass_data.dim() == 1:  # 1D tensor case
        mass_data = mass_data.unsqueeze(1)  # shape (N, 1)
    
    for i in range(len(trajectory_dict["time"]) - 1):
        time_feature = trajectory_dict["time"][i].repeat(N,1)
        position_features = trajectory_dict["positions"][i]

        x_features = torch.cat([
            trajectory_dict["positions"][i],    # (N,dim)
            trajectory_dict["velocities"][i],   # (N, dim)
            mass_data,       # (N, 1)
            time_feature    # (N, 1)
        ], dim=1)   # tensor of shape (N, 2 + 2 + 1 + 1) + (N, 6) if dim = 2

        velocity_update = trajectory_dict["velocities"][i+1] - trajectory_dict["velocities"][i]
        
        y_target = torch.tensor(velocity_update, dtype=torch.float)
        
        edge_list = []
        
        if self_loop:
            edge_list.extend([j,j] for j in range(N))
        
        if complete_graph:
            edge_list.extend([k,j] for k in range(N) for j in range(N) if k != j)           
            
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                
                

        data_list.append(Data(x=x_features, y=y_target, edge_index=edge_index))

    return data_list