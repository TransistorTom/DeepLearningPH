import torch
import numpy as np
from torch_geometric.data import Data

def dynamic_dataloader(position_time_data_list, mass=1.0):
    """
    Converts a list of time-step data of the form [[r1], [r2], ..., [rn], t]
    into PyTorch Geometric data objects.

    Parameters:
        position_time_data_list (list of np.ndarray): A list where each element is 
            a numpy array containing [[r1], [r2], ..., [rn], t] for a given time step.
        mass (float): The mass of each body (default: 1.0).
    
    Returns:
        List[Data]: A list of PyTorch Geometric data objects for the GNN.
    """

    data_list = []

    for time_step_idx in range(len(position_time_data_list) - 1):  
        # Extract current time step data
        current_time_step = position_time_data_list[time_step_idx]
        next_time_step = position_time_data_list[time_step_idx + 1]

        # Extract positions (all but last entry)
        positions = np.array(current_time_step[:-1], dtype=np.float64)  # Exclude last scalar t

        # Extract the shared time value (last entry)
        t = current_time_step[-1]

        # Compute velocity using finite difference
        next_positions = np.array(next_time_step[:-1], dtype=np.float64)  # Next step positions
        delta_t = next_time_step[-1] - t  # Time step difference

        velocity = (next_positions - positions) / delta_t  # Finite difference for velocity

        # Compute velocity updates
        next_velocity = (np.array(next_time_step[:-1], dtype=np.float64) - positions) / delta_t
        velocity_update = next_velocity - velocity

        # Create PyTorch Geometric nodes for each body
        for i in range(len(positions)):  # Loop over all bodies
            pos = positions[i]
            vel = velocity[i]

            # Node features: position (r), velocity (v), and time (t)
            x = torch.tensor(np.concatenate([pos, vel, [t]]), dtype=torch.float)

            # Target: velocity update
            y = torch.tensor(velocity_update[i], dtype=torch.float)

            # Create all-to-all bidirectional edges
            edge_index = []
            for j in range(len(positions)):
                if i != j:  # Skip self-loop as the edge is already defined (self-loop is just for this example)
                    edge_index.append([i, j])  # Edge from node i to node j

            # Convert edge_index to a tensor
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # Create a PyTorch Geometric Data object
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

    return data_list
