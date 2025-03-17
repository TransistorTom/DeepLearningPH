import torch
from torch_geometric.data import Data

def node_data_list(trajectory_dict, self_loop=True, complete_graph=True):
    """
    Converts a trajectory dictionary into a list of PyTorch Geometric `Data` objects 
    representing graph-structured data for a node-based learning task.

    Parameters:
    -----------
    trajectory_dict : dict
        A dictionary containing simulation data with the following keys:
        - "masses" (torch.Tensor): Mass values for the nodes (particles).
        - "time" (torch.Tensor): Time steps in the trajectory.
        - "positions" (list of torch.Tensor): List of position tensors at each time step.
        - "velocities" (list of torch.Tensor): List of velocity tensors at each time step.

    self_loop : bool, optional (default=True)
        If True, self-loops (edges from a node to itself) are included in the graph.

    complete_graph : bool, optional (default=True)
        If True, a fully connected graph is created where each node is connected to every other node.

    Returns:
    --------
    data_list : list of torch_geometric.data.Data
        A list of `Data` objects, each representing a graph at a given time step.
        Each `Data` object contains:
        - `x` (torch.Tensor): Node features with shape (N, num_features), where `num_features` includes
          position, velocity, mass, and time.
        - `y` (torch.Tensor): Target values representing velocity updates.
        - `edge_index` (torch.Tensor): Graph connectivity in COO format with shape (2, num_edges).

    Notes:
    ------
    - The function assumes a constant number of nodes (N) throughout the trajectory.
    - If `masses` is a scalar, it is broadcasted to all nodes.
    - Edge indices are stored in COO format (two-row tensor).

    Example:
    --------
    >>> trajectory_dict = {
    ...     "masses": torch.tensor(1.0),
    ...     "time": torch.arange(3),
    ...     "positions": [torch.rand(5, 2) for _ in range(3)],
    ...     "velocities": [torch.rand(5, 2) for _ in range(3)]
    ... }
    >>> graphs = node_data_list(trajectory_dict)
    >>> print(graphs[0])
    Data(x=[5, 6], y=[5, 2], edge_index=[2, 25])
    """

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
        acceleration = velocity_update / (trajectory_dict["time"][i+1] - trajectory_dict["time"][i])

        y_target = torch.tensor(acceleration, dtype=torch.float32)
        
        edge_list = []
        
        if self_loop:
            edge_list.extend([j,j] for j in range(N))
        
        if complete_graph:
            edge_list.extend([k,j] for k in range(N) for j in range(N) if k != j)           
            
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                
                

        data_list.append(Data(x=x_features.float(), y=y_target, edge_index=edge_index))

    return data_list
