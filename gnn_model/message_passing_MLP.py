import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# define the message passing class to initiate MLP's among the nodes
class GNN_MLP(MessagePassing):
    def __init__(self, n_f, m_dim, hidden_channels, out_channels, single_node = False):
        super(GNN_MLP, self).__init__(aggr='add')  # "Add" aggregation for summing over forces
        
        # initialising the MLP by creating the self.MLP attribute. 2 * in_channels to account for the fact that it may use both it's own and the other nodes features.
        self.mess_mlp = nn.Sequential(
            nn.Linear(2 * n_f, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, m_dim)
        )

        self.agg_mlp = nn.Sequential(
            nn.Linear(m_dim + n_f, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        """
        Forward calls propagate to initiate message passing for all nodes in edge_index
        """
        x=x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)  # Triggers message passing
    
    def message(self, x_i, x_j):
        """
        Applying mlp to every directed edge in edge_index for [x_i, x_j]
        """
        edge_features = torch.cat([x_i, x_j], dim=1)  # Concatenating node features for edge
        return self.mess_mlp(edge_features)  # Pass through MLP

    def update(self, aggr_out, x=None):
        """
        Updates node features with passed messages.
        """
        update_features = torch.cat([x, aggr_out], dim=1)
        
        return self.agg_mlp(update_features)