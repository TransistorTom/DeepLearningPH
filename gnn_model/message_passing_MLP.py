import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# define the message passing class to initiate MLP's among the nodes
class GNN_MLP(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_MLP, self).__init__(aggr='add')  # "Add" aggregation for summing over forces
        
        # initialising the MLP by creating the self.MLP attribute. 2 * in_channels to account for the fact that it may use both it's own and the other nodes features.
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

        def forward(self, x, edge_index):
            """
            Forward calls propagate to initiate message passing for all nodes in edge_index
            """
            return self.propagate(edge_index, x=x)  # Triggers message passing
        
        def message(self, x_i, x_j):
            """
            Applying mlp to every directed edge in edge_index for [x_i, x_j]
            """
            edge_features = torch.cat([x_i, x_j], dim=1)  # Concatenating node features for edge
            return self.mlp(edge_features)  # Pass through MLP

        def update(self, aggr_out):
            """
            Updates node features with passed messages.
            """
            return aggr_out