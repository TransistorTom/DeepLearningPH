import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define the message passing class to initiate MLP's among the nodes
class GNN_MLP(MessagePassing):
    def __init__(self, n_f, m_dim, hidden_channels, out_channels, single_node = False):
        super(GNN_MLP, self).__init__(aggr='add')  # "Add" aggregation for summing over forces
        
        # initialising the MLP by creating the self.MLP attribute. 2 * in_channels to account for the fact that it may use both it's own and the other nodes features.
        self.mess_mlp = nn.Sequential(
            nn.Linear(4, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels,hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, m_dim)
        )

        self.agg_mlp = nn.Sequential(
            nn.Linear(m_dim + n_f, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

        self.single_node = single_node
        self.message_storage = []
        self.store_messages = False
        self.current_time = None
        self.current_mass = None

    def forward(self, x, edge_index, save_messages=False):
        """
        Forward calls propagate to initiate message passing for all nodes in edge_index
        """
        self.store_messages = save_messages
        # Extract the time from the last column of x (assumes time is broadcasted to all nodes)
        if save_messages:
            # self.current_time = round(x[0, -1].item(), 4) # Just take it from the first node
            self.current_time = x[0, -1].item()

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    
    def message(self, x_i, x_j, edge_index_i, edge_index_j):
        
        # Compute dx, dy (difference in positions)
        dx = x_i[:, 0] - x_j[:, 0]
        dy = x_i[:, 1] - x_j[:, 1]
        r = torch.sqrt(dx**2 + dy**2)

        # Extract mass of sender node j
        m_j = x_j[:, -2]  # assuming mass is at position -2

        # Stack the new edge features: [dx, dy, r, m_j]
        edge_features = torch.stack([dx, dy, r, m_j], dim=1)
        edge_features = edge_features.to(device)
        messages = self.mess_mlp(edge_features)

        if self.store_messages:
            for i in range(messages.size(0)):
                msg = messages[i].detach().cpu().numpy()
                record = {
                    'edge': (edge_index_i[i].item(), edge_index_j[i].item()),
                    'message_x': msg[0],
                    'message_y': msg[1],
                    'dx': dx[i].item(),
                    'dy': dy[i].item(),
                    'r': r[i].item(),
                    'mass_j': m_j[i].item(),
                    'time': self.current_time
                }
                self.message_storage.append(record)

        return messages

    def update(self, aggr_out, x=None):
        """
        Updates node features with passed messages.
        """
        if self.single_node:
            return aggr_out
        
        else:
            update_features = torch.cat([x, aggr_out], dim=1)
            return self.agg_mlp(update_features)