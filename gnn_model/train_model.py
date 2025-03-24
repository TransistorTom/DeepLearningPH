import numpy as np
import torch
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def train_model(model, train_data, epochs=100, lr=0.01):
    """
    Train a GNN model using the provided training and validation loaders.
    
    Parameters:
    -----------
    model : torch.nn.Module
        A PyTorch module representing the GNN model.
    train_loader : torch_geometric.loader.DataLoader
        A DataLoader object containing training data.
    val_loader : torch_geometric.loader.DataLoader
        A DataLoader object containing validation data.
    epochs : int
        Number of epochs to train the model.
    lr : float
        Learning rate for the optimizer.
    
    Returns:
    --------
    model : torch.nn.Module
        The trained model.
    """
    
    # Only convert to DataLoader if not already in DataLoader format
    if isinstance(train_data, DataLoader):
        train_data = DataLoader(train_data, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        final_epoch = (epoch == epochs - 1)
        for data in train_data:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, save_messages=final_epoch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return model