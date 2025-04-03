import numpy as np
import torch
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn

class RelativeL1Loss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        rel_error = torch.abs(pred - target) / (torch.abs(target) + self.eps)
        return rel_error.mean()

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
        train_data = DataLoader(train_data, batch_size=16, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = RelativeL1Loss()
    
    for epoch in range(epochs):
        total_loss = 0
        final_epoch = (epoch == epochs - 1)
        relative_errors = []
        for data in train_data:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, save_messages=final_epoch) #no longer needed per se, decided to 
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            eps = 1e-8  # to avoid division by zero
            rel_err = torch.abs(out - data.y) / (torch.abs(data.y) + eps)
            mean_rel_err = rel_err.mean().item()
            relative_errors.append(mean_rel_err)
            # relative_errors.append(relative_error.item())
        
        avg_loss = total_loss / len(train_data)
        avg_rel_err = np.mean(relative_errors)


        print(f"Epoch {epoch+1:03}: MSE = {avg_loss:.6f}, Mean Relative Error = {avg_rel_err:.6f}")


    
    return model