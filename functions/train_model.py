import numpy as np
import torch
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class RelativeL1Loss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        rel_error = torch.abs(pred - target) / (torch.abs(target) + self.eps)
        return rel_error.mean()

def train_model(model, train_data, epochs=100, lr=0.01, batch_size = 128):
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
    if isinstance(train_data, list):
        train_loader = DataLoader(train_data, batch_size, shuffle=False)
    else: 
        train_loader = train_data
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = RelativeL1Loss()

    loss_history = {"Epoch": [], "L1R": []}

    for epoch in range(epochs):
        total_loss = 0
        final_epoch = (epoch == epochs - 1)
        relative_errors = []

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, save_messages=final_epoch) #no longer needed per se, decided to 
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (1 / (data.x.shape[0]-1))

        loss_e = total_loss / len(train_data) # gives average L1R loss per application of the message function    

        loss_history["Epoch"].append(epoch + 1)
        loss_history["L1R"].append(loss_e)
        
        print(f"Epoch {epoch+1:03}: MSE = {loss_e:.6f}")


    
    return model, loss_history