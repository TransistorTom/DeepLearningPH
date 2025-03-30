import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

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

def train_model(model, train_data, batch_size, epochs=100, lr=0.01):
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
    if not isinstance(train_data, DataLoader):
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device.type == "cuda")
        )

    else:
        train_loader = train_data


    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = RelativeL1Loss()

    use_amp = device.type == 'cuda'
    scaler = GradScaler(device='cuda' if use_amp else 'cpu')

    loss_history = {"Epoch": [], "L1R": []}

    for epoch in range(epochs):
        total_loss = 0
        final_epoch = (epoch == epochs - 1)
        model.train()
        
        for data in train_loader:
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda' if use_amp else 'cpu'):
                out = model(data.x, data.edge_index, save_messages=final_epoch)
                loss = criterion(out, data.y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * (1 / (data.x.shape[0]-1))

        loss_e = total_loss / len(train_data) # gives average L1R loss per application of the message function    

        loss_history["Epoch"].append(epoch + 1)
        loss_history["L1R"].append(loss_e)
        
        print(f"Epoch {epoch+1:03}: MSE = {loss_e:.6f}")


    
    return model, loss_history