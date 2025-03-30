from torch.utils.data import Dataset
import torch
import os

class GraphDataset(Dataset):
    def __init__(self, directory):
        self.files = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".pt")
        ])
        self.index_map = []
        for file_idx, file in enumerate(self.files):
            graphs = torch.load(file, map_location='cuda', weights_only=False)
            for graph_idx in range(len(graphs)):
                self.index_map.append((file, graph_idx))
        self.cache = {}

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, graph_idx = self.index_map[idx]
        if file_path not in self.cache:
            self.cache[file_path] = torch.load(file_path, map_location='cuda', weights_only=False)
        return self.cache[file_path][graph_idx]