import numpy as np
import torch.nn as nn

class MLP(nn.Module):
    """
    Creates a simple linear MLP AutoEncoder.

    `in_dim`: input and output dimension   
    `bottleneck_dim`: dimension at bottleneck  
    `width`: width of model   
    `device`: which device to use   
    """
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, dropout: float, device: str = 'cpu'):
        super(MLP, self).__init__()
        # Class variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Model layer sizes
        sizes = list()
        sizes.extend(np.logspace(np.log2(in_dim), np.log2(out_dim), base=2, num=n_layers+1, dtype=int).tolist())
        sizes[0] = self.in_dim
        sizes[-1] = self.out_dim

        # Define model layers
        self.layers = []
        for idx in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[idx], sizes[idx+1]))
            if idx != (len(sizes)-2):
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(self, x):
        sindy_loss = x.get("sindy_loss", None)
        x = x["final_hidden_state"]
        out = self.model(x)
        out = self.dropout(out)
        return {
            "output": out,
            "sindy_loss": sindy_loss
        }

class CNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, dropout: float, device: str = 'cpu'):
        super().__init__()
        # Class variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Model layer sizes
        sizes = list()
        sizes.extend(np.logspace(np.log2(in_dim), np.log2(out_dim), base=2, num=n_layers+1, dtype=int).tolist())
        sizes[0] = self.in_dim
        sizes[-1] = self.out_dim

        # Define model layers
        self.layers = []
        for idx in range(len(sizes)-1):
            self.layers.append(nn.Conv1d(sizes[idx], sizes[idx+1], kernel_size=3, padding=1))
            if idx != (len(sizes)-2):
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(self, x):
        sindy_loss = x.get("sindy_loss", None)
        x = x["final_hidden_state"] # 128 x 8 
        x = x.unsqueeze(-1)
        out = self.model(x) 
        out = self.dropout(out)
        out = out.squeeze(-1)
        return {
            "output": out,
            "sindy_loss": sindy_loss
        }