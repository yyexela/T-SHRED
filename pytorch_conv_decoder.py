import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional
from time import time


class Decoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.tconv_0 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=3)
        self.tconv_1 = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=4, stride=3)
        self.tconv_2 = nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=5, stride=2)
        self.tconv_3 = nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=8, stride=2)
        self.tconv_4 = nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=10, stride=2)
        self.dec_dense = nn.Linear(in_features=self.latent_dim, out_features=96)
        self.layer_norm = nn.LayerNorm(96)

    def forward(self, latent_state):
        output = torch.tanh(self.dec_dense(latent_state))
        output = self.layer_norm(output)
        output = output.reshape(-1, 32, 3)  # (batch, channels, length)
        output = F.gelu(self.tconv_0(output))
        output = F.gelu(self.tconv_1(output))
        output = F.gelu(self.tconv_2(output))
        output = F.gelu(self.tconv_3(output))
        output = self.tconv_4(output)
        return output.squeeze(1)  # Remove channel dimension


class Conv_SHRED(nn.Module):

    def __init__(
        self,
        in_size,
        out_size,
        hidden_size: int,
        activation: Callable = F.relu,
    ):
        """Initialize model. 

        Parameters
        ----------
        in_size : int
            Dimensionality of the input sensor measurements.
        out_size : int
            Dimensionality of the state to reconstruct.
        hidden_size : int
            Dimensionality of the GRU hidden state.
        activation : Callable, optional
            Activation function applied between linear layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation

        self.cell1 = nn.GRUCell(
            input_size=in_size, hidden_size=hidden_size * 4
        )
        self.cell2 = nn.GRUCell(
            input_size=hidden_size * 4, hidden_size=hidden_size
        )

        self.decoder = Decoder(latent_dim=hidden_size)

    def forward(
        self,
        input_sensors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the Conv_SHRED model.

        Parameters
        ----------
        input_sensors : torch.Tensor
            Input sequence, shape (batch_size, sequence_length, in_size).

        Returns
        -------
        torch.Tensor
            Model output, shape (batch_size, out_size).
        """
        batch_size, seq_len, _ = input_sensors.shape
        device = input_sensors.device

        hidden1 = torch.zeros(batch_size, 4 * self.hidden_size, device=device)
        
        # First GRU layer
        seq_outputs = []
        for t in range(seq_len):
            hidden1 = self.cell1(input_sensors[:, t, :], hidden1)
            seq_outputs.append(hidden1)
        
        seq = torch.stack(seq_outputs, dim=1)  # (batch, seq_len, hidden*4)
        
        # Second GRU layer
        hidden2 = torch.zeros(batch_size, self.hidden_size, device=device)
        for t in range(seq_len):
            hidden2 = self.cell2(seq[:, t, :], hidden2)
        
        out = self.decoder(hidden2)
        return out

    def embed(
        self,
        input_sensors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get embeddings from the GRU layers.

        Parameters
        ----------
        input_sensors : torch.Tensor
            Input sequence, shape (batch_size, sequence_length, in_size).

        Returns
        -------
        torch.Tensor
            Embeddings, shape (batch_size, sequence_length, hidden_size).
        """
        batch_size, seq_len, _ = input_sensors.shape
        device = input_sensors.device

        hidden1 = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # First GRU layer
        seq_outputs = []
        for t in range(seq_len):
            hidden1 = self.cell1(input_sensors[:, t, :], hidden1)
            seq_outputs.append(hidden1)
        
        seq = torch.stack(seq_outputs, dim=1)
        
        # Second GRU layer
        hidden2 = torch.zeros(batch_size, self.hidden_size, device=device)
        embed_outputs = []
        for t in range(seq_len):
            hidden2 = self.cell2(seq[:, t, :], hidden2)
            embed_outputs.append(hidden2)
        
        return torch.stack(embed_outputs, dim=1)

    def decode(self, out):
        return self.decoder(out)


class GRU_SHRED(nn.Module):
    """
    A two-layer GRU-based SHRED model with dropout, implemented in PyTorch.

    Attributes
    ----------
    in_size : int
        Dimensionality of the input sensor measurements.
    out_size : int
        Dimensionality of the state to reconstruct.
    hidden_size : int
        Dimensionality of the GRU hidden state.
    cell1 : nn.GRUCell
        First recurrent layer.
    cell2 : nn.GRUCell
        Second recurrent layer.
    linear1 : nn.Linear
        First linear layer of decoder.
    linear2 : nn.Linear
        Second linear layer of decoder.
    linear3 : nn.Linear
        Output layer of decoder.
    dropout1 : nn.Dropout
        First dropout layer.
    dropout2 : nn.Dropout
        Second dropout layer.
    activation : Callable
        Activation function, default relu.

    Methods
    -------
    forward(input_sensors)
        Forward pass through network for batched sequence input.
    """

    def __init__(
        self,
        in_size,
        out_size,
        hidden_size: int,
        lin_sizes: list = [350, 400],
        dropout: float = 0.1,
        activation: Callable = F.relu,
    ):
        """Initialize model. 

        Parameters
        ----------
        in_size : int
            Dimensionality of the input sensor measurements.
        out_size : int
            Dimensionality of the state to reconstruct.
        hidden_size : int
            Dimensionality of the GRU hidden state.
        lin_sizes : list of int, optional
            Output dimensions of the first and second linear layers.
        dropout : float, optional
            Dropout probability.
        activation : Callable, optional
            Activation function applied between linear layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation

        self.cell1 = nn.GRUCell(
            input_size=in_size, hidden_size=hidden_size
        )
        self.cell2 = nn.GRUCell(
            input_size=hidden_size, hidden_size=hidden_size
        )

        self.linear1 = nn.Linear(
            in_features=hidden_size, out_features=lin_sizes[0]
        )
        self.linear2 = nn.Linear(
            in_features=lin_sizes[0], out_features=lin_sizes[1]
        )
        self.linear3 = nn.Linear(
            in_features=lin_sizes[1], out_features=out_size
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        input_sensors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the GRU_SHRED model.

        Parameters
        ----------
        input_sensors : torch.Tensor
            Input sequence, shape (batch_size, sequence_length, in_size).

        Returns
        -------
        torch.Tensor
            Model output, shape (batch_size, out_size).
        """
        batch_size, seq_len, _ = input_sensors.shape
        device = input_sensors.device

        hidden1 = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # First GRU layer
        seq_outputs = []
        for t in range(seq_len):
            hidden1 = self.cell1(input_sensors[:, t, :], hidden1)
            seq_outputs.append(hidden1)
        
        seq = torch.stack(seq_outputs, dim=1)  # (batch, seq_len, hidden)
        
        # Second GRU layer
        hidden2 = torch.zeros(batch_size, self.hidden_size, device=device)
        for t in range(seq_len):
            hidden2 = self.cell2(seq[:, t, :], hidden2)
        
        out = self.linear1(hidden2)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        out = self.linear3(out)
        return out

    def embed(
        self,
        input_sensors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get embeddings from the GRU layers.

        Parameters
        ----------
        input_sensors : torch.Tensor
            Input sequence, shape (batch_size, sequence_length, in_size).

        Returns
        -------
        torch.Tensor
            Embeddings, shape (batch_size, sequence_length, hidden_size).
        """
        batch_size, seq_len, _ = input_sensors.shape
        device = input_sensors.device

        hidden1 = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # First GRU layer
        seq_outputs = []
        for t in range(seq_len):
            hidden1 = self.cell1(input_sensors[:, t, :], hidden1)
            seq_outputs.append(hidden1)
        
        seq = torch.stack(seq_outputs, dim=1)
        
        # Second GRU layer
        hidden2 = torch.zeros(batch_size, self.hidden_size, device=device)
        embed_outputs = []
        for t in range(seq_len):
            hidden2 = self.cell2(seq[:, t, :], hidden2)
            embed_outputs.append(hidden2)
        
        return torch.stack(embed_outputs, dim=1)

    def decode(self, out):
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        out = self.linear3(out)
        return out


def compute_loss(model, x, y):
    """
    Compute MSE loss over a batch.

    x should have shape (batch, seq_len, in_size) and y should have shape
    (batch, out_size).
    """
    preds = model(x)
    loss = torch.mean((preds - y) ** 2)
    return loss


def evaluate(model, val_inputs, val_targets):
    """
    Compute validation MSE.
    """
    model.eval()
    with torch.no_grad():
        preds = model(val_inputs)
        return torch.mean((preds - val_targets) ** 2).item()


def train(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = 'cpu'
):
    """
    Training loop for GRU_SHRED model.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_inputs : torch.Tensor
        Training inputs of shape (num_samples, seq_len, in_size).
    train_targets : torch.Tensor
        Training targets of shape (num_samples, out_size).
    val_inputs : torch.Tensor
        Validation inputs.
    val_targets : torch.Tensor
        Validation targets.
    num_epochs : int, optional
        Number of training epochs. Default is 100.
    batch_size : int, optional
        Size of each mini-batch. Default is 64.
    learning_rate : float, optional
        Learning rate for optimizer. Default is 1e-3.
    device : str, optional
        Device to train on. Default is 'cpu'.

    Returns
    -------
    model : nn.Module
        Trained model.
    """
    model = model.to(device)
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)

    val_loss_list = []
    num_samples = train_inputs.shape[0]
    steps_per_epoch = num_samples // batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        t_start = time()
        model.train()
        
        # Shuffle data
        perm = torch.randperm(num_samples)
        inputs_shuffled = train_inputs[perm]
        targets_shuffled = train_targets[perm]

        epoch_loss = 0.0

        for i in range(steps_per_epoch):
            batch_x = inputs_shuffled[i * batch_size:(i + 1) * batch_size]
            batch_y = targets_shuffled[i * batch_size:(i + 1) * batch_size]
            
            optimizer.zero_grad()
            loss = compute_loss(model, batch_x, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # Evaluate validation and training losses
        print('evaluating...')
        train_loss = evaluate(model, train_inputs[::100], train_targets[::100])
        val_loss = evaluate(model, val_inputs[::20], val_targets[::20])
        t_end = time()
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.6f}")
        print(f"Average Val. Loss: {val_loss}")
        print(f"Epoch time: {t_end - t_start}")
        val_loss_list.append(val_loss)

    return model


def create_lagged_array(data, lags, subsample_factor=16, step=1):
    """
    Create time-delayed and subsampled array from input data.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        lags: Number of time lags
        subsample_factor: 16
        step: Number of timesteps between each lagged measurement
    
    Returns:
        Array of shape (n_samples - (lags-1)*step - 1, lags, n_features//subsample_factor)
    """
    n_samples, n_features = data.shape

    # Subsample to reduce features
    subsampled_data = data[:, ::subsample_factor]

    # Create all lagged versions at once using broadcasting
    max_lag = (lags - 1) * step
    indices = np.arange(n_samples - max_lag - 1)[:, None] + np.arange(lags)[None, :] * step + 1

    return subsampled_data[indices]

