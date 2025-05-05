import torch
from torch.utils.data import DataLoader
import numpy as np
import math
import torch.nn as nn
class SHRED(torch.nn.Module):
    '''SHRED model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state, hidden_size, number of LSTM layers,
    size of fully-connected layers, and dropout parameter'''
    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.0):
        super(SHRED,self).__init__()
        #Initialize lstm layer to receive time series data
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=hidden_layers, batch_first=True)
        print(input_size)
        #initialize linear layers 
        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)
        #Add dropout layers
        self.dropout = torch.nn.Dropout(dropout)

        #Add parameters
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        

    def forward(self, x):
        #Establish h_0 and c_0
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        c_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)
    
        output = self.linear3(output)

        return output


class SDN(torch.nn.Module):
    '''SDN model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state,
    size of fully-connected layers, and dropout parameter'''
    def __init__(self, input_size, output_size, l1=350, l2=400, dropout=0.0):
        super(SDN,self).__init__()
        #Add linear layers
        self.linear1 = torch.nn.Linear(input_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)
        #Add dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        #Model
        output = self.linear1(x)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)
        
        output = self.linear3(output)

        return output

# Basically training the model
def fit(model, train_dataset, valid_dataset, batch_size=64, num_epochs=4000, lr=1e-3, verbose=False, patience=5):
    '''Function for training SHRED and SDN models'''
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size) # Initializes train loader
    criterion = torch.nn.MSELoss() # Uses mean squared error loss because of linear regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Uses adam optimizer
    val_error_list = [] #Initializes validation error list
    patience_counter = 0 # Patience counter for early stopping
    best_params = model.state_dict()
    for epoch in range(1, num_epochs + 1):
        
        for k, data in enumerate(train_loader):
            model.train()
            outputs = model(data[0])
            optimizer.zero_grad()
            loss = criterion(outputs, data[1])
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(val_outputs - valid_dataset.Y)
                val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)

            if verbose == True:
                print('Training epoch ' + str(epoch))
                print('Error ' + str(val_error_list[-1]))

            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0 
                best_params = model.state_dict()
            else:
                patience_counter += 1


            if patience_counter == patience:
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list).cpu()

    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()

"""
def forecast(forecaster, reconstructor, test_dataset):
    '''Takes model and corresponding test dataset, returns tensor containing the
    inputs to generate the first forecast and then all subsequent forecasts 
    throughout the test dataset.'''
    initial_in = test_dataset.X[0:1].clone()
    vals = []
    for i in range(0, test_dataset.X.shape[1]):
        vals.append(initial_in[0, i, :].detach().cpu().clone().numpy())

    for i in range(len(test_dataset.X)):
        scaled_output = forecaster(initial_in).detach().cpu().numpy()

        vals.append(scaled_output.reshape(test_dataset.X.shape[2]))
        temp = initial_in.clone()
        initial_in[0,:-1] = temp[0,1:]
        initial_in[0,-1] = torch.tensor(scaled_output)

    device = 'cuda' if next(reconstructor.parameters()).is_cuda else 'cpu'
    forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
    reconstructions = []
    for i in range(len(forecasted_vals) - test_dataset.X.shape[1]):
        recon = reconstructor(forecasted_vals[i:i+test_dataset.X.shape[1]].reshape(1, test_dataset.X.shape[1], 
                                    test_dataset.X.shape[2])).detach().cpu().numpy()
        reconstructions.append(recon)
    reconstructions = np.array(reconstructions)
    return forecasted_vals, reconstructions
"""

"""
# Standard positional encoder according to Attention is All you Need paper
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, sequence_length: int = 5400, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos_encoding = torch.zeros(sequence_length, 1, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pos_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the positional encodings for a maximum sequence length
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(max_sequence_length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        pos_encoding = pos_encoding.unsqueeze(0)  # Shape: (1, max_sequence_length, d_model)
        self.register_buffer('pe', pos_encoding)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]  # Adjust to the sequence length of the input
        x = x + pe
        return self.dropout(x)

# U-Net decoder component
import torch
import torch.nn as nn

class UNetDecoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(d_model, 256, kernel_size=2, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer after conv1
        self.conv2 = nn.Conv1d(256, 1024, kernel_size=4, padding=1)
        self.conv_transpose1 = nn.ConvTranspose1d(in_channels=512, out_channels=1024, kernel_size=2, padding=1)
        self.conv_transpose2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=4, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer after conv2
        self.conv3 = nn.Conv1d(1024, 2048, kernel_size=2, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer after conv3
        self.conv4 = nn.Conv1d(283, 128, kernel_size=2, padding=1)
        self.conv5 = nn.Conv1d(128, 283, kernel_size=4, padding=1)
        
        self.relu = nn.ReLU()
        self.gelu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assuming x has shape [batch_size, sequence_length, d_model]
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, d_model, sequence_length]

        # Pass through the Conv1d, BatchNorm, GELU, and MaxPool layers
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        """
        x = self.gelu(self.b10(self.conv10(x)))
        x = self.pool10(x)
        x = self.gelu(self.b11(self.conv11(x)))
        x = self.pool11(x)
        """
        
        # Optionally, permute back to the original shape if needed
        x = x.permute(0, 2, 1)  # Change shape back to [batch_size, sequence_length, d_model]
        return x
    
# TimeSeries_UTransformer with U-Net decoder
class TimeSeries_UTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, sequence_length: int = 500, dropout: float = 0.2):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, sequence_length, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, activation=nn.GELU())
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.input_embedding = nn.GRU(input_size=3, hidden_size=d_model, num_layers=2)
        self.relu = nn.ReLU()
        self.unet_decoder = UNetDecoder(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply input embedding
        x, _= self.input_embedding(x)

        # Apply positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x, self._generate_square_subsequent_mask(x.size(0)))

        # Apply U-Net decoder
        x = self.unet_decoder(x)
        return x

    def _generate_square_subsequent_mask(self, sequence_length: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sequence_length, sequence_length)) == 0
        return mask

"""
def fit(model, train_dataset, valid_dataset, batch_size=500, num_epochs=4000, lr=1e-3, verbose=False, patience=5): 
    '''Function for training SHRED, SDN, and TimeSeries_UTransformer models'''
    
    # Setup DataLoader
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    # Loss function and optimizer
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)

    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        
        for batch_idx, data in enumerate(train_loader):
            inputs, targets = data
            
            # Reshape inputs and targets if necessary
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(2)
            if targets.dim() == 2:
                targets = targets.unsqueeze(2)
                
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Check shapes
            print(f"Epoch {epoch}, Batch {batch_idx}, Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")

            # Trim outputs and targets to the same size
            min_seq_len = min(outputs.shape[1], targets.shape[1])
            outputs = outputs[:, :min_seq_len, :]
            targets = targets[:, :min_seq_len, :]
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if verbose and batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_inputs = valid_dataset.X
                if val_inputs.dim() == 2:
                    val_inputs = val_inputs.unsqueeze(2)

                val_outputs = model(val_inputs)
                val_targets = valid_dataset.Y
                if val_targets.dim() == 2:
                    val_targets = val_targets.unsqueeze(2)

                # Trim validation outputs and targets to the same size
                min_seq_len = min(val_outputs.shape[1], val_targets.shape[1])
                val_outputs = val_outputs[:, :min_seq_len, :]
                val_targets = val_targets[:, :min_seq_len, :]

                val_error = criterion(val_outputs, val_targets).item()
                val_error_list.append(val_error)

                if verbose:
                    print(f'Epoch {epoch}, Validation Error: {val_error}')

                # Early stopping check
                if val_error == min(val_error_list):
                    patience_counter = 0
                    best_params = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    model.load_state_dict(best_params)
                    return torch.tensor(val_error_list).cpu()

    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()
"""
"""
import torch
from torch.utils.data import DataLoader

def fit(model, train_dataset, valid_dataset, batch_size=500, num_epochs=4000, lr=1e-3, verbose=False, patience=5):
    '''Function for training SHRED, SDN, and TimeSeries_UTransformer models'''
    if isinstance(model, (SHRED, SDN)):
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        val_error_list = []
        patience_counter = 0
        best_params = model.state_dict()
        for epoch in range(1, num_epochs + 1):
            
            for _, data in enumerate(train_loader):
                model.train()
                outputs = model(data[0])
                optimizer.zero_grad()
                loss = criterion(outputs, data[1].unsqueeze(2))
                loss.backward()
                optimizer.step()

            if epoch % 20 == 0 or epoch == 1:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(valid_dataset.X)
                    val_error = torch.linalg.norm(val_outputs - valid_dataset.Y)
                    val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                    val_error_list.append(val_error)

                if verbose:
                    print('Training epoch ' + str(epoch))
                    print('Error ' + str(val_error_list[-1]))

                if val_error == torch.min(torch.tensor(val_error_list)):
                    patience_counter = 0 
                    best_params = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter == patience:
                    model.load_state_dict(best_params)
                    return torch.tensor(val_error_list).cpu()

        model.load_state_dict(best_params)
        return torch.tensor(val_error_list).detach().cpu().numpy()

    elif isinstance(model, TimeSeries_UTransformer):
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        val_error_list = []
        patience_counter = 0
        best_params = model.state_dict()
        
        for epoch in range(1, num_epochs + 1):
            model.train()
            for k, data in enumerate(train_loader):
                if verbose:
                    print(f"Batch {k}")
                    print(f"Input shape: {data[0].shape}")
                    print(f"Target shape: {data[1].shape}")
                
                # Ensure input is 3D: (batch_size, sequence_length, features)
                inputs = data[0]
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(2)
                elif inputs.dim() == 4:
                    inputs = inputs.squeeze(2)
                
                if verbose:
                    print(f"Adjusted input shape: {inputs.shape}")
                print(inputs.shape)
                
                outputs = model(inputs)
                
                if verbose:
                    print(f"Output shape: {outputs.shape}")
                
                # Ensure targets have the right shape
                targets = data[1]
                if targets.dim() == 2:
                    targets = targets.unsqueeze(2)
                
                # Ensure outputs and targets have the same shape
                min_seq_len = min(outputs.shape[1], targets.shape[1])
                outputs = outputs[:, :min_seq_len, :283]
                targets = targets[:, :min_seq_len, :283]
                
                if verbose:
                    print(f"Final output shape: {outputs.shape}")
                    print(f"Final target shape: {targets.shape}")
                
                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            if epoch % 20 == 0 or epoch == 1:
                model.eval()
                with torch.no_grad():
                    val_inputs = valid_dataset.X
                    if val_inputs.dim() == 2:
                        val_inputs = val_inputs.unsqueeze(2)
                    elif val_inputs.dim() == 4:
                        val_inputs = val_inputs.squeeze(2)
                    val_outputs = model(val_inputs)
                    
                    val_targets = valid_dataset.Y
                    if val_targets.dim() == 2:
                        val_targets = val_targets.unsqueeze(2)
                    
                    # Ensure val_outputs and val_targets have the same shape
                    min_seq_len = min(val_outputs.shape[1], val_targets.shape[1])
                    val_outputs = val_outputs[:, :min_seq_len, :283]
                    val_targets = val_targets[:, :min_seq_len, :283]
                    
                    val_error = torch.linalg.norm(val_outputs - val_targets)
                    val_error = val_error / torch.linalg.norm(val_targets)
                    val_error_list.append(val_error)
                    print(val_error)

                if verbose:
                    print('Training epoch ' + str(epoch))
                    print('Error ' + str(val_error_list[-1]))

                if val_error == torch.min(torch.tensor(val_error_list)):
                    patience_counter = 0 
                    best_params = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter == patience:
                    model.load_state_dict(best_params)
                    return torch.tensor(val_error_list).cpu()

        model.load_state_dict(best_params)
        return torch.tensor(val_error_list).detach().cpu().numpy()

    else:
        raise ValueError("Invalid model type")
"""
"""
def forecast(forecaster, reconstructor, test_dataset):
    '''Takes model and corresponding test dataset, returns tensor containing the
    inputs to generate the first forecast and then all subsequent forecasts 
    throughout the test dataset.'''
    if isinstance(forecaster, (SHRED, SDN)):
        initial_in = test_dataset.X[0:1].clone()
        vals = []
        for i in range(0, test_dataset.X.shape[1]):
            vals.append(initial_in[0, i, :].detach().cpu().clone().numpy())

        for i in range(len(test_dataset.X)):
            scaled_output = forecaster(initial_in).detach().cpu().numpy()

            vals.append(scaled_output.reshape(test_dataset.X.shape[2]))
            temp = initial_in.clone()
            initial_in[0,:-1] = temp[0,1:]
            initial_in[0,-1] = torch.tensor(scaled_output)

        device = 'cuda' if next(reconstructor.parameters()).is_cuda else 'cpu'
        forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
        reconstructions = []
        for i in range(len(forecasted_vals) - test_dataset.X.shape[1]):
            recon = reconstructor(forecasted_vals[i:i+test_dataset.X.shape[1]].reshape(1, test_dataset.X.shape[1], 
                                        test_dataset.X.shape[2])).detach().cpu().numpy()
            reconstructions.append(recon)
        reconstructions = np.array(reconstructions)
        return forecasted_vals, reconstructions

    elif isinstance(forecaster, TimeSeries_UTransformer):
        initial_in = test_dataset.X[0:1].clone()
        vals = []
        for i in range(0, test_dataset.X.shape[1]):
            vals.append(initial_in[0, i, :].detach().cpu().clone().numpy())

        for i in range(len(test_dataset.X)):
            scaled_output = forecaster(initial_in).detach().cpu().numpy()

            vals.append(scaled_output.reshape(test_dataset.X.shape[2]))
            temp = initial_in.clone()
            initial_in[0,:-1] = temp[0,1:]
            initial_in[0,-1] = torch.tensor(scaled_output)

        device = 'cuda' if next(reconstructor.parameters()).is_cuda else 'cpu'
        forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
        reconstructions = []
        for i in range(len(forecasted_vals) - test_dataset.X.shape[1]):
            recon = reconstructor(forecasted_vals[i:i+test_dataset.X.shape[1]].reshape(1, test_dataset.X.shape[1], 
                                        test_dataset.X.shape[2])).detach().cpu().numpy()
            reconstructions.append(recon)
        reconstructions = np.array(reconstructions)
        return forecasted_vals, reconstructions

    else:
        raise ValueError("Invalid model type. Only SHRED, SDN, and TimeSeries_UTransformer models are supported.")
"""
