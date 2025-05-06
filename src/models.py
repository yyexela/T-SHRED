import torch
from torch.utils.data import DataLoader
import numpy as np
import math
import torch.nn as nn

def get_model(args):
    #UTransformer = TimeSeries_UTransformer(d_model=args.d_model, n_heads=args.n_heads, sequence_length=args.window_length, dropout=args.dropout).to(args.device)

    pass

    return 

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
    def __init__(self, d_model: int, n_heads: int, sequence_length: int = 500, dropout: float = 0.2):
        super().__init__()
        # TODO: Positional encoder
        #self.pos_encoder = PositionalEncoding(d_model, sequence_length, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128, dropout=dropout, activation=nn.GELU())
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.input_embedding = nn.GRU(input_size=3, hidden_size=d_model, num_layers=2)
        self.relu = nn.ReLU()
        self.unet_decoder = UNetDecoder(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply input embedding
        x, _= self.input_embedding(x)

        # Apply positional encoding
        #x = self.pos_encoder(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x, self._generate_square_subsequent_mask(x.size(0)))

        # Apply U-Net decoder
        x = self.unet_decoder(x)
        return x

    def _generate_square_subsequent_mask(self, sequence_length: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sequence_length, sequence_length)) == 0
        return mask

class UNET(torch.nn.Module):
    def __init__(self, dropout: float = 0.1, conv1: int = 256, conv2: int = 1024, d_model: int = 10, output_size: int = 1):
        super().__init__()
        self.c1 = conv1
        self.c2 = conv2
        self.d_model = d_model
        self.dropout = dropout
        self.output_size = output_size

        self.initialize(self.d_model, self.output_size)

    def initialize(self, d_model, output_size):
        """
        Initialize the SDNDecoder with input and output sizes.

        Parameters:
        -----------
        d_model : int
            Size of the input features.
        output_size : int
            Size of the output features.
        """
        # self.dropoutLayer = nn.Dropout(self.dropout)
        self.conv1 = nn.Conv1d(d_model, self.c1, kernel_size=2, padding=1)
        # self.pool1 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer after conv1
        self.conv2 = nn.Conv1d(self.c1, self.c2, kernel_size=4, padding=1)
        # self.conv_transpose1 = nn.ConvTranspose1d(in_channels=512, out_channels=1024, kernel_size=2, padding=1)
        # self.conv_transpose2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=4, padding=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer after conv2
        self.conv3 = nn.Conv1d(self.c2, output_size, kernel_size=2, padding=1)
        # self.pool3 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer after conv3
        # self.conv4 = nn.Conv1d(283, 128, kernel_size=2, padding=1)
        # self.conv5 = nn.Conv1d(128, 283, kernel_size=4, padding=1)
        # self.relu = nn.ReLU()
        self.gelu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x["sequence_output"] # TODO: why? Why not final_hidden_state
        # Assuming x has shape [batch_size, sequence_length, d_model]
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, d_model, sequence_length]
        # Pass through the Conv1d, BatchNorm, GELU, and MaxPool layers
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        # Optionally, permute back to the original shape if needed
        x = x.permute(0, 2, 1)  # Change shape back to [batch_size, sequence_length, d_model]
        x = torch.mean(x, dim=1)
        return x

class TRANSFORMER_SINDY(torch.nn.Module):
    def __init__(self, d_model: int = 128, dropout: float = 0.05,
                 poly_order: int = 1, include_sine: bool = False,
                 num_sindy_layers: int = 2, dim_feedforward: int = 128,
                 window_length: int = 500,
                 hidden_size: int = 64,
                 activation=nn.GELU()): # Added SINDy params
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.num_sindy_layers = num_sindy_layers
        self.window_length = window_length

        # Store config for layers
        self.layer_config = {
            "d_model": d_model,
            "hidden_size": hidden_size,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": activation,
            "poly_order": poly_order,
            "include_sine": include_sine
        }

        self.output_size = d_model

        self.initialize(self.d_model, self.window_length)

    def initialize(self, d_model:int, window_length:int, **kwargs):
        #self.pos_encoder = PositionalEncoding(self.d_model,
                                              #window_length, # Use actual sequence length
                                              #self.dropout)

        # Input GRU embedding
        self.input_embedding = nn.GRU(input_size=self.d_model,
                                      hidden_size=self.hidden_size,
                                      num_layers=2, # Or adjust as needed
                                      batch_first=True,
                                      dropout=self.dropout if self.num_sindy_layers > 1 else 0.0) # Add dropout if multiple layers

        # Create the stack of SINDy layers (units) attention
        layers = [SINDyLayer(**self.layer_config) for _ in range(self.num_sindy_layers)]
        self.sindy_encoder = nn.Sequential(*layers) # Use nn.Sequential for simplicity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Dictionary containing sequence output and final hidden state
        """
        # apply input embedding
        # GRU output: (output_seq, final_hidden_state)
        x, _ = self.input_embedding(x) # Shape: (batch_size, seq_len, d_model)

        # perform positional encoding
        #x = self.pos_encoder(x) # Shape: (batch_size, seq_len, d_model)

        # apply SINDy encoder/attention layers
        x = self.sindy_encoder(x) # Shape: (batch_size, seq_len, d_model)

        # Return dictionary format as before
        return {
            "sequence_output": x, # [batch_size, sequence_length, d_model]
            "final_hidden_state": x[:,-1,:] # last timestep hidden state [batch_size, d_model]
        }

    # Keep the mask generation function if needed elsewhere, but it's not used by SINDyLayer directly
    def _generate_square_subsequent_mask(self, sequence_length: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sequence_length, sequence_length, device=device), diagonal=1).bool()
        return mask

    @property
    def model_name(self):
        return f"SINDyTransformer_L{self.num_sindy_layers}_P{self.poly_order}_S{self.include_sine}"

    # not working right now: Method to apply thresholding to all SINDy layers
    def threshold_all_layers(self, threshold):
        print(f"Applying threshold {threshold} to all SINDy layers...")
        for i, layer in enumerate(self.sindy_encoder):
            if isinstance(layer, SINDyLayer):
                print(f"Layer {i}:")
                layer.thresholding(threshold)

class SINDyLayer(torch.nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, activation: nn.Module,
                 poly_order: int, include_sine: bool, sindy_threshold: float = 0.01, hidden_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = calculate_library_dim(hidden_size, poly_order, include_sine)
        self.sindy_threshold_val = sindy_threshold # For thresholding (sparsify)

        # SINDy coefficients (nn.parameter which is learnable)
        self.coefficients = nn.Parameter(torch.Tensor(self.library_dim, self.hidden_size))
        nn.init.xavier_uniform_(self.coefficients) # Initialize coefficients

        # Coefficient mask (not learnable, used for thresholding)
        self.register_buffer('coefficient_mask', torch.ones(self.library_dim, self.hidden_size), persistent=False)

        # Standard Transformer components
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_size)
        self.activation = activation


    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Input tensor of shape (batch_size, sequence_length, hidden_size)
        Returns:
            Output tensor of shape (batch_size, sequence_length, hidden_size)
        """
        batch_size, seq_len, hidden_size = src.shape
        src_residual = src

        ############################# SINDy unit #############################
        # Reshape src for sindy_library: (batch_size * seq_len, hidden_size)
        src_flat = src.reshape(-1, hidden_size)

        # Calculate SINDy library features
        library_Theta = sindy_library_torch(src_flat, self.hidden_size, self.poly_order, self.include_sine)

        # Calculate SINDy update (use masked coefficients)
        # effective_coefficients = self.coefficients * self.coefficient_mask.to(self.coefficients.device) # Ensure mask is on correct device
        ############################## Simplified SINDy update (without mask) #############################
        sindy_update = library_Theta @ self.coefficients


        # Reshape update back to (batch_size, seq_len, hidden_size)
        sindy_update = sindy_update.reshape(batch_size, seq_len, hidden_size)
        ############################## End SINDy unit #############################

        # Apply dropout and skip connection + normalization
        src = self.norm1(src_residual + self.dropout1(sindy_update))

        ############################## MLP Block##############################
        src_residual = src
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        ######################################################################

        # dropout and skip connection + norm
        src = self.norm2(src_residual + self.dropout3(ff_output))

        return src

    # may cause numerical instability, temp optional
    def thresholding(self, threshold=None):
        if threshold is None:
            threshold = self.sindy_threshold_val
        with torch.no_grad():
            mask = torch.abs(self.coefficients.data) > threshold
            self.coefficients.data *= mask
            self.coefficient_mask.copy_(mask.float()) # Update buffer if needed for inspection
            print(f"SINDyLayer: Applied threshold {threshold}. Non-zero coeffs: {mask.sum().item()}/{mask.numel()}")

def calculate_library_dim(latent_dim, poly_order, include_sine):
    dim = 1 # Constant term
    # Polynomial terms (using combinations with replacement)
    current_dim = latent_dim
    dim += current_dim
    if poly_order > 1:
        current_dim = current_dim * (latent_dim + 1) // 2
        dim += current_dim
    if poly_order > 2:
        current_dim = current_dim * (latent_dim + 2) // 3
        dim += current_dim
    if poly_order > 3:
        current_dim = current_dim * (latent_dim + 3) // 4
        dim += current_dim
    if poly_order > 4:
        current_dim = current_dim * (latent_dim + 4) // 5
        dim += current_dim

    if include_sine:
        dim += latent_dim
    return dim

def sindy_library_torch(z, latent_dim, poly_order, include_sine=False):
    device = z.device # Get device from input tensor
    library = [torch.ones(z.shape[0], device=device)]

    # Add polynomials up to poly_order
    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(z[:,i] * z[:,j]) # Use element-wise multiplication

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i] * z[:,j] * z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p] * z[:,q])

    # Add sine terms if requested
    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, axis=1)

class MixedModel(torch.nn.Module):
    """
    Main function to generate mixes of models
    """
    def __init__(self, args): # Added SINDy params
        super().__init__()

        if args.encoder == "sindy_transformer":
            self.encoder = TRANSFORMER_SINDY(
                d_model=args.d_model,
                dropout=args.dropout,
                poly_order=args.poly_order,
                include_sine=args.include_sine,
                num_sindy_layers=args.num_sindy_layers,
                dim_feedforward=args.dim_feedforward,
                window_length=args.window_length,
                hidden_size=args.hidden_size,
                activation=nn.GELU()
            )
        else:
            raise NotImplementedError
        
        if args.decoder == "unet":
            self.decoder = UNET(
                dropout = args.dropout,
                conv1 = args.conv1,
                conv2 = args.conv2,
                d_model = args.hidden_size,
                output_size = args.output_size,
            )
        else:
            raise NotImplementedError

        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Input tensor of shape (batch_size, sequence_length, n_sensors, d_model)
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_sensors, d_model)
        """

        src_encoded = self.encoder(src)
        src_decoded = self.decoder(src_encoded)

        return src_decoded
