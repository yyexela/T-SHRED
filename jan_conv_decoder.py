import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from typing import Callable, Optional

from time import time
jax.config.update("jax_enable_x64", True)

class Decoder(eqx.Module):

    tconv_0: eqx.nn.Conv1d
    tconv_1: eqx.nn.Conv1d
    tconv_2: eqx.nn.Conv1d
    tconv_3: eqx.nn.Conv1d
    tconv_4: eqx.nn.Conv1d
    dec_dense: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm
    latent_dim: int

    def __init__(self, latent_dim, *, key=jax.random.key(0)):

        self.latent_dim = latent_dim
        keys = jax.random.split(key, 7)
        self.tconv_0 = eqx.nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=3, key=keys[0])
        self.tconv_1 = eqx.nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=4, stride=3, key=keys[1])
        self.tconv_2 = eqx.nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=5, stride=2, key=keys[2])
        self.tconv_3 = eqx.nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=8, stride=2, key=keys[3])
        self.tconv_4 = eqx.nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=10, stride=2, key=keys[4])
        self.dec_dense = eqx.nn.Linear(in_features=self.latent_dim, out_features=96, key=keys[6])
        self.layer_norm = eqx.nn.LayerNorm(96)

    def __call__(self, latent_state):
        output = jnp.tanh(self.dec_dense(latent_state))
        output = self.layer_norm(output)
        output = output.reshape(32, 3)
        output = jax.nn.gelu(self.tconv_0(output))
        output = jax.nn.gelu(self.tconv_1(output))
        output = jax.nn.gelu(self.tconv_2(output))
        output = jax.nn.gelu(self.tconv_3(output))
        output = self.tconv_4(output)
        return jnp.squeeze(output)




class Conv_SHRED(eqx.Module):

    hidden_size: int
    in_size: int
    out_size: int
    cell1: eqx.nn.GRUCell
    cell2: eqx.nn.GRUCell
    decoder: eqx.Module
    activation: Callable
    inference: bool

    def __init__(
        self,
        in_size,
        out_size,
        hidden_size: int,
        activation: Callable = jax.nn.relu,
        inference: bool = False,
        *,
        key
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
        inference : bool, optional
            Whether the model is in inference mode (dropout disabled).
        key : jax.random.PRNGKey
            Random key used to initialize weights and dropout layers.
        """
        r1key, r2key, l1key, l2key, l3key = jax.random.split(key, 5)
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation

        self.cell1 = eqx.nn.GRUCell(
            input_size=in_size, hidden_size=hidden_size * 4, key=r1key
        )
        self.cell2 = eqx.nn.GRUCell(
            input_size=hidden_size * 4, hidden_size=hidden_size, key=r2key
        )

        self.decoder = Decoder(latent_dim=hidden_size,key=l1key)
        self.inference = inference


    def __call__(
        self,
        input_sensors: jnp.array,
        key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Forward pass through the GRU_SHRED model for a single, unbatched seq.

        Parameters
        ----------
        input_sensors : jnp.ndarray
            Input sequence, shape (sequence_length, in_size).
        key : jax.random.PRNGKey, optional
            PRNG key for dropout, (required if not in inference mode).

        Returns
        -------
        jnp.ndarray
            Model output, shape (out_size,).
        """
        if not self.inference:
            key1, key2 = jax.random.split(key)
        else:
            key1, key2 = None, None

        hidden = jnp.zeros(4 * self.hidden_size)

        def f1(carry, inp):
            next_state = self.cell1(inp, carry)
            return next_state, next_state

        _, seq = jax.lax.scan(f1, hidden, input_sensors)

        hidden = jnp.zeros(self.hidden_size)
        def f2(carry, inp):
            next_state = self.cell2(inp, carry)
            return next_state, next_state

        out, _ = jax.lax.scan(f2, hidden, seq)

        out = self.decoder(out)
        return out
    
    def embed(
        self,
        input_sensors: jnp.array,
        key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Forward pass through the GRU_SHRED model for a single, unbatched seq.

        Parameters
        ----------
        input_sensors : jnp.ndarray
            Input sequence, shape (sequence_length, in_size).
        key : jax.random.PRNGKey, optional
            PRNG key for dropout, (required if not in inference mode).

        Returns
        -------
        jnp.ndarray
            Model output, shape (out_size,).
        """
        if not self.inference:
            key1, key2 = jax.random.split(key)
        else:
            key1, key2 = None, None

        hidden = jnp.zeros(self.hidden_size)

        def f1(carry, inp):
            next_state = self.cell1(inp, carry)
            return next_state, next_state

        _, seq = jax.lax.scan(f1, hidden, input_sensors)

        def f2(carry, inp):
            next_state = self.cell2(inp, carry)
            return next_state, next_state

        _, seq = jax.lax.scan(f2, hidden, seq)

        return seq

    def decode(self, out, key=None):
        return self.decoder(out)

class GRU_SHRED(eqx.Module):
    """
    A two-layer GRU-based SHRED model with dropout, implemented in Equinox and JAX.

    Attributes
    ----------
    in_size : int
        Dimensionality of the input sensor measurements.
    out_size : int
        Dimensionality of the state to reconstruct.
    hidden_size : int
        Dimensionality of the GRU hidden state.
    cell1 : eqx.nn.GRUCell
        First recurrent layer.
    cell2 : eqx.nn.GRUCell
        Second recurrent layer. Equinox does not provide the simple flag to
        stack recurrent layers that pytorch does, so there's some more
        bookkeeping here.
    linear1 : eqx.nn.Linear
        First linear layer of decoder.
    linear2 : eqx.nn.Linear
        Second linear layer of decoder.
    linear3 : eqx.nn.Linear
        Output layer of decoder.
    dropout1 : eqx.nn.Dropout
        First dropout layer.
    dropout2 : eqx.nn.Dropout
        Second dropout layer.
    activation : Callable
        Activation function, default relu.
    inference : bool
        Flag to tell the model if dropout should not be used (inference=True).

    Methods
    -------
    __call__(input_sensors, key)
        Forward pass through network for single, unbatched sequence input.
    
    """

    hidden_size: int
    in_size: int
    out_size: int
    cell1: eqx.nn.GRUCell
    cell2: eqx.nn.GRUCell
    linear1: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    linear2: eqx.nn.Linear
    dropout2: eqx.nn.Dropout
    linear3: eqx.nn.Linear
    activation: Callable
    inference: bool

    def __init__(
        self,
        in_size,
        out_size,
        hidden_size: int,
        lin_sizes: list[int] = [350, 400],
        dropout: float = 0.1,
        activation: Callable = jax.nn.relu,
        inference: bool = False,
        *,
        key
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
        inference : bool, optional
            Whether the model is in inference mode (dropout disabled).
        key : jax.random.PRNGKey
            Random key used to initialize weights and dropout layers.
        """
        r1key, r2key, l1key, l2key, l3key = jax.random.split(key, 5)
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation

        self.cell1 = eqx.nn.GRUCell(
            input_size=in_size, hidden_size=hidden_size, key=r1key
        )
        self.cell2 = eqx.nn.GRUCell(
            input_size=hidden_size, hidden_size=hidden_size, key=r2key
        )

        self.linear1 = eqx.nn.Linear(
            in_features=hidden_size, out_features=lin_sizes[0], key=l1key
        )
        self.linear2 = eqx.nn.Linear(
            in_features=lin_sizes[0], out_features=lin_sizes[1], key=l2key
        )
        self.linear3 = eqx.nn.Linear(
            in_features=lin_sizes[1], out_features=out_size, key=l3key
        )

        self.dropout1 = eqx.nn.Dropout(dropout)
        self.dropout2 = eqx.nn.Dropout(dropout)
        self.inference = inference


    def __call__(
        self,
        input_sensors: jnp.array,
        key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Forward pass through the GRU_SHRED model for a single, unbatched seq.

        Parameters
        ----------
        input_sensors : jnp.ndarray
            Input sequence, shape (sequence_length, in_size).
        key : jax.random.PRNGKey, optional
            PRNG key for dropout, (required if not in inference mode).

        Returns
        -------
        jnp.ndarray
            Model output, shape (out_size,).
        """
        if not self.inference:
            key1, key2 = jax.random.split(key)
        else:
            key1, key2 = None, None

        hidden = jnp.zeros(self.hidden_size)

        def f1(carry, inp):
            next_state = self.cell1(inp, carry)
            return next_state, next_state

        _, seq = jax.lax.scan(f1, hidden, input_sensors)

        def f2(carry, inp):
            next_state = self.cell2(inp, carry)
            return next_state, next_state

        out, _ = jax.lax.scan(f2, hidden, seq)

        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout1(out, key=key1)
        out = self.linear2(out)
        out = self.activation(out)
        out = self.dropout2(out, key=key2)
        out = self.linear3(out)
        return out
    
    def embed(
        self,
        input_sensors: jnp.array,
        key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Forward pass through the GRU_SHRED model for a single, unbatched seq.

        Parameters
        ----------
        input_sensors : jnp.ndarray
            Input sequence, shape (sequence_length, in_size).
        key : jax.random.PRNGKey, optional
            PRNG key for dropout, (required if not in inference mode).

        Returns
        -------
        jnp.ndarray
            Model output, shape (out_size,).
        """
        if not self.inference:
            key1, key2 = jax.random.split(key)
        else:
            key1, key2 = None, None

        hidden = jnp.zeros(self.hidden_size)

        def f1(carry, inp):
            next_state = self.cell1(inp, carry)
            return next_state, next_state

        _, seq = jax.lax.scan(f1, hidden, input_sensors)

        def f2(carry, inp):
            next_state = self.cell2(inp, carry)
            return next_state, next_state

        _, seq = jax.lax.scan(f2, hidden, seq)

        return seq

    def decode(self, out, key=None):
        key = jax.random.key(0)
        key1, key2 = jax.random.split(key)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout1(out, key=key1)
        out = self.linear2(out)
        out = self.activation(out)
        out = self.dropout2(out, key=key2)
        out = self.linear3(out)
        return out
    
def compute_loss(model, x, y, key):
    """
    Compute MSE loss over a batch.

    x should have shape (batch, seq_len, in_size) and y should have shape
    (batch, out_size).
    """
    def single_forward(xi, keyi):
        return model(xi, key=keyi)

    keys = jax.random.split(key, x.shape[0])

    # you can just eqx.filter_vmap over split keys when necessary
    preds = eqx.filter_vmap(single_forward)(x, keys)

    loss = jnp.mean((preds - y) ** 2)
    return loss


@eqx.filter_jit
def make_step(model, optimizer, opt_state, x, y, key):
    """
    Perform one optimizer step and return updated model and loss.
    """
    loss, grad = eqx.filter_value_and_grad(compute_loss)(model, x, y, key)

    # You have to use eqx.filter to tell the optimizer which components of the
    # model should be considerd (only arrays)
    updates, opt_state = optimizer.update(
        grad, opt_state, eqx.filter(model, eqx.is_array)
    )

    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

@eqx.filter_jit
def evaluate(model, val_inputs, val_targets):
    """
    Compute validation MSE.
    """

    def forward(x):
        return model(x)

    preds = eqx.filter_vmap(forward)(val_inputs)
    return jnp.mean((preds - val_targets) ** 2)


def train(
    model: GRU_SHRED,
    train_inputs: jnp.ndarray,
    train_targets: jnp.ndarray,
    val_inputs: jnp.ndarray,
    val_targets: jnp.ndarray,
    *,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    key: jax.random.PRNGKey
):
    """
    Training loop for GRU_SHRED model.

    Parameters
    ----------
    model : GRU_SHRED
        The model to train.
    train_inputs : jnp.ndarray
        Training inputs of shape (num_samples, seq_len, in_size).
    train_targets : jnp.ndarray
        Training targets of shape (num_samples, out_size).
    num_epochs : int, optional
        Number of training epochs. Default is 100.
    batch_size : int, optional
        Size of each mini-batch. Default is 32.
    learning_rate : float, optional
        Learning rate for optimizer. Default is 1e-3.
    key : jax.random.PRNGKey
        Random key for shuffling and dropout.

    Returns
    -------
    model : GRU_SHRED
        Trained model.
    """
    val_loss_list = []
    num_samples = train_inputs.shape[0]
    steps_per_epoch = num_samples // batch_size
    optimizer = optax.adam(learning_rate=learning_rate)

    # Have to initialize optimizer state with only the arrays in the model
    # Same as above using the eqx.filter function
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


    for epoch in range(num_epochs):
        t_start = time()
        key, shuffle_key, dropout_key = jax.random.split(key, 3)
        perm = jax.random.permutation(shuffle_key, num_samples)
        inputs_shuffled = train_inputs[perm]
        targets_shuffled = train_targets[perm]

        epoch_loss = 0.0

        for i in range(steps_per_epoch):
            batch_x = inputs_shuffled[i * batch_size:(i + 1) * batch_size]
            batch_y = targets_shuffled[i * batch_size:(i + 1) * batch_size]
            step_key, dropout_key = jax.random.split(dropout_key)
            model, opt_state, loss = make_step(
                model, optimizer, opt_state, batch_x, batch_y, step_key
            )
            epoch_loss += loss


        # Evaluate validation and training losses
        print('evaluating...')
        inference_model = eqx.nn.inference_mode(model)

        val_loss = evaluate(inference_model, val_inputs[::20], val_targets[::20])
        train_loss = evaluate(inference_model, train_inputs[::100], train_targets[::100])
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
        data: Input array of shape
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
