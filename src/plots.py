import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec

top_dir = str(Path(__file__).parent.parent)
figure_dir = Path(top_dir) / 'figures'
figure_dir.mkdir(parents=True, exist_ok=True)

def plot_field_comparison(prediction: torch.Tensor, target: torch.Tensor, save: bool = False, fname: str = None) -> None:
    """
    Plot comparison between predicted and target fields using matplotlib, with one row per dimension. Ensure that each row has a single colorbar that is scaled to the minimum and maximum of the target field. Each row has a separate colorbar with a separate scale.
    
    Args:
        prediction (torch.Tensor): Predicted field of shape (rows, cols, dim)
        target (torch.Tensor): Target field of shape (rows, cols, dim)
        save (bool, optional): Whether to save the figure to a file. Defaults to False.
        fname (str, optional): If saving, the filename to save to. Required if save=True. Defaults to None.
    """
    # Move tensors to CPU and convert to numpy
    prediction = prediction.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    # Get dimensions
    n_dims = prediction.shape[2]
    
    # Create figure with GridSpec for better control over subplot spacing
    fig = plt.figure(figsize=(24, 4*n_dims), constrained_layout=True)
    gs = GridSpec(n_dims, 5, figure=fig, width_ratios=[1, 1, 0.05, 1, 0.05], wspace=0.3)
    
    # Plot each dimension
    for i in range(n_dims):
        # Get min/max for this dimension from target
        vmin = target[:,:,i].min()
        vmax = target[:,:,i].max()
        
        # Prediction subplot
        ax_pred = fig.add_subplot(gs[i, 0])
        im_pred = ax_pred.imshow(prediction[:,:,i], vmin=vmin, vmax=vmax)
        ax_pred.set_title(f'Prediction (dim {i})')
        
        # Target subplot
        ax_target = fig.add_subplot(gs[i, 1])
        im_target = ax_target.imshow(target[:,:,i], vmin=vmin, vmax=vmax)
        ax_target.set_title(f'Target (dim {i})')

        # Add colorbar for first two images
        cbar_ax = fig.add_subplot(gs[i, 2])
        plt.colorbar(im_target, cax=cbar_ax)
        
        # Error subplot
        ax_error = fig.add_subplot(gs[i, 3])
        error = np.abs(prediction[:,:,i] - target[:,:,i])
        im_error = ax_error.imshow(error)
        ax_error.set_title(f'Absolute Error (dim {i})')
        
        # Add colorbar for this row
        cbar_ax = fig.add_subplot(gs[i, 4])
        plt.colorbar(im_error, cax=cbar_ax)
    
    # Remove tight_layout call since we're using constrained_layout
    if save:
        if fname is None:
            raise ValueError("Filename must be provided when save=True")
        plt.savefig(figure_dir / f"{fname}.pdf", bbox_inches='tight', dpi=300)
    else:
        plt.show()
    
    plt.close()

def plot_losses(training_loss: list[float], validation_loss: list[float], saved_epoch: int, save: bool = False, fname: str = None) -> None:
    """
    Plot training and validation losses with a marker at the saved epoch.
    
    Args:
        training_loss (list[float]): List of training loss values per epoch
        validation_loss (list[float]): List of validation loss values per epoch
        saved_epoch (int): The epoch number where the model was saved
        save (bool, optional): Whether to save the figure to a file. Defaults to False.
        fname (str, optional): If saving, the filename to save to. Required if save=True. Defaults to None.
    """
    # Create x-axis values (epochs)
    epochs = list(range(1, len(training_loss) + 1))
    
    # Create the figure
    fig = go.Figure()
    
    # Add training loss line
    fig.add_trace(go.Scatter(
        x=epochs,
        y=training_loss,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue')
    ))
    
    # Add validation loss line
    fig.add_trace(go.Scatter(
        x=epochs,
        y=validation_loss,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='red')
    ))
    
    # Add marker at saved epoch
    fig.add_trace(go.Scatter(
        x=[saved_epoch],
        y=[validation_loss[saved_epoch - 1]],
        mode='markers',
        name='Saved Model',
        marker=dict(
            size=15,
            color='yellow',
            symbol='star',
            line=dict(
                color='black',
                width=2
            )
        )
    ))
    
    # Update layout
    fig.update_layout(
        title='Training and Validation Losses',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        showlegend=True,
        template='plotly_white'
    )
    
    # Show or save the plot
    if save:
        if fname is None:
            raise Exception(f"Filename fname ({fname}) must not be None.")
        fig.write_image(figure_dir / f'{fname}.pdf', format='pdf')
    else:
        fig.show()
