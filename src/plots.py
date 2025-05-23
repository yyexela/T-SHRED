import torch
import kaleido
import palettable
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec

top_dir = str(Path(__file__).parent.parent)
figure_dir = Path(top_dir) / 'figures'

def plot_field_comparison(prediction: torch.Tensor, target: torch.Tensor, dataset: str, sensors: list[tuple[int, int]], sensors_all = False, save: bool = False, fname: str = None) -> None:
    """
    Plot comparison between predicted and target fields using matplotlib, with one row per dimension. Ensure that each row has a single colorbar that is scaled to the minimum and maximum of the target field. Each row has a separate colorbar with a separate scale.
    
    Args:
        prediction (torch.Tensor): Predicted field of shape (rows, cols, dim)
        target (torch.Tensor): Target field of shape (rows, cols, dim)
        dataset (str): Name of the dataset to use for figure size
        sensors (list[tuple[int, int]]): List of sensor positions to plot
        sensors_all (bool, optional): Whether to plot sensors on all plots. Defaults to False.
        save (bool, optional): Whether to save the figure to a file. Defaults to False.
        fname (str, optional): If saving, the filename to save to. Required if save=True. Defaults to None.
    """
    # Move tensors to CPU and convert to numpy
    prediction = prediction.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    # Get dimensions
    n_dims = prediction.shape[2]
    
    # Create figure with GridSpec for better control over subplot spacing
    if dataset in ['planetswe', 'planetswe_pod', 'planetswe_full']:
        figsize = (15, 2*n_dims)
        width_ratios = [1, 1, 0.05, 1, 0.05]
    elif dataset in ['sst']:
        figsize = (15, 2*n_dims)
        width_ratios = [1, 1, 0.05, 1, 0.05]
    elif dataset in ['gray_scott_reaction_diffusion', 'gray_scott_reaction_diffusion_pod']:
        figsize = (14, 4*n_dims)
        width_ratios = [1, 1, 0.05, 1, 0.05]
    elif dataset in ['plasma']:
        figsize = (8, 4*n_dims)
        width_ratios = [1, 1, 0.1, 1, 0.1]
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(n_dims, 5, figure=fig, width_ratios=width_ratios, wspace=0.3)
    
    # Plot each dimension
    for i in range(n_dims):
        # Get min/max for this dimension from target
        vmin = target[:,:,i].min()
        vmax = target[:,:,i].max()
        
        # Prediction subplot
        ax_pred = fig.add_subplot(gs[i, 0])
        im_pred = ax_pred.imshow(prediction[:,:,i], vmin=vmin, vmax=vmax)
        if n_dims >= 2:
            ax_pred.set_title(f'Prediction (dim {i})')
        else:
            ax_pred.set_title(f'Prediction')
        
        # Target subplot
        ax_target = fig.add_subplot(gs[i, 1])
        im_target = ax_target.imshow(target[:,:,i], vmin=vmin, vmax=vmax)
        if n_dims >= 2:
            ax_target.set_title(f'Target (dim {i})')
        else:
            ax_target.set_title(f'Target')

        # Add colorbar for first two images
        cbar_ax = fig.add_subplot(gs[i, 2])
        plt.colorbar(im_target, cax=cbar_ax)
        
        # Error subplot
        ax_error = fig.add_subplot(gs[i, 3])
        error = np.abs(prediction[:,:,i] - target[:,:,i])
        im_error = ax_error.imshow(error)
        if n_dims >= 2:
            ax_error.set_title(f'Absolute Error (dim {i})')
        else:
            ax_error.set_title(f'Absolute Error')

        # Add sensor markers to error subplot
        if dataset in ["sst", "planetswe_full"]:
            if sensors:
                for sensor in sensors:
                    x, y = sensor
                    ax_error.plot(y, x, 'ro', markersize=2)
                    if sensors_all:
                        ax_pred.plot(y, x, 'ro', markersize=2)
                        ax_target.plot(y, x, 'ro', markersize=2)
        
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

def plot_model_results_scatter(results: list[dict], dataset: str, top_n: int = None, save: bool = False, fname: str = None) -> None:
    """
    Create a scatter plot of model results using plotly, where:
    - y-axis shows the results (test loss) on a log scale
    - x-axis is ordered by performance on a log scale
    - colors are based on encoder and decoder combinations
    - only shows results for the specified dataset
    - optionally shows only the top N performing models
    
    Args:
        results (list[dict]): List of dictionaries containing model results and hyperparameters
        dataset (str): Name of the dataset to filter results for
        top_n (int, optional): If provided, only show the top N performing models. Defaults to None.
    """
    # Filter results for the specified dataset
    filtered_results = [r for r in results if r['hyperparameters']['dataset'] == dataset]
    
    # Sort results by test loss (ascending)
    filtered_results.sort(key=lambda x: x.get('test_loss', x.get('test_loss_pod', None)), reverse=True)

    # Assert to make sure filtered_results does not contain None
    assert None not in filtered_results, "filtered_results contains None"

    # If top_n is specified, only keep the top N models
    if top_n is not None:
        filtered_results = filtered_results[-top_n:]
    
    # Get unique encoders and decoders
    unique_encoders = ["lstm", "gru", "sindy_loss_lstm", "sindy_loss_gru", "vanilla_transformer", "sindy_loss_transformer", "sindy_attention_transformer", "sindy_attention_sindy_loss_transformer"]
    unique_decoders = ["mlp", "unet"]
    
    # Create color mappings
    encoder_colors = palettable.cartocolors.qualitative.Prism_8.hex_colors
    encoder_color_map = {encoder: encoder_colors[i % len(encoder_colors)] for i, encoder in enumerate(unique_encoders)}
    decoder_colors = palettable.cartocolors.qualitative.Pastel_3.hex_colors
    decoder_color_map = {decoder: decoder_colors[i % len(decoder_colors)] for i, decoder in enumerate(unique_decoders)}
    
    # Create the figure
    fig = go.Figure()
    
    # First add dummy traces for encoders to create legend entries
    for encoder in unique_encoders:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name=f'Encoder: {encoder}',
            marker=dict(
                color="white",
                line=dict(
                    color=encoder_color_map[encoder],
                    width=4
                ),
                size=12,
            ),
            showlegend=True
        ))
    
    # Then add dummy traces for decoders to create legend entries
    for decoder in unique_decoders:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name=f'Decoder: {decoder}',
            marker=dict(
                color=decoder_color_map[decoder],
                line=dict(
                    color="white",
                    width=4
                ),
                size=12,
            ),
            showlegend=True
        ))
    
    # Add actual data points
    for r in filtered_results:
        encoder = r['hyperparameters']['encoder']
        encoder_depth = r['hyperparameters']['encoder_depth']
        decoder = r['hyperparameters']['decoder']
        decoder_depth = r['hyperparameters']['decoder_depth']
        test_loss = r.get('test_loss', r.get('test_loss_pod', None))
        
        if test_loss is not None:
            fig.add_trace(go.Scatter(
                x=[filtered_results.index(r) + 1],  # Add 1 to avoid log(0)
                y=[test_loss],
                mode='markers',
                name=f'{encoder}-{decoder}',
                marker=dict(
                    color=decoder_color_map[decoder],
                    line=dict(
                        color=encoder_color_map[encoder],
                        width=4
                    ),
                    size=12
                ),
                hovertemplate=(
                    f"Encoder: {encoder} (x{encoder_depth})<br>"
                    f"Decoder: {decoder} (x{decoder_depth})<br>"
                    f"LR: {r['hyperparameters']['lr']:.2e}<br>"
                    f"Test Loss: {test_loss:.2e}<br>"
                    f"<extra></extra>"
                ),
                showlegend=False  # Don't show these in legend
            ))
    
    # Update layout with log scales
    fig.update_layout(
        title=f'Model Results for {dataset} Dataset' + (f' (Top {top_n})' if top_n is not None else ''),
        xaxis_title='Model',
        yaxis_title='Test Loss',
        showlegend=True,
        template='plotly_white',
        hovermode='closest',
        xaxis=dict(
            title='Model',
            showticklabels=False
        ),
        yaxis=dict(
            type='log',
            title='Test Loss',
            exponentformat = 'E'
        ),
        legend=dict(
            title='Legend'
        ),
        height=400,
        width=800
    )
    
    # Show or save the plot
    if save:
        if fname is None:
            raise Exception(f"Filename fname ({fname}) must not be None.")
        fig.write_image(figure_dir / f'{fname}.pdf', engine='kaleido')
        print(f"Saved {figure_dir/fname}.pdf")
        fig.show()
    else:
        fig.show()
