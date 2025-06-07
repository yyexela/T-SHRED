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

def plot_field_comparison(prediction: torch.Tensor, target: torch.Tensor, dataset: str, sensors: list[tuple[int, int]], sensors_all = False, save: bool = False, fname: str = None, title_fontsize=20, label_fontsize=20, tick_fontsize=20) -> None:
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
        title_fontsize (int, optional): Font size for plot titles. Defaults to 16.
        label_fontsize (int, optional): Font size for axis labels. Defaults to 14.
        tick_fontsize (int, optional): Font size for tick labels. Defaults to 12.
    """
    # Move tensors to CPU and convert to numpy
    prediction = prediction.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    # Get dimensions
    n_dims = prediction.shape[2]
    
    # Create figure with GridSpec for better control over subplot spacing
    if dataset in ['planetswe']:
        figsize = (15, 2*n_dims)
        width_ratios = [1, 1, 0.05, 1, 0.05]
    elif dataset in ['sst']:
        figsize = (15, 2*n_dims)
        width_ratios = [1, 1, 0.05, 1, 0.05]
    elif dataset in ['plasma']:
        figsize = (8, 1.8*n_dims)
        width_ratios = [1, 1, 0.1, 1, 0.1]
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(n_dims, 5, figure=fig, width_ratios=width_ratios, wspace=0.3)

    planetswe_fields = ["$u$", "$v$", "$h$"]

    cmap = palettable.mycarta.Cube1_20.mpl_colormap
    
    # Plot each dimension
    for i in range(n_dims):
        # Get min/max for this dimension from both prediction and target
        vmin = min(prediction[:,:,i].min(), target[:,:,i].min())
        vmax = max(prediction[:,:,i].max(), target[:,:,i].max())
        
        # Prediction subplot
        ax_pred = fig.add_subplot(gs[i, 0])
        im_pred = ax_pred.imshow(prediction[:,:,i], vmin=vmin, vmax=vmax, cmap=cmap)
        if n_dims >= 2:
            ax_pred.set_title(f'Prediction (dim {i}: {planetswe_fields[i]})', fontsize=title_fontsize)
        else:
            ax_pred.set_title(f'Prediction', fontsize=title_fontsize)
        # Remove all ticks
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        
        # Target subplot
        ax_target = fig.add_subplot(gs[i, 1])
        im_target = ax_target.imshow(target[:,:,i], vmin=vmin, vmax=vmax, cmap=cmap)
        if n_dims >= 2:
            ax_target.set_title(f'Target (dim {i}: {planetswe_fields[i]})', fontsize=title_fontsize)
        else:
            ax_target.set_title(f'Target', fontsize=title_fontsize)
        # Remove all ticks
        ax_target.set_xticks([])
        ax_target.set_yticks([])

        # Add colorbar for first two images
        cbar_ax = fig.add_subplot(gs[i, 2])
        cbar = plt.colorbar(im_target, cax=cbar_ax)
        # Set colorbar ticks at extremes only
        cbar.set_ticks([vmin, vmax])
        cbar.ax.tick_params(labelsize=tick_fontsize)
        
        # Error subplot
        ax_error = fig.add_subplot(gs[i, 3])
        error = np.abs(prediction[:,:,i] - target[:,:,i])
        im_error = ax_error.imshow(error, cmap=cmap)
        if n_dims >= 2:
            ax_error.set_title(f'Absolute Error (dim {i}: {planetswe_fields[i]})', fontsize=title_fontsize)
        else:
            ax_error.set_title(f'Absolute Error', fontsize=title_fontsize)
        # Remove all ticks
        ax_error.set_xticks([])
        ax_error.set_yticks([])

        # Add sensor markers to error subplot
        if dataset in ["sst", "planetswe"]:
            if sensors:
                for sensor in sensors:
                    x, y = sensor
                    ax_error.plot(y, x, 'ro', markersize=2)
                    if sensors_all:
                        ax_pred.plot(y, x, 'ro', markersize=2)
                        ax_target.plot(y, x, 'ro', markersize=2)
        
        # Add colorbar for this row
        cbar_ax = fig.add_subplot(gs[i, 4])
        cbar = plt.colorbar(im_error, cax=cbar_ax)
        # Set colorbar ticks at extremes only
        error_min, error_max = error.min(), error.max()
        cbar.set_ticks([error_min, error_max])
        cbar.ax.tick_params(labelsize=tick_fontsize)
    
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

def plot_model_results_scatter(results: list[dict], dataset: str, top_n: int = None, save: bool = False, fname: str = None, title_fontsize: int = 16, axes_fontsize: int = 14, legend_fontsize: int = 12, reverse: bool = False) -> None:
    """
    Create a scatter plot of model results using plotly, where:
    - y-axis shows the results (test loss) on a log scale
    - x-axis is ordered by performance on a log scale
    - colors are based on encoder and decoder combinations
    - only shows results for the specified dataset
    - optionally shows only the top N performing models
    - groups results by model configuration (excluding seed) and shows mean with error bars
    
    Args:
        results (list[dict]): List of dictionaries containing model results and hyperparameters
        dataset (str): Name of the dataset to filter results for
        top_n (int, optional): If provided, only show the top N performing models. Defaults to None.
        save (bool, optional): Whether to save the figure to a file. Defaults to False.
        fname (str, optional): If saving, the filename to save to. Required if save=True. Defaults to None.
        title_fontsize (int, optional): Font size for the plot title. Defaults to 16.
        axes_fontsize (int, optional): Font size for axes titles and tick labels. Defaults to 14.
        legend_fontsize (int, optional): Font size for legend text. Defaults to 12.
        reverse (bool, optional): Whether to reverse the order of the models. Defaults to False.
    """
    # Filter results for the specified dataset
    filtered_results = [r for r in results if r['hyperparameters']['dataset'] == dataset]

    # Filter to only include transformer encoders
    #filtered_results = [r for r in filtered_results if 'transformer' in r['hyperparameters']['encoder']]
    
    # Group results by model configuration (excluding seed)
    from collections import defaultdict
    model_groups = defaultdict(list)
    
    for r in filtered_results:
        # Create a key from hyperparameters excluding 'seed'
        hyperparams = r['hyperparameters'].copy()
        hyperparams.pop('seed', None)  # Remove seed if it exists
        
        # Convert to a frozenset of items to make it hashable
        key = f"{hyperparams['encoder']}_{hyperparams['decoder']}_{hyperparams['dataset']}_e{hyperparams['encoder_depth']}_d{hyperparams['decoder_depth']}_lr{hyperparams['lr']:0.2e}_p{hyperparams['poly_order']}"

        test_loss = r.get('test_loss', None)
        
        if test_loss is not None:
            model_groups[key].append({
                'test_loss': test_loss,
                'hyperparameters': hyperparams,
                'original_result': r
            })
        else:
            raise Exception("Test loss is None for", r)
    
    # Calculate mean and std for each model configuration
    aggregated_results = []
    for i, (key, group) in enumerate(model_groups.items()):
        test_losses = [item['test_loss'] for item in group]
        mean_loss = np.mean(test_losses)

        if i == 0:
            print("test losses:", test_losses)
        std_loss = np.std(test_losses) if len(test_losses) > 1 else 0.0
        
        # Use the hyperparameters from the first result in the group
        hyperparams = group[0]['hyperparameters']
        
        aggregated_results.append({
            'mean_test_loss': mean_loss,
            'std_test_loss': std_loss,
            'n_seeds': len(test_losses),
            'hyperparameters': hyperparams
        })
    
    # Sort results by mean test loss (ascending - best models first)
    aggregated_results.sort(key=lambda x: x['mean_test_loss'], reverse=reverse)

    # If top_n is specified, only keep the top N models
    if top_n is not None:
        aggregated_results = aggregated_results[:top_n]
    
    # Reverse list
    aggregated_results = aggregated_results[::-1]
    
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
        encoder_name = None
        if encoder == "lstm":
            encoder_name = "LSTM"
        elif encoder == "gru":
            encoder_name = "GRU"
        elif encoder == "sindy_loss_lstm":
            encoder_name = "SL-LSTM"
        elif encoder == "sindy_loss_gru":
            encoder_name = "SL-GRU"
        elif encoder == "vanilla_transformer":
            encoder_name = "T"
        elif encoder == "sindy_loss_transformer":
            encoder_name = "SL-T"
        elif encoder == "sindy_attention_transformer":
            encoder_name = "SA-T"
        elif encoder == "sindy_attention_sindy_loss_transformer":
            encoder_name = "SASL-T"
            
            
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name=f'{encoder_name}',
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
        decoder_name = None
        if decoder == "mlp":
            decoder_name = "MLP"
        elif decoder == "unet":
            decoder_name = "CNN"
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name=f'{decoder_name}',
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
    
    # Add actual data points with error bars
    for i, r in enumerate(aggregated_results):
        encoder = r['hyperparameters']['encoder']
        encoder_depth = r['hyperparameters']['encoder_depth']
        decoder = r['hyperparameters']['decoder']
        decoder_depth = r['hyperparameters']['decoder_depth']
        mean_test_loss = r['mean_test_loss']
        std_test_loss = r['std_test_loss']
        n_seeds = r['n_seeds']
        
        fig.add_trace(go.Scatter(
            x=[i + 1],  # Add 1 to avoid log(0)
            y=[mean_test_loss],
            error_y=dict(
                type='data',
                array=[std_test_loss],
                visible=True,
                color='black',
                thickness=1,
                width=3
            ),
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
                f"Mean Test Loss: {mean_test_loss:.2e}<br>"
                f"Std Test Loss: {std_test_loss:.2e}<br>"
                f"Seeds: {n_seeds}<br>"
                f"<extra></extra>"
            ),
            showlegend=False  # Don't show these in legend
        ))
    
    # Update layout with log scales
    dataset_name = None
    if dataset == "sst":
        dataset_name = "SST"
    elif dataset == "planetswe":
        dataset_name = "PlanetSWE"
    elif dataset == "plasma":
        dataset_name = "Plasma"

    title_prefix = "Top" if not reverse else "Bottom"
        
    fig.update_layout(
        title=dict(
            text=f'{dataset_name}',
            font=dict(
                size=title_fontsize
            )
        ),
        xaxis_title='Model',
        yaxis_title='Test Loss',
        showlegend=True,
        template='plotly_white',
        hovermode='closest',
        xaxis=dict(
            title=dict(
                text=f'{title_prefix} {top_n if top_n else len(aggregated_results)} Models',
                font=dict(
                    size=axes_fontsize
                )
            ),
            showticklabels=False,
            tickfont=dict(
                size=axes_fontsize
            )
        ),
        yaxis=dict(
            type='log',
            title=dict(
                text='Test Loss',
                font=dict(
                    size=axes_fontsize
                )
            ),
            exponentformat='E',
            tickformat='1.2e',
            nticks=5,
            tickfont=dict(
                size=axes_fontsize
            )
        ),
        legend=dict(
            title=dict(
                text='Legend',
                font=dict(
                    size=legend_fontsize
                )
            ),
            font=dict(
                size=legend_fontsize
            )
        ),
        height=500,
        width=500
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
