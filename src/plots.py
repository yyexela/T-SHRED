import torch
import kaleido
import palettable
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('agg')
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

def plot_model_results_scatter(results: list[dict], dataset: str, top_n: int = None, save: bool = False, fname: str = None, title_fontsize: int = 16, axes_fontsize: int = 14, legend_fontsize: int = 12, reverse: bool = False, encoders: list[str] = None, results_type: str = "hyper_opt") -> None:
    """
    Create a scatter plot of model results using plotly, where:
    - y-axis shows the results (test loss) on a log scale
    - x-axis is ordered by performance on a log scale
    - colors are based on encoder and decoder combinations
    - only shows results for the specified dataset
    - optionally shows only the top N performing models
    - groups results by model configuration (excluding seed) and shows mean with error bars
    - results_type can be "hyper_opt" or "pickle"
    
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
        encoders (list[str], optional): List of encoders to include in the plot. Defaults to None, which uses all encoders.
    """
    # Filter results for the specified dataset
    if results_type == "hyper_opt":
        filtered_results = [r for r in results if r['final_config']['model']['dataset'] == dataset]
    elif results_type == "pickle":
        filtered_results = [r for r in results if r['hyperparameters']['dataset'] == dataset]
    else:
        raise Exception("Invalid results_type:", results_type)

    # Filter to only include transformer encoders
    #filtered_results = [r for r in filtered_results if 'transformer' in r['hyperparameters']['encoder']]
    
    # Group results by model configuration (excluding seed)
    from collections import defaultdict
    model_groups = defaultdict(list)
    
    for r in filtered_results:
        if results_type == "hyper_opt":
            if encoders is not None and r['final_config']['model']['encoder'] not in encoders:
                continue
        elif results_type == "pickle":
            if encoders is not None and r['hyperparameters']['encoder'] not in encoders:
                continue

        # Create a key  from hyperparameters excluding 'seed'
        if results_type == "hyper_opt":
            hyperparams = r['final_config']['model'].copy()
        elif results_type == "pickle":
            hyperparams = r['hyperparameters'].copy()

        # Modify encoder if rollout is not 1
        if "rollout" in hyperparams['encoder']:
            if r['hyperparameters']['forecast_length'] != 1:
                hyperparams['encoder'] = hyperparams['encoder'] + f"_{r['hyperparameters']['forecast_length']}"
        
        # Convert to a frozenset of items to make it hashable
        key = f"{hyperparams['encoder']}_{hyperparams['decoder']}_{hyperparams['dataset']}"

        if results_type == "hyper_opt":
            test_loss = r['best_value']
        elif results_type == "pickle":
            test_loss = r['test_loss']
        
        if test_loss is not None:
            model_groups[key].append({
                'test_loss': test_loss,
                'hyperparameters': hyperparams,
            })
        else:
            raise Exception("Test loss is None for", r)
    
    # Calculate mean and std for each model configuration
    aggregated_results = []
    for i, (key, group) in enumerate(model_groups.items()):
        test_losses = [item['test_loss'] for item in group]
        mean_loss = np.mean(test_losses)

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
    if not reverse:
        aggregated_results = aggregated_results[::-1]
    
    # Get unique encoders and decoders
    unique_encoders = list(set([r['hyperparameters']['encoder'] for r in aggregated_results]))
    unique_decoders = list(set([r['hyperparameters']['decoder'] for r in aggregated_results]))
    
    # Create color mappings
    encoder_colors = palettable.cartocolors.qualitative.Prism_9.hex_colors
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
        elif encoder == "sindy_attention_transformer_rollout":
            encoder_name = "SAR-T"
        elif encoder == "sindy_attention_sindy_loss_transformer_rollout":
            encoder_name = "SASLR-T"
        elif encoder == "sindy_attention_transformer_rollout_5":
            encoder_name = "SAR-T-5"
        elif encoder == "sindy_attention_sindy_loss_transformer_rollout_5":
            encoder_name = "SASLR-T-5"
            
            
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
        elif decoder == "cnn":
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
        decoder = r['hyperparameters']['decoder']
        mean_test_loss = r['mean_test_loss']
        std_test_loss = r['std_test_loss']
        n_seeds = r['n_seeds']

        if r['hyperparameters']['coord_descent']:
            hover_template = (
                f"Encoder: {encoder}<br>"
                f"Decoder: {decoder}<br>"
                f"Mean Test Loss: {mean_test_loss:.2e}<br>"
                f"Std Test Loss: {std_test_loss:.2e}<br>"
                f"Seeds: {n_seeds}<br>"
                f"<extra></extra>"
            )
        else:
            hover_template = (
                f"Encoder: {encoder}<br>"
                f"Decoder: {decoder}<br>"
                f"Mean Test Loss: {mean_test_loss:.2e}<br>"
                f"Std Test Loss: {std_test_loss:.2e}<br>"
                f"Seeds: {n_seeds}<br>"
                f"<extra></extra>"
            )
        
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
            hovertemplate=hover_template,
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
        height=550,
        width=650
    )
    
    # Show or save the plot
    if save:
        if fname is None:
            raise Exception(f"Filename fname ({fname}) must not be None.")
        fig.write_image(figure_dir / f'{fname}.pdf', engine='kaleido')
        print(f"Saved {figure_dir/fname}.pdf")
        #fig.show()
    else:
        fig.show()

def plot_eigvs(eigvs_np, save: bool = False, fname: str = None) -> None:
    """
    Plot eigenvalue trajectories in the complex plane using matplotlib. eigvs_np is a numpy array of shape [steps, terms], 
    where steps is the number of time steps and terms is the number of different eigenvalues being tracked.
    The x-axis represents the real component and y-axis represents the imaginary component of eigenvalues.
    Time evolution is shown through color: first time step in green, last in red, intermediate in blue.
    Each eigenvalue term is connected with a trajectory line.
    
    Args:
        eigvs_np (np.ndarray): Array of complex eigenvalues with shape [steps, terms]
        save (bool, optional): Whether to save the figure to a file. Defaults to False.
        fname (str, optional): If saving, the filename to save to. Required if save=True. Defaults to None.
    """
    # Convert to numpy array for easier handling
    eigvs_array = np.array(eigvs_np)
    
    # Ensure we have a 2D array
    if eigvs_array.ndim == 1:
        eigvs_array = eigvs_array.reshape(-1, 1)
    
    steps, terms = eigvs_array.shape
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract real and imaginary parts
    real_parts = np.real(eigvs_array)
    imag_parts = np.imag(eigvs_array)
    
    # Generate colors for time evolution (green -> blue -> red)
    time_colors = []
    if steps == 1:
        time_colors = ['green']
    elif steps == 2:
        time_colors = ['green', 'red']
    else:
        # First green, last red, intermediate blue
        time_colors = ['green'] + ['blue'] * (steps - 2) + ['red']
    
    # Plot each eigenvalue trajectory
    for i in range(terms):
        # Get trajectory for this eigenvalue term
        real_traj = real_parts[:, i]
        imag_traj = imag_parts[:, i]
        
        # Plot trajectory line connecting all time steps for this eigenvalue
        ax.plot(real_traj, imag_traj, 
                color='gray', linewidth=1, alpha=0.5, zorder=1)
        
        # Plot individual time points with color coding
        for t in range(steps):
            ax.scatter(real_traj[t], imag_traj[t], 
                      c=time_colors[t], s=50, alpha=0.8, zorder=2,
                      edgecolors='black', linewidth=0.5)
        
        # Add label for the first point of each eigenvalue
        ax.annotate(f'λ{i+1}', 
                   xy=(real_traj[0], imag_traj[0]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(f'Eigenvalue Trajectories in Complex Plane\n({terms} eigenvalue{"s" if terms > 1 else ""}, {steps} time step{"s" if steps > 1 else ""})')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add unit circle for reference (common in eigenvalue analysis)
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.3)
    ax.add_patch(circle)
    
    # Create custom legend for time evolution
    from matplotlib.lines import Line2D
    legend_elements = []
    if steps > 1:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                                    markersize=8, label='First time step', markeredgecolor='black'))
        if steps > 2:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                        markersize=8, label='Intermediate steps', markeredgecolor='black'))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                    markersize=8, label='Last time step', markeredgecolor='black'))
        legend_elements.append(Line2D([0], [0], color='gray', alpha=0.5, label='Trajectory'))
    else:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                                    markersize=8, label='Eigenvalue', markeredgecolor='black'))
    
    ax.legend(handles=legend_elements, loc='best')
    
    # Save or show the plot
    if save:
        if fname is None:
            raise ValueError("Filename must be provided when save=True")
        plt.savefig(figure_dir / f"{fname}.pdf", bbox_inches='tight', dpi=300)
    else:
        plt.show()
    
    plt.close()
