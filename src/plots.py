import numpy as np
from pathlib import Path
import plotly.graph_objects as go

top_dir = str(Path(__file__).parent.parent)

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
        fig.write_image(Path(top_dir) / 'figures' / f'{fname}.pdf', format='pdf')
    else:
        fig.show()
