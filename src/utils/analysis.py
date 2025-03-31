import os
from typing import Any

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.utils.config import PlotConfig

config = PlotConfig()
config.apply()  # Explicitly apply global font and style settings


def filter_dataframe(df: pd.DataFrame, filters: dict[str, Any] | None = None) -> pd.DataFrame:
    """Filter a DataFrame based on permissible values provided in a dictionary."""
    if filters is None:
        return df
    else:
        mask = pd.Series(True, index=df.index)
        for col, allowed_values in filters.items():
            if not isinstance(allowed_values, list):
                allowed_values = [allowed_values]
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            mask &= df[col].isin(allowed_values)
        return df[mask]


def plot_test_accuracies(test_name: str, plot_type: str, filters: dict[str, Any] | None = None) -> None:
    """Plot the results of a test and save the plot to a file."""
    # Load the test dataframe
    df = pd.read_csv(f'outputs/tests/{test_name}/results.csv')

    # Apply filters
    df = filter_dataframe(df, filters)

    # Get model names
    model_names = df['model_name'].unique().tolist()

    # Plot accuracy versus test percolation for each model
    if plot_type == 'acc_vs_perc':
        # Get train percolations
        train_percolations = []
        for model_name in model_names:
            train_percolation = df[df['model_name'] == model_name]['train_percolation'].unique()
            if len(train_percolation) == 1:
                train_percolations.append(train_percolation[0])
            else:
                raise ValueError(
                    f'Expected single train_percolation value for {model_name = }, got {train_percolation = }'
                )

        # Sort model names and train percolations by train percolations
        model_names, train_percolations = map(
            list, zip(*sorted(zip(model_names, train_percolations, strict=False), key=lambda x: x[1]), strict=False)
        )

        # Ensure only one maze size for each model
        maze_sizes = df['test_maze_size'].unique()
        if len(maze_sizes) > 1:
            raise NotImplementedError(f'Expected single maze size for each model, got {maze_sizes = }')

        # Create figure
        plt.figure(figsize=(14, 6), dpi=600)

        # Set up color mapping
        small_value = 1e-3
        color_mapping_values = [tp if tp > 0 else small_value for tp in train_percolations]
        cmap = cm.get_cmap('plasma', len(train_percolations))
        norm = mcolors.Normalize(vmin=min(color_mapping_values), vmax=max(color_mapping_values))

        # Generate colors using the modified mapping list
        colors = [cmap(norm(tp if tp > 0 else small_value)) for tp in train_percolations]

        # Plot accuracy versus test percolation for each model
        for model_name, train_percolation, color in zip(model_names, train_percolations, colors, strict=False):
            df_subset = df[df['model_name'] == model_name]

            df_subset = df_subset.groupby('test_percolation')['correct'].mean().reset_index()
            df_subset = df_subset.sort_values(by='test_percolation')  # Ensure sorted order

            plt.plot(
                df_subset['test_percolation'],
                df_subset['correct'],
                marker='o',
                linestyle='-',
                linewidth=5,
                markersize=10,
                label=f'{train_percolation:.3f}',
                color=color,
                zorder=3,
            )

            # Add vertical dashed line at train_percolation
            plt.axvline(x=train_percolation, color=color, linestyle='dashed', linewidth=5, zorder=1)

        # Set plot title and labels
        plt.xlabel('Test Percolation')
        plt.ylabel('Test Accuracy')
        legend = plt.legend(title='Train Percolation', loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.setp(legend.get_title(), fontweight=config.legend_title_weight, fontsize=config.legend_title_fontsize)

        # Set axis limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Save the plot
        plt.savefig(f'outputs/tests/{test_name}/acc_vs_perc.pdf', bbox_inches='tight')
        plt.close()

    # Plot heatmap of test accuracy versus maze size and test percolation for each model
    elif plot_type == 'acc_vs_size_perc':
        # Columns we explicitly check for filtering
        columns_to_check = {'model_name', 'test_iter'}
        required_axes = {'test_maze_size', 'test_percolation'}

        # Ensure required axes have multiple unique values
        for col in required_axes:
            if df[col].nunique() <= 1:
                raise ValueError(f"Column '{col}' must have multiple unique values for a heatmap.")

        # Identify columns by uniqueness
        unique_counts = df.nunique()
        multi_value_cols = {col for col, count in unique_counts.items() if count > 1 and col in columns_to_check}

        # Validate and apply filtering
        filters = filters or {}
        for col, value in filters.items():
            if col not in df.columns:
                raise ValueError(f"Invalid column: '{col}' does not exist in the dataframe.")
            if col in required_axes:
                raise ValueError(f"Cannot filter on required axis column: '{col}'")
            if col in multi_value_cols and value not in df[col].unique():
                raise ValueError(f"Invalid value for '{col}': {value}. Must be one of {df[col].unique()}.")

        # Ensure all multi-value columns in `columns_to_check` are specified in `filters`
        unspecified_cols = multi_value_cols - filters.keys()
        if unspecified_cols:
            raise ValueError(f'Columns {unspecified_cols} have multiple values. Please specify one in `filters`.')

        # Apply filtering
        filtered_df = filter_dataframe(df, filters)

        # Get model name (dt_net or it_net)
        if len(filtered_df['model_name'].unique()) != 1:
            raise ValueError(
                f'Expected exactly one model_name after filtering, but got {len(filtered_df["model_name"].unique())}.'
            )
        model_name = filtered_df['model_name'].unique()[0]
        if 'dt_net' in model_name:
            model_name = 'dt_net'
        elif 'it_net' in model_name:
            model_name = 'it_net'
        else:
            raise ValueError(f'Unexpected model_name found: {model_name}')

        # Get train percolation for the model
        train_percolation = filtered_df['train_percolation'].unique()
        if len(train_percolation) != 1:
            raise ValueError(f'Expected single train_percolation value for {model_name = }, got {train_percolation = }')
        train_percolation = train_percolation[0]

        # Get test_iter
        test_iter = filtered_df['test_iter'].unique()
        if len(test_iter) != 1:
            raise ValueError(f'Expected single test_iter value for {model_name = }, got {test_iter = }')
        test_iter = test_iter[0]

        # Compute heatmap values: mean of is_correct
        heatmap_data = filtered_df.groupby(['test_percolation', 'test_maze_size'])['correct'].mean().unstack()

        # Extract axis labels
        x_labels = heatmap_data.columns.values  # Maze sizes
        y_labels = heatmap_data.index.values  # Test percolation values
        heatmap_values = heatmap_data.values  # Heatmap matrix

        # Plot heatmap using Matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(heatmap_values, cmap='coolwarm', aspect='auto')

        # Add color bar
        cbar = fig.colorbar(cax)
        cbar.set_label('Test Accuracy')

        # Set axis labels and ticks
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Test Maze Size')
        ax.set_ylabel('Test Percolation')
        fig.suptitle(
            'Test Accuracy',
            fontsize=config.axes_title_size,
            fontweight=config.axes_title_weight,
        )
        subtitle_text = f'Model: {model_name} | Train Perc: {train_percolation:.3f} | Test Iter: {test_iter}'
        ax.set_title(
            subtitle_text,
            fontsize=config.font_size * 0.8,  # smaller than the main title
            fontweight='normal',  # not bold
            pad=10,  # extra spacing between subtitle and plot
        )

        plt.tight_layout(rect=[0, 0, 1, 1])  # type: ignore

        ax.invert_yaxis()

        # Save the plot
        plt.savefig(
            f'outputs/tests/{test_name}/acc_vs_size_perc_{model_name}_perc-{train_percolation:.3f}_iter-{test_iter}.pdf',
            bbox_inches='tight',
        )
        plt.close()

    # Plot overall accuracy for each model across all test percolations and maze sizes versus train percolation,
    # with separate curves for each iteration value
    elif plot_type == 'overall_acc_vs_perc':
        # Get sorted unique iteration values
        iters = np.sort(df['test_iter'].unique())

        # Set up logarithmic normalization and colormap for iteration values
        norm = mcolors.LogNorm(vmin=iters.min(), vmax=iters.max())
        cmap = plt.get_cmap('viridis')

        # Create figure
        plt.figure(figsize=(14, 6), dpi=600)

        # For each iteration, compute overall accuracy vs train percolation and plot the curve
        for iter in iters:
            df_iter = df[df['test_iter'] == iter]
            # Group by train_percolation and compute the mean 'correct' value (overall accuracy)
            acc = df_iter.groupby('train_percolation')['correct'].mean().reset_index()
            acc.sort_values('train_percolation', inplace=True)
            plt.plot(
                acc['train_percolation'],
                acc['correct'],
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=6,
                label=iter,
                color=cmap(norm(iter)),
            )

        # Set labels, title, and legend
        plt.xlabel('Train Percolation')
        plt.xlim(0, 1)
        plt.ylabel('Overall Test Accuracy')
        plt.ylim(0, 1)
        plt.title('Overall Test Accuracy vs Train Percolation', fontsize=config.font_size * 1.2)
        plt.legend(title='Iterations', loc='upper left', bbox_to_anchor=(1.05, 1))

        # Save and close the plot
        plt.savefig(f'outputs/tests/{test_name}/overall_acc_vs_perc.pdf', bbox_inches='tight')
        plt.close()

    else:
        raise ValueError(f'Invalid plot_type: {plot_type}.')


def plot_mazes(
    inputs: torch.Tensor | None = None,
    solutions: torch.Tensor | None = None,
    predictions: torch.Tensor | None = None,
    file_name: str = 'outputs/visuals/mazes/mazes',
) -> None:
    """Plot mazes inputs, solutions, and/or predictions, batched or not, and save the plot to a file."""
    if all(x is None for x in [inputs, solutions, predictions]):
        raise ValueError('At least one of inputs, solutions, or predictions must be provided.')

    # Ensure valid batch size
    for tensor, name in zip([inputs, solutions, predictions], ['Inputs', 'Solutions', 'Predictions'], strict=False):
        if tensor is not None and tensor.size(0) == 0:
            raise ValueError(f'{name} tensor has zero batch size.')

    # Ensure tensors have batch dimension
    if inputs is not None and inputs.dim() == 3:
        inputs = inputs.unsqueeze(0)
    if solutions is not None and solutions.dim() == 2:
        solutions = solutions.unsqueeze(0)
    if predictions is not None and predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)

    batch_size = max(x.size(0) if x is not None else 0 for x in [inputs, solutions, predictions])

    # Filter out None values
    mazes = [
        (maze, title)
        for maze, title in zip([inputs, solutions, predictions], ['Inputs', 'Solutions', 'Predictions'], strict=False)
        if maze is not None
    ]
    num_cols = len(mazes)

    # Create figure with proper spacing
    fig, axes = plt.subplots(batch_size, num_cols, figsize=(3.5 * num_cols, 3.5 * batch_size), dpi=300, squeeze=False)

    for row in range(batch_size):
        for col, (maze, title) in enumerate(mazes):
            ax = axes[row, col] if batch_size > 1 else axes[col]  # Ensure correct indexing
            ax.imshow(
                maze[row].permute(1, 2, 0).cpu().numpy() if title == 'Inputs' else maze[row].cpu().numpy(), cmap='gray'
            )
            ax.set_title(title, fontsize=config.subplot_title_size, pad=10)  # Ensure titles appear
            ax.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.2)  # Increase space between subplots

    # Save the plot
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    plt.savefig(f'{file_name}.pdf', bbox_inches='tight')
    plt.close()


def plot_predictions(test_name: str, correct: bool | None = None) -> None:
    """Plot model predictions, possibly filtering by correctness, and save the plot to a file."""
    # Load the test dataframe
    df = pd.read_csv(f'outputs/tests/{test_name}/results.csv')

    # Get model names
    # model_names = df['model_name'].unique().tolist()

    # Ensure only one maze size for each model
    maze_sizes = df['test_maze_size'].unique()
    if len(maze_sizes) > 1:
        raise NotImplementedError(f'Expected single maze size for each model, got {maze_sizes = }')
