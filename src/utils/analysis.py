import logging
import os
from typing import Any

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.utils.config import LOGGING_LEVEL, PlotConfig

config = PlotConfig()
config.apply()  # Explicitly apply global font and style settings

# Create logger
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL, logging.INFO),  # Default to INFO if LOGGING_LEVEL is invalid
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


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


def plot_test_accuracies(test_names: str | list[str], plot_type: str, filters: dict[str, Any] | None = None) -> None:
    """Plot the results of one or more tests and save the plot(s) to file."""
    # Allow test_names to be either a single test or a list of tests.
    if isinstance(test_names, str):
        test_names = [test_names]

    if plot_type == 'overall_acc_vs_perc':
        # ... (existing overall_acc_vs_perc code remains unchanged) ...
        test_data = []
        test_iters = []
        for test_name in test_names:
            df_temp = pd.read_csv(f'outputs/tests/{test_name}/results.csv')
            df_temp = filter_dataframe(df_temp, filters)
            unique_iter = df_temp['test_iter'].unique()
            if len(unique_iter) != 1:
                raise ValueError(f'Expected single test_iter for {test_name}, got {unique_iter}')
            test_iter = unique_iter[0]
            test_iters.append(test_iter)

            model_names = df_temp['model_name'].unique()
            if all('dt_net' in model_name for model_name in model_names):
                model_label = 'RNN'
            elif all('it_net' in model_name for model_name in model_names):
                model_label = 'INN'
            else:
                raise ValueError('Model names contain both dt_net and it_net. Unable to determine model type.')

            acc = df_temp.groupby('train_percolation')['correct'].mean().reset_index()
            acc.sort_values('train_percolation', inplace=True)
            test_data.append((test_iter, model_label, acc))

        norm = mcolors.LogNorm(vmin=min(test_iters), vmax=max(test_iters))
        cmap = plt.get_cmap('viridis')

        plt.figure(figsize=(14, 6), dpi=600)
        for i, (test_iter, model_label, acc) in enumerate(test_data):
            marker = 's' if i == 1 else 'o'
            linestyle = '--' if i == 1 else '-'
            plt.plot(
                acc['train_percolation'],
                acc['correct'],
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=6,
                label=f'{model_label}',
                color=cmap(norm(test_iter)),
            )

        plt.xlabel('Train Percolation')
        plt.xlim(0, 0.01)
        plt.ylabel('Overall Test Accuracy')
        plt.ylim(0, 1)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        for test_name in test_names:
            plt.savefig(f'outputs/tests/{test_name}/overall_acc_vs_perc.pdf', bbox_inches='tight')
        plt.close()

    else:
        # For other plot types, loop over each test and produce separate plots (or a summary for heatmaps)
        if plot_type == 'acc_vs_perc':
            for test_name in test_names:
                df = pd.read_csv(f'outputs/tests/{test_name}/results.csv')
                df = filter_dataframe(df, filters)
                model_names = df['model_name'].unique().tolist()

                if plot_type == 'acc_vs_perc':
                    train_percolations = []
                    for model_name in model_names:
                        train_percolation = df[df['model_name'] == model_name]['train_percolation'].unique()
                        if len(train_percolation) == 1:
                            train_percolations.append(train_percolation[0])
                        else:
                            raise ValueError(
                                f'Expected single train_percolation value for {model_name = }, '
                                f'got {train_percolation = }'
                            )

                    model_names, train_percolations = map(
                        list,
                        zip(
                            *sorted(zip(model_names, train_percolations, strict=False), key=lambda x: x[1]),
                            strict=False,
                        ),
                    )

                    maze_sizes = df['test_maze_size'].unique()
                    if len(maze_sizes) > 1:
                        raise NotImplementedError(f'Expected single maze size for each model, got {maze_sizes = }')

                    plt.figure(figsize=(14, 6), dpi=600)
                    small_value = 1e-3
                    color_mapping_values = [tp if tp > 0 else small_value for tp in train_percolations]
                    cmap_local = cm.get_cmap('plasma', len(train_percolations))
                    norm_local = mcolors.Normalize(vmin=min(color_mapping_values), vmax=max(color_mapping_values))
                    colors = [cmap_local(norm_local(tp if tp > 0 else small_value)) for tp in train_percolations]

                    for model_name, train_percolation, color in zip(
                        model_names, train_percolations, colors, strict=False
                    ):
                        df_subset = df[df['model_name'] == model_name]
                        df_subset = df_subset.groupby('test_percolation')['correct'].mean().reset_index()
                        df_subset = df_subset.sort_values(by='test_percolation')
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
                        plt.axvline(x=train_percolation, color=color, linestyle='dashed', linewidth=5, zorder=1)

                    plt.xlabel('Test Percolation')
                    plt.ylabel('Test Accuracy')
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.title('Test Accuracy vs Test Percolation', fontsize=config.font_size * 1.2)
                    legend = plt.legend(title='Train Percolation', loc='upper left', bbox_to_anchor=(1.05, 1))
                    plt.setp(
                        legend.get_title(), fontweight=config.legend_title_weight, fontsize=config.legend_title_fontsize
                    )
                    plt.savefig(f'outputs/tests/{test_name}/acc_vs_perc.pdf', bbox_inches='tight')
                    plt.close()

        elif plot_type == 'acc_vs_size_perc':
            # Gather heatmap data for each test into a summary list.
            summary_list = []
            for test_name in test_names:
                df = pd.read_csv(f'outputs/tests/{test_name}/results.csv')
                df = filter_dataframe(df, filters)

                # Validate required axes and allowed filters.
                required_axes = {'test_maze_size', 'test_percolation'}
                for col in required_axes:
                    if df[col].nunique() <= 1:
                        raise ValueError(f"Column '{col}' must have multiple unique values for a heatmap.")
                allowed_filters = {'model_name', 'test_iter'}
                filters = filters or {}
                for col, value in filters.items():
                    if col not in df.columns:
                        raise ValueError(f"Invalid column: '{col}' does not exist in the dataframe.")
                    if col in required_axes:
                        raise ValueError(f"Cannot filter on required axis column: '{col}'")
                    if col in allowed_filters and value not in df[col].unique():
                        raise ValueError(f"Invalid value for '{col}': {value}. Must be one of {df[col].unique()}.")

                multi_value_cols = {col for col in allowed_filters if df[col].nunique() > 1}
                if multi_value_cols - filters.keys():
                    raise ValueError(
                        f'Columns {multi_value_cols - filters.keys()} have multiple values. '
                        f'Please specify one in `filters`.'
                    )

                filtered_df = filter_dataframe(df, filters)

                model_names = filtered_df['model_name'].unique()
                if len(model_names) != 1:
                    raise ValueError(f'Expected exactly one model_name after filtering, but got {len(model_names)}.')
                model_name = model_names[0]
                if 'dt_net' in model_name:
                    model_name = 'dt_net'
                elif 'it_net' in model_name:
                    model_name = 'it_net'
                else:
                    raise ValueError(f'Unexpected model_name found: {model_name}')

                train_percolation_vals = filtered_df['train_percolation'].unique()
                if len(train_percolation_vals) != 1:
                    raise ValueError(
                        f'Expected single train_percolation value for {model_name}, got {train_percolation_vals}.'
                    )
                train_percolation = train_percolation_vals[0]

                test_iter_vals = filtered_df['test_iter'].unique()
                if len(test_iter_vals) != 1:
                    raise ValueError(f'Expected single test_iter value for {model_name}, got {test_iter_vals}.')
                test_iter = test_iter_vals[0]

                heatmap_data = filtered_df.groupby(['test_percolation', 'test_maze_size'])['correct'].mean().unstack()
                x_labels = heatmap_data.columns.values
                y_labels = heatmap_data.index.values
                heatmap_values = heatmap_data.values

                dx = (x_labels[-1] - x_labels[0]) / (len(x_labels) - 1) if len(x_labels) > 1 else 1
                x_extent = [x_labels[0] - dx / 2, x_labels[-1] + dx / 2]
                extent = [x_extent[0], x_extent[1], y_labels[0], y_labels[-1]]

                train_maze_size_vals = filtered_df['train_maze_size'].unique()
                if len(train_maze_size_vals) != 1:
                    raise ValueError(
                        f'Expected single train_maze_size value for {model_name}, got {train_maze_size_vals}.'
                    )
                train_maze_size = train_maze_size_vals[0]

                # Store the data for this test.
                summary_list.append(
                    {
                        'test_name': test_name,
                        'train_percolation': train_percolation,
                        'test_iter': test_iter,
                        'model_name': model_name,
                        'heatmap_values': heatmap_values,
                        'x_labels': x_labels,
                        'y_labels': y_labels,
                        'extent': extent,
                        'train_maze_size': train_maze_size,
                    }
                )

            # Order the summary list by training percolation (ascending)
            summary_list.sort(key=lambda d: d['train_percolation'])

            # Create a summary figure with one row of subplots.
            n_plots = len(summary_list)
            fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
            if n_plots == 1:
                axes = [axes]
            for ax, data in zip(axes, summary_list, strict=False):
                cax = ax.imshow(
                    data['heatmap_values'], cmap='coolwarm', aspect='auto', origin='lower', extent=tuple(data['extent'])
                )
                ax.set_xticks(data['x_labels'])
                ax.set_xticklabels(data['x_labels'])
                ax.set_yticks(data['y_labels'])
                ax.set_yticklabels(data['y_labels'])
                ax.set_xlabel('Test Maze Size')
                ax.set_ylabel('Test Percolation')
                title_text = (
                    f'{data["model_name"]} | Train Perc: {data["train_percolation"]:.3f} | Iter: {data["test_iter"]}'
                )
                ax.set_title(title_text)
                ax.scatter(
                    data['train_maze_size'],
                    data['train_percolation'],
                    marker='*',
                    s=400,
                    color='gold',
                    edgecolors='gold',
                    linewidths=1.5,
                    zorder=4,
                    clip_on=False,
                    label='training distribution',
                )
                ax.legend()
            fig.colorbar(cax, ax=axes, orientation='vertical', label='Test Accuracy')
            plt.tight_layout()
            summary_filename = 'outputs/tests/acc_vs_size_perc_summary.pdf'
            plt.savefig(summary_filename, bbox_inches='tight')
            logger.info(f'Saved summary heatmap to {summary_filename}')
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
