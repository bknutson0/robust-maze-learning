from typing import Any

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_test_accuracies(test_name: str, plot_type: str, filters: dict[str, Any] | None = None) -> None:
    """Plot the results of a test and save the plot to a file."""
    # Load the test dataframe
    df = pd.read_csv(f'outputs/tests/{test_name}/results.csv')

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
            # [0.003, 0.9, 0.0, 0.3, 0.01, 0.6, 0.1, 0.5, 0.4, 0.03, 0.7, 0.001]
            if train_percolation in [0.003, 0.9, 0.0, 0.3, 0.01, 0.6, 0.1, 0.5, 0.4, 0.03, 0.7, 0.001]:
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

                # # Compute and plot average 'matches_solution' as well
                # df_subset = df[df['model_name'] == model_name]
                # df_subset = df_subset.groupby('test_percolation')['matches_solution'].mean().reset_index()
                # df_subset = df_subset.sort_values(by='test_percolation')
                # plt.plot(
                #     df_subset['test_percolation'],
                #     df_subset['matches_solution'],
                #     marker='o',
                #     linestyle='-',
                #     linewidth=5,
                #     markersize=10,
                #     label=f'{train_percolation:.3f}',
                #     color='black',
                #     zorder=3,
                # )

                # Add vertical dashed line at train_percolation
                plt.axvline(x=train_percolation, color=color, linestyle='dashed', linewidth=5, zorder=1)

        # Set plot title and labels
        plt.xlabel('Test Percolation')
        plt.ylabel('Test Accuracy')
        plt.legend(title='Train Percolation', loc='upper left', bbox_to_anchor=(1.05, 1))

        # Set axis limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # # Adjust layout to prevent clipping
        # plt.tight_layout()

        # Save the plot
        plt.savefig(f'outputs/tests/{test_name}/acc_vs_perc.png', bbox_inches='tight')
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
        filtered_df = df.copy()
        for col, val in filters.items():
            filtered_df = filtered_df[filtered_df[col] == val]

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
        ax.set_title('Test Accuracy')
        ax.invert_yaxis()

        # Save the plot
        plt.savefig(f'outputs/tests/{test_name}/acc_vs_size_perc.png', bbox_inches='tight')
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
