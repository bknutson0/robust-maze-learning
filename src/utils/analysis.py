import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


def plot_test_results(test_name: str) -> None:
    """Plot the results of a test and save the plot to a file."""
    # Load the test dataframe
    df = pd.read_csv(f'outputs/tests/{test_name}/results.csv')

    # Get model names
    model_names = df['model_name'].unique().tolist()

    # Get train percolations
    train_percolations = []
    for model_name in model_names:
        train_percolation = df[df['model_name'] == model_name]['train_percolation'].unique()
        if len(train_percolation) == 1:
            train_percolations.append(train_percolation[0])
        else:
            raise ValueError(f'Expected single train_percolation value for {model_name = }, got {train_percolation = }')

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
    cmap = cm.get_cmap('plasma', len(train_percolations))
    norm = mcolors.Normalize(vmin=min(train_percolations), vmax=max(train_percolations))
    colors = [cmap(norm(tp)) for tp in train_percolations]

    # Plot accuracy versus test percolation for each model
    for model_name, train_percolation, color in zip(model_names, train_percolations, colors, strict=False):
        df_subset = df[df['model_name'] == model_name]
        df_subset = df_subset.groupby('test_percolation')['correct'].mean().reset_index()  # Get mean accuracy
        df_subset = df_subset.sort_values(by='test_percolation')  # Ensure sorted order

        plt.plot(
            df_subset['test_percolation'],
            df_subset['correct'],
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=5,
            label=f'{train_percolation:.3f}',
            color=color,
            zorder=2,
        )

        # Add vertical dashed line at train_percolation
        plt.axvline(x=train_percolation, color=color, linestyle='dashed', linewidth=5, zorder=1)

    # Set plot title and labels
    plt.xlabel('Test Percolation')
    plt.ylabel('Accuracy')
    plt.legend(title='Train Percolation', loc='upper left', bbox_to_anchor=(1.05, 1))

    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0.9, 1)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'outputs/tests/{test_name}/plot.png', bbox_inches='tight')
    plt.close()
