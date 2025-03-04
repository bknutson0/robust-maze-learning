import pandas as pd

from src.utils.analysis import plot_test_accuracies


def main() -> None:
    """Analyze test results."""
    test_name = '2025-03-04_12:26:21'

    df = pd.read_csv(f'outputs/tests/{test_name}/results.csv')
    print(df.head(20))

    # For each model_name, print train_percolation
    model_names = df['model_name'].unique().tolist()
    for model_name in model_names:
        train_percolation = df[df['model_name'] == model_name]['train_percolation'].unique()
        print(f'{model_name = }, {train_percolation[0] = }')

    plot_test_accuracies(
        test_name,
        'acc_vs_perc',
        filters={'model_name': 'models/dt_net/20250211_174547/best.pth', 'test_iter': 30},
    )

    # #Plot mazes
    # params = Hyperparameters()
    # params.num_mazes = 10
    # params.percolation = 0.99
    # inputs, solutions = load_mazes(params)
    # maze_index = 0
    # inputs = inputs[maze_index : maze_index + 10]
    # solutions = solutions[maze_index : maze_index + 10]
    # print(inputs.sum().item() + solutions.sum().item())
    # plot_mazes(inputs, solutions, None, 'outputs/visuals/mazes/mazes')


if __name__ == '__main__':
    main()
