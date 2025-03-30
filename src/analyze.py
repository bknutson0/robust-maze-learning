import pandas as pd

from src.utils.analysis import plot_test_accuracies


def main() -> None:
    """Analyze test results."""
    # Random initial weights without progressive learning dt-net models (03-01 to 03-03)
    # test_name = '2025-03-04_12:26:21'
    # Random initial weights and progressive learning dt-net models (03-11 to 03-13)
    # test_name = '2025-03-26_18:12:08'
    # test_name = '2025-03-27_16:35:05'
    # test_name = '2025-03-29_15:40:05'
    # test_name = '2025-03-29_16:19:02'
    test_name = '2025-03-30_11:45:35'

    df = pd.read_csv(f'outputs/tests/{test_name}/results.csv')
    # print(df.head(20))
    # Print unique values in each column
    for column in df.columns:
        unique_values = df[column].unique()
        print(f'{column}: {unique_values}')

    # # For each model_name, print train_percolation
    # model_names = df['model_name'].unique().tolist()
    # for model_name in model_names:
    #     train_percolation = df[df['model_name'] == model_name]['train_percolation'].unique()
    #     print(f'{model_name = }, {train_percolation[0] = }')

    plot_test_accuracies(
        test_name,
        'acc_vs_perc',
        filters={'test_iter': 100, 'test_maze_size': 9},
        # filters={'model_name': 'models/dt_net/20250211_174547/best.pth', 'test_iter': 30},
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
