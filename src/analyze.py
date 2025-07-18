# ruff: noqa: F841

import pandas as pd

from src.utils.analysis import plot_test


def main() -> None:
    """Analyze test results."""
    # dt_net
    dt_net_original_test_name = '2025-04-27_17:08:33'
    # Random initial weights without progressive learning dt-net models (03-01 to 03-03)
    # dt_net_test_name = '2025-03-04_12:26:21'
    # Random initial weights and progressive learning dt-net models (03-11 to 03-13)
    # dt_net_test_name = '2025-03-30_18:36:53'
    # dt_net_test_name = '2025-04-16_17:27:21'  # larger range of maze sizes
    dt_net_test_name = '2025-04-27_14:07:53'  # Add matches_deadend_fill column

    # it_net
    # it_net_test_name = '2025-03-27_16:35:05'
    # it_net_test_name = '2025-03-29_15:40:05'
    # it_net_test_name = '2025-03-29_16:19:02'
    # it_net_test_name = '2025-03-30_11:45:35' # Contractive it_net and num_mazes = 100
    # it_net_test_name = '2025-04-16_17:27:53'  # larger range of maze sizes
    it_net_test_name = '2025-04-28_12:21:42'  # Add matches_deadend_fill column

    # ff_net
    # ff_net_test_name = '2025-04-21_12:48:32'
    ff_net_test_name = '2025-04-27_14:07:46'  # Add matches_deadend_fill column

    # pi_net
    pi_net_test_name = '2025-05-22_12:19:48'

    # deadend_fill
    deadend_fill_test_name = '2025-04-26_21:46:00'

    # test_names = [ff_net_test_name, dt_net_test_name, it_net_test_name]
    test_names = [
        # dt_net_original_test_name,
        # pi_net_test_name,
        ff_net_test_name,
        dt_net_test_name,
        it_net_test_name,
        # deadend_fill_test_name
    ]

    df = pd.read_csv('./outputs/tests/' + pi_net_test_name + '/results.csv')
    # print(df.head(20))
    # Print unique values in each column
    for column in df.columns:
        if column != 'maze_index':
            unique_values = df[column].unique()
            # print(f'{column}: {unique_values}')

    # # For each model_name, print train_percolation
    # model_names = df['model_name'].unique().tolist()
    # for model_name in model_names:
    #     train_percolation = df[df['model_name'] == model_name]['train_percolation'].unique()
    #     print(f'{model_name = }, {train_percolation[0] = }')

    # Plot test accuracy versus test percolation for models trained with different percolations
    plot_test(
        test_names=[dt_net_original_test_name],
        plot_type='acc_vs_perc',
        filters={
            'test_iter': 200,
            'test_maze_size': 9,
        },
    )
    plot_test(
        test_names=[pi_net_test_name],
        plot_type='acc_vs_perc',
        filters={
            'test_iter': 200,
            'test_maze_size': 9,
        },
    )

    # # Plot test accuracy heatmap versus maze size and test percolation for models trained with different percolations
    # for percolation in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    #     plot_test_accuracies(
    #         it_net_test_name,
    #         'acc_vs_size_perc',
    #         filters={
    #             'test_iter': 30,
    #             'train_percolation': percolation,
    #         },
    #     )

    # # Create heatmap for agreement with deadend_fill
    # plot_test(
    #     test_names,
    #     plot_type='value_vs_size_perc',
    #     filters={'train_percolation': [0.0, 0.001, 0.1, 0.5, 0.99], 'test_iter': 200},
    #     # filters={'test_iter': 200},
    #     value='matches_deadend_fill',
    # )

    # # Plot test accuracy versus train percolation
    # plot_test_accuracies(
    #     test_names,
    #     'acc_vs_perc',
    #     filters={
    #         'train_percolation': [0.0, 0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    #         'test_iter': 200,
    #         'test_maze_size': 9,
    #     },
    # )

    # # Plot overall test accuracy versus test percolation
    # plot_test_accuracies(
    #     test_names,
    #     'overall_acc_vs_perc',
    #     filters={
    #         'test_iter': 200,
    #         'test_maze_size': [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
    #     },
    # )

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

    # #  Plot test accuracy versus test iters
    # plot_test(
    #     test_names,
    #     plot_type='accuracy_by_iter',
    #     filters={'train_percolation': [0.0, 0.001, 0.01, 0.1, 0.5, 0.99], 'test_maze_size': 9},
    # )


if __name__ == '__main__':
    main()
