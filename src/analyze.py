from src.utils.analysis import plot_test_accuracies
from src.utils.config import PlotConfig

# Instantiate and apply the plot configuration
plot_config = PlotConfig()
plot_config.apply()


def main() -> None:
    """Analyze test results."""
    # dt_net
    # Random initial weights without progressive learning dt-net models (03-01 to 03-03)
    # dt_net_test_name = '2025-03-04_12:26:21'
    # Random initial weights and progressive learning dt-net models (03-11 to 03-13)
    # dt_net_test_name = '2025-03-30_18:36:53'
    dt_net_test_name = '2025-04-16_17:27:21'  # larger range of maze sizes

    # it_net
    # it_net_test_name = '2025-03-27_16:35:05'
    # it_net_test_name = '2025-03-29_15:40:05'
    # it_net_test_name = '2025-03-29_16:19:02'
    # it_net_test_name = '2025-03-30_11:45:35' # Contractive it_net and num_mazes = 100
    it_net_test_name = '2025-04-16_17:27:53'  # larger range of maze sizes

    # df = pd.read_csv(f'outputs/tests/{it_net_test_name}/results.csv')
    # print(df.head(20))
    # # Print unique values in each column
    # for column in df.columns:
    #     unique_values = df[column].unique()
    #     print(f'{column}: {unique_values}')

    # # For each model_name, print train_percolation
    # model_names = df['model_name'].unique().tolist()
    # for model_name in model_names:
    #     train_percolation = df[df['model_name'] == model_name]['train_percolation'].unique()
    #     print(f'{model_name = }, {train_percolation[0] = }')

    test_names = [dt_net_test_name, it_net_test_name]
    # # Plot test accuracy versus test percolation for models trained with different percolations
    # plot_test_accuracies(
    #     it_net_test_name,
    #     'acc_vs_perc',
    #     filters={
    #         'test_iter': 300,
    #         'test_maze_size': 9,
    #         'train_percolation': [0.0, 0.2, 0.4, 0.6, 0.8, 0.99],
    #     },
    # )

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

    # Plot overall test accuracy for each model
    plot_test_accuracies(
        it_net_test_name,
        'acc_vs_size_perc',
        filters={'train_percolation': [0.0, 0.001, 0.01, 0.1, 0.5, 0.99], 'test_iter': 200},
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
