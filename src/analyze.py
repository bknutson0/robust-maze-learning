import pandas as pd

from src.utils.analysis import plot_test_accuracies


def main() -> None:
    """Analyze test results."""
    test_name = '2025-02-13_14:52:51'

    plot_test_accuracies(test_name)

    df = pd.read_csv(f'outputs/tests/{test_name}/results.csv')

    # model_names = df['model_name'].unique().tolist()
    # for model_name in model_names:
    #     train_percolation = df[df['model_name'] == model_name]['train_percolation'].unique()
    #     if len(train_percolation) == 1:
    #         print(f'{model_name = }, {train_percolation = }')
    #     else:
    #         raise ValueError(f'Expected single train_percolation value for {model_name = }, got {train_percolation = }')


if __name__ == '__main__':
    main()
