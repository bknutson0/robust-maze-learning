import pandas as pd

from src.utils.analysis import plot_test_accuracies


def main() -> None:
    """Analyze test results."""
    test_name = '2025-02-16_14:34:51'

    df = pd.read_csv(f'outputs/tests/{test_name}/results.csv')

    # For each model_name, print train_percolation
    model_names = df['model_name'].unique().tolist()
    for model_name in model_names:
        train_percolation = df[df['model_name'] == model_name]['train_percolation'].unique()
        print(f'{model_name = }, {train_percolation[0] = }')

    plot_test_accuracies(
        test_name,
        'acc_vs_size_perc',
        filters={'model_name': 'models/dt_net/20250211_174547/best.pth', 'test_iter': 300},
    )


if __name__ == '__main__':
    main()
