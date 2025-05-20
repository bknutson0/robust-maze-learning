from src.utils.config import TDAParameters
from src.utils.tda import tda


def main() -> None:
    """Main function to run TDA analysis."""
    # Select TDA parameters
    params = TDAParameters()
    params.model_name = [
        'dt_net/original.pth',
        'models/dt_net/2025-04-17_13:30:06/best.pth',  # dt_net trained on percolation 0.0
        'pi_net/original.pth',
        'models/it_net/2025-03-27_16:16:36/best.pth',  # it_net trained on percolation 0.0
    ]
    params.num_mazes = 3
    params.percolation = [0]

    # Perform TDA
    df = tda(params=params)

    # Analyze TDA results
    print(df.head(10))


if __name__ == '__main__':
    main()
