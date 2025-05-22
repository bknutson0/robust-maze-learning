import numpy as np

from src.utils.config import TDAParameters
from src.utils.tda import tda


def main() -> None:
    """Main function to run TDA analysis."""
    # Select TDA parameters
    params = TDAParameters()
    params.model_name = [
        'models/dt_net/original.pth',
        'models/dt_net/2025-04-17_13:30:06/best.pth',  # trained on percolation 0.0
        'models/pi_net/original.pth',
        'models/it_net/2025-03-27_16:16:36/best.pth',  # trained on percolation 0.0
    ]
    params.num_mazes = 100
    params.maze_size = [9, 19, 29, 39, 49, 59, 69]
    params.iters = list(range(3001, 3401))
    params.percolation = [0.0]
    params.dtype = np.float64

    # Perform TDA
    tda(params=params)


if __name__ == '__main__':
    main()
