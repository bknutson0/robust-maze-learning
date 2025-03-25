from src.utils.config import TestParameters
from src.utils.model_loading import get_all_model_names


def main() -> None:
    """Perform test according to test parameters."""
    model_names = get_all_model_names()

    # Retrieve best dt_net models from 2025-03-18
    best_dt_net_model_names = [
        name for name in model_names if ('dt_net' in name) and ('best' in name) and ('2025-03-18_17:52:53' in name)
    ]
    # Retrieve best it_net models from 2025-03-19
    best_it_net_model_names = [
        name for name in model_names if ('it_net' in name) and ('best' in name) and ('2025-03-19_17:52:53' in name)
    ]

    # Define test parameters
    test_params = TestParameters()
    test_params.model_name = best_dt_net_model_names
    test_params.maze_size = [9]
    test_params.num_mazes = 10000
    test_params.iters = [30]

    # Perform test
    # test(test_params)


if __name__ == '__main__':
    main()
