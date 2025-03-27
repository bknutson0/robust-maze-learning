from src.utils.config import TestParameters
from src.utils.model_loading import get_all_model_names
from src.utils.testing import test


def main() -> None:
    """Perform test according to test parameters."""
    model_names = get_all_model_names()

    # dt-net
    # Pre-trained dt-net models (02-26 to 02-28)
    # model_names = [name for name in model_names if ('dt_net' in name) and ('best' in name) and ('2025-02-2' in name)]
    # Random initial weights dt-net models (03-01 to 03-03)
    # model_names = [name for name in model_names if ('it_net' in name) and ('best' in name) and ('2025-03-0' in name)]
    # Random initial weights and progressive learning dt-net models (03-11 to 03-13)
    # model_names = [name for name in model_names if ('dt_net' in name) and ('best' in name) and ('2025-03-1' in name)]

    # it-net
    # Random iters, JFB, without contraction it-net models (03-25 to 03-26)
    model_names = [
        name
        for name in model_names
        if ('dt_net' in name) and ('best' in name) and ('2025-03-25' in name or '2025-03-26' in name)
    ]
    # Random iters, JFB, with contraction it-net models (03-25 to 03-26)
    model_names = [
        name
        for name in model_names
        if ('dt_net' in name) and ('best' in name) and ('2025-03-25' in name or '2025-03-26' in name)
    ]

    # Define test parameters
    test_params = TestParameters()
    test_params.model_name = sorted(model_names)
    test_params.maze_size = [9, 13, 17, 21]
    test_params.num_mazes = 10000
    test_params.iters = [30, 60, 100]

    # Perform test
    test(test_params)


if __name__ == '__main__':
    main()
