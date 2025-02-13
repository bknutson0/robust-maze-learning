from src.utils.config import TestParameters
from src.utils.testing import test


def main() -> None:
    """Perform test according to test parameters."""
    # Define test parameters
    test_params = TestParameters()

    # Perform test
    test(test_params)


if __name__ == '__main__':
    main()
