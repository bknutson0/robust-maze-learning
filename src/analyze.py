from src.utils.analysis import plot_test_results


def main() -> None:
    """Analyze test results."""
    test_name = '2025-02-13_12:08:52'

    plot_test_results(test_name)


if __name__ == '__main__':
    main()
