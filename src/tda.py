from src.utils.config import TDAParameters
from src.utils.tda import tda


def main() -> None:
    """Main function to run TDA analysis."""
    # Select TDA parameters
    params = TDAParameters()
    params.num_mazes = 10

    # Perform TDA
    df = tda(params=params)

    # Analyze TDA results
    print(df.head(10))


if __name__ == '__main__':
    main()
