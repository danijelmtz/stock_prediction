"""Script used for download raw stock data."""

import utils as utils
from pathlib import Path


def main():
    """Main function to used for downloading raw data."""
    # path is currently hard-coded. Should be read from a configuration
    # In future the data should be read from a database.
    path_to_data = Path(__file__).resolve().parents[2]/'data/raw'

    # API Key to download the data.
    API_KEY = utils.get_api_key(name_of_api_key='alphavantage')

    # List of stocks we are interested in.
    list_of_tickers = ['FB', 'AAPL', 'GOOG', 'NFLX', 'AMZN']

    for ticker in list_of_tickers:
        utils.download_stock_data(api_key=API_KEY, ticker=ticker,
                                  output_dir=path_to_data)


if __name__ == '__main__':
    main()
