"""Helper functions for data downloading, loading and preprocessing."""

from configparser import ConfigParser
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from pathlib import Path
from pandas import DataFrame


def get_api_key(name_of_api_key: str):
    """
    Return api key from key_config.cfg.

    Params:
        name_of_api_key (str): Name of the API Key from where
                                the key is loaded.
    """
    config = ConfigParser()
    config.read('../config/keys_config.cfg')
    return config[name_of_api_key]['api_key']


def load_stock_data(ticker: str) -> pd.DataFrame():
    """
    Load stock csv data into a pd.DataFrame.

    Params:
        ticker (str): Name of the Stock that will be loaded.

    Return:
        stock_dataframe (pd.DataFrame): Dataframe with stock data.
    """
    path_to_data = Path(__file__).resolve().parents[2]/'data/raw'
    filename = 'stock_market_data-%s.csv' % ticker
    full_path = Path(path_to_data)/filename
    return pd.read_csv(full_path, sep=',')


def split_train_and_test_data(stock_df: DataFrame, test_ratio: float):
    """
    Split historical stock data into train and test subsets.

    Params:
        stock_df (pd.DataFrame): DataFrame containing stock data.
        test_ration (float): Ratio based on which the data will be split.

    Return:
        train_df (pd.DataFrame): DataFrame with training data.
        test_df (pd.DataFrame): DataFrame with test data.
    """
    training_ratio = 1 - test_ratio

    train_size = int(training_ratio * len(stock_df))
    test_size = int(test_ratio * len(stock_df))

    train_df = stock_df[:train_size][['date', '4. close']]
    test_df = stock_df[-test_size:][['date', '4. close']]
    return train_df, test_df


def download_stock_data(api_key: str, ticker: str, output_dir: Path) -> None:
    """
    Download, sort and save the stock data.

    params:
        api_key (str): Api key for Alpha Vantange from where the data is downloaded.
        ticker (str): Symbol for which we are downloading the data.
        output_dir (pathlib.Path): Path where the data will be stored.
    """
    filename = 'stock_market_data-%s.csv' % ticker
    full_path = Path(output_dir)/filename

    print('Downloading {} stock data from Alpha Vantange'.format(ticker))
    pd_timeseries = TimeSeries(key=api_key, output_format='pandas')

    stock_data, _ = pd_timeseries.get_daily(
        symbol=ticker, outputsize='full')
    # sort the data
    sorted_stock_data = stock_data.sort_values('date')
    sorted_stock_data.to_csv(full_path, mode='w+')
