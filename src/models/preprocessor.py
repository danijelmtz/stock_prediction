"""Script used for preprocessing data needed for LSTM."""
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


def get_trainseqX_and_outcomeY(dataframe: DataFrame, window_size: int, offset: int):
    """
    Split the dataframe in a training sequence X and outcome value Y.

    Params:
        dataframe (pd.DataFrame): Dataframe containing stock data.
        window_size (int): len of the training sequence X and outcome value Y.
        offset (int): starting position for the extraction.

    Return:
        train_x (np_array): Array containing input data for LSTM.
        train_y (np_array): Array containing labels to input data for LSTM.
    """
    dataframe = scale_data(dataframe)

    X = [dataframe[i-window_size:i] for i in range(offset, len(dataframe))]
    y = [dataframe[i] for i in range(offset, len(dataframe))]
    return np.array(X), np.array(y)


def scale_data(dataframe_to_scale: DataFrame):
    """
    Scale training and test data.

    Params:
        dataframe_to_scale(pd.DataFrame): DataFrame to scale.

    Return:
        scaled_dataframe (pd.DataFrame): Scaled Dataframe using StandardScaler.
    """
    scalar = StandardScaler()
    scaled_df = scalar.fit_transform(dataframe_to_scale[['4. close']])
    return scaled_df[:dataframe_to_scale.shape[0]]


def get_testseqX(original_df: DataFrame, window_size: int, test_df: DataFrame):
    """
    Get test seq X for LSTM.

    Params:
        dataframe (pd.DataFrame): Dataframe containing stock data.
        window_size (int): len of the test sequence X.
        test_df (pd.DataFrame): Dataframe with test data.
    """
    close_prices = original_df['4. close'][len(
        original_df) - len(test_df) - window_size:].values

    scaler = StandardScaler()
    close_prices = close_prices.reshape(-1, 1)
    close_prices = scaler.fit_transform(close_prices)

    X_test = [close_prices[i-window_size:i, 0]
              for i in range(window_size, close_prices.shape[0])]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test


def undo_scaling(original_data: DataFrame, data_to_unscale: DataFrame):
    """
    Inverse transform scaled data.

    Params:
        original_data (pd.DataFrame): Dataframe used to fit the scaling.
        data_to_unscale (pd.DataFrame): Dataframe to unscale.
    Return:
        unscaled_data (pd.DataFrame): Unscaled Dataframe.
    """
    scalar = StandardScaler()
    scaled_df = scalar.fit_transform(original_data[['4. close']])
    return scalar.inverse_transform(data_to_unscale)
