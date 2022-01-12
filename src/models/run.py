import models.preprocessor as preprocessing
import models.model as m
import numpy as np
import models.helper_functions as hf
from pandas import DataFrame


def train_and_eval_lstm(full_df: DataFrame,
                        train_df: DataFrame,
                        test_df: DataFrame,
                        window_size: int,
                        epochs: int,
                        hidden_units: int):
    """
    Run training and evaluation of a simple LSTM.

    Params:
        full_df (pd.Dataframe): Dataframe containing full data (train+test).
        train_df (pd.Dataframe): Dataframe containing train data.
        test_df (pd.Dataframe): Dataframe containing test data.
        window_size (int): Number of points in history the model is looking at.
        epochs (int): Number of epochs to train the model for.
        hidden_units (int): How many hidden units the LSTM layer will have.

    Return:
        mape_lstm (float): Mean Absolute Percentage Error of the model.
        rmse_lstm (float): Root Mean Squared Error of the model.
        test_df (pd.Dataframe): Dataframe containing test labels and model predictions.
    """
    # Scale training data.

    train_X, train_y = preprocessing.get_trainseqX_and_outcomeY(
        train_df, window_size, window_size)

    lstm_model = m.lstm(train_X,  hidden_units)

    lstm_model.fit(train_X, train_y,
                   epochs=epochs,
                   batch_size=32,
                   verbose=1,
                   validation_split=0.1,
                   shuffle=False)
    test_X = preprocessing.get_testseqX(
        original_df=full_df, window_size=window_size, test_df=test_df)

    predictions = lstm_model.predict(test_X)

    unscaled_predictions = preprocessing.undo_scaling(test_df, predictions)

    test_df['predicted'] = unscaled_predictions

    mape_lstm = hf.mean_absolute_percentage_error(
        np.array(test_df['4. close']), np.array(test_df['predicted']))
    rmse_lstm = hf.root_mean_squared_error(
        np.array(test_df['4. close']), np.array(test_df['predicted']))
    return mape_lstm, rmse_lstm, test_df, lstm_model


def run_evaluation(model, full_df: DataFrame, test_df: DataFrame, window_size: int):
    """
    Evaluate model on the test dataframe.

    Params:
        full_df (pd.Dataframe): Dataframe containing full data (train+test).
        test_df (pd.Dataframe): Dataframe containing test data.
        window_size (int): Number of points in history the model is looking at.

    Return:
        mape_lstm (float): Mean Absolute Percentage Error of the model.
        rmse_lstm (float): Root Mean Squared Error of the model.
        test_df (pd.Dataframe): Dataframe containing test labels and model predictions.
    """
    test_X = preprocessing.get_testseqX(
        original_df=full_df, window_size=window_size, test_df=test_df)
    # Use model to get the predictions.
    predictions = model.predict(test_X)

    # Return the predicted results to original scale.
    unscaled_predictions = preprocessing.undo_scaling(full_df, predictions)

    # Add predictions to the test data.
    test_df['predicted'] = unscaled_predictions

    # Calculate MAPE and RSME.
    mape_lstm = hf.mean_absolute_percentage_error(
        np.array(test_df['4. close']), np.array(test_df['predicted']))
    rmse_lstm = hf.root_mean_squared_error(
        np.array(test_df['4. close']), np.array(test_df['predicted']))
    return mape_lstm, rmse_lstm, test_df


def get_baseline(full_dataframe: DataFrame, train_df: DataFrame, window_baseline: int):
    """
    Get MAPE, RSME and results of the baseline model.

    Params:
        full_dataframe (pd.DataFrame): Dataframe with all stock prices.
        train_df (pd.DataFrame): Dataframe with only train examples.
        window_baseline (int): Number of past points.

    Return:
        mape_lstm (float): Mean Absolute Percentage Error of the model.
        rmse_lstm (float): Root Mean Squared Error of the model.
        test_df (pd.Dataframe): Dataframe containing model predictions.
    """
    full_dataframe['baseline'] = full_dataframe['4. close'].rolling(
        window_baseline).mean()

    mape_baseline = hf.mean_absolute_percentage_error(np.array(full_dataframe[len(
        train_df):]['4. close']), np.array(full_dataframe[len(train_df):]['baseline']))

    rmse_baseline = hf.root_mean_squared_error(np.array(full_dataframe[len(
        train_df):]['4. close']), np.array(full_dataframe[len(train_df):]['baseline']))
    return mape_baseline, rmse_baseline, full_dataframe
