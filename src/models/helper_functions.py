"""Helper functions for model evaluation."""
import numpy as np
from pathlib import Path
from keras.models import load_model


def root_mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate the Root Mean Squared Error(RMSE).

    Params:
        y_true (np.array): List of true values.
        y_pred (np.array): List of predicted values.
    Output:
        RMSE (float): Root Mean Squared Error

    """
    return np.sqrt(np.mean((y_true-y_pred)**2))


def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %.

    Params:
        y_true (np.array): List of true values.
        y_pred (np.array): List of predicted values.
    Output:
        MAPE (float): Mean Absolute Percentage Error in %.
    """
    return np.mean(np.abs((y_true-y_pred)/y_true))*100


def save_trained_model(model):
    """
    Save trained model.

    Params:
        model (tf.model): Tensorflow model that we want to save.
    """
    path_to_save_model = Path(__file__).resolve().parents[2]/'trained_models/'
    print('Saving model....')
    model.save(path_to_save_model/'best_model.h5')


def load_trained_model():
    """
    Load trained model.

    Returns:
        train_model (tf.model): Returns a trained model.
    """
    # currently we only have one  model that is being loaded.
    # TODO: Make this function more flexible.
    path_to_trained_model = Path(__file__).resolve(
    ).parents[2]/'trained_models/best_model.h5'
    return load_model(path_to_trained_model)
