from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np


def lstm(input: np.array, hidden_units: int):
    """
    Simple LSTM model used to predict stock prices.

    Params:
        input (np.array): Input array used to create the input an output shape of the model.
        hidden_units (int): Number of hidden units of the LSTM layer.

    Return:
        model (Keras.model): LSTM model that can be used for training and evaluation.
    """
    model = Sequential()
    model.add(LSTM(units=hidden_units, return_sequences=True,
              input_shape=(input.shape[1], input.shape[2])))
    model.add(LSTM(units=hidden_units, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=input.shape[2]))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
