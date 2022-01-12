"""Script containing functions for plotting."""
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from pandas import DataFrame


def standard_plot(data: DataFrame, ticker: str) -> go.Figure:
    """
    Standard plot of financial data.

    Params:
        stock_df (pd.DataFrame): Pandas DataFrame with historical stock prices.
        ticker (str): Name of the Stock.

    Return:
        fig (go.Figure): Ploty figure with the results.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['date'], y=data['4. close'], name=' Close Price in $'))
    fig.layout.update(title_text="Time Series Data",
                      xaxis_rangeslider_visible=True)
    fig.update_layout(autosize=True)
    return fig


def candlestick_plot(stock_df: DataFrame, ticker: str) -> go.Figure:
    """
    Create Candlestick plot.

    Params:
        stock_df (pd.DataFrame): Pandas DataFrame with historical stock prices.
        ticker (str): Name of the Stock.

    Return:
        fig (go.Figure): Ploty figure with the results.
    """
    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Candlestick(x=stock_df['date'],
                                 open=stock_df['1. open'], high=stock_df['2. high'],
                                 low=stock_df['3. low'], close=stock_df['4. close']),
                  secondary_y=True)
    fig.layout.yaxis2.showgrid = False
    return fig


def plot_train_test_df(train_df: DataFrame, test_df: DataFrame, predictions: bool = False) -> go.Figure():
    """
    Plot train and validation dataframes.

    Params:
        train_df (pd.DataFrame): Dataframe with train data.
        test_df (pd.DataFrame): Dataframe with test data.

    Return:
        fig (go.Figure): Ploty figure with the results.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_df['date'], y=train_df['4. close'], name='Train data Close Price in $'))
    fig.add_trace(go.Scatter(
        x=test_df['date'], y=test_df['4. close'], name='Test data Close Price in $'))
    if predictions:
        fig.add_trace(go.Scatter(
            x=test_df['date'], y=test_df['predicted'], name='Predicted Close Price in $'))
    fig.layout.update(title_text="Time Series Train and Test Data",
                      xaxis_rangeslider_visible=True)
    return fig


def plot_model_results_on_full_data(test_df: DataFrame) -> go.Figure():
    """
    Plot model results on full data.

    Params:
        test_df (pd.DataFrame): DataFrame with prediction and actual stock data.

    Return:
        fig (go.Figure): Ploty figure with the results.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_df['date'], y=test_df['4. close'], name='Actual Close Price in $'))
    fig.add_trace(go.Scatter(
        x=test_df['date'], y=test_df['predicted'], name='Predicted Close Price in $'))
    fig.layout.update(title_text="Time Series Train and Test Data",
                      xaxis_rangeslider_visible=True)
    return fig

def plot_baseline_model(full_df: DataFrame) -> go.Figure():
    """
    Plot results of the baseline model.

    Params:
        full_df (pd.DataFrame): DataFrame with baseline results and actual stock data.

    Return:
        fig (go.Figure): Ploty figure with the results.

    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=full_df['date'], y=full_df['4. close'], name='Actual Close Price in $'))
    fig.add_trace(go.Scatter(
        x=full_df['date'], y=full_df['baseline'], name='Baseline Close Price in $'))
    fig.layout.update(title_text="Baseline vs Actual predictions.",
                      xaxis_rangeslider_visible=True)
    return fig
