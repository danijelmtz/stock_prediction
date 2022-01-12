
import streamlit as st
from data import utils, plot
from models import helper_functions
from models import run

header = st.container()

av_api = st.container()

datasets = st.container()

model_training = st.container()


with header:
    st.title('Stock prediction with Machine Learning!')
    """
    Goal is to predict stock movement using Machine Learning models.

    Disclaimer: Predicting the stock price on the basis of historical data is really hard
    because technical analysis results are easily influenced by outside events(news).

    """

with av_api:
    st.header('Alpha Vantange API')
    """
    In this example, we will be working with the following stocks daily data:

    * **FB** - Meta Platforms formerly known as Facebook
    * **AMZN** - Amazon
    * **AAPL** - Apple
    * **NFLX** - Netflix
    * **GOOG** - Alphabet formerly known as Google.

   [Alpha Vantange](https://www.alphavantage.co/documentation/) offers a free API
   for historical and real-time stock market data.
   The API provides 20+ years of daily, weekly, monthly data.
   Besides the Time Series stock data , the API offers Fundamental data, Physical and Digital /Crypto Currencies (e.g. Bitcoin), Economical and Technical Indicators.
   """
with datasets:
    st.header('Visualization of the Stock data')
    # Currently we only the FAANG stocks to chose from
    stocks_to_select_from = ('AAPL', "GOOG", "FB", "AMZN", "NFLX")
    selected_stocks = st.selectbox("Select dataset for visualization", stocks_to_select_from)

    # Load data into a pandas dataframe.
    stock_df = utils.load_stock_data(selected_stocks)

    # display the dataframe in streamlit
    st.dataframe(stock_df)

    # Select what kind of plot you want.
    plot_type = ('Standard Chart', 'Candlestick Chart')
    selected_plot_type = st.selectbox('Select type of the plot', plot_type)

    if selected_plot_type == 'Standard Chart':
        st.plotly_chart(plot.standard_plot(stock_df, selected_stocks),
                        use_container_width=True)
    elif selected_plot_type == 'Candlestick Chart':
        st.plotly_chart(plot.candlestick_plot(stock_df, selected_stocks),
                        use_container_width=True)

with model_training:
    st.header('Model training')
    """Select the stock for training the model."""
    selected_stocks_for_training = st.selectbox(
        "Select Stock for Training", stocks_to_select_from)
    stock_df_for_training = utils.load_stock_data(selected_stocks_for_training)

    """
    Enter the train/test split coefficient.
    """
    test_train_split = st.number_input(
        label="Train/Test Split", min_value=0.1, max_value=0.9, step=0.1, format="%.1f")

    # Split the original dataframe into train and test data.
    train_df, test_df = utils.split_train_and_test_data(
        stock_df_for_training, test_train_split)

    # Plot the train and test data.
    st.plotly_chart(plot.plot_train_test_df(train_df, test_df),
                    use_container_width=True)

    """
    ### Baseline model - Simple Moving Average
    """
    # Select the window for the baseline model
    window_baseline = st.slider(
        'Baseline window.', min_value=10, max_value=100, value=50, step=10)

    # Get the results from the baseline model.
    mape_baseline, rmse_baseline, baseline_df = run.get_baseline(
        stock_df_for_training, train_df, window_baseline=window_baseline)

    # Create two col to display the RMSE and MAPE results.
    rsme_col_baseline, mape_col_baseline = st.columns(2)
    mape_col_baseline.subheader(
        'Mean Absolute Percentage Error  baseline model is: ')
    mape_col_baseline.write(mape_baseline)
    rsme_col_baseline.subheader('Root Mean Square Error baseline model is: ')
    rsme_col_baseline.write(rmse_baseline)

    # Plot the results of the baseline model.
    st.plotly_chart(plot.plot_baseline_model(baseline_df),
                    use_container_width=True)
    """
    ### Select parameters for training the model.
    """

    select_col, display_col = st.columns(2)

    # Get the window size to be used for the model.
    window_size = select_col.slider(
        'How far in the past should we look?', min_value=10, max_value=100, value=50, step=10)

    # Number of hidden unites that will be used.
    hidden_units = select_col.number_input(
        label="How many hidden units should our model have?", min_value=10, max_value=100, step=1, value=32, format="%d")

    # Number of epochs to train for.
    cur_epoch = select_col.number_input(
        label="How many epochs to train for?", min_value=1, max_value=50, step=1, value=1, format="%d")

    # Get model results.
    mape_lstm, rmse_lstm, test_df, model = run.train_and_eval_lstm(
        stock_df_for_training, train_df, test_df, window_size, int(cur_epoch), int(hidden_units))

    display_col.subheader('Mean Absolute Percentage Error of the model is: ')
    display_col.write(mape_lstm)
    display_col.subheader('Root Mean Sqaure Error of the model is: ')
    display_col.write(rmse_lstm)
    """
    ## Plot evaluation results.
    """
    # Plot the results of the model together with actual predictions.
    st.plotly_chart(plot.plot_train_test_df(train_df, test_df, predictions=True),
                    use_container_width=True)
    # Option to save the model or not.
    to_save_or_not_to_save = ("Yes", "No")
    save_model = st.selectbox("Save the model?", to_save_or_not_to_save)

    if save_model:
        helper_functions.save_trained_model(model)
