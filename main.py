# forecast monthly births with xgboost
# Taken from
# https://machinelearningmastery.com/xgboost-for-time-series-forecasting/
import numpy
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
# from matplotlib import pyplot
import matplotlib.pyplot as plt

import xgboost

print("xgboost", xgboost.__version__)


# load dataset
def read_data():
    """
    read_data reads data from a csv file
    :return:
    series  is a dataframe object
    values  is series.values
    """
    series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
    values = series.values
    # plot dataset
    # plt.plot(values)
    # plt.show()
    return series, values


# transform a time series dataset into a supervised learning dataset
# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """

    :param data: time series data list
    :param n_in: the number of time steps before the current time + n_out time steps
    :param n_out:the number of time steps after the current time step
    :param dropnan: drop any columns of a dataframe containing NaNs
    :return: return a matrix transformed supervised learning task. The first n_in columns are "X_train" and n_out columns y_train i.e. target.
    array([[nan, nan, nan, nan, nan, nan, 35.],
       [nan, nan, nan, nan, nan, 35., 32.],
       [nan, nan, nan, nan, 35., 32., 30.],
       [nan, nan, nan, 35., 32., 30., 31.],
       [nan, nan, 35., 32., 30., 31., 44.],
       [nan, 35., 32., 30., 31., 44., 29.],
       [35., 32., 30., 31., 44., 29., 45.],
       [32., 30., 31., 44., 29., 45., 43.]])
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def train_test_split(data, n_test: int) -> numpy.array:
    """
    split the data from beginning leaving the last n_test time instants to test and the rest for training.
    :param data:
    :param n_test:
    :return:
    """
    return data[:-n_test, :], data[-n_test:, :]


# fit an xgboost model and make a one-step prediction
def xgboost_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(asarray([testX]))
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = xgboost_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    series, values = read_data()
    # series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
    # values = series.values

    # transform the time series data into supervised learning
    data = series_to_supervised(values, n_in=6)

    # evaluate
    pass
    mae, y, yhat = walk_forward_validation(data, 12)
    print('MAE: %.3f' % mae)
    # plot expected vs predicted
    plt.plot(y, label='Expected')
    plt.plot(yhat, label='Predicted')
    plt.legend()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
