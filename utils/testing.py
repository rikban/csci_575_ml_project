import numpy as np

def as_1d_np_array(array):
    if type(array) in [list, tuple]:
        array = np.array(array)
    if not (type(array) is np.ndarray):
        raise Exception("input must be a list, tuple or numpy array")
    if len(array.shape) != 1:
        raise Exception("input must be 1 dimensional")
    return array

# Mean Absolute Error
import matplotlib.pyplot as plt
import pdb

def MAE(actual, predicted):
    actual = as_1d_np_array(actual)
    predicted = as_1d_np_array(predicted)
    return np.abs(actual - predicted).sum() / len(actual)

# Mean Absolute Percent Error
def MAPE(actual, predicted):
    actual = as_1d_np_array(actual)
    predicted = as_1d_np_array(predicted)
    return 100 * np.abs((actual - predicted)/actual).sum() / len(actual)

# takes tuples of (method_name, predicted_values)
def plot_results(actual, method_predictions):
    plt.figure("Actual vs Predicted")
    plt.plot(actual, label="Actual Prices")
    for name, values in method_predictions:
        assert(len(values) == len(actual))
        plt.plot(values, label=name)
    plt.legend()

# compares the stats of different models
# stat: function taking actual and predicted values
# data: tuples of (name, expected, predicted)
def stat_chart(title, stat, data):
    assert(callable(stat))
    plt.figure(title)
    names, values = [], []
    for name, expected, predicted in data:
        names.append(name)
        values.append(stat(expected, predicted))
    plt.bar(names, values)
