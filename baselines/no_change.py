# Baseline that predicts no day-to-day change.
# Each prediction is equal to the previous day's price
# returns list of price predictions and index in original
# data where the predictions start
def no_change_predictions(data):
    return [datum.price for datum in data[:-1]], 1, -1
