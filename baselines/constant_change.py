# Baseline that predicts a constant day-to-day change
# Each prediction is equal to the previous day's price plus
# the change betwee the previous two days
# returns list of price predictions and index in original
# data where the predictions start
def constant_change_predictions(data):
    return [2 * data[i+1].price - data[i].price
            for i in range(len(data) - 2)], 2, -1
