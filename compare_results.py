from utils.data_manager import DataManager
from utils.result_reader import ResultReader, NNResultReader
from utils import testing
from baselines.no_change import no_change_predictions
from baselines.constant_change import constant_change_predictions
import numpy as np
import matplotlib.pyplot as plt
import pdb

def main(model_names):
    # Load the data
    data = DataManager().data
    print("data:", len(data))
    actual = [datum.price for datum in data]
    dates = [datum.date for datum in data]

    # First index in data with results from all models
    max_start_index = 0
    min_end_index = -1
    # Collect models as dicts with keys "name", "predictions", "start_index"
    models = []

    ### Collect predictions from models ###

    if "no_change" in model_names:
        predictions, start_index, end_index = no_change_predictions(data)
        max_start_index = max(max_start_index, start_index)
        min_end_index = min(min_end_index, end_index)
        models.append({"name": "no_change", "predictions": predictions, "start_index": start_index, "end_index": end_index})

    if "constant_change" in model_names:
        predictions, start_index, end_index = constant_change_predictions(data)
        max_start_index = max(max_start_index, start_index)
        min_end_index = min(min_end_index, end_index)
        models.append({"name": "constant_change", "predictions": predictions, "start_index": start_index, "end_index": end_index})

    if "GAN" in model_names:
        reader = ResultReader("predictedGAN.csv")
        predictions, start_index, end_index = reader.get_predictions()
        reader.check_start_date(dates[start_index], dates[end_index])
        max_start_index = max(max_start_index, start_index)
        min_end_index = min(min_end_index, end_index)
        models.append({"name": "GAN", "predictions": predictions, "start_index": start_index, "end_index": end_index})

    if "NN" in model_names:
        reader = NNResultReader("predictedNN.csv", len(data))
        NNpredictions = reader.get_predictions()
        #max_start_index = max(max_start_index, start_index)
        #min_end_index = min(min_end_index, end_index)
        #models.append({"name": "NN", "predictions": predictions, "start_index": start_index, "end_index": end_index})

    
    ### Generate stats and graphs ###


    chart_data = []
    for model in models:
        if model["end_index"] < -1:
            expected = actual[model["start_index"]: model["end_index"] + 1]
        else:
            expected = actual[model["start_index"]:]
        chart_data.append((model["name"], expected, model["predictions"]))
        print((model["name"], len(expected), len(model["predictions"])))
    NNexpected = np.flip(np.array([datum.price for datum in data[:552]]))
    chart_data.append(("RNN", NNexpected, NNpredictions))
    testing.stat_chart("MAPE", testing.MAPE, chart_data)
    testing.plot_results(NNexpected, [("RNN", NNpredictions)])
    plt.show()
    return
    
    actual = actual[max_start_index:min_end_index]
    for model in models:
        model["predictions"] = model["predictions"][max_start_index - model["start_index"]:]
        if model["end_index"] > min_end_index:
            model["predictions"] = model["predictions"][0:-1 + min_end_index - model["end_index"]]

    clip = [-100, -1]
    if not clip is None:
        for model in models:
            model["predictions"] = model["predictions"][clip[0]:clip[1]]
        actual = actual[clip[0]:clip[1]]

    model_graph_data = [(m["name"], m["predictions"]) for m in models]
    testing.plot_results(actual, model_graph_data)
    plt.show()

if __name__ == "__main__":
    model_names = ["no_change", "constant_change", "GAN", "NN"]
    main(model_names)
