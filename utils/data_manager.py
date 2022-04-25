import os
import numpy as np

DATA_COLUMNS = ["Date", "Price", "Open", "High", "Low", "Volume", "Change"]

class Datum():
    def __init__(self, values):
        self.date = values[0]
        self.price, self.open, self.high, self.low, self.volume, self.change = values[1:].astype(np.float32)

    def as_row(self):
        return [self.date, self.price, self.open, self.high, self.low, self.volume, self.change]
    def __repr__(self):
        values = self.as_row()
        return ", ".join([DATA_COLUMNS[i] + ": " + str(values[i]) for i in range(len(values))])

class DataManager():
    def __init__(self, filename="data.csv", rows=2263):
        # filename should be relative to repo base directory
        path = "/".join(__file__.split("/")[:-2] + [filename])
        assert(os.path.exists(path))
        with open(path, "r") as f:
            raw = np.genfromtxt(f, dtype='str', delimiter=',', invalid_raise=False, usecols=np.arange(0, 7))
            self.data = [Datum(raw[i,:]) for i in range(1, rows+1)]
