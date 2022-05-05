import os
import numpy as np
import pdb

class ResultReader():
    def __init__(self, filename):
        # filename should be relative to model_results dierectory
        path = "/".join(__file__.split("/")[:-2] + ["model_results", filename])
        assert(os.path.exists(path))
        with open(path, "r") as f:
            raw = np.genfromtxt(f, dtype='str', delimiter=',', invalid_raise=False, usecols=np.arange(0, 2))
            self.data = raw[1:,:]

    def get_predictions(self):
        return self.data[:, 1].astype(np.float32), self.start_index(), self.end_index()

    def get_start_date(self):
        return self.data[0, 0]

    def get_end_date(self):
        return self.data[-1, 0]

    def start_index(self):
        return 1068

    def end_index(self):
        return -24

    def check_start_date(self, expected_start, expected_end):
        print("\nSanity check: start date =", self.get_start_date())
        print("  expected start date = ", expected_start)
        print("   actual end date =", self.get_end_date())
        print("  expected end date = ", expected_end)

class NNResultReader():
    def __init__(self, filename, last_day):
        # filename should be relative to model_results dierectory
        path = "/".join(__file__.split("/")[:-2] + ["model_results", filename])
        assert(os.path.exists(path))
        with open(path, "r") as f:
            raw = np.genfromtxt(f, dtype='str', delimiter=',', invalid_raise=False)
            self.data = raw.transpose()
            print(self.data[0,:])
            print(self.data[-1,:])

    def get_predictions(self):
        return self.data[:, 1].astype(np.float32)
