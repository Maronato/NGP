import csv
import numpy as np


def load():
    return UnicampDataset()


class UnicampDataset:

    def __init__(self):
        self.load_dataset()
        self.load_H()
        self.load_W()

    def load_dataset(self):
        with open("Data/cleaned_Unicamp.csv", "r") as f:
            data = self.read(f)

        self.dataset = data

    def load_H(self):
        with open("Data/UnicampH.csv", "r") as f:
            data = self.read(f)

        self.H = data

    def load_W(self):
        with open("Data/UnicampW.csv", "r") as f:
            data = self.read(f)

        self.W = data

    def read(self, file):
        data = []
        reader = csv.reader(file)
        for row in reader:
            data.append([float(x) for x in row])
        return np.array(data)
