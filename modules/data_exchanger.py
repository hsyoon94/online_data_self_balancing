# DE

import numpy as np
import torch

class DataExchanger():
    def __init__(self, dataset):
        self.dataset = dataset

    def exchange(self, online_novel_data):
        discard_data = self.select_discard_data()

        self.dataset.discard(discard_data)
        self.dataset.append(online_novel_data)

    def select_discard_data(self):

        # TODO: Define criteria of discarding data with probability function
        discard_data = self.dataset[0]
        return discard_data
