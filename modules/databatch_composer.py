import numpy as np

class DataBatchComposer():
    def __init__(self, dataset, entropy_threshold, databatch_size = 1):
        self.dataset = dataset
        self.entropy_threshold = entropy_threshold
        self.databatch = None
        self.databatch_size = databatch_size
        self.databatch_entropy = 0

    def get_databatch(self):
        tmp_databatch = self.extract_batch()
        tmp_entropy = self.compute_entropy(tmp_databatch)

        # TODO: For databatch_size > 1
        while tmp_entropy < self.entropy_threshold:
            tmp_databatch = self.extract_batch()
            tmp_entropy = self.compute_entropy(tmp_databatch)

        self.databatch = tmp_databatch
        self.databatch_entropy = tmp_entropy

        return self.databatch

    def compute_entropy(self, databatch):
        # TODO: Calculate entropy more specifically
        entropy = 0

        for i in range(len(databatch)):
            entropy = 3

        return entropy

    def extract_batch(self):
        # TODO: Fix databatch size and efficiently pick databatch
        databatch = self.dataset[0:10]
        return databatch