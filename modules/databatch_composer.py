import numpy as np

class DataBatchComposer():
    def __init__(self, dataset_dir, data_list, entropy_threshold, databatch_size = 1):
        self.dataset_dir = dataset_dir
        self.data_list = data_list
        self.entropy_threshold = entropy_threshold
        self.databatch_list = None
        self.databatch_size = databatch_size
        self.databatch_entropy = 0

    def get_databatch_list(self):

        # tmp_databatch = self.extract_batch()
        # tmp_entropy = self.compute_entropy(tmp_databatch)
        #
        # while tmp_entropy < self.entropy_threshold:
        #     tmp_databatch = self.extract_batch()
        #     tmp_entropy = self.compute_entropy(tmp_databatch)
        #
        # self.databatch = tmp_databatch
        # self.databatch_entropy = tmp_entropy

        self.databatch_list = np.random.choice(len(self.data_list) - 1, int(len(self.data_list) / 100), replace=True)

        return self.databatch_list

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