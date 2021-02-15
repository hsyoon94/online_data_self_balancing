import numpy as np
import math

class DataBatchComposer():
    def __init__(self, dataset_dir, data_list, entropy_threshold, databatch_size = 1):
        self.dataset_dir = dataset_dir
        self.data_list = data_list
        self.entropy_threshold = entropy_threshold
        self.databatch_list = None
        self.databatch_size = databatch_size
        self.databatch_entropy = 0
        self.mt_net = None
        self.ms_net = None
        self.mb_net = None
        self.po = None

    def get_databatch_list(self):

        tmp_databatch = self.extract_random_batch()
        tmp_entropy = self.compute_entropy(tmp_databatch)

        while tmp_entropy < self.entropy_threshold:
            print("current databatch entropy", tmp_entropy)
            tmp_databatch = self.extract_random_batch()
            tmp_entropy = self.compute_entropy(tmp_databatch)

        self.databatch = tmp_databatch
        self.databatch_entropy = tmp_entropy

        return self.databatch

    def compute_entropy(self, databatch):
        entropy = 0
        for i in range(len(databatch)):
            entropy = entropy + self.prob(databatch[i]) * math.log(self.prob(databatch[i]))

        entropy = -1 * entropy

        entropy = 100
        return entropy

    def extract_random_batch(self):
        # Random sampling
        databatch = np.random.choice(len(self.data_list) - 1, int(len(self.data_list) / 1000), replace=True)
        return databatch

    def prob(self, data):
        # prob = self.mt_net(data) * self.ms_net(data) * self.mb_net(data) * self.po(data)
        prob = 0.5
        return prob

    def compute_prob(self):
        # TODO: SORT WITH THE INDEX
        return