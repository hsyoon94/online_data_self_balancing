import numpy as np
from os import listdir
from os.path import isfile, join
import shutil
import json


class DataExchanger():
    def __init__(self, online_data_dir, dataset_dir):
        self.data_removal_dir = '/media/hsyoon/hard2/SDS/data_removal/'
        self.online_data_dir = online_data_dir
        self.dataset_dir = dataset_dir
        self.dataset_name_list = [f1 for f1 in listdir(self.dataset_dir) if isfile(join(self.dataset_dir, f1))]
        self.online_data_name_list = [f2 for f2 in listdir(self.online_data_dir) if isfile(join(self.online_data_dir, f2))]

    def update_data_list(self):
        self.dataset_name_list = [f1 for f1 in listdir(self.dataset_dir) if isfile(join(self.dataset_dir, f1))]
        self.online_data_name_list = [f2 for f2 in listdir(self.online_data_dir) if isfile(join(self.online_data_dir, f2))]

    # Exchange data one by one.
    def exchange(self, online_novel_data_name):
        discard_data_name = self.select_discard_data()

        # discard dataset data to trash
        shutil.move(self.dataset_dir + discard_data_name, self.data_removal_dir + discard_data_name)
        # append dataset data to trash
        shutil.move(self.online_data_dir + online_novel_data_name, self.dataset_dir + online_novel_data_name)
        self.update_data_list()

    def select_discard_data(self):

        # TODO: Define criteria of discarding data with probability function
        # TODO: Sort the data with the probability in "daytime" to efficiently use time. (Night time is for updating MNet)
        index = np.random.choice(len(self.dataset_name_list)-1, replace=False)

        discard_data_name = self.dataset_name_list[index]
        return discard_data_name