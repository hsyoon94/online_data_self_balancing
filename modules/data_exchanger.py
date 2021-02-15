import numpy as np
from os import listdir
from os.path import isfile, join
import shutil
import json


class DataExchanger():
    def __init__(self, online_data_dir, online_data_image_dir, dataset_dir, dataset_image_dir):
        self.data_removal_dir = '/media/hsyoon/hard2/SDS/data_removal/'
        self.online_data_dir = online_data_dir
        self.online_data_image_dir = online_data_image_dir
        self.dataset_dir = dataset_dir
        self.dataset_image_dir = dataset_image_dir
        self.dataset_name_list = [f1 for f1 in listdir(self.dataset_dir) if isfile(join(self.dataset_dir, f1))]
        self.pmt = None
        self.pms = None
        self.pmb = None
        self.pm = None


    def update_data_list(self):
        self.dataset_name_list = [f1 for f1 in listdir(self.dataset_dir) if isfile(join(self.dataset_dir, f1))]
        self.dataset_name_list.sort()

    # Exchange data one by one.
    def exchange(self, online_novel_data_name):
        discard_data_name = self.select_discard_data()

        # discard dataset data to trash
        shutil.move(self.dataset_dir + discard_data_name, self.data_removal_dir + discard_data_name)
        shutil.move(self.dataset_image_dir + discard_data_name.split('.')[0] + '.png', self.data_removal_dir + discard_data_name.split('.')[0] + '.png')
        # append dataset data to trash
        shutil.move(self.online_data_dir + online_novel_data_name, self.dataset_dir + online_novel_data_name)
        shutil.move(self.online_data_image_dir + online_novel_data_name.split('.')[0] + '.png', self.dataset_image_dir + online_novel_data_name.split('.')[0] + '.png')
        self.update_data_list()

    def select_discard_data(self):

        # TODO: Define criteria of discarding data with probability function
        # TODO: Sort the data with the probability in "daytime" to efficiently use time. (Night time is for updating MNet)
        index = np.random.choice(len(self.dataset_name_list)-1, replace=False)

        discard_data_name = self.dataset_name_list[index]
        return discard_data_name

    def exchange_whole(self, online_data_name_list):
        self.online_data_name_list_length = len(online_data_name_list)
        self.list_to_remove = list()

        for file_index in range(len(self.dataset_name_list)):
            with open(self.dataset_dir + self.dataset_name_list[file_index]) as tmp_json:
                json_data = json.load(tmp_json)

                final_prob = max(self.pmt(json_data['state'])) * max(self.pms(json_data['state'])) * max(self.pmb(json_data['state']))

                self.list_to_remove.append(final_prob)

        # TODO
        self.index_list_to_remove = self.list_to_remove.largest(self.online_data_name_list_length)

        for i in range(self.online_data_name_list_length):
            # Discard from dataset
            shutil.move(self.dataset_dir + self.dataset_name_list[self.index_list_to_remove], self.data_removal_dir + self.dataset_name_list[self.index_list_to_remove])
            shutil.move(self.dataset_image_dir + self.dataset_name_list[self.index_list_to_remove].split('.')[0] + '.png', self.data_removal_dir + self.dataset_name_list[self.index_list_to_remove].split('.')[0] + '.png')

            # Append novel data to dataset
            shutil.move(self.online_data_dir + online_data_name_list[i], self.dataset_dir + online_data_name_list[i])
            shutil.move(self.online_data_image_dir + online_data_name_list[i].split('.')[0] + '.png', self.dataset_image_dir + online_data_name_list[i].split('.')[0] + '.png')

