import numpy as np
from os import listdir
from os.path import isfile, join
import shutil
import json

# DE
import numpy as np
import torch
# import probability
import torch.nn as nn
import math
from numpy import linalg as LA
from arguments import get_args

torch.set_num_threads(2)

is_cuda = torch.cuda.is_available()

# device = torch.device('cuda' if is_cuda else 'cpu')
device = 'cpu'

import modules.swav.src.resnet50 as resnet_models
from .bmnet import MNet
from .probability import motion_probability as PM_NET
from os import listdir
from os.path import isfile, join

MNET_MODEL_DIR = "/home/hsyoon/job/SDS/trained_models/mnet/"

PMT_MODEL_DIR = "/home/hsyoon/job/SDS/trained_models/pmt/"
PMS_MODEL_DIR = "/home/hsyoon/job/SDS/trained_models/pms/"
PMB_MODEL_DIR = "/home/hsyoon/job/SDS/trained_models/pmb/"
PO_MODEL_DIR = "/home/hsyoon/job/SDS/trained_models/po/"

THROTTLE_DISCR_DIM = 2
throttle_discr_th = 0.1

STEER_DISCR_DIM = 3
steer_discr_th1 = -0.2
steer_discr_th2 = 0.2

BRAKE_DISCR_DIM = 2
brake_discr_th = 0.1

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def sort_with_index(list_with_prob, num):
    sort = sorted(range(len(list_with_prob)), key=lambda k: list_with_prob[k])
    return sort[-1*num:]

class DataExchanger():
    def __init__(self, online_data_dir, online_data_image_dir, dataset_dir, dataset_image_dir, mnet_state_size, mnet_state_dim):
        self.data_removal_dir = '/media/hsyoon/hard2/SDS/data_removal/'
        self.online_data_dir = online_data_dir
        self.online_data_image_dir = online_data_image_dir
        self.dataset_dir = dataset_dir
        self.dataset_image_dir = dataset_image_dir
        self.dataset_name_list = [f1 for f1 in listdir(self.dataset_dir) if isfile(join(self.dataset_dir, f1))]

        self.probability_with_index = list()

        self.pmt_net = PM_NET(mnet_state_size, mnet_state_dim, THROTTLE_DISCR_DIM, device)
        self.pms_net = PM_NET(mnet_state_size, mnet_state_dim, STEER_DISCR_DIM, device)
        self.pmb_net = PM_NET(mnet_state_size, mnet_state_dim, BRAKE_DISCR_DIM, device)
        self.pm = None

    def exchange_whole(self, online_data_name_list):
        self.online_data_name_list_length = len(online_data_name_list)
        self.probability_with_index = list()  # Initialize

        # FOR THE LATEST MODELS
        mnet_list = [f for f in listdir(MNET_MODEL_DIR) if isfile(join(MNET_MODEL_DIR, f))]
        pmt_list = [ft for ft in listdir(PMT_MODEL_DIR) if isfile(join(PMT_MODEL_DIR, ft))]
        pms_list = [fs for fs in listdir(PMS_MODEL_DIR) if isfile(join(PMS_MODEL_DIR, fs))]
        pmb_list = [fb for fb in listdir(PMB_MODEL_DIR) if isfile(join(PMB_MODEL_DIR, fb))]
        po_list = [fo for fo in listdir(PO_MODEL_DIR) if isfile(join(PO_MODEL_DIR, fo))]

        mnet_list.sort()
        pmt_list.sort()
        pms_list.sort()
        pmb_list.sort()
        po_list.sort()

        self.pmt_gt_index = 0
        self.pms_gt_index = 0
        self.pmb_gt_index = 0

        self.pmt_net.load_state_dict(torch.load(PMT_MODEL_DIR + pmt_list[-1]))
        self.pms_net.load_state_dict(torch.load(PMS_MODEL_DIR + pms_list[-1]))
        self.pmb_net.load_state_dict(torch.load(PMB_MODEL_DIR + pmb_list[-1]))

        for file_index in range(len(self.dataset_name_list)):
            with open(self.dataset_dir + self.dataset_name_list[file_index]) as tmp_json:
                json_data = json.load(tmp_json)
                with torch.no_grad():
                    state_tensor = torch.tensor(json_data['state']).to(device)
                    state_tensor = torch.reshape(state_tensor, (state_tensor.shape[0], state_tensor.shape[3], state_tensor.shape[1], state_tensor.shape[2])).float()
                    # TODO I: Implement selecting max value in list

                    output_pmt = self.pmt_net(state_tensor).cpu().squeeze().detach().numpy().tolist()
                    output_pms = self.pms_net(state_tensor).cpu().squeeze().detach().numpy().tolist()
                    output_pmb = self.pmb_net(state_tensor).cpu().squeeze().detach().numpy().tolist()
                    final_prob = max(output_pmt) * max(output_pms) * max(output_pmb)

                self.probability_with_index.append(final_prob)

        self.index_list_to_remove = sort_with_index(self.probability_with_index, self.online_data_name_list_length)

        for i in range(self.online_data_name_list_length):
            # Discard from dataset
            shutil.move(self.dataset_dir + self.dataset_name_list[self.index_list_to_remove[i]], self.data_removal_dir + self.dataset_name_list[self.index_list_to_remove[i]])
            shutil.move(self.dataset_image_dir + self.dataset_name_list[self.index_list_to_remove[i]].split('.')[0] + '.png', self.data_removal_dir + self.dataset_name_list[self.index_list_to_remove[i]].split('.')[0] + '.png')

            # Append novel data to dataset
            shutil.move(self.online_data_dir + online_data_name_list[i], self.dataset_dir + online_data_name_list[i])
            shutil.move(self.online_data_image_dir + online_data_name_list[i].split('.')[0] + '.png', self.dataset_image_dir + online_data_name_list[i].split('.')[0] + '.png')

        # Update dataset name list
        self.dataset_name_list = [fd for fd in listdir(self.dataset_dir) if isfile(join(self.dataset_dir, fd))]



