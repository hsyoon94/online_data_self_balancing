import numpy as np
import math
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

class DataBatchComposer():
    def __init__(self, dataset_dir, entropy_threshold, databatch_size, mnet_state_size, mnet_state_dim, date, time):
        self.dataset_dir = dataset_dir
        self.data_list = None
        self.entropy_threshold = entropy_threshold
        self.databatch_list = None
        self.databatch_size = databatch_size
        self.databatch_entropy = 0
        self.pmt_net = PM_NET(mnet_state_size, mnet_state_dim, THROTTLE_DISCR_DIM, device)
        self.pms_net = PM_NET(mnet_state_size, mnet_state_dim, STEER_DISCR_DIM, device)
        self.pmb_net = PM_NET(mnet_state_size, mnet_state_dim, BRAKE_DISCR_DIM, device)
        self.date = date
        self.time = time

        self.data_name_list_length = 0
        self.probability_with_index = list()

    def extract_databatch_wigh_high_entropy(self):

        data_batch_index = self.extract_random_batch()
        batch_entropy = self.compute_entropy(data_batch_index, self.probability_with_index)

        while batch_entropy < self.entropy_threshold:
            data_batch_index = self.extract_random_batch()
            batch_entropy = self.compute_entropy(data_batch_index, self.probability_with_index)

        return data_batch_index

    def compute_entropy(self, data_batch_index, databatch_prob):
        entropy = 0
        for i in range(len(data_batch_index)):
            entropy = entropy + databatch_prob[data_batch_index[i]] * math.log(databatch_prob[data_batch_index[i]])

        fin_entropy = -1 * entropy
        return fin_entropy

    def extract_random_batch(self):
        # Random sampling
        start_index =  len(self.data_list) - 1
        fin_index = int(len(self.data_list) / 100)
        rnd_databatch = np.random.choice(start_index, fin_index ,replace=True)
        return rnd_databatch

    def update_probability(self):
        self.data_list = [f for f in listdir(self.dataset_dir) if isfile(join(self.dataset_dir, f))]
        self.data_name_list_length = len(self.data_list)
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

        # Compute probability
        for file_index in range(len(self.data_list)):
            with open(self.dataset_dir + self.data_list[file_index]) as tmp_json:
                json_data = json.load(tmp_json)
                with torch.no_grad():
                    state_tensor = torch.tensor(json_data['state']).to(device)
                    state_tensor = torch.reshape(state_tensor, (
                        state_tensor.shape[0], state_tensor.shape[3], state_tensor.shape[1],
                        state_tensor.shape[2])).float()

                    output_pmt = self.pmt_net(state_tensor).cpu().squeeze().detach().numpy().tolist()
                    output_pms = self.pms_net(state_tensor).cpu().squeeze().detach().numpy().tolist()
                    output_pmb = self.pmb_net(state_tensor).cpu().squeeze().detach().numpy().tolist()
                    final_prob = max(output_pmt) * max(output_pms) * max(output_pmb)

                self.probability_with_index.append(final_prob)

        self.probability_with_index = softmax(self.probability_with_index)

        dataset_entropy = self.compute_entropy(list(range(0, len(self.probability_with_index))), self.probability_with_index)
        print("DATASET ENTROPY:", dataset_entropy)

        dataset_entropy_txt = open('/home/hsyoon/job/SDS/log/' + self.date + '/' + self.time + '/dataset_entropy.txt', 'a')
        dataset_entropy_txt.write(str(dataset_entropy) + '\n')
        dataset_entropy_txt.close()