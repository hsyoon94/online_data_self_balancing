# DE

import numpy as np
import torch
# import probability
import torch.nn as nn
import math
from numpy import linalg as LA

torch.set_num_threads(2)

is_cuda = torch.cuda.is_available()

# device = torch.device('cuda' if is_cuda else 'cpu')
device = 'cpu'

from .bmnet import MNet
from .probability import motion_probability as PM_NET
from os import listdir
from os.path import isfile, join

MNET_MODEL_DIR = "/home/hsyoon/job/SDS/trained_models/mnet/"

PMT_MODEL_DIR = "/home/hsyoon/job/SDS/trained_models/pmt/"
PMS_MODEL_DIR = "/home/hsyoon/job/SDS/trained_models/pms/"
PMB_MODEL_DIR = "/home/hsyoon/job/SDS/trained_models/pmb/"

def mse_loss(arr1, arr2):
    error = 0
    for i in range(arr1.shape[0]):
        error = error + math.pow(arr1[i] - arr2[i], 2)

    error = error / arr1.shape[0]

    return error

class DataFilter():
    def __init__(self, mnet_state_size, mnet_state_dim, mnet_motion_size):
        self.pm = None
        self.po = None

        self.pm_threshold = 0.5
        self.po_threshold = 0.5
        self.pm_mse_threshold = 0.33

        self.pmo_threshold = 0.1
        self.MSELOSS = nn.MSELoss()
        # MNET for novelty with dropout
        self.mnet_filter = MNet(mnet_state_size, mnet_state_dim, mnet_motion_size, device)

        # PM_NET for
        self.pms_net = PM_NET(mnet_state_size, mnet_state_dim, 10, device)
        self.pmt_net = PM_NET(mnet_state_size, mnet_state_dim, 10, device)
        self.pmb_net = PM_NET(mnet_state_size, mnet_state_dim, 10, device)

        # FOR THE LATEST MNET
        mnet_list = [f for f in listdir(MNET_MODEL_DIR) if isfile(join(MNET_MODEL_DIR, f))]
        mnet_list.sort()

        # pms_list = [f1 for f1 in listdir(PMS_MODEL_DIR) if isfile(join(PMS_MODEL_DIR, f1))]
        # pms_list.sort()

        self.mnet_filter.load_state_dict(torch.load(MNET_MODEL_DIR + mnet_list[-1]))
        # self.pms_net.load_state_dict(torch.load(PMS_MODEL_DIR + pms_list[-1]))
        self.accumulated_mse_error = 0
        self.motion_sequence = list()
        self.ensemble_frequency = 3

    def is_novel(self, online_state, gt_motion):

        with torch.no_grad():
            online_state_tensor = torch.tensor(online_state).to(device)
            online_state_tensor = torch.reshape(online_state_tensor, (online_state_tensor.shape[0], online_state_tensor.shape[3], online_state_tensor.shape[1], online_state_tensor.shape[2])).float()
            try:
                for iter in range(self.ensemble_frequency):
                    with torch.no_grad():
                        pm = self.mnet_filter(online_state_tensor)
                        pm = pm.cpu().detach().numpy().squeeze()

                    self.motion_sequence.append(pm)


                motion_sequence_numpy = np.array(self.motion_sequence)
                motion_sequence_mean = np.mean(motion_sequence_numpy, axis=0)
                motion_sequence_std = np.std(motion_sequence_numpy, axis=0)

                self.pm = motion_sequence_mean

                with torch.no_grad():
                    pms = self.pms_net(online_state_tensor)
                    pmt = self.pmt_net(online_state_tensor)
                    pmb = self.pmb_net(online_state_tensor)

                    # 0.7 under or not the maximum, then it is indicated to be novel!

                mse_error = mse_loss(gt_motion, self.pm)

                self.accumulated_mse_error = self.accumulated_mse_error + mse_error
                self.motion_sequence = list()

                total_std = LA.norm(motion_sequence_std)

                print("total_std", total_std)

                if total_std > self.pmo_threshold:
                    return False
                else:
                    return False

            except RuntimeError as e:
                print("what?", e)
                return False

    def get_mse_loss(self):
        return self.accumulated_mse_error