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


THROTTLE_DISCR_DIM = 2
throttle_discr_th = 0.1

STEER_DISCR_DIM = 3
steer_discr_th1 = -0.2
steer_discr_th2 = 0.2

BRAKE_DISCR_DIM = 2
brake_discr_th = 0.1

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

        self.BCELOSS = nn.BCELoss()

        self.pm_probability_novelty = False
        self.pm_uncertainty_novelty = False

        self.pm_threshold = 0.5
        self.po_threshold = 0.5
        self.pmt_threshold = 0.7
        self.pms_threshold = 0.7
        self.pmb_threshold = 0.7

        self.pm_mse_threshold = 0.33

        self.pmo_threshold = 0.035
        self.MSELOSS = nn.MSELoss()
        # MNET for novelty with dropout
        self.mnet_filter = MNet(mnet_state_size, mnet_state_dim, mnet_motion_size, device)

        # PM_NET for
        self.pmt_net = PM_NET(mnet_state_size, mnet_state_dim, THROTTLE_DISCR_DIM, device)
        self.pms_net = PM_NET(mnet_state_size, mnet_state_dim, STEER_DISCR_DIM, device)
        self.pmb_net = PM_NET(mnet_state_size, mnet_state_dim, BRAKE_DISCR_DIM, device)

        # FOR THE LATEST MODELS
        mnet_list = [f for f in listdir(MNET_MODEL_DIR) if isfile(join(MNET_MODEL_DIR, f))]
        pmt_list = [ft for ft in listdir(PMT_MODEL_DIR) if isfile(join(PMT_MODEL_DIR, ft))]
        pms_list = [fs for fs in listdir(PMS_MODEL_DIR) if isfile(join(PMS_MODEL_DIR, fs))]
        pmb_list = [fb for fb in listdir(PMB_MODEL_DIR) if isfile(join(PMB_MODEL_DIR, fb))]

        mnet_list.sort()
        pmt_list.sort()
        pms_list.sort()
        pmb_list.sort()

        self.pmt_gt_index = 0
        self.pms_gt_index = 0
        self.pmb_gt_index = 0

        self.mnet_filter.load_state_dict(torch.load(MNET_MODEL_DIR + mnet_list[-1]))

        self.pmt_net.load_state_dict(torch.load(PMT_MODEL_DIR + pmt_list[-1]))
        self.pms_net.load_state_dict(torch.load(PMS_MODEL_DIR + pms_list[-1]))
        self.pmb_net.load_state_dict(torch.load(PMB_MODEL_DIR + pmb_list[-1]))

        self.accumulated_mse_error = 0
        self.motion_sequence = list()
        self.ensemble_frequency = 3
        self.motion_std = None

    def is_novel(self, online_state, gt_motion):

        self.pm_probability_novelty = False
        self.pm_uncertainty_novelty = False

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
                    pms = self.pms_net(online_state_tensor).cpu().detach().squeeze()
                    pmt = self.pmt_net(online_state_tensor).cpu().detach().squeeze()
                    pmb = self.pmb_net(online_state_tensor).cpu().detach().squeeze()

                    if gt_motion[0] < throttle_discr_th:
                        self.pmt_gt_index = 0
                    else:
                        self.pmt_gt_index = 1

                    if gt_motion[1] <= steer_discr_th1:
                        self.pms_gt_index = 0
                    elif steer_discr_th1 <= gt_motion[1] <= steer_discr_th2:
                        self.pms_gt_index = 1
                    else:
                        self.pms_gt_index = 2

                    if gt_motion[2] <= brake_discr_th:
                        self.pmb_gt_index = 0
                    else:
                        self.pmb_gt_index = 1

                    if pmt[self.pmt_gt_index] <= self.pmt_threshold or pms[self.pms_gt_index] <= self.pms_threshold or pmb[self.pmb_gt_index] <= self.pmb_threshold:
                        self.pm_probability_novelty = True
                    else:
                        self.pm_probability_novelty = False

                mse_error = mse_loss(gt_motion, self.pm)

                self.accumulated_mse_error = self.accumulated_mse_error + mse_error
                self.motion_sequence = list()

                self.motion_std = LA.norm(motion_sequence_std)

                if self.motion_std > self.pmo_threshold:
                    self.pm_uncertainty_novelty = True
                else:
                    self.pm_uncertainty_novelty = False

                # TODO: Change wording. pm -> pb for pmt, pms, pmb
                if self.pm_probability_novelty is True and self.pm_uncertainty_novelty is True:
                    return True
                else:
                    return False

            except RuntimeError as e:
                print("Error", e)
                return False

    def get_mse_loss(self):
        return self.accumulated_mse_error

    def get_motion_std(self):
        return self.motion_std