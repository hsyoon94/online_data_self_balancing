# DE

import numpy as np
import torch
# import probability
import torch.nn as nn
import math

torch.set_num_threads(2)

is_cuda = torch.cuda.is_available()

# device = torch.device('cuda' if is_cuda else 'cpu')
device = 'cpu'

from .bmnet import MNet
from os import listdir
from os.path import isfile, join

MNET_MODEL_DIR = "/home/hsyoon/job/SDS/trained_models/"

def mse_loss(arr1, arr2):
    error = 0
    for i in range(arr1.shape[0]):
        error = error + math.pow(arr1[i] - arr2[i], 2)

    error = error / arr1.shape[0]

    return error

class DataFilter():
    def __init__(self, mnet_state_size, mnet_state_dim, mnet_motion_size):
        self.pm = None
        self.pb = None
        self.po = None

        self.pm_threshold = 0.5
        self.pb_threshold = 0.5
        self.po_threshold = 0.5
        self.pm_mse_threshold = 0.0001

        self.pmbo_threshold = 0.7
        self.MSELOSS = nn.MSELoss()
        self.mnet_filter = MNet(mnet_state_size, mnet_state_dim, mnet_motion_size, device)
        # FOR THE LATEST MNET
        mnet_list = [f for f in listdir(MNET_MODEL_DIR) if isfile(join(MNET_MODEL_DIR, f))]
        mnet_list.sort()
        self.mnet_filter.load_state_dict(torch.load(MNET_MODEL_DIR + mnet_list[-1]))
        self.accumulated_mse_error = 0

    def is_novel(self, online_state, gt_motion):

        with torch.no_grad():
            online_state_tensor = torch.tensor(online_state).to(device)
            online_state_tensor = torch.reshape(online_state_tensor, (online_state_tensor.shape[0], online_state_tensor.shape[3], online_state_tensor.shape[1], online_state_tensor.shape[2])).float()

            try:
                pm = self.mnet_filter(online_state_tensor)
                pm = pm.cpu().detach().numpy().squeeze()
                self.pm = pm

                mse_error = mse_loss(gt_motion, self.pm)
                self.accumulated_mse_error = self.accumulated_mse_error + mse_error

                random_num = np.random.uniform(0, 1, 1)
                # if mse_error > self.pm_mse_threshold:
                if random_num > 0.5:
                    return True
                else:
                    return False

            except RuntimeError:
                return False




    def get_mse_loss(self):
        return self.accumulated_mse_error