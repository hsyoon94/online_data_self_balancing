# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import numpy as np
import torch

class MultiCropDataset():
    def __init__(self, device):
        self.state = None
        self.device = device

    def augment_data(self, state, crop_size):
        # Randomly crop 2 augmented data
        state = torch.tensor(state).to(self.device).float()
        state = torch.reshape(state, (state.shape[0], state.shape[3], state.shape[1], state.shape[2]))
        print("state dim", state.shape)

        aug1_rndx = np.random.randint(state.shape[2] - crop_size, size=1)[0]
        aug1_rndy = np.random.randint(state.shape[2] - crop_size, size=1)[0]

        state_aug1 = state[:, :, aug1_rndx:aug1_rndx+crop_size, aug1_rndy:aug1_rndy+crop_size]

        aug2_rndx = np.random.randint(state.shape[2] - crop_size, size=1)[0]
        aug2_rndy = np.random.randint(state.shape[2] - crop_size, size=1)[0]

        state_aug2 = state[:, :, aug2_rndx:aug2_rndx + crop_size, aug2_rndy:aug2_rndy + crop_size]

        return state_aug1, state_aug2