import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as utils
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from os import listdir
from os.path import isfile, join
import json

import math
from PIL import Image
import PIL.Image as pilimg
import numpy as np
import time
from datetime import datetime
import os

class MNet(nn.Module):
    def __init__(self, state_shape, state_channel, motion_shape, device):

        self.state_shape = state_shape
        self.state_channel = state_channel
        self.motion_shape = motion_shape
        self.device = device
        self.cnn_mid_channel = 16

        self.F1_DIM = 128
        self.F2_DIM = 64
        self.ReLU = nn.ReLU().to(self.device)
        self.ReLU = nn.Sigmoid().to(self.device)
        self.CNN1 = nn.CNN(self.state_channel, self.cnn_mid_channel, stride=3).to(self.device)
        self.CNN2 = nn.CNN(self.cnn_mid_channel, 3, stride=3).to(self.device)
        self.MaxPool1 = nn.MaxPool2d(3, 1).to(self.device)

        self.F1 = nn.Linear(6724, self.F1_DIM).to(self.device)
        self.F2 = nn.Linear(self.F1_DIM, self.F2_DIM).to(self.device)
        self.F3 = nn.Linear(self.F2_DIM, self.motion_shape).to(self.device)

        self.model = nn.Sequential(
            self.CNN1,
            self.CNN2,
            self.MaxPool1,
            self.F1,
            self.ReLU,
            self.F2,
            self.ReLU,
            self.F3,
            self.Sigmoid
        )

    def forward(self, state):
        motion = self.model(state)
        return motion


class BMNet(nn.Module):
    def __init__(self, state_shape, behavior_shape, motion_shape):

        # TODO: Design BMNet specifically

        self.state_shape = state_shape
        self.behavior_shape = behavior_shape
        self.motion_shape = motion_shape
        # TODO: Separate bnet and mnet
        self.model = nn.Sequential()

    def forward(self, state):

        # TODO: Seperately implement forward function of bnet and mnet
        behavior, motion = self.model(state)

        return behavior, motion
