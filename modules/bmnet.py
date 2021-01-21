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
        super(MNet, self).__init__()
        self.state_shape = state_shape
        self.state_channel = state_channel
        self.motion_shape = motion_shape
        self.device = device
        self.cnn_mid_channel = 16
        self.dropout= nn.Dropout(p=0.1)

        self.F1_DIM = 128
        self.F2_DIM = 64
        self.ReLU = nn.ReLU().to(self.device)
        self.Sigmoid = nn.Sigmoid().to(self.device)
        self.Tanh = nn.Tanh().to(self.device)
        self.CNN1 = nn.Conv2d(self.state_channel, self.cnn_mid_channel, 1, stride=3).to(self.device)
        self.CNN2 = nn.Conv2d(self.cnn_mid_channel, 3, 1, stride=3).to(self.device)
        self.MaxPool1 = nn.MaxPool2d(3, 1).to(self.device)

        self.F1 = nn.Linear(324, self.F1_DIM).to(self.device)
        self.F2 = nn.Linear(self.F1_DIM, self.F2_DIM).to(self.device)
        self.F3 = nn.Linear(self.F2_DIM, self.motion_shape).to(self.device)

    def forward(self, state):

        motion = self.CNN1(state)
        motion = self.dropout(motion)
        motion = self.CNN2(motion)
        motion = self.dropout(motion)
        motion = self.MaxPool1(motion)

        self.F1_DIM = motion.shape[0] * motion.shape[1] * motion.shape[2] * motion.shape[3]
        motion = motion.view(-1, self.F1_DIM)  # reshape Variable

        motion = self.F1(motion)
        motion = self.dropout(motion)
        motion = self.ReLU(motion)

        motion = self.F2(motion)
        motion = self.dropout(motion)
        motion = self.ReLU(motion)

        motion = self.F3(motion)

        motion[0][0] = self.Sigmoid(motion[0][0])
        motion[0][1] = self.Tanh(motion[0][1])
        motion[0][2] = self.Sigmoid(motion[0][2])

        return motion
