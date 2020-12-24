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
import databatch_composer as DataBatchComposer
import os

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
