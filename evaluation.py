# train and update LPNet

from modules.bmnet import MNet
from modules.probability import motion_probability, state_probability
from modules.data_filter import DataFilter
from modules.databatch_composer import DataBatchComposer
from modules.data_exchanger import DataExchanger
import carla
import json
import numpy as np
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from os import listdir
from os.path import isfile, join
# import carla.collect_online_data as run_file
import os
import shutil

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
from logging import getLogger

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist

logger = getLogger()

from modules.swav.src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)

from arguments import get_args
from modules.swav.src.multicropdataset import MultiCropDataset
import modules.swav.src.resnet50 as resnet_models

torch.set_num_threads(2)

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

MODEL_SAVE = True

MULTI_CROP_SIZE = 32
STATE_SIZE = 64
STATE_DIM = 3
MOTION_SIZE = 3
CLUSTER_DIM = 1000

THROTTLE_DISCR_DIM = 2
throttle_discr_th = 0.1

STEER_DISCR_DIM = 3
steer_discr_th1 = -0.2
steer_discr_th2 = 0.2

BRAKE_DISCR_DIM = 2
brake_discr_th = 0.1

DATASET_DIR = '/media/hsyoon/hard2/SDS/dataset/'
DATASET_IMAGE_DIR = '/media/hsyoon/hard2/SDS/dataset_image/image/'
ONLINE_DATA_DIR = '/media/hsyoon/hard2/SDS/dataset_online/'
ONLINE_DATA_IMAGE_DIR = '/media/hsyoon/hard2/SDS/dataset_online_image/image/'
REMOVAL_DATA_DIR = '/media/hsyoon/hard2/SDS/data_removal/'
TEST_DATA_DIR = '/media/hsyoon/hard2/SDS/test_dataset/'

MNET_MODEL_SAVE_DIR = './trained_models/mnet/'
PMT_MODEL_SAVE_DIR = './trained_models/pmt/'
PMS_MODEL_SAVE_DIR = './trained_models/pms/'
PMB_MODEL_SAVE_DIR = './trained_models/pmb/'
PO_MODEL_SAVE_DIR = './trained_models/po/'

def get_date():
    now = datetime.now()
    now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
    return now_date

def get_time():
    now = datetime.now()
    now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)
    return now_time

def test_model(model, pmtnet, pmsnet, pmbnet,
                test_dataset_dir, data_list, criterion_mse, criterion_bce,
                date, time, device):

    total_loss_mnet = 0
    total_loss_pt = 0
    total_loss_ps = 0
    total_loss_pb = 0

    for data_name in data_list:

        try:
            with open(test_dataset_dir + '/' + data_name) as tmp_json:
                json_data = json.load(tmp_json)
        except ValueError:
            print("JSON value error with ", data_name)
            continue
        except IOError:
            print("JSON IOerror with ", data_name)
            continue

        # Network(MNet, PMT, PMS, PMB) update
        if json_data['state'] is not None and len(json_data['motion']) is not 0 and len(json_data['state']) is 3:
            state_tensor = torch.tensor(json_data['state']).to(device)
            state_tensor = torch.reshape(state_tensor, (state_tensor.shape[0], state_tensor.shape[3], state_tensor.shape[1], state_tensor.shape[2])).float()


            model_output = model.forward(state_tensor).squeeze()
            loss = criterion_mse(model_output, torch.tensor(json_data['motion']).to(device))
            total_loss_mnet = total_loss_mnet + loss.cpu().detach().numpy()

            # PMNet update
            if json_data['motion'][0] <= throttle_discr_th:
                pmt_output_gt = torch.tensor([1, 0]).cuda().float()
            else:
                pmt_output_gt = torch.tensor([0, 1]).cuda().float()

            if json_data['motion'][1] <= steer_discr_th1:
                pms_output_gt = torch.tensor([1, 0, 0]).cuda().float()
            elif steer_discr_th1 <= json_data['motion'][0] <= steer_discr_th2:
                pms_output_gt = torch.tensor([0, 1, 0]).cuda().float()
            else:
                pms_output_gt = torch.tensor([0, 0, 1]).cuda().float()

            if json_data['motion'][2] <= brake_discr_th:
                pmb_output_gt = torch.tensor([1, 0]).cuda().float()
            else:
                pmb_output_gt = torch.tensor([0, 1]).cuda().float()

            with torch.no_grad():
                pmtnet_output = pmtnet.forward(state_tensor).squeeze()
                loss_pmt = criterion_bce(pmtnet_output, pmt_output_gt)
                total_loss_pt = total_loss_pt + loss_pmt.cpu().detach().numpy()

                pmsnet_output = pmsnet.forward(state_tensor).squeeze()
                loss_pms = criterion_bce(pmsnet_output, pms_output_gt)
                total_loss_ps = total_loss_ps + loss_pms.cpu().detach().numpy()

                pmbnet_output = pmbnet.forward(state_tensor).squeeze()
                loss_pmb = criterion_bce(pmbnet_output, pmb_output_gt)
                total_loss_pb = total_loss_pb + loss_pmb.cpu().detach().numpy()

    # Save loss!
    loss_mnet_txt = open('/home/hsyoon/job/SDS/log/' + date + '/' + time + '/test_loss_mnet.txt', 'a')
    loss_mnet_txt.write(str(total_loss_mnet) + '\n')
    loss_mnet_txt.close()

    loss_pmt_txt = open('/home/hsyoon/job/SDS/log/' + date + '/' + time + '/test_loss_pmt.txt', 'a')
    loss_pmt_txt.write(str(total_loss_pt) + '\n')
    loss_pmt_txt.close()

    loss_pms_txt = open('/home/hsyoon/job/SDS/log/' + date + '/' + time + '/test_loss_pms.txt', 'a')
    loss_pms_txt.write(str(total_loss_ps) + '\n')
    loss_pms_txt.close()

    loss_pmb_txt = open('/home/hsyoon/job/SDS/log/' + date + '/' + time + '/test_loss_pmb.txt', 'a')
    loss_pmb_txt.write(str(total_loss_pb) + '\n')
    loss_pmb_txt.close()


def main():

    model = MNet(STATE_SIZE, STATE_DIM, MOTION_SIZE, device)

    pmt_prob_model = motion_probability(STATE_SIZE, STATE_DIM, THROTTLE_DISCR_DIM, device)
    pms_prob_model = motion_probability(STATE_SIZE, STATE_DIM, STEER_DISCR_DIM, device)
    pmb_prob_model = motion_probability(STATE_SIZE, STATE_DIM, BRAKE_DISCR_DIM, device)

    start_date = get_date()
    start_time = get_time()

    if os.path.exists('/home/hsyoon/job/SDS/log/' + start_date + '/') is False:
        os.mkdir('/home/hsyoon/job/SDS/log/' + start_date)

    if os.path.exists('/home/hsyoon/job/SDS/log/' + start_date + '/' + start_time + '/') is False:
        os.mkdir('/home/hsyoon/job/SDS/log/' + start_date + '/' + start_time + '/')

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

    total_day_length = [f for f in listdir(MNET_MODEL_SAVE_DIR) if isfile(join(MNET_MODEL_SAVE_DIR, f))]
    total_day_length = len(total_day_length)

    test_data_list = [f2 for f2 in listdir(TEST_DATA_DIR) if isfile(join(TEST_DATA_DIR, f2))]

    # Test
    for i in range(total_day_length):

        model.load_state_dict(torch.load(MNET_MODEL_SAVE_DIR + 'day' + str(i) + '.pt'))
        pmt_prob_model.load_state_dict(torch.load(PMT_MODEL_SAVE_DIR + 'day' + str(i) + '.pt'))
        pms_prob_model.load_state_dict(torch.load(PMS_MODEL_SAVE_DIR + 'day' + str(i) + '.pt'))
        pmb_prob_model.load_state_dict(torch.load(PMB_MODEL_SAVE_DIR + 'day' + str(i) + '.pt'))

        test_model(model, pmt_prob_model, pms_prob_model, pmb_prob_model, TEST_DATA_DIR, test_data_list, criterion_mse, criterion_bce, start_date, start_time, device)
        print("Day", i, "test complete")

if __name__ == '__main__':
    main()