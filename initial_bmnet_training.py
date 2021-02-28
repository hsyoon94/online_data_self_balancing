# train and update MNet

from modules.bmnet import MNet
from modules.probability import motion_probability
from modules.databatch_composer import DataBatchComposer

import carla
import json
import numpy as np
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import os

torch.set_num_threads(2)

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

MODEL_SAVE = True

STATE_SIZE = 64
STATE_DIM = 3
MOTION_SIZE = 3

THROTTLE_DISCR_DIM = 2
throttle_discr_th = 0.1

STEER_DISCR_DIM = 3
steer_discr_th1 = -0.2
steer_discr_th2 = 0.2

BRAKE_DISCR_DIM = 2
brake_discr_th = 0.1

TRAINING_ITERATION = 10000
DATASET_DIR = '/media/hsyoon/hard2/SDS/dataset_initial/'
# ONLINE_DATA_DIR = '/media/hsyoon/hard2/SDS/dataset_online/'
# REMOVAL_DATA_DIR = '/media/hsyoon/hard2/SDS/data_removal/'

MNET_MODEL_SAVE_DIR = './trained_models/mnet/'
PMT_MODEL_SAVE_DIR = './trained_models/pmt/'
PMS_MODEL_SAVE_DIR = './trained_models/pms/'
PMB_MODEL_SAVE_DIR = './trained_models/pmb/'

MNET_MODEL0_FILE = './trained_models/mnet/day0.pt'
PMT_MODEL0_FILE = './trained_models/pmt/day0.pt'
PMS_MODEL0_FILE = './trained_models/pms/day0.pt'
PMB_MODEL0_FILE = './trained_models/pmb/day0.pt'

def get_date():
    now = datetime.now()
    now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
    return now_date

def get_time():
    now = datetime.now()
    now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)
    return now_time

def train_model(day, iteration, model, pmtnet, pmsnet, pmbnet, dataset_dir, data_list, model_save_dir, pmt_save_dir, pms_save_dir, pmb_save_dir, criterion_mse, criterion_bce, optimizer_mnet, optimizer_pmt, optimizer_pms, optimizer_pmb, date, time, device):

    databatch_composer = DataBatchComposer(dataset_dir, 0.0, 1, 64, 3, date, time)

    total_loss_mnet = 0
    total_loss_pt = 0
    total_loss_ps = 0
    total_loss_pb = 0

    for iter in range(iteration):

        batch_index = databatch_composer.extract_random_batch_for_initial_training()
        for i in range(batch_index.shape[0]):

            try:
                with open(dataset_dir + '/' + data_list[batch_index[i]]) as tmp_json:
                    json_data = json.load(tmp_json)
            except ValueError:
                print("JSON value error with ", data_list[batch_index[i]])
                continue
            except IOError:
                print("JSON IOerror with ", data_list[batch_index[i]])
                continue

            # Network(MNet, PMT, PMS, PMB) update
            if json_data['state'] is not None and len(json_data['motion']) is not 0 and len(json_data['state']) is 3:
                state_tensor = torch.tensor(json_data['state']).to(device)
                state_tensor = torch.reshape(state_tensor, (state_tensor.shape[0], state_tensor.shape[3], state_tensor.shape[1], state_tensor.shape[2])).float()

                optimizer_mnet.zero_grad()
                model_output = model.forward(state_tensor).squeeze()
                loss = criterion_mse(model_output, torch.tensor(json_data['motion']).to(device))
                loss.backward()
                optimizer_mnet.step()
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

                optimizer_pmt.zero_grad()
                pmtnet_output = pmtnet.forward(state_tensor).squeeze()
                # loss_pmt = criterion_bce(pmtnet_output, pmt_output_gt)
                loss_pmt = criterion_mse(pmtnet_output, pmt_output_gt)
                loss_pmt.backward()
                optimizer_pmt.step()
                total_loss_pt = total_loss_pt + loss_pmt.cpu().detach().numpy()

                optimizer_pms.zero_grad()
                pmsnet_output = pmsnet.forward(state_tensor).squeeze()
                # loss_pms = criterion_bce(pmsnet_output, pms_output_gt)
                loss_pms = criterion_mse(pmsnet_output, pms_output_gt)
                loss_pms.backward()
                optimizer_pms.step()
                total_loss_ps = total_loss_ps + loss_pms.cpu().detach().numpy()

                optimizer_pmb.zero_grad()
                pmbnet_output = pmtnet.forward(state_tensor).squeeze()
                # loss_pmb = criterion_bce(pmbnet_output, pmb_output_gt)
                loss_pmb = criterion_mse(pmbnet_output, pmb_output_gt)
                loss_pmb.backward()
                optimizer_pmb.step()
                total_loss_pb = total_loss_pb + loss_pmb.cpu().detach().numpy()

        # if MODEL_SAVE is True and iter % 50 == 0:
        #     # torch.save(model.state_dict(), model_save_dir + get_date() + '_' + get_time() + '_iter' + str(iter) + '.pt')
        #     print("Iteration", iter, "for day", day)
        #
        #     torch.save(model.state_dict(), model_save_dir + 'iter' + str(iter + 1) + '.pt')
        #     torch.save(pmtnet.state_dict(), pmt_save_dir + 'iter' + str(iter + 1) + '.pt')
        #     torch.save(pmsnet.state_dict(), pms_save_dir + 'iter' + str(iter + 1) + '.pt')
        #     torch.save(pmbnet.state_dict(), pmb_save_dir + 'iter' + str(iter + 1) + '.pt')

        if iter % 10 == 0:
            loss_mnet_txt = open('/home/hsyoon/job/SDS/log/' + date + '/' + time + '/initial_training_loss_mnet.txt', 'a')
            loss_mnet_txt.write(str(total_loss_mnet) + '\n')
            loss_mnet_txt.close()

            loss_pmt_txt = open('/home/hsyoon/job/SDS/log/' + date + '/' + time + '/initial_training_loss_pmt.txt', 'a')
            loss_pmt_txt.write(str(total_loss_pt) + '\n')
            loss_pmt_txt.close()

            loss_pms_txt = open('/home/hsyoon/job/SDS/log/' + date + '/' + time + '/initial_training_loss_pms.txt', 'a')
            loss_pms_txt.write(str(total_loss_ps) + '\n')
            loss_pms_txt.close()

            loss_pmb_txt = open('/home/hsyoon/job/SDS/log/' + date + '/' + time + '/initial_training_loss_pmb.txt', 'a')
            loss_pmb_txt.write(str(total_loss_pb) + '\n')
            loss_pmb_txt.close()

            total_loss_mnet = 0
            total_loss_pt = 0
            total_loss_ps = 0
            total_loss_pb = 0

            print("Iter", iter, "training done for total interation ", iteration, "...")

    if MODEL_SAVE is True:
        torch.save(model.state_dict(), model_save_dir +'initial_day' + str(day + 1) + '.pt')
        torch.save(pmtnet.state_dict(), pmt_save_dir + 'initial_day' + str(day + 1) + '.pt')
        torch.save(pmsnet.state_dict(), pms_save_dir + 'initial_day' + str(day + 1) + '.pt')
        torch.save(pmbnet.state_dict(), pmb_save_dir + 'initial_day' + str(day + 1) + '.pt')
        print("Day", day, "training ends and model saved to", get_date() + '_' + get_time() + '/final_of_day' + str(day + 1) + '.pt')

    else:
        print("[FAKE: NOT SAVED] Day", day, "training ends and model saved to", get_date() + '_' + get_time() + '/final_of_day' + str(day + 1) + '.pt')


def main():

    # data_exchanger = DataExchanger()
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

    optimizer_mnet = optim.Adam(model.parameters(), lr=0.0001)
    optimizer_pmt = optim.Adam(pmt_prob_model.parameters(), lr=0.0001)
    optimizer_pms = optim.Adam(pms_prob_model.parameters(), lr=0.0001)
    optimizer_pmb = optim.Adam(pmb_prob_model.parameters(), lr=0.0001)

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

    day = 0

    print("[", get_date(), "-", get_time()[0:2], ":", get_time()[2:] , "]", "INITIAL INCREMENTAL INTELLIGENCE SYSTEM OPERATING...", sep="")

    data_list = [f for f in listdir(DATASET_DIR) if isfile(join(DATASET_DIR, f))]

        # Train dataset
    train_model(day, TRAINING_ITERATION, model, pmt_prob_model, pms_prob_model, pmb_prob_model,
                    DATASET_DIR, data_list, MNET_MODEL_SAVE_DIR, PMT_MODEL_SAVE_DIR, PMS_MODEL_SAVE_DIR, PMB_MODEL_SAVE_DIR,
                    criterion_mse, criterion_bce, optimizer_mnet, optimizer_pmt, optimizer_pms, optimizer_pmb, start_date, start_time, device)

    # if MODEL_SAVE is True:
    #     model.load_state_dict(torch.load(MNET_MODEL_SAVE_DIR + 'day' + str(day) + '.pt'))

    return 0

if __name__ == '__main__':
    main()