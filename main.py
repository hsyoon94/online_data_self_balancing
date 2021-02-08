# train and update LPNet

from modules.bmnet import MNet
from modules.probability import motion_probability
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

torch.set_num_threads(2)

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

MODEL_SAVE = False

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

TRAINING_ITERATION = 100
DATASET_DIR = '/media/hsyoon/hard2/SDS/dataset/'
ONLINE_DATA_DIR = '/media/hsyoon/hard2/SDS/dataset_online/'
REMOVAL_DATA_DIR = '/media/hsyoon/hard2/SDS/data_removal/'

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

    databatch_composer = DataBatchComposer(dataset_dir, data_list, entropy_threshold=1.0, databatch_size=1)

    for iter in range(iteration):

        total_loss_mnet = 0
        total_loss_pt = 0
        total_loss_ps = 0
        total_loss_pb = 0

        batch_index = databatch_composer.get_databatch_list()
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
                loss_pmt = criterion_bce(pmtnet_output, pmt_output_gt)
                loss_pmt.backward()
                optimizer_pmt.step()
                total_loss_pt = total_loss_pt + loss_pmt.cpu().detach().numpy()

                optimizer_pms.zero_grad()
                pmsnet_output = pmsnet.forward(state_tensor).squeeze()
                loss_pms = criterion_bce(pmsnet_output, pms_output_gt)
                loss_pms.backward()
                optimizer_pms.step()
                total_loss_ps = total_loss_ps + loss_pms.cpu().detach().numpy()

                optimizer_pmb.zero_grad()
                pmbnet_output = pmtnet.forward(state_tensor).squeeze()
                loss_pmb = criterion_bce(pmbnet_output, pmb_output_gt)
                loss_pmb.backward()
                optimizer_pmb.step()
                total_loss_pb = total_loss_pb + loss_pmb.cpu().detach().numpy()


        if iter % 10 == 0:
            print("Iteration", iter, "for day", day)

    # Save loss!
    loss_mnet_txt = open('/home/hsyoon/job/SDS/log/' + date + '_' + time + '_training_loss_mnet.txt', 'a')
    loss_mnet_txt.write(str(total_loss_mnet) + '\n')
    loss_mnet_txt.close()

    loss_pmt_txt = open('/home/hsyoon/job/SDS/log/' + date + '_' + time + '_training_loss_pmt.txt', 'a')
    loss_pmt_txt.write(str(total_loss_pt) + '\n')
    loss_pmt_txt.close()

    loss_pms_txt = open('/home/hsyoon/job/SDS/log/' + date + '_' + time + '_training_loss_pms.txt', 'a')
    loss_pms_txt.write(str(total_loss_ps) + '\n')
    loss_pms_txt.close()

    loss_pmb_txt = open('/home/hsyoon/job/SDS/log/' + date + '_' + time + '_training_loss_pmb.txt', 'a')
    loss_pmb_txt.write(str(total_loss_pb) + '\n')
    loss_pmb_txt.close()

    if MODEL_SAVE is True:
        torch.save(model.state_dict(), model_save_dir +'day' + str(day + 1) + '.pt')
        torch.save(pmtnet.state_dict(), pmt_save_dir + 'day' + str(day + 1) + '.pt')
        torch.save(pmsnet.state_dict(), pms_save_dir + 'day' + str(day + 1) + '.pt')
        torch.save(pmbnet.state_dict(), pmb_save_dir + 'day' + str(day + 1) + '.pt')
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

    # torch.save(model.state_dict(), MODEL_SAVE_DIR + 'day0.pt')
    # model.load_state_dict(torch.load(MNET_MODEL0_FILE))
    # pmt_prob_model.load_state_dict(torch.load(PMT_MODEL0_FILE))
    # pms_prob_model.load_state_dict(torch.load(PMS_MODEL0_FILE))
    # pmb_prob_model.load_state_dict(torch.load(PMB_MODEL0_FILE))

    optimizer_mnet = optim.Adam(model.parameters(), lr=0.0001)
    optimizer_pmt = optim.Adam(pmt_prob_model.parameters(), lr=0.0001)
    optimizer_pms = optim.Adam(pms_prob_model.parameters(), lr=0.0001)
    optimizer_pmb = optim.Adam(pmb_prob_model.parameters(), lr=0.0001)

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()
    forever = True
    day = 0

    # TODO: We have to initialize Pm0, Pb0, Po0 with D0, MNet0, Pb0, Po0 at this point.

    data_exchanger = DataExchanger(ONLINE_DATA_DIR, DATASET_DIR)

    print("[", get_date(), "-", get_time()[0:2], ":", get_time()[2:] , "]", "INCREMENTAL INTELLIGENCE SYSTEM OPERATING...", sep="")
    while forever:
        # Collect novel online data in daytime.
        print("[", get_date(), "-", get_time()[0:2], ":", get_time()[2:] , "]", "START COLLECTING ONLINE DATA...", sep="")
        command = 'python /home/hsyoon/job/SDS/carla/collect_online_data.py --date ' + start_date + ' --time ' + start_time
        print("COMMAND:", command)
        os.system(command)
        print("[", get_date(), "-", get_time()[0:2], ":", get_time()[2:] , "]", "END COLLECTING ONLINE DATA AND START UPDATING DATASET WITH APPENDING NOVEL ONLINE DATA...", sep="")

        # Update dataset then discard wrong data
        online_data_name_list = [f for f in listdir(ONLINE_DATA_DIR) if isfile(join(ONLINE_DATA_DIR, f))]
        for online_data_index in range(len(online_data_name_list)):
            try:
                with open(ONLINE_DATA_DIR + online_data_name_list[online_data_index]) as tmp_json:
                    json_data = json.load(tmp_json)
            except ValueError:
                print("ONLINE JSON value error with ", online_data_name_list[online_data_index])
                os.remove(ONLINE_DATA_DIR + online_data_name_list[online_data_index])
                # shutil.move(, REMOVAL_DATA_DIR + online_data_name_list[online_data_index])

            except IOError:
                print("ONLINE JSON IOerror with ", online_data_name_list[online_data_index])
                os.remove(ONLINE_DATA_DIR + online_data_name_list[online_data_index])
                # shutil.move(, REMOVAL_DATA_DIR + online_data_name_list[online_data_index])

        # Update online data name list
        online_data_name_list = [f for f in listdir(ONLINE_DATA_DIR) if isfile(join(ONLINE_DATA_DIR, f))]

        print("DATA EXCHANGE STARTS...")

        online_data_length = len(online_data_name_list)
        for odi in range(online_data_length):
            data_exchanger.exchange(online_data_name_list[odi])

        print("DATA EXCHANGE ENDS...")

        # TODO : Update data distribution index/rank in this step. (D0 -> D1, Pm0 -> Pm1, Pb0 -> Pb1, Po0 -> Po1)

        print("[", get_date(), "-", get_time()[0:2], ":", get_time()[2:] , "]", "DATASET UPDATE COMPLETE THEN GET READY FOR NEURAL NETWORK TRAINING...", sep="")

        data_list = [f for f in listdir(DATASET_DIR) if isfile(join(DATASET_DIR, f))]

        # Update MNet
        train_model(day, TRAINING_ITERATION, model, pmt_prob_model, pms_prob_model, pmb_prob_model,
                    DATASET_DIR, data_list, MNET_MODEL_SAVE_DIR, PMT_MODEL_SAVE_DIR, PMS_MODEL_SAVE_DIR, PMB_MODEL_SAVE_DIR,
                    criterion_mse, criterion_bce, optimizer_mnet, optimizer_pmt, optimizer_pms, optimizer_pmb, start_date, start_time, device)

        # Go to next day and update policy network parameter.
        day = day + 1

        if MODEL_SAVE is True:
            model.load_state_dict(torch.load(MNET_MODEL_SAVE_DIR + 'day' + str(day) + '.pt'))
            pms_prob_model.load_state_dict(torch.load(PMS_MODEL_SAVE_DIR + 'day' + str(day) + '.pt'))
            pmt_prob_model.load_state_dict(torch.load(PMT_MODEL_SAVE_DIR + 'day' + str(day) + '.pt'))
            pmb_prob_model.load_state_dict(torch.load(PMB_MODEL_SAVE_DIR + 'day' + str(day) + '.pt'))

    return 0

if __name__ == '__main__':
    main()