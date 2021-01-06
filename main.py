# train and update LPNet

from modules.bmnet import BMNet, MNet
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

MODEL_SAVE = True

STATE_SIZE = 64
STATE_DIM = 3
MOTION_SIZE = 3

TRAINING_ITERATION = 500
DATASET_DIR = '/media/hsyoon/hard2/SDS/dataset/'
ONLINE_DATA_DIR = '/media/hsyoon/hard2/SDS/dataset_online/'
REMOVAL_DATA_DIR = '/media/hsyoon/hard2/SDS/data_removal/'
MODEL_SAVE_DIR = './trained_models/'
MODEL0_FILE = './trained_models/day0.pt'

def get_date():
    now = datetime.now()
    now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
    return now_date

def get_time():
    now = datetime.now()
    now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)
    return now_time

def train_model(day, iteration, model, dataset_dir, data_list, model_save_dir, criterion, optimizer, device):

    databatch_composer = DataBatchComposer(dataset_dir, data_list, entropy_threshold=0.0, databatch_size=1)

    for iter in range(iteration):

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

            if json_data['state'] is not None and len(json_data['motion']) is not 0 and len(json_data['state']) is 3:
                state_tensor = torch.tensor(json_data['state']).to(device)
                state_tensor = torch.reshape(state_tensor, (state_tensor.shape[0], state_tensor.shape[3], state_tensor.shape[1], state_tensor.shape[2])).float()

                optimizer.zero_grad()
                model_output = model.forward(state_tensor).squeeze()
                loss = criterion(model_output, torch.tensor(json_data['motion']).to(device))
                loss.backward()
                optimizer.step()

        if MODEL_SAVE is True and iter % 100 == 0:
            # torch.save(model.state_dict(), model_save_dir + get_date() + '_' + get_time() + '_iter' + str(iter) + '.pt')
            print("Iteration", iter, "for day", day)

    if MODEL_SAVE is True:
        torch.save(model.state_dict(), model_save_dir +'day' + str(day + 1) + '.pt')
        print("Day", day, "training ends and model saved to", get_date() + '_' + get_time() + '/final_of_day' + str(day + 1) + '.pt')
    print("[FAKE] Day", day, "training ends and model saved to", get_date() + '_' + get_time() + '/final_of_day' + str(day + 1) + '.pt')


def main():

    # data_exchanger = DataExchanger()
    model = MNet(STATE_SIZE, STATE_DIM, MOTION_SIZE, device)
    start_date = get_date()
    start_time = get_time()

    # torch.save(model.state_dict(), MODEL_SAVE_DIR + 'day0.pt')
    # model.load_state_dict(torch.load(MODEL0_FILE))

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    forever = True
    day = 0

    data_exchanger = DataExchanger(ONLINE_DATA_DIR, DATASET_DIR)

    print("[", get_date(), "-", get_time()[0:2], ":", get_time()[2:] , "]", "INCREMENTAL INTELLIGENCE SYSTEM OPERATING...", sep="")
    while forever:
        # Collect novel online data in daytime.
        print("[", get_date(), "-", get_time()[0:2], ":", get_time()[2:] , "]", "START COLLECTING ONLINE DATA...", sep="")
        command = 'python /home/hsyoon/job/SDS/carla/collect_online_data.py --date ' + start_date + ' --time ' + start_time
        print("COMMAND:", command)
        os.system(command)
        # os.system('/home/hsyoon/job/SDS/carla/collect_online_data.py --date', start_date, '--time', start_time)
        print("[", get_date(), "-", get_time()[0:2], ":", get_time()[2:] , "]", "END COLLECTING ONLINE DATA AND START UPDATING DATASET WITH APPENDING NOVEL ONLINE DATA...", sep="")

        # Update dataset then discard wrong data
        online_data_name_list = [f for f in listdir(ONLINE_DATA_DIR) if isfile(join(ONLINE_DATA_DIR, f))]
        for online_data_index in range(len(online_data_name_list)):
            try:
                with open(ONLINE_DATA_DIR + online_data_name_list[online_data_index]) as tmp_json:
                    json_data = json.load(tmp_json)
            except ValueError:
                print("ONLINE JSON value error with ", online_data_name_list[online_data_index])
                shutil.move(ONLINE_DATA_DIR + online_data_name_list[online_data_index], REMOVAL_DATA_DIR + online_data_name_list[online_data_index])

            except IOError:
                print("ONLINE JSON IOerror with ", online_data_name_list[online_data_index])
                shutil.move(ONLINE_DATA_DIR + online_data_name_list[online_data_index], REMOVAL_DATA_DIR + online_data_name_list[online_data_index])

        # Update online data name list
        online_data_name_list = [f for f in listdir(ONLINE_DATA_DIR) if isfile(join(ONLINE_DATA_DIR, f))]

        online_data_length = len(online_data_name_list)
        for odi in range(online_data_length):
            data_exchanger.exchange(online_data_name_list[odi])

        print("[", get_date(), "-", get_time()[0:2], ":", get_time()[2:] , "]", "DATASET UPDATE COMPLETE THEN GET READY FOR NEURAL NETWORK TRAINING...", sep="")

        data_list = [f for f in listdir(DATASET_DIR) if isfile(join(DATASET_DIR, f))]
        # data_list.sort()

        # Train dataset
        train_model(day, TRAINING_ITERATION, model, DATASET_DIR, data_list, MODEL_SAVE_DIR, criterion, optimizer, device)

        # Go to next day and update policy network parameter.
        day = day + 1

        if MODEL_SAVE is True:
            model.load_state_dict(torch.load(MODEL_SAVE_DIR + 'day' + str(day) + '.pt'))

    return 0

if __name__ == '__main__':
    main()