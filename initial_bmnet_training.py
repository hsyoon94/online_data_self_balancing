# train and update initial BMNet or MNet

from modules.bmnet import BMNet, MNet

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
import shutil

torch.set_num_threads(2)

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

MODEL_SAVE = True

STATE_SIZE = 64
STATE_DIM = 3
MOTION_SIZE = 3

TRAINING_ITERATION = 2000
DATASET_DIR = '/media/hsyoon/hard2/SDS/dataset/'
MODEL_SAVE_DIR = './trained_models/mnet/'
MODEL0_FILE = './trained_models/mnet/day0.pt'

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

        if MODEL_SAVE is True and iter % 5 == 0:
            torch.save(model.state_dict(), model_save_dir + get_date() + '_' + get_time() + '_iter' + str(iter) + '.pt')
            print("Iteration", iter, "for day", day)

    if MODEL_SAVE is True:
        torch.save(model.state_dict(), model_save_dir +'day' + str(day + 1) + '.pt')
        print("Day", day, "training ends and model saved to", get_date() + '_' + get_time() + '/final_of_day' + str(day + 1) + '.pt')
    print("[FAKE] Day", day, "training ends and model saved to", get_date() + '_' + get_time() + '/final_of_day' + str(day + 1) + '.pt')

def main():

    # data_exchanger = DataExchanger()
    # TODO: Also initially train pm3nets
    model = MNet(STATE_SIZE, STATE_DIM, MOTION_SIZE, device)
    start_date = get_date()
    start_time = get_time()

    # torch.save(model.state_dict(), MODEL_SAVE_DIR + 'day0.pt')
    # model.load_state_dict(torch.load(MODEL0_FILE))

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    forever = True
    day = 0
    data_list = [f for f in listdir(DATASET_DIR) if isfile(join(DATASET_DIR, f))]

    train_model(day, TRAINING_ITERATION, model, DATASET_DIR, data_list, MODEL_SAVE_DIR, criterion, optimizer, device)
    print("TRAINING ENDS!")

if __name__ == '__main__':
    main()