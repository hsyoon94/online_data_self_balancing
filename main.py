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

torch.set_num_threads(2)

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

def get_date():
    now = datetime.now()
    now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
    return now_date

def get_time():
    now = datetime.now()
    now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)
    return now_time

def train_model(day, iteration, model, dataset_dir, data_list, model_save_dir, criterion, optimizer, device):

    # databatch_composer = DataBatchComposer(dataset_dir, data_list)
    #
    # for iter in range(iteration):
    #
    #     batch_index = databatch_composer.get_databatch()
    #
    #     for i in range(batch_index.shape[0]):
    #
    #         try:
    #             with open(dataset_dir + '/' + data_list[batch_index[i]]) as tmp_json:
    #                 json_data = json.load(tmp_json)
    #         except ValueError:
    #             print("JSON value error with ", data_list[batch_index[i]])
    #             continue
    #         except IOError:
    #             print("JSON IOerror with ", data_list[batch_index[i]])
    #
    #         state_tensor = torch.tensor(json_data['state']).to(device)
    #
    #         optimizer.zero_grad()
    #         model_output = model.forward(state_tensor)
    #         loss = criterion(model_output, torch.tensor(json_data['motion']).to(device))
    #         loss.backward()
    #         optimizer.step()
    #
    #     if iter % 10 == 0:
    #         torch.save(model.state_dict(), model_save_dir + get_date() + '_' + get_time() + '_iter' + str(iter) + '.pt')

    torch.save(model.state_dict(), model_save_dir +'day' + str(day + 1) + '.pt')
    print("Day", day, "training ends and model saved to", get_date() + '_' + get_time() + '/final_of_day' + str(day+1) + '.pt')

def main():

    STATE_SIZE = 64
    STATE_DIM = 3
    MOTION_SIZE = 3

    TRAINING_ITERATION = 10000
    DATASET_DIR = '/media/hsyoon/hard2/SDS/dataset'
    MODEL_SAVE_DIR = './trained_models/'
    MODEL0_FILE = './trained_models/day0.pt'

    # data_exchanger = DataExchanger()
    model = MNet(STATE_SIZE, STATE_DIM, MOTION_SIZE, device)
    model.load_state_dict(torch.load(MODEL0_FILE))

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    forever = True
    day = 0

    print("INCREMENTAL INTELLIGENCE SYSTEM OPERATING...")
    while forever:
        # Collect novel online data in daytime.
        print("START COLLECTING ONLINE DATA...")

        # exec(open("./carla/collect_online_data.py").read())
        # run_file.run()

        os.system('/home/hsyoon/job/SDS/carla/collect_online_data.py')

        # TODO: update dataset with DataExchanger
        # TODO: No change in dataset yet.

        print("END COLLECTING ONLINE DATA...")

        data_list = [f for f in listdir(DATASET_DIR) if isfile(join(DATASET_DIR, f))]
        data_list.sort()

        train_model(day, TRAINING_ITERATION, model, DATASET_DIR, data_list, MODEL_SAVE_DIR, criterion, optimizer, device)

        # Go to next day and update policy network parameter.
        day = day + 1

        model.load_state_dict(torch.load(MODEL_SAVE_DIR + 'day' + str(day) + '.pt'))

    return 0

if __name__ == '__main__':
    main()