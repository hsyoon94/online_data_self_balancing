# train and update LPNet

from modules.bmnet import BMNet, MNet
from modules.data_filter import DataFilter
from modules.databatch_composer import DataBatchComposer
from modules.data_exchanger import DataExchanger
import carla
import json
import numpy as np
from carla.run import CarlaWorld
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from os import listdir
from os.path import isfile, join

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
    databatch_composer = DataBatchComposer(dataset_dir, data_list)

    for iter in range(iteration):

        batch_index = databatch_composer.get_databatch()

        for i in range(batch_index.shape[0]):

            try:
                with open(dataset_dir + '/' + data_list[batch_index[i]]) as tmp_json:
                    json_data = json.load(tmp_json)
            except ValueError:
                print("JSON value error with ", data_list[batch_index[i]])
                continue
            except IOError:
                print("JSON IOerror with ", data_list[batch_index[i]])

            state_tensor = torch.tensor(json_data['state']).to(device)

            optimizer.zero_grad()
            model_output = model.forward(state_tensor)
            loss = criterion(model_output, torch.tensor(json_data['motion']).to(device))
            loss.backward()
            optimizer.step()

        if iter % 10 == 0:
            torch.save(model.state_dict(), model_save_dir + get_date() + '_' + get_time() + '_iter' + str(iter) + '.pt')

    torch.save(model.state_dict(), model_save_dir + get_date() + '_' + get_time() + '/final_of_day' + day + '.pt')
    print("Day", day, "training ends and model saved to", get_date() + '_' + get_time() + '/final_of_day' + day + '.pt')

def main():

    STATE_SIZE = 64
    MOTION_SIZE = 3

    TRAINING_ITERATION = 10000
    DATASET_DIR = '/media/hsyoon/hard2/SDS/dataset'
    MODEL_SAVE_DIR = './trained_model/'
    MODEL0_FILE = './trained_model/day0.pt'

    data_exchanger = DataExchanger()
    model = MNet(STATE_SIZE, MOTION_SIZE)
    model.load_state_dict(torch.load(MODEL0_FILE))

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSE()
    forever = True
    day = 0

    print("INCREMENTAL INTELLIGENCE SYSTEM OPERATING...")
    while forever:
        # Collect novel online data in daytime.
        print("START COLLECTING ONLINE DATA...")

        # TODO: exec('./carla/collect_online_data.py') (filtering is done at this file and novel data saved to dataset_online
        # TODO: update dataset with DataExchanger

        print("END COLLECTING ONLINE DATA...")

        data_list = [f for f in listdir(DATASET_DIR) if isfile(join(DATASET_DIR, f))]
        data_list.sort()

        train_model(day, TRAINING_ITERATION, model, DATASET_DIR, data_list, MODEL_SAVE_DIR, criterion, optimizer, device)

        # Go to next day and update policy network parameter.
        day = day + 1
        # TODO: Unify the file saved name with using just "date"
        model.load_state_dict(torch.load(MODEL_SAVE_DIR + 'day' + day + '.pt'))

    return 0

if __name__ == '__main__':
    main()