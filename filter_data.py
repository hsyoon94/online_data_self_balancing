from os import listdir
from os.path import isfile, join
import json
import numpy as np
import time
from datetime import datetime
import os
import shutil

data_raw_dir  = '/media/hsyoon/hard2/SDS/dataset_test/'
PROBABILITY_THRESHOLD = 0.90

data_name_list = [f for f in listdir(data_raw_dir) if isfile(join(data_raw_dir, f))]

out_count = 0
print(len(data_name_list))

for j in range(len(data_name_list)):
    # # FILTERING WITH ERROR
    # try:
    #     with open(data_raw_dir + data_name_list[j]) as tmp_json:
    #         json_data = json.load(tmp_json)
    # except ValueError:
    #     print("ONLINE JSON value error with ", data_name_list[j])
    #     os.remove(data_raw_dir + data_name_list[j])
    #     # shutil.move(, REMOVAL_DATA_DIR + online_data_name_list[online_data_index])
    #
    # except IOError:
    #     print("ONLINE JSON value error with ", data_name_list[j])
    #     os.remove(data_raw_dir + data_name_list[j])

    # RANDOM REMOVAL
    try:
        with open(data_raw_dir + '/' + data_name_list[j]) as myfile:
            json_myfile = json.load(myfile)
    except ValueError as val_e:
        print("VALUE ERROR", val_e)
        os.remove(data_raw_dir + '/' + data_name_list[j])
        continue
    except IOError as io_e:
        print("IO ERROR", io_e)
        os.remove(data_raw_dir + '/' + data_name_list[j])
        continue

    probability = np.random.uniform(0, 1, 1)

    if probability <= PROBABILITY_THRESHOLD:
        os.remove(data_raw_dir + '/' + data_name_list[j])
        print(j,"th FILE with", len(data_name_list), "total length moved with probability", probability)
        out_count = out_count + 1

print(out_count, "DATA REMOVED")