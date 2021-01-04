# from raw dataset, it combines 3 states and 1 actions as json file.

from os import listdir
from os.path import isfile, join
import json

import math
from PIL import Image
import PIL.Image as pilimg
import numpy as np
import time
from datetime import datetime
import numpy

IMAGE_DIR = '/media/hsyoon/hard2/SDS/dataset_raw/image/'
MOTION_DIR = '/media/hsyoon/hard2/SDS/dataset_raw/motion/'

DATA_SAVE_DIR = '/media/hsyoon/hard2/SDS/dataset/'

IMAGE_SEQUENCE = 3
IMAGE_SIZE = 64
count = 0

image_data_name_list = [fi for fi in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, fi))]
image_data_name_list.sort()

motion_data_name_list = [fm for fm in listdir(MOTION_DIR) if isfile(join(MOTION_DIR, fm))]
motion_data_name_list.sort()

im = Image.open(IMAGE_DIR + image_data_name_list[0])
im = im.resize((64, 64))

print(image_data_name_list[0])
print(motion_data_name_list[0])

if len(image_data_name_list) != len(motion_data_name_list):
    print("DATASET ERROR!")

for index in range(len(image_data_name_list)):
    if len(image_data_name_list) - index > 2:
        numpy_to_save = list()
        motion = None
        for seq in range(IMAGE_SEQUENCE):
            im = Image.open(IMAGE_DIR + image_data_name_list[index + seq])
            im = im.resize((IMAGE_SIZE, IMAGE_SIZE))

            numpy_im_tmp = numpy.asarray(im)
            numpy_im_tmp = numpy_im_tmp[:,:,:3]
            list_tmp = list(numpy_im_tmp)
            numpy_to_save.append(list_tmp)

        motion = np.loadtxt(MOTION_DIR + motion_data_name_list[index + IMAGE_SEQUENCE - 1])
        numpy_to_save = np.array(numpy_to_save)

        now = datetime.now()
        now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
        now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

        data = dict()
        data['state'] = numpy_to_save.tolist()
        data['motion'] = motion.tolist()

        print("state shape!", numpy_to_save.shape)
        if numpy_to_save.shape[0] == 3:
            with open(DATA_SAVE_DIR + now_date + "_" + now_time + "_" + str(index) + ".json", 'w', encoding='utf-8') as make_file:
                json.dump(data, make_file, indent="\t")