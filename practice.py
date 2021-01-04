import json
import numpy as np

json1 = '/media/hsyoon/hard2/SDS/dataset_online/210104_1747_305540.json'
json2 = '/media/hsyoon/hard2/SDS/dataset/201228_1320_104.json'


with open(json1) as tmp_json1:
    print(tmp_json1)
    file = json.load(tmp_json1)
    state = file['state']
    motion = file['motion']


    print(np.array(state).shape)
    print(np.array(motion).shape)



with open(json2) as tmp_json2:
    print(tmp_json2)
    file = json.load(tmp_json2)
    state = file['state']
    motion = file['motion']


    print(np.array(state).shape)
    print(np.array(motion).shape)