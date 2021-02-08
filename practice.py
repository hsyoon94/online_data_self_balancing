import torch
import numpy as np
import json

model = torch.hub.load('facebookresearch/swav', 'resnet50')

json_file_dir = '/media/hsyoon/hard2/SDS/dataset/210201_0914_4790160.json'

with open(json_file_dir) as tmp_json:
    json_data = json.load(tmp_json)

# print(json_data['state'])
# print(model)
json_data = torch.tensor(json_data['state']).float()

json_data = torch.reshape(json_data, [64, 3, 64, 3])
print(json_data.shape)
print("output", model(json_data))
print("output shape", model(json_data).shape)