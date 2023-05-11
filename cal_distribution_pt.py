from data_load_sti import Dataset_dict, collate_fn
from torchvision import transforms, utils
import torch
import json
import pickle
import os
import numpy as np

train_dict_root = "/data/xbj/0929_STI_catTIMIT_withNOISE_DATA/train"
data_transform = transforms.Compose([transforms.Resize([224, 224],antialias=True)])
failed_file = "/data/xbj/0929_STI_catTIMIT_withNOISE_DATA/train$ ls Six_Config/Six_Config_BatteryBenson_left_Six_Config_DR4_FDKN0_TIMIT_S_1000dB-0.pt"

train_transformed_dataset = Dataset_dict(root_dir=train_dict_root, transform=data_transform,
                                             start_freq=0, end_freq=29,
                                             failed_file=failed_file, random_choose_slice=False)

print("len of train dataset:", len(train_transformed_dataset))

data_dict={}

for i in range(len(train_transformed_dataset)):
    #print(i)
    print(train_transformed_dataset[i])
    label=round(train_transformed_dataset[i]["sti"][0].item(),1)
    
    if label not in data_dict:
        data_dict[label] = 0    
    data_dict[label] = data_dict[label] + 1 

arr=np.array(list(data_dict.items()))
np.save('add_data_distribution_pt.npy', arr)     
print("saved")
input("Press Enter to continue...")

