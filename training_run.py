 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1
Date: 29-03-2023
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""

from parameters import *
from configuration import *
from Training.train import *
from Training.dataset import *
from Preprocessing.split_dataset import *


########################################################################################################################
# Creating loaders for training and validating network
########################################################################################################################
# class Training(MetaParameters):
#     def __init__(self):         
#         super(MetaParameters, self).__init__()


train_ds = GetData(train_list).generated_data_list()    
valid_ds = GetData(valid_list).generated_data_list()

train_ds_origin = train_ds[0]
train_ds_mask = train_ds[1]
train_ds_names = train_ds[2]

valid_ds_origin = valid_ds[0]
valid_ds_mask = valid_ds[1]
valid_ds_names = valid_ds[2]

if meta.CROPPING is True:
    kernel_sz = meta.CROPP_KERNEL
elif meta.CROPPING is False:
    kernel_sz = meta.KERNEL

train_set = MyDataset(meta.NUM_LAYERS, train_ds_origin, train_ds_mask, train_ds_names, kernel_sz, default_transform)
for i in range(7):
    train_set += MyDataset(meta.NUM_LAYERS, train_ds_origin, train_ds_mask, train_ds_names, kernel_sz, transform)
train_loader = DataLoader(train_set, meta.BT_SZ, drop_last=True, shuffle=True, pin_memory=False)


valid_set = MyDataset(meta.NUM_LAYERS, valid_ds_origin, valid_ds_mask, valid_ds_names, kernel_sz, default_transform)
valid_batch_size = len(valid_set)
valid_loader = DataLoader(valid_set, valid_batch_size, drop_last=True, shuffle=True, pin_memory=False)


print(f'Train size: {len(train_set)} | Valid size: {len(valid_set)}')
model = TrainNetwork(model, optimizer, loss_function, train_loader, valid_loader, meta, ds).train()




