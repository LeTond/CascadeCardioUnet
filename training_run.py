 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1
Date: 29-03-2023
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""

from Training.train import *
from parameters import *
from configuration import *


########################################################################################################################
# Read datasets saved into pickle files
########################################################################################################################
train_ds_origin = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_train_{meta.FOLD_NAME}_origin.pickle'
)
train_ds_mask = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_train_{meta.FOLD_NAME}_mask.pickle'
)
train_ds_names = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_train_{meta.FOLD_NAME}_sub_names.pickle'
)

valid_ds_origin = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_valid_{meta.FOLD_NAME}_origin.pickle'
)
valid_ds_mask = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_valid_{meta.FOLD_NAME}_mask.pickle'
)
valid_ds_names = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_valid_{meta.FOLD_NAME}_sub_names.pickle'
)

########################################################################################################################
# If we want split all train_list_full to train and valid sets randomly by slices (not by full subjects)  
########################################################################################################################
# length_list2 = len(train_ds_names_full)

# train_ds_origin = train_ds_origin_full[:round(0.8*length_list2)]
# train_ds_mask = train_ds_mask_full[:round(0.8*length_list2)]
# train_ds_names = train_ds_names_full[:round(0.8*length_list2)]

# valid_ds_origin = train_ds_origin_full[round(0.8*length_list2):]
# valid_ds_mask = train_ds_mask_full[round(0.8*length_list2):]
# valid_ds_names = train_ds_names_full[round(0.8*length_list2):]

########################################################################################################################
# Creating loaders for training and validating network
########################################################################################################################
train_set = MyDataset(meta.NUM_LAYERS, train_ds_origin, train_ds_mask, train_ds_names, meta.KERNEL_SZ, target_transform,
                      target_transform)
for i in range(3):
    train_set += MyDataset(meta.NUM_LAYERS, train_ds_origin, train_ds_mask, train_ds_names, meta.KERNEL_SZ, transform, target_transform)
train_loader = DataLoader(train_set, meta.BT_SZ, drop_last=True, shuffle=True, pin_memory=True)

valid_set = MyDataset(meta.NUM_LAYERS, valid_ds_origin, valid_ds_mask, valid_ds_names, meta.KERNEL_SZ, target_transform,
                      target_transform)
valid_batch_size = len(valid_set)
valid_loader = DataLoader(valid_set, valid_batch_size, drop_last=True, shuffle=True, pin_memory=True)


print(f'Train size: {len(train_set)} | Valid size: {len(valid_set)}')
model = TrainNetwork(device, model, optimizer, loss_function, train_loader, valid_loader, meta, ds).train()
# model = TrainNetwork(device, model, optimizer, loss_function, valid_loader, train_loader, meta, ds).train()
