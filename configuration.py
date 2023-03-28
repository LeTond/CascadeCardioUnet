import torch
import random
import sys
import time
import cv2
import matplotlib
import os
import pickle
import platform

import nibabel as nib
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import statsmodels.api as sm

from torch import nn
# from matplotlib import pylab as plt
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import preprocessing  # pip install scikit-learn

from Training.dataset import MyDataset
from parameters import MetaParameters
from Preprocessing.dirs_logs import create_dir, create_dir_log, log_stats
from Model.unet2D import UNet_2D_mini, UNet_2D, UNet_2D_AttantionLayer, U_Net, CNN, UNetResnet, SegNet
from Model.FCT.utils.model import FCT
from Model.resnet import ResNet, BasicBlock
from Training.ranger import Ranger
# from Model.models import bounding_box_CNN
from Training.optimizer import Lion
# from Preprocessing.save_to_pickle import SaveDataset

########################################################################################################################
# Show software and harware
########################################################################################################################
print(f"Python Platform: {platform.platform()}")
print(f'python version: {sys.version}')
print(f'torch version: {torch.__version__}')
print(f'numpy version: {np.__version__}')
print(f'pandas version: {pd.__version__}')


# from torchsummary import summary
# device = 'cpu'
# model = UNet_2D_AttantionLayer().to(device)
# summary(model,input_size=(1,256,256))


def device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


device = device()
print(device)

########################################################################################################################
# COMMENTS
########################################################################################################################
meta = MetaParameters()

create_dir_log(meta.PROJECT_NAME)

try:
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)
    model = torch.load(f'{meta.PROJECT_NAME}/{meta.MODEL_NAME}.pth').to(device=device)
    model.eval()
    # print(f'model loaded: {model}')
    print(f'model loaded: {meta.PROJECT_NAME}/{meta.MODEL_NAME}.pth')
except:
    print('no trained models')
    # model = UNet_2D_mini(drop = dropout, init_features = init_features).to(device)
    # model = UNet_2D().to(device)
    model = UNet_2D_AttantionLayer().to(device)
    # model = SegNet().to(device)
    # model = U_Net().to(device)
    # model = FCT().to(device)
    
loss_function = nn.CrossEntropyLoss(weight=meta.CE_WEIGHTS).to(device)
# loss_function = nn.CrossEntropyLoss().to(device)
# loss_function = FocalLoss(weight = weight_CE).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=meta.LR, weight_decay=meta.WDC)
# optimizer = Lion(model.parameters(), lr = learning_rate, betas=(0.9, 0.99), weight_decay = wdc)
# optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = wdc, amsgrad=False)
# optimizer = torch.optim.SGD(model.parameters(), lr=meta.LR, weight_decay=meta.WDC, momentum=0.9, nesterov=True)
# optimizer = Ranger(model.parameters(), lr = learning_rate, k = 6, N_sma_threshhold = 5, weight_decay = wdc)

scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=meta.TMAX, eta_min=0.0, last_epoch=-1, verbose=True)

########################################################################################################################
## Main image transforms in Dataloder
########################################################################################################################
target_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((meta.KERNEL_SZ, meta.KERNEL_SZ)),
    transforms.ToTensor(),
])
transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((meta.KERNEL_SZ, meta.KERNEL_SZ)),
    transforms.RandomRotation((-45, 45)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
])

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

test_ds_origin = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_test_origin.pickle'
)
test_ds_mask = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_test_mask.pickle'
)
test_ds_names = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_test_sub_names.pickle'
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
for i in range(7):
    train_set += MyDataset(meta.NUM_LAYERS, train_ds_origin, train_ds_mask, train_ds_names, meta.KERNEL_SZ, transform, target_transform)
train_loader = DataLoader(train_set, meta.BT_SZ, drop_last=True, shuffle=True, pin_memory=True)

valid_set = MyDataset(meta.NUM_LAYERS, valid_ds_origin, valid_ds_mask, valid_ds_names, meta.KERNEL_SZ, target_transform,
                      target_transform)
valid_batch_size = len(valid_set)
valid_loader = DataLoader(valid_set, valid_batch_size, drop_last=True, shuffle=True, pin_memory=True)

test_set = MyDataset(meta.NUM_LAYERS, test_ds_origin, test_ds_mask, test_ds_names, meta.KERNEL_SZ, target_transform, target_transform)
test_batch_size = len(test_set)
test_loader = DataLoader(test_set, test_batch_size//8, drop_last=True, shuffle=False, pin_memory=True)
