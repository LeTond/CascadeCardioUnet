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
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from sklearn import preprocessing  # pip install scikit-learn

from Training.dataset import MyDataset
from Training.ranger import Ranger
from Training.optimizer import Lion

from parameters import MetaParameters
from Preprocessing.dirs_logs import create_dir, create_dir_log, log_stats
from Model.unet2D import UNet_2D_mini, UNet_2D, UNet_2D_AttantionLayer, U_Net, CNN, UNetResnet, SegNet
from Model.FCT.utils.model import FCT
from Model.resnet import ResNet, BasicBlock

# from Model.models import bounding_box_CNN


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
    transforms.RandomRotation((-15, 15)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
])

