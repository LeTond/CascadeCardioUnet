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
import torch.nn.functional as F
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
# from Model.FCT.utils.model import FCT
# from Model.resnet import ResNet, BasicBlock

# from Model.models import bounding_box_CNN


########################################################################################################################
# Show software and harware
########################################################################################################################
print(f"Python Platform: {platform.platform()}")
print(f'python version: {sys.version}')
print(f'torch version: {torch.__version__}')
print(f'numpy version: {np.__version__}')
print(f'pandas version: {pd.__version__}')


global device


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


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight = None, gamma = 2,reduction = 'mean'):    #reduction='sum'
        super(FocalLoss, self).__init__(weight,reduction = reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

########################################################################################################################
# COMMENTS
########################################################################################################################
meta = MetaParameters()

create_dir_log(meta.PROJ_NAME)
create_dir_log(meta.CROPP_PROJ_NAME)


if meta.CROPPING is True:
    projec_name = meta.CROPP_PROJ_NAME
elif meta.CROPPING is False:
    projec_name = meta.PROJ_NAME

try:
    model = torch.load(f'{projec_name}/{meta.MODEL_NAME}.pth').to(device=device)
    model.eval()
    print(f'model loaded: {projec_name}/{meta.MODEL_NAME}.pth')
except:
    print('no trained models')
    model = UNet_2D_AttantionLayer().to(device=device)
    
loss_function = nn.CrossEntropyLoss(weight=meta.CE_WEIGHTS).to(device)
# loss_function = nn.CrossEntropyLoss().to(device)
# loss_function = FocalLoss(weight = meta.CE_WEIGHTS).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=meta.LR, weight_decay=meta.WDC)
# optimizer = Lion(model.parameters(), lr = meta.LR, betas=(0.9, 0.99), weight_decay = meta.WDC)
# optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = wdc, amsgrad=False)
# optimizer = torch.optim.SGD(model.parameters(), lr=meta.LR, weight_decay=meta.WDC, momentum=0.9, nesterov=True)
# optimizer = Ranger(model.parameters(), lr = meta.LR, k = 6, N_sma_threshhold = 5, weight_decay = meta.WDC)

scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=meta.TMAX, eta_min=0.0, last_epoch=-1, verbose=True)

########################################################################################################################
## Main image transforms in Dataloder
########################################################################################################################
default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

transform_01 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation((-15, 15), expand=False),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
])

transform_02 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomAffine(degrees=(-45, 45), translate=(0.05, 0.15), scale=(0.75, 1.5)),
    transforms.ToTensor(),
])

transform_03 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.GaussianBlur(19), 
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.3), scale=(0.5, 2.0)),
    transforms.ToTensor(),
])

transform_04 = transforms.Compose([
    transforms.ToPILImage(),    
    transforms.GaussianBlur(19), 
    transforms.RandomRotation((-15, 15), expand=False),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
])

transform_05 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.GaussianBlur(19), 
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomRotation((-15, 15), expand=False),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    # transforms.GaussianBlur(19), 
    # transforms.RandomResizedCrop(meta.KERNEL),
    transforms.RandomAffine(degrees=(-45, 45), translate=(0.05, 0.15), scale=(0.75, 1.5)),
    # transforms.RandomPerspective(distortion_scale=0.7, p=1, interpolation=2, fill=0),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    # transforms.RandomCrop(meta.KERNEL//2, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
    # transforms.Resize((meta.KERNEL, meta.KERNEL)),
    transforms.ToTensor(),
])


transforms_list = [transform_01, transform_02, transform_03, transform_04, transform_05]

# It is important to note that if we use expand=True, the image size will be changed. 
# The output will try to include the whole image after rotation. 
# It can be observed that in the following figure, image sizes are different, which is decided by rotation degrees. 
# If you want all the training data to have the same size, the Resize() transform then should be placed after rotation.

# my_transform = transforms.Compose([
#  transforms.RandomPerspective(distortion_scale=0.7,p=1, interpolation=2, fill=0),
#  transforms.ToTensor()

# ])
# torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
# transforms.RandomCrop(size,
#                       padding=None,
#                       pad_if_needed=False,
#                       fill=0,
#                       padding_mode='constant')


# from torchsummary import summary
# device = 'cpu'
# model = UNet_2D_AttantionLayer().to(device)
# summary(model,input_size=(1,256,256))

