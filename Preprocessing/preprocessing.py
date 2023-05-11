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
# import pydicom as dicom
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import statsmodels.api as sm
import matplotlib.pyplot as plt

from torch import nn
from skimage.transform import resize, rescale, downscale_local_mean
# from matplotlib import pylab as plt
from torch.utils.data import DataLoader
from sklearn import preprocessing        #pip install scikit-learn
from parameters import MetaParameters


class ReadImages():
    def __init__(self, path_to_file):
        self.path_to_file = path_to_file

    def get_nii(self):
        # matplotlib.use('TkAgg')
        img = nib.load(self.path_to_file)
        return img

    def get_dcm(self):

        origin_dicom = dicom.dcmread(self.path_to_file)
        new_dicom = np.array(origin_dicom.pixel_array)
        
        if len(list(new_dicom.shape)) == 2:
            new_dicom = new_dicom[:, :, np.newaxis]
        
        return new_dicom

    def get_nii_fov(self):
        # matplotlib.use('TkAgg')
        img = nib.load(self.path_to_file)
        return img.header.get_zooms()

    def view_matrix(self):
        # np.set_printoptions(threshold=sys.maxsize)
        return np.array(self.get_nii().dataobj)

    def get_file_list(self):
        files = os.listdir(self.path_to_file)
        files.sort()
        return files

    def get_file_path_list(self):

        path_list = []
        for root, subfolder, files in os.walk(self.path_to_file):
            for item in files:
                if item.endswith('.nii') or item.endswith('.dcm'):
                    filenamepath = str(os.path.join(root, item)).split('/')[-1]
                    path_list.append(filenamepath)

        return path_list

    def get_dataset_list(self):
        return list(self.get_file_list())


class PreprocessData(MetaParameters):

    def __init__(self, image, mask = None, names = None):      
        super(MetaParameters, self).__init__()
        self.image = image
        self.mask = mask
        self.names = names

    def preprocessing(self, kernel_sz):
        image = np.array(self.image, dtype = np.float32)
        image = self.clipping(image)
        image = self.normalization(image)
        # image = self.z_normalization(image)
        image = self.equalization_matrix(matrix = image)
        image = self.rescale_matrix(kernel_sz, matrix = image, order = None)
        image = np.array(image.reshape(kernel_sz, kernel_sz, 1), dtype = np.float32)

        if self.mask is not None:
            mask = np.array(self.mask, dtype = np.float32)
            mask = self.equalization_matrix(matrix = mask)
            mask = self.rescale_matrix(kernel_sz, matrix = mask, order = 0)
            mask = np.array(mask.reshape(kernel_sz, kernel_sz, 1), dtype = np.float32)
        else:
            mask = None
        return image, mask

    def clipping(self, image):
        image_max = np.max(image)
        if self.CLIP_RATE is not None:
            image = np.clip(image, self.CLIP_RATE[0] * image_max, self.CLIP_RATE[1] * image_max)
        return image

    @staticmethod
    def normalization(image):
        return image / np.max(image)

    @staticmethod
    def z_normalization(image):
        mean, std = np.mean(image), np.std(image)
        image = (image - mean) / std
        image += abs(np.min(image))
        return image / np.max(image)

    @staticmethod
    def equalization_matrix(matrix):
        max_kernel = max(matrix.shape[0], matrix.shape[1])
        zero_matrix = np.zeros((max_kernel, max_kernel))
        zero_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        matrix = zero_matrix
        return matrix

    @staticmethod
    def center_cropping(matrix):
        y, x = matrix.shape
        min_kernel = min(matrix.shape[0], matrix.shape[1])
        startx = (x - min_kernel)//4*3
        starty = (y - min_kernel)//4*3
        return matrix[starty:starty + min_kernel, startx:startx + min_kernel]

    def rescale_matrix(self, kernel_sz, matrix, order=None):
        shp = matrix.shape
        max_kernel = max(matrix.shape[0], matrix.shape[1])
        scale =  kernel_sz / max_kernel
        # print(f"Start rescaling from {shp} to {self.KERNEL, self.KERNEL}")        
        return rescale(matrix, (scale, scale), anti_aliasing = False, order=order)

    def shuff_dataset(self):
        temp = list(zip(self.images, self.masks, self.names))
        random.shuffle(temp)
        images, masks, names = zip(*temp)
        return list(images), list(masks), list(names)


class EvalPreprocessData(MetaParameters):

    def __init__(self, images = None, masks = None):         
        super(MetaParameters, self).__init__()
        self.images = images
        self.masks = masks

    def presegmentation_tissues(self):
        list_top, list_bot, list_left, list_right = [], [], [], []

        shp = self.images.shape
        base_kernel = min(shp[0], shp[1])
        count = 0

        for slc in range(shp[2]):

            image = self.images[:, :, slc]
            mask = self.masks[:, :, slc]

            if (mask != 0).any():
                count += 1
                predict_mask = np.where(mask != 0)

                list_top.append(np.min(predict_mask[0]))
                list_left.append(np.min(predict_mask[1]))
                list_bot.append(np.max(predict_mask[0]))
                list_right.append(np.max(predict_mask[1]))

            else:

                count += 1
                list_top.append(shp[0] // 2 - 32)
                list_left.append(shp[1] // 2 - 32)
                list_bot.append(shp[0] // 2 + 32)
                list_right.append(shp[1] // 2 + 32)

        mean_top = np.array(list_top).sum() // count
        mean_left = np.array(list_left).sum() // count
        mean_bot = np.array(list_bot).sum() // count 
        mean_right = np.array(list_right).sum() // count

        center_row = (mean_bot + mean_top) // 2
        center_column = (mean_left + mean_right) // 2 

        ## TODO: подумать об обрезке не квадратной а по контуру ровно...
        # max_kernel = max((mean_bot - mean_top), (mean_right - mean_left))
        # gap = max_kernel // 2 + round(0.05 * base_kernel)
        gap = 32

        images = self.images[center_row - gap: center_row + gap, center_column - gap: center_column + gap, :]
        masks = self.masks[center_row - gap: center_row + gap, center_column - gap: center_column + gap, :]

        return images, masks, [center_row, center_column]


class ViewData():

    def view_img(self, img):
        width, height, queue = img.shape
        array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        print(width, height, queue)
        num = 1
        for i in range(0, queue, 1):
            img_arr = img.dataobj[:, :, i]
            plt.subplot(4, 5, num)
            plt.imshow(img_arr, cmap='gray')
            num += 1
        plt.show()


class SaveData():
    ...

class ExportImage():
    ...
