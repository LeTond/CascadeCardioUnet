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
from skimage.transform import resize, rescale, downscale_local_mean
# from matplotlib import pylab as plt
from torch.utils.data import DataLoader
from sklearn import preprocessing        #pip install scikit-learn
from parameters import MetaParameters


########################################################################################################################
# COMMENTS
########################################################################################################################
## TODO: переделать все на классы
def read_nii(path_to_nii):
    # matplotlib.use('TkAgg')
    img = nib.load(path_to_nii)
    return img


def view_img(img):
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


def view_matrix(img):
    # np.set_printoptions(threshold=sys.maxsize)
    return np.array(img.dataobj)


def compare_images(img_1, img_2, numsl):
    plt.subplot(1, 2, 1)
    img_arr = img_1.dataobj[:, :, numsl]
    plt.imshow(img_arr, cmap = 'gray')
    plt.subplot(1, 2, 2)
    img_arr2 = img_2.dataobj[:, :, numsl]
    plt.imshow(img_arr2, cmap = 'gray')
    plt.show()


def view_matrix_size(path_1, path_2):
    matrix3d = view_matrix(read_nii(path_1)).shape
    matrix3d2 = view_matrix(read_nii(path_2)).shape
    print(matrix3d, matrix3d2)


def create_info(info_file, list_):
    with open(info_file, 'w') as f:
        for i in list_:
            for num in range(0, 10):
                f.write(f"{i} {num}\n")


def read_info(info_file):
    with open(info_file, 'r') as f:
        return f.read()


def get_file_list(path_to_dir):
    files = os.listdir(path_to_dir)
    files.sort()
    return files


def save_pickle(pickle_file, new_data):
    with open(pickle_file, 'wb') as f:
        pickle.dump(new_data, f)


def read_pickle(pickle_file):
    # np.set_printoptions(threshold=sys.maxsize)
    with open(pickle_file, 'rb') as f:
        rp = np.array(pickle.load(f))
    return rp


def shuff_dataset(images, masks, names):
    temp = list(zip(images, masks, names))
    random.shuffle(temp)
    images, masks, names = zip(*temp)
    return list(images), list(masks), list(names)


def get_dataset_list(path_to_files):
    dataset_list = get_file_list(path_to_files)
    return dataset_list


class ReadImages():
    def read_nii(self, path_to_nii):
        # matplotlib.use('TkAgg')
        img = nib.load(path_to_nii)
        return img

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

    def view_matrix(self, img):
        # np.set_printoptions(threshold=sys.maxsize)
        return np.array(img.dataobj)

    def compare_images(self, img_1, img_2, numsl):
        plt.subplot(1, 2, 1)
        img_arr = img_1.dataobj[:, :, numsl]
        plt.imshow(img_arr, cmap = 'gray')
        plt.subplot(1, 2, 2)
        img_arr2 = img_2.dataobj[:, :, numsl]
        plt.imshow(img_arr2, cmap = 'gray')
        plt.show()

    def view_matrix_size(self, path_1, path_2):
        matrix3d = view_matrix(read_nii(path_1)).shape
        matrix3d2 = view_matrix(read_nii(path_2)).shape
        print(matrix3d, matrix3d2)

    def create_info(self, info_file, list_):
        with open(info_file, 'w') as f:
            for i in list_:
                for num in range(0, 10):
                    f.write(f"{i} {num}\n")

    def read_info(self, info_file):
        with open(info_file, 'r') as f:
            return f.read()

    def get_file_list(self, path_to_dir):
        files = os.listdir(path_to_dir)
        files.sort()
        return files

    def save_pickle(self, pickle_file, new_data):
        with open(pickle_file, 'wb') as f:
            pickle.dump(new_data, f)

    def read_pickle(self, pickle_file):
        # np.set_printoptions(threshold=sys.maxsize)
        with open(pickle_file, 'rb') as f:
            rp = np.array(pickle.load(f))
        return rp

    def shuff_dataset(self, images, masks, names):
        temp = list(zip(images, masks, names))
        random.shuffle(temp)
        images, masks, names = zip(*temp)
        return list(images), list(masks), list(names)

    def get_dataset_list(self, path_to_files):
        dataset_list = get_file_list(path_to_files)
        return dataset_list


class ExportImage():
    ...


class PreprocessData(MetaParameters):

    def __init__(self, image, mask = None):         
        super(MetaParameters, self).__init__()
        self.image = image
        self.mask = mask

    def preprocessing(self, kernel_sz):
        image = np.array(self.image, dtype = np.float32)
        image = self.clipping(image)
        image = self.normalization(image)
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
        mean, std = np.mean(image), np.std(image)
        # image = (image - mean) / std
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image / np.max(image)

        return image

    @staticmethod
    def equalization_matrix(matrix):
        max_kernel = max(matrix.shape[0], matrix.shape[1])
        zero_matrix = np.zeros((max_kernel, max_kernel))
        zero_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        matrix = zero_matrix

        return matrix

    def rescale_matrix(self, kernel_sz, matrix, order=None):
        shp = matrix.shape
        max_kernel = max(matrix.shape[0], matrix.shape[1])
        scale =  kernel_sz / max_kernel
        # print(f"Start rescaling from {shp} to {self.KERNEL_SZ, self.KERNEL_SZ}")        

        return rescale(matrix, (scale, scale), anti_aliasing = False, order=order)


