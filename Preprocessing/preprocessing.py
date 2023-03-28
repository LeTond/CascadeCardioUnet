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

    def __init__(self, image, mask):         
        super(MetaParameters, self).__init__()
        self.image = image
        self.mask = mask

    def expand_matrix(self, mask, row_img, column_img, def_cord = None):
        
        zero_matrix = np.zeros((row_img, column_img))  
        row_msk, column_msk = self.mask.shape

        if def_cord is None:
            start_row = row_img // 2 - row_msk // 2
            start_column = column_img // 2 - column_msk // 2
            zero_matrix[start_row: start_row + row_msk, start_column: start_column + column_msk] = self.mask
        else:
            X = (def_cord[0] - 112 // 2)
            Y = (def_cord[1] - 112 // 2)
            start_row = row_img // 2 - row_msk // 2 + X
            start_column = column_img // 2 - column_msk // 2 + Y
            zero_matrix[start_row: start_row + row_msk, start_column: start_column + column_msk] = self.mask

        return zero_matrix

    def normalization(self):
        image = np.array(self.image, dtype = np.float32)
        mask = np.array(self.mask, dtype = np.float32)

        shp = image.shape
        max_kernel = max(image.shape[0], image.shape[1])

        image = resize(image, (max_kernel, max_kernel), anti_aliasing = False)
        mask = resize(mask, (max_kernel, max_kernel), anti_aliasing = False, order=0)

        image_max = np.max(image)

        if self.CLIP_RATE is not None:
            image = np.clip(image, self.CLIP_RATE[0] * image_max, self.CLIP_RATE[1] * image_max)

        image_max = np.max(image)
        image = image / image_max

        scale = self.KERNEL_SZ / max_kernel 
        image = rescale(image, scale, anti_aliasing = False)
        mask = rescale(mask, scale, anti_aliasing = False, order=0)

        mask = np.array(mask.reshape(self.KERNEL_SZ, self.KERNEL_SZ, 1), dtype = np.float32)
        image = np.array(image.reshape(self.KERNEL_SZ, self.KERNEL_SZ, 1), dtype = np.float32)
        
        return image, mask

    # def crop_center_3D(img, kernel_sz):
    #     y, x, z = img.shape
    #     startx = x // 2-(kernel_sz // 2)
    #     starty = y // 2-(kernel_sz // 2)    

    #     return img[starty: starty + kernel_sz, startx: startx + kernel_sz, :]

    # def crop_transforms_3D(image, mask, kernel_sz):
    #     image = crop_center_3D(image, kernel_sz)
    #     mask = crop_center_3D(mask, kernel_sz)

    #     return image, mask

    # def crop_center(img, cropx, cropy):
    #     y, x = img.shape
    #     startx = x // 2 - (cropx // 2)
    #     starty = y // 2 - (cropy // 2)
    #     return img[starty: starty + cropy, startx: startx + cropx]

    # def crop_transforms(image, mask, kernel_sz):
        
    #     crop_image = crop_center(image, kernel_sz, kernel_sz)
    #     crop_mask = crop_center(mask, kernel_sz, kernel_sz)
        
    #     return crop_image, crop_mask




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


def expand_matrix(mask, row_img, column_img, def_cord = None):
    
    zero_matrix = np.zeros((row_img, column_img))  
    row_msk, column_msk = mask.shape

    if def_cord is None:
        start_row = row_img // 2 - row_msk // 2
        start_column = column_img // 2 - column_msk // 2
        zero_matrix[start_row: start_row + row_msk, start_column: start_column + column_msk] = mask
    else:
        X = (def_cord[0] - 112 // 2)
        Y = (def_cord[1] - 112 // 2)
        start_row = row_img // 2 - row_msk // 2 + X
        start_column = column_img // 2 - column_msk // 2 + Y
        zero_matrix[start_row: start_row + row_msk, start_column: start_column + column_msk] = mask

    return zero_matrix


# def crop_center(img, cropx, cropy):
#     y, x = img.shape
#     startx = x // 2 - (cropx // 2)
#     starty = y // 2 - (cropy // 2)
#     return img[starty: starty + cropy, startx: startx + cropx]


# def crop_transforms(image, mask, kernel_sz):
    
#     crop_image = crop_center(image, kernel_sz, kernel_sz)
#     crop_mask = crop_center(mask, kernel_sz, kernel_sz)
    
#     return crop_image, crop_mask


# def crop_center_3D(img, kernel_sz):
#     y, x, z = img.shape
#     startx = x // 2-(kernel_sz // 2)
#     starty = y // 2-(kernel_sz // 2)    

#     return img[starty: starty + kernel_sz, startx: startx + kernel_sz, :]


# def crop_transforms_3D(image, mask, kernel_sz):
#     image = crop_center_3D(image, kernel_sz)
#     mask = crop_center_3D(mask, kernel_sz)

#     return image, mask


# def revision_mean_value(mean_val, base_kernel, gap):
    # if mean_val > (base_kernel - gap):
    #     mean_val = (base_kernel - gap)

    # elif mean_val < gap:
    #     mean_val = gap

    # return mean_val
