from torch import nn

import torch
import cv2
import numpy as np
import nibabel as nib
# import pydicom as dicom

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from matplotlib import pylab as plt
from skimage.transform import resize, rescale       #pip install scikit-image
from matplotlib.backends.backend_pdf import PdfPages

from Model.unet2D import UNet_2D, UNet_2D_AttantionLayer
from parameters import *
from Preprocessing.preprocessing import *
from Preprocessing.dirs_logs import create_dir
from skimage.transform import resize, rescale, downscale_local_mean
from configuration import *


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

  
class PredictionMask(MetaParameters):

    def __init__(self, model, kernel_sz, images, image_shp, def_cord):

        super(MetaParameters, self).__init__()

        self.model = model
        self.image_shp = image_shp
        self.device = device
        self.images = images
        self.def_cord = def_cord
        self.kernel_sz = kernel_sz

    def expand_matrix(self, mask, row_img, column_img):
        
        zero_matrix = np.zeros((row_img, column_img))

        ## After prediction of the resized and rescaled image
        if self.def_cord is None:
            row_msk, column_msk = mask.shape
            max_kernel = max(row_img, column_img)
            mask = rescale(mask, (max_kernel/mask.shape[0], max_kernel/mask.shape[1]), anti_aliasing = False, order=0)            
            zero_matrix = mask[: row_img, : column_img]

        ## After prediction of cropped and rescaled image
        elif self.def_cord is not None:
            X = (self.def_cord[0] - self.CROPP_KERNEL // 2)
            Y = (self.def_cord[1] - self.CROPP_KERNEL // 2)
            zero_matrix[X: X + self.CROPP_KERNEL, Y: Y + self.CROPP_KERNEL] = mask

        return zero_matrix

    def predict(self, image):
        self.model.eval()

        with torch.no_grad():

            image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, 1)
            image = torch.from_numpy(image).to(device)

            predict = torch.softmax(self.model(image), dim = 1)
            predict = torch.argmax(predict, dim = 1).cpu()

        return predict, image

    def get_predicted_mask(self):
        mask_list = []

        for image in self.images:

            predict, image = self.predict(image)
            predict = np.reshape(predict, (self.kernel_sz, self.kernel_sz))
            predict = np.array(predict, dtype = np.float32)
            
            if self.def_cord is not None:
                predict = self.expand_matrix(predict, self.image_shp[0], self.image_shp[1])
            else:
                predict = self.expand_matrix(predict, self.image_shp[0], self.image_shp[1])

            predict = resize(predict, (self.image_shp[0], self.image_shp[1]), anti_aliasing_sigma = False)
            mask_list.insert(0, predict)

        mask_list = self.back_flip_matrix(mask_list)

        return mask_list

    @staticmethod
    def back_flip_matrix(matrix_list):
        matrix_list = np.array(matrix_list, dtype = np.float32)
        matrix_list = np.rot90(matrix_list, k = 1, axes = (0, 1))
        matrix_list = matrix_list.transpose(0,2,1)
        matrix_list = np.flip(matrix_list, (1,2))
        matrix_list = np.flip(matrix_list, (1))
        matrix_list = np.flip(matrix_list, (0))
        matrix_list = np.round(matrix_list)

        return matrix_list


class NiftiSaver(MetaParameters):
    def __init__(self, masks_list, file_name, evaluate_directory):         
        super(MetaParameters, self).__init__()

        self.masks_list = masks_list
        self.file_name = file_name
        self.evaluate_directory = evaluate_directory

    def save_nifti(self):
        new_image = nib.Nifti1Image(self.masks_list, affine = np.eye(4))
        nib.save(new_image, f'{self.evaluate_directory}/{self.file_name}')


class DicomSaver(MetaParameters):
    def __init__(self, masks_list, file_name, evaluate_directory):         
        super(MetaParameters, self).__init__()

        self.masks_list = masks_list
        self.file_name = file_name
        self.evaluate_directory = evaluate_directory
        self.orig_dir = f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/'

    def old_dicom(self):
        old_dicom = dicom.dcmread(self.orig_dir + self.file_name)
        return old_dicom

    def change_name(self, old_dicom):

        seq_name = old_dicom[0x0018, 0x1030]
        seq_name.value += '_Mask'
        seq_number = old_dicom[0x0020, 0x0011]
        seq_number.value = int(seq_number.value) + 1000

        return old_dicom        

    def change_grey_to_color(self, old_dicom):

        old_dicom.PhotometricInterpretation = 'RGB'
        old_dicom.SamplesPerPixel = 3
        old_dicom.BitsAllocated = 8
        old_dicom.BitsStored = 8
        old_dicom.HighBit = 7
        old_dicom.add_new(0x00280006, 'US', 0)

        return old_dicom

    def new_dicom_array(self):

        dcm2 = self.old_dicom().pixel_array

        new_dicom_array = cv2.cvtColor(dcm2, cv2.COLOR_GRAY2RGB)
        new_dicom_array = new_dicom_array / 4095 * 255
        new_dicom_array = new_dicom_array.astype(np.uint8)

        mask = self.masks_list[:,:,0].astype(np.float16)

        # new_dicom_array[:,:,1][mask == 1] -= 50
        new_dicom_array[:,:,2][mask == 2] -= 50
        new_dicom_array[:,:,2][mask == 3] += 50

        return new_dicom_array

    def change_value_range_info(self, old_dicom):
        
        old_dicom.SmallestImagePixelValue = np.min(self.new_dicom_array())
        old_dicom.LargestImagePixelValue = np.max(self.new_dicom_array())

        return old_dicom

    def dicom_file_name(self):
        
        new_file_name = self.file_name.split('/')[-1]

        return new_file_name

    def save_dicom_mask(self):

        old_dicom = self.change_name(self.old_dicom())
        mask = self.masks_list[:,:,0].astype(np.float16)
        old_dicom.PixelData = mask.tostring()

        new_file_name = f'{self.evaluate_directory}{self.file_name}'.split('/')[-1]
        new_dir_name = f'{self.evaluate_directory}{self.file_name}'.rstrip(new_file_name)
        create_dir(new_dir_name)

        old_dicom.save_as(f'{self.evaluate_directory}{self.file_name}')

    def save_dicom(self):

        old_dicom = self.change_name(self.old_dicom())
        old_dicom = self.change_grey_to_color(old_dicom)
        old_dicom = self.change_value_range_info(old_dicom)
        old_dicom.PixelData = self.new_dicom_array().tostring()

        new_dir_name = old_dicom.PatientName
        create_dir(f'{self.evaluate_directory}/{new_dir_name}')

        print(f'{self.evaluate_directory}/{new_dir_name}/')
        # old_dicom.save_as(f'{self.evaluate_directory}/{new_dir_name}/{self.file_name}')
        old_dicom.save_as(f'{self.evaluate_directory}/{new_dir_name}/{self.dicom_file_name()}')


class PdfSaver():
    def __init__(self, file_name, dataset_path, evaluate_directory):

        self.file_name = file_name
        self.dataset_path = dataset_path
        self.evaluate_directory = evaluate_directory

        self.images_list = ReadImages(f"{self.dataset_path}{self.file_name}").view_matrix()
        self.masks_list = ReadImages(f"{self.evaluate_directory}/{self.file_name}").view_matrix()

        self.images_list = self.images_list.transpose(2,0,1)
        self.masks_list = self.masks_list.transpose(2,0,1)

    def get_stats_parameters(self):
        stack_id = 0
        smooth = 0.0001
        full_lv_volume, full_myo_volume, full_fib_volume = 0, 0, 0
        volumes_lv, volumes_myo, volumes_fib, related_volume = [], [], [], []
        
        #TODO: get pixel size from header.zoom
        fov = ReadImages(f"{self.dataset_path}{self.file_name}").get_nii_fov()
        volume_size = fov[0] * fov[1] * fov[2]

        for mask in self.masks_list:

            mask_lv = (mask == 1)
            mask_myo = (mask == 2)
            mask_fib = (mask == 3)
            
            lv_volume = round((mask_lv.sum()) * volume_size, 0)
            myo_volume = round((mask_myo.sum()) * volume_size, 0)
            fib_volume = round((mask_fib.sum()) * volume_size, 0)

            volumes_lv.append(lv_volume) 
            volumes_myo.append(myo_volume)
            volumes_fib.append(fib_volume)
            related_volume.append(round((fib_volume / (myo_volume + fib_volume + smooth)) * 100, 0)) 

            full_lv_volume += lv_volume
            full_myo_volume += myo_volume
            full_fib_volume += fib_volume
        
        related_full_fib_volume = round(((full_fib_volume) / (full_myo_volume + full_fib_volume + smooth)) * 100, 0)
        
        return full_lv_volume, full_myo_volume, full_fib_volume, volumes_lv, volumes_myo, volumes_fib, related_volume, related_full_fib_volume

    def save_pdf(self):
        
        rows = 3
        full_lv_volume, full_myo_volume, full_fib_volume, volumes_lv, volumes_myo, volumes_fib, related_volume, related_full_fib_volume = self.get_stats_parameters()
        num_chunk = len(self.images_list) % rows

        chunk_list_masks = list(divide_chunks(self.masks_list, rows))
        chunk_list_images = list(divide_chunks(self.images_list, rows))

        chunk_volumes_lv = list(divide_chunks(volumes_lv, rows))
        chunk_volumes_myo = list(divide_chunks(volumes_myo, rows))
        chunk_volumes_fib = list(divide_chunks(volumes_fib, rows))
        chunk_related_volume = list(divide_chunks(related_volume, rows))
        
        num_chunk = len(chunk_list_images)
        pp = PdfPages(f'{self.evaluate_directory}/{self.file_name}_results.pdf')
        
        for chunk in range(num_chunk):
            masks = chunk_list_masks[chunk]
            images = chunk_list_images[chunk]

            LVv = chunk_volumes_lv[chunk]
            MYOv = chunk_volumes_myo[chunk]
            FIBv = chunk_volumes_fib[chunk]
            relVolume = chunk_related_volume[chunk]
            
            len_chunk = len(masks)
            
            if len_chunk > 1:
                figure, ax = plt.subplots(nrows = len_chunk, ncols = 2, figsize = (8, 8))        
                for i in range(len_chunk):
                    ax[i, 0].imshow(images[i], plt.get_cmap('gray'))
                    ax[i, 1].imshow(images[i], plt.get_cmap('gray'))
                    ax[i, 1].imshow(masks[i], alpha = 0.5)
                    
                    ax[i, 1].set_title(f'rel vol: {relVolume[i]} % '
                                       f'LV vol: {LVv[i] / 1000} ml, ' 
                                       f'MYO vol: {MYOv[i] / 1000} ml, '
                                       f'FIB vol: {FIBv[i] / 1000} ml', 
                                       fontsize = 8, fontweight = 'bold', loc = 'right')
                    
                    figure.tight_layout()
                pp.savefig(figure)
                
            elif len_chunk == 1:
                figure, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 8))             
                ax[0].imshow(images[0], plt.get_cmap('gray'))
                ax[1].imshow(images[0], plt.get_cmap('gray'))
                ax[1].imshow(masks[0], alpha = 0.5)
                ax[1].set_title(f'rel vol: {relVolume[0]} % '
                                f'LV vol: {LVv[0] / 1000} ml, '  
                                f'MYO vol: {MYOv[0] / 1000} ml, ' 
                                f'FIB vol: {FIBv[0] / 1000} ml',
                                fontsize = 8, fontweight = 'bold', loc = 'right')

                figure.tight_layout()
                pp.savefig(figure)
        
        # fig = plt.figure(figsize=(8, 8))
        # text = fig.text(0.2, 0.7,
        #                 f'Full left ventricular volume: ≈ {full_lv_volume} mm3 \n'
        #                 f'Full myocardium volume: ≈ {full_myo_volume} mm3 \n'
        #                 f'Full fibrous volume: ≈ {full_fib_volume} mm3 \n'
        #                 f'Full relative fibrous to myocardium volume: ≈ {related_full_fib_volume} % \n',
        #                 ha = 'left', va = 'top', size = 14)
        # text.set_path_effects([path_effects.Normal()])
        # pp.savefig(fig)
        #
        pp.close()


class GetListImages(MetaParameters):
    
    def __init__(self, file_name, path_to_data, dataset_path, preseg):         
        super(MetaParameters, self).__init__()
        self.file_name = file_name
        self.path_to_data = path_to_data
        self.dataset_path = dataset_path
        self.preseg = preseg

    def array_list(self, kernel_sz):
        
        list_images = []
        def_coord = None

        images = ReadImages(f"{self.dataset_path}{self.file_name}").view_matrix()
        orig_img_shape = images.shape

        if self.preseg:
            masks = ReadImages(f"{self.path_to_data}{self.file_name}").view_matrix()
            images, masks, def_coord = EvalPreprocessData(images, masks).presegmentation_tissues()
            
        for slc in range(images.shape[2]):

            normalized = PreprocessData(images[:, :, slc], mask=None).preprocessing(kernel_sz)[0]
            list_images.append(normalized)

        return list_images, orig_img_shape, def_coord

    def dicom_array(self, kernel_sz):
        
        def_coord = None

        image = ReadImages(f"{self.dataset_path}{self.file_name}").get_dcm()
        orig_img_shape = image.shape

        if self.preseg:

            mask = ReadImages(f"{self.path_to_data}{self.file_name}").get_dcm()
            image, mask, def_coord = EvalPreprocessData(image, mask).presegmentation_tissues()
        
        normalized_image = PreprocessData(image[:, :, 0], mask=None).preprocessing(kernel_sz)[0]
        normalized_image = list([normalized_image])

        return normalized_image, orig_img_shape, def_coord


# def benchmark(func):
#     def wrapper():
#         start = time.time()
#         func()
#         end = time.time()
#         print('[*] Время выполнения: {} секунд.'.format(end-start))
#     return wrapper




