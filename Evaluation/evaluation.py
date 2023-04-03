from torch import nn

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from matplotlib import pylab as plt
from skimage.transform import resize, rescale       #pip install scikit-image
from matplotlib.backends.backend_pdf import PdfPages

from Model.unet2D import UNet_2D, UNet_2D_AttantionLayer
from parameters import *
from Preprocessing.preprocessing import *
from skimage.transform import resize, rescale, downscale_local_mean
from configuration import *

#########################################################################################################################
##TODO: COMMENTS
#########################################################################################################################
class PdfSaver():
    ...

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

        
## Transfer to Preprocessing.preprocessing
def read_nii_zoom(path_to_nii):
    # matplotlib.use('TkAgg')
    img = nib.load(path_to_nii)
    return img.header.get_zooms()


def plot_to_pdf(file_name, origImage, predMask, full_lv_volume, full_myo_volume, full_fib_volume, volumes_lv, volumes_myo, volumes_fib, related_volume, related_full_fib_volume, kernel_sz, evaluate_directory):
    n = 3
    num_chunk = len(origImage) % n
    
    chunk_list_mask = list(divide_chunks(predMask, n))
    chunk_list_image = list(divide_chunks(origImage, n))
    chunk_volumes_lv = list(divide_chunks(volumes_lv, n))
    chunk_volumes_myo = list(divide_chunks(volumes_myo, n))
    chunk_volumes_fib = list(divide_chunks(volumes_fib, n))
    chunk_related_volume = list(divide_chunks(related_volume, n))
    
    num_chunk = len(chunk_list_image)
    pp = PdfPages(f'{evaluate_directory}/{file_name}_results.pdf')
    
    for chunk in range(num_chunk):
        mask = chunk_list_mask[chunk]
        orig = chunk_list_image[chunk]
        LVv = chunk_volumes_lv[chunk]
        MYOv = chunk_volumes_myo[chunk]
        FIBv = chunk_volumes_fib[chunk]
        relVolume = chunk_related_volume[chunk]
        
        len_chunk = len(mask)
        
        if len_chunk > 1:
            figure, ax = plt.subplots(nrows = len_chunk, ncols = 2, figsize = (8, 8))        
            for i in range(len_chunk):
                ax[i, 0].imshow(np.resize(orig[i].cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
                ax[i, 1].imshow(np.resize(orig[i].cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
                ax[i, 1].imshow(np.resize(mask[i].cpu(), (kernel_sz, kernel_sz)), alpha = 0.5)
                
                ax[i, 1].set_title(f'rel vol: {relVolume[i]} % '
                                   f'LV vol: {LVv[i]} mm2 ' 
                                   f'MYO vol: {MYOv[i]} mm2 '
                                   f'FIB vol: {FIBv[i]} mm2', 
                                   fontsize = 8, fontweight = 'bold', loc = 'right')
                
                figure.tight_layout()
            pp.savefig(figure)
            
        elif len_chunk == 1:
            figure, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 8))             
            ax[0].imshow(np.resize(orig[0].cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
            ax[1].imshow(np.resize(orig[0].cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
            ax[1].imshow(np.resize(mask[0].cpu(), (kernel_sz, kernel_sz)), alpha = 0.5)
            ax[1].set_title(f'rel vol: {relVolume[0]} % '
                            f'LV vol: {LVv[0]} mm2 '  
                            f'MYO vol: {MYOv[0]} mm2 ' 
                            f'FIB vol: {FIBv[0]} mm2',
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


def pdf_predictions(Net, file_name, kernel_sz, images, image_shp, fov_size, evaluate_directory):
    mask_list = []
    stack_id = 0
    smooth = 0.0001
    full_lv_volume, full_myo_volume, full_fib_volume = 0, 0, 0
    origins, predicts, volumes_lv, volumes_myo, volumes_fib, related_volume = [], [], [], [], [], []   
    
    volume_size = fov_size[0] * fov_size[1]
    
    Net.eval()
        
    for image in images:
        with torch.no_grad():
            image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, 1)
            image = torch.from_numpy(image).to(device)

            predict = torch.softmax(Net(image), dim = 1)
            predict = torch.argmax(predict, dim = 1)

            predicts.append(predict)
            origins.append(image)
            
            pred_lv = (predict == 1).cpu()
            pred_myo = (predict == 2).cpu()
            pred_fib = (predict == 3).cpu()
            
            lv_volume = round((pred_lv.numpy().sum()) * volume_size, 0)
            myo_volume = round((pred_myo.numpy().sum()) * volume_size, 0)
            fib_volume = round((pred_fib.numpy().sum()) * volume_size, 0)

            volumes_lv.append(lv_volume) 
            volumes_myo.append(myo_volume)
            volumes_fib.append(fib_volume)
            related_volume.append(round((fib_volume / (myo_volume + fib_volume + smooth)) * 100, 0)) 

            full_lv_volume += lv_volume
            full_myo_volume += myo_volume
            full_fib_volume += fib_volume
    
    related_full_fib_volume = round(((full_fib_volume) / (full_myo_volume + full_fib_volume + smooth)) * 100, 0)

    plot_to_pdf(file_name, origins, predicts, full_lv_volume, full_myo_volume, full_fib_volume, volumes_lv, volumes_myo, volumes_fib, related_volume, related_full_fib_volume, kernel_sz, evaluate_directory)


class PredictionMask():
    def __init__(self, model):         
        self.model = model
        self.device = device

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, 1)
            image = torch.from_numpy(image).to(device)

            predict = torch.softmax(self.model(image), dim = 1)
            predict = torch.argmax(predict, dim = 1).cpu()

        return predict


class NiftiSaver(MetaParameters):
    def __init__(self):         
        super(MetaParameters, self).__init__()

    def expand_matrix(self, mask, row_img, column_img, def_cord):
        
        zero_matrix = np.zeros((row_img, column_img))

        ## After prediction of the resized and rescaled image
        if def_cord is None:
            row_msk, column_msk = mask.shape
            max_kernel = max(row_img, column_img)
            mask = rescale(mask, (max_kernel/mask.shape[0], max_kernel/mask.shape[1]), anti_aliasing = False, order=0)            
            zero_matrix = mask[: row_img, : column_img]

        ## After prediction of cropped and rescaled image
        elif def_cord is not None:
            X = (def_cord[0] - self.CROPP_KERNEL // 2)
            Y = (def_cord[1] - self.CROPP_KERNEL // 2)
            zero_matrix[X: X + self.CROPP_KERNEL, Y: Y + self.CROPP_KERNEL] = mask

        return zero_matrix

    def save_predictions(self, Net, file_name, kernel_sz, images, image_shp, fov_size, def_cord, evaluate_directory):

        mask_list = []
        for image in images:
            
            predict = PredictionMask(Net).predict(image)
            predict = np.reshape(predict, (kernel_sz, kernel_sz))
            predict = np.array(predict, dtype = np.float32)
            
            if def_cord is not None:
                predict = self.expand_matrix(predict, image_shp[0], image_shp[1], def_cord)
            else:
                predict = self.expand_matrix(predict, image_shp[0], image_shp[1], None)

            predict = resize(predict, (image_shp[0], image_shp[1]), anti_aliasing_sigma = False)
            mask_list.insert(0, predict)

        mask_list = np.array(mask_list, dtype = np.float32)
        mask_list = np.rot90(mask_list, k = 1, axes = (0, 1))
        mask_list = mask_list.transpose(0,2,1)
        mask_list = np.flip(mask_list, (1,2))
        mask_list = np.flip(mask_list, (1))
        mask_list = np.flip(mask_list, (0))
        mask_list = np.round(mask_list)

        new_image = nib.Nifti1Image(mask_list, affine = np.eye(self.NUM_LAYERS))
        print(type(new_image), image_shp)

        nib.save(new_image, f'{evaluate_directory}/{file_name}')


class GetListImages(MetaParameters):
    
    def __init__(self, file_name, path_to_data, dataset_path, preseg):         
        super(MetaParameters, self).__init__()
        self.file_name = file_name
        self.path_to_data = path_to_data
        self.dataset_path = dataset_path
        self.preseg = preseg

    def array_list(self, kernel_sz):
        
        list_images = []
        count = 0
        def_coord = None

        fov = read_nii_zoom(f"{self.dataset_path}{self.file_name}")
        images = view_matrix(read_nii(f"{self.dataset_path}{self.file_name}"))

        orig_img_shape = images.shape

        if self.preseg:
            masks = view_matrix(read_nii(f"{self.path_to_data}{self.file_name}"))
            images, masks, def_coord = EvalPreprocessData(images, masks).presegmentation_tissues()

        for slc in range(images.shape[2]):
            count += 1
            image = images[:, :, slc]
                
            normalized = PreprocessData(image, mask=None).preprocessing(kernel_sz)[0]
            list_images.append(normalized)

        return list_images, orig_img_shape, fov, def_coord


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

        mean_top = np.array(list_top).sum() // count
        mean_left = np.array(list_left).sum() // count
        mean_bot = np.array(list_bot).sum() // count 
        mean_right = np.array(list_right).sum() // count

        center_row = (mean_bot + mean_top) // 2
        center_column = (mean_left + mean_right) // 2 

        ## TODO: подумать о обрезке не квадратной а по контуру ровно...
        # max_kernel = max((mean_bot - mean_top), (mean_right - mean_left))
        # gap = max_kernel // 2 + round(0.05 * base_kernel)
        gap = 32

        images = self.images[center_row - gap: center_row + gap, center_column - gap: center_column + gap, :]
        masks = self.masks[center_row - gap: center_row + gap, center_column - gap: center_column + gap, :]

        return images, masks, [center_row, center_column]


# def benchmark(func):
#     def wrapper():
#         start = time.time()
#         func()
#         end = time.time()
#         print('[*] Время выполнения: {} секунд.'.format(end-start))
#     return wrapper




