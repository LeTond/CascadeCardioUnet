from configuration import *
from Validation.metrics import *
from Validation.validation import *

from Training.dataset import *
from Preprocessing.split_dataset import *


class PlotResults(MetaParameters):

    def __init__(self):         
        super(MetaParameters, self).__init__()

        if self.CROPPING is True:
            self.kernel_sz = self.CROPP_KERNEL
        elif self.CROPPING is False:
            self.kernel_sz = self.KERNEL

    def save_plot(self, sub_names, origImage, origMask, predMask):

        pp = PdfPages('results.pdf')
        slices = len(sub_names)
        figure, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 8))

        origImage = np.resize(origImage.cpu(), (self.kernel_sz, self.kernel_sz, slices))
        predMask = np.resize(predMask.cpu(), (self.kernel_sz, self.kernel_sz, slices))
        origMask = np.resize(origMask.cpu(), (self.kernel_sz, self.kernel_sz, slices))

        for slc in range(slices):
            
            ax[0].imshow(origImage[slc], plt.get_cmap('gray'))
            ax[1].imshow(origImage[slc], plt.get_cmap('gray'))
            ax[1].imshow(predMask[slc], alpha = 0.5)
            ax[2].imshow(origImage[slc], plt.get_cmap('gray'))
            ax[2].imshow(origMask[slc], alpha = 0.5)
            
            ax[0].set_title(f"{sub_names[slc]}", fontsize = 8, fontweight = 'bold')
            ax[1].set_title("Predicted mask", fontsize = 8, fontweight='bold')
            ax[2].set_title("Manual mask", fontsize = 8, fontweight ='bold')
            figure.tight_layout()
            pp.savefig(figure)
        pp.close()

    def prepare_plot(self, sub_names, origImage, origMask, predMask):

        figure, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 8))

        origImage = np.resize(origImage.cpu(), (self.kernel_sz, self.kernel_sz))
        predMask = np.resize(predMask.cpu(), (self.kernel_sz, self.kernel_sz))
        origMask = np.resize(origMask.cpu(), (self.kernel_sz, self.kernel_sz))

        ax[0].imshow(origImage, plt.get_cmap('gray'))
        ax[1].imshow(origImage, plt.get_cmap('gray'))
        ax[1].imshow(predMask, alpha = 0.5)
        ax[2].imshow(origImage, plt.get_cmap('gray'))
        ax[2].imshow(origMask, alpha = 0.5)
        
        ax[0].set_title(f"{sub_names}", fontsize = 8, fontweight = 'bold')
        ax[1].set_title(f"Dice: LV - {self.dice_lv} || MYO - {self.dice_myo} || FIB - {self.dice_fib} || Prec - {self.precision} || Rec - {self.recall}", fontsize = 8, fontweight='bold')
        ax[2].set_title("", fontsize = 8, fontweight ='bold')
        figure.tight_layout()
        
        return figure

    def make_predictions(self, predicted_masks):

        for i in range(predicted_masks[0][0]):
            if predicted_masks[5][i].sum() < 10:

                predicted_masks[3][i] += predicted_masks[5][i]
                predicted_masks[5][i] = predicted_masks[5][i] * 0
                predicted_masks[10][i][predicted_masks[10][i] == 3]= 2

            fib_metrics = tm.get_image_metrics(predicted_masks[6][i], predicted_masks[5][i], 'FIB')
            
            self.dice_lv = round((float(ds(predicted_masks[1][i], predicted_masks[2][i]))), 3)
            self.dice_myo = round((float(ds(predicted_masks[3][i], predicted_masks[4][i]))), 3)
            self.dice_fib = round((float(ds(predicted_masks[5][i], predicted_masks[6][i]))), 3)
            self.precision, self.recall, self.accur = fib_metrics[0], fib_metrics[1], fib_metrics[2]

            # if self.dice_fib < 0.2:
            self.prepare_plot(predicted_masks[7][i], predicted_masks[8][i], predicted_masks[9][i], predicted_masks[10][i])


#############################
# class Validation(MetaParameters):

#     def __init__(self):         
#         super(MetaParameters, self).__init__()

def bland_altman_per_subject(unet, test_list, meta, kernel_sz):
    
    GT_myo, CM_myo, GT_fib, CM_fib, true_Myo_vol, Myo_vol, true_Fib_vol, Fib_vol = [], [], [], [], [], [], [], []

    for subj in test_list:

        test_ds = GetData([subj]).generated_data_list()
        test_ds_origin = test_ds[0]
        test_ds_mask = test_ds[1]
        test_ds_names = test_ds[2]
        
        try:
            test_set = MyDataset(meta.NUM_LAYERS, test_ds_origin, test_ds_mask, test_ds_names, kernel_sz, default_transform)
            test_batch_size = len(test_set)
            test_loader = DataLoader(test_set, test_batch_size, drop_last=True, shuffle=False, pin_memory=True)

            metrics = TissueMetrics(unet, test_loader).bland_altman_metrics()
            GT_myo.append(metrics[0])
            CM_myo.append(metrics[1])
            GT_fib.append(metrics[2])
            CM_fib.append(metrics[3])
            true_Myo_vol.append(metrics[4])
            Myo_vol.append(metrics[5])
            true_Fib_vol.append(metrics[6])
            Fib_vol.append(metrics[7])
        
        except ValueError:
            print(f'Subject {subj} has no suitable images !!!!')


    return GT_myo, CM_myo, GT_fib, CM_fib, true_Myo_vol, Myo_vol, true_Fib_vol, Fib_vol


def stats_per_subject(unet, test_list, meta, kernel_sz):
    stats_lv, stats_myo, stats_fib = [], [], []
    Precision_FIB, Recall_FIB, Accuracy, Dice_list = [], [], [], []

    for subj in test_list:

        test_ds = GetData([subj]).generated_data_list()
        test_ds_origin = test_ds[0]
        test_ds_mask = test_ds[1]
        test_ds_names = test_ds[2]

        try:
            test_set = MyDataset(meta.NUM_LAYERS, test_ds_origin, test_ds_mask, test_ds_names, kernel_sz, default_transform)
            test_batch_size = len(test_set)
            test_loader = DataLoader(test_set, test_batch_size, drop_last=True, shuffle=False, pin_memory=True)

            tm = TissueMetrics(unet, test_loader)
            counted_parameters = tm.image_metrics()

            stats_lv.append(np.mean(tm.main_stat_parameters(counted_parameters[0])[1]))
            stats_myo.append(np.mean(tm.main_stat_parameters(counted_parameters[1])[1]))
            stats_fib.append(np.mean(tm.main_stat_parameters(counted_parameters[2])[1]))

            Precision_FIB.append(np.mean(counted_parameters[3]))
            Recall_FIB.append(np.mean(counted_parameters[4]))
            Accuracy.append(np.mean(counted_parameters[5]))
            Dice_list.append(counted_parameters[2])

        except ValueError:
            print(f'Subject {subj} has no suitable images !!!!')

    return stats_lv, stats_myo, stats_fib, Precision_FIB, Recall_FIB, Accuracy, Dice_list


test_ds = GetData(test_list).generated_data_list()
test_ds_origin = test_ds[0]
test_ds_mask = test_ds[1]
test_ds_names = test_ds[2]


if meta.CROPPING is False:
    unet = torch.load(f'{meta.PROJ_NAME}/{meta.MODEL_NAME}.pth').to(device=device)
    kernel_sz = meta.KERNEL

elif meta.CROPPING is True:
    unet = torch.load(f'{meta.CROPP_PROJ_NAME}/{meta.MODEL_NAME}.pth').to(device=device)
    kernel_sz = meta.CROPP_KERNEL 

test_set = MyDataset(meta.NUM_LAYERS, test_ds_origin, test_ds_mask, test_ds_names, kernel_sz, default_transform)
# test_set = MyDataset(meta.NUM_LAYERS, test_ds_origin, test_ds_mask, test_ds_names, kernel_sz, transform)
test_batch_size = len(test_set)
test_loader = DataLoader(test_set, test_batch_size, drop_last=True, shuffle=False, pin_memory=True)

ds = DiceLoss()
show_predicted_masks = MaskPrediction().prediction_masks(unet, test_loader)
tm = TissueMetrics(unet, test_loader)


try:
    GT_myo, CM_myo, GT_fib, CM_fib, true_Myo_vol, Myo_vol, true_Fib_vol, Fib_vol = bland_altman_per_subject(unet, test_list, meta, kernel_sz)
    stats_lv, stats_myo, stats_fib, Precision_FIB, Recall_FIB, Accuracy, Dice_list = stats_per_subject(unet, test_list, meta, kernel_sz)
except ValueError:
    print(f'Subjects has no suitable images !!!!')




