from configuration import *
from Validation.metrics import *

#########################################################################################################################
##TODO: COMMENTS
#########################################################################################################################
class Validation(MetaParameters):

    def __init__(self, device):         
        super(MetaParameters, self).__init__()
        self.device = device


class PlotResults(MetaParameters):

    def __init__(self):         
        super(MetaParameters, self).__init__()

    def save_plot(self, sub_names, origImage, origMask, predMask):
        pp = PdfPages('results.pdf')
        num = len(sub_names)
        figure, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 8))
        for i in range(num):

            ax[0].imshow(np.resize(origImage[i].cpu(), (self.KERNEL_SZ, self.KERNEL_SZ)), plt.get_cmap('gray'))
            ax[1].imshow(np.resize(origImage[i].cpu(), (self.KERNEL_SZ, self.KERNEL_SZ)), plt.get_cmap('gray'))
            ax[1].imshow(np.resize(predMask[i].cpu(), (self.KERNEL_SZ, self.KERNEL_SZ)), alpha = 0.5)
            ax[2].imshow(np.resize(origImage[i].cpu(), (self.KERNEL_SZ, self.KERNEL_SZ)), plt.get_cmap('gray'))
            ax[2].imshow(np.resize(origMask[i].cpu(), (self.KERNEL_SZ, self.KERNEL_SZ)), alpha = 0.5)
            
            ax[0].set_title(f"{sub_names[i]}", fontsize = 8, fontweight = 'bold')
            ax[1].set_title("Predicted mask", fontsize = 8, fontweight='bold')
            ax[2].set_title("Manual mask", fontsize = 8, fontweight ='bold')
            figure.tight_layout()
            pp.savefig(figure)
        pp.close()


    def prepare_plot(self, sub_names, origImage, origMask, predMask):
        figure, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 8))
        ax[0].imshow(np.resize(origImage.cpu(), (self.KERNEL_SZ, self.KERNEL_SZ)), plt.get_cmap('gray'))
        ax[1].imshow(np.resize(origImage.cpu(), (self.KERNEL_SZ, self.KERNEL_SZ)), plt.get_cmap('gray'))
        ax[1].imshow(np.resize(predMask.cpu(), (self.KERNEL_SZ, self.KERNEL_SZ)), alpha = 0.5)
        ax[2].imshow(np.resize(origImage.cpu(), (self.KERNEL_SZ, self.KERNEL_SZ)), plt.get_cmap('gray'))
        ax[2].imshow(np.resize(origMask.cpu(), (self.KERNEL_SZ, self.KERNEL_SZ)), alpha = 0.5)
        
        ax[0].set_title(f"{sub_names}", fontsize = 10, fontweight = 'bold')
        ax[1].set_title(f"Dice: LV - {self.dice_lv} || MYO - {self.dice_myo} || FIB - {self.dice_fib} || Prec - {self.precision} || Rec - {self.recall}", fontsize = 10, fontweight='bold')
        ax[2].set_title("", fontsize = 10, fontweight ='bold')
        figure.tight_layout()
        
        return figure

    def make_predictions(self, predicted_masks):

        ds = DiceLoss()

        for i in range(predicted_masks[0][0]):
            if predicted_masks[5][i].sum() < 10:
                predicted_masks[10][i][predicted_masks[10][i] == 3]= 2

            fib_metrics = metrics(predicted_masks[7][i], predicted_masks[6][i], predicted_masks[5][i], 'FIB')
            self.dice_lv = round((float(ds(predicted_masks[1][i], predicted_masks[2][i]))), 3)
            self.dice_myo = round((float(ds(predicted_masks[3][i], predicted_masks[4][i]))), 3)
            self.dice_fib = fib_metrics[3]
            # dice_fib = round((float(ds(predicted_masks[5][i], predicted_masks[6][i]))), 3)
            # dice = round((self.dice_lv + self.dice_myo + self.dice_fib) / 3, 3)
            self.precision, self.recall, self.accur = fib_metrics[0], fib_metrics[1], fib_metrics[2]
            if self.dice_fib < 0.2:
                self.prepare_plot(predicted_masks[7][i], predicted_masks[8][i], predicted_masks[9][i], predicted_masks[10][i])


test_ds_origin = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_test_origin.pickle'
)
test_ds_mask = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_test_mask.pickle'
)
test_ds_names = pd.read_pickle(
    f'{meta.PREPROCESSED_DATASET}{meta.DATASET_NAME}_test_sub_names.pickle'
)

test_set = MyDataset(meta.NUM_LAYERS, test_ds_origin, test_ds_mask, test_ds_names, meta.KERNEL_SZ, target_transform, target_transform)
test_batch_size = len(test_set)
test_loader = DataLoader(test_set, test_batch_size, drop_last=True, shuffle=False, pin_memory=True)

unet = torch.load(f'{meta.PROJECT_NAME}/{meta.MODEL_NAME}.pth').to(device=device)
predicted_masks = prediction_masks(unet, device, test_loader)







