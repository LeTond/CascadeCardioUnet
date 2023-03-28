


from configuration import *
from Evaluation.metrics import *

kernel_sz = meta.KERNEL_SZ


#########################################################################################################################
##TODO: COMMENTS
#########################################################################################################################
class Validation(MetaParameters):

    def __init__(self, device):         
        super(MetaParameters, self).__init__()
        self.device = device


class PlotResults():

    def __init__(self):         
        super(MetaParameters, self).__init__()

    def pdf_plot(sub_names, origImage, origMask, predMask):
        pp = PdfPages('results.pdf')
        num = len(sub_names)
        figure, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 8))
        for i in range(num):

            ax[0].imshow(np.resize(origImage[i].cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
            ax[1].imshow(np.resize(origImage[i].cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
            ax[1].imshow(np.resize(predMask[i].cpu(), (kernel_sz, kernel_sz)), alpha = 0.5)
            ax[2].imshow(np.resize(origImage[i].cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
            ax[2].imshow(np.resize(origMask[i].cpu(), (kernel_sz, kernel_sz)), alpha = 0.5)
            
            ax[0].set_title(f"{sub_names[i]}", fontsize = 8, fontweight = 'bold')
            ax[1].set_title("Predicted mask", fontsize = 8, fontweight='bold')
            ax[2].set_title("Manual mask", fontsize = 8, fontweight ='bold')
            figure.tight_layout()
            pp.savefig(figure)
        pp.close()

    def pdf_plot2(sub_names, origImage, origMask, predMask, dice_lv, dice_myo, dice_fib, precision, recall, accur):
        figure, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 8))
        ax[0].imshow(np.resize(origImage.cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
        ax[1].imshow(np.resize(origImage.cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
        ax[1].imshow(np.resize(predMask.cpu(), (kernel_sz, kernel_sz)), alpha = 0.5)
        ax[2].imshow(np.resize(origImage.cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
        ax[2].imshow(np.resize(origMask.cpu(), (kernel_sz, kernel_sz)), alpha = 0.5)
        
        ax[0].set_title(f"{sub_names}", fontsize = 8, fontweight = 'bold')
        ax[1].set_title(f"DC_LV:{dice_lv}/DC_MYO:{dice_myo}/DC_FIB:{dice_fib}/Prec:{precision}/Recall:{recall}/Accur:{accur}", fontsize = 8, fontweight='bold')
        ax[2].set_title("", fontsize = 8, fontweight ='bold')
        figure.tight_layout()
                
        return figure


    def prepare_plot(sub_names, origImage, origMask, predMask, dice_lv, dice_myo, dice_fib, precision, recall, accur):
        figure, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 8))
        ax[0].imshow(np.resize(origImage.cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
        ax[1].imshow(np.resize(origImage.cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
        ax[1].imshow(np.resize(predMask.cpu(), (kernel_sz, kernel_sz)), alpha = 0.5)
        ax[2].imshow(np.resize(origImage.cpu(), (kernel_sz, kernel_sz)), plt.get_cmap('gray'))
        ax[2].imshow(np.resize(origMask.cpu(), (kernel_sz, kernel_sz)), alpha = 0.5)
        
        ax[0].set_title(f"{sub_names}", fontsize = 8, fontweight = 'bold')
        ax[1].set_title(f"Dice: LV - {dice_lv} || MYO - {dice_myo} || FIB - {dice_fib} || Prec - {precision} || Rec - {recall}", fontsize = 8, fontweight='bold')
        ax[2].set_title("", fontsize = 8, fontweight ='bold')
        figure.tight_layout()
        
        return figure

    def make_predictions(predicted_masks):

        ds = DiceLoss()

        for i in range(predicted_masks[0][0]):
            if predicted_masks[5][i].sum() < 10:
                predicted_masks[10][i][predicted_masks[10][i] == 3]= 2

            fib_metrics = metrics(predicted_masks[7][i], predicted_masks[6][i], predicted_masks[5][i], 'FIB')
            dice_lv = round((float(ds(predicted_masks[1][i], predicted_masks[2][i]))), 3)
            dice_myo = round((float(ds(predicted_masks[3][i], predicted_masks[4][i]))), 3)
            # dice_fib = round((float(ds(predicted_masks[5][i], predicted_masks[6][i]))), 3)
            dice_fib = fib_metrics[3]
            dice = round((dice_lv + dice_myo + dice_fib) / 3, 3)
            
            precision, recall, accur = fib_metrics[0], fib_metrics[1], fib_metrics[2]
            
            # if dice_fib < 0.5:
            prepare_plot(predicted_masks[7][i], predicted_masks[8][i], predicted_masks[9][i], predicted_masks[10][i], dice_lv, dice_myo, dice_fib, precision, recall, accur)
                        # break
                    # pdf_plot2(sub_names[i], inputs[i], labels[i], predict[i], dice_lv, dice_myo, dice_fib, precision, recall, accur)