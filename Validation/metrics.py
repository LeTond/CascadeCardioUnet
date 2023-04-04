from Preprocessing.preprocessing import view_matrix, read_nii
from parameters import meta 
from Preprocessing.dirs_logs import *
from Training.dataset import *

from configuration import *


#########################################################################################################################
##TODO: COMMENTS
#########################################################################################################################
path_to_origs = meta.ORIGS_DIR
path_to_masks = meta.MASKS_DIR


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(pred, target):
        smooth = 0.0001
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return (2.0 * intersection + smooth) / (A_sum + B_sum + smooth)


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


def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))


def volumes(labe_fib, pred_fib, labe_myo, pred_myo):
    smooth = 0.0001

    GT_fib = labe_fib.sum()
    CM_fib = pred_fib.sum()
    GT_myo = labe_myo.sum()
    CM_myo = pred_myo.sum()

    Vgt = round(float((GT_fib + smooth) / (GT_myo + GT_fib + smooth) * 100), 2)
    Vcm = round(float((CM_fib + smooth) / (CM_myo + CM_fib + smooth) * 100), 2)
    
    # return Vgt, Vcm
    return GT_myo, CM_myo, GT_fib, CM_fib


def bland_altman(predicted_masks):
    GT_fib, CM_fib = [], []
    GT_myo, CM_myo = [], []

    lv_vol, myo_vol, fib_vol = 0, 0, 0
    True_lv_vol, True_myo_vol, True_fib_vol = 0, 0, 0

    for i in range(predicted_masks[0][0]):

        GT_myo.append(volumes(predicted_masks[5][i], predicted_masks[6][i], predicted_masks[4][i], predicted_masks[3][i])[0])
        CM_myo.append(volumes(predicted_masks[5][i], predicted_masks[6][i], predicted_masks[4][i], predicted_masks[3][i])[1])

        GT_fib.append(volumes(predicted_masks[5][i], predicted_masks[6][i], predicted_masks[4][i], predicted_masks[3][i])[2])
        CM_fib.append(volumes(predicted_masks[5][i], predicted_masks[6][i], predicted_masks[4][i], predicted_masks[3][i])[3])

        lv_vol += predicted_masks[1][i].numpy().sum()
        myo_vol += predicted_masks[3][i].numpy().sum()
        fib_vol += predicted_masks[5][i].numpy().sum()

        True_lv_vol += predicted_masks[2][i].numpy().sum()
        True_myo_vol += predicted_masks[4][i].numpy().sum()
        True_fib_vol += predicted_masks[6][i].numpy().sum()                

    mean_True_lv_vol = round(True_lv_vol / 1000 * 32)
    mean_True_myo_vol = round(True_myo_vol / 1000 * 32)
    mean_True_fib_vol = round(True_fib_vol / 1000 * 32)

    mean_lv_vol = round(lv_vol / 1000 * 32)
    mean_myo_vol = round(myo_vol / 1000 * 32)
    mean_fib_vol = round(fib_vol / 1000 * 32)

    mean_GT_myo = np.sum(GT_myo) * 32 / 1000
    mean_CM_myo = np.sum(CM_myo) * 32 / 1000

    mean_GT_fib = np.sum(GT_fib) * 32 / 1000
    mean_CM_fib = np.sum(CM_fib) * 32 / 1000

    return mean_GT_myo, mean_CM_myo, mean_GT_fib, mean_CM_fib, mean_True_myo_vol, mean_myo_vol, mean_True_fib_vol, mean_fib_vol


def create_hist(value_list: list):
    img_np = np.array(value_list)
    plt.hist(img_np.ravel(), bins=20, density=False)
    plt.xlabel("DSC")
    plt.ylabel("Images")
    plt.title("Distribution of dice")


def main_stat_parameters(value_list: list):
    median_value = round(float(np.median(value_list)), 2)
    mean_value = round(float(np.mean(value_list)), 2)
    std_value = round(float(np.std(value_list)), 2)
    
    return median_value, mean_value, std_value


def image_parameters(predicted_masks):
    ds = DiceLoss()
    precision, recall, accur = 0, 0, 0
    dice, dice_lv, dice_myo, dice_fib = 0, 0, 0, 0
    all_values_lv, all_values_myo, all_values_fib = [], [], []
                
    for i in range(predicted_masks[0][0]):

        #Замена пикселей фиброза на пиксели миокарда
        if predicted_masks[5][i].sum() < 10:
                predicted_masks[3][i] += predicted_masks[5][i]
                predicted_masks[5][i] = predicted_masks[5][i] * 0

        all_values_lv.append(ds(predicted_masks[1][i], predicted_masks[2][i]))
        all_values_myo.append(ds(predicted_masks[3][i], predicted_masks[4][i]))
        all_values_fib.append(ds(predicted_masks[5][i], predicted_masks[6][i]))

        fib_metrics = TissueContrast().metrics(predicted_masks[6][i], predicted_masks[5][i], 'FIB')
        myo_metrics = TissueContrast().metrics(predicted_masks[4][i], predicted_masks[3][i], 'MYO')
        lv_metrics = TissueContrast().metrics(predicted_masks[2][i], predicted_masks[1][i], 'LV')

        precision += fib_metrics[0]
        recall += fib_metrics[1]
        accur += fib_metrics[2]

        # precision += myo_metrics[0]
        # recall += myo_metrics[1]
        # accur += myo_metrics[2]

        # precision += lv_metrics[0]
        # recall += lv_metrics[1]
        # accur += lv_metrics[2]

    size = predicted_masks[0][0]
    
    mean_precision = round(precision / size, 2) 
    mean_recall = round(recall / size, 2)
    mean_accur = round(accur / size, 2)

    return all_values_lv, all_values_myo, all_values_fib, mean_precision, mean_recall, mean_accur


def prediction_masks(Net, loader_):
    size = len(loader_.dataset)
    
    Net.eval()
    
    with torch.no_grad():
        
        for inputs, labels, sub_names in loader_:
            inputs, labels, sub_names = inputs.to(device), labels.to(device), list(sub_names)   

            predict = torch.softmax(Net(inputs), dim = 1)
            predict = torch.argmax(predict, dim = 1)
            labels = torch.argmax(labels, dim = 1)
            
            pred_lv = (predict == 1).cpu()
            labe_lv = (labels == 1).cpu()
            pred_myo = (predict == 2).cpu()
            labe_myo = (labels == 2).cpu()
            pred_fib = (predict == 3).cpu()
            labe_fib = (labels == 3).cpu()

            shp = predict.shape

    return shp, pred_lv, labe_lv, pred_myo, labe_myo, pred_fib, labe_fib, sub_names, inputs, labels, predict


class MaskPrediction():

    def prediction_masks(self, Net, loader_):
        size = len(loader_.dataset)
        Net.eval()
        with torch.no_grad():
            for inputs, labels, sub_names in loader_:
                inputs, labels, sub_names = inputs.to(device), labels.to(device), list(sub_names)   
                predict = torch.softmax(Net(inputs), dim = 1)
                self.predict = torch.argmax(predict, dim = 1)
                self.labels = torch.argmax(labels, dim = 1)
                
        self.pred_lv = (self.predict == 1).cpu()
        self.labe_lv = (self.labels == 1).cpu()
        self.pred_myo = (self.predict == 2).cpu()
        self.labe_myo = (self.labels == 2).cpu()
        self.pred_fib = (self.predict == 3).cpu()
        self.labe_fib = (self.labels == 3).cpu()

        self.shp = predict.shape

        # return shp, pred_lv, labe_lv, pred_myo, labe_myo, pred_fib, labe_fib, sub_names, inputs, labels, predict


def get_orig_slice(sub_name):

    sub_name_list = sub_name.split(' ')
    
    return sub_name_list


class TissueContrast(MetaParameters):

    def __init__(self):         
        super(MetaParameters, self).__init__()

    def image_contrast(self, predicted_masks):

        smooth = 1e-5
        for i in range(predicted_masks[0][0]):

            fib_matrix = predicted_masks[5][i]
            myo_matrix = predicted_masks[3][i]
            fiblab_matrix = predicted_masks[6][i]
            myolab_matrix = predicted_masks[4][i]
            lvlab_matrix = predicted_masks[2][i]
            sub_name = predicted_masks[7][i]

            metric_dice = self.metrics(fib_matrix, fiblab_matrix, 'FIB')[3]
            metric_dice2 = self.metrics(myo_matrix, myolab_matrix, 'MYO')[3]

            # if metric_dice < 0.4:

            orig_slc = int(get_orig_slice(sub_name)[2]) 
            orig_sub = str(get_orig_slice(sub_name)[0])
            
            images_matrix = view_matrix(read_nii(f"{self.ORIGS_DIR}/{orig_sub}.nii"))[:,:,-orig_slc] 
            masks_matrix = view_matrix(read_nii(f"{self.MASKS_DIR}/{orig_sub}.nii"))[:,:,-orig_slc] 

            # orig_matrix = view_matrix(read_nii(f"{path_to_origs}/{orig_sub}.nii"))[:,:,-orig_slc]            
            # orig_matrix = crop_center(orig_matrix, kernel_sz, kernel_sz)
            # origlab_matrix = view_matrix(read_nii(f"{path_to_masks}/{orig_sub}.nii"))[:,:,-orig_slc]
            # origlab_matrix = crop_center(origlab_matrix, kernel_sz, kernel_sz) 

            fib_matrix = np.copy(masks_matrix)
            fib_matrix[fib_matrix != 3] = 0   #2
            fib_matrix[fib_matrix == 3] = 1   #3
            fib_orig_matrix = images_matrix * fib_matrix #4
            summ_fib_matrix = fib_matrix.sum()          #5
            summ_fib_orig_matrix = fib_orig_matrix.sum()    #6
            mean_contrast_fib = round(float((summ_fib_orig_matrix + smooth) / (summ_fib_matrix + smooth)), 2)   #7

            myo_matrix = np.copy(masks_matrix)
            myo_matrix[myo_matrix != 2] = 0   #2
            myo_matrix[myo_matrix == 2] = 1   #3
            myo_orig_matrix = images_matrix * myo_matrix   #4
            summ_myo_matrix = myo_matrix.sum()   #5
            summ_myo_orig_matrix = myo_orig_matrix.sum()   #6
            mean_contrast_myo = round(float((summ_myo_orig_matrix + smooth) / (summ_myo_matrix + smooth)), 2)   #7

            lv_matrix = np.copy(masks_matrix)
            lv_matrix[lv_matrix != 1] = 0   #2
            lv_matrix[lv_matrix == 1] = 1   #3
            lv_orig_matrix = images_matrix * lv_matrix #4
            summ_lv_matrix = lv_matrix.sum()          #5
            summ_lv_orig_matrix = lv_orig_matrix.sum()    #6
            mean_contrast_lv = round(float((summ_lv_orig_matrix + smooth) / (summ_lv_matrix + smooth)), 2)   #7


            # diff_contrast = round((mean_contrast_fib - mean_contrast_lv + 1) / (mean_contrast_lv + 1) * 100, 2)

            # if mean_contrast_lv < 10:
                # pass 
            # else:
            # diff_contrast = round((mean_contrast_fib + smooth) / (mean_contrast_lv + smooth) * 100, 2)

            # output_massage = f"{predicted_masks[7][i]} || Lab_FIB: {summ_fib_matrix} || Pred_FIB: {predicted_masks[5][i].numpy().sum()} || MeanValLV: {mean_contrast_lv} || MeanValFib: {mean_contrast_fib} || MeanValMyo: {mean_contrast_myo} || DiffVal: {diff_contrast} || DiceFib: {metric_dice}"
            # output_massage = f"{predicted_masks[7][i]} || DiffVal: {diff_contrast} || DiceFib: {metric_dice}"        
            
            # output_massage = f"{predicted_masks[7][i]} || DiffVal: {diff_contrast} || {summ_fib_matrix} || DiceFib: {metric_dice}"
            # print(output_massage)
            # print(f'{predicted_masks[7][i]} || label_pixels_myo: {summ_myo_matrix} || pred_pixels_myo: {predicted_masks[3][i].numpy().sum()} || Mean value MYO: {mean_contrast_myo}')

    def metrics(self, label, prediction, metric_name):
        smooth = 1e-5
        GT = label.sum()
        CM = prediction.sum()
        TP = (label * prediction).sum()
        FN = np.abs(GT - TP)
        FP = np.abs(CM - TP)
        TN = np.abs(self.KERNEL * self.KERNEL - GT - FP)
        
        precision = round(float((TP + smooth) / (TP + FP + smooth)), 2)
        recall = round(float((TP + smooth) / (TP + FN + smooth)), 2)    
        accuracy = round(float((TP + TN + smooth) / (TP + TN + FP + FN + smooth)), 2)
        dice = round(float((2 * TP + smooth) / (2 * TP + FP + FN + smooth)), 2)

        return precision, recall, accuracy, dice 

    def precision_recall_accuracy(self):

        fib_metrics = self.metrics(predicted_masks[6][i], predicted_masks[5][i], 'FIB')
        myo_metrics = self.metrics(predicted_masks[4][i], predicted_masks[3][i], 'MYO')
        lv_metrics = self.metrics(predicted_masks[2][i], predicted_masks[1][i], 'LV')

        precision += fib_metrics[0]
        recall += fib_metrics[1]
        accur += fib_metrics[2]

        # precision += myo_metrics[0]
        # recall += myo_metrics[1]
        # accur += myo_metrics[2]

        # precision += lv_metrics[0]
        # recall += lv_metrics[1]
        # accur += lv_metrics[2]

        return precision, recall, accuracy








