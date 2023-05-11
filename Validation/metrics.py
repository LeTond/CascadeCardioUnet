from Preprocessing.preprocessing import ReadImages
from parameters import meta 
from Preprocessing.dirs_logs import *
from Training.dataset import *

from configuration import *



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


def create_hist(value_list: list):
    img_np = np.array(value_list)
    plt.hist(img_np.ravel(), bins=20, density=False)
    plt.xlabel("DSC")
    plt.ylabel("Images")
    plt.title("Distribution of dice")


class MaskPrediction():

    def prediction_masks(self, model_net, dataset_):
        
        size = len(dataset_.dataset)
        model_net.eval()
        
        with torch.no_grad():
            for inputs, labels, sub_names in dataset_:
                inputs, labels, sub_names = inputs.to(device), labels.to(device), list(sub_names)   
                predict = torch.softmax(model_net(inputs), dim = 1)
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


class TissueMetrics(MetaParameters, MaskPrediction):

    def __init__(self, Net, dataset):         
        super(MetaParameters, self).__init__()
        super(MaskPrediction, self).__init__()

        model_net = Net
        dataset_ = dataset

        mp = MaskPrediction().prediction_masks(model_net, dataset_)

        if meta.CROPPING is True:
            self.kernel_sz = self.CROPP_KERNEL
        elif meta.CROPPING is False:
            self.kernel_sz = self.KERNEL

        self.shp = mp[0]
        self.pred_lv = mp[1]
        self.labe_lv = mp[2]
        self.pred_myo = mp[3]
        self.labe_myo = mp[4]
        self.pred_fib = mp[5]
        self.labe_fib = mp[6]
        self.sub_names = mp[7]
        self.inputs = mp[8]
        self.labels = mp[9]
        self.predict = mp[10]

    @staticmethod
    def get_orig_slice(sub_name):
        sub_name_list = sub_name.split(' ')
        
        return sub_name_list

    def get_image_contrast(self, num_label):
        smooth = 1e-5
        masks_matrix = np.copy(self.masks_matrix)
        masks_matrix[masks_matrix != num_label] = 0
        masks_matrix[masks_matrix == num_label] = 1
        orig_matrix = np.copy(self.images_matrix) * masks_matrix
        summ_mask_matrix = masks_matrix.sum()
        summ_orig_matrix = orig_matrix.sum()
        mean_contrast = round(float((summ_orig_matrix + smooth) / (summ_mask_matrix + smooth)), 2)   #7

        return mean_contrast

    def get_image_metrics(self, label, prediction, metric_name: str):

        smooth = 1e-5
        GT = label.sum()
        CM = prediction.sum()
        TP = (label * prediction).sum()
        FN = np.abs(GT - TP)
        FP = np.abs(CM - TP)
        TN = np.abs(self.kernel_sz * self.kernel_sz - GT - FP)
        
        precision = round(float((TP + smooth) / (TP + FP + smooth)), 2)
        recall = round(float((TP + smooth) / (TP + FN + smooth)), 2)    
        accuracy = round(float((TP + TN + smooth) / (TP + TN + FP + FN + smooth)), 2)
        dice = round(float((2 * TP + smooth) / (2 * TP + FP + FN + smooth)), 2)

        return precision, recall, accuracy, dice 

    def main_stat_parameters(self, value_list: list):
        median_value = round(float(np.median(value_list)), 2)
        mean_value = round(float(np.mean(value_list)), 2)
        std_value = round(float(np.std(value_list)), 2)
        
        return median_value, mean_value, std_value

    def image_contrast(self):

        smooth = 1e-5
        for i in range(self.shp[0]):

            orig_slc = int(self.get_orig_slice(self.sub_names[i])[2]) 
            orig_sub = str(self.get_orig_slice(self.sub_names[i])[0])
            
            self.images_matrix = ReadImages(f"{self.ORIGS_DIR}/{orig_sub}.nii").view_matrix()[:,:,-orig_slc] 
            self.masks_matrix = ReadImages(f"{self.MASKS_DIR}/{orig_sub}.nii").view_matrix()[:,:,-orig_slc] 

            mean_contrast_lv = self.get_image_contrast(num_label = 1)
            mean_contrast_myo = self.get_image_contrast(num_label = 2)
            mean_contrast_fib = self.get_image_contrast(num_label = 3)

            print(f'{self.sub_names[i]} - mean LV {mean_contrast_lv}, mean Myo {mean_contrast_myo}, mean Fib {mean_contrast_fib}')

    def image_metrics(self):

        precision, recall, accur, dice = 0, 0, 0, 0
        lv_dice_list, myo_dice_list, fib_dice_list = [], [], []
                    
        for i in range(self.shp[0]):

            #Замена пикселей фиброза на пиксели миокарда
            if self.pred_fib[i].sum().item() < 10:
                    self.pred_myo[i] += self.pred_fib[i]
                    self.pred_fib[i] = self.pred_fib[i] * 0

            fib_metrics = self.get_image_metrics(self.labe_fib[i], self.pred_fib[i], 'FIB')
            myo_metrics = self.get_image_metrics(self.labe_myo[i], self.pred_myo[i], 'MYO')
            lv_metrics = self.get_image_metrics(self.labe_lv[i], self.pred_lv[i], 'LV')

            lv_dice_list.append(lv_metrics[3])
            myo_dice_list.append(myo_metrics[3])
            fib_dice_list.append(fib_metrics[3])

            precision += fib_metrics[0]
            recall += fib_metrics[1]
            accur += fib_metrics[2]
            dice += fib_metrics[3]

        mean_precision = round(precision / self.shp[0], 2) 
        mean_recall = round(recall / self.shp[0], 2)
        mean_accur = round(accur / self.shp[0], 2)
        mean_dice = round(dice / self.shp[0], 2)

        return lv_dice_list, myo_dice_list, fib_dice_list, mean_precision, mean_recall, mean_accur, dice

    @staticmethod
    def get_volume(label):
        lab_volume = label.sum().item()

        return lab_volume

    def get_recent_volume(self):
        Vgt = round(float((GT_fib + smooth) / (GT_myo + GT_fib + smooth) * 100), 2)
        Vcm = round(float((CM_fib + smooth) / (CM_myo + CM_fib + smooth) * 100), 2)
        
        return Vgt, Vcm

    def bland_altman_metrics(self):
        GT_fib, CM_fib = [], []
        GT_myo, CM_myo = [], []

        lv_vol, myo_vol, fib_vol = 0, 0, 0
        True_lv_vol, True_myo_vol, True_fib_vol = 0, 0, 0

        for i in range(self.shp[0]):

            GT_myo.append(self.get_volume(self.labe_myo[i]))
            CM_myo.append(self.get_volume(self.pred_myo[i]))
            GT_fib.append(self.get_volume(self.labe_fib[i]))
            CM_fib.append(self.get_volume(self.pred_fib[i]))

            lv_vol += self.pred_lv[i].numpy().sum()
            myo_vol += self.pred_myo[i].numpy().sum()
            fib_vol += self.pred_fib[i].numpy().sum()

            True_lv_vol += self.labe_lv[i].numpy().sum()
            True_myo_vol += self.labe_myo[i].numpy().sum()
            True_fib_vol += self.labe_fib[i].numpy().sum()                

        mean_True_lv_vol = round(True_lv_vol / 1000 * 32)   #TODO: read pixel size from header.zoom
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



