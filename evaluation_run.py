from Evaluation.evaluation import *
from Preprocessing.preprocessing import get_dataset_list
from parameters import MetaParameters
from configuration import *


#########################################################################################################################
##TODO: COMMENTS
#########################################################################################################################
class Evaluation(MetaParameters):

    def __init__(self, device):         
        super(MetaParameters, self).__init__()
        self.device = device

    def segmentation(self, file_name):
        ####################################################################################
        ##  Presegmenation with matrix size 112x112
        ####################################################################################

        evaluate_directory = f'{self.SEGMENT_DIR}{self.DATASET_NAME}_{self.KERNEL_SZ}mask_NEW'
        create_dir(evaluate_directory)

        self.FOLD = f'{self.KERNEL_SZ}Fold_{self.FOLD_NAME}'
        dataset_dir = f'{self.SEGMENT_DIR}{self.DATASET_NAME}_'
        dataset_path = f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/'

        self.PROJECT_NAME = f'./Results/{self.FOLD}/Lr({self.LR})_Drp({self.DROPOUT})_batch({self.BT_SZ})_L2({self.WDC})'
        
        neural_model = torch.load(f'{self.PROJECT_NAME}/{self.MODEL_NAME}.pth').to(device=self.device)
        images, image_shp, fov_size, def_cord = GetListImages(file_name, dataset_dir, dataset_path).array_list()
        

        pdf_predictions(neural_model, file_name, self.KERNEL_SZ, images, image_shp, fov_size, evaluate_directory)

        NiftiSaver().save_predictions(neural_model, file_name, self.KERNEL_SZ, images, image_shp, fov_size, def_cord, evaluate_directory)


    def presegmentation(self, file_name): 
        ####################################################################################
        ##  Segmenation with matrix size 64x64
        ####################################################################################

        evaluate_directory = f'{self.SEGMENT_DIR}{self.DATASET_NAME}_{self.KERNEL_SZ}mask_NEW'
        create_dir(evaluate_directory)


        ## TODO: Прочитать директорию с 128 fov
        self.FOLD = f'{self.KERNEL_SZ}Fold_{self.FOLD_NAME}'
        dataset_dir = f'{self.SEGMENT_DIR}{self.DATASET_NAME}_'
        dataset_path = f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/'

        self.FOLD = f'{self.KERNEL_SZ}Fold_{self.FOLD_NAME}'
        self.PROJECT_NAME = f'./Results/{self.FOLD}/Lr({self.LR})_Drp({self.DROPOUT})_batch({self.BT_SZ})_L2({self.WDC})'
        
        neural_model = torch.load(f'{self.PROJECT_NAME}/{self.MODEL_NAME}.pth').to(device=self.device)        
        images, image_shp, fov_size, def_cord = GetListImages(file_name, dataset_dir, dataset_path).array_list()


        pdf_predictions(neural_model, file_name, self.PRESEG_KERNEL, images, image_shp, fov_size, evaluate_directory)

        NiftiSaver().save_predictions(neural_model, file_name, self.PRESEG_KERNEL, images, image_shp, fov_size, def_cord, evaluate_directory)


    def run_process(self):
        dataset_dir = f'{self.DATASET_DIR}{self.DATASET_NAME}_'
        dataset_path = f'{dataset_dir}origin_new/'
        dataset_list = list(get_dataset_list(dataset_path))

        for file_name in dataset_list:
            if file_name.endswith('.nii'):
                self.segmentation(file_name)
                if self.PRESEGMENTATION is True:
                    self.presegmentation(file_name)


if __name__ == "__main__":
    Evaluation(device).run_process()
