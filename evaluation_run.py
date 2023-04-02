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
        self.fold = f'{self.KERNEL_SZ}Fold_{self.FOLD_NAME}'
        self.dataset_dir = f'{self.SEGMENT_DIR}{self.DATASET_NAME}_{self.KERNEL_SZ}mask_NEW/'
        self.dataset_path = f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/'
        self.project_name = f'./Results/{self.fold}/Lr({self.LR})_Drp({self.DROPOUT})_batch({self.BT_SZ})_L2({self.WDC})'

    def base_evaluation(self, file_name):
        ...

    def evaluation(self, file_name):
        ####################################################################################
        ##  Presegmenation with matrix size 112x112
        ####################################################################################
        evaluate_directory = f'{self.SEGMENT_DIR}{self.DATASET_NAME}_{self.KERNEL_SZ}mask_NEW'        
        create_dir(evaluate_directory)
        print(self.project_name)
        neural_model = torch.load(f'{self.project_name}/{self.MODEL_NAME}.pth').to(device=self.device)
        images, image_shp, fov_size, def_cord = GetListImages(file_name, self.dataset_dir, self.dataset_path, preseg = False).array_list(self.KERNEL_SZ)
        # pdf_predictions(neural_model, file_name, self.KERNEL_SZ, images, image_shp, fov_size, evaluate_directory)
        NiftiSaver().save_predictions(neural_model, file_name, self.KERNEL_SZ, images, image_shp, fov_size, def_cord, evaluate_directory)

    def preseg_evaluation(self, file_name):
        ####################################################################################
        ##  Segmenation with matrix size 64x64
        ####################################################################################
        fold = f'PRESEG_{self.PRESEG_KERNEL}Fold_{self.FOLD_NAME}'
        seg_dir = f'./Results/{self.KERNEL_SZ}Fold_{self.FOLD_NAME}/'

        evaluate_directory = f'{seg_dir}{self.DATASET_NAME}_{self.PRESEG_KERNEL}mask_NEW'
        dataset_dir = f'{seg_dir}{self.DATASET_NAME}_{self.KERNEL_SZ}mask_NEW/'

        # ./Results/Preseg_64Fold_01/ALMAZ_128mask_NEW/S

        create_dir(evaluate_directory)
        project_name = f'./Results/{fold}/Lr({self.LR})_Drp({self.DROPOUT})_batch({self.BT_SZ})_L2({self.WDC})'

        neural_model = torch.load(f'{project_name}/{self.MODEL_NAME}.pth').to(device=self.device)        
        images, image_shp, fov_size, def_cord = GetListImages(file_name, dataset_dir, self.dataset_path, preseg = True).array_list(self.PRESEG_KERNEL)

        pdf_predictions(neural_model, file_name, self.PRESEG_KERNEL, images, image_shp, fov_size, evaluate_directory)

        NiftiSaver().save_predictions(neural_model, file_name, self.PRESEG_KERNEL, images, image_shp, fov_size, def_cord, evaluate_directory)


    def run_process(self):
        dataset_dir = f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/'
        dataset_list = list(get_dataset_list(dataset_dir))

        for file_name in dataset_list:
            if file_name.endswith('.nii'):
                print('Stage 1')
                self.evaluation(file_name)
                print('Evaluation complete')
                if self.PRESEGMENTATION is True:
                    print('Stage 2')
                    self.preseg_evaluation(file_name)



# class Evaluation(MetaParameters):

#     def __init__(self, device):         
#         super(MetaParameters, self).__init__()
#         self.device = device

#     def segmentation(self, file_name):
#         ####################################################################################
#         ##  Presegmenation with matrix size 112x112
#         ####################################################################################

#         evaluate_directory = f'{self.SEGMENT_DIR}{self.DATASET_NAME}_{self.KERNEL_SZ}mask_NEW'
#         create_dir(evaluate_directory)

#         self.FOLD = f'{self.KERNEL_SZ}Fold_{self.FOLD_NAME}'
#         dataset_dir = f'{self.SEGMENT_DIR}{self.DATASET_NAME}_{self.KERNEL_SZ}mask_NEW/'
#         dataset_path = f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/'

#         self.PROJECT_NAME = f'./Results/{self.FOLD}/Lr({self.LR})_Drp({self.DROPOUT})_batch({self.BT_SZ})_L2({self.WDC})'
        
#         neural_model = torch.load(f'{self.PROJECT_NAME}/{self.MODEL_NAME}.pth').to(device=self.device)
#         images, image_shp, fov_size, def_cord = GetListImages(file_name, dataset_dir, dataset_path).array_list(self.KERNEL_SZ)
#         pdf_prediction = pdf_predictions(neural_model, file_name, self.KERNEL_SZ, images, image_shp, fov_size, evaluate_directory)
#         NiftiSaver().save_predictions(neural_model, file_name, self.KERNEL_SZ, images, image_shp, fov_size, def_cord, evaluate_directory)

#     def presegmentation(self, file_name): 
#         ####################################################################################
#         ##  Segmenation with matrix size 64x64
#         ####################################################################################

#         evaluate_directory = f'{self.SEGMENT_DIR}{self.DATASET_NAME}_{self.KERNEL_SZ}mask_NEW'
#         create_dir(evaluate_directory)

#         self.FOLD = f'{self.KERNEL_SZ}Fold_{self.FOLD_NAME}'
#         dataset_dir = f'{self.SEGMENT_DIR}{self.DATASET_NAME}_{self.KERNEL_SZ}mask_NEW/'
#         dataset_path = f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/'

#         self.FOLD = f'{self.KERNEL_SZ}Fold_{self.FOLD_NAME}'
#         self.PROJECT_NAME = f'./Results/Preseg{self.FOLD}/Lr({self.LR})_Drp({self.DROPOUT})_batch({self.BT_SZ})_L2({self.WDC})'
        
#         neural_model = torch.load(f'{self.PROJECT_NAME}/{self.MODEL_NAME}.pth').to(device=self.device)        
#         images, image_shp, fov_size, def_cord = GetListImages(file_name, dataset_dir, dataset_path).array_list(self.PRESEG_KERNEL)
#         pdf_prediction = pdf_predictions(neural_model, file_name, self.PRESEG_KERNEL, images, image_shp, fov_size, evaluate_directory)
#         NiftiSaver().save_predictions(neural_model, file_name, self.PRESEG_KERNEL, images, image_shp, fov_size, def_cord, evaluate_directory)

#     def run_process(self):
#         dataset_dir = f'{self.DATASET_DIR}{self.DATASET_NAME}_'
#         dataset_path = f'{dataset_dir}origin_new/'
#         dataset_list = list(get_dataset_list(dataset_path))

#         for file_name in dataset_list:
#             if file_name.endswith('.nii'):
#                 self.segmentation(file_name)
#                 if self.PRESEGMENTATION is True:
#                     self.presegmentation(file_name)


if __name__ == "__main__":
    Evaluation(device).run_process()
