from Evaluation.evaluation import *
from Preprocessing.preprocessing import get_dataset_list
from parameters import MetaParameters
from configuration import *


#########################################################################################################################
##TODO: COMMENTS
#########################################################################################################################
class Evaluation(MetaParameters):

    def __init__(self):         
        super(MetaParameters, self).__init__()
        self.dataset_dir = f'{self.EVAL_DIR}{self.DATASET_NAME}_mask_NEW/'
        self.dataset_path = f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/'
        self.project_name = f'./Results/{self.FOLD}/Lr({self.LR})_Drp({self.DROPOUT})_batch({self.BT_SZ})_L2({self.WDC})'

    def base_evaluation(self, file_name):
        ...

    def evaluation(self, file_name):
        ##  Presegmenation with matrix size KERNEL by KERNEL
        evaluate_directory = f'{self.EVAL_DIR}{self.DATASET_NAME}_mask_NEW'        
        create_dir(evaluate_directory)
        neural_model = torch.load(f'{self.project_name}/{self.MODEL_NAME}.pth').to(device=device)
        images, image_shp, fov_size, def_cord = GetListImages(file_name, self.dataset_dir, self.dataset_path, preseg = False).array_list(self.KERNEL)
        # pdf_predictions(neural_model, file_name, self.KERNEL, images, image_shp, fov_size, evaluate_directory)
        NiftiSaver().save_predictions(neural_model, file_name, self.KERNEL, images, image_shp, fov_size, def_cord, evaluate_directory)

    def preseg_evaluation(self, file_name):
        ##  Segmenation with matrix size CROPP_KERNEL by CROPP_KERNEL
        evaluate_directory = f'{self.CROPP_EVAL_DIR}{self.DATASET_NAME}_mask_NEW'
        dataset_dir = f'{self.EVAL_DIR}{self.DATASET_NAME}_mask_NEW/'

        create_dir(evaluate_directory)
        project_name = f'./Results/{self.CROPP_FOLD}/Lr({self.LR})_Drp({self.DROPOUT})_batch({self.BT_SZ})_L2({self.WDC})'

        neural_model = torch.load(f'{project_name}/{self.MODEL_NAME}.pth').to(device=device)        
        images, image_shp, fov_size, def_cord = GetListImages(file_name, dataset_dir, self.dataset_path, preseg = True).array_list(self.CROPP_KERNEL)

        pdf_predictions(neural_model, file_name, self.CROPP_KERNEL, images, image_shp, fov_size, evaluate_directory)
        NiftiSaver().save_predictions(neural_model, file_name, self.CROPP_KERNEL, images, image_shp, fov_size, def_cord, evaluate_directory)


    def run_process(self):
        dataset_dir = f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/'
        dataset_list = list(get_dataset_list(dataset_dir))

        for file_name in dataset_list:
            if file_name.endswith('.nii'):
                print('Stage 1')
                self.evaluation(file_name)
                print('Evaluation complete')
                if self.CROPPING is True:
                    print('Stage 2')
                    self.preseg_evaluation(file_name)
                    print('Cropped Evaluation complete')



if __name__ == "__main__":
    Evaluation().run_process()
